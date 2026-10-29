// NVN 空设备的 Metal 执行层，见 nvn_mtl.h。
// Metal execution layer of the NVN null device, see nvn_mtl.h.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CAMetalDisplayLink.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <objc/runtime.h>
#include "m3_compat.h"   // DM_NVN_M3：Metal 3 兼容层
#include <fcntl.h>
#include "nvn_mtl.h"
#include "nvn_obj.h"
#include "../tex/nvn_fmt_mtl.h"
#include "../tex/nvn_deswizzle.h"
#include "../tex/nvn_deswizzle_gpu.h"
#include "totk_shader_rt.h"
#include "totk_mfx.h"
#include "totk_mfx_up.h"
#include "totk_sysload.h"
/* TOTK_MFX_UPSCALE: for the one compose draw that samples the smaller scene, the scene texture and its MetalFX upscale
   (mfxup_draw before the draw, encode_args swaps the binding). */
static __strong id<MTLTexture> g_mfxup_src, g_mfxup_out; static __unsafe_unretained id g_mfxup_sh;
static _Atomic uint64_t g_mfxup_last;   // game frame of the last MetalFX upscale (the FPS box shows " MFX" while recent)
static _Atomic uint64_t g_mfxi_last;    // game frame of the last interpolated frame shown (the FPS box shows "+<generated>" while recent)
static uint64_t g_gen_shown;            // interpolated frames handed to the screen (present threads)
#include "../tex/nvn_bcdec.h"

#include "nvn_dirty.h"
#include "nvn_depth_clear.h"
#include "nvn_mip_range.h"
#if DM_NVN_M3
#include "nvn_scaled_copy.h"
#endifstatic int encode_args(id<MTLCommandEncoder> en, NvShader *sh, int nvn_stage, int metal_vertex) {
    if (!sh->ae) return 1;
    if (nvn_stage == 1 && !strncmp(sh->key, "0ac9a54938974396", 16) && getenv("DM_NVN_FLASHHDR") && R.rt.n && R.rt.color[0])
        g_flash_tonemap_target = tex_mtl(R.rt.color[0]);
    uint64_t slots[NVN_REFL_MAXS] = {0};
    int uniform_textures = 0;
    for (int i=0;i<sh->rf.nb;i++) if (sh->rf.b[i].kind == NVN_RB_TEXU || sh->rf.b[i].kind == NVN_RB_SAMPU) uniform_textures=1;
    args_sync_dirty();
    static int ab_noreuse = -1; if (ab_noreuse < 0) ab_noreuse = getenv("DM_NVN_AB_NOREUSE") != NULL;   // TOTK diagnostic: never reuse a shader's argument buffer within a pass
    int epoch_mode = arg_epoch_on() && nvn_stage != 5 && arg_epoch_safe(sh);
    arg_epoch_key epoch = {0}; int epoch_hit = 0;
    if (epoch_mode) {
        // Serialize with descriptor publication, as the full builder does. The
        // key precedes resource lookup: later changes cannot validate old bytes.
        registry_lock(); epoch = arg_epoch_current(nvn_stage);
        if (g_arg_epoch_negative && sh->ab_buf) epoch.bindings = sh->ab_epoch.bindings;
        epoch_hit = sh->ab_buf && !memcmp(&epoch, &sh->ab_epoch, sizeof epoch);
        registry_unlock();
    }
    int reuse = !uniform_textures && !ab_noreuse && !passdump() && sh->ab_buf && sh->ab_pass == g_pass.serial && nvn_stage != 5 &&
        (epoch_mode ? epoch_hit : !g_args_dirty[nvn_stage]);
    int verify_epoch = reuse && epoch_mode && atomic_load_explicit(&g_arg_epoch_check, memory_order_relaxed);
    if (reuse) {
        if (epoch_mode) { atomic_fetch_add_explicit(&g_arg_epoch_hits, 1, memory_order_relaxed); if (g_args_dirty[nvn_stage]) atomic_fetch_add_explicit(&g_arg_epoch_extra, 1, memory_order_relaxed); }
        if (!verify_epoch) {
            bind_args(en, nvn_stage, metal_vertex, sh->ab_buf, sh->ab_gpu, sh->ab_off, sh->rf.abi);
            g_args_dirty[nvn_stage] = 0; g_stat_args_reuse++;
            return 1;
        }
    }
#if DM_NVN_M3
    // Residency alone does not tell Metal 3 about reads/writes through argument buffers.
    #define USE(r, u) do { id resource_ = (r); m3_indirect_use(en, resource_, (u), nvn_stage, metal_vertex, slots[aid]); if (__builtin_expect(g_res_check_on, 0)) res_check(resource_, sh->key, reg); } while (0)
#else
    #define USE(r, u) do { (void)(u); if (__builtin_expect(g_res_check_on, 0)) res_check((r), sh->key, reg); } while (0)
#endif
    // DM_NVN_PASSDUMP：每通道每阶段第一次新编时记着色器键（K）与各绑定的实际来源（B：UBO 地址 / 长度 / 前 4 个浮点，贴图号与序号或「哑」）
    // DM_NVN_PASSDUMP: on the first new encode per pass per stage, records the shader key (K) and the actual source of each binding (B: UBO address / length / first 4 floats, texture ID and serial, or 「哑」(dummy))
    static uint64_t bl_pass[6]; static int bl_all = -1; if (bl_all < 0) bl_all = getenv("DM_NVN_PASSDUMP_ALL") != NULL || getenv("TOTK_INPUT_FILE") != NULL;   // 每次新编都记（否则只记每通道每阶段第一次） | Record on every new encode (otherwise only the first time per pass per stage)
    int bl = passdump() && (bl_all || bl_pass[nvn_stage] != g_stat_pass);
    if (bl) { bl_pass[nvn_stage] = g_stat_pass; PD("K %llu st%d key=%s nb=%d prog=%p", g_stat_pass, nvn_stage, sh->key[0] ? sh->key : sh->builtin ? sh->builtin : "?", sh->rf.nb, (const void *)S.prog); }
    int slot_mode = tex_slots_on() && !verify_epoch && nvn_stage != 5 && arg_epoch_safe(sh) && !ab_noreuse &&
        !passdump() && !g_res_check_on && g_use != USE_REC && !tex_slots_diagnostics() && !nan_trace_match(sh);
    int slot_check = slot_mode && atomic_load_explicit(&g_tex_slots_check, memory_order_relaxed);
    tex_slot_key slot_key = {0}; uint64_t slot_mask = 0, slot_hits = 0, slot_misses = 0;
    uint64_t handles[NVN_REFL_MAXB];
    registry_lock();   // 贴图 / 采样器登记表可能正被游戏线程扩容或改写 | The texture / sampler registration tables may be being grown or rewritten by the game thread
    int slot_context = 0;
    if (slot_mode) {
        slot_key = (tex_slot_key){g_pass.serial, g_arena_epoch, g_reg_gen, g_ser_gen,
            nvn_dirty_now(), g_owned_gen, (uint64_t)nvn_stage};
        if (g_tex_slots_negative == 21) slot_key.registry = sh->tex_slots.key.registry;
        slot_context = sh->tex_slots.valid && !memcmp(&slot_key, &sh->tex_slots.key, sizeof slot_key);
    }
    for (int i = 0; i < sh->rf.nb; i++) {
        const nvn_rbind *b = &sh->rf.b[i];
        int reg = b->reg, aid = b->aid;
        if (slot_mode && tex_slot_binding(b)) {
            // A preceding miss may create/replace a resource, or a guest write
            // may arrive since the initial key. Then finish this block normally.
            if (slot_context && (slot_key.serial != g_ser_gen || slot_key.dirty != nvn_dirty_now() ||
                slot_key.owned != g_owned_gen || slot_key.arena != g_arena_epoch)) slot_context = 0;
            handles[aid] = reg < 128 ? (b->kind == NVN_RB_SAMP ? S.samp[nvn_stage][reg] : S.tex[nvn_stage][reg]) : 0;
            if (slot_context && (g_tex_slots_negative == 20 || handles[aid] == sh->tex_slots.handles[aid])) {
                slot_mask |= UINT64_C(1) << aid; slot_hits++;
                if (!slot_check) { slots[aid] = sh->tex_slots.slots[aid]; continue; }
            } else slot_misses++;
        }
        if (b->kind == NVN_RB_CBUF) {
            const p_bind_buf *u = reg < 16 ? &S.ubo[nvn_stage][reg] : NULL;
            void *mb; uint64_t mo, gpu;
            if (!u || !u->addr || !mtl_gpu_addr(u->addr, 1, &mb, &mo, &gpu)) { RED_ONCE(16, "阶段 %d cb%d 没有绑定 uniform 缓冲", nvn_stage, reg); continue; }
            use_rec("buf", u->addr, u->size, 0, NULL);
            uint64_t ren = g_nupd ? upd_shadow(u->addr, u->size) : 0;   // TOTK: renamed inline updates (see upd_shadow)
            if (ren) { gpu = ren; if (nvn_stage == 0 && reg == 0 && g_ui_moff != 0) RED_ONCE(58, "UI 矩阵修补读的是内存，跳过了含内联更新的 uniform 副本"); }
            else if (nvn_stage == 0 && reg == 0 && g_ui_moff != 0) { uint64_t pg = ui_mat_patch(u, mb, mo); if (pg) gpu = pg; }
            else { const usnap_t *sc = usnap_find(u->addr, u->size ? u->size : 1); if (sc) gpu = sc->gpu + (u->addr - sc->lo); }   // TOTK: contents as of the submit call
            if (bl) {
                PD("B %llu st%d cb reg%d addr=%#llx size=%llu pool=%#llx+%#llx f=%g,%g,%g,%g", g_stat_pass, nvn_stage, reg, (unsigned long long)u->addr, (unsigned long long)u->size,
                   (unsigned long long)(u->addr - mo), (unsigned long long)mo, ((float *)u->addr)[0], ((float *)u->addr)[1], ((float *)u->addr)[2], ((float *)u->addr)[3]);
                // DM_NVN_CBDUMP=目录：同时把整块 uniform 原样存盘（F帧_p通道_st阶段_cb槽_地址.bin），供真值侧按客体地址逐字节对
                // DM_NVN_CBDUMP=directory: also saves the whole uniform block to disk as-is (F<frame>_p<pass>_st<stage>_cb<slot>_<address>.bin), for the ground-truth side to compare byte by byte by guest address
                static const char *cbd; static int cbd_init; if (!cbd_init) { cbd = getenv("DM_NVN_CBDUMP"); cbd_init = 1; }
                if (cbd) {
                    char p[1024]; snprintf(p, sizeof p, "%s/F%llu_p%llu_st%d_cb%d_%llx.bin", cbd, (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass, nvn_stage, reg, (unsigned long long)u->addr);
                    FILE *f = fopen(p, "wb"); if (f) { fwrite((const void *)u->addr, 1, u->size, f); fclose(f); } else RED_ONCE(41, "CBDUMP 写不出 %s", p);
                    const uint8_t *actual = bound_uniform_bytes(gpu, (size_t)u->size, (__bridge id<MTLBuffer>)mb);
                    if (actual) {
                        snprintf(p, sizeof p, "%s/F%llu_p%llu_st%d_cb%d_%llx.bound-%llx.bin", cbd,
                            (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass, nvn_stage, reg, (unsigned long long)u->addr, (unsigned long long)gpu);
                        FILE *bound = fopen(p, "wb"); if (bound) { fwrite(actual, 1, u->size, bound); fclose(bound); }
                        uint64_t head[2] = {0}; memcpy(head, actual, MIN((size_t)u->size, sizeof head));
                        PD("E %llu st%d cb%d addr=%#llx gpu=%#llx bytes=%llu head=%016llx,%016llx", g_stat_pass, nvn_stage, reg,
                            (unsigned long long)u->addr, (unsigned long long)gpu, (unsigned long long)u->size,
                            (unsigned long long)head[0], (unsigned long long)head[1]);
                    } else PD("E %llu st%d cb%d addr=%#llx gpu=%#llx unavailable", g_stat_pass, nvn_stage, reg, (unsigned long long)u->addr, (unsigned long long)gpu);
                }
            }
            slots[aid] = gpu;
            USE((__bridge id<MTLBuffer>)mb, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_TEXU || b->kind == NVN_RB_SAMPU) {
            // Ryujinx BindlessToArray maps vec*2 + component/2: one handle every 8 bytes.
            id last_texture_resource=nil;
            for (unsigned element=0; element<b->count; element++) {
                aid = b->aid + element;
                uint32_t handle=uniform_texture_handle(b,nvn_stage,element);
                if (bl) PD("I %llu st%d key=%s kind=%d cb=%u off=%u element=%u handle=%#x tid=%u sid=%u",
                    g_stat_pass,nvn_stage,sh->key,b->kind,b->reg,b->off,element,handle,handle&0xfffff,handle>>20);
                if (b->kind == NVN_RB_SAMPU) {
                    unsigned sid=handle>>20;
                    // Unused table records also contain ordinary material floats.
                    // Resolve only registered samplers; samplers are not Metal
                    // resources and do not need per-entry useResource calls.
                    if(sid && sid<g_nsreg && g_sreg_ok[sid])samp_fast(sid);
                    slots[aid]=sid && sid<g_nsreg && g_srid[sid] ? g_srid[sid] : dummy_samp_rid();
                } else {
                    unsigned tid=handle & 0xfffff; void *view=NULL;
                    uint64_t rid=tid && tid<g_ntreg && g_treg[tid].ok ? tex_fast_rid(tid,&view) : 0;
                    id<MTLTexture> texture=(__bridge id<MTLTexture>)view;
                    if (!texture) { texture=dummy_tex(b->dtt,b->dpf); rid=g_dummy_tex_rid[b->dtt&7][b->dpf&3]; }
                    else if (texture.textureType != b->dtt) {
                        id<MTLTexture> adapted=tex_type_view(texture,(MTLTextureType)b->dtt);
                        if (adapted) { texture=adapted; rid=adapted.gpuResourceID._impl; }
                    }
                    slots[aid]=rid;
                    // Large sparse tables repeat dummy/identical resources.
                    // One declaration covers consecutive entries of that resource.
                    if(texture!=last_texture_resource){USE(texture,MTLResourceUsageRead);last_texture_resource=texture;}
                }
            }
        } else if (b->kind == NVN_RB_SAMP || b->kind == NVN_RB_SAMPT) {   // SAMPT：合并句柄（贴图号 | 采样器号 << 20）的采样器部分 | SAMPT: the sampler part of a combined handle (texture ID | sampler ID << 20)
            int sid = reg < 128 ? (int)((b->kind == NVN_RB_SAMPT ? S.tex[nvn_stage][reg] : S.samp[nvn_stage][reg]) >> 20) : 0;
            __unsafe_unretained id<MTLSamplerState> ss = sid ? (__bridge id<MTLSamplerState>)samp_fast(sid) : nil;
            slots[aid] = sid && sid < g_nsreg && g_srid[sid] ? g_srid[sid] : dummy_samp_rid();
            USE(ss ? ss : dummy_samp(), MTLResourceUsageRead);
            if (bl && sid > 0) { int ok = sid < g_nsreg && g_sreg_ok[sid]; samp_par sp = ok ? g_sreg[sid] : (samp_par){0};   // 本段已持 g_rmu | g_rmu is already held in this section
                if (ok) PD("B %llu st%d samp reg%d sid=%d mag=%d min=%d cmp=%d/%d wrap=%d,%d,%d lod=%g..%g（0 最近点 / 1 双线性；2–5 带 mip）", g_stat_pass, nvn_stage, reg, sid, (int)sp.magf, (int)sp.minf, (int)sp.cmp_mode, (int)sp.cmp_func, (int)sp.ws, (int)sp.wt, (int)sp.wr, sp.lod_min, sp.lod_max); }
        } else if (b->kind == NVN_RB_TEX) {
            int tid = reg < 128 ? (int)(S.tex[nvn_stage][reg] & 0xfffff) : 0;
            void *tv = NULL; uint64_t trid = tid ? tex_fast_rid(tid, &tv) : 0; __unsafe_unretained id<MTLTexture> t = (__bridge id<MTLTexture>)tv;
            if (nvn_stage == 1 && reg == 0 && !strncmp(sh->key, "0ac9a54938974396", 16) && getenv("DM_NVN_FLASHHDR")) g_flash_hdr_source = t;
            if (t && tid < g_ntreg) use_rec("tex", (uint64_t)g_treg[tid].t.mem, g_treg[tid].t.L.size, tid, &g_treg[tid].t);
            if (t && passdump() && tid < g_ntreg) PD("R %llu st%d ser=%llu mem=%p sz=%llu", g_stat_pass, nvn_stage, g_treg[tid].t.serial, (void *)g_treg[tid].t.mem, (unsigned long long)g_treg[tid].t.L.size);
            if (t && tid < g_ntreg) tp_note_fs(sh->key, g_treg[tid].t.serial, &g_treg[tid].t);   // TOTK: DM_NVN_TEXPNG_FS
            if (bl) PD("B %llu st%d tex reg%d tid=%d %s ser=%llu nvnfmt=%#x vfmt=%#x fmt=%lu %lux%lux%lu view_lv=%lu+%lu view_sl=%lu array=%lu target=%lu", g_stat_pass, nvn_stage, reg, tid, t ? "实" : "哑", t && tid < g_ntreg ? g_treg[tid].t.serial : 0,
                       tid < g_ntreg ? g_treg[tid].t.d.format : 0, tid < g_ntreg && g_treg[tid].has_v && g_treg[tid].v.set_format ? g_treg[tid].v.format : 0ull,
                       (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.depth,
                       (unsigned long)(t.parentTexture ? t.parentRelativeLevel : 0), (unsigned long)t.mipmapLevelCount,
                       (unsigned long)(t.parentTexture ? t.parentRelativeSlice : 0), (unsigned long)t.arrayLength, (unsigned long)t.textureType);
            if (nan_trace_match(sh) && t) { static int wid = -2; if (wid == -2) { const char *e = getenv("TOTK_NAN_TRACE_WATCH"); wid = e ? atoi(e) : 7; }
                if (aid == wid && (__bridge void *)t != g_nt_watch_ptr) { g_nt_watch = t; g_nt_watch_ptr = (__bridge void *)t; g_nt_watch_ser = tid < g_ntreg ? g_treg[tid].t.serial : 0; } }
            if (nan_trace_match(sh)) {   /* diagnostic TOTK_NAN_TRACE: what each texture slot of the traced shader holds, once per frame */
                static long nt_frame = -1; static int nt_count; long fr = (long)atomic_load(&g_cur_frame);
                if (fr != nt_frame) { nt_frame = fr; nt_count = 0; }
                if (nt_count++ < sh->rf.nb) MLOG("[nan-trace] bind F%ld id%d reg%d tid=%d %s ser=%llu nvnfmt=%#x vfmt=%#x mtl=%lu %lux%lux%lu lv%lu mem=%p size=%llu flags=%#x levels=%u mtlobj=%p parent=%p", fr, aid, reg, tid, t ? "tex" : "dummy",
                    t && tid < g_ntreg ? g_treg[tid].t.serial : 0ull, tid < g_ntreg ? g_treg[tid].t.d.format : 0, tid < g_ntreg && g_treg[tid].has_v && g_treg[tid].v.set_format ? g_treg[tid].v.format : 0u,
                    (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.depth, (unsigned long)t.mipmapLevelCount,
                    tid < g_ntreg ? (void *)g_treg[tid].t.mem : NULL, tid < g_ntreg ? (unsigned long long)g_treg[tid].t.L.size : 0ull, tid < g_ntreg ? g_treg[tid].t.d.flags : 0u,
                    tid < g_ntreg ? g_treg[tid].t.d.levels : 0u, (__bridge void *)t, (__bridge void *)t.parentTexture);
            }
            if (!t) { t = dummy_tex(b->dtt, b->dpf); trid = g_dummy_tex_rid[b->dtt & 7][b->dpf & 3]; }
            else if ((b->dtt == MTLTextureType2DArray && t.textureType == MTLTextureType2D) || (b->dtt == MTLTextureType2D && t.textureType == MTLTextureType2DArray)) {
                /* TOTK: the shader indexes a 2D texture as a 2D array (the UI glyph atlas: NVN clamps the layer to 0) or the reverse;
                   Metal needs a matching texture type in the argument buffer, so bind a cached 1-slice view of the right type. */
                id<MTLTexture> av = tex_type_view(t, (MTLTextureType)b->dtt);
                if (av) { t = av; trid = av.gpuResourceID._impl; } }
            if (g_mfxup_out && (__bridge void *)sh == (__bridge void *)g_mfxup_sh && t == g_mfxup_src) { t = g_mfxup_out; trid = t.gpuResourceID._impl; }   // TOTK_MFX_UPSCALE
            slots[aid] = trid;
            USE(t, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_IMG) {   // 可写贴图：NVN 图像槽 r（BindImages）↔ D3D u_r | Writable texture: NVN image slot r (BindImages) ↔ D3D u_r
            int tid = reg < 32 ? (int)(S.img[nvn_stage][reg] & 0xfffff) : 0;
            void *tv = NULL; uint64_t trid = tid ? tex_fast_rid(tid, &tv) : 0; __unsafe_unretained id<MTLTexture> t = (__bridge id<MTLTexture>)tv;
            if (passdump()) PD("W %llu st%d reg%d tid=%d %s ser=%llu", g_stat_pass, nvn_stage, reg, tid, t ? "实" : "哑", t && tid < g_ntreg ? g_treg[tid].t.serial : 0ull);
            if (t && !(t.usage & MTLTextureUsageShaderWrite)) { RED_ONCE(32, "图像槽绑的贴图没有可写用途（贴图号 %d）", tid); t = nil; }
            if (!t) { t = dummy_img(b->dtt, b->dpf); trid = g_dummy_img_rid[b->dtt & 7][b->dpf & 3]; }   // 未绑：D3D 语义写入丢弃——写进专用哑贴图，不污染读出 0 的那套 | Unbound: D3D semantics discard the write - write into a dedicated dummy texture so the set that reads 0 is not polluted
            slots[aid] = trid;
            USE(t, MTLResourceUsageRead | MTLResourceUsageWrite);
        } else if (b->kind == NVN_RB_TBUF) {   // 类型化缓冲：贴图槽 r 里的缓冲贴图（目标 BUFFER） | Typed buffer: the buffer texture (target BUFFER) in texture slot r
            int tid = reg < 128 ? (int)(S.tex[nvn_stage][reg] & 0xfffff) : 0;
            void *tv = NULL; uint64_t trid = tid ? tex_fast_rid(tid, &tv) : 0; __unsafe_unretained id<MTLTexture> t = (__bridge id<MTLTexture>)tv;
            if (t && t.textureType != MTLTextureTypeTextureBuffer) { RED_ONCE(40, "类型化缓冲槽绑的不是缓冲贴图（贴图号 %d 类型 %lu）", tid, (unsigned long)t.textureType); t = nil; }
            if (bl) PD("B %llu st%d tbuf reg%d tid=%d %s ser=%llu", g_stat_pass, nvn_stage, reg, tid, t ? "实" : "哑", t && tid < g_ntreg ? g_treg[tid].t.serial : 0ull);
            if (!t) { t = dummy_tbuf(b->dpf); trid = g_dummy_tbuf_rid[b->dpf & 3]; }
            slots[aid] = trid;
            USE(t, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_SBUFU) {   // TOTK: the shader loads this storage buffer's address from a user uniform block (slot reg, byte offset off); resolve that pointer now
            const p_bind_buf *u = reg < 16 ? &S.ubo[nvn_stage][reg] : NULL;
            uint64_t va = 0; int pr = 0;
            if (u && u->addr && u->size >= (uint64_t)b->off + 8) {   // TOTK: the block may have been written by a pending inline update / GPU copy of this submission: read what it will hold
                static int pend_env = -1; if (pend_env < 0) { const char *e = getenv("TOTK_PEND_READ"); pend_env = !e || atoi(e); }   // bisect knob: TOTK_PEND_READ=0 reads memory only
                if (pend_env) pr = pend_read(u->addr + b->off, &va, 8, 0); else memcpy(&va, (const void *)(uintptr_t)(u->addr + b->off), 8);
                if (pr < 0) { static uint64_t n; if (++n <= 4) MLOG("storage-buffer pointer in uniform slot %d+%u: a pending write overlaps it only partly or the pending table overflowed; reading memory (%llu so far)", reg, b->off, (unsigned long long)n); }
            }
            void *mb; uint64_t mo, gpu;
            if (va && g_nupd && upd_overlap_seq(va, va + 65536)) { g_ren_late++; RED_ONCE(60, "uniform 槽里的存储缓冲指针 %#llx 在编码器打开后才发现与待写入的 uniform 更新重叠", (unsigned long long)va); }
            if (va && b->rw) usnap_drop(va, 65536);   // TOTK: the shader may write there (extent unknown: 64 KiB as for the update check)
            if (va && mtl_gpu_addr(va, 1, &mb, &mo, &gpu)) { use_rec("buf", va, 1, 0, NULL); slots[aid] = gpu; USE((__bridge id<MTLBuffer>)mb, b->rw ? MTLResourceUsageRead | MTLResourceUsageWrite : MTLResourceUsageRead); }
            else { RED_ONCE(40, "阶段 %d uniform 槽 %d 偏移 %u 处的存储缓冲指针 %#llx 不在内存池里（uniform %s）", nvn_stage, reg, b->off, (unsigned long long)va, u && u->addr ? "已绑" : "未绑"); slots[aid] = dummy_buf_gpu(); USE(dummy_buf(), MTLResourceUsageRead);
                /* TOTK diagnostic (first 12): which bound uniform block of this stage holds a pool pointer at this offset, and the head of every bound block */
                static uint64_t dn; if (dn < 12 && sh) { dn++; char line[1400]; size_t o = 0;
                    for (int r = 0; r < 16; r++) { const p_bind_buf *q = &S.ubo[nvn_stage][r]; void *qb; uint64_t qo; if (!q->addr || q->size < 16 || !mtl_addr(q->addr, q->size, &qb, &qo)) continue; const uint32_t *w = (const uint32_t *)(uintptr_t)q->addr; uint64_t v = q->size >= (uint64_t)b->off + 8 ? *(const uint64_t *)(uintptr_t)(q->addr + b->off) : 0;
                        void *mb2; uint64_t mo2, gpu2; int ok = v && mtl_gpu_addr(v, 1, &mb2, &mo2, &gpu2);
                        o += (size_t)snprintf(line + o, sizeof line - o, " r%d[%llu]=%08x %08x %08x %08x%s", r, (unsigned long long)q->size, w[0], w[1], w[2], w[3], ok ? "<pool-ptr>" : ""); if (o >= sizeof line - 100) break; }
                    MLOG("sbufu miss: shader %.16s stage %d reg %d+%u = %#llx;%s", sh->key, nvn_stage, reg, b->off, (unsigned long long)va, line); } }
            if (bl) PD("B %llu st%d sbufu reg%d+%u -> %#llx%s", g_stat_pass, nvn_stage, reg, b->off, (unsigned long long)va, pr > 0 ? " (pending write)" : "");
        } else if (b->kind == NVN_RB_SBUFP || b->kind == NVN_RB_UBOP) {   // B：存储缓冲 / 计算阶段 uniform 是设备指针，按 NVN 槽取地址 | B: storage buffers / compute-stage uniforms are device pointers; the address is taken by NVN slot
            const p_bind_buf *u = reg < 16 ? (b->kind == NVN_RB_UBOP ? &S.ubo[nvn_stage][reg] : &S.ssbo[nvn_stage][reg]) : NULL;
            void *mb; uint64_t mo, gpu;
            { /* TOTK: DM_NVN_WATCH_SB=<shader key prefix>:<reg>: arm the guest read/write watch on this storage buffer (see runtime_watch_set) */
              static const char *wsb = NULL; static int wsb_init, wsb_reg = -1, wsb_done; if (!wsb_init) { wsb_init = 1; const char *e = getenv("DM_NVN_WATCH_SB"); if (e && strchr(e, ':')) { size_t L = strchr(e, ':') - e; char *k = malloc(L + 1); memcpy(k, e, L); k[L] = 0; wsb = k; wsb_reg = atoi(strchr(e, ':') + 1); } }
              if (wsb && !wsb_done && u && u->addr && reg == wsb_reg && sh && sh->key && !strncmp(sh->key, wsb, strlen(wsb))) { extern void runtime_watch_set(uintptr_t, size_t); runtime_watch_set((uintptr_t)u->addr, (size_t)u->size); wsb_done = 1; MLOG("[watch] armed on storage buffer reg%d of %.16s addr=%#llx size=%llu", reg, sh->key, (unsigned long long)u->addr, (unsigned long long)u->size); } }
            uint64_t ren = 0;
            if (u && u->addr && g_nupd) {   // TOTK: renamed inline updates: uniforms read the private copy; storage access must already see memory
                if (b->kind == NVN_RB_UBOP) ren = upd_shadow(u->addr, u->size);
                else if (upd_overlap_seq(u->addr, u->addr + (u->size ? u->size : 1))) { g_ren_late++; RED_ONCE(59, "存储缓冲 %#llx 在编码器打开后才发现与待写入的 uniform 更新重叠", (unsigned long long)u->addr); }
            }
            if (b->kind == NVN_RB_SBUFP && b->rw && u && u->addr) usnap_drop(u->addr, u->size ? u->size : 1);   // TOTK: the shader may write it: no stale uniform copy from here on
            const usnap_t *usc = !ren && b->kind == NVN_RB_UBOP && u && u->addr ? usnap_find(u->addr, u->size ? u->size : 1) : NULL;   // TOTK: compute uniforms as of the submit call
            if (ren) { slots[aid] = ren; USE(dummy_buf(), MTLResourceUsageRead); }
            else if (usc) { slots[aid] = usc->gpu + (u->addr - usc->lo); USE(dummy_buf(), MTLResourceUsageRead); }
            else if (u && u->addr && mtl_gpu_addr(u->addr, u->size ? u->size : 1, &mb, &mo, &gpu)) { use_rec("buf", u->addr, u->size, 0, NULL); slots[aid] = gpu; USE((__bridge id<MTLBuffer>)mb, b->rw ? MTLResourceUsageRead | MTLResourceUsageWrite : MTLResourceUsageRead); }
            else { slots[aid] = dummy_buf_gpu(); USE(dummy_buf(), MTLResourceUsageRead); }
            if (bl) { /* TOTK: first 16 bytes and the count of non-zero 32-bit words (CPU view of the pool: what earlier GPU work left there) */
                uint64_t nz = 0, nw = 0; uint32_t w4[4] = {0, 0, 0, 0}; void *mb2; uint64_t mo2;
                if (u && u->addr && u->size && mtl_addr(u->addr, u->size, &mb2, &mo2)) { const uint32_t *q = (const uint32_t *)((const uint8_t *)[(__bridge id<MTLBuffer>)mb2 contents] + mo2); nw = u->size / 4; for (uint64_t i = 0; i < nw; i++) nz += q[i] != 0; memcpy(w4, q, nw >= 4 ? 16 : nw * 4); }
                PD("B %llu st%d sbufp reg%d %s addr=%#llx size=%llu %s nonzero_words=%llu/%llu head=%08x %08x %08x %08x", g_stat_pass, nvn_stage, reg, u && u->addr ? "实" : "哑", u ? (unsigned long long)u->addr : 0ull, u ? (unsigned long long)u->size : 0ull, b->rw ? "rw" : "ro",
                   (unsigned long long)nz, (unsigned long long)nw, w4[0], w4[1], w4[2], w4[3]);
                /* TOTK: DM_NVN_CBDUMP also saves storage buffers (F<frame>_p<pass>_st<stage>_sb<slot>_<address>.bin, at most 1 MiB) */
                static const char *sbd; static int sbd_init; if (!sbd_init) { sbd = getenv("DM_NVN_CBDUMP"); sbd_init = 1; }
                if (sbd && u && u->addr && u->size) {
                    char p2[1024]; snprintf(p2, sizeof p2, "%s/F%llu_p%llu_st%d_sb%d_%llx.bin", sbd, (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass, nvn_stage, reg, (unsigned long long)u->addr);
                    FILE *f2 = fopen(p2, "wb"); if (f2) { fwrite((const void *)u->addr, 1, u->size > (1u << 20) ? (1u << 20) : (size_t)u->size, f2); fclose(f2); }
                } }
        } else if (b->kind == NVN_RB_C1) {      // B：着色器自带常数段，取着色器时建好 | B: the shader's own constant segment, built when the shader is fetched
            id<MTLBuffer> cb = sh->c1buf ? sh->c1buf : zero_bank();   // c1 长 0（全库 600 份）而着色器仍声明：读 0 | c1 has length 0 (600 of them in the whole library) yet the shader still declares it: reads 0
            slots[aid] = (sh->c1buf ? sh->c1_gpu : zero_bank_gpu()) + (sh->c1buf ? sh->c1off : 0);
            USE(cb, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_SUPPORT) {   // support_buffer：缩放 1.0、BGRA 互换 0、视口变换未关闭（视口字段不读） | support_buffer: scale 1.0, BGRA swap 0, viewport transform not disabled (viewport fields are not read)
            slots[aid] = support_gpu();
            USE(support_buf(), MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_DRV0 || b->kind == NVN_RB_DRV2) {
            // NVN 驱动常量段：出包时静态统计证明不读的供 0；读的按当前绑定现填——c0 = 存储缓冲表（基址 + 0x10 r：地址 64 位、大小 32 位），
            // c2 = 贴图句柄表（0x168 + 8 r）与采样器句柄表（0x568 + 8 r），值即游戏绑定时给的 NVN 句柄；读到表外的位置出包时已登记、加载时报红
            // NVN driver constant segments: those proven unread by static analysis at packaging time are supplied as 0; those read are filled on the spot from the current bindings - c0 = storage buffer table (base + 0x10 r: address 64 bits, size 32 bits),
            // c2 = texture handle table (0x168 + 8 r) and sampler handle table (0x568 + 8 r), the values being the NVN handles the game gave at bind time; positions read outside the tables are registered at packaging time and reported a failure at load time
            int bank = b->kind == NVN_RB_DRV2 ? 2 : 0;
            if (!sh->rf.drv_read[bank] || (sh->drv_scanned[bank] && !drvfill_on())) { slots[aid] = zero_bank_gpu(); USE(zero_bank(), MTLResourceUsageRead); }
            else {
                id<MTLBuffer> db; slots[aid] = driver_bank_gpu(bank, nvn_stage, &db);
                USE(db, MTLResourceUsageRead);
            }
        } else RED_ONCE(46, "绑定种类 %d 未实现", b->kind);
    }
    #undef USE
    if (slot_mode) {
        for (int i = 0; i < sh->rf.nb; i++) {
            const nvn_rbind *b = &sh->rf.b[i]; int aid = b->aid;
            if (!tex_slot_binding(b)) continue;
            if (slot_check && (slot_mask & (UINT64_C(1) << aid)) && slots[aid] != sh->tex_slots.slots[aid])
                MDIE("texture slot mismatch: shader %s stage %d binding %d kind %d", sh->key, nvn_stage, aid, b->kind);
            sh->tex_slots.handles[aid] = handles[aid]; sh->tex_slots.slots[aid] = slots[aid];
        }
        sh->tex_slots.key = slot_key; sh->tex_slots.valid = 1;
        atomic_fetch_add_explicit(&g_tex_slots_hits, slot_hits, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_tex_slots_misses, slot_misses, memory_order_relaxed);
        if (slot_check) atomic_fetch_add_explicit(&g_tex_slots_verified, slot_hits, memory_order_relaxed);
    }
    registry_unlock();
    if (verify_epoch) {
        const uint64_t *prior = (const uint64_t *)((const uint8_t *)sh->ab_buf.contents + sh->ab_off);
        for (int i = 0; i < sh->rf.nb; i++) {
            const nvn_rbind *b = &sh->rf.b[i]; int aid = b->aid;
            if (slots[aid] == prior[aid]) continue;
            // Fresh driver-bank allocations have different addresses but must hold identical bytes.
            if (b->kind == NVN_RB_DRV0 || b->kind == NVN_RB_DRV2) {
                const uint8_t *a = bound_uniform_bytes(prior[aid], 0xA00, zero_bank());
                const uint8_t *b_ = bound_uniform_bytes(slots[aid], 0xA00, zero_bank());
                if (a && b_ && !memcmp(a, b_, 0xA00)) continue;
            }
            MDIE("argument epoch mismatch: shader %s stage %d binding %d kind %d", sh->key, nvn_stage, aid, b->kind);
        }
        atomic_fetch_add_explicit(&g_arg_epoch_verified, 1, memory_order_relaxed);
        bind_args(en, nvn_stage, metal_vertex, sh->ab_buf, sh->ab_gpu, sh->ab_off, sh->rf.abi);
        g_args_dirty[nvn_stage] = 0; g_stat_args_reuse++;
        return 1;
    }
    NSUInteger arg_len = sh->arg_len ? sh->arg_len : sh->ae.encodedLength;
    if (arg_len > sizeof slots) MDIE("Argument buffer exceeds resource capacity: %lu", (unsigned long)arg_len);
    id<MTLBuffer> ab; size_t off = arena_alloc(arg_len, &ab);
    memcpy((uint8_t *)ab.contents + off, slots, arg_len);
    bind_args(en, nvn_stage, metal_vertex, ab, g_cur_arena->gpu + off, off, sh->rf.abi);
    sh->ab_buf = ab; sh->ab_gpu = g_cur_arena->gpu + off; sh->ab_off = off; sh->ab_pass = g_pass.serial; sh->ab_epoch = epoch; g_args_dirty[nvn_stage] = 0;
    g_stat_args_enc++;
    return 1;
}
static id<MTLBuffer> g_quad_ib; static uint64_t g_quad_gpu;
static id<MTL4RenderCommandEncoder> rp_begin(void);
// ---------- 通道前瞻：本通道是否用到深度 / 模板 ----------
// 开通道时从当前绘制的下一条起往后扫，模拟深度模板状态的变化，直到本通道必然结束处（重设目标、拷贝、计算派发——
// 与回放里结束通道的操作一一对应）。期间没有任何绘制开深度测试 / 深度写 / 模板测试，则本通道可不挂深度附件：
// 深度内容一字节不动，TBDR 上省掉整张深度的读入和写出（真机实测：单绘制全屏通道占 GPU 一半，多挂着 D32S8 读入写出）。
// 扫描按真实回放顺序跨子列表返回、跨本提交余下句柄；提交末尾 = 通道必然结束。通道中途清除、嵌套过深等无法判定的情况一律算「用到」。
// ---------- Pass lookahead: does this pass use depth / stencil ----------
// When opening a pass, scan forward from the op after the current draw, simulating the changes of depth-stencil state, up to where this pass must end (target reset, copy, compute dispatch -
// matching one-to-one the ops that end a pass in the replay). If no draw in between enables depth test / depth write / stencil test, this pass may leave the depth attachment unattached:
// not one byte of the depth content is touched, and on TBDR the load and store of the whole depth texture are saved (measured on a real device: a single-draw fullscreen pass takes half the GPU, with an extra D32S8 load and store attached).
// The scan follows real replay order across sub-list returns and across the remaining handles of this submission; end of submission = the pass must end. Undecidable cases such as a mid-pass clear or too-deep nesting always count as 'used'.
static const uint8_t *g_exec_next;   // 回放中当前绘制的下一条操作 | The op following the current draw in the replay
static const uint8_t *g_ret[16]; static int g_ret_n;   // 回放的调用返回栈（OP_CALL 处压入调用方的下一条） | Call return stack of the replay (at OP_CALL the caller's next op is pushed)
static const uint64_t *g_rest; static int g_nrest;      // 本次提交里当前句柄之后还要回放的句柄 | Handles in this submission still to be replayed after the current handle
void mtl_exec_rest(const uint64_t *h, int n) { g_rest = h; g_nrest = n; }
static uint64_t g_stat_ds_omit;      // 因前瞻证明不用而少挂深度模板的通道数（含只挂模板 / 只挂深度） | Number of passes that attach less depth-stencil because the lookahead proved it unused (including stencil-only / depth-only)
static uint64_t g_stat_s_only, g_stat_d_only;   // 其中只挂模板 / 只挂深度的通道数 | Of those, the number of passes that attach only stencil / only depth
// 纯清除通道前提（hoist_try）：已提前执行、回放走到原位置时要跳过的操作（各组的 SetRenderTargets 与其中的清除）。
// 执行循环、写回前瞻（sdx 索引）、深度模板前瞻（ds_scan）都按它跳过——否则前瞻会把原位置的清除当成「再次被清」，误判提前清好的内容多余
// Clear-only pass hoist (hoist_try): ops that were already executed early and must be skipped when replay reaches their original position (each group's SetRenderTargets and the clears in it).
// The exec loop, the store lookahead (sdx index) and the depth-stencil lookahead (ds_scan) all skip by it - otherwise the lookahead would take the clear at the original position as 'cleared again' and wrongly judge the early-cleared content redundant
#define HOIST_MAX 40   // 最多 4 组 ×（1 条设目标 + 8 条清除） | At most 4 groups × (1 set-target op + 8 clears)
static const uint8_t *g_hoist_skip[HOIST_MAX]; static int g_hoist_n; static uint64_t g_hoist_groups;
static int hoist_skipped(const uint8_t *p) { for (int i = 0; i < g_hoist_n; i++) if (g_hoist_skip[i] == p) return 1; return 0; }
typedef struct { const uint8_t *p, *ret[8]; int ret_n, rest_i, bad; } fz_cursor;
static int fz_next(fz_cursor *c, const uint8_t **out, op_hdr *oh, const void **oa) {
    for (int guard = 0; guard < 1000000; guard++) {
        if (!c->p) return 0;
        op_hdr h; memcpy(&h, c->p, sizeof h); const void *a = c->p + 8;
        switch (h.op) {
            case OP_END:
                if (c->ret_n) c->p = c->ret[--c->ret_n];
                else if (c->rest_i < g_nrest) c->p = (const uint8_t *)g_rest[c->rest_i++];
                else { c->p = NULL; return 0; }
                continue;
            case OP_JUMP: memcpy(&c->p, a, 8); continue;
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (c->ret_n >= 8) { c->bad = 1; return 0; }
                c->ret[c->ret_n++] = c->p + h.size; c->p = (const uint8_t *)t; continue; }
            default: break;
        }
        const uint8_t *p = c->p; c->p += h.size;
        if (g_hoist_n && hoist_skipped(p)) continue;
        if (out) *out = p; if (oh) *oh = h; if (oa) *oa = a; return 1;
    }
    c->bad = 1; return 0;
}
// 可读 tile 通道合并（合并 B 第 2 步）的状态：前一个通道（阶段 1）与后一个通道（阶段 2）共用一个渲染通道，附件取并集
// State of the readable-tile pass merge (merge B step 2): the earlier pass (phase 1) and the later pass (phase 2) share one render pass, and the attachments are the union
typedef struct {
    int active, phase, n1, nskip;
    int nreuse;
    const uint8_t *rt2_op, *reuse[8], *skip[12];
    p_set_rt rt1, rt2;                 // 两个通道各自的逻辑目标（R.rt 在合并期间是并集） | Logical targets of each of the two passes (R.rt is the union during the merge)
    int8_t p2map[8];                   // 后一个通道的逻辑颜色号 → 物理附件 | Logical color number of the later pass → physical attachment
    uint8_t lmap1[8], lmap2[8];        // 物理附件 → 本阶段绘制的逻辑颜色号（0xff = 不写） | Physical attachment → logical color number of this phase's draws (0xff = not written)
    uint8_t clr2[8], cd2f, cs2f; MTLClearColor cc2[8]; double cd2; uint32_t cs2;   // 后一个通道开头的清除（局内双渲染重画时照原样用） | Clears at the start of the later pass (used as-is when the in-run dual render redraws)
    uint8_t wrote[8];                   // 后一个通道写过的物理附件 | Physical attachments written by the later pass
    uint8_t cov2[8];                     // 后一个通道首绘覆盖的物理附件 | Physical attachments covered by the first draw of the later pass
} fuse_t;
static fuse_t g_fz; static uint64_t g_fz_n, g_fz_draws, g_fz_fallback;
typedef struct { int idx; p_draw dw; p_draw_idx di; p_scissor sc; int has_sc; void *S_; uint8_t pm[8]; } defer_snap;   // pm：组 P 目标下标 → 本次绘制的逻辑颜色号（自身绘制为恒等；被吸收的上游组为复合映射） | pm: group P target index → logical color number of this draw (identity for its own draws; a composed mapping for absorbed upstream groups)
typedef struct {
    int active, at_q, in_p, n, reason, reason_op, logged, reset_between; uint64_t born;   // born：创建序号（回放按它排，不按槽位） | born: creation serial (replay is ordered by it, not by slot)
    p_set_rt p_rt, q_rt;
    const uint8_t *p_end_op, *q_op;
    uint8_t clear_c[8], lmap[8];
    MTLClearColor cc[8];
    uint8_t clear_d, clear_s;
    double cd; uint32_t cs;
    uint64_t p_reads[512]; int nreads, p_dw, p_st;
    defer_snap snap[64];
} defer_t;
static defer_t g_defer_slots[4];
static defer_t *g_defer_cur = &g_defer_slots[0];
#define g_defer (*g_defer_cur)
static int sub_eq(const p_set_rt *a, int ka, const p_set_rt *b, int kb);
static uint64_t g_defer_n, g_defer_draws, g_defer_fallbacks;
static const uint8_t *g_defer_pend;   // 在 P 区间那一组的 P 区间结束操作（没有则 NULL）：执行循环每条操作只比这一个指针 | The P-range end op of the group that is in its P range (NULL if none): the exec loop compares each op against just this one pointer
static int g_defer_nact, g_defer_natq;   // 活动组数、已到 Q 等回放的组数（热路径只看计数，不逐槽扫） | Number of active groups, number of groups that reached Q and await replay (the hot path only looks at the counts, no per-slot scan)
static int g_defer_ab = -1, g_defer_ab_on;
static void draw(const p_draw *dw, const p_draw_idx *di);
static int defer_has_in_p(void) { return g_defer_pend != NULL; }
static int defer_has_active(void) { return g_defer_nact > 0; }
static int defer_slot_free(void) { for (int i = 0; i < 4; i++) if (!g_defer_slots[i].active) return i; return -1; }
static int defer_same_q_depth(const defer_t *a, const p_set_rt *q) {
    if (!a->q_rt.depth || !q->depth) return 1;
    return sub_eq(&a->q_rt, 8, q, 8);
}
static int defer_next(int at_q_only, uint64_t after) {   // 下一个（创建序号最小且 > after）活动组的槽号；没有返回 −1 | Slot number of the next active group (smallest creation serial that is > after); returns −1 if none
    int best = -1; for (int i = 0; i < 4; i++) { const defer_t *d = &g_defer_slots[i]; if (!d->active || (at_q_only && !d->at_q) || d->born <= after) continue; if (best < 0 || d->born < g_defer_slots[best].born) best = i; }
    return best;
}
static void defer_release_one(defer_t *d) { if (d->active) { g_defer_nact--; if (d->at_q) g_defer_natq--; if (d->in_p) g_defer_pend = NULL; } for (int i = 0; i < d->n; i++) free(d->snap[i].S_); memset(d, 0, sizeof *d); }
static void defer_clear_all(void) { for (int i = 0; i < 4; i++) defer_release_one(&g_defer_slots[i]); g_defer_nact = g_defer_natq = 0; g_defer_pend = NULL; }
static int hoist_take(const uint8_t *p) { for (int i = 0; i < g_hoist_n; i++) if (g_hoist_skip[i] == p) { g_hoist_skip[i] = g_hoist_skip[--g_hoist_n]; return 1; } return 0; }
enum { DS_FREE = 0, DS_USED = 1, DS_RET = 2 };
static int g_ds_why; static int g_ds_scanned; static int g_ds_need;   // g_ds_need：扫描中累计的需求，1 深度 2 模板   // 诊断：判「用到」时触发的操作码（绘制则记 1000 + dt*100 + dw*10 + st）、扫过的操作数 | g_ds_need: need accumulated during the scan, 1 depth 2 stencil   // Diagnostics: the opcode that triggered the 'used' verdict (for a draw, records 1000 + dt*100 + dw*10 + st), and the number of ops scanned
static int prog_ps_writes_depth(const prog_t *pg);
static int ds_scan(const uint8_t *p, ds_st *ds, int depth) {
    for (int guard = 0; guard < 1000000; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const void *a = p + 8; g_ds_scanned++;
        if (g_hoist_n && hoist_skipped(p)) { p += h.size; continue; }   // 已提前执行的清除通道 | Clear pass already executed early
        switch (h.op) {
            case OP_END: return DS_RET;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (depth >= 8) return DS_USED;
                int r = ds_scan((const uint8_t *)t, ds, depth + 1); if (r != DS_RET) return r; break; }
            case OP_SET_RT: case OP_COPY_B2B: case OP_UPDATE_UB: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX: case OP_DISPATCH: return DS_FREE;
            case OP_CLEAR_COLOR: case OP_CLEAR_DS: g_ds_why = h.op; return DS_USED;
            case OP_BIND_STATE: { const p_bind_state *b = a; if (b->kind == 5) memcpy(ds, b->bytes, sizeof *ds); break; }
            case OP_BIND_PROGRAM: if (prog_ps_writes_depth(((const p_bind_program *)a)->prog)) { g_ds_need |= 1; g_ds_why = 2000; if (g_ds_need == 3) return DS_USED; } break;   /* TOTK: depth-writing fragment function */
            case OP_DRAW: case OP_DRAW_IDX:   // 深度测试 / 写入 → 要深度；模板测试 → 要模板；两项都要即可停 | Depth test / write → depth needed; stencil test → stencil needed; can stop once both are needed
                g_ds_need |= (ds->dt || ds->dw ? 1 : 0) | (ds->st ? 2 : 0);
                if (g_ds_need == 3) { g_ds_why = 1000 + ds->dt * 100 + ds->dw * 10 + ds->st; return DS_USED; } break;
            default: break;
        }
        p += h.size;
    }
    return DS_USED;
}
// 写回前瞻的判定（逐条命令，按真实回放顺序）：
//   多余：再次被设为附件后、任何绘制之前被整块清除（颜色：该序号四通道全写；深度：深度清除；模板：模板 8 位全清。深度与模板分开判、分开设写回）
//   需要：有绘制 / 派发时它绑在任一阶段的贴图或图像槽（含扫描开始时已绑着的）；作拷贝源 / 拷贝目标 / 清贴图对象；再次作附件后先画后清（读入）
// 同一贴图的不同级 / 层分开判（降采样链写第 k 级、读第 k−1 级）；采样绑定按整张贴图算读取（保守）
// Store lookahead decision (command by command, in real replay order):
//   Redundant: after being set as an attachment again and before any draw, it is cleared in full (color: all four channels of that index written; depth: depth clear; stencil: all 8 stencil bits cleared. Depth and stencil are decided separately and their write-back is set separately)
//   Needed: at a draw / dispatch it is bound to a texture or image slot of any stage (including ones already bound when the scan starts); used as copy source / copy destination / texture-clear target; drawn before cleared after becoming an attachment again (load)
// Different levels / layers of the same texture are decided separately (a downsample chain writes level k and reads level k−1); a sampler binding counts as a read of the whole texture (conservative)
static const uint8_t *g_exec_cur;   // 回放中当前这条操作（含） | The op currently being replayed (inclusive)
static void sd_need(sd_att *t, int why) { if (!t->state) { t->state = 1; t->why = (uint8_t)why; } }   // why：1 绘制作贴图读 2 作附件先画后清 3 拷贝 4 清贴图 5 扫到末尾未定 6 无起点 / 索引不可用 | why: 1 a draw reads it as a texture 2 as an attachment, drawn before cleared 3 copy 4 texture clear 5 scanned to the end undecided 6 no start point / index unavailable
static int sdx_decide(uint8_t st[9]);   // 写回前瞻判定（索引见 mtl_sub_index） | Store lookahead decision (for the index see mtl_sub_index)
// ---------- TOTK: layered passes (see g_lay) ----------
static int lay_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("TOTK_LAYER_MERGE"); v = !(e && !strcmp(e, "0")); if (!v) MLOG("TOTK_LAYER_MERGE=0：分层通道合并关闭"); } return v && !tbdr_on(); }
// One colour target, no depth, a view selecting one layer: *layer = that layer.
static int lay_single(const p_set_rt *r, int *layer) {
    if (r->n != 1 || !r->color[0] || r->depth || !r->has_cview[0] || !r->cview[0].set_layers || r->cview[0].nlayers != 1) return 0;
    *layer = r->cview[0].min_layer; return 1;
}
// Same texture, and the same view apart from the selected layer.
static int lay_same_group(const p_set_rt *a, const p_set_rt *b) {
    view_t v = b->cview[0]; v.min_layer = a->cview[0].min_layer;
    return a->color[0]->serial == b->color[0]->serial && !memcmp(&v, &a->cview[0], sizeof v);
}
// Highest layer in [first, nlayers) selected by a later SetRenderTargets of the group before an op at which the replay ends
// the pass anyway. The range only sizes the attachment: lay_continue and the replay's own checks decide each step.
static int lay_scan(const uint8_t *start, const p_set_rt *rt, int first, int nlayers) {
    fz_cursor cur = {.p = start}; int hi = first;
    for (int guard = 0; guard < 4096; guard++) {
        const uint8_t *q; op_hdr h; const void *a;
        if (!fz_next(&cur, &q, &h, &a)) break;
        switch (h.op) {
            case OP_SET_RT: { int l;
                if (!lay_single(a, &l) || !lay_same_group(rt, a) || l < first || l >= nlayers || l - first >= LAY_MAX) return hi;
                if (l > hi) hi = l;
                continue; }
            case OP_UPDATE_UB: case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_COND: case OP_RENDER_ENABLE: case OP_CLEAR_COLOR:
            case OP_DRAW: case OP_DRAW_IDX: case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_BIND_PROGRAM: case OP_BIND_STATE:
            case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF: case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP:
            case OP_BIND_IMG: case OP_STENCIL: case OP_POLY_OFFSET: case OP_BLEND_COLOR: case OP_COPY_STRIDE: case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD:
            case OP_BARRIER: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: continue;
            default: return hi;   // copies, dispatches, texture / depth-stencil clears, events
        }
    }
    return hi;
}
// The draw can run in a layered pass: triangles, and a vertex function that can write the layer.
static int lay_draw_ok(NvProg *pr, const p_draw *dw, const p_draw_idx *di) {
    int prim = dw ? dw->prim : di ? di->prim : -1;
    return (prim == 4 || prim == 5 || prim == 7) && pr && vs_layerable(pr->sh[0]);
}
// rp_begin_ex: the number of layers of a layered pass starting at *first, or 0 for a plain pass. Only for a pass opened by a
// replayed draw or colour clear (g_exec_cur), with no pending clear (a Clear load action would clear every layer of the range).
static int lay_decide(int clear_only, int *first) {
    if (g_lay.refuse) { g_lay.refuse = 0; return 0; }
    if (clear_only || !lay_on() || !g_exec_cur || R.clear_c[0] || !lay_single(&R.rt, first)) return 0;
    op_hdr h; memcpy(&h, g_exec_cur, sizeof h);
    if (h.op != OP_DRAW && h.op != OP_DRAW_IDX && h.op != OP_CLEAR_COLOR) return 0;
    id<MTLTexture> base = tex_mtl(R.rt.color[0]);
    if (!base || base.sampleCount != 1 || pf_block_compressed(base.pixelFormat) || (base.textureType != MTLTextureTypeCube && base.textureType != MTLTextureType2DArray)) return 0;
    NSUInteger lv = R.rt.cview[0].set_levels ? R.rt.cview[0].base_level : 0, layers = base.textureType == MTLTextureTypeCube ? 6 : base.arrayLength;
    if (lv >= base.mipmapLevelCount || mip_extent(base.width, lv) > LAY_MAX_EXTENT || mip_extent(base.height, lv) > LAY_MAX_EXTENT || (NSUInteger)*first >= layers) return 0;
    if ((g_fd_dw || g_fd_di) && !lay_draw_ok(S.prog ? prog_get(S.prog) : nil, g_fd_dw, g_fd_di)) return 0;
    int hi = lay_scan(g_exec_cur + h.size, &R.rt, *first, (int)layers);
    return hi > *first ? hi - *first + 1 : 0;
}
// SetRenderTargets in a layered pass: another layer of its range continues the pass (g_rt_merged: a clear or a draw that
// samples the attachment ends it, as for a same-target continuation).
static int lay_continue(const p_set_rt *r) {
    int l, c;
    if (g_lay_probe_neg == 2 || !lay_single(r, &l) || !lay_single(&R.rt, &c) || !lay_same_group(&R.rt, r) || l < g_lay.base || l >= g_lay.base + g_lay.count) return 0;
    if (!g_lay.merges++) MLOG("[totk] layered pass: first SetRenderTargets merged (layer %d of %d-%d)", l, g_lay.base, g_lay.base + g_lay.count - 1);
    PD("M %llu 分层通道续用 层 %d", g_stat_pass, l);
    g_lay.cur = l; R.rt = *r; g_rt_merged = 1; return 1;
}
static uint64_t lay_gpu(int offset) {   // GPU address of the constant `offset` (the layer within the pass)
    static id<MTLBuffer> b; static uint64_t gpu;
    if (!b) { uint32_t v[LAY_MAX]; for (int i = 0; i < LAY_MAX; i++) v[i] = (uint32_t)i;
        b = [g_dev newBufferWithBytes:v length:sizeof v options:MTLResourceStorageModeShared]; if (!b) MDIE("layer constant buffer"); res_add(b); gpu = b.gpuAddress; }
    if (offset < 0 || offset >= LAY_MAX) MDIE("layer %d outside the layered pass", offset);
    return gpu + 4u * (uint64_t)offset;
}
void mtl_layer_stats(uint64_t v[4]) { v[0] = g_stat_pass; v[1] = g_lay.passes; v[2] = g_lay.merges; v[3] = g_lay.refused; }
static void sd_finish(void) {
    if (!g_nsd) return;
    if (!tbdr_on()) {   // 优化全关：一律写回 | All optimizations off: always write back
        for (int i = 0; i < g_nsd; i++) { sd_att *t = &g_sd[i];
            enc_r_store(g_enc, t->depth, MTLStoreActionStore, (NSUInteger)t->idx); }
        g_nsd = 0; return; }
    uint8_t nst[9];
    if (sdx_decide(nst)) { for (int i = 0; i < g_nsd; i++) if (nst[i] == 2) g_sd[i].state = 2; else sd_need(&g_sd[i], nst[i] >> 2); }
    else for (int i = 0; i < g_nsd; i++) sd_need(&g_sd[i], 6);   // 列表外结束的通道 / 索引不可用：一律写回 | Pass ended outside the list / index unavailable: always write back
    for (int i = 0; i < g_nsd; i++) {
        sd_att *t = &g_sd[i]; int udc_dead = udc_on() && t->load && !t->wrote;
        // 与窗口贴图写在同一个通道里的颜色附件不进账本：它是出图链上的一环，不是「每帧先重置再用」的瞬态附件。
        // 局内「放大」通道一次写两张——场景贴图 + 窗口贴图，界面合成并进了同一通道，所以局内没人回读场景贴图；游戏一停画 3D（读取画面）
        // 就改由单独的合成通道回头采它。「要进读取画面」在上一帧提交之后才知道，预判拦不住：iPad 真全屏每个安装第一次换幕必撞一次
        // （帧 N 那张 2732×2048 附件没写回，帧 N+1 起被连采 4 帧；键永久禁用）。iPhone 比例下同尺寸的键在试用期就见到读者、被静默禁用。
        // A color attachment written in the same pass as the window texture does not enter the ledger: it is a link in the present chain, not a transient attachment that is 'reset every frame before use'.
        // The in-run 'upscale' pass writes two textures at once - scene texture + window texture - and the UI composite was merged into the same pass, so in-run nobody reads back the scene texture; as soon as the game stops drawing 3D (loading screen)
        // a separate composite pass samples it afterwards instead. 'About to enter the loading screen' is only known after the previous frame was submitted, so prediction cannot stop it: on iPad true fullscreen every install hits it once on the first act change
        // (the 2732×2048 attachment of frame N was not written back and is sampled for 4 consecutive frames starting at frame N+1; the key is permanently disabled). At the iPhone aspect ratio the same-size key already sees a reader during the trial period and is silently disabled.
        int xv_win = 0; for (int k = 0; k < R.rt.n && k < 8; k++) if (R.rt.color[k] && xv_present(R.rt.color[k]->serial)) xv_win = 1;
        uint64_t xtag = 0; int xvcand = xv_on() && !xv_win && t->depth == 0 && t->state != 2 && t->why == 5 && xv_first_good(t->ser, t->lv, t->sl) && !xv_present(t->ser)
                                        && !xv_disabled(xtag = xv_tag_pass(R.rt.color[t->idx], t->lv, t->idx, &R.rt));
        int xvdead = xvcand && xv_trial_get(xtag, 0) >= XV_TRIAL;   // 过了试用期才真的不写回 | Only after the trial period is the write-back really skipped
        if (xvcand && !xvdead) xv_set(t->ser, t->lv, t->sl, XV_PROBE, xtag);   // 试用期：照常写回，只跟踪下一次触碰 | Trial period: write back as usual, only track the next touch
        int dead = t->state == 2 || udc_dead || xvdead; MTLStoreAction sa = dead ? MTLStoreActionDontCare : MTLStoreActionStore;
        if (xvdead) { xv_set(t->ser, t->lv, t->sl, XV_INVALID, xtag); g_xv_dc++; }
        enc_r_store(g_enc, t->depth, sa, (NSUInteger)t->idx);
        g_sd_n_all++; if (dead) { g_sd_n_dead++; g_sd_dead_bytes += t->bytes; g_bw_store -= t->bytes; if (udc_dead) g_udc_saved_bytes += t->bytes; }
        if (!t->state) t->why = 5;
        PD("Z %llu %s ser=%llu lv%u sl%u %s why=%d by=%s slot=%d", g_stat_pass, t->depth == 1 ? "d" : t->depth == 2 ? "s" : "c", (unsigned long long)t->ser, t->lv, t->sl, xvdead ? "dontcare-xv" : dead ? "dontcare" : "store", dead ? 0 : t->why, t->why == 1 && t->wkey ? t->wkey : "-", t->wslot);
        if (udc_on()) PD("U %llu %s ser=%llu lv%u sl%u runtime=%s reason=%s state=%d load=%d", g_stat_pass, t->depth == 1 ? "d" : t->depth == 2 ? "s" : "c", (unsigned long long)t->ser, t->lv, t->sl, t->wrote ? "写过" : "没写", t->wwhy == 1 ? "图像写" : t->wwhy == 2 ? "颜色写掩码" : t->wwhy == 3 ? "片元输出" : t->wwhy == 4 ? "深度写" : t->wwhy == 5 ? "模板写" : "无", t->state, t->load);
    }
    g_nsd = 0;
}
// 按真实回放顺序接着扫：当前列表 → 各层调用方（返回栈自顶向下）→ 本提交余下句柄；提交结束时通道必然结束（commit 前 rp_flush）
// 覆盖证明（全部满足才成立；任何一项认不出按不成立）：
//   着色器：顶点位置 xyz 原样取自一个顶点属性、w 为 1 或同属性 .w（dmsh_build raster_facts）；片元不丢弃、不写 sample_mask、写该颜色号
//   状态：该附件不混合、写掩码满、无 alpha-to-coverage、填充模式、无深度测试（或 ALWAYS）、无模板测试、单采样、非 3D
//   顶点：非索引或 16 / 32 位索引绘制，索引与位置都从 CPU 可见的池内存读出；剪裁 / 视口 / 几何 / 朝向 / 深度裁剪的判定在 nvn_cover.h（纯函数，有单测）
// 读出本次绘制全部顶点的属性 a（非索引或 16 / 32 位索引；顶点号 = first + k 或 索引值 + base_vertex），缺分量按取数规则补 0 / 1。
// 成功返回 0，失败返回原因码（6 顶点数 / 索引类型，7 属性号，8 格式 / 流，9 越界，10 地址）
// Continue scanning in real replay order: current list -> callers at each level (return stack from the top down) -> remaining handles of this submit; the render pass necessarily ends when the submit ends (rp_flush before commit)
// Coverage proof (holds only if all are satisfied; anything unrecognized is treated as not holding):
//   Shader: the vertex position xyz is taken as-is from one vertex attribute, w is 1 or the same attribute's .w (dmsh_build raster_facts); the fragment does not discard, does not write sample_mask, and writes this color index
//   State: this attachment has no blending, a full write mask, no alpha-to-coverage, fill mode, no depth test (or ALWAYS), no stencil test, single sample, not 3D
//   Vertices: non-indexed or 16 / 32-bit indexed draws, with indices and positions both read from CPU-visible pool memory; the scissor / viewport / geometry / winding / depth clipping decisions are in nvn_cover.h (pure functions, with unit tests)
// Read out attribute a of all vertices of this draw (non-indexed or 16 / 32-bit indexed; vertex number = first + k or index value + base_vertex), with missing components filled with 0 / 1 per the fetch rules.
// Returns 0 on success, a reason code on failure (6 vertex count / index type, 7 attribute index, 8 format / stream, 9 out of bounds, 10 address)
static int attr_read(int a, const p_draw *dw, const p_draw_idx *di, float P[64][4]) {
    int n = dw ? dw->count : di->count;
    if (n < 1 || n > 64 || (dw && dw->first < 0) || (di && (di->itype < 1 || di->itype > 2))) return 6;
    if (a < 0 || a >= S.nattr || S.attr[a].stream >= 16) return 7;
    va_st at = S.attr[a]; vs_st sm = S.stream[at.stream]; NVNbufferRange vb = S.vbuf[at.stream];
    MTLVertexFormat vf = vfmt(at.fmt); int nc = vf == MTLVertexFormatFloat2 ? 2 : vf == MTLVertexFormatFloat3 ? 3 : vf == MTLVertexFormatFloat4 ? 4 : 0;
    if (!nc || !sm.stride || sm.divisor || !vb.address) return 8;
    const uint8_t *ib = NULL; int isz = di ? (di->itype == 1 ? 2 : 4) : 0;
    if (di) { void *m; uint64_t o; if (!mtl_addr(di->ibuf, (uint64_t)n * isz, &m, &o) || !(ib = [(__bridge id<MTLBuffer>)m contents])) return 10; ib += o; }
    for (int k = 0; k < n; k++) {
        int64_t vid = dw ? (int64_t)dw->first + k : (isz == 2 ? (int64_t)((const uint16_t *)ib)[k] : (int64_t)((const uint32_t *)ib)[k]) + di->base_vertex;
        uint64_t off = (uint64_t)vid * sm.stride + at.offset;
        if (vid < 0 || off + 4 * nc > vb.size) return 9;
        void *mb; uint64_t mo; const uint8_t *base;
        if (!mtl_addr(vb.address + off, 4 * nc, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) return 10;
        float v[4] = {0, 0, 0, 1}; memcpy(v, base + mo, 4 * nc); memcpy(P[k], v, sizeof v);
    }
    return 0;
}
static int first_draw_covers(int i, id<MTLTexture> t, NSUInteger level) {
    const p_draw *dw = g_fd_dw; const p_draw_idx *di = g_fd_di;
    int n = dw ? dw->count : di->count, prim = dw ? dw->prim : di->prim;
    __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; if (!pr) return g_fd_why = 1, 0;
    NvShader *vs = pr->sh[0], *fs = pr->sh[1];
    if (!vs || !fs || vs->rf.pos_attr < 0 || fs->rf.may_discard || fs->rf.writes_smask || !(fs->rf.color_mask >> i & 1)) return g_fd_why = 2, 0;
    if (!t || t.sampleCount != 1 || t.textureType == MTLTextureType3D) return g_fd_why = 3, 0;
    if ((S.color.blend_en >> i & 1) || ((S.chmask.mask >> (4 * i)) & 0xf) != 0xf || S.ms.a2c || S.poly.mode != NVN_POLYGON_MODE_FILL) return g_fd_why = 4, 0;
    if (R.rt.depth && ((S.ds.dt && S.ds.df != NVN_DEPTH_FUNC_ALWAYS) || S.ds.st)) return g_fd_why = 5, 0;
    if (n < 3 || n > 64) return g_fd_why = 6, 0;
    float P[64][4]; int why = attr_read(vs->rf.pos_attr, dw, di, P);
    if (why) return g_fd_why = why, 0;
    if (vs->rf.pos_w1) for (int k = 0; k < n; k++) P[k][3] = 1.0f;   // 位置 w 为常数 1（不取属性 .w） | Position w is the constant 1 (the attribute's .w is not taken)
    int sc[4] = {R.sc.x, R.sc.y, R.sc.w, R.sc.h}; float vp[4] = {S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]};
    g_fd_why = 11;
    return nvn_cover((const float (*)[4])P, n, prim, (int)S.poly.cull, S.poly.front == NVN_FRONT_FACE_CCW,
                     S.has_vp, vp, R.has_sc, sc, (long)MAX(t.width >> level, 1), (long)MAX(t.height >> level, 1));
}

static int state_op(uint32_t op, const void *a) {
    switch (op) {
        case OP_SCISSOR: R.sc = *(const p_scissor *)a; R.has_sc = 1; break;
        case OP_BIND_PROGRAM: { const prog_t *np = ((const p_bind_program *)a)->prog; if (np != S.prog) { S.prog = np; g_pso_dirty = 1; args_program_changed(); } break; }
        case OP_BIND_STATE: {
            const p_bind_state *b = a; void *dst = NULL; size_t n = 0;
            blend_st bs;
            switch (b->kind) {
                case 0: memcpy(&bs, b->bytes, sizeof bs); dst = &S.blend[bs.target]; n = sizeof bs; break;
                case 1: dst = &S.color; n = sizeof S.color; break;
                case 2: dst = &S.chmask; n = sizeof S.chmask; break;
                case 3: dst = &S.ms; n = sizeof S.ms; break;
                case 4: dst = &S.poly; n = sizeof S.poly; break;
                case 5: dst = &S.ds; n = sizeof S.ds; break;
            }
            if (dst && memcmp(dst, b->bytes, n)) { memcpy(dst, b->bytes, n); if (b->kind == 5) g_dss_dirty = 1; else g_pso_dirty = 1; }
            break;
        }
        case OP_BIND_VATTR: { const p_bind_vattr *b = a; if (S.nattr != b->n || memcmp(S.attr, b->st, 4 * b->n)) { memcpy(S.attr, b->st, 4 * b->n); S.nattr = b->n; g_pso_dirty = 1; } break; }
        case OP_BIND_VSTREAM: { const p_bind_vstream *b = a; if (S.nstream != b->n || memcmp(S.stream, b->st, 8 * b->n)) { memcpy(S.stream, b->st, 8 * b->n); S.nstream = b->n; g_pso_dirty = 1; } break; }
        case OP_BIND_VBUF: { const p_bind_vbuf *b = a; memcpy(&S.vbuf[b->first], b->r, sizeof *b->r * b->n); break; }
        case OP_BIND_UBO: { const p_bind_buf *b = a; if (b->stage < 6 && b->index < 16 && memcmp(&S.ubo[b->stage][b->index], b, sizeof *b)) { S.ubo[b->stage][b->index] = *b; args_dirty_stage(b->stage); } break; }
        case OP_BIND_SSBO: { const p_bind_buf *b = a; if (b->stage < 6 && b->index < 16 && memcmp(&S.ssbo[b->stage][b->index], b, sizeof *b)) { S.ssbo[b->stage][b->index] = *b; args_dirty_stage(b->stage); } break; }
        case OP_BIND_TEX: { const p_bind_handles *b = a; if (b->stage < 6 && b->first + b->n <= 128 && memcmp(&S.tex[b->stage][b->first], b->h, 8 * b->n)) { memcpy(&S.tex[b->stage][b->first], b->h, 8 * b->n); args_dirty_stage(b->stage); if (b->first + b->n > g_tex_hi[b->stage]) g_tex_hi[b->stage] = b->first + b->n; } break; }
        case OP_BIND_SAMP: { const p_bind_handles *b = a; if (b->stage < 6 && b->first + b->n <= 128 && memcmp(&S.samp[b->stage][b->first], b->h, 8 * b->n)) { memcpy(&S.samp[b->stage][b->first], b->h, 8 * b->n); args_dirty_stage(b->stage); } break; }
        case OP_BIND_IMG: { const p_bind_handles *b = a; if (b->stage < 6 && b->first >= 0 && b->first + b->n <= 32) { if (memcmp(&S.img[b->stage][b->first], b->h, 8 * b->n)) { memcpy(&S.img[b->stage][b->first], b->h, 8 * b->n); args_dirty_stage(b->stage); } } else RED_ONCE(31, "图像槽越界 阶段 %d 起 %d 数 %d", b->stage, b->first, b->n); break; }
        case OP_STENCIL: {
            const p_stencil *b = a;
            if (b->which != 2) g_dss_dirty = 1;
            for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) S.st_vmask[f] = b->value_mask; else if (b->which == 1) S.st_wmask[f] = b->mask; else S.st_ref[f] = b->ref; }
            break;
        }
        case OP_POLY_OFFSET: S.po = *(const p_poly_offset *)a; break;
        case OP_VIEWPORTS: { const p_viewports *b = a; for (int i = 0; i < b->n; i++) memcpy(S.vp[b->first + i], b->v[i], 16); S.has_vp = 1; break; }
        case OP_DEPTH_RANGE: { const p_depth_range *b = a; S.dr[0] = b->n; S.dr[1] = b->f; break; }
        case OP_BLEND_COLOR: memcpy(S.blend_color, ((const p_blend_color *)a)->rgba, 16); break;
        case OP_COPY_STRIDE: S.copy_stride = ((const p_copy_stride *)a)->stride; break;
        case OP_DEPTH_CLAMP: S.depth_clamp = ((const p_flag *)a)->value; break;
        case OP_RENDER_ENABLE: S.render_disabled = !((const p_flag *)a)->value; break;
        case OP_RENDER_COND: S.render_disabled = !cond_eval(a, (const uint8_t *)a - 8); PD("E cond addr=%#llx mode=%d -> %s", (unsigned long long)((const p_render_cond *)a)->addr, ((const p_render_cond *)a)->mode, S.render_disabled ? "disabled" : "enabled"); break;
        case OP_RASTER_DISCARD: S.raster_discard = ((const p_flag *)a)->value; break;
        default: return 0;
    }
    return 1;
}

static void rt_sub(const p_set_rt *r, int k, uint64_t *ser, int *lv, int *sl);

// 并行预编的预测（见 pso_get 前的说明）：从当前操作之后扫到本次提交结束，只重放状态；遇到绘制就按「那时的状态 + 那个通道的附件格式」算管线键，没建过的丢后台。
// 通道实际挂不挂深度 / 模板要到回放时才定（纯深度 / 纯模板 / 都不挂的省略），所以每种可能的挂法各算一条键——猜多了只是后台白编，不影响结果。
// Prediction for parallel precompilation (see the note before pso_get): scan from after the current operation to the end of this submit, replaying only state; at a draw, compute the pipeline key from "the state at that time + that render pass's attachment formats" and dispatch those not yet built to the background.
// Whether a render pass actually attaches depth / stencil is only decided at replay (depth-only / stencil-only / neither attached are omitted), so one key is computed for each possible attach mode -- guessing too many only means wasted background compiles and does not affect the result.
static void pso_predict_rest(void) {
    static int enabled = -1; if (enabled < 0) { const char *e = getenv("TOTK_PSO_PREDICT"); enabled = !e || atoi(e); }
    if (!enabled) return;
    if (!g_exec_next) return;
    static typeof(S) Ssave; memcpy(&Ssave, &S, sizeof S); typeof(R) Rsave = R;
    int pso_dirty = g_pso_dirty, dss_dirty = g_dss_dirty;
    uint32_t cf[8]; memcpy(cf, g_pass.cfmt, sizeof cf); uint32_t df = g_pass.dfmt, sf = g_pass.sfmt, samples = g_pass.samples, dfull = df, sfull = sf; int kicked0 = (int)g_pp_kicked, draws = 0;
    fz_cursor cur = {.p = g_exec_next};
    // 先收齐再丢：按「附件原样挂」的键排在前（最可能用到），其余挂法排在后——后台并发有限，先丢的先编完
    // Collect them all first, then dispatch: keys for "attachments attached as-is" go first (most likely to be used), the other attach modes after -- background concurrency is limited, and what is dispatched first is compiled first
    enum { QN = 160 }; static struct { pso_key k; void *vs, *fs; } qa[QN], qb[QN]; int na = 0, nb = 0;
    for (int guard = 0; guard < 200000 && na < QN; guard++) {
        const uint8_t *q; op_hdr hq; const void *aq;
        if (!fz_next(&cur, &q, &hq, &aq)) break;
        if (hq.op == OP_SET_RT) {
            const p_set_rt *r = aq; memset(cf, 0, sizeof cf); dfull = sfull = 0; samples = 1;
            /* Prediction must not acquire/recreate/upload textures on the active GPU timeline. */
            for (int i = 0; i < r->n && i < 8; i++) { MTLTextureDescriptor *d = r->color[i] ? tex_desc(r->color[i], tex_mirrored(r->color[i])) : nil;
                if (d) { cf[i] = (uint32_t)attachment_format(r->color[i], &r->cview[i], r->has_cview[i], d.pixelFormat); samples = (uint32_t)d.sampleCount; } }
            for (int i = 0; i < 8; i++) if (pf_block_compressed(cf[i])) memset(cf, 0, sizeof cf), r = NULL;   // TOTK: such passes are handled (or reported) when they begin
            if (!r) { cf[0] = 0; df = sf = dfull = sfull = 0; continue; }
            MTLTextureDescriptor *dd = r->depth ? tex_desc(r->depth, tex_mirrored(r->depth)) : nil;
            if (dd) { MTLPixelFormat f = attachment_format(r->depth, &r->dview, r->has_dview, dd.pixelFormat); samples = (uint32_t)dd.sampleCount;
                dfull = f != MTLPixelFormatStencil8 ? (uint32_t)f : 0; sfull = (f == MTLPixelFormatStencil8 || f == MTLPixelFormatDepth32Float_Stencil8) ? (uint32_t)f : 0; }
            df = dfull; sf = sfull; continue;
        }
        if (hq.op == OP_DRAW || hq.op == OP_DRAW_IDX) {
            draws++;
            __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; if (!pr || !pr->sh[0]) continue;
            __unsafe_unretained NvShader *vs = pr->sh[0], *fs = pr->sh[1];
            const uint32_t dv[4] = {dfull, 0, 0, dfull}, sv[4] = {sfull, 0, sfull, 0};
            for (int v = 0; v < 4; v++) {
                if (v && dv[v] == dv[0] && sv[v] == sv[0]) continue;
                if (v == 2 && !sfull) continue; if (v == 3 && (!dfull || !sfull)) continue;
                pso_key k; pso_key_fill(&k, vs, fs, cf, dv[v], sv[v], samples, 0);
                if (hm_get_id(&g_pso, &k)) continue;
                int dup = 0; for (int i = 0; i < (v ? nb : na) && !dup; i++) dup = !memcmp(&(v ? qb : qa)[i].k, &k, sizeof k);
                if (dup) continue;
                if (!v && na < QN) { qa[na].k = k; qa[na].vs = (__bridge void *)vs; qa[na].fs = (__bridge void *)fs; na++; }
                else if (v && nb < QN) { qb[nb].k = k; qb[nb].vs = (__bridge void *)vs; qb[nb].fs = (__bridge void *)fs; nb++; }
            }
            continue;
        }
        if (hq.op == OP_CLEAR_COLOR || hq.op == OP_CLEAR_DS || hq.op == OP_RENDER_COND) continue;
        state_op(hq.op, aq);
    }
    (void)df; (void)sf;
    for (int i = 0; i < na; i++) pso_pend_kick(&qa[i].k, (__bridge NvShader *)qa[i].vs, (__bridge NvShader *)qa[i].fs);
    for (int i = 0; i < nb; i++) pso_pend_kick(&qb[i].k, (__bridge NvShader *)qb[i].vs, (__bridge NvShader *)qb[i].fs);
    memcpy(&S, &Ssave, sizeof S); R = Rsave; args_dirty_all(); g_pso_dirty = pso_dirty; g_dss_dirty = dss_dirty;
    static int logged; if (g_pp_kicked > (uint64_t)kicked0 && logged++ < 20) MLOG("并行预编：本次提交余下 %d 次绘制，丢给后台 %d 条管线", draws, (int)g_pp_kicked - kicked0);
}

/* Prediction visits later scissor commands but must leave the active draw's state untouched. */
void dm_nvn_probe_prediction(void) {
    typeof(R) saved = R; const uint8_t *next = g_exec_next;
    struct { op_hdr sc; p_scissor rect; op_hdr end; } ops = {
        {OP_SCISSOR, 8 + sizeof(p_scissor)}, {101, 202, 303, 404}, {OP_END, 8}
    };
    R.sc = (p_scissor){3, 5, 7, 11}; R.has_sc = 0;
    g_exec_next = (const uint8_t *)&ops;
    pso_predict_rest();
    p_scissor expected = {3, 5, 7, 11};
    if (R.has_sc || memcmp(&R.sc, &expected, sizeof expected)) MDIE("pipeline prediction changed active scissor");
    struct {op_hdr h;p_render_cond c;op_hdr end;} cond={{OP_RENDER_COND,8+sizeof(p_render_cond)},{0},{OP_END,8}};
    const void *op=&cond;unsigned slot=((uintptr_t)op>>3)&1023;cond_cache_ent cache=g_cond_cache[slot];uint64_t unknown=g_draw_skip[14];
    g_exec_next=(const uint8_t *)&cond;pso_predict_rest();
    if(getenv("TOTK_PROBE_NEGATIVE"))state_op(OP_RENDER_COND,&cond.c);
    if(memcmp(&cache,&g_cond_cache[slot],sizeof cache)||g_draw_skip[14]!=unknown)MDIE("pipeline prediction evaluated a future conditional counter");
    R = saved; g_exec_next = next;
}

static void fz_probe_cov2(const p_set_rt *r2, const uint8_t *first, const int8_t *p2map, const uint8_t *clr, int n1, int n) {
    static typeof(S) Ssave; memcpy(&Ssave, &S, sizeof S); typeof(R) Rsave = R;
    const p_draw *fdw = g_fd_dw; const p_draw_idx *fdi = g_fd_di; int fdwhy = g_fd_why;
    int pso_dirty = g_pso_dirty, dss_dirty = g_dss_dirty;
    op_hdr fh; memcpy(&fh, first, sizeof fh);
    g_fd_dw = fh.op == OP_DRAW ? (const p_draw *)(first + 8) : NULL;
    g_fd_di = fh.op == OP_DRAW_IDX ? (const p_draw_idx *)(first + 8) : NULL;
    R.rt = *r2; R.has_rt = 1;
    fz_cursor cur = {.p = g_exec_next};
    for (;;) {
        const uint8_t *q; op_hdr hq; const void *aq;
        if (!fz_next(&cur, &q, &hq, &aq)) break;
        if (q == first) break;
        if (hq.op == OP_DRAW || hq.op == OP_DRAW_IDX || hq.op == OP_SET_RT || hq.op == OP_CLEAR_COLOR || hq.op == OP_CLEAR_DS) continue;
        state_op(hq.op, aq);
    }
    for (int j = n1; j < n && j < 8; j++) {
        int k = -1;
        for (int q = 0; q < 8; q++) if (p2map[q] == j) { k = q; break; }
        if (k < 0 || clr[k] || !r2->color[k]) continue;
        uint64_t ser; int level, slice;
        rt_sub(r2, k, &ser, &level, &slice);
        (void)ser; (void)slice;
        g_fz.cov2[j] = (uint8_t)first_draw_covers(k, tex_mtl(r2->color[k]), (NSUInteger)level);
    }
    memcpy(&S, &Ssave, sizeof S); args_dirty_all(); R = Rsave;
    g_fd_dw = fdw; g_fd_di = fdi; g_fd_why = fdwhy;
    g_pso_dirty = pso_dirty; g_dss_dirty = dss_dirty;
}

static int fz_load_covers(int i, id<MTLTexture> t, NSUInteger level) {
    if (g_fz.active && g_fz.phase == 1 && i >= g_fz.n1) return g_fz.cov2[i];
    return first_draw_covers(i, t, level);
}

// 探针收尾：每个被改成清哨兵色的附件回读该级 / 层，与同格式 1x1 哨兵像素逐像素比，数剩余（> 0 即覆盖证明有误，报红）
// Probe wrap-up: for every attachment changed to clear to the sentinel color, read back that level / layer, compare pixel by pixel with a same-format 1x1 sentinel pixel, and count what remains (> 0 means the coverage proof is wrong, report a failure)
static void lp_finish(void) {
    id<MTLCommandBuffer> dcb = q_cb();
    NSArray *ps = g_lp; g_lp = nil;
    for (NSArray *e in ps) {
        id<MTLTexture> t = e[0]; NSUInteger lv = [e[1] unsignedIntegerValue], sl = [e[2] unsignedIntegerValue];
        NSUInteger W = MAX(t.width >> lv, 1), H = MAX(t.height >> lv, 1), bpp = (NSUInteger)(att_bytes(t, lv) / (W * H));
        MTLTextureDescriptor *rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat width:1 height:1 mipmapped:NO];
        rd.usage = MTLTextureUsageRenderTarget; rd.storageMode = MTLStorageModePrivate;
        id<MTLTexture> ref = [g_dev newTextureWithDescriptor:rd];
        MTLRenderPassDescriptor *pd = [MTLRenderPassDescriptor renderPassDescriptor];
        pd.colorAttachments[0].texture = ref; pd.colorAttachments[0].loadAction = MTLLoadActionClear; pd.colorAttachments[0].storeAction = MTLStoreActionStore;
        pd.colorAttachments[0].clearColor = [e[6] intValue] == 3 ? MTLClearColorMake(NAN, NAN, NAN, NAN) : [e[6] intValue] == 2 ? MTLClearColorMake(0.777, 0.222, 0.555, 0.888) : MTLClearColorMake(0.123, 0.456, 0.789, 0.321);
        [[dcb renderCommandEncoderWithDescriptor:pd] endEncoding];
        id<MTLBuffer> bt = [g_dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared], br = [g_dev newBufferWithLength:bpp options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> b = [dcb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:sl sourceLevel:lv sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1) toBuffer:bt destinationOffset:0 destinationBytesPerRow:W * bpp destinationBytesPerImage:W * H * bpp];
        [b copyFromTexture:ref sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(1, 1, 1) toBuffer:br destinationOffset:0 destinationBytesPerRow:bpp destinationBytesPerImage:bpp];
    enc_b_end(b);
        unsigned long long pass = [e[3] unsignedLongLongValue], ser = [e[5] unsignedLongLongValue]; int idx = [e[4] intValue];
        [dcb addCompletedHandler:^(id<MTLCommandBuffer> c) { (void)c;
            const uint8_t *p = bt.contents, *r = br.contents; uint64_t left = 0; char at[96] = ""; size_t ao = 0;
            for (NSUInteger k = 0; k < W * H; k++) if (!memcmp(p + k * bpp, r, bpp)) { if (left++ < 4) ao += snprintf(at + ao, sizeof at - ao, " (%lu,%lu)", (unsigned long)(k % W), (unsigned long)(k / W)); }
            if (left) MLOG("【红】读入可省探针：通道 %llu c%d ser=%llu lv%lu sl%lu %lux%lu 剩哨兵像素 %llu，前几个位置%s", pass, idx, ser, (unsigned long)lv, (unsigned long)sl, (unsigned long)W, (unsigned long)H, (unsigned long long)left, at);
            else MLOG("读入可省探针：通道 %llu c%d ser=%llu lv%lu sl%lu %lux%lu 全覆盖（哨兵剩 0）", pass, idx, ser, (unsigned long)lv, (unsigned long)sl, (unsigned long)W, (unsigned long)H);
        }];
    }
    cb_commit(dcb);
}
/* TOTK (2026-10-17): the replay state that outlives a submit and holds texture copies of recordings (hle_nvn.c pin_sweep):
   the current render targets, which the next submit's first SetRenderTargets compares by serial (rt_same). Everything
   else holding them (cube groups, deferred and fused passes, layer merges) ends with its pass or submit. */
void mtl_pin_refs(void (*mark)(const void *)) {
    if (!R.has_rt) return;
    for (int i = 0; i < R.rt.n && i < 8; i++) mark(R.rt.color[i]);
    mark(R.rt.depth);
}
void mtl_probe_pin_rt(const tex_t *t) { memset(&R.rt, 0, sizeof R.rt); R.has_rt = t != NULL; R.rt.n = t ? 1 : 0; R.rt.color[0] = t; }
static int rt_same(const p_set_rt *a, const p_set_rt *b) {   // 同一组目标：按贴图序号与视图比（不比对象指针：游戏可能按值拷贝贴图对象） | Same set of targets: compare by texture serial and view (not by object pointer: the game may copy texture objects by value)
    if (a->n != b->n || !a->depth != !b->depth || (a->depth && a->depth->serial != b->depth->serial) || a->has_dview != b->has_dview) return 0;
    if (a->has_dview && memcmp(&a->dview, &b->dview, sizeof a->dview)) return 0;
    for (int k = 0; k < a->n && k < 8; k++) {
        if (!a->color[k] != !b->color[k] || (a->color[k] && a->color[k]->serial != b->color[k]->serial) || a->has_cview[k] != b->has_cview[k]) return 0;
        if (a->has_cview[k] && memcmp(&a->cview[k], &b->cview[k], sizeof a->cview[k])) return 0;
    }
    return 1;
}
// 实验RT范围候选：只忽略实际渲染附件未使用的正nlevels差异，其它字段保持逐字相同。
// Experimental RT range candidate: only ignore positive nlevels differences not used by the actual render attachments; other fields stay word-for-word identical.
#define NVN_RT_RANGE_IMPL 1
static int rt_range_shape(const p_set_rt *a, const p_set_rt *b) {
    if (a->n != 1 || b->n != 1 || !a->color[0] || !b->color[0] || !a->has_cview[0] || !b->has_cview[0] ||
        !a->depth != !b->depth || a->has_dview != b->has_dview ||
        (a->has_dview && memcmp(&a->dview,&b->dview,sizeof a->dview))) return 0;
    const tex_t *x=a->color[0], *y=b->color[0];
    if (!x->serial || x->serial != y->serial || x->d.target != NVN_TEXTURE_TARGET_2D || y->d.target != NVN_TEXTURE_TARGET_2D ||
        x->d.samples > 1 || y->d.samples > 1 || tex_mirrored(x) || tex_mirrored(y) ||
        memcmp(&x->d,&y->d,sizeof x->d) || x->mem != y->mem || x->pool != y->pool || x->off != y->off) return 0;
    const view_t *v=&a->cview[0], *w=&b->cview[0]; int levels=x->d.levels;
    if (!v->set_levels || !w->set_levels || v->base_level < 0 || v->base_level >= levels ||
        v->nlevels <= 0 || v->nlevels > levels-v->base_level || w->nlevels <= 0 || w->nlevels > levels-v->base_level ||
        (v->set_layers && (v->min_layer || v->nlayers != 1)) || (v->set_format && v->format != x->d.format) ||
        (v->set_target && v->target != NVN_TEXTURE_TARGET_2D)) return 0;
    view_t cv=*w; cv.nlevels=v->nlevels;
    if (memcmp(v,&cv,sizeof cv)) return 0;
    if (a->depth) { const tex_t *d=a->depth,*e=b->depth;
        if (!d->serial || d->serial != e->serial || d->d.target != NVN_TEXTURE_TARGET_2D || e->d.target != NVN_TEXTURE_TARGET_2D ||
            d->d.samples > 1 || e->d.samples > 1 || tex_mirrored(d) || tex_mirrored(e) ||
            memcmp(&d->d,&e->d,sizeof d->d) || d->mem != e->mem || d->pool != e->pool || d->off != e->off) return 0;
        int dl=a->has_dview && a->dview.set_levels ? a->dview.base_level : 0;
        if (dl < 0 || dl >= d->d.levels || mip_extent(d->d.width,dl) < mip_extent(x->d.width,v->base_level) ||
            mip_extent(d->d.height,dl) < mip_extent(x->d.height,v->base_level)) return 0; }

    return v->nlevels != w->nlevels;
}
#define NVN_RT_RANGE_DEPTH_IMPL 1
static void rt_range_note_depth(void) { if (rt_range_on() && S.ds.dw) g_pass.range_depth_nonro=1; }
static int rt_range_same(const p_set_rt *a, const p_set_rt *b) {
    if (!rt_range_on() || !g_enc || g_fz.active || g_pass.range_external || g_pass.range_depth_nonro || !S.ds.dt || S.ds.dw || S.ds.st || g_enc_mode != 4 || !g_pass.hd || g_pass.hs ||
        (g_vis_on && !g_enc_has_vis) || g_pass.reg_gen != g_reg_gen || g_pass.ser_gen != g_ser_gen || g_pass.range_dirty != nvn_dirty_now() || !rt_range_shape(a,b) || !b->depth) return 0;
    id<MTLTexture> color=tex_mtl(b->color[0]), depth=tex_mtl(b->depth);
    int lv=b->cview[0].base_level;
    return mip_extent(color.width,lv) == g_pass.tw && mip_extent(color.height,lv) == g_pass.th &&
        color == g_pass.range_color && depth == g_pass.range_depth && color.textureType == MTLTextureType2D &&
        depth.textureType == MTLTextureType2D && color.sampleCount == 1 && depth.sampleCount == 1 &&
        g_pass.reg_gen == g_reg_gen && g_pass.ser_gen == g_ser_gen && g_pass.range_dirty == nvn_dirty_now();
}
static int draw_samples_rt(NvProg *pr) {   // 采样附件保持原ser级保守guard，不分级/层 | Sampled attachments keep the original conservative ser-level guard, not split by level/layer
    uint64_t rs[9]; int nr = 0;
    for (int k = 0; k < R.rt.n && k < 8; k++) if (R.rt.color[k]) rs[nr++] = R.rt.color[k]->serial;
    if (R.rt.depth) rs[nr++] = R.rt.depth->serial;
    if (!pr) return 1;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h;
            if(b->kind==NVN_RB_TEXU){
                for(unsigned element=0;element<b->count;element++){
                    unsigned tid=uniform_texture_handle(b,st,element)&0xfffff;if(!tid)continue;
                    registry_lock();uint64_t ser=tid<(unsigned)g_ntreg&&g_treg[tid].ok?g_treg[tid].t.serial:0;registry_unlock();
                    for(int j=0;j<nr;j++)if(ser&&ser==rs[j])return 1;
                }
                continue;
            }
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 && sh->stage < 6 ? S.img[sh->stage][b->reg] : 0; else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 && sh->stage < 6 ? S.tex[sh->stage][b->reg] : 0; else continue;
            int tid = (int)(h & 0xfffff); if (!tid) continue;
            registry_lock(); uint64_t ser = tid < g_ntreg && g_treg[tid].ok ? g_treg[tid].t.serial : 0; registry_unlock();
            for (int j = 0; j < nr; j++) if (ser && ser == rs[j]) return 1; } }
    return 0;
}
// 整通道对深度 / 模板的需求（位：1 深度 2 模板；3 = 两项都要或证不了）。从开通道的这次绘制起，按真实回放顺序扫到通道结束：
// 挂上的清除与本次绘制的状态先计入；之后的绘制按其深度模板状态累计；中途清除、嵌套过深等证不了的一律按两项都要
// The whole render pass's need for depth / stencil (bits: 1 depth 2 stencil; 3 = both needed or cannot be proven). Starting from the draw that opens the render pass, scan in real replay order to the end of the render pass:
// the attached clears and this draw's state are counted first; later draws accumulate by their depth-stencil state; anything unprovable such as a mid-pass clear or nesting too deep is treated as both needed
/* TOTK: a fragment function with a [[depth]] output needs the pass to keep its depth attachment (Metal refuses the pipeline otherwise),
   even when depth test and write are off. */
static int prog_ps_writes_depth(const prog_t *pg) { if (!pg) return 0; __unsafe_unretained NvProg *pr = prog_get(pg); return pr && pr->sh[1] && pr->sh[1]->writes_depth; }
static int pass_ds_need(void) {
    int need = (R.clear_d || S.ds.dt || S.ds.dw || prog_ps_writes_depth(S.prog) ? 1 : 0) | (R.clear_s || S.ds.st ? 2 : 0);
    if (!R.rt.depth || need == 3 || !g_exec_next || g_ret_n > 16) {
        PD("S %llu 不前瞻 depth=%d clear_d=%d clear_s=%d dt=%d dw=%d st=%d next=%d ret=%d", g_stat_pass + 1, R.rt.depth != NULL, R.clear_d, R.clear_s, (int)S.ds.dt, (int)S.ds.dw, (int)S.ds.st, g_exec_next != NULL, g_ret_n);
        return 3;
    }
    ds_st ds = S.ds; g_ds_why = -1; g_ds_scanned = 0; g_ds_need = need;
    int r = ds_scan(g_exec_next, &ds, 0);
    for (int k = g_ret_n - 1; r == DS_RET && k >= 0; k--) r = ds_scan(g_ret[k], &ds, 0);
    for (int k = 0; r == DS_RET && k < g_nrest; k++) r = ds_scan((const uint8_t *)g_rest[k], &ds, 0);
    int out = r == DS_USED ? 3 : g_ds_need;
    PD("S %llu 前瞻 结果=%d 需求=%d（1 深度 2 模板）原因=%d 扫过 %d 条 返回栈 %d 余句柄 %d", g_stat_pass + 1, r, out, g_ds_why, g_ds_scanned, g_ret_n, g_nrest);
    return out;
}
// 纯清除通道前提：开通道时向后扫，形如「本通道若干绘制 → 若干组（SetRenderTargets + 只有清除）→ 回到与本通道相同的目标」，
// 就把这几组清除提到本通道之前执行、原位置跳过，回到同一目标时由同目标续用接上——本通道的附件不再为夹在中间的清除整块写回再读入
// （选角页：grunge 与粒子都画 28 / 27 第 0 级 + 深度，中间只清第 1、2 级）。
// 等价条件（全部满足才前提，否则照原样）：扫描区间只有绘制、状态与这些清除，没有拷贝 / 派发 / 清贴图 / 子列表；被清的子资源（贴图 + 级 + 层）
// 与本通道附件不相交；被清的贴图既不在当前任一贴图 / 图像槽上、扫描区间里也没被绑上（保守：不论程序采不采样）；清除时的剪裁覆盖整块（同原位置执行的要求）
// Pure-clear render pass hoisting: when a render pass is opened, scan forward, and if the shape is "several draws of this render pass -> several groups (SetRenderTargets + only clears) -> back to the same targets as this render pass",
// then hoist those groups of clears to execute before this render pass and skip them at their original positions, and on returning to the same targets the same-target continuation picks up -- this render pass's attachments are no longer stored whole and loaded again for the clears sandwiched in between
// (character select page: grunge and particles both draw 28 / 27 level 0 + depth, with only levels 1 and 2 cleared in between).
// Equivalence conditions (hoist only if all hold, otherwise leave as-is): the scan range has only draws, state and these clears, no copies / dispatches / texture clears / sub-lists; the cleared subresources (texture + level + layer)
// are disjoint from this render pass's attachments; the cleared textures are neither on any current texture / image slot nor bound within the scan range (conservative: regardless of whether the program samples them); the scissor at the time of the clear covers the whole block (same requirement as executing at the original position)
static uint64_t h_ser(uint64_t h) { int tid = (int)(h & 0xfffff); return tid ? atomic_load_explicit(&g_tser[tid], memory_order_relaxed) : 0; }
// ---------- 写回前瞻索引：提交开始按回放顺序线性扫一遍，按贴图记事件；通道结束时每个附件二分找「之后第一个有效事件」定论 ----------
// 取代原「每个通道结束向后整段扫」（通道数 × 命令数，营地每帧 49 通道）。与原扫描逐附件对拍：画面闸门 17 页 + 营地局约 34 万次附件判定（多余约 4.6 万）0 不一致后删去原扫描。
// 事件（seq = 回放序号）：READ 绘制 / 派发按程序反射读到该贴图；ATT 绘制时它是当前附件；CLEAR 作附件被整块清除；COPY / CLRTEX 作拷贝两端 / 被清贴图。
// 判定（首个有效事件定论）：READ / COPY / CLRTEX → 需要；ATT 且其 SetRenderTargets 在通道结束之后 → 需要（读入）；CLEAR 且同上 → 多余；无事件 → 需要。
// 任何不确定（回放序号与索引对不上、嵌套过深）索引作废，本次提交余下通道一律写回（写回总是正确的，只是少省带宽）。
// ---------- Store lookahead index: at the start of a submit do one linear scan in replay order and record events per texture; when a render pass ends, binary-search each attachment for "the first valid event afterwards" to decide ----------
// Replaces the original "scan the whole remainder forward at every render pass end" (pass count x command count, 49 render passes per frame in camp). Cross-checked per attachment against the original scan: after 0 mismatches over the 17 pages of the screenshot gate + a camp run with about 340,000 attachment decisions (about 46,000 redundant), the original scan was removed.
// Events (seq = replay sequence number): READ a draw / dispatch reads this texture per program reflection; ATT it is a current attachment at a draw; CLEAR cleared as a whole block while an attachment; COPY / CLRTEX either end of a copy / the cleared texture.
// Decision (the first valid event decides): READ / COPY / CLRTEX -> needed; ATT whose SetRenderTargets is after the render pass end -> needed (load); CLEAR with the same -> redundant; no event -> needed.
// On any uncertainty (replay sequence number does not match the index, nesting too deep) the index is invalidated and all remaining render passes of this submit store (storing is always correct, it just saves less bandwidth).
typedef struct { uint32_t seq, rtseq; uint64_t ser; const uint8_t *op; uint16_t lv, sl; uint8_t kind, plane, anyls, why; } sdx_ev;   // plane 0 颜色 8 深度模板（ATT）1 深度 2 模板（CLEAR） | plane 0 color 8 depth-stencil (ATT) 1 depth 2 stencil (CLEAR)
enum { SDX_READ = 1, SDX_NEED, SDX_ATT, SDX_CLEAR };
#define SDX_NONE 0xffffffffu
static sdx_ev *g_sdx, *g_sdx_tmp; static uint32_t g_sdx_n, g_sdx_cap; static int g_sdx_ok;   // 索引有效 | Index valid
// 候选贴图（当过附件的）→ 桶：开放定址；事件发生时尚未当过附件的贴图，对此前结束的任何通道都无关，不记
// Candidate textures (those that have been attachments) -> buckets: open addressing; a texture that has not yet been an attachment when an event occurs is irrelevant to any render pass that ended before then, so it is not recorded
#define SDX_HB 2048
static uint64_t g_sdx_hk[SDX_HB]; static uint16_t g_sdx_hb[SDX_HB]; static uint32_t g_sdx_nb, g_sdx_bcnt[SDX_HB + 1], g_sdx_bst[SDX_HB + 1];
static int sdx_bucket(uint64_t ser, int add) {   // 返回桶号；不在集合（且不加）返回 −1；满了作废索引 | Returns the bucket number; returns -1 if not in the set (and not adding); invalidates the index when full
    if (!ser) return -1;
    for (uint32_t i = (uint32_t)(ser * 0x9E3779B97F4A7C15ull >> 53), n = 0; n < SDX_HB; i = (i + 1) & (SDX_HB - 1), n++) {
        if (g_sdx_hk[i] == ser) return g_sdx_hb[i];
        if (!g_sdx_hk[i]) { if (!add) return -1; if (g_sdx_nb >= SDX_HB / 2) { g_sdx_ok = 0; return -1; } g_sdx_hk[i] = ser; g_sdx_hb[i] = (uint16_t)g_sdx_nb; return (int)g_sdx_nb++; }
    }
    return -1;
}
static const uint8_t **g_sdx_op; static uint32_t g_sdx_nop, g_sdx_opcap;
static uint32_t g_exec_seq;   // 回放中当前操作的序号 | Sequence number of the current operation during replay
static uint64_t g_pp_slot[6][160]; static const prog_t *g_pp_prog; static const p_set_rt *g_pp_rt; static uint32_t g_pp_rtseq;
static uint32_t *g_rtx_seq, *g_drx_seq; static uint32_t g_rtx_n, g_rtx_cap, g_drx_n, g_drx_cap;   // 本次提交的 SET_RT / 绘制序号表（可读 tile 合并的预筛用） | This submit's SET_RT / draw sequence number tables (used by the pre-screen of the readable tile merge)
static void seq_push(uint32_t **v, uint32_t *n, uint32_t *cap, uint32_t x) { if (*n == *cap) { *cap = *cap ? *cap * 2 : 1024; *v = realloc(*v, *cap * sizeof **v); } (*v)[(*n)++] = x; }
static void sdx_ev_add(sdx_ev e) { if (sdx_bucket(e.ser, 0) < 0) return; if (g_sdx_n == g_sdx_cap) { g_sdx_cap = g_sdx_cap ? g_sdx_cap * 2 : 4096; g_sdx = realloc(g_sdx, g_sdx_cap * sizeof *g_sdx); g_sdx_tmp = realloc(g_sdx_tmp, g_sdx_cap * sizeof *g_sdx); } g_sdx[g_sdx_n++] = e; }
static void sdx_lvsl(const view_t *v, int hv, uint16_t *lv, uint16_t *sl) { *lv = (uint16_t)(hv && v->set_levels ? v->base_level : 0); *sl = (uint16_t)(hv && v->set_layers ? v->min_layer : 0); }
static int g_sdx_ren = 1;   // TOTK: render-enable state along the walk (SetRenderEnable / conditional, see cond_eval)
static void sdx_walk(const uint8_t *p, int depth) {
    if (depth > 64) { g_sdx_ok = 0; return; }
    for (uint32_t guard = 0; guard < 4000000 && p && g_sdx_ok; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        if (g_sdx_nop == g_sdx_opcap) { g_sdx_opcap = g_sdx_opcap ? g_sdx_opcap * 2 : 16384; g_sdx_op = realloc(g_sdx_op, g_sdx_opcap * sizeof *g_sdx_op); }
        uint32_t seq = g_sdx_nop; g_sdx_op[g_sdx_nop++] = p;
        switch (h.op) {
            case OP_END: return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); sdx_walk((const uint8_t *)t, depth + 1); break; }
            case OP_SET_RT: { const p_set_rt *r = a; g_pp_rt = r; g_pp_rtseq = seq; seq_push(&g_rtx_seq, &g_rtx_n, &g_rtx_cap, seq);
                for (int k = 0; k < r->n && k < 8; k++) if (r->color[k]) sdx_bucket(r->color[k]->serial, 1);
                if (r->depth) sdx_bucket(r->depth->serial, 1);
                break; }
            case OP_RENDER_ENABLE: g_sdx_ren = ((const p_flag *)a)->value; break;
            case OP_RENDER_COND: g_sdx_ren = cond_eval(a, p); break;
            case OP_CLEAR_COLOR: { const p_clear_color *c = a; const p_set_rt *r = g_pp_rt;
                if (!g_sdx_ren) break;   // TOTK: this clear will not run
                if (!r || (c->mask & 0xf) != 0xf || c->index < 0 || c->index >= r->n || c->index >= 8 || !r->color[c->index]) break;
                uint16_t lv, sl; sdx_lvsl(&r->cview[c->index], r->has_cview[c->index], &lv, &sl); int last = -1;   // 同一子资源挂在多个序号时按最后一个算 | When the same subresource is attached at several indices, the last one counts
                for (int k = 0; k < r->n && k < 8; k++) { uint16_t l2, s2; if (!r->color[k] || r->color[k]->serial != r->color[c->index]->serial) continue; sdx_lvsl(&r->cview[k], r->has_cview[k], &l2, &s2); if (l2 == lv && s2 == sl) last = k; }
                if (last == c->index) sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->color[c->index]->serial, p, lv, sl, SDX_CLEAR, 0, 0, 0});
                break; }
            case OP_CLEAR_DS: { const p_clear_ds *c = a; const p_set_rt *r = g_pp_rt; if (!g_sdx_ren || !r || !r->depth) break;
                uint16_t lv, sl; sdx_lvsl(&r->dview, r->has_dview, &lv, &sl);
                if (c->depth_mask) sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->depth->serial, p, lv, sl, SDX_CLEAR, 1, !r->has_dview, 0});
                if ((c->stencil_mask & 0xff) == 0xff) sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->depth->serial, p, lv, sl, SDX_CLEAR, 2, !r->has_dview, 0});
                break; }
            case OP_BIND_TEX: case OP_BIND_IMG: { const p_bind_handles *b = a; int base = h.op == OP_BIND_IMG ? 128 : 0, lim = h.op == OP_BIND_IMG ? 32 : 128;
                if (b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) g_pp_slot[b->stage][base + b->first + k] = h_ser(b->h[k]);
                break; }
            case OP_BIND_PROGRAM: g_pp_prog = ((const p_bind_program *)a)->prog; break;
            case OP_DRAW: case OP_DRAW_IDX: case OP_DISPATCH: {
                if (h.op != OP_DISPATCH) seq_push(&g_drx_seq, &g_drx_n, &g_drx_cap, seq);
                __unsafe_unretained NvProg *pr = g_pp_prog ? prog_get(g_pp_prog) : nil;
                if (!pr) { for (int st = 0; st < 6; st++) for (int k = 0; k < 160; k++) if (g_pp_slot[st][k]) sdx_ev_add((sdx_ev){seq, 0, g_pp_slot[st][k], p, 0, 0, SDX_READ, 0, 0, 1}); }
                else for (int st = 0; st < 6; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || (h.op == OP_DISPATCH) != (st == 5)) continue;
                    for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; int k = b->kind == NVN_RB_IMG ? 128 + b->reg : (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) ? b->reg : -1;
                        if (k < 0 || k >= 160 || sh->stage < 0 || sh->stage >= 6) continue; uint64_t ser = g_pp_slot[sh->stage][k];
                        if (ser) sdx_ev_add((sdx_ev){seq, 0, ser, p, 0, 0, SDX_READ, 0, 0, 1}); } }
                const p_set_rt *r = g_pp_rt;
                if (h.op != OP_DISPATCH && r) {
                    for (int k = 0; k < r->n && k < 8; k++) if (r->color[k]) { uint16_t lv, sl; sdx_lvsl(&r->cview[k], r->has_cview[k], &lv, &sl); sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->color[k]->serial, p, lv, sl, SDX_ATT, 0, 0, 2}); }
                    if (r->depth) { uint16_t lv, sl; sdx_lvsl(&r->dview, r->has_dview, &lv, &sl); sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->depth->serial, p, lv, sl, SDX_ATT, 8, !r->has_dview, 2}); }
                }
                break; }
            case OP_COPY_T2T: { const p_copy_t2t *c = a; if (c->src) sdx_ev_add((sdx_ev){seq, 0, c->src->serial, p, 0, 0, SDX_NEED, 0, 0, 3}); if (c->dst) sdx_ev_add((sdx_ev){seq, 0, c->dst->serial, p, 0, 0, SDX_NEED, 0, 0, 3}); break; }
            case OP_COPY_T2B: case OP_COPY_B2T: { const p_copy_b2t *c = a; if (c->tex) sdx_ev_add((sdx_ev){seq, 0, c->tex->serial, p, 0, 0, SDX_NEED, 0, 0, 3}); break; }
            case OP_CLEAR_TEX: { const p_clear_tex *c = a; if (c->tex) sdx_ev_add((sdx_ev){seq, 0, c->tex->serial, p, 0, 0, SDX_NEED, 0, 0, 4}); break; }
            default: break;
        }
        p += h.size;
    }
}
void mtl_sub_index(const uint64_t *h, int n) {   // 提交开始（hle_nvn.c QueueSubmitCommands）：建本次提交的写回前瞻索引 | Start of submit (hle_nvn.c QueueSubmitCommands): build this submit's store lookahead index
    g_xv_ntouch = 0; g_sdx_n = g_sdx_nop = 0; g_rtx_n = g_drx_n = 0; g_exec_seq = 0; g_sdx_ok = tbdr_on();
    if (!g_sdx_ok) { g_xv_n = 0; g_xv_block = 1; return; }   // 索引不可用：账本清空、本次提交不新增（没法核下一次触碰） | Index unavailable: clear the ledger and add nothing in this submit (the next touch cannot be verified)
    g_xv_block = 0;
    memset(g_sdx_hk, 0, sizeof g_sdx_hk); g_sdx_nb = 0;
    for (int i = 0; i < g_xv_n; i++) sdx_bucket(g_xv[i].ser, 1);   // 账本里的贴图进候选集：否则它在本次提交里只被采样、不作附件时读事件不入索引，兜底看不见 | Textures in the ledger enter the candidate set: otherwise, when one is only sampled and not used as an attachment in this submit, its read events are not indexed and the fallback cannot see them
    if (R.has_rt) { for (int k = 0; k < R.rt.n && k < 8; k++) if (R.rt.color[k]) sdx_bucket(R.rt.color[k]->serial, 1); if (R.rt.depth) sdx_bucket(R.rt.depth->serial, 1); }   // 提交开始时的目标（首个绘制可能不经 SetRenderTargets 重开通道） | The targets at the start of the submit (the first draw may reopen the render pass without going through SetRenderTargets)
    g_pp_prog = S.prog; g_pp_rt = NULL; g_pp_rtseq = SDX_NONE;
    for (int st = 0; st < 6; st++) { for (int k = 0; k < 128; k++) g_pp_slot[st][k] = S.tex[st][k] ? h_ser(S.tex[st][k]) : 0;
                                     for (int k = 0; k < 32; k++) g_pp_slot[st][128 + k] = S.img[st][k] ? h_ser(S.img[st][k]) : 0; }
    g_sdx_ren = 1;
    for (int i = 0; i < n && g_sdx_ok; i++) sdx_walk((const uint8_t *)h[i], 0);
    if (!g_sdx_ok) return;
    // 计数排序到桶：事件本按序号追加，同一序号内 READ 先于 ATT（同一绘制先判读后判附件），稳定散列后桶内有序
    // Counting sort into buckets: events are appended in sequence-number order, and within the same sequence number READ precedes ATT (the same draw is judged for reads first, then attachments), so after a stable scatter each bucket is ordered
    memset(g_sdx_bcnt, 0, (g_sdx_nb + 1) * sizeof *g_sdx_bcnt);
    for (uint32_t i = 0; i < g_sdx_n; i++) g_sdx_bcnt[sdx_bucket(g_sdx[i].ser, 0)]++;
    for (uint32_t b = 0, acc = 0; b <= g_sdx_nb; b++) { g_sdx_bst[b] = acc; acc += b < g_sdx_nb ? g_sdx_bcnt[b] : 0; }
    for (uint32_t b = 0; b < g_sdx_nb; b++) g_sdx_bcnt[b] = g_sdx_bst[b];
    for (uint32_t i = 0; i < g_sdx_n; i++) g_sdx_tmp[g_sdx_bcnt[sdx_bucket(g_sdx[i].ser, 0)]++] = g_sdx[i];
    sdx_ev *t = g_sdx; g_sdx = g_sdx_tmp; g_sdx_tmp = t;
    xv_forward();
}
static void sdx_seq(const uint8_t *p) {   // mtl_exec 每条操作：序号与索引核对（不符即作废，余下一律写回） | Every operation in mtl_exec: check the sequence number against the index (on mismatch invalidate, and store everything for the rest)
    if (!g_sdx_ok) return;
    if (g_exec_seq >= g_sdx_nop || g_sdx_op[g_exec_seq] != p) { RED_ONCE(40, "写回前瞻索引与回放顺序不符（序号 %u / %u），本次提交余下通道一律写回", g_exec_seq, g_sdx_nop); g_sdx_ok = 0; return; }
    g_exec_seq++;
}
static int sdx_decide(uint8_t st[9]) {   // st[i]：2 多余；否则 why << 2（需要的原因，同 sd_need）。返回 0 = 索引不可用 | st[i]: 2 redundant; otherwise why << 2 (the reason it is needed, same as sd_need). Returns 0 = index unavailable
    if (!g_sdx_ok || !g_exec_cur || !g_exec_seq || g_sdx_op[g_exec_seq - 1] != g_exec_cur) return 0;
    uint32_t P = g_exec_seq - 1;
    for (int i = 0; i < g_nsd; i++) { const sd_att *t = &g_sd[i]; st[i] = 5 << 2;
        int b = sdx_bucket(t->ser, 0); if (b < 0) continue;   // 不在候选集：本提交里没有任何相关事件 → 扫到末尾未定，需要 | Not in the candidate set: no related event at all in this submit -> undecided after scanning to the end, needed
        uint32_t lo = g_sdx_bst[b], end = g_sdx_bst[b + 1], hi = end;   // 桶内第一个 seq ≥ P | First seq >= P in the bucket
        while (lo < hi) { uint32_t m = (lo + hi) / 2; if (g_sdx[m].seq < P) lo = m + 1; else hi = m; }
        for (uint32_t j = lo; j < end; j++) { const sdx_ev *e = &g_sdx[j];
            if (g_hoist_n && hoist_skipped(e->op)) continue;
            if (e->kind == SDX_READ || e->kind == SDX_NEED) { st[i] = (uint8_t)(e->why << 2); break; }
            if (e->rtseq == SDX_NONE || e->rtseq < P || (g_hoist_n && hoist_skipped(g_sdx_op[e->rtseq]))) continue;   // 该目标在通道结束之前设的：通道结束之后未重设为附件，不算 | This target was set before the render pass ended: not re-set as an attachment after the render pass ended, so it does not count
            int hit = t->depth ? (e->kind == SDX_ATT ? e->plane == 8 : e->plane == t->depth) && (e->anyls || (e->lv == t->lv && e->sl == t->sl))
                               : e->plane == 0 && e->lv == t->lv && e->sl == t->sl;
            if (!hit) continue;
            st[i] = e->kind == SDX_CLEAR ? 2 : (uint8_t)(2 << 2); break; }
    }
    return 1;
}
static void xv_forward(void) {
    if (!g_xv_n) return;
    if (!g_sdx_ok) { g_xv_n = 0; g_xv_block = 1; return; }
    for (int i = 0; i < g_xv_n;) {
        xv_ent *e = &g_xv[i]; int b = sdx_bucket(e->ser, 0), action = 0;
        if (b >= 0) for (uint32_t j = g_sdx_bst[b]; j < g_sdx_bst[b + 1]; j++) { sdx_ev *v = &g_sdx[j];
            if (v->kind == SDX_READ || v->kind == SDX_NEED) { xv_ent copy = *e; xv_violation(&copy, "提交前瞻读/需求"); action = 2; break; }
            if (v->lv != e->lv || v->sl != e->sl) continue;
            if (v->kind == SDX_CLEAR) { xv_good(e); action = 1; break; }
            if (v->kind == SDX_ATT) break;
        }
        if (!action) i++;
    }
}
static void rt_sub(const p_set_rt *r, int k, uint64_t *ser, int *lv, int *sl) {   // 附件 k（0–7 颜色，8 深度）的子资源 | Subresource of attachment k (0–7 color, 8 depth)
    const tex_t *t = k == 8 ? r->depth : r->color[k]; const view_t *v = k == 8 ? &r->dview : &r->cview[k]; int hv = k == 8 ? r->has_dview : r->has_cview[k];
    *ser = t ? t->serial : 0; *lv = hv && v->set_levels ? v->base_level : 0; *sl = hv && v->set_layers ? v->min_layer : 0;
}
static void hoist_try(void) {
    if (g_hoist_n || !g_exec_next) return;
    typedef struct { const uint8_t *rt_op; const p_set_rt *rt; uint8_t clear_c[8]; MTLClearColor cc[8]; uint8_t clear_d, clear_s; double cd; uint32_t cs;
                     p_scissor sc; int has_sc; const uint8_t *ops[8]; int nops; } grp;
    grp G[4]; int ng = 0, in_grp = 0; p_scissor sc = R.sc; int has_sc = R.has_sc;
    // 「被清的贴图不在任一槽上」要的绑定集合：只在真找到可前提的清除组、回到本通道目标时才查表（绝大多数通道走不到那一步）。
    // 扫描区间里绑上的句柄先记下，查表时与当前槽一并换成贴图序号；扫描只读命令，期间当前槽 S 不变，结果与开头就查相同
    // The binding set needed for "the cleared texture is not on any slot": the table is looked up only when a hoistable clear group is really found and we are back to this render pass's targets (the vast majority of render passes never get that far).
    // Handles bound within the scan range are noted first and converted to texture serials together with the current slots at lookup time; the scan only reads commands and the current slots S do not change meanwhile, so the result is the same as checking at the start
    uint64_t bh[512]; int nbh = 0;
    const uint8_t *p = g_exec_next;
    for (int guard = 0; guard < 200000; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        switch (h.op) {
            case OP_END: case OP_CALL: case OP_COPY_T2T: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH: return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_DRAW: case OP_DRAW_IDX: if (in_grp) return; break;
            case OP_SCISSOR: sc = *(const p_scissor *)a; has_sc = 1; break;
            case OP_RENDER_ENABLE: if (!((const p_flag *)a)->value) return; break;   // TOTK: clears recorded while rendering is disabled do not run
            case OP_RENDER_COND: if (!cond_eval(a, p)) return; break;
            case OP_BIND_TEX: case OP_BIND_IMG: { const p_bind_handles *b = a; for (int k = 0; k < b->n && k < 32 && nbh < 512; k++) if (b->h[k]) bh[nbh++] = b->h[k]; break; }
            case OP_CLEAR_COLOR: case OP_CLEAR_DS: {
                if (!in_grp || G[ng - 1].nops >= 8) return;
                grp *g = &G[ng - 1];
                if (h.op == OP_CLEAR_COLOR) { const p_clear_color *c = a;
                    if (c->index < 0 || c->index >= g->rt->n || !g->rt->color[c->index] || c->mask != 0xf) return;
                    const tex_t *t = g->rt->color[c->index];
                    if (has_sc && !(sc.x <= 0 && sc.y <= 0 && sc.x + sc.w >= (int)t->d.width && sc.y + sc.h >= (int)t->d.height)) return;
                    g->clear_c[c->index] = 1; g->cc[c->index] = MTLClearColorMake(c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]); }
                else { const p_clear_ds *c = a; if (!g->rt->depth) return;
                    if (c->depth_mask) { g->clear_d = 1; g->cd = c->depth; } if (c->stencil_mask) { g->clear_s = 1; g->cs = (uint32_t)c->stencil; } }
                g->sc = sc; g->has_sc = has_sc; g->ops[g->nops++] = p; break; }
            case OP_SET_RT: { const p_set_rt *r = a;
                if (rt_same(&R.rt, r)) {   // 回到本通道的目标：前提成立的话执行 | Back to this render pass's targets: execute if hoisting holds
                    if (!ng) return;
                    uint64_t bound[512]; int nb = 0;   // 原「开头即查」的顺序：当前各槽（阶段 × 贴图 128 + 图像 32），再接扫描区间里绑上的 | The order of the original "check at the start": the current slots (stage x textures 128 + images 32), followed by those bound within the scan range
                    for (int st = 0; st < 6; st++) { for (int k = 0; k < 128 && nb < 512; k++) if (S.tex[st][k]) bound[nb++] = h_ser(S.tex[st][k]);
                                                     for (int k = 0; k < 32 && nb < 512; k++) if (S.img[st][k]) bound[nb++] = h_ser(S.img[st][k]); }
                    for (int k = 0; k < nbh && nb < 512; k++) bound[nb++] = h_ser(bh[k]);
                    for (int gi = 0; gi < ng; gi++) {   // 不相交、未被绑 | Disjoint and not bound
                        grp *g = &G[gi]; if (!g->nops) return;
                        for (int k = 0; k < 9; k++) { if (k < 8 ? !g->clear_c[k] : !(g->clear_d || g->clear_s)) continue;
                            uint64_t ser; int lv, sl; rt_sub(g->rt, k, &ser, &lv, &sl); if (!ser) return;
                            for (int j = 0; j < nb; j++) if (bound[j] == ser) return;
                            for (int m = 0; m < 9; m++) { uint64_t s2; int l2, sl2; if (m < 8 && m >= R.rt.n) continue; rt_sub(&R.rt, m, &s2, &l2, &sl2);
                                if (s2 == ser && l2 == lv && sl2 == sl) return; } } }
                    // 先登记全部跳过项再执行：前提通道结束时写回前瞻就往后扫，原位置的清除必须已在跳过集合里
                    // Register all skip entries first, then execute: when a hoisted render pass ends the store lookahead scans forward, and the clears at their original positions must already be in the skip set
                    for (int gi = 0; gi < ng; gi++) { g_hoist_skip[g_hoist_n++] = G[gi].rt_op; for (int k = 0; k < G[gi].nops; k++) g_hoist_skip[g_hoist_n++] = G[gi].ops[k]; }
                    p_set_rt rt0 = R.rt; p_scissor sc0 = R.sc; int hs0 = R.has_sc;   // 本通道的目标、剪裁与挂起的清除：执行完原样恢复 | This render pass's targets, scissor and pending clears: restored as they were after execution
                    uint8_t cc0f[8]; MTLClearColor cc0[8]; uint8_t cd0f = R.clear_d, cs0f = R.clear_s; double cd0 = R.cd; uint32_t cs0 = R.cs;
                    memcpy(cc0f, R.clear_c, 8); memcpy(cc0, R.cc, sizeof cc0);
                    for (int gi = 0; gi < ng; gi++) {   // 执行：临时换上该组目标与清除，开只为清除的通道 | Execute: temporarily swap in that group's targets and clears and open a clear-only render pass
                        grp *g = &G[gi];
                        R.rt = *g->rt; memcpy(R.clear_c, g->clear_c, 8); memcpy(R.cc, g->cc, sizeof R.cc); R.clear_d = g->clear_d; R.clear_s = g->clear_s; R.cd = g->cd; R.cs = g->cs;
                        R.sc = g->sc; R.has_sc = g->has_sc;
                        PD("H %llu 前提纯清除通道（原位置在本通道之后）", g_stat_pass + 1);
                        rp_begin_ex(1); rp_end();
                    }
                    R.rt = rt0; R.sc = sc0; R.has_sc = hs0; memcpy(R.clear_c, cc0f, 8); memcpy(R.cc, cc0, sizeof cc0); R.clear_d = cd0f; R.clear_s = cs0f; R.cd = cd0; R.cs = cs0;
                    if (!g_hoist_groups) MLOG("纯清除通道前提：首例 %d 组（通道 %llu 前）", ng, g_stat_pass + 1);
                    g_hoist_groups += (uint64_t)ng;
                    return;
                }
                if (in_grp && !G[ng - 1].nops) return;   // 上一组没有清除（只设目标）：不认 | The previous group has no clear (only set targets): not recognized
                if (ng >= 4) return;
                memset(&G[ng], 0, sizeof G[ng]); G[ng].rt_op = p; G[ng].rt = r; ng++; in_grp = 1; break; }
            default: break;
        }
        p += h.size;
    }
}
// ---------- 可读 tile 通道合并（合并 B 第 2 步） ----------
// 后一个通道的绘制在本像素采样前一个通道写的颜色附件时，两个通道并成一个渲染通道：前一个通道写的内容留在 tile 里，
// 后一个通道用可读 tile 版本（<键>_bf，函数常量特化）直接读，不为被采样而先落主存再读回。
// 规划（fuse_try，前一个通道开头）只做结构检查；本像素寻址的闭式证明在后一个通道每次绘制时做（fz_p2_draw），证不出就结束合并通道、
// 按后一个通道的原目标重开普通通道（前一个通道的结果照常写回，等价于不合并）。
// ---------- Readable tile render pass merge (merge B step 2) ----------
// When a draw of the later pass samples, at the same pixel, a color attachment written by the earlier pass, the two passes are merged into one render pass: what the earlier pass wrote stays in the tile,
// and the later pass reads it directly with the readable tile version (<key>_bf, function-constant specialization), instead of first landing in main memory and being read back just to be sampled.
// Planning (fuse_try, at the start of the earlier pass) does only structural checks; the closed-form proof of same-pixel addressing is done at every draw of the later pass (fz_p2_draw), and if it cannot be proven the merged render pass is ended and
// a normal render pass is reopened with the later pass's original targets (the earlier pass's result is stored as usual, equivalent to not merging).
static void fz_unskip(void) { for (int i = 0; i < g_fz.nskip; i++) for (int j = 0; j < g_hoist_n; j++) if (g_hoist_skip[j] == g_fz.skip[i]) { g_hoist_skip[j] = g_hoist_skip[--g_hoist_n]; break; } }
static int fz_reuse_op(const uint8_t *p) { for (int i = 0; i < g_fz.nreuse; i++) if (g_fz.reuse[i] == p) return 1; return 0; }
// 诊断 DM_NVN_FUSE_CHECK=1：局内双渲染。后一个通道证得出的绘制逐次留状态快照；合并通道结束后，把后一个通道写过的附件拷一份（合并路径的结果），
// 再关掉全部前瞻、按后一个通道的原目标与原清除用普通路径逐次重画，两份逐字节比（同一帧同一输入，不依赖跨局确定性）。重画结果留在附件上（相同即无影响）
// Diagnostic DM_NVN_FUSE_CHECK=1: in-run double rendering. For the later pass's draws that can be proven, keep a state snapshot per draw; after the merged render pass ends, make a copy of the attachments the later pass wrote (the merge path's result),
// then turn off all lookahead and redraw one by one through the normal path with the later pass's original targets and original clears, and compare the two byte by byte (same frame, same input, no dependence on cross-run determinism). The redraw result is left on the attachment (no effect if identical)
static int fz_check_on(void) { static int v = -1; if (v < 0) v = getenv("DM_NVN_FUSE_CHECK") != NULL; return v; }
typedef struct { int idx; p_draw dw; p_draw_idx di; p_scissor sc; int has_sc; void *S_; } fz_snap;
static fz_snap g_fzs[8]; static int g_nfzs, g_fzs_bad, g_fz_checking; static uint64_t g_fzc_same, g_fzc_diff;
static void draw(const p_draw *dw, const p_draw_idx *di);
static void fz_snap_take(const p_draw *dw, const p_draw_idx *di) {
    if (g_nfzs >= 8) { g_fzs_bad = 1; return; }
    fz_snap *z = &g_fzs[g_nfzs++]; z->idx = di != NULL; if (dw) z->dw = *dw; if (di) z->di = *di; z->sc = R.sc; z->has_sc = R.has_sc;
    z->S_ = malloc(sizeof S); memcpy(z->S_, &S, sizeof S);
}
static void fz_check_run(const fuse_t *fz) {
    if (!g_nfzs || g_fzs_bad) { if (g_fzs_bad) MLOG("【红】局内双渲染：后一个通道绘制超过 8 次，未比"); goto out; }
    {
    id<MTLTexture> att[8], scr[8]; uint64_t sers[8]; int na = 0;
    // fz_end 已把 R.rt 从物理并集还原为后通道逻辑目标；wrote 仍按物理槽登记，必须经 p2map 对位。
    // fz_end has already restored R.rt from the physical union to the later pass's logical targets; wrote is still registered by physical slot and must be aligned through p2map.
    for (int j = 0; j < fz->rt2.n && j < 8; j++) if (fz->p2map[j] >= 0 && fz->wrote[fz->p2map[j]] && fz->rt2.color[j]) {
        id<MTLTexture> t = tex_mtl(fz->rt2.color[j]); if (!t) continue;
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat width:t.width height:t.height mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
        id<MTLTexture> c = [g_dev newTextureWithDescriptor:d]; if (!c) continue;
        enc_b_copy_texture(blit(), t, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(t.width, t.height, 1), c, 0, 0, MTLOriginMake(0, 0, 0));
        att[na] = t; scr[na] = c; sers[na] = fz->rt2.color[j]->serial; na++; }
    if (!na) { MLOG("【红】局内双渲染：已留 %d 次绘制快照，但没有可比附件", g_nfzs); goto out; }
    static typeof(S) Ssave; memcpy(&Ssave, &S, sizeof S); typeof(R) Rsave = R; const uint8_t *en0 = g_exec_next;
    g_fz_checking = 1; g_exec_next = NULL;   // 重画时关掉全部前瞻（续用 / 前提 / 合并 / 深度模板按需）：普通路径 | Turn off all lookahead during the redraw (continuation / hoisting / merging / depth-stencil on demand): normal path
    R.rt = fz->rt2; memcpy(R.clear_c, fz->clr2, 8); memcpy(R.cc, fz->cc2, sizeof R.cc); R.clear_d = fz->cd2f; R.clear_s = fz->cs2f; R.cd = fz->cd2; R.cs = fz->cs2;
    for (int i = 0; i < g_nfzs; i++) { fz_snap *z = &g_fzs[i]; memcpy(&S, z->S_, sizeof S); args_dirty_all(); R.sc = z->sc; R.has_sc = z->has_sc; g_pso_dirty = g_dss_dirty = 1;
        draw(z->idx ? NULL : &z->dw, z->idx ? &z->di : NULL); }
    rp_end();
    memcpy(&S, &Ssave, sizeof S); args_dirty_all(); R = Rsave; g_exec_next = en0; g_fz_checking = 0; g_pso_dirty = g_dss_dirty = 1;
    for (int a = 0; a < na; a++) {
        id<MTLTexture> t = att[a], c = scr[a]; NSUInteger W = t.width, H = t.height, bpp = (NSUInteger)(att_bytes(t, 0) / (W * H));
        id<MTLBuffer> b1 = [g_dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared], b2 = [g_dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared];
        id<MTL4ComputeCommandEncoder> b = blit();
        enc_b_copy_texture_buffer(b, c, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), b1, 0, W * bpp, W * H * bpp);
        enc_b_copy_texture_buffer(b, t, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), b2, 0, W * bpp, W * H * bpp);
        unsigned long long ser = sers[a], pass = g_stat_pass, frame = atomic_load(&g_cur_frame); NSUInteger nb = W * H * bpp;
        [g_commit_blocks addObject:^{
            const uint8_t *x = b1.contents, *y = b2.contents; uint64_t dpx = 0, maxd = 0;
            for (NSUInteger k = 0; k < nb; k += bpp) if (memcmp(x + k, y + k, bpp)) { dpx++; for (NSUInteger q = 0; q < bpp; q++) { uint64_t dd = (uint64_t)abs((int)x[k + q] - (int)y[k + q]); if (dd > maxd) maxd = dd; } }
            if (dpx) { g_fzc_diff++; MLOG("【红】局内双渲染：帧 %llu 通道 %llu 附件 ser=%llu 合并 vs 普通 不同像素 %llu / %lu（逐字节最大差 %llu）", frame, pass, ser, (unsigned long long)dpx, (unsigned long)(nb / bpp), (unsigned long long)maxd); }
            else { g_fzc_same++; MLOG("局内双渲染：帧 %llu 通道 %llu 附件 ser=%llu 合并 vs 普通 逐字节相同（%lu 像素）", frame, pass, ser, (unsigned long)(nb / bpp)); }
            if ((g_fzc_same + g_fzc_diff) % 300 == 0) MLOG("局内双渲染累计：逐字节相同 %llu 次、不同 %llu 次", (unsigned long long)g_fzc_same, (unsigned long long)g_fzc_diff);
        }];
    }
    }
out:
    for (int i = 0; i < g_nfzs; i++) free(g_fzs[i].S_);
    g_nfzs = 0; g_fzs_bad = 0;
}
static void fz_end(void) {   // 合并通道结束（rp_end 调）：还没进后一个通道就被打断 → 撤回跳过项、目标还原成前一个通道的；否则还原成后一个通道的 | End of a merged render pass (called by rp_end): interrupted before entering the later pass -> retract the skip entries and restore the targets to the earlier pass's; otherwise restore to the later pass's
    if (!g_fz.active) return;
    if (g_fz.phase == 1) { fz_unskip(); R.rt = g_fz.rt1; } else R.rt = g_fz.rt2;
    g_fz.active = 0;
    if (g_fz.phase == 2 && fz_check_on() && !g_fz_checking) { fuse_t fz = g_fz; R.rt = fz.rt2; fz_check_run(&fz); }
    else { for (int i = 0; i < g_nfzs; i++) free(g_fzs[i].S_); g_nfzs = 0; g_fzs_bad = 0; }
}
static int sub_eq(const p_set_rt *a, int ka, const p_set_rt *b, int kb) { uint64_t s1, s2; int l1, l2, x1, x2; rt_sub(a, ka, &s1, &l1, &x1); rt_sub(b, kb, &s2, &l2, &x2); return s1 && s1 == s2 && l1 == l2 && x1 == x2; }
static int prog_samples_ser(const prog_t *pg, uint64_t tex[6][128], uint64_t img[6][32], uint64_t ser) {   // TOTK: a program with an untranslated stage counts as reading everything   // 程序反射声明的贴图 / 图像 / 类型化缓冲槽里有没有绑着该贴图 | Whether this texture is bound in any texture / image / typed buffer slot declared by the program's reflection
    __unsafe_unretained NvProg *pr = pg ? prog_get(pg) : nil; if (!pr || pr->missing) return 1;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? img[sh->stage][b->reg] : 0; else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? tex[sh->stage][b->reg] : 0; else continue;
            if (h && h_ser(h) == ser) return 1; } }
    return 0;
}
static int g_defer_env_ab, g_defer_env_off;   // 环境开关：每次提交开始（defer_reset）读一次，热路径不调 getenv | Environment switches: read once at the start of each submit (defer_reset); the hot path does not call getenv
static void defer_env(void) { const char *x = getenv("DM_NVN_DEFER_AB"), *e = getenv("DM_NVN_DEFER"); g_defer_env_ab = x && !strcmp(x, "1"); g_defer_env_off = e && !strcmp(e, "0"); }
static int defer_on(void) {
    if (!g_defer_env_ab) return tbdr_on() && !g_defer_env_off;
    if (g_defer_ab < 0) { g_defer_ab = 0; g_defer_ab_on = 0; }
    return tbdr_on() && (g_defer_ab || g_defer_ab_on); }
static int defer_handle_sub(uint64_t h, const p_set_rt *r, int k, int cover) {
    int tid = (int)(h & 0xfffff); if (!tid || tid >= g_ntreg || !g_treg[tid].ok) return 0;
    uint64_t ser; int lv, sl; rt_sub(r, k, &ser, &lv, &sl); if (!ser || g_treg[tid].t.serial != ser) return 0;
    if (!g_treg[tid].has_v || !g_treg[tid].v.set_levels) return cover;
    view_t v = g_treg[tid].v; int n = v.nlevels > 0 ? v.nlevels : 1;
    return lv >= v.base_level && lv < v.base_level + n && (!v.set_layers || (sl >= v.min_layer && sl < v.min_layer + (v.nlayers > 0 ? v.nlayers : 1)));
}
static int defer_prog_hits(const prog_t *pg, uint64_t tex[6][128], uint64_t img[6][32], const p_set_rt *r, int color_only, int cover) {
    __unsafe_unretained NvProg *pr = pg ? prog_get(pg) : nil; if (!pr || pr->missing) return 1;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage < 0 || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h = 0;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? img[sh->stage][b->reg] : 0;
            else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? tex[sh->stage][b->reg] : 0;
            else continue;
            if (!h) continue;
            for (int k = 0; k < (color_only ? r->n : 9); k++) if ((color_only ? r->color[k] : (k < 8 ? r->color[k] : r->depth)) && defer_handle_sub(h, r, k, cover)) return 1;
        }
    }
    return 0;
}
static void defer_collect_reads(defer_t *d, const prog_t *pg, uint64_t tex[6][128], uint64_t img[6][32]) {
    __unsafe_unretained NvProg *pr = pg ? prog_get(pg) : nil; if (!pr || pr->missing) { d->nreads = 513; return; }
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage < 0 || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h = 0;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? img[sh->stage][b->reg] : 0;
            else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? tex[sh->stage][b->reg] : 0;
            else continue;
            if (!h) continue; int seen = 0; for (int j = 0; j < d->nreads; j++) seen |= d->p_reads[j] == h;
            if (!seen && d->nreads < 512) d->p_reads[d->nreads++] = h;
        }
    }
}
static int defer_rt_hits_reads(const p_set_rt *r, int idx) {
    if (idx < 0 || idx >= r->n || !r->color[idx]) return 0;
    for (int i = 0; i < g_defer.nreads; i++) if (defer_handle_sub(g_defer.p_reads[i], r, idx, 1)) return 1;
    return 0;
}
static int defer_q_superset(const p_set_rt *q) {
    memset(g_defer.lmap, 0xff, sizeof g_defer.lmap);
    for (int i = 0; i < g_defer.p_rt.n; i++) if (g_defer.p_rt.color[i]) { int found = 0;
        // 下标须相同：普通片元函数的 color(n) 输出写到物理第 n 个附件，lmap 只改写掩码 / 混合、搬不动输出（能搬的只有带输出映射常量的 _bf 版本）。
        // 下标不同时推迟会把 P 的输出写进 Q 的别的附件。
        // The indices must be the same: a normal fragment function's color(n) output is written to physical attachment n, and lmap only changes the write mask / blend and cannot move the output (only the _bf version with output mapping constants can).
        // With different indices, deferral would write P's output into a different attachment of Q.
        if (i < q->n && q->color[i] && sub_eq(&g_defer.p_rt, i, q, i)) { g_defer.lmap[i] = (uint8_t)i; found = 1; }
        if (!found) return 0;
    }
    return 1;
}
static int defer_take(const p_draw *dw, const p_draw_idx *di) {
    if (g_defer.n >= 64) { g_defer.reason = 20; return 0; }
    defer_snap *z = &g_defer.snap[g_defer.n++]; z->idx = di != NULL; if (dw) z->dw = *dw; if (di) z->di = *di;
    z->sc = R.sc; z->has_sc = R.has_sc; z->S_ = malloc(sizeof S); memcpy(z->S_, &S, sizeof S); for (int k = 0; k < 8; k++) z->pm[k] = (uint8_t)k; g_defer_draws++; return 1;
}
static void defer_release(void) { defer_release_one(&g_defer); }
static void defer_env(void);
static void defer_reset(void) { defer_clear_all(); g_defer_replaying = 0; defer_env(); }
static void defer_apply_p_clear(void) {
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    for (int j = 0; j < g_defer.q_rt.n; j++) { int i = g_defer.lmap[j]; if (i >= 0 && g_defer.clear_c[i]) { R.clear_c[j] = 1; R.cc[j] = g_defer.cc[i]; } }
    if (g_defer.clear_d) { R.clear_d = 1; R.cd = g_defer.cd; } if (g_defer.clear_s) { R.clear_s = 1; R.cs = g_defer.cs; }
}
static int defer_begin_q(const uint8_t *op) {
    int found = 0, first = -1;
    for (int i = 0; i < 4; i++) if (g_defer_slots[i].active && !g_defer_slots[i].at_q && g_defer_slots[i].q_op == op) { found = 1; if (first < 0) first = i; }
    if (!found) return 0;
    rp_flush();
    g_defer_cur = &g_defer_slots[first]; R.rt = g_defer.q_rt; R.has_rt = 1;
    for (int i = 0; i < 4 && !R.rt.depth; i++) { const defer_t *d = &g_defer_slots[i]; if (d->active && !d->at_q && d->q_op == op && d->q_rt.depth) { R.rt.depth = d->q_rt.depth; R.rt.dview = d->q_rt.dview; R.rt.has_dview = d->q_rt.has_dview; } }   // 任一组需要的深度模板都挂上（同 Q 各组深度已核相容） | Attach the depth-stencil needed by any group (the depths of groups with the same Q have been checked compatible)
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    for (int i = 0; i < 4; i++) { defer_t *d = &g_defer_slots[i]; if (!d->active || d->at_q || d->q_op != op) continue;
        for (int j = 0; j < d->q_rt.n; j++) { int k = d->lmap[j]; if (k >= 0 && d->clear_c[k]) { R.clear_c[j] = 1; R.cc[j] = d->cc[k]; } }
        if (d->clear_d) { R.clear_d = 1; R.cd = d->cd; } if (d->clear_s) { R.clear_s = 1; R.cs = d->cs; }
        d->at_q = 1; g_defer_natq++;
    }
    return 1;
}
static void defer_replay(void) {
    if (g_defer_replaying) return;
    int any = 0; for (int i = 0; i < 4; i++) if (g_defer_slots[i].active && g_defer_slots[i].at_q) any = 1;
    if (!any) return;
    typeof(S) save; memcpy(&save, &S, sizeof S); p_scissor sc = R.sc; int has_sc = R.has_sc; int vis = g_vis_counting;
    const uint8_t *en0 = g_exec_next; g_defer_replaying = 1; g_exec_next = NULL;
    for (int gi, after = 0; (gi = defer_next(1, after)) >= 0;) { defer_t *d = &g_defer_slots[gi]; after = d->born;   // 按创建先后：各组 P 区间不交叠且先后排列，组序即原绘制顺序 | In creation order: the groups' P ranges do not overlap and are arranged in sequence, so the group order is the original draw order
        g_defer_cur = d; if (d->reset_between) g_vis_counting = 0;
        { static int once; if (!once++) MLOG("推迟合并通道：首例 通道 %llu，组 %d 回放绘制 %d 次", g_stat_pass, gi, d->n); }   // 多组时首个回放前可能已建多组，不能按 g_defer_n == 1 判 | With multiple groups, several groups may already have been created before the first replay, so it cannot be judged by g_defer_n == 1
        for (int i = 0; i < d->n; i++) { defer_snap *z = &d->snap[i]; static uint8_t lm[8]; for (int j = 0; j < 8; j++) lm[j] = d->lmap[j] == 0xff || d->lmap[j] >= 8 ? 0xff : z->pm[d->lmap[j]];
            memcpy(&S, z->S_, sizeof S); args_dirty_all(); R.sc = z->sc; R.has_sc = z->has_sc; g_pso_dirty = g_dss_dirty = 1; g_pso_lmap = lm; draw(z->idx ? NULL : &z->dw, z->idx ? &z->di : NULL); }
        defer_release_one(d);
    }
    g_vis_counting = vis; memcpy(&S, &save, sizeof S); args_dirty_all(); R.sc = sc; R.has_sc = has_sc; g_pso_lmap = NULL; g_pso_fsfn = nil; memset(&g_pso_fz, 0, sizeof g_pso_fz); g_pso_dirty = g_dss_dirty = 1; g_exec_next = en0; g_defer_replaying = 0; g_defer_cur = &g_defer_slots[0];
}
static void defer_fallback(void) {
    if (g_defer_replaying || !defer_has_active()) return;
    g_defer_fallbacks++; MLOG("【红】推迟合并通道：提交结束仍有未回放组，按各组 P 原目标兜底");
    if (g_enc) { enc_r_end(g_enc); g_enc = nil; }
    typeof(S) save; memcpy(&save, &S, sizeof S); p_set_rt rt = R.rt; g_defer_replaying = 1;
    for (int gi, after = 0; (gi = defer_next(0, after)) >= 0;) { defer_t *d = &g_defer_slots[gi]; after = d->born; g_defer_cur = d; R.rt = d->p_rt; R.has_rt = 1; memcpy(R.clear_c, d->clear_c, 8); memcpy(R.cc, d->cc, sizeof R.cc);
        R.clear_d = d->clear_d; R.clear_s = d->clear_s; R.cd = d->cd; R.cs = d->cs;
        for (int i = 0; i < d->n; i++) { defer_snap *z = &d->snap[i]; memcpy(&S, z->S_, sizeof S); args_dirty_all(); R.sc = z->sc; R.has_sc = z->has_sc; g_pso_dirty = g_dss_dirty = 1; g_pso_lmap = z->pm; draw(z->idx ? NULL : &z->dw, z->idx ? &z->di : NULL); }
        rp_end(); defer_release_one(d);
    }
    memcpy(&S, &save, sizeof S); args_dirty_all(); R.rt = rt; g_defer_replaying = 0; g_defer_cur = &g_defer_slots[0];
}
static const uint8_t *defer_cont(int *ci) {   // 列表结束后的下一段（同一次提交内）：先调用方返回栈，再本次提交余下的句柄；没有了返回 NULL | The next segment after a list ends (within the same submit): first the caller return stack, then the remaining handles of this submit; returns NULL when there are none left
    for (;;) { int k = (*ci)++;
        if (k < g_ret_n) { if (g_ret_n > 16) return NULL; return g_ret[g_ret_n - 1 - k]; }
        k -= g_ret_n; if (k < g_nrest) return (const uint8_t *)g_rest[k];
        return NULL; }
}
// 推迟预筛（写回前瞻索引二分，零扫描）：P 每个颜色子资源在本次绘制之后、首个「作附件 / 被清」事件必须是「另一次 SetRenderTargets 下作附件、
// 且那次目标覆盖 P 全部颜色」；先被清 / 本次提交再没作附件 → 不可能推迟，直接返回。读事件不带级别，交给完整前瞻按级别核。
// Deferral pre-screen (binary search on the store lookahead index, zero scanning): for every color subresource of P, the first "used as attachment / cleared" event after this draw must be "used as attachment under a different SetRenderTargets,
// whose targets cover all of P's colors"; cleared first / never used as attachment again in this submit -> deferral is impossible, return directly. Read events carry no level; handed to the full lookahead to check by level.
static int defer_pre(void) {
    if (!g_sdx_ok || !g_exec_cur || !g_exec_seq || g_sdx_op[g_exec_seq - 1] != g_exec_cur) return 0;
    uint32_t P = g_exec_seq - 1;
    for (int k = 0; k < R.rt.n; k++) { if (!R.rt.color[k]) continue;
        uint64_t ser; int lv, sl; rt_sub(&R.rt, k, &ser, &lv, &sl);
        int b = sdx_bucket(ser, 0); if (b < 0) return 0;
        uint32_t lo = g_sdx_bst[b], end = g_sdx_bst[b + 1], hi = end;
        while (lo < hi) { uint32_t m = (lo + hi) / 2; if (g_sdx[m].seq <= P) lo = m + 1; else hi = m; }
        int ok = 0, same = 0; uint32_t walk = 0, prt = lo > g_sdx_bst[b] && g_sdx[lo - 1].seq == P ? g_sdx[lo - 1].rtseq : SDX_NONE;   // P 本通道的 SetRenderTargets 序号（取本次绘制的作附件事件） | SetRenderTargets sequence number of P's own render pass (taken from this draw's used-as-attachment event)
        for (uint32_t j = lo; j < end; j++) { const sdx_ev *e = &g_sdx[j];
            if (++walk > 64) return 0;   // 事件太多（被大量采样的贴图）：不当候选，保守且免长走 | Too many events (a heavily sampled texture): not taken as a candidate, conservative and avoids a long walk
            if (g_hoist_n && hoist_skipped(e->op)) continue;
            if (e->kind == SDX_READ || e->kind == SDX_NEED) continue;   // 读事件不带级别（读别的级不冲突），交给完整前瞻按级别核 | Read events carry no level (reading another level is not a conflict); handed to the full lookahead to check by level
            if (e->plane != 0 || e->lv != lv || e->sl != sl) continue;
            if (e->kind == SDX_CLEAR) return 0;
            if (e->rtseq == SDX_NONE || e->rtseq >= g_sdx_nop) return 0;
            const uint8_t *rop = g_sdx_op[e->rtseq]; const p_set_rt *r = (const p_set_rt *)(rop + 8);
            if (rt_same(r, &R.rt)) { if (e->rtseq != prt) same = 1; continue; }   // 同目标：紧接的是 P 续段、隔了别的通道的是 Q；预筛不分，走完仍无别的候选时交给完整前瞻 | Same target: one immediately following is a continuation segment of P, one separated by other render passes is Q; the pre-screen does not distinguish them, and if there is still no other candidate at the end it is handed to the full lookahead
            for (int i = 0; i < R.rt.n; i++) { if (!R.rt.color[i]) continue; int f = 0;
                for (int q = 0; q < r->n && !f; q++) if (r->color[q] && sub_eq(&R.rt, i, r, q)) f = 1;
                if (!f) return 0; }
            ok = 1; break; }
        if (!ok && !same) return 0; }
    return 1;
}
static void defer_try(const p_draw *dw, const p_draw_idx *di) {
    if (!defer_on() || defer_has_in_p() || g_defer_replaying || g_fz.active || g_fz_checking || !g_exec_next || !R.rt.n) return;
    int slot = defer_slot_free(); if (slot < 0) return;
    g_defer_cur = &g_defer_slots[slot];
    int ncol = 0; for (int i = 0; i < R.rt.n; i++) ncol += R.rt.color[i] != NULL; if (!ncol) return;   // 只挂深度模板的通道（阴影图等）：Q 的定义是覆盖 P 的颜色，无颜色的 P 不可能推迟——原先预筛对它空转放行、完整前瞻白扫几十次绘制 | Render passes with only depth-stencil attached (shadow maps etc.): Q is defined as covering P's colors, so a P without color can never be deferred -- previously the pre-screen let it through idly and the full lookahead scanned dozens of draws for nothing
    int has_clear = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) has_clear |= R.clear_c[i]; if (!has_clear) return;
    if (!defer_pre()) return;
    memset(&g_defer, 0, sizeof g_defer); g_defer.p_rt = R.rt; memcpy(g_defer.clear_c, R.clear_c, 8); memcpy(g_defer.cc, R.cc, sizeof R.cc);
    g_defer.clear_d = R.clear_d; g_defer.clear_s = R.clear_s; g_defer.cd = R.cd; g_defer.cs = R.cs;
    uint64_t tex[6][128], img[6][32]; memcpy(tex, S.tex, sizeof tex); memcpy(img, S.img, sizeof img); const prog_t *pg = S.prog; ds_st ds = S.ds;
    const uint8_t *p = g_exec_next, *p_end = NULL, *qop = NULL; const p_set_rt *q = NULL; p_set_rt scan_rt = R.rt; int srt_hit = -1, ci = 0, draws = 1, p_has_draw = 1; defer_collect_reads(&g_defer, pg, tex, img);   // 当前这次绘制就是 P 的第一次 | The current draw is P's first
    for (int guard = 0; guard < 200000; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        if (g_hoist_n && hoist_skipped(p)) { p += h.size; continue; }
        switch (h.op) {
            case OP_END: { const uint8_t *nx = defer_cont(&ci); if (!nx) { g_defer.reason = 1; goto fail; } p = nx; continue; }
            case OP_CALL: g_defer.reason = 2; goto fail;
            case OP_JUMP: memcpy(&p, a, 8); continue;
            case OP_BIND_PROGRAM: pg = ((const p_bind_program *)a)->prog; break;
            case OP_BIND_TEX: case OP_BIND_IMG: { const p_bind_handles *b = a; int lim = h.op == OP_BIND_IMG ? 32 : 128;
                if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (h.op == OP_BIND_IMG) img[b->stage][b->first + k] = b->h[k]; else tex[b->stage][b->first + k] = b->h[k]; } break; }
            case OP_BIND_STATE: { const p_bind_state *b = a; if (b->kind == 5) memcpy(&ds, b->bytes, sizeof ds); break; }
            case OP_DRAW: case OP_DRAW_IDX:
                if (defer_prog_hits(pg, tex, img, &g_defer.p_rt, 1, 1)) { /* 建议在这里识别需要输出映射但缺少 _bf 后调用 bf_want_note。 | Suggestion: here, after identifying that an output mapping is needed but _bf is missing, call bf_want_note. */ g_defer.reason = 7; goto fail; }
                if (p_end) {   // 中间区间的绘制：不得写 P 读的贴图、不得写 P 的深度模板（P 写模板时也不得测模板） | Draws in the intermediate range: must not write textures P reads, must not write P's depth-stencil (and must not test stencil when P writes stencil)
                    if (srt_hit < 0) { srt_hit = 0; for (int k = 0; k < scan_rt.n && !srt_hit; k++) if (scan_rt.color[k] && defer_rt_hits_reads(&scan_rt, k)) srt_hit = 1; }   // 附件只在 SetRenderTargets 变：每段算一次 | Attachments change only at SetRenderTargets: computed once per segment
                    if (srt_hit) { g_defer.reason = 12; goto fail; }
                    if (scan_rt.depth && g_defer.p_rt.depth && scan_rt.depth->serial == g_defer.p_rt.depth->serial && (ds.dw || (g_defer.p_st && ds.st))) { g_defer.reason = 13; goto fail; }
                    break; }
                draws++; if (draws > 64) { g_defer.reason = 20; goto fail; } p_has_draw = 1; defer_collect_reads(&g_defer, pg, tex, img); g_defer.p_dw |= ds.dw; g_defer.p_st |= ds.st && ((S.st_wmask[0] | S.st_wmask[1]) != 0) && (ds.fail_f != NVN_STENCIL_OP_KEEP || ds.zfail_f != NVN_STENCIL_OP_KEEP || ds.zpass_f != NVN_STENCIL_OP_KEEP || ds.fail_b != NVN_STENCIL_OP_KEEP || ds.zfail_b != NVN_STENCIL_OP_KEEP || ds.zpass_b != NVN_STENCIL_OP_KEEP); if (g_defer.p_dw) { g_defer.reason = 8; goto fail; } break;
            case OP_RESET_COUNTER: g_defer.reset_between = 1; break;
            case OP_COPY_T2T: { const p_copy_t2t *c = a;   // 贴图拷贝：两端都不碰 P 的附件、目标不是 P 采样的贴图才放行 | Texture copy: allowed only if neither end touches P's attachments and the destination is not a texture P samples
                uint64_t ss = c->src ? c->src->serial : 0, ds_ = c->dst ? c->dst->serial : 0; int bad = !ss || !ds_;
                for (int k = 0; k < g_defer.p_rt.n && !bad; k++) if (g_defer.p_rt.color[k] && (g_defer.p_rt.color[k]->serial == ss || g_defer.p_rt.color[k]->serial == ds_)) bad = 1;
                if (g_defer.p_rt.depth && (g_defer.p_rt.depth->serial == ss || g_defer.p_rt.depth->serial == ds_)) bad = 1;
                for (int k = 0; k < g_defer.nreads && !bad; k++) if (h_ser(g_defer.p_reads[k]) == ds_) bad = 1;
                if (g_defer.nreads > 512) bad = 1;
                if (bad) { g_defer.reason = 3; g_defer.reason_op = h.op; goto fail; } break; }
            case OP_REPORT_COUNTER: case OP_UPDATE_UB: case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH: g_defer.reason = 3; g_defer.reason_op = h.op; goto fail;
            case OP_CLEAR_COLOR: { const p_clear_color *c = a; if (!p_end) { g_defer.reason = 16; goto fail; }   // P 续段里的清除：不推迟 | A clear inside P's continuation segment: do not defer
                if (c->index >= 0 && c->index < scan_rt.n && (defer_rt_hits_reads(&scan_rt, c->index) || (scan_rt.color[c->index] && defer_rt_hits_reads(&g_defer.p_rt, c->index)))) { g_defer.reason = 5; goto fail; } break; }
            case OP_CLEAR_DS: if (!p_end) { g_defer.reason = 16; goto fail; } break;
            case OP_SET_RT: { const p_set_rt *r = a;
                if (!p_end && rt_same(r, &g_defer.p_rt)) break;   // 同目标再设一次（游戏常连发）：仍是 P 的续段，绘制一并推迟 | The same target set once more (the game often sends it repeatedly): still a continuation segment of P, and its draws are deferred together
                if (!p_end) p_end = p; int shared = 0; for (int i = 0; i < g_defer.p_rt.n; i++) for (int j = 0; j < r->n; j++) if (g_defer.p_rt.color[i] && r->color[j] && sub_eq(&g_defer.p_rt, i, r, j)) shared = 1;
                if (shared) {   // 空通道（设完目标不画不清就又换目标）不碰任何附件：既不当 Q 也不算冲突 | An empty render pass (targets set, then changed again without drawing or clearing) touches no attachment: it is neither taken as Q nor counted as a conflict
                    const uint8_t *e = p + h.size; int ec = ci, empty = 0;
                    for (int g2 = 0; g2 < 200000; g2++) { op_hdr he; memcpy(&he, e, sizeof he);
                        if (he.op == OP_JUMP) { memcpy(&e, e + 8, 8); continue; }
                        if (he.op == OP_END) { const uint8_t *nx = defer_cont(&ec); if (!nx) break; e = nx; continue; }
                        if (he.op == OP_SET_RT) { empty = 1; break; }
                        if (he.op == OP_DRAW || he.op == OP_DRAW_IDX || he.op == OP_CLEAR_COLOR || he.op == OP_CLEAR_DS || he.op == OP_CALL || he.op == OP_DISPATCH || he.op == OP_CLEAR_TEX || he.op == OP_COPY_T2T || he.op == OP_COPY_B2T || he.op == OP_COPY_T2B) break;
                        e += he.size; }
                    if (empty) break; }
                if (!shared) break; if (!p_has_draw || !defer_q_superset(r)) { g_defer.reason = 6; goto fail; } q = r; qop = p; goto q_found; }
            default: break;
        }
        if (h.op == OP_SET_RT) { scan_rt = *(const p_set_rt *)a; srt_hit = -1; }
        p += h.size;
    }
fail:
    if (g_defer.reason && passdump()) PD("G %llu 不成立 reason=%d op=%d", g_stat_pass + 1, g_defer.reason, g_defer.reason_op); memset(&g_defer, 0, sizeof g_defer); return;
q_found:
    p = qop; op_hdr qh; memcpy(&qh, p, sizeof qh); p += qh.size; int q_body = 0, q_clears = 0;   // 绑定跟踪接着主扫描的状态（不回到 P 开头的 S） | Binding tracking continues from the main scan's state (does not go back to S at the start of P)
    for (int guard = 0; guard < 200000; guard++) { op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8; if (h.op == OP_JUMP) { memcpy(&p, a, 8); continue; }
        // Q 整段（到下一个 SetRenderTargets / 列表结束）：任何绘制都不得按覆盖 P 级别的视图采样 P 的颜色附件——推迟后 P 的结果还在 tile 里没落主存，采样会读到旧内容
        // The whole Q segment (up to the next SetRenderTargets / end of list): no draw may sample P's color attachments through a view covering P's level -- after deferral P's result is still in the tile and has not landed in main memory, so sampling would read old content
        if (h.op == OP_DRAW || h.op == OP_DRAW_IDX) { if (defer_prog_hits(pg, tex, img, &g_defer.p_rt, 1, 1)) { g_defer.reason = 14; goto fail; } q_body = 1; p += h.size; continue; }
        if (q_body && (h.op == OP_SET_RT || h.op == OP_END)) break;
        if (!q_body && h.op == OP_END) { const uint8_t *nx = defer_cont(&ci); if (!nx) { g_defer.reason = 9; goto fail; } p = nx; continue; }
        if (h.op == OP_CALL) { g_defer.reason = 9; g_defer.reason_op = h.op; goto fail; }
        // Q 的 SetRenderTargets 之后、首次绘制与任何清除之前又来一次 SetRenderTargets（游戏重复设目标，前一次是空通道）：新目标仍覆盖 P 就以它为 Q，否则不推迟
        // After Q's SetRenderTargets and before its first draw and any clear, another SetRenderTargets arrives (the game sets targets repeatedly, the previous one being an empty render pass): if the new targets still cover P take it as Q, otherwise do not defer
        if (!q_body && h.op == OP_SET_RT && !q_clears) { const p_set_rt *r2 = a; if (!defer_q_superset(r2)) { g_defer.reason = 15; goto fail; } q = r2; qop = p; p += h.size; continue; }
        if (h.op == OP_CLEAR_COLOR || h.op == OP_CLEAR_DS) q_clears = 1;
        if (!q_body && (h.op == OP_END || h.op == OP_SET_RT || h.op == OP_UPDATE_UB || h.op == OP_COPY_B2B || h.op == OP_DISPATCH)) { g_defer.reason = 9; g_defer.reason_op = h.op; goto fail; }
        if (h.op == OP_BIND_PROGRAM) pg = ((const p_bind_program *)a)->prog; else if (h.op == OP_BIND_TEX || h.op == OP_BIND_IMG) { const p_bind_handles *b = a; int lim = h.op == OP_BIND_IMG ? 32 : 128; if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (h.op == OP_BIND_IMG) img[b->stage][b->first + k] = b->h[k]; else tex[b->stage][b->first + k] = b->h[k]; } }
        else if (h.op == OP_CLEAR_COLOR) { const p_clear_color *c = a; if (c->index >= 0 && c->index < q->n && q->color[c->index]) for (int i = 0; i < g_defer.p_rt.n; i++) if (g_defer.p_rt.color[i] && sub_eq(q, c->index, &g_defer.p_rt, i)) { g_defer.reason = 10; goto fail; } }
        p += h.size;
    }
    if (!q || !defer_q_superset(q) || g_defer.n >= 64) { g_defer.reason = 11; goto fail; }
    for (int i = 0; i < 4; i++) if (&g_defer_slots[i] != &g_defer && g_defer_slots[i].active && g_defer_slots[i].q_op == qop && !defer_same_q_depth(&g_defer_slots[i], q)) { g_defer.reason = 17; goto fail; }
    g_defer.q_rt = *q; if (!q->depth && g_defer.p_rt.depth) { g_defer.q_rt.depth = g_defer.p_rt.depth; g_defer.q_rt.dview = g_defer.p_rt.dview; g_defer.q_rt.has_dview = g_defer.p_rt.has_dview; } { int na = 0; for (int i = 0; i < 4; i++) { const defer_t *d = &g_defer_slots[i]; if (d != &g_defer && d->active && d->at_q) na += d->n; }
      if (na + 1 > 64) { g_defer.reason = 20; goto fail; } }
    g_defer.q_op = qop; g_defer.p_end_op = p_end; g_defer.in_p = 1; g_defer.active = 1; g_defer_nact++; g_defer_pend = p_end; g_defer_n++; g_defer.born = g_defer_n;
    // 链式：本通道正是上游组的 Q（它们 at_q、等本通道开通道时回放），本通道又整体推迟——上游组按创建先后并进新组：
    // 上游组的目标空间 = 本通道目标 = 新组 P 目标，快照的映射复合为 pm' = 上游 lmap ∘ pm；上游的清除已在 R 里，下面随新组带走
    // Chained: this render pass is exactly the Q of upstream groups (they are at_q, waiting to be replayed when this render pass opens), and this render pass is itself deferred as a whole -- the upstream groups are merged into the new group in creation order:
    // the target space of an upstream group = this render pass's targets = the new group's P targets, and the snapshot's mapping is composed as pm' = upstream lmap o pm; the upstream clears are already in R and are carried along with the new group below
    for (int gi, after = 0; (gi = defer_next(1, after)) >= 0;) { defer_t *d = &g_defer_slots[gi]; after = d->born; if (d == &g_defer) continue;
        for (int k = 0; k < 8; k++) if (d->lmap[k] != 0xff && d->lmap[k] != k) { g_defer.reason = 18; goto fail; }   // 只并恒等映射的上游组（见 defer_q_superset） | Only merge in upstream groups with an identity mapping (see defer_q_superset)
        for (int i = 0; i < d->n; i++) { defer_snap z = d->snap[i]; uint8_t pm[8]; for (int k = 0; k < 8; k++) pm[k] = d->lmap[k] == 0xff || d->lmap[k] >= 8 ? 0xff : z.pm[d->lmap[k]];
            memcpy(z.pm, pm, 8); g_defer.snap[g_defer.n++] = z; }
        g_defer.reset_between |= d->reset_between; d->n = 0; defer_release_one(d); } if (!defer_take(dw, di)) { defer_release_one(&g_defer); return; }
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0; PD("G %llu 成立 组 %d P→Q draws=%d", g_stat_pass + 1, slot, draws);
}
// 合并的必要条件预筛（按提交索引二分，失败零扫描）：合并要求「后一个通道（下一个目标不同、未被提前的 SET_RT）的首次绘制
// 采样前一个通道的某张颜色附件」，即那次绘制的序号上有该附件的 SDX_READ 事件。不满足就必然不合并，省掉下面逐条扫描；
// 满足才走原规划（决定与不预筛时完全相同）。索引不可用时照旧规划。
// Pre-screen of the necessary condition for merging (binary search on the submit index, zero scanning on failure): merging requires that "the first draw of the later pass (the next SET_RT with a different target that has not been hoisted)
// samples some color attachment of the earlier pass", i.e. there is an SDX_READ event for that attachment at that draw's sequence number. If not satisfied there is certainly no merge, saving the per-operation scan below;
// only if satisfied does it go through the original planning (the decision is exactly the same as without the pre-screen). When the index is unavailable, plan as before.
static int fz_prefilter(void) {
    if (!g_sdx_ok || !g_exec_seq || g_sdx_op[g_exec_seq - 1] != g_exec_cur) return 1;
    uint32_t P = g_exec_seq - 1, lo = 0, hi = g_rtx_n;
    while (lo < hi) { uint32_t m = (lo + hi) / 2; if (g_rtx_seq[m] <= P) lo = m + 1; else hi = m; }
    for (; lo < g_rtx_n; lo++) { const uint8_t *op = g_sdx_op[g_rtx_seq[lo]]; if (g_hoist_n && hoist_skipped(op)) continue; if (!rt_same(&R.rt, (const p_set_rt *)(op + 8))) break; }
    if (lo >= g_rtx_n) return 0;
    uint32_t r2 = g_rtx_seq[lo], a = 0, b = g_drx_n;
    while (a < b) { uint32_t m = (a + b) / 2; if (g_drx_seq[m] <= r2) a = m + 1; else b = m; }
    if (a >= g_drx_n) return 0;
    uint32_t fd = g_drx_seq[a];
    for (uint32_t k = lo + 1; k < g_rtx_n && g_rtx_seq[k] < fd; k++) if (!(g_hoist_n && hoist_skipped(g_sdx_op[g_rtx_seq[k]]))) return 0;   // 首绘前又换目标：原规划第二步也放弃 | The target changes again before the first draw: step two of the original planning gives up as well
    for (int j = 0; j < R.rt.n && j < 8; j++) { if (!R.rt.color[j]) continue; int bk = sdx_bucket(R.rt.color[j]->serial, 0); if (bk < 0) continue;
        uint32_t e0 = g_sdx_bst[bk], e1 = g_sdx_bst[bk + 1];
        while (e0 < e1) { uint32_t m = (e0 + e1) / 2; if (g_sdx[m].seq < fd) e0 = m + 1; else e1 = m; }
        for (; e0 < g_sdx_bst[bk + 1] && g_sdx[e0].seq == fd; e0++) if (g_sdx[e0].kind == SDX_READ) return 1; }
    return 0;
}
static void fuse_try(void) {
    if (g_fz.active || !g_exec_next || g_hoist_n + 12 > HOIST_MAX || !R.rt.n) return;
    if (!fz_prefilter()) return;
    // 一：前一个通道到下一次 SetRenderTargets，只许绘制与状态
    // One: from the earlier pass to the next SetRenderTargets, only draws and state are allowed
    const uint8_t *p = NULL, *r2op = NULL, *reuse[8]; int nreuse = 0; const p_set_rt *r2 = NULL; p_scissor sc = R.sc; int has_sc = R.has_sc;
    fz_cursor cur = {.p = g_exec_next}; op_hdr h; const void *a;
    for (int guard = 0; guard < 200000 && !r2; guard++) {
        if (!fz_next(&cur, &p, &h, &a)) return;
        switch (h.op) {
            case OP_COPY_T2T: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH:
            case OP_CLEAR_COLOR: case OP_CLEAR_DS: case OP_UPDATE_UB: case OP_COPY_B2B: return;
            case OP_SCISSOR: sc = *(const p_scissor *)a; has_sc = 1; break;
            case OP_SET_RT: {
                int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
                if (rt_same(&R.rt, a)) {
                    if (g_enc_mode == 1 || pend || !R.has_rt || !tbdr_on() || nreuse >= 8) return;
                    reuse[nreuse++] = p; continue;
                }
                r2 = a; r2op = p; continue;
            }
            default: break;
        }
    }
    if (!r2 || rt_same(&R.rt, r2)) return;
    // 二：后一个通道开头到首次绘制，只许清除（不清共享附件）与状态
    // Two: from the start of the later pass to its first draw, only clears (not clearing shared attachments) and state are allowed
    const uint8_t *cl[8]; int ncl = 0; uint8_t clr[8] = {0}; MTLClearColor cc[8]; int cd_f = 0, cs_f = 0; double cdv = 0; uint32_t csv = 0;
    if (!fz_next(&cur, &p, &h, &a)) return;   // 接着第一步的游标（已越过后一个通道的 SET_RT，保留跨句柄 / 返回栈位置） | Continue from the cursor of step one (already past the later pass's SET_RT, keeping the cross-handle / return stack position)
    for (int guard = 0; guard < 200000; guard++) {
        if (h.op == OP_DRAW || h.op == OP_DRAW_IDX) break;
        switch (h.op) {
            case OP_COPY_T2T: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH: case OP_SET_RT: case OP_UPDATE_UB: case OP_COPY_B2B: return;
            case OP_SCISSOR: sc = *(const p_scissor *)a; has_sc = 1; break;
            case OP_RENDER_ENABLE: if (!((const p_flag *)a)->value) return; break;   // TOTK: conditional clears (see cond_eval)
            case OP_RENDER_COND: if (!cond_eval(a, p)) return; break;
            case OP_CLEAR_COLOR: { const p_clear_color *c = a; if (ncl >= 8 || c->index < 0 || c->index >= r2->n || !r2->color[c->index] || c->mask != 0xf) return;
                const tex_t *t = r2->color[c->index];
                if (has_sc && !(sc.x <= 0 && sc.y <= 0 && sc.x + sc.w >= (int)t->d.width && sc.y + sc.h >= (int)t->d.height)) return;
                clr[c->index] = 1; cc[c->index] = MTLClearColorMake(c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]); cl[ncl++] = p; break; }
            case OP_CLEAR_DS: { const p_clear_ds *c = a; if (ncl >= 8 || !r2->depth) return;
                if (c->depth_mask) { cd_f = 1; cdv = c->depth; } if (c->stencil_mask) { cs_f = 1; csv = (uint32_t)c->stencil; } cl[ncl++] = p; break; }
            default: break;
        }
        if (!fz_next(&cur, &p, &h, &a)) return;
    }
    // 三：附件并集（前一个通道的附件保持原序号，后一个通道独有的接在后面）、尺寸一致、深度相容
    // Three: union of attachments (the earlier pass's attachments keep their original indices, those unique to the later pass are appended), consistent sizes, compatible depth
    p_set_rt U = R.rt; int n = R.rt.n; int8_t p2map[8]; memset(p2map, -1, sizeof p2map);
    for (int k = 0; k < r2->n && k < 8; k++) { if (!r2->color[k]) continue;
        int j = -1; for (int m = 0; m < R.rt.n; m++) if (R.rt.color[m] && sub_eq(&R.rt, m, r2, k)) { j = m; break; }
        if (j < 0) { if (n >= 8) return; j = n++; U.color[j] = r2->color[k]; U.cview[j] = r2->cview[k]; U.has_cview[j] = r2->has_cview[k]; }
        else if (clr[k]) return;   // 后一个通道开头要清前一个通道正在画的附件：合并后无处安放这个清除 | The start of the later pass wants to clear an attachment the earlier pass is drawing to: after merging this clear has nowhere to go
        p2map[k] = (int8_t)j; }
    U.n = n;
    { uint64_t s0; int l0, x0; rt_sub(&U, 0, &s0, &l0, &x0); if (!U.color[0]) return;
      uint32_t w0 = MAX(U.color[0]->d.width >> l0, 1u), h0 = MAX(U.color[0]->d.height >> l0, 1u);
      for (int j = 1; j < n; j++) { uint64_t s_; int l_, x_; rt_sub(&U, j, &s_, &l_, &x_); if (!U.color[j] || MAX(U.color[j]->d.width >> l_, 1u) != w0 || MAX(U.color[j]->d.height >> l_, 1u) != h0) return; } }
    int depth_from_r2 = 0;
    if (R.rt.depth && r2->depth && !sub_eq(&R.rt, 8, r2, 8)) return;
    if (!R.rt.depth && r2->depth) { U.depth = r2->depth; U.dview = r2->dview; U.has_dview = r2->has_dview; depth_from_r2 = 1; }
    if ((cd_f || cs_f) && !depth_from_r2) return;   // 深度在前一个通道里就在用：后一个通道开头的深度清除无处安放 | The depth is already in use in the earlier pass: the depth clear at the start of the later pass has nowhere to go
    // 四：前一个通道的绘制不得采样后一个通道带来的附件；深度来自后一个通道时不得开深度模板测试
    // Four: draws of the earlier pass must not sample attachments brought in by the later pass; when the depth comes from the later pass, depth-stencil testing must not be enabled
    static uint64_t tex[6][128], img[6][32]; memcpy(tex, S.tex, sizeof tex); memcpy(img, S.img, sizeof img);
    const prog_t *pg = S.prog; ds_st ds = S.ds;
    cur = (fz_cursor){.p = g_exec_next};
    for (;;) {   // 第一次循环核开通道的这次绘制（当前状态），之后按区间里的绘制逐个核 | The first loop iteration checks the draw that opens the render pass (current state), then checks the draws in the range one by one
        if (depth_from_r2 && (ds.dt || ds.dw || ds.st)) return;
        for (int j = R.rt.n; j < n; j++) if (prog_samples_ser(pg, tex, img, U.color[j]->serial)) return;
        if (depth_from_r2 && prog_samples_ser(pg, tex, img, U.depth->serial)) return;
        for (;;) {
            const uint8_t *q; op_hdr hq; const void *aq;
            if (!fz_next(&cur, &q, &hq, &aq)) return;
            if (q == r2op) goto planned;
            if (hq.op == OP_BIND_PROGRAM) pg = ((const p_bind_program *)aq)->prog;
            else if (hq.op == OP_BIND_STATE) { const p_bind_state *b = aq; if (b->kind == 5) memcpy(&ds, b->bytes, sizeof ds); }
            else if (hq.op == OP_BIND_TEX || hq.op == OP_BIND_IMG) { const p_bind_handles *b = aq; int lim = hq.op == OP_BIND_IMG ? 32 : 128;
                if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (hq.op == OP_BIND_IMG) img[b->stage][b->first + k] = b->h[k]; else tex[b->stage][b->first + k] = b->h[k]; } }
            else if (hq.op == OP_DRAW || hq.op == OP_DRAW_IDX) break;
        }
    }
planned:
    {   // 五：预筛后一个通道的第一次绘制（静态条件），免得白挂并集附件：跟踪到那次绘制时的程序与贴图绑定 | Five: pre-screen the first draw of the later pass (static conditions), to avoid attaching the union of attachments for nothing: track the program and texture bindings up to that draw
        const prog_t *pg2 = S.prog; static uint64_t t2[6][128], i2[6][32]; memcpy(t2, S.tex, sizeof t2); memcpy(i2, S.img, sizeof i2);
        cur = (fz_cursor){.p = g_exec_next};
        for (;;) { const uint8_t *q; op_hdr hq; const void *aq; if (!fz_next(&cur, &q, &hq, &aq)) return; if (q == p) break;
            if (hq.op == OP_BIND_PROGRAM) pg2 = ((const p_bind_program *)aq)->prog;
            else if (hq.op == OP_BIND_TEX || hq.op == OP_BIND_IMG) { const p_bind_handles *b = aq; int lim = hq.op == OP_BIND_IMG ? 32 : 128;
                if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (hq.op == OP_BIND_IMG) i2[b->stage][b->first + k] = b->h[k]; else t2[b->stage][b->first + k] = b->h[k]; } }
        }
        NvProg *pr2 = pg2 ? prog_get(pg2) : nil; NvShader *vs2 = pr2 ? pr2->sh[0] : nil, *fs2 = pr2 ? pr2->sh[1] : nil;
        if (!vs2 || !fs2) return;
        int touches = 0;
        for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr2->sh[st]; if (!sh || sh->stage >= 6) continue;
            for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *bb = &sh->rf.b[i]; uint64_t h;
                if (bb->kind == NVN_RB_IMG) h = bb->reg < 32 ? i2[sh->stage][bb->reg] : 0; else if (bb->kind == NVN_RB_TEX || bb->kind == NVN_RB_TBUF) h = bb->reg < 128 ? t2[sh->stage][bb->reg] : 0; else continue;
                uint64_t ser = h ? h_ser(h) : 0; if (!ser) continue;
                int j = -1; for (int m = 0; m < R.rt.n; m++) if (R.rt.color[m] && R.rt.color[m]->serial == ser) { j = m; break; }
                if (j < 0) continue;
                if (sh != fs2 || bb->kind != NVN_RB_TEX || (fs2->rf.fc_mixed[bb->reg >> 6] >> (bb->reg & 63) & 1)) return;
                int any = 0;
                for (int c = 0; c < fs2->rf.nfc; c++) if (fs2->rf.fc_reg[c] == bb->reg) { int l = fs2->rf.fc_locn[c];
                    if (l >= 32 || (vs2->rf.uv_attr[l] < 0 && vs2->rf.uv_aff_attr[l] < 0)) return; any = 1; }
                if (!any) return;
                touches = 1; } }
        if (!touches) return;   // 后一个通道不读前一个通道的附件：合并没有收益 | The later pass does not read the earlier pass's attachments: merging has no benefit
        int ident = 1; for (int c = 0; c < 8; c++) if (fs2->rf.color_mask >> c & 1) { if (p2map[c] < 0) return; if (p2map[c] != c) ident = 0; }
        if (!ident && !fs2->rf.nfo) return;
    }
    memset(&g_fz, 0, sizeof g_fz);
    g_fz.active = 1; g_fz.phase = 1; g_fz.n1 = R.rt.n; g_fz.nreuse = nreuse; g_fz.rt1 = R.rt; g_fz.rt2 = *r2; g_fz.rt2_op = r2op; memcpy(g_fz.reuse, reuse, sizeof reuse); memcpy(g_fz.p2map, p2map, 8);
    for (int j = 0; j < 8; j++) g_fz.lmap1[j] = j < R.rt.n ? (uint8_t)j : 0xff;
    fz_probe_cov2(r2, p, p2map, clr, g_fz.n1, n);
    g_fz.skip[g_fz.nskip++] = r2op; for (int i = 0; i < ncl; i++) g_fz.skip[g_fz.nskip++] = cl[i];
    for (int i = 0; i < g_fz.nskip; i++) g_hoist_skip[g_hoist_n++] = g_fz.skip[i];
    R.rt = U;
    memcpy(g_fz.clr2, clr, 8); memcpy(g_fz.cc2, cc, sizeof cc); g_fz.cd2f = (uint8_t)cd_f; g_fz.cs2f = (uint8_t)cs_f; g_fz.cd2 = cdv; g_fz.cs2 = csv;
    for (int k = 0; k < 8; k++) if (clr[k]) { R.clear_c[p2map[k]] = 1; R.cc[p2map[k]] = cc[k]; }
    if (cd_f) { R.clear_d = 1; R.cd = cdv; } if (cs_f) { R.clear_s = 1; R.cs = csv; }
    if (!g_fz_n++) MLOG("可读 tile 通道合并：首例 通道 %llu（物理附件 %d 个）", g_stat_pass + 1, n);
    PD("F %llu 合并下一个通道（可读 tile）：物理附件 %d 个，前一个通道 %d 个", g_stat_pass + 1, n, R.rt.n == n ? g_fz.n1 : g_fz.n1);
}
static id<MTLFunction> bf_fn(NvShader *fs, const int *fx, uint8_t fbmask, const uint8_t *lmap) {   // 可读 tile 版本按常量特化（缓存） | The readable tile version is specialized by constants (cached)
    os_unfair_lock_lock(&g_fz_fn_lock);
    if (!g_bf_fn_cache) { g_bf_fn_cache = [NSMutableDictionary dictionary]; g_bf_lib_cache = [NSMutableDictionary dictionary]; }
    size_t kl = strlen(fs->key); NSString *base = [NSString stringWithFormat:@"%.*s", (int)(kl > 2 ? kl - 2 : kl), fs->key];
    id<MTLLibrary> lib = g_bf_lib_cache[base];
    if (!lib) {
        NSError *e = nil; NSString *path = [NSString stringWithFormat:@"%@/%@_bf.metallib", g_msl_dir, base];
        lib = lib_from_file(path, &e);
        if (!lib) MLOG("【红】可读 tile 版本载入失败 %s：%s", path.UTF8String, e.localizedDescription.UTF8String);
        else g_bf_lib_cache[base] = lib;
    }
    if (!lib) { os_unfair_lock_unlock(&g_fz_fn_lock); return nil; }
    NSMutableString *k = [NSMutableString stringWithFormat:@"%02x", fbmask];
    for (int i = 0; i < fs->rf.nfc; i++) [k appendFormat:@",%d", fx[i]];
    for (int j = 0; j < 8; j++) [k appendFormat:@"/%02x", lmap[j]];
    NSString *cacheKey = [NSString stringWithFormat:@"%@|%@", base, k];
    id<MTLFunction> f = g_bf_fn_cache[cacheKey];
    if (f) { os_unfair_lock_unlock(&g_fz_fn_lock); return f; }
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    for (int i = 0; i < fs->rf.nfc; i++) { int v = fx[i]; [cv setConstantValue:&v type:MTLDataTypeInt atIndex:(NSUInteger)i]; }
    for (int q = 0; q < 8; q++) { bool b = fbmask >> q & 1; [cv setConstantValue:&b type:MTLDataTypeBool atIndex:(NSUInteger)(32 + q)]; }
    for (int j = 0; j < fs->rf.nfo; j++) for (int q = 0; q < 8; q++) { bool b = lmap[q] == fs->rf.fo_color[j]; [cv setConstantValue:&b type:MTLDataTypeBool atIndex:(NSUInteger)(64 + 8 * j + q)]; }
    NSError *e = nil; f = [lib newFunctionWithName:@"main0" constantValues:cv error:&e];
    if (!f) { MLOG("【红】可读 tile 版本特化失败 %s（%s）：%s", fs->key, k.UTF8String, e.localizedDescription.UTF8String); os_unfair_lock_unlock(&g_fz_fn_lock); return nil; }
    g_bf_fn_cache[cacheKey] = f; os_unfair_lock_unlock(&g_fz_fn_lock); return f;
}
static int ubo_float(int slot, uint16_t ref, float *out) {   // uniform 槽的第 vec 个向量第 cmp 分量（CPU 侧读；落在本提交待执行的缓冲写里就不读） | Component cmp of vector vec of the uniform slot (read on the CPU side; not read if it falls within a buffer write pending execution in this submit)
    int sl = ref >> 8, vec = ref >> 2 & 63, cmp = ref & 3;
    if (slot < 0 || slot >= 6 || sl < 0 || sl >= 16) return 0;
    const p_bind_buf *u = &S.ubo[slot][sl]; uint64_t off = (uint64_t)vec * 16 + (uint64_t)cmp * 4;
    if (!u->addr || off + 4 > u->size || pend_hits(u->addr + off, 4)) return 0;
    void *mb; uint64_t mo; const uint8_t *base;
    if (!mtl_addr(u->addr + off, 4, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) return 0;
    memcpy(out, base + mo, 4); return 1;
}
static int zskip_fact_for(const char *key, zskip_fact *out) {
    if (!key) return 0;
    for (int i = 0; i < g_zskip_nf; i++) if (!strncmp(key, g_zskip_facts[i].key, strlen(g_zskip_facts[i].key))) { *out = g_zskip_facts[i]; return 1; }   // 运行时键带 _b 后缀 | Runtime keys carry the _b suffix
    return 0;
}
static int zskip_view_zero(int tid) {
    if (!tid || tid >= g_ntreg || !g_treg[tid].ok) return 0;
    treg_t *r = &g_treg[tid]; const tex_t *t = &r->t; int lv0 = r->has_v && r->v.set_levels ? r->v.base_level : 0;
    int nl = r->has_v && r->v.set_levels && r->v.nlevels > 0 ? r->v.nlevels : 1;
    int sl0 = r->has_v && r->v.set_layers ? r->v.min_layer : 0;
    int ns = r->has_v && r->v.set_layers && r->v.nlayers > 0 ? r->v.nlayers : 1;
    for (int l = 0; l < nl; l++) for (int s = 0; s < ns; s++) if (!zskip_has(t->serial, lv0 + l, sl0 + s)) return 0;
    return 1;
}
// 遮挡查询在计数时，被跳过的绘制本该贡献通过采样数：只有「下一次清零之前没有报告」才无关紧要。
// 从本次绘制之后顺着回放顺序看：先遇到采样数清零 → 无关（返回 1）；先遇到采样数报告、或看不到头（提交结束）→ 有关（返回 0）
// While an occlusion query is counting, a skipped draw should have contributed passed samples: it only does not matter if "there is no report before the next reset".
// Looking forward in replay order from after this draw: sample count reset encountered first -> does not matter (return 1); sample count report encountered first, or the end cannot be seen (submit ends) -> matters (return 0)
static int zskip_vis_dead(void) {
    if (!g_vis_counting) return 1;
    const uint8_t *p = g_exec_next; int ci = 0; if (!p) return 0;
    for (int guard = 0; guard < 200000; guard++) { op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        if (h.op == OP_JUMP) { memcpy(&p, a, 8); continue; }
        if (h.op == OP_END) { p = defer_cont(&ci); if (!p) return 0; continue; }
        if (h.op == OP_CALL) return 0;
        if (h.op == OP_RESET_COUNTER || h.op == OP_REPORT_COUNTER) { uint64_t t; memcpy(&t, a, 8); if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) return h.op == OP_RESET_COUNTER; }
        p += h.size; }
    return 0;
}
static int zskip_can_draw(NvProg *pr) {
    // 跳过条件（全部满足）：片元键在事实表、它反射声明的每个贴图槽绑的视图只覆盖已知全零的子资源、系数有限、
    // 不写深度不做模板、没在数遮挡查询、要写的每个颜色附件本来就是 0（本通道整张清 0 的候选，或已知全零）——写 0 盖 0。
    // 不成立原因（PASSDUMP 的 N 行）：1 深度 / 模板 2 遮挡查询计数中且清零前有报告 3 系数读不到或非有限 4 图像槽 5 被采样视图不是已知全零 6 无目标 7 目标不是 0
    // Skip conditions (all must hold): the fragment key is in the fact table, the view bound to every texture slot declared by its reflection covers only subresources known to be all zero, the coefficient is finite,
    // no depth write and no stencil, not counting an occlusion query, and every color attachment to be written is already 0 (a candidate cleared entirely to 0 in this render pass, or known all zero) -- writing 0 over 0.
    // Reasons for not holding (the N line of PASSDUMP): 1 depth / stencil 2 occlusion query counting with a report before the reset 3 coefficient unreadable or non-finite 4 image slot 5 sampled view not known all zero 6 no target 7 target is not 0
    if (!zskip_on() || !pr) return 0;
    NvShader *fs = nil; for (int st = 0; st < 5; st++) if (pr->sh[st] && pr->sh[st]->stage == 1) fs = pr->sh[st];
    zskip_fact f; float coeff; int why = 0;
    if (!fs || !zskip_fact_for(fs->key, &f)) return 0;
    if (S.ds.dw || S.ds.st) why = 1;
    else if (!zskip_vis_dead()) why = 2;
    else if (f.ubo_slot >= 0 && (!ubo_float(1, (uint16_t)((f.ubo_slot << 8) | (f.vec << 2) | f.comp), &coeff) || !isfinite(coeff))) why = 3;   // ubo_slot < 0：该着色器没有系数 | ubo_slot < 0: this shader has no coefficient
    for (int i = 0; !why && i < fs->rf.nb; i++) { nvn_rbind *b = &fs->rf.b[i];
        if (b->kind == NVN_RB_IMG) { why = 4; break; }
        if (b->kind != NVN_RB_TEX && b->kind != NVN_RB_TBUF) continue;
        int tid = b->reg < 128 ? (int)(S.tex[1][b->reg] & 0xfffff) : 0; if (!zskip_view_zero(tid)) why = 5; }
    if (!why && (!R.has_rt || !R.rt.n)) why = 6;
    for (int i = 0; !why && i < R.rt.n && i < 8; i++) { if (!R.rt.color[i]) continue;
        int pending = g_enc ? g_zskip_cand_ok[i] : zskip_zero_clear(i);   // 通道已开：看开通道时记下的候选（还没被画过）；未开：看挂起的清除 | Render pass already open: look at the candidates noted when the render pass was opened (not yet drawn to); not open: look at the pending clears
        if (!pending && (g_enc || R.clear_c[i] || !zskip_target_zero(R.rt.color[i], &R.rt.cview[i], R.rt.has_cview[i]))) why = 7; }
    if (why) PD("N %llu 零源不跳过 reason=%d 集合 %d", g_stat_pass + 1, why, g_zskip_n);
    return !why;
}
static int uv_proof(NvShader *vs, int locn, const p_draw *dw, const p_draw_idx *di) {   // 插值输出 locn 在每个像素中心恰为本像素的纹素中心：u = (x + 1) / 2、v = (1 − y) / 2，w = 1，视口与剪裁覆盖整块 | The interpolated output locn at every pixel center is exactly this pixel's texel center: u = (x + 1) / 2, v = (1 - y) / 2, w = 1, viewport and scissor cover the whole block
    if (locn < 0 || locn >= 32 || vs->rf.pos_attr < 0) return 0;
    int direct = vs->rf.uv_attr[locn] >= 0, aff = vs->rf.uv_aff_attr[locn] >= 0;
    if (!direct && !aff) return 0;
    float sc[4] = {1, 0, 1, 0};   // 原样直传 = 缩放 1、偏移 0 | Pass-through as-is = scale 1, offset 0
    if (!direct) for (int q = 0; q < 4; q++) if (!ubo_float(0, vs->rf.uv_aff[locn][q], &sc[q])) return 0;
    NSUInteger W = g_pass.tw, H = g_pass.th;
    if (S.has_vp && !(S.vp[0][0] == 0 && S.vp[0][1] == 0 && S.vp[0][2] == (float)W && S.vp[0][3] == (float)H)) return 0;
    if (R.has_sc && !(R.sc.x <= 0 && R.sc.y <= 0 && R.sc.x + R.sc.w >= (int)W && R.sc.y + R.sc.h >= (int)H)) return 0;
    float P[64][4], Q[64][4]; int n = dw ? dw->count : di->count;
    if (attr_read(vs->rf.pos_attr, dw, di, P) || attr_read(direct ? vs->rf.uv_attr[locn] : vs->rf.uv_aff_attr[locn], dw, di, Q)) return 0;
    for (int k = 0; k < n; k++) { float w = vs->rf.pos_w1 ? 1.0f : P[k][3];
        float u = direct ? Q[k][0] : fmaf(Q[k][0], sc[0], sc[1]), v = direct ? Q[k][1] : fmaf(Q[k][1], sc[2], sc[3]);   // 与着色器同一个 fma | The same fma as the shader
        if (w != 1.0f || u != (P[k][0] + 1.0f) * 0.5f || v != (1.0f - P[k][1]) * 0.5f) { PD("F %llu uv 证明：顶点 %d 位置 %g,%g uv %g,%g（缩放 / 偏移 %g,%g,%g,%g）", g_stat_pass, k, P[k][0], P[k][1], u, v, sc[0], sc[1], sc[2], sc[3]); return 0; } }
    return 1;
}
// 后一个通道的一次绘制：核对并给出特化。返回 0 = 证不出（调用方结束合并通道、按原目标重开）
// One draw of the later pass: verify and produce the specialization. Returns 0 = cannot be proven (the caller ends the merged render pass and reopens with the original targets)
static int fz_p2_draw(NvProg *pr, const p_draw *dw, const p_draw_idx *di, uint8_t lmap[8], __strong id<MTLFunction> *fn) {
    NvShader *vs = pr->sh[0], *fs = pr->sh[1]; *fn = nil;
    if (!vs || !fs) { PD("F %llu 证不出：无顶点 / 片元着色器", g_stat_pass); return 0; }
    if (!g_fz.rt2.depth && R.rt.depth && (S.ds.dt || S.ds.dw || S.ds.st)) { PD("F %llu 证不出：深度来自前一个通道而本次绘制开了深度模板测试", g_stat_pass); return 0; }
    int fx[32]; for (int i = 0; i < 32; i++) fx[i] = -1; uint8_t fbmask = 0;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? S.img[sh->stage][b->reg] : 0; else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? S.tex[sh->stage][b->reg] : 0; else continue;
            uint64_t ser = h ? h_ser(h) : 0; if (!ser) continue;
            int j = -1; for (int m = 0; m < g_fz.n1; m++) if (R.rt.color[m] && R.rt.color[m]->serial == ser) { j = m; break; }
            if (j < 0) continue;   // 采样的不是前一个通道的附件 | What is sampled is not an attachment of the earlier pass
            if (sh != fs || b->kind != NVN_RB_TEX) { PD("F %llu 证不出：非片元阶段或非贴图槽读前一个通道的附件", g_stat_pass); return 0; }
            if (fs->rf.fc_mixed[b->reg >> 6] >> (b->reg & 63) & 1) { PD("F %llu 证不出：槽 %d 混用", g_stat_pass, b->reg); return 0; }
            uint64_t s_; int lv, sl; rt_sub(&R.rt, j, &s_, &lv, &sl);
            if (lv || sl || MAX(R.rt.color[j]->d.width, 1u) != g_pass.tw || MAX(R.rt.color[j]->d.height, 1u) != g_pass.th) { PD("F %llu 证不出：附件不是第 0 级第 0 层或尺寸不符", g_stat_pass); return 0; }
            {   // 采样的那个视图也得正好是附件那一张（原先只核附件是第 0 级第 0 层，没核着色器绑的视图——同一张贴图按第 k 级 / 第 k 层 / 换格式的视图采样时， | The sampled view must also be exactly the attachment itself (previously only the attachment being level 0 layer 0 was checked, not the view bound by the shader -- when the same texture is sampled through a view of level k / layer k / a changed format,
                // 证明照样成立，采样会被改成读片上第 0 级 / 原格式，画面就错了）。视图起始级 / 层须为 0、格式须与附件实际格式相同，否则不合并
                // the proof would still hold, the sampling would be changed to read on-tile level 0 / the original format, and the picture would be wrong). The view's starting level / layer must be 0 and its format must equal the attachment's actual format, otherwise do not merge
                int tid_ = (int)(h & 0xfffff); void *tv_ = NULL; if (tid_) tex_fast_rid(tid_, &tv_); __unsafe_unretained id<MTLTexture> tv = (__bridge id<MTLTexture>)tv_;
                uint32_t apf_ = tv.parentTexture ? (uint32_t)tv.parentTexture.pixelFormat : (uint32_t)tv.pixelFormat; uint8_t swz_[4];
                if (R.rt.has_cview[j] && R.rt.cview[j].set_format && !(nvn_fmt_to_mtl(R.rt.cview[j].format, &apf_, swz_) && apf_)) apf_ = 0;   // 附件按换格式视图挂：取那个格式；认不出就不合 | The attachment is attached through a format-changing view: take that format; if unrecognized, do not merge
                MTLPixelFormat apf = (MTLPixelFormat)apf_;
                if (!tv || (tv.parentTexture && (tv.parentRelativeLevel || tv.parentRelativeSlice)) || tv.pixelFormat != apf) {
                    static int lg; if (lg++ < 8) LOG("[nvn-mtl] 片上读合并：槽 %d 采样视图（起始级 %lu 层 %lu 格式 %lu）与附件（第 0 级第 0 层 格式 %lu）不符，不合并（帧 %llu 通道 %llu）", b->reg,
                        tv && tv.parentTexture ? (unsigned long)tv.parentRelativeLevel : 0ul, tv && tv.parentTexture ? (unsigned long)tv.parentRelativeSlice : 0ul, (unsigned long)tv.pixelFormat, (unsigned long)apf,
                        (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass);
                    PD("F %llu 证不出：槽 %d 采样视图不是附件那一张（级 / 层 / 格式不符）", g_stat_pass, b->reg); return 0; } }
            int any = 0;
            for (int c = 0; c < fs->rf.nfc; c++) if (fs->rf.fc_reg[c] == b->reg) {
                if (!uv_proof(vs, fs->rf.fc_locn[c], dw, di)) { PD("F %llu 证不出：槽 %d 的采样坐标不是本像素（插值输出 %d）", g_stat_pass, b->reg, fs->rf.fc_locn[c]); return 0; }
                fx[c] = j; fbmask |= (uint8_t)(1u << j); any = 1; }
            if (!any) { PD("F %llu 证不出：槽 %d 的采样不在可改表里", g_stat_pass, b->reg); return 0; } } }
    memset(lmap, 0xff, 8);
    for (int c = 0; c < 8; c++) if (fs->rf.color_mask >> c & 1) { if (g_fz.p2map[c] < 0) { PD("F %llu 证不出：输出颜色 %d 无对应附件", g_stat_pass, c); return 0; } lmap[g_fz.p2map[c]] = (uint8_t)c; }
    int ident = !fbmask; for (int c = 0; c < 8; c++) if ((fs->rf.color_mask >> c & 1) && g_fz.p2map[c] != c) ident = 0;
    if (ident) { for (int q = 0; q < 8; q++) if (lmap[q] != 0xff) g_fz.wrote[q] = 1; if (fz_check_on()) fz_snap_take(dw, di); return 1; }   // 不读 tile、输出不移位：用原版本（写掩码仍按映射） | Does not read the tile and the output is not shifted: use the original version (the write mask still follows the mapping)
    if (!fs->rf.nfo) { bf_want_note(fs->key); PD("F %llu 证不出：没有可读 tile 版本（输出须移位）", g_stat_pass); return 0; }
    *fn = bf_fn(fs, fx, fbmask, lmap);
    if (!*fn) return 0;
    memset(&g_fz_desc, 0, sizeof g_fz_desc); g_fz_desc.kind = 1; g_fz_desc.fbmask = fbmask; for (int i = 0; i < 32; i++) g_fz_desc.fx[i] = (int8_t)fx[i];
    for (int q = 0; q < 8; q++) if (lmap[q] != 0xff) g_fz.wrote[q] = 1;
    if (fz_check_on()) fz_snap_take(dw, di);
    g_fz_draws++; return 1;
}
static void bloomq_reset(void) { memset(&g_bloomq, 0, sizeof g_bloomq); }
static int bloomq_enabled(void) {
    if (g_bloom_q < 0) { const char *e = getenv("DM_NVN_BLOOM_Q"); g_bloom_q = !e || strcmp(e, "0"); }
    return g_bloom_q;
}
static int bloomq_target(uint64_t *ser, int *lv, int *sl) {
    if (!R.has_rt || R.rt.n != 1 || !R.rt.color[0]) return 0;
    rt_sub(&R.rt, 0, ser, lv, sl); return *ser != 0;
}
static int bloomq_make_pipeline(MTLPixelFormat fmt) {
    if (g_bloomq_pso && g_bloomq_fmt == fmt) return 1;
    static const char *src =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct V { float4 p [[position]]; float2 uv; };\n"
        "vertex V vs(uint id [[vertex_id]]) { float2 uv=float2((id<<1)&2,id&2); V o; o.p=float4(uv*float2(2,-2)+float2(-1,1),0,1); o.uv=uv; return o; }\n"
        "struct P { float threshold; float coeff_g; float coeff_b; float dm; float sh; };\n"
        "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant P& p [[buffer(0)]]) { float2 d=float2(1.0/p.dm,1.0/p.sh); float3 c=0; for (int y=-1;y<=1;y+=2) for (int x=-1;x<=1;x+=2) { float3 v=t.sample(s,in.uv+float2(x,y)*d).rgb; float l=v.r*0.2126+v.g*p.coeff_g+v.b*p.coeff_b; if(l>=p.threshold) c+=v/(1.0+l); } return float4(c*0.25,1); }\n";
    NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src) options:nil error:&e];
    if (!lib) { MLOG("【红】泛光 1/4 着色器编译失败：%s", e.localizedDescription.UTF8String); return 0; }
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new]; pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"fs"]; pd.colorAttachments[0].pixelFormat = fmt;
    g_bloomq_pso = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!g_bloomq_pso) { MLOG("【红】泛光 1/4 管线失败：%s", e.localizedDescription.UTF8String); return 0; }
    g_bloomq_fmt = fmt;
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear; sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge; sd.supportArgumentBuffers = YES;
    g_bloomq_samp = [g_dev newSamplerStateWithDescriptor:sd]; return g_bloomq_samp != nil;
}
static int bloomq_capture(void) {
    uint64_t ser; int lv, sl; if (!bloomq_target(&ser, &lv, &sl) || !g_bloomq.s_tex) return 0;
    id<MTLTexture> base = tex_mtl(R.rt.color[0]), src = tex_mtl(g_bloomq.s_tex); if (!base || !src) return 0;
    id<MTLTexture> dst = base; if (lv || sl) { static id<MTLTexture> vb; static int vl, vs; if (vb != base || vl != lv || vs != sl) { g_bloomq_view = [base newTextureViewWithPixelFormat:base.pixelFormat textureType:base.textureType levels:NSMakeRange(lv, 1) slices:NSMakeRange(sl, 1)]; if (g_bloomq_view) res_add(g_bloomq_view); vb = base; vl = lv; vs = sl; } dst = g_bloomq_view; }
    if (!dst || !bloomq_make_pipeline(dst.pixelFormat)) return 0;
    rp_end();
    MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new]; rp.colorAttachments[0].texture = dst; rp.colorAttachments[0].loadAction = MTLLoadActionDontCare; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    blit_close(); rs_sync(); pt_mark(g_cb, 'R', rp); id<MTL4RenderCommandEncoder> en = [g_cb renderCommandEncoderWithDescriptor:rp]; es_reset(); if (!en) return 0;
    ((void)0); m3_attach(g_bloomq_args, en, M3_STAGE_FRAGMENT);
    [g_bloomq_args setTexture:src atIndex:0]; [g_bloomq_args setSamplerState:g_bloomq_samp atIndex:0];
    float *p = g_bloomq_const[g_bloomq_i].contents; p[0] = g_bloomq.threshold; p[1] = g_bloomq.coeff_g; p[2] = g_bloomq.coeff_b; p[3] = src.width; p[4] = src.height; [g_bloomq_args setAddress:g_bloomq_const[g_bloomq_i].gpuAddress atIndex:0];
    [en setViewport:(MTLViewport){0,0,dst.width,dst.height,0,1}]; [en setRenderPipelineState:g_bloomq_pso]; [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; m4_end(en);
    g_bloomq_i = (g_bloomq_i + 1) % 3; args_dirty_all(); g_pso_dirty = g_dss_dirty = 1; return 1;
}
// ---------- 拖货跟手图：商人货架上的货没有「拿起」态，拖动时由回放层把「画这件货图标的那次绘制」在同一趟末尾平移到手指处再画一遍 ----------
// 界面那一趟每个矩形是一次 4 顶点绘制（顶点 20 字节：位置 3 浮点 + 贴图坐标 2 浮点，位置经顶点常量 0 号的矩阵到裁剪坐标）。
// 前端（front_ios.m）给出手指按下的那一点、货架格网与跟手图中心（归一化，左上原点）；这里在界面趟里找「盖住这一点、四边都压在格线上、面积最小」的矩形 = 那件货
// （只按点认会认错：按下约 0.1 秒后游戏在按下点画一个以它为中心、一格大小的矩形，不压格线——真机上两次把它当成了货）
// （不依赖游戏的选中态：选中要等手柄光标落到货上约 0.1 秒，按下就拖的快拖等不到）。同一矩形上有格底和图标两次绘制：
// 格底用公共贴图（本趟很多矩形都用它），图标每件一张——取「本趟用到次数最少」的那张贴图的那次。认到的矩形回报给前端（货架原位盖暗色用）。
// 界面矩形在真实画面里的纵向位置与高度（都按占画面高的比例：y = 0 贴顶，h = 1 即整个画面）；前端登记，每次绘制现读（切主机 / 掌机档会变）。
// 游戏把界面画在界面贴图的上部 h 那一截（视口是整张贴图，投影矩阵里带着缩放）。
// ---------- Drag-follow image for shop items: items on a vendor's shelf have no "picked up" state, so during a drag the replay layer takes "the draw that draws this item's icon", translates it to the finger at the end of the same pass and draws it once more ----------
// In the UI pass each rectangle is one 4-vertex draw (20-byte vertices: position 3 floats + texture coordinates 2 floats; the position goes to clip coordinates through the matrix in vertex constant 0).
// The front end (front_ios.m) gives the point where the finger pressed, the shelf grid and the center of the drag-follow image (normalized, top-left origin); here, in the UI pass, find the rectangle that "covers this point, has all four edges on grid lines, and has the smallest area" = that item
// (identifying by the point alone gets it wrong: about 0.1 seconds after the press the game draws a one-cell-sized rectangle centered on the press point, not on grid lines -- on a real device it was twice taken for the item)
// (does not depend on the game's selected state: selection has to wait about 0.1 seconds for the controller cursor to land on the item, and a quick drag that starts on press cannot wait for that). The same rectangle has two draws, cell background and icon:
// the cell background uses a common texture (many rectangles in this pass use it), while each item has its own icon texture -- take the draw with the texture "used the fewest times in this pass". The recognized rectangle is reported back to the front end (used to cover the original shelf position with a dark color).
// Vertical position and height of the UI rectangle in the real picture (both as a fraction of the picture height: y = 0 touches the top, h = 1 is the whole picture); registered by the front end and read fresh on every draw (changes when switching between docked / handheld mode).
// The game draws the UI into the upper h portion of the UI texture (the viewport is the whole texture, and the projection matrix carries the scaling).
static void (*g_ui_rect_fn)(double *y, double *h);
void mtl_set_ui_rect_fn(void (*fn)(double *y, double *h)) { g_ui_rect_fn = fn; }
static double ui_hfrac(void) { double y = 0, h = 1; if (g_ui_rect_fn) g_ui_rect_fn(&y, &h); return h > 0 && h < 1 ? h : 1; }
static _Atomic int g_ghost_on; static _Atomic uint64_t g_ghost_pt, g_ghost_at, g_ghost_found, g_ghost_grid;
void mtl_ui_ghost(int on, float px, float py, const float grid[4], float cx, float cy) {
    if (on) { uint64_t gv = 0; for (int i = 0; i < 4; i++) gv |= (uint64_t)(uint16_t)(fminf(fmaxf(grid[i], 0), 1) * 65535 + .5f) << (16 * i); atomic_store(&g_ghost_grid, gv);
              atomic_store(&g_ghost_pt, (uint64_t)(uint32_t)lroundf(fminf(fmaxf(px, 0), 1) * 65535) | (uint64_t)(uint32_t)lroundf(fminf(fmaxf(py, 0), 1) * 65535) << 32);
              atomic_store(&g_ghost_at, (uint64_t)(uint32_t)(int32_t)lroundf(cx * 65535) | (uint64_t)(uint32_t)(int32_t)lroundf(cy * 65535) << 32); }
    else atomic_store(&g_ghost_found, 0);
    atomic_store(&g_ghost_on, on);
}
int mtl_ui_ghost_rect(float r[4]) { uint64_t v = atomic_load(&g_ghost_found); if (!v) return 0; for (int i = 0; i < 4; i++) r[i] = (float)((v >> (16 * i)) & 0xffff) / 65535.f; return 1; }
static struct { uint64_t pass; int n, nt, busy; float area, r[4]; struct { void *S_; p_draw dw; uint64_t tex; } c[4]; uint64_t tid[128]; uint16_t tn[128]; } g_gh;
static uint64_t ghost_tex(void) {   // 这次绘制的片元着色器实际采样的那张贴图（按着色器反射取槽位；S.tex 里别的槽是之前绘制留下的） | The texture actually sampled by this draw's fragment shader (slot taken from shader reflection; the other slots in S.tex are left over from earlier draws)
    __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; NvShader *f = pr ? pr->sh[1] : nil; if (!f) return 0;
    for (int i = 0; i < f->rf.nb; i++) if ((f->rf.b[i].kind == NVN_RB_TEX || f->rf.b[i].kind == NVN_RB_SAMP) && f->rf.b[i].reg < 128 && S.tex[1][f->rf.b[i].reg]) return S.tex[1][f->rf.b[i].reg];
    return 0; }
// 当前这次界面绘制是不是「不带贴图、正好盖满整个界面矩形」（界面矩形占贴图上部 hf）
// Whether the current UI draw is "untextured and exactly covering the whole UI rectangle" (the UI rectangle occupies the upper hf of the texture)
static int ui_cover_all(double hf) {
    if (S.vbuf[0].size != 80 || ghost_tex()) return 0;
    const p_bind_buf *u = &S.ubo[0][0]; void *vb, *mb; uint64_t vo, mo; const uint8_t *vp_, *mp_;
    if (!u->addr || u->size < 64 || pend_hits(u->addr, 64) || pend_hits(S.vbuf[0].address, 80)) return 0;
    if (!mtl_addr(S.vbuf[0].address, 80, &vb, &vo) || !(vp_ = [(__bridge id<MTLBuffer>)vb contents]) || !mtl_addr(u->addr, 64, &mb, &mo) || !(mp_ = [(__bridge id<MTLBuffer>)mb contents])) return 0;
    float m[16], x0 = 9, y0 = 9, x1 = -9, y1 = -9; memcpy(m, mp_ + mo, 64);
    for (int i = 0; i < 4; i++) { float f[2]; memcpy(f, vp_ + vo + 20 * i, 8); float nx = (m[0] * f[0] + m[4] * f[1] + m[12] + 1) / 2, ny = (m[1] * f[0] + m[5] * f[1] + m[13] + 1) / 2;
        x0 = fminf(x0, nx); x1 = fmaxf(x1, nx); y0 = fminf(y0, ny); y1 = fmaxf(y1, ny); }
    return x0 < 0.002f && x1 > 0.998f && y0 < 0.002f && fabsf(y1 - (float)hf) < 0.002f;
}
static void ghost_note(const p_draw *dw) {
    if (g_gh.busy || dw->count != 4 || S.nattr != 2 || S.stream[0].stride != 20 || S.vbuf[0].size != 80) return;
    if (g_gh.pass != g_stat_pass) { for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = g_gh.nt = 0; g_gh.pass = g_stat_pass; }
    uint64_t t = ghost_tex(); int k = 0; for (; k < g_gh.nt; k++) if (g_gh.tid[k] == t) break;
    if (k == g_gh.nt && g_gh.nt < 128) { g_gh.tid[k] = t; g_gh.tn[k] = 0; g_gh.nt++; } if (k < g_gh.nt) g_gh.tn[k]++;
    const p_bind_buf *u = &S.ubo[0][0]; void *vb, *mb; uint64_t vo, mo; const uint8_t *vp_, *mp_;
    if (!u->addr || u->size < 64 || pend_hits(u->addr, 64) || pend_hits(S.vbuf[0].address, 80)) return;
    if (!mtl_addr(S.vbuf[0].address, 80, &vb, &vo) || !(vp_ = [(__bridge id<MTLBuffer>)vb contents]) || !mtl_addr(u->addr, 64, &mb, &mo) || !(mp_ = [(__bridge id<MTLBuffer>)mb contents])) return;
    float m[16], x0 = 9, y0 = 9, x1 = -9, y1 = -9; memcpy(m, mp_ + mo, 64);
    for (int i = 0; i < 4; i++) { float f[2]; memcpy(f, vp_ + vo + 20 * i, 8); float nx = (m[0] * f[0] + m[4] * f[1] + m[12] + 1) / 2, ny = (m[1] * f[0] + m[5] * f[1] + m[13] + 1) / 2;
        x0 = fminf(x0, nx); x1 = fmaxf(x1, nx); y0 = fminf(y0, ny); y1 = fmaxf(y1, ny); }
    { float hf = (float)ui_hfrac(); y0 /= hf; y1 /= hf; }   // 前端给的点是界面矩形里的归一化坐标；界面只占贴图上部 hf 那一截时换到同一口径（hf = 1 时不变） | The points given by the front end are normalized coordinates within the UI rectangle; when the UI occupies only the upper hf portion of the texture, convert to the same convention (unchanged when hf = 1)
    uint64_t pv = atomic_load(&g_ghost_pt); float qx = (float)(uint32_t)pv / 65535.f, qy = (float)(pv >> 32) / 65535.f;
    if (qx < x0 || qx > x1 || qy < y0 || qy > y1) return;
    uint64_t gv = atomic_load(&g_ghost_grid); float gx = (float)(gv & 0xffff) / 65535.f, gy = (float)((gv >> 16) & 0xffff) / 65535.f, gw = (float)((gv >> 32) & 0xffff) / 65535.f, gh = (float)(gv >> 48) / 65535.f;
    if (gw <= 0 || gh <= 0) return;
    float w = x1 - x0, h = y1 - y0, area = w * h, c0 = (x0 - gx) / gw, r0 = (y0 - gy) / gh, cn = w / gw, rn = h / gh;   // 左上角在第几格、占几格 | Which cell the top-left corner is in, and how many cells it occupies
    if (fabsf(c0 - roundf(c0)) > 0.15f || fabsf(r0 - roundf(r0)) > 0.15f || fabsf(cn - roundf(cn)) > 0.15f || fabsf(rn - roundf(rn)) > 0.15f || cn < 0.85f || rn < 0.85f || cn > 2.15f || rn > 4.15f) return;   // 一件货最大 2×4 格 | One item is at most 2x4 cells
    if (g_gh.n && area > g_gh.area * 1.03f) return;                       // 已有更小的 | A smaller one already exists
    if (g_gh.n && area < g_gh.area * 0.97f) { for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = 0; }   // 找到更小的：前面的作废 | Found a smaller one: the earlier ones are invalidated
    if (g_gh.n == 4) { free(g_gh.c[0].S_); memmove(&g_gh.c[0], &g_gh.c[1], 3 * sizeof g_gh.c[0]); g_gh.n = 3; }
    g_gh.area = area; g_gh.r[0] = x0; g_gh.r[1] = y0; g_gh.r[2] = x1; g_gh.r[3] = y1;
    typeof(g_gh.c[0]) *c = &g_gh.c[g_gh.n++]; c->S_ = malloc(sizeof S); memcpy(c->S_, &S, sizeof S); c->dw = *dw; c->tex = t;
}
static void ghost_flush(void) {   // 渲染通道结束前：把认到的图标绘制平移到手指处再画一遍（挪视口，不改顶点；剪裁放开） | Before the render pass ends: translate the recognized icon draw to the finger and draw it once more (moves the viewport, does not change vertices; the scissor is opened up)
    if (g_gh.busy || !g_gh.n) return;
    if (g_gh.pass != g_stat_pass || !atomic_load(&g_ghost_on)) { for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = 0; return; }
    int best = -1, bn = 1 << 30; for (int i = 0; i < g_gh.n; i++) { int cnt = 0; for (int k = 0; k < g_gh.nt; k++) if (g_gh.tid[k] == g_gh.c[i].tex) cnt = g_gh.tn[k]; if (cnt < bn) { bn = cnt; best = i; } }
    typeof(g_gh.c[0]) *c = &g_gh.c[best]; uint64_t at = atomic_load(&g_ghost_at); float ax = (float)(int32_t)(uint32_t)at / 65535.f, ay = (float)(int32_t)(uint32_t)(at >> 32) / 65535.f;
    float W = (float)g_pass.tw, H = (float)g_pass.th, ccx = (g_gh.r[0] + g_gh.r[2]) / 2, ccy = (g_gh.r[1] + g_gh.r[3]) / 2;
    { uint64_t v = 0; for (int i = 0; i < 4; i++) v |= (uint64_t)(uint16_t)(fminf(fmaxf(g_gh.r[i], 0), 1) * 65535 + .5f) << (16 * i); atomic_store(&g_ghost_found, v ? v : 1); }
    g_gh.busy = 1; typeof(S) save; memcpy(&save, &S, sizeof S); p_scissor sc = R.sc; int has_sc = R.has_sc;
    memcpy(&S, c->S_, sizeof S); if (!S.has_vp) { S.vp[0][0] = S.vp[0][1] = 0; S.vp[0][2] = W; S.vp[0][3] = H; S.has_vp = 1; }
    S.vp[0][0] += (ax - ccx) * W; S.vp[0][1] += (ay - ccy) * H * (float)ui_hfrac(); R.has_sc = 0;
    args_dirty_all(); g_pso_dirty = g_dss_dirty = 1; draw(&c->dw, NULL);
    { static int once; if (!once++) MLOG("拖货跟手图：首例 通道 %llu，候选 %d 个，取贴图本趟用到 %d 次的那次", g_stat_pass, g_gh.n, bn); }
    memcpy(&S, &save, sizeof S); args_dirty_all(); R.sc = sc; R.has_sc = has_sc; g_pso_dirty = g_dss_dirty = 1; g_gh.busy = 0;
    for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = 0;
}
// TOTK: an 8-bit index buffer that GPU work earlier in this submit may have written is widened on the GPU timeline
// (compute, before the render pass that draws with it) instead of from CPU memory.
/* Widened buffers per draw record of the current submit: deferred draws are replayed from inside another draw, where
   the render pass must not be ended, so a record widened once is reused. */
#define WIDE_MAX 4096
static struct { const p_draw_idx *di; __unsafe_unretained id<MTLBuffer> buf; size_t off; } g_wide[WIDE_MAX]; static int g_nwide; static uint64_t g_wide_sub;
static int g_draw_depth;
static int wide_find(const p_draw_idx *di) { if (g_wide_sub != g_sub_serial) { g_nwide = 0; g_wide_sub = g_sub_serial; } for (int i = 0; i < g_nwide; i++) if (g_wide[i].di == di) return i; return -1; }
/* Widens count 8-bit indices at GPU address src (range checked by the caller) into this submit's arena. */
static void widen_u8_gpu(uint64_t src, uint32_t count, id<MTLBuffer> *out, size_t *off) {
    static id<MTLComputePipelineState> ps; static id<MTL4ArgumentTable> at;
    if (!ps) {
        NSError *e = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void widen(device const uchar *s [[buffer(0)]], device ushort *d [[buffer(1)]], constant uint2 &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {"
            " if (i < n.x) { uchar v = s[n.y + i]; d[i] = v == 0xff ? ushort(0xffff) : ushort(v); } }" options:nil error:&e];
        id<MTLFunction> fn = [lib newFunctionWithName:@"widen"];
        ps = fn ? [g_dev newComputePipelineStateWithFunction:fn error:&e] : nil;
        MTL4ArgumentTableDescriptor *ad = [MTL4ArgumentTableDescriptor new]; ad.maxBufferBindCount = 3; at = [M3ArgTable new];
        if (!ps || !at) MDIE("8-bit index widening kernel: %s", e.localizedDescription.UTF8String);
    }
    /* Buffer addresses are bound 16-byte aligned; the source's byte offset travels as a constant (index data may start anywhere). */
    uint64_t base = src & ~15ull; uint32_t shift = (uint32_t)(src - base);
    id<MTLBuffer> nb; size_t no = arena_alloc(16, &nb); uint32_t *np = (uint32_t *)((uint8_t *)nb.contents + no); np[0] = count; np[1] = shift;
    id<MTLBuffer> wb; *off = arena_alloc((size_t)count * 2 + 2, &wb); *out = wb;
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    id<MTL4ComputeCommandEncoder> ce = (id<MTL4ComputeCommandEncoder>)enc_c_begin(NULL); m3_attach(at, ce, M3_STAGE_COMPUTE);
    [at setAddress:base atIndex:0]; [at setAddress:wb.gpuAddress + *off atIndex:1]; [at setAddress:nb.gpuAddress + no atIndex:2];
    enc_c_pso(ce, ps); enc_c_dispatch(ce, MTLSizeMake(((NSUInteger)count + 63) / 64, 1, 1), MTLSizeMake(64, 1, 1));
    ((void)0);
    enc_c_end(ce); args_dirty_all();
}
static void widen_indices_gpu(const p_draw_idx *di) {
    gpu_path_on("TOTK_GPU_WIDEN", "GPU widening of 8-bit indices written earlier in the submit");
    void *mb; uint64_t mo, gpu;
    if (!mtl_gpu_addr(di->ibuf, (uint64_t)di->count, &mb, &mo, &gpu)) MDIE("8-bit index buffer range %#llx+%d not in a memory pool", (unsigned long long)di->ibuf, di->count);
    if (wide_find(di) >= 0) return;
    if (g_draw_depth > 1) MDIE("8-bit indices written earlier in the submit need GPU widening inside a replayed deferred draw");
    if (g_nwide == WIDE_MAX) MDIE("too many GPU-widened index buffers in one submit");
    id<MTLBuffer> wb; size_t wo; widen_u8_gpu(gpu, (uint32_t)di->count, &wb, &wo);   /* arena buffers live until the submit completes */
    g_wide[g_nwide++] = (typeof(g_wide[0])){di, wb, wo};
    static int once; if (!once++) MLOG("[totk] 8-bit indices written earlier in the submit are widened on the GPU");
}
// TOTK: color clears in the middle of a render pass, with a partial channel mask or a partial scissor are drawn as a
// full-target triangle that writes only that attachment's masked channels inside the scissor (depth/stencil untouched).
static void clear_quad_core(id<MTL4RenderCommandEncoder> en, const uint32_t cfmt[8], uint32_t dfmt, uint32_t sfmt, uint32_t samples,
                            NSUInteger tw, NSUInteger th, int index, int mask, const float rgba[4], MTLScissorRect sc) {
    static NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *cache; static id<MTLDepthStencilState> dss; static id<MTLLibrary> libs[8];
    if (!cache) {
        cache = [NSMutableDictionary dictionary];
        MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new]; dd.depthCompareFunction = MTLCompareFunctionAlways; dd.depthWriteEnabled = NO; dss = [g_dev newDepthStencilStateWithDescriptor:dd];
    }
    if (index < 0 || index > 7) MDIE("clear index %d", index);
    if (!libs[index]) {
        NSString *src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\nconstant float4 cc [[function_constant(0)]];\n"
            "vertex float4 cq_vs(uint id [[vertex_id]]) { float2 p = float2((id << 1) & 2, id & 2); return float4(p * 2 - 1, 0, 1); }\n"
            "struct O { float4 c [[color(%d)]]; };\nfragment O cq_fs() { O o; o.c = cc; return o; }\n", index];
        NSError *e = nil; libs[index] = [g_dev newLibraryWithSource:src options:totk_precise_opts() error:&e];
        if (!libs[index]) MDIE("clear shader: %s", e.localizedDescription.UTF8String);
    }
    NSString *key = [NSString stringWithFormat:@"%d/%x/%u,%u,%u,%u,%u,%u,%u,%u/%u/%u/%u/%a,%a,%a,%a", index, mask, cfmt[0], cfmt[1], cfmt[2], cfmt[3], cfmt[4], cfmt[5], cfmt[6], cfmt[7],
                     dfmt, sfmt, samples, rgba[0], rgba[1], rgba[2], rgba[3]];
    id<MTLRenderPipelineState> ps = cache[key];
    if (!ps) {
        MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new]; float v[4] = {rgba[0], rgba[1], rgba[2], rgba[3]};
        [cv setConstantValue:v type:MTLDataTypeFloat4 atIndex:0];
        NSError *e = nil;
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [libs[index] newFunctionWithName:@"cq_vs"]; d.fragmentFunction = [libs[index] newFunctionWithName:@"cq_fs" constantValues:cv error:&e];
        for (int i = 0; i < 8; i++) if (cfmt[i]) { d.colorAttachments[i].pixelFormat = (MTLPixelFormat)cfmt[i]; d.colorAttachments[i].writeMask = MTLColorWriteMaskNone; }
        d.colorAttachments[index].writeMask = (mask & 1 ? MTLColorWriteMaskRed : 0) | (mask & 2 ? MTLColorWriteMaskGreen : 0) | (mask & 4 ? MTLColorWriteMaskBlue : 0) | (mask & 8 ? MTLColorWriteMaskAlpha : 0);
        d.depthAttachmentPixelFormat = (MTLPixelFormat)dfmt; d.stencilAttachmentPixelFormat = (MTLPixelFormat)sfmt; d.rasterSampleCount = samples ? samples : 1;
        ps = d.fragmentFunction ? [g_dev newRenderPipelineStateWithDescriptor:d error:&e] : nil;
        if (!ps) MDIE("clear pipeline (index %d format %u): %s", index, cfmt[index], e.localizedDescription.UTF8String);
        cache[key] = ps;
    }
    enc_r_pso(en, ps); g_enc_pso = nil;
    enc_r_dss(en, dss); g_enc_dss = nil;
    if (g_enc_vis) { enc_r_visibility(en, MTLVisibilityResultModeDisabled, 0); g_enc_vis = 0; }   /* the clear is not a counted draw */
    enc_r_viewport(en, (MTLViewport){0, 0, (double)tw, (double)th, 0, 1});
    enc_r_scissor(en, sc);
    enc_r_cull(en, MTLCullModeNone); enc_r_fill(en, MTLTriangleFillModeFill); enc_r_depth_bias(en, 0, 0, 0);
    enc_r_draw(en, MTLPrimitiveTypeTriangle, 0, 3);
}
static void clear_quad(const p_clear_color *c) {
    gpu_path_on("TOTK_GPU_MIDCLEAR", "mid-pass / partial-mask / scissored color clear");
    const tex_t *t = R.rt.color[c->index];
    int f = R.rt.has_cview[c->index] && R.rt.cview[c->index].set_format ? R.rt.cview[c->index].format : t->d.format;
    if (cls_of(f)) MDIE("partial color clear of integer render target format %#x not implemented", f);
    id<MTL4RenderCommandEncoder> en = rp_begin(); if (!en) return;
    NSUInteger tw = g_pass.tw, th = g_pass.th; MTLScissorRect sc = {0, 0, tw, th};
    if (R.has_sc) { long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y < 0 ? 0 : R.sc.y, x1 = (long)R.sc.x + R.sc.w, y1 = (long)R.sc.y + R.sc.h;
        if (x1 > (long)tw) x1 = (long)tw; if (y1 > (long)th) y1 = (long)th; if (x1 <= x0 || y1 <= y0) return;
        sc = (MTLScissorRect){(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0)}; }
    if (!g_pass.cfmt[c->index]) MDIE("partial color clear: attachment %d not in the pass", c->index);
    clear_quad_core(en, g_pass.cfmt, g_pass.hd ? g_pass.dfmt : 0, g_pass.hs ? g_pass.sfmt : 0, g_pass.samples, tw, th, c->index, c->mask, c->rgba, sc);
}
/* Depth/stencil clears obey the scissor and stencil bit mask even inside an active pass. */
static void clear_ds_core(id<MTL4RenderCommandEncoder> en, const uint32_t cfmt[8], uint32_t dfmt, uint32_t sfmt, uint32_t samples,
                          NSUInteger tw, NSUInteger th, const p_clear_ds *c, MTLScissorRect sc) {
    static NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *pipelines;
    static NSMutableDictionary<NSNumber *, id<MTLDepthStencilState>> *states; static id<MTLLibrary> lib;
    if (!lib) {
        NSError *e = nil; lib = [g_dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nconstant float depth [[function_constant(0)]];\n"
            "vertex float4 ds_vs(uint id [[vertex_id]]) { float2 p = float2((id << 1) & 2, id & 2); return float4(p * 2 - 1, depth, 1); }\n" options:totk_precise_opts() error:&e];
        if (!lib) MDIE("depth/stencil clear shader: %s", e.localizedDescription.UTF8String);
        pipelines = [NSMutableDictionary dictionary]; states = [NSMutableDictionary dictionary];
    }
    float depth = c->depth_mask && dfmt ? fminf(1, fmaxf(0, c->depth)) : 0;
    NSString *key = [NSString stringWithFormat:@"%u,%u,%u,%u,%u,%u,%u,%u/%u/%u/%u/%a", cfmt[0], cfmt[1], cfmt[2], cfmt[3], cfmt[4], cfmt[5], cfmt[6], cfmt[7], dfmt, sfmt, samples, depth];
    id<MTLRenderPipelineState> ps = pipelines[key];
    if (!ps) {
        MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new]; [cv setConstantValue:&depth type:MTLDataTypeFloat atIndex:0];
        NSError *e = nil; MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [lib newFunctionWithName:@"ds_vs" constantValues:cv error:&e];
        for (int i = 0; i < 8; i++) if (cfmt[i]) { d.colorAttachments[i].pixelFormat = (MTLPixelFormat)cfmt[i]; d.colorAttachments[i].writeMask = MTLColorWriteMaskNone; }
        d.depthAttachmentPixelFormat = (MTLPixelFormat)dfmt; d.stencilAttachmentPixelFormat = (MTLPixelFormat)sfmt; d.rasterSampleCount = samples ? samples : 1;
        ps = d.vertexFunction ? [g_dev newRenderPipelineStateWithDescriptor:d error:&e] : nil;
        if (!ps) MDIE("depth/stencil clear pipeline: %s", e.localizedDescription.UTF8String); pipelines[key] = ps;
    }
    unsigned sm = sfmt ? c->stencil_mask & 255 : 0, dw = dfmt && c->depth_mask;
    NSNumber *dk = @(sm | (dw << 8)); id<MTLDepthStencilState> ds = states[dk];
    if (!ds) {
        MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new]; d.depthCompareFunction = MTLCompareFunctionAlways; d.depthWriteEnabled = dw;
        if (sm) { MTLStencilDescriptor *st = [MTLStencilDescriptor new]; st.stencilCompareFunction = MTLCompareFunctionAlways;
            st.depthStencilPassOperation = MTLStencilOperationReplace; st.writeMask = sm; d.frontFaceStencil = d.backFaceStencil = st; }
        ds = [g_dev newDepthStencilStateWithDescriptor:d]; states[dk] = ds;
    }
    enc_r_pso(en, ps); g_enc_pso = nil; enc_r_dss(en, ds); g_enc_dss = nil;
    if (g_enc_vis) { enc_r_visibility(en, MTLVisibilityResultModeDisabled, 0); g_enc_vis = 0; }
    enc_r_stencil_ref(en, c->stencil, c->stencil); enc_r_viewport(en, (MTLViewport){0, 0, (double)tw, (double)th, 0, 1}); enc_r_scissor(en, sc);
    enc_r_cull(en, MTLCullModeNone); enc_r_fill(en, MTLTriangleFillModeFill); enc_r_depth_bias(en, 0, 0, 0); enc_r_depth_clip(en, YES);
    enc_r_draw(en, MTLPrimitiveTypeTriangle, 0, 3);
}
static void clear_ds_quad(const p_clear_ds *c) {
    gpu_path_on("TOTK_GPU_MIDCLEAR", "mid-pass / masked / scissored depth-stencil clear");
    if (g_enc && ((c->depth_mask && !g_pass.hd) || (c->stencil_mask && !g_pass.hs))) rp_end();
    id<MTL4RenderCommandEncoder> en = rp_begin(); if (!en) return;
    NSUInteger tw = g_pass.tw, th = g_pass.th; MTLScissorRect sc = {0, 0, tw, th};
    if (R.has_sc) { int64_t x0 = MAX(R.sc.x, 0), y0 = MAX(R.sc.y, 0), x1 = MIN((int64_t)R.sc.x + R.sc.w, (int64_t)tw), y1 = MIN((int64_t)R.sc.y + R.sc.h, (int64_t)th);
        if (x1 <= x0 || y1 <= y0) return; sc = (MTLScissorRect){x0, y0, x1 - x0, y1 - y0}; }
    clear_ds_core(en, g_pass.cfmt, g_pass.hd ? g_pass.dfmt : 0, g_pass.hs ? g_pass.sfmt : 0, g_pass.samples, tw, th, c, sc);
}
static void prepare_draw_textures(NvProg *pr) {
    if (!pr) return;
    int planned = texture_plan_on();
    /* The direct-slot plan cannot describe handles stored in uniforms. */
    for(int st=0;st<2;st++)if(pr->sh[st])for(int i=0;i<pr->sh[st]->rf.nb;i++)
        if(pr->sh[st]->rf.b[i].kind==NVN_RB_TEXU)planned=0;
    if (planned) {
        unsigned total = 0;
        for (int st = 0; st < 2; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh) continue;
            if (!sh->texture_plan_ready) { sh->texture_count = nvn_texture_plan(&sh->rf, sh->texture_slots); sh->texture_plan_ready = 1; }
            if (atomic_load(&g_texture_plan_check)) {
                unsigned n = 0;
                for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; int slot = -1;
                    if ((b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) && b->reg < 128) slot = b->reg;
                    else if (b->kind == NVN_RB_IMG && b->reg < 32) slot = 128 + b->reg;
                    if (slot >= 0 && (n >= sh->texture_count || sh->texture_slots[n++] != slot)) MDIE("texture plan differs from reflection order");
                }
                if (n != sh->texture_count) MDIE("texture plan differs from reflection count");
                g_texture_plan_verified++;
            }
            total += sh->texture_count; g_texture_plan_total += sh->rf.nb; g_texture_plan_selected += sh->texture_count;
        }
        if (!total) return;
    }
    for (;;) {
        g_tex_prepare_split = 0; g_tex_preparing = 1;
        registry_lock();
        for (int st = 0; st < 2; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh) continue;
            int count = planned ? sh->texture_count : sh->rf.nb;
            for (int i = 0; i < count; i++) { int tid = 0;
                if (planned) { unsigned slot = sh->texture_slots[i]; tid = (int)((slot < 128 ? S.tex[st][slot] : S.img[st][slot - 128]) & 0xfffff); }
                else { const nvn_rbind *b = &sh->rf.b[i];
                    if(b->kind==NVN_RB_TEXU){
                        for(unsigned element=0;element<b->count;element++){
                            unsigned indirect=uniform_texture_handle(b,st,element)&0xfffff;
                            if(indirect)(void)tex_fast(indirect);
                            if(g_tex_prepare_split)break;
                        }
                    }
                    if ((b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) && b->reg < 128) tid = (int)(S.tex[st][b->reg] & 0xfffff);
                    else if (b->kind == NVN_RB_IMG && b->reg < 32) tid = (int)(S.img[st][b->reg] & 0xfffff);
                }
                if (tid > 0) (void)tex_fast(tid);
                if (g_tex_prepare_split) break;
            }
            if (g_tex_prepare_split) break;
        }
        registry_unlock(); g_tex_preparing = 0;
        if (!g_tex_prepare_split) return;
        /* rp_end may flush deferred draws which also use the registry. */
        g_tex_prepare_split = 0; rp_end();
    }
}
/* TOTK_MFX: the camera of the G-buffer pass, from its first draw: vertex uniform 1, the 2368-byte block (view, view-projection,
   projection, inverse view, near / far, aspect, field of view; floats 84..167 hold the camera of two frames earlier). */
static void mfx_take_camera(void) {
    g_mfx.want_cam = 0;
    const p_bind_buf *u = &S.ubo[0][1]; float cam[84];
    if (u->addr && u->size == 2368) {
        pend_materialize(u->addr, (uint8_t *)cam, sizeof cam, 0);   // the bytes this draw sees: memory plus pending inline updates
        mfx_note_camera(cam, 84);
        /* TOTK_MFX_CAMLOG=<first>-<last frame>: every captured camera (diagnostic) */
        static long c0 = -2, c1; if (c0 == -2) { const char *e = getenv("TOTK_MFX_CAMLOG"); c0 = e ? atol(e) : -1; c1 = e && strchr(e, '-') ? atol(strchr(e, '-') + 1) : c0; }
        long f = (long)atomic_load(&g_cur_frame);
        if (c0 >= 0 && f >= c0 && f <= c1) MLOG("[mfx] camera F%ld pass %llu block %#llx size %.0fx%.0f near %g fov %.4f yaw %.3f pos %.3f %.3f %.3f vp0 %.5f %.5f %.5f %.3f",
            f, g_stat_pass, (unsigned long long)u->addr, cam[72], cam[73], cam[56], cam[70], atan2(cam[2], cam[0]) * 180 / M_PI, cam[47], cam[51], cam[55], cam[12], cam[13], cam[14], cam[15]);
    } else { static int said; if (said++ < 5) MLOG("[mfx] G-buffer pass %llu: vertex uniform 1 is %#llx / %llu bytes, not the 2368-byte camera block (frame %llu)", g_stat_pass,
                                                 (unsigned long long)u->addr, (unsigned long long)u->size, (unsigned long long)atomic_load(&g_cur_frame)); }
}
static void draw_inner(const p_draw *dw, const p_draw_idx *di);
/* Diagnostic DM_NVN_NANDRAW=<first>-<last frame> (log DM_NVN_NANDRAW_LOG, default analysis/nandraw.tsv): after every
   draw into a full-size RG11B10F or R32F colour attachment (the HDR target, the linear depth), end the pass and count
   Inf / NaN texels in the centre 128x128 of each such attachment, so the first draw that writes a non-finite value can be
   named. Same Metal 4 queue as the draws, so each count belongs to the state right after that draw. Slow: one pass per draw. */
static void nan_draw_probe(const p_draw *dw, const p_draw_idx *di) {
    static long f0 = -2, f1; static FILE *f; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    static long last_frame = -1; static unsigned seq;
    if (f0 == -2) {
        const char *e = getenv("DM_NVN_NANDRAW"), *path = getenv("DM_NVN_NANDRAW_LOG");
        f0 = e ? atol(e) : -1; f1 = e && strchr(e, '-') ? atol(strchr(e, '-') + 1) : f0;
        if (f0 >= 0 && !(f = fopen(path ? path : "analysis/nandraw.tsv", "w"))) MDIE("DM_NVN_NANDRAW_LOG: cannot open %s", path ? path : "analysis/nandraw.tsv");
        if (f) fprintf(f, "frame\tseq\tpass\tatt\tfmt\ttex\tvs\tfs\tblend\twmask\tcount\tinst\tbase\tmean0\tmean1\tmean2\tnf0\tnf1\tnf2\n");
    }
    if (f0 < 0 || !g_enc || g_defer_replaying) return;
    long fr = (long)atomic_load(&g_cur_frame);
    if (fr < f0 || fr > f1) return;
    id<MTLTexture> srcs[8]; int atts[8], fmts[8], n = 0; uint64_t sers[8];
    for (int i = 0; i < R.rt.n && i < 8; i++) {
        if (!R.rt.color[i]) continue;
        id<MTLTexture> t = tex_mtl(R.rt.color[i]);
        if (!t || t.width < 1280 || t.height < 720 || (t.pixelFormat != MTLPixelFormatRG11B10Float && t.pixelFormat != MTLPixelFormatR32Float)) continue;
        srcs[n] = t; atts[n] = i; fmts[n] = t.pixelFormat == MTLPixelFormatR32Float ? 32 : 11; sers[n] = R.rt.color[i]->serial;
        if (t.pixelFormat == MTLPixelFormatR32Float) { static uint64_t said_ser; if (said_ser != R.rt.color[i]->serial) { said_ser = R.rt.color[i]->serial;
            MLOG("[nandraw] R32F attachment F%ld c%d ser=%llu mem=%p size=%llu flags=%#x levels=%u mtlobj=%p parent=%p", fr, i, (unsigned long long)R.rt.color[i]->serial, (void *)R.rt.color[i]->mem,
                 (unsigned long long)R.rt.color[i]->L.size, R.rt.color[i]->d.flags, R.rt.color[i]->d.levels, (__bridge void *)t, (__bridge void *)t.parentTexture); } }
        n++;
    }
    if (!n) return;
    if (fr != last_frame) { last_frame = fr; seq = 0; }
    unsigned sq = seq++;
    __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil;
    char vs[17] = "-", fs[17] = "-";
    if (pr && pr->sh[0]) snprintf(vs, sizeof vs, "%.16s", pr->sh[0]->key);
    if (pr && pr->sh[1]) snprintf(fs, sizeof fs, "%.16s", pr->sh[1]->key);
    NSString *vss = @(vs), *fss = @(fs);   /* blocks cannot capture C arrays */
    unsigned blend = S.color.blend_en, wmask = S.chmask.mask, count = dw ? (unsigned)dw->count : di ? (unsigned)di->count : 0;
    unsigned inst = dw ? (unsigned)dw->instances : di ? (unsigned)di->instances : 0; int base = di ? di->base_vertex : dw ? (int)dw->first : 0;
    uint64_t pass = g_stat_pass;
    rp_end();   /* the next draw reopens the pass; pending clears were consumed when it was opened */
    for (int k = 0; k < n; k++) {
        id<MTLTexture> src = srcs[k]; int att = atts[k], fm = fmts[k]; uint64_t ser = sers[k];
        id<MTLBuffer> rb; size_t ro = arena_alloc(128 * 128 * 4, &rb);
        blit_close(); rs_sync(); id<MTL4ComputeCommandEncoder> be = [g_cb computeCommandEncoder];
        ((void)0);
        enc_b_copy_texture_buffer(be, src, 0, 0, MTLOriginMake((src.width - 128) / 2, (src.height - 128) / 2, 0), MTLSizeMake(128, 128, 1), rb, ro, 128 * 4, 128 * 128 * 4);
        m4_end(be);
        [g_commit_blocks addObject:^{
            const uint32_t *v = (const uint32_t *)((const uint8_t *)rb.contents + ro); double sum[3] = {0}; unsigned nf[3] = {0}, finite = 0;
            for (unsigned q = 0; q < 128 * 128; q++) {
                uint32_t z = v[q]; int bad = 0;
                if (fm == 32) { if (((z >> 23) & 0xff) == 0xff) { nf[0]++; bad = 1; } else { float x; memcpy(&x, &z, 4); sum[0] += x; } }
                else { uint32_t c[3] = {z & 0x7ff, (z >> 11) & 0x7ff, (z >> 22) & 0x3ff};
                    for (int j = 0; j < 3; j++) if ((c[j] >> (j == 2 ? 5 : 6)) == 31) { nf[j]++; bad = 1; }
                    if (!bad) for (int j = 0; j < 3; j++) sum[j] += c[j]; }
                if (!bad) finite++;
            }
            pthread_mutex_lock(&mu);
            fprintf(f, "%ld\t%u\t%llu\t%d\t%d\t%llu\t%s\t%s\t%#x\t%#x\t%u\t%u\t%d\t%.3f\t%.1f\t%.1f\t%u\t%u\t%u\n", fr, sq, (unsigned long long)pass, att, fm, (unsigned long long)ser,
                    vss.UTF8String, fss.UTF8String, blend, wmask, count, inst, base, finite ? sum[0] / finite : 0, finite ? sum[1] / finite : 0, finite ? sum[2] / finite : 0, nf[0], nf[1], nf[2]);
            fflush(f); pthread_mutex_unlock(&mu); }];
    }
}
/* TOTK_EXP_SKIP_INVISIBLE=1 (MEASUREMENT ONLY, the picture may be wrong): every draw gets its own visibility counter; a
   draw whose key (program + mesh range + index buffer, not its constants) produced no sample in each of the last two
   frames is not encoded. Static scene: the saved GPU / CPU time is the cost of fully occluded or off-screen draws
   (2026-10-16: about half of all draws, 37% of the vertices). Objects that share a key are skipped only when all of them
   were invisible. Not a feature: a camera turn makes it skip things that just became visible. */
typedef struct { uint64_t key; uint32_t last_frame, nonzero_frame, stable_after; uint16_t streak; uint16_t used; } skv_ent;   // stable_after: no skipping before this frame (a key that came back after being skipped flickers at the visibility boundary)
enum { SKV_CAP = 1 << 16, SKV_STREAK = 4, SKV_HOLD = 120 };   // table size; frames without a sample before a key is skipped; frames a revived key is left alone
static skv_ent *g_skv; static int g_skv_cur = -1;
/* Invisible-draw skipping (on by default on the Mac from 2026-10-16 17:50 to that night, then off again: the user saw holes
   in the ground and black shadows while playing, see skv_on). A draw whose key (program, mesh range, index buffer, pass type)
   passed no sample for 4 consecutive frames is not drawn; every K frames (TOTK_SKIP_RETEST, default 8) it is drawn and counted
   again and comes back when visible. TOTK_SKIP_INVISIBLE=1 turns it on; TOTK_EXP_SKIP_INVISIBLE=1 alone is the old
   measurement form (no retest). Off in the headless probes. Runtime: `flag skip_invisible 0|1`. Known flaw: keys shared by
   several draws of a frame (see skv_on). */
static int skv_retest(void) {
    static int K = -1;
    if (K < 0) { const char *r = getenv("TOTK_SKIP_RETEST"); if (!r) r = getenv("TOTK_EXP_SKIP_RETEST");
        K = r ? atoi(r) : (getenv("TOTK_EXP_SKIP_INVISIBLE") && !getenv("TOTK_SKIP_INVISIBLE") ? 0 : 8); if (K < 0) K = 0; }
    return K;
}
static int skv_skip_storers(void) {   /* TOTK_SKIP_STORERS (default 1): hidden geometry whose material stores the per-tile class flag may be skipped; 0 = strict */
    static int v = -1; if (v < 0) { const char *e = getenv("TOTK_SKIP_STORERS"); v = e ? atoi(e) != 0 : 1; } return v;
}
static _Atomic int g_skv_on = -1;
static int skv_on(void) {
    int v = atomic_load_explicit(&g_skv_on, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("TOTK_SKIP_INVISIBLE"), *x = getenv("TOTK_EXP_SKIP_INVISIBLE");
        /* Off by default everywhere since 2026-10-16 night: in real play (camera turns) the user saw holes in the ground and
           black shadows, the gain is 0.6 ms, and a draw sharing its key with a later visible draw of the same frame could stay
           skipped indefinitely (the first draw sets last_frame, the later ones keep nonzero_frame current, the streak never
           resets and the retest never comes). TOTK_SKIP_INVISIBLE=1 / "flag skip_invisible 1" for experiments only. */
        int def = 0;
        v = e ? atoi(e) != 0 : x ? atoi(x) != 0 : def;
        if (getenv("TOTK_HEADLESS")) v = 0;   // probes replay fixed scenes frame after frame: no per-draw counters, nothing skipped
        if (v && !g_skv) g_skv = calloc(SKV_CAP, sizeof *g_skv);
        if (v) MLOG("不可见绘制跳过：开（连续 %d 帧没通过一个样本的绘制不画，每 %d 帧重测%s，重测后又可见的 %d 帧内不再跳；着色器会写内存的屏幕空间绘制（≤6 顶点）不跳，会写每格类别标志的几何%s；TOTK_SKIP_INVISIBLE=0 关，运行中 flag skip_invisible 0|1）", SKV_STREAK, skv_retest(), skv_retest() ? "" : "——不重测，只量开销", SKV_HOLD, skv_skip_storers() ? "照跳（TOTK_SKIP_STORERS=0 严格）" : "也不跳（严格，TOTK_SKIP_STORERS=0）");
        else MLOG("不可见绘制跳过：关（TOTK_SKIP_INVISIBLE=1 / flag skip_invisible 1 开）");
        atomic_store(&g_skv_on, v);
    }
    return v;
}
static void skv_set(int on) {   // runtime switch: turning it on allocates the table, turning it off stops skipping at the next draw
    if (on && !g_skv) g_skv = calloc(SKV_CAP, sizeof *g_skv);
    atomic_store(&g_skv_on, on != 0); MLOG("[flag] skip_invisible = %d（每 %d 帧重测）", on != 0, skv_retest());
}
static int perdraw_vis_on(void) { return anom_vis_on() || skv_on(); }
static uint64_t skv_hash(uint64_t h, uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); return h * 0xff51afd7ed558ccdull; }
/* returns 1 when the draw is to be skipped; otherwise leaves its table index in g_skv_cur for skv_note */
static int skv_check(const void *pr, const p_draw *dw, const p_draw_idx *di) {
    g_skv_cur = -1;
    /* "no sample passed" means "wrote nothing" only when a failed sample cannot write: depth test on with an ordinary function
       (not ALWAYS / NEVER) and no stencil write on fail / depth-fail (the material-class marker draws write stencil on
       depth fail: 0 samples passed, yet they decide how Link is lit -- skipping them painted him black). */
    { int keep = 1 /* NVN_STENCIL_OP_KEEP */;
      int df_ok = S.ds.dt && S.ds.df != 8 && S.ds.df != 1;
      int st_ok = !S.ds.st || ((S.ds.fail_f == keep && S.ds.zfail_f == keep && S.ds.fail_b == keep && S.ds.zfail_b == keep) || (!S.st_wmask[0] && !S.st_wmask[1]));
      if (!df_ok || !st_ok) return 0;
      /* shader side effects (stores to storage buffers / images in the vertex or fragment stage) happen whether or not a
         sample passes (no shader declares early_fragment_tests). rw comes from the MSL's stores (refl_mark_stores): the
         translator flag said read-only for every shader, so the deferred-lighting quads (dbe0acd9 ...), which store a per-tile
         feedback word the next frame's class marker reads, were skipped as "0 samples" -- the title mural went dark except on
         retest frames (2026-10-16 night). Every material fragment shader stores too: an unconditional per-tile class-group flag
         (`tile = 1u`, indexed by the fragment position) that the next frame's marker turns into class depth for the tile's own
         pixels. A hidden fragment's flag is redundant for the visible pixels of its tile (they flag it themselves) and cannot
         light a pixel that has no class depth, so hidden geometry may still be skipped (TOTK_SKIP_STORERS=0: never). The
         screen-space draws (<= 6 vertices per instance: the lighting / marker / post quads, whose stores are the conditional
         feedback itself) are never skipped: skipping them broke the loop at the title. Strict rule costs the whole gain
         (11.50 -> 11.22 ms vs 10.4 with it). */
      __unsafe_unretained NvProg *np = (__bridge NvProg *)pr;
      int storer = 0;
      for (int st = 0; st < 2 && !storer; st++) { NvShader *sh = np ? np->sh[st] : nil; if (!sh) continue;
          for (int i = 0; i < sh->rf.nb; i++) if (sh->rf.b[i].rw) { storer = 1; break; } }
      if (storer) { uint32_t cnt = dw ? dw->count : di->count; if (cnt <= 6 || !skv_skip_storers()) { g_skv_effects++; return 0; } } }
    uint64_t h = skv_hash(0x1234567, (uint64_t)(uintptr_t)pr);
    /* the view matters: the same mesh is invisible from one environment-cube face and visible from the next, and the
       cube faces take turns frame by frame; key the render targets too (colour 0 and depth: serial, level, slice) */
    /* pass type (size and formats: G-buffer / shadow cascade / cube face ...) and the view's level and layer, not the target's serial:
       many targets are rebuilt every frame, so their serials never match across frames */
    for (int i = 0; i < 2; i++) { const tex_t *t = i ? R.rt.depth : (R.rt.n ? R.rt.color[0] : NULL); const view_t *v = i ? &R.rt.dview : &R.rt.cview[0]; int hv = i ? R.rt.has_dview : (R.rt.n ? R.rt.has_cview[0] : 0);
        h = skv_hash(h, t ? ((uint64_t)t->d.format << 40 | (uint64_t)t->d.width << 20 | t->d.height) : 0); h = skv_hash(h, hv && v->set_levels ? (uint64_t)v->base_level + 1 : 0); h = skv_hash(h, hv && v->set_layers ? (uint64_t)v->min_layer + 1 : 0); }
    if (dw) { h = skv_hash(h, (uint64_t)(uint32_t)dw->first << 32 | (uint32_t)dw->count); h = skv_hash(h, (uint64_t)(uint32_t)dw->instances << 32 | (uint32_t)dw->base_instance); }
    else { h = skv_hash(h, di->ibuf); h = skv_hash(h, (uint64_t)(uint32_t)di->count << 32 | (uint32_t)di->base_vertex); h = skv_hash(h, (uint64_t)(uint32_t)di->instances << 32 | (uint32_t)di->itype); }
    uint32_t cur = (uint32_t)atomic_load(&g_cur_frame);
    for (uint32_t i = (uint32_t)h & (SKV_CAP - 1), n = 0; n < 64; i = (i + 1) & (SKV_CAP - 1), n++) {
        skv_ent *e = &g_skv[i];
        if (!e->used || (uint32_t)(cur - e->last_frame) > 900) {   // free, or not seen for 900 frames (streamed-out meshes): the table ages instead of filling up
            e->used = 1; e->key = h; e->streak = 0; e->last_frame = cur; e->nonzero_frame = cur; e->stable_after = cur; g_skv_cur = (int)i; return 0; }
        if (e->key != h) continue;
        /* skip after SKV_STREAK consecutive frames without a sample, and not within SKV_HOLD frames of coming back: a key that
           was skipped and then seen again sits at the visibility boundary (swaying grass, particles); skipping it again right
           away shows it only on retest frames, i.e. a one-frame flicker (the anomaly detector counted 2-4 per session) */
        if (e->streak >= SKV_STREAK && e->last_frame + 1 >= cur && e->last_frame != cur && (int32_t)(cur - e->stable_after) >= 0) {
            /* TOTK_EXP_SKIP_RETEST=<K>: a skipped draw is drawn (and counted) again every K frames, so something that became
               visible shows up at most K frames late; without it a draw once invisible for 3 frames was skipped for ever
               (the character-class lighting quads during the fade-in: Link stayed black). The entries' phases differ. */
            int K = skv_retest();
            if (K > 0 && (cur - e->nonzero_frame) % (uint32_t)K == 0) { g_skv_cur = (int)i; g_skv_retests++; return 0; }
            e->last_frame = cur; g_skv_skipped++; return 1; }
        g_skv_cur = (int)i; return 0;
    }
    g_skv_full++; return 0;   // 64 probes without a free or matching slot: drawn, never skipped
}
static void skv_note(uint32_t slot) {   /* this draw's counter slot in g_vis */
    if (g_skv_cur < 0) return;
    if (g_skv_n == g_skv_cap) { g_skv_cap = g_skv_cap ? g_skv_cap * 2 : 8192; g_skv_list = realloc(g_skv_list, (size_t)g_skv_cap * sizeof *g_skv_list); if (!g_skv_list) MDIE("skv list"); }
    g_skv_list[g_skv_n++] = (skv_item){(uint32_t)g_skv_cur, slot, (void *)CFBridgingRetain(g_vis)}; g_skv_tested++; g_skv_cur = -1;
}
static void skv_done(skv_item *l, int n, uint32_t frame) {   /* submission complete: read the counters, update the streaks */
    for (int j = 0; j < n; j++) { uint64_t v = ((const uint64_t *)((__bridge id<MTLBuffer>)l[j].buf).contents)[l[j].slot]; skv_ent *e = &g_skv[l[j].idx]; if (v) e->nonzero_frame = frame; }
    for (int j = 0; j < n; j++) { skv_ent *e = &g_skv[l[j].idx]; if (e->last_frame != frame) { e->last_frame = frame; if (e->nonzero_frame == frame) { if (e->streak >= SKV_STREAK) { g_skv_revived++; e->stable_after = frame + SKV_HOLD; } e->streak = 0; } else if (e->streak < 1000) e->streak++; } CFRelease(l[j].buf); }
    free(l);
}
static void draw(const p_draw *dw, const p_draw_idx *di) { g_draw_depth++; draw_inner(dw, di); if (g_draw_depth == 1) nan_draw_probe(dw, di); g_draw_depth--; }
// TOTK: an update makes the argument buffers of the stages whose bound uniform ranges overlap it stale.
static void upd_dirty_stages(uint64_t lo, uint64_t hi) {
    for (int st = 0; st < 6; st++) for (int r = 0; r < 16; r++) {
        const p_bind_buf *u = &S.ubo[st][r];
        if (u->addr && u->addr < hi && lo < u->addr + (u->size ? u->size : 1)) { args_dirty_stage(st); break; }
    }
}
// TOTK: non-uniform buffer accesses of one stage that may touch renamed uniform updates (checked before an encoder opens).
static void upd_guard_stage(NvShader *sh, int st) {
    if (!g_nupd || !sh || st < 0 || st > 5) return;
    for (int i = 0; i < sh->rf.nb && g_nupd; i++) {
        const nvn_rbind *b = &sh->rf.b[i]; int reg = b->reg;
        if (b->kind == NVN_RB_SBUFP && reg < 16) { const p_bind_buf *u = &S.ssbo[st][reg]; if (u->addr) upd_guard(u->addr, u->addr + (u->size ? u->size : 1)); }
        else if (b->kind == NVN_RB_SBUFU && reg < 16) {
            const p_bind_buf *u = &S.ubo[st][reg]; uint64_t va = 0;
            if (u->addr && u->size >= (uint64_t)b->off + 8) { pend_read(u->addr + b->off, &va, 8, 0); if (va) upd_guard(va, va + 65536); }
        } else if (b->kind == NVN_RB_TBUF && reg < 128) {
            int tid = (int)(S.tex[st][reg] & 0xfffff); uint64_t lo = 0, n = 0;
            registry_lock(); if (tid > 0 && tid < g_ntreg && g_treg[tid].ok) { lo = (uint64_t)(uintptr_t)g_treg[tid].t.mem; n = g_treg[tid].t.L.size; } registry_unlock();
            if (lo) upd_guard(lo, lo + (n ? n : 1));
        }
    }
}
static void upd_guard_draw(NvProg *pr, const p_draw_idx *di) {
    if (!g_nupd) return;
    for (int s_ = 0; s_ < 16 && g_nupd; s_++) if (S.vbuf[s_].address) upd_guard(S.vbuf[s_].address, S.vbuf[s_].address + (S.vbuf[s_].size ? S.vbuf[s_].size : 1));
    if (di && di->ibuf && g_nupd) upd_guard(di->ibuf, di->ibuf + (uint64_t)di->count * (di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4));
    if (pr) { upd_guard_stage(pr->sh[0], 0); upd_guard_stage(pr->sh[1], 1); }
}
/* TOTK_ANOMALY=<dir> (diagnostic, 2026-10-15): catch single-frame picture anomalies -- jumping now and then renders a
   distant part of the map in front of the camera for one frame -- together with the draws that made them. Every draw's
   identity and the vertex-stage data it reads go into a ring of the last ANOM_FRAMES game frames: uniform blocks and
   storage buffers as the GPU will see them at that point (pend_materialize: memory plus this submission's pending
   inline updates; GPU-written buffers show their previous contents), as address, size, hash of the first 4 KiB and the
   first 16 floats, plus the vertex textures' serials. anom_present (at each present) reads four strips of the picture
   back and writes the ring and the pictures to <dir> when one frame differs from both neighbours. */
enum { ANOM_FRAMES = 8, ANOM_BUFS = 8, ANOM_TEX = 4, ANOM_HASH_MAX = 4096 };
typedef struct { uint64_t addr, hash; uint32_t size; uint8_t reg, kind; float f[16]; } anom_buf;
typedef struct { uint64_t ser, mem; const void *mtl; uint32_t w, h, fmt; } anom_tex;
typedef struct {
    uint64_t pass, ib, vb0, tex[ANOM_TEX]; const char *vs, *fs;
    int32_t first, count, inst, bv; uint8_t nb, ntex, nft, nrt; anom_buf b[ANOM_BUFS];
    anom_tex ft[8], rt[3];   // fragment-stage textures; colour targets 0-1 and the depth target
    uint64_t ih, vh[4]; uint32_t ilen, vlen[4]; uint8_t nvs;   // content hashes of the index range and of the first 4 KiB of the vertex streams the attributes use (from base vertex on)
    int64_t smp;   // samples the draw passed (its own visibility counter; -1 = not counted, -2 = pending)
    char kind;     // 'D' draw, 'U' inline uniform update (ib = destination, count = bytes, b[0].f = data), 'Y' buffer copy (vb0 = source)
} anom_draw;
static struct { pthread_mutex_t mu; uint64_t frame; anom_draw *d; uint32_t n, cap; } g_anom[ANOM_FRAMES];
static const char *g_anom_dir; static int g_anom_on = -1, g_anom_rec = 1;   // TOTK_ANOMALY_RECORD=0: detect and save pictures only (no per-draw records: less CPU per draw)
static int anom_on(void) {
    if (g_anom_on < 0) {
        g_anom_dir = getenv("TOTK_ANOMALY"); g_anom_on = g_anom_dir && *g_anom_dir;
        for (int i = 0; i < ANOM_FRAMES; i++) pthread_mutex_init(&g_anom[i].mu, NULL);
        const char *r = getenv("TOTK_ANOMALY_RECORD"); g_anom_rec = !(r && !strcmp(r, "0"));
        r = getenv("TOTK_ANOMALY_INITIAL");
        int unset = -1; atomic_compare_exchange_strong(&g_anom_requested, &unset, !(r && !atoi(r)));
        if (g_anom_on && mkdir(g_anom_dir, 0755) != 0 && errno != EEXIST) MLOG("【红】[anom] cannot create %s: %s (pictures and records will be missing)", g_anom_dir, strerror(errno));
        if (g_anom_on) MLOG("[anom] single-frame anomaly capture armed -> %s (initially %s, per-draw records %s)", g_anom_dir, atomic_load(&g_anom_requested) ? "on" : "off", g_anom_rec ? "on" : "off");
    }
    return g_anom_on && atomic_load(&g_anom_requested);
}
static uint64_t anom_hash(const uint8_t *p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull ^ n; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); h = (h ^ w) * 0x100000001b3ull; h ^= h >> 29; }
    for (; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}
static void anom_note(NvProg *pr, const p_draw *dw, const p_draw_idx *di) {
    uint64_t f = atomic_load(&g_cur_frame); int k = (int)(f % ANOM_FRAMES);
    static uint8_t tmp[ANOM_HASH_MAX];   // submit thread only
    pthread_mutex_lock(&g_anom[k].mu);
    if (g_anom[k].frame != f) { g_anom[k].frame = f; g_anom[k].n = 0; }
    if (g_anom[k].n == g_anom[k].cap) {
        uint32_t c = g_anom[k].cap ? g_anom[k].cap * 2 : 4096; anom_draw *d = realloc(g_anom[k].d, c * sizeof *d);
        if (!d) MDIE("anomaly ring allocation"); g_anom[k].d = d; g_anom[k].cap = c;
    }
    anom_draw *a = &g_anom[k].d[g_anom[k].n++]; memset(a, 0, sizeof *a); a->smp = -1; a->kind = 'D';
    if (g_anom_vis_slot >= 0 && g_vis) {
        if (g_anom_nvis == g_anom_capvis) { g_anom_capvis = g_anom_capvis ? g_anom_capvis * 2 : 8192; g_anom_vis = realloc(g_anom_vis, g_anom_capvis * sizeof *g_anom_vis); if (!g_anom_vis) MDIE("anomaly counters"); }
        g_anom_vis[g_anom_nvis++] = (anom_vis_ent){k, f, g_anom[k].n - 1, (uint32_t)g_anom_vis_slot, (void *)CFBridgingRetain(g_vis)}; a->smp = -2;
    }
    g_anom_vis_slot = -1;
    a->pass = g_stat_pass; a->vs = pr && pr->sh[0] ? pr->sh[0]->key : "-"; a->fs = pr && pr->sh[1] ? pr->sh[1]->key : "-";
    a->first = dw ? dw->first : 0; a->count = dw ? dw->count : di->count; a->inst = dw ? 1 : di->instances; a->bv = di ? di->base_vertex : 0;
    a->ib = di ? di->ibuf : 0; a->vb0 = S.vbuf[0].address;
    __unsafe_unretained NvShader *v = pr ? pr->sh[0] : nil;
    for (int i = 0; v && i < v->rf.nb; i++) {
        const nvn_rbind *b = &v->rf.b[i];
        if (b->kind == NVN_RB_TEX && b->reg < 128 && a->ntex < ANOM_TEX) {
            int tid = (int)(S.tex[0][b->reg] & 0xfffff);
            registry_lock(); a->tex[a->ntex++] = tid && tid < g_ntreg ? g_treg[tid].t.serial : 0; registry_unlock();
            continue;
        }
        const p_bind_buf *u = b->kind == NVN_RB_CBUF && b->reg < 16 ? &S.ubo[0][b->reg] : b->kind == NVN_RB_SBUFP && b->reg < 16 ? &S.ssbo[0][b->reg] : NULL;
        if (!u || !u->addr || a->nb == ANOM_BUFS) continue;
        anom_buf *o = &a->b[a->nb++]; o->addr = u->addr; o->size = (uint32_t)u->size; o->reg = (uint8_t)b->reg; o->kind = b->kind == NVN_RB_CBUF ? 'c' : 's';
        size_t n = u->size ? (u->size < ANOM_HASH_MAX ? (size_t)u->size : ANOM_HASH_MAX) : 64;
        pend_materialize(u->addr, tmp, n, 0);
        o->hash = anom_hash(tmp, n); memcpy(o->f, tmp, n < sizeof o->f ? n : sizeof o->f);
    }
    __unsafe_unretained NvShader *fsh = pr ? pr->sh[1] : nil;
    registry_lock();
    for (int i = 0; fsh && i < fsh->rf.nb && a->nft < 8; i++) {
        const nvn_rbind *b = &fsh->rf.b[i];
        if (b->kind != NVN_RB_TEX || b->reg >= 128) continue;
        int tid = (int)(S.tex[1][b->reg] & 0xfffff); anom_tex *t = &a->ft[a->nft++];
        if (!tid || tid >= g_ntreg) continue;
        const tex_t *x = &g_treg[tid].t; void *tv = NULL; (void)tex_fast_rid(tid, &tv);
        *t = (anom_tex){x->serial, (uint64_t)x->mem, tv, x->d.width, x->d.height, x->d.format};
    }
    registry_unlock();
    for (int i = 0; i < 2 && i < R.rt.n; i++) if (R.rt.color[i]) { const tex_t *x = R.rt.color[i]; a->rt[i] = (anom_tex){x->serial, (uint64_t)x->mem, NULL, x->d.width, x->d.height, x->d.format}; }
    if (R.rt.depth) { const tex_t *x = R.rt.depth; a->rt[2] = (anom_tex){x->serial, (uint64_t)x->mem, NULL, x->d.width, x->d.height, x->d.format}; }
    a->nrt = (uint8_t)(R.rt.n < 2 ? R.rt.n : 2);
    if (di && di->ibuf && di->count > 0) {   // index bytes as the GPU will read them (zero-copy memory)
        size_t isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4, n = (size_t)di->count * isz; if (n > ANOM_HASH_MAX) n = ANOM_HASH_MAX;
        void *mb; uint64_t mo; if (mtl_addr(di->ibuf, n, &mb, &mo)) { a->ih = anom_hash((const uint8_t *)(uintptr_t)di->ibuf, n); a->ilen = (uint32_t)n; }
    }
    uint32_t seen = 0;
    for (int i = 0; i < S.nattr && i < 16 && a->nvs < 4; i++) {
        int st = S.attr[i].stream; if (st < 0 || st >= 16 || (seen >> st & 1) || !S.vbuf[st].address) continue; seen |= 1u << st;
        uint64_t stride = S.stream[st].stride, off = di && stride ? (uint64_t)(int64_t)di->base_vertex * stride : 0, sz = S.vbuf[st].size;
        if (sz && off >= sz) continue;
        size_t n = sz ? (size_t)(sz - off) : 256; if (n > ANOM_HASH_MAX) n = ANOM_HASH_MAX;
        void *mb; uint64_t mo; if (!mtl_addr(S.vbuf[st].address + off, n, &mb, &mo)) continue;
        a->vh[a->nvs] = anom_hash((const uint8_t *)(uintptr_t)(S.vbuf[st].address + off), n); a->vlen[a->nvs++] = (uint32_t)n;
    }
    pthread_mutex_unlock(&g_anom[k].mu);
}
static int anom_vis_on(void) { return anom_on() && g_anom_rec; }
static void anom_note_xfer(char kind, uint64_t src, uint64_t dst, uint64_t n, const void *data) {
    if (!anom_vis_on()) return;
    uint64_t f = atomic_load(&g_cur_frame); int k = (int)(f % ANOM_FRAMES);
    pthread_mutex_lock(&g_anom[k].mu);
    if (g_anom[k].frame != f) { g_anom[k].frame = f; g_anom[k].n = 0; }
    if (g_anom[k].n == g_anom[k].cap) {
        uint32_t c = g_anom[k].cap ? g_anom[k].cap * 2 : 4096; anom_draw *d = realloc(g_anom[k].d, c * sizeof *d);
        if (!d) MDIE("anomaly ring allocation"); g_anom[k].d = d; g_anom[k].cap = c;
    }
    anom_draw *a = &g_anom[k].d[g_anom[k].n++]; memset(a, 0, sizeof *a);
    a->kind = kind; a->smp = -1; a->pass = g_stat_pass; a->vs = a->fs = "-"; a->ib = dst; a->vb0 = src; a->count = (int32_t)n;
    a->nb = 1; a->b[0].addr = dst; a->b[0].size = (uint32_t)n;
    size_t m = n < sizeof a->b[0].f ? (size_t)n : sizeof a->b[0].f;
    if (data) memcpy(a->b[0].f, data, m); else pend_materialize(src, (uint8_t *)a->b[0].f, m, 0);
    pthread_mutex_unlock(&g_anom[k].mu);
}
static void anom_vis_done(anom_vis_ent *e, int n) {   // on completion of the submission: each draw's sample count into its ring record
    for (int j = 0; j < n; j++) {
        uint64_t v = ((const uint64_t *)((__bridge id<MTLBuffer>)e[j].buf).contents)[e[j].slot];
        pthread_mutex_lock(&g_anom[e[j].k].mu);
        if (g_anom[e[j].k].frame == e[j].f && e[j].i < g_anom[e[j].k].n) g_anom[e[j].k].d[e[j].i].smp = (int64_t)v;
        pthread_mutex_unlock(&g_anom[e[j].k].mu);
        CFRelease(e[j].buf);
    }
}
/* Opt-in geometry input capture on a requested passdump frame. Copy storage and indices
   on the same GPU queue before this draw, then save on completion. Uniforms are the
   submission snapshot plus inline updates. This deliberately splits passes; no FPS use. */
static void geometry_dump(NvProg *pr, const p_draw *dw, const p_draw_idx *di) {
    static const char *prefix, *dir; static int enabled = -1;
    static uint64_t frame = ~0ull; static unsigned seq;
    if (enabled < 0) { prefix = getenv("DM_NVN_GEODUMP_VS"); dir = getenv("DM_NVN_CBDUMP"); enabled = prefix && *prefix && dir && *dir; }
    uint64_t fs = atomic_load(&g_diag_geometry_fs);
    if ((!enabled && !fs) || !dir || !*dir || !passdump() || g_defer_replaying || !pr || !pr->sh[0]) return;
    if (fs) {
        if (!pr->sh[1] || !diag_shader_prefix_matches(pr->sh[1]->key, fs)) return;
    } else if (strncmp(pr->sh[0]->key, prefix, strlen(prefix))) return;
    uint64_t fr = atomic_load(&g_cur_frame); if (fr != frame) { frame = fr; seq = 0; }
    if (seq >= 64) return;
    NSString *base = [NSString stringWithFormat:@"%s/geo-F%llu-%u", dir, (unsigned long long)fr, seq++];
    NSMutableDictionary *info = [@{@"frame":@(fr), @"vs":@(pr->sh[0]->key), @"fs":@(pr->sh[1] ? pr->sh[1]->key : ""),
        @"count":@(dw ? dw->count : di->count), @"first":@(dw ? dw->first : 0), @"base_vertex":@(di ? di->base_vertex : 0),
        @"index_type":@(di ? di->itype : -1), @"instances":@(dw ? dw->instances : di->instances)} mutableCopy];
    NSMutableArray *attributes = [NSMutableArray array], *streams = [NSMutableArray array];
    uint16_t used = 0;
    for (int i = 0; i < pr->sh[0]->rf.nv; i++) {
        unsigned a = pr->sh[0]->rf.v[i].nvn;
        if (a >= 16 || a >= (unsigned)S.nattr) continue;
        va_st attr = S.attr[a]; if (attr.stream >= 16) continue;
        [attributes addObject:@{@"nvn":@(a), @"format":@((unsigned)attr.fmt), @"stream":@((unsigned)attr.stream), @"offset":@((unsigned)attr.offset)}];
        used |= (uint16_t)(1u << attr.stream);
    }
    for (unsigned st = 0; st < 16; st++) {
        if (!(used & (1u << st)) || !S.vbuf[st].address || !S.vbuf[st].size) continue;
        uint64_t addr = S.vbuf[st].address, requested = S.vbuf[st].size;
        void *mb; uint64_t mo; if (!mtl_addr(addr, 1, &mb, &mo)) continue;
        id<MTLBuffer> source = (__bridge id<MTLBuffer>)mb;
        uint64_t size = MIN(MIN(requested, source.length - mo), 2u << 20);
        if (!size) continue;
        upd_guard(addr, addr + size);
        id<MTLBuffer> rb; size_t ro = arena_alloc((size_t)size, &rb);
        enc_b_copy_buffer(blit(), source, mo, rb, ro, (NSUInteger)size); enc_b_end(g_blit); g_blit = nil;
        NSString *path = [base stringByAppendingFormat:@"-vertex%u.bin", st];
        const uint8_t *bytes = (const uint8_t *)rb.contents + ro;
        [g_commit_blocks addObject:^{ FILE *f = fopen(path.UTF8String, "wb"); if (f) { fwrite(bytes, 1, (size_t)size, f); fclose(f); } }];
        [streams addObject:@{@"stream":@(st), @"address":@(addr), @"size":@(size), @"requested_size":@(requested),
            @"truncated":@(size != requested), @"stride":@(S.stream[st].stride), @"divisor":@(S.stream[st].divisor)}];
    }
    info[@"attributes"] = attributes; info[@"vertex_streams"] = streams;
    for (int i = -1; i < pr->sh[0]->rf.nb; i++) {
        uint64_t addr = 0, size = 0; NSString *suffix;
        if (i < 0) { if (!di || !di->ibuf || di->count <= 0) continue;
            addr = di->ibuf; size = (uint64_t)di->count * (di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4); suffix = @"indices";
        } else {
            const nvn_rbind *b = &pr->sh[0]->rf.b[i]; if (b->reg >= 16) continue;
            const p_bind_buf *u = &S.ubo[0][b->reg];
            if (b->kind == NVN_RB_CBUF && u->addr && u->size && u->size <= 65536) {
                NSMutableData *data = [NSMutableData dataWithLength:(NSUInteger)u->size]; pend_materialize(u->addr, data.mutableBytes, (size_t)u->size, 0);
                [data writeToFile:[base stringByAppendingFormat:@"-cb%d.bin", b->reg] atomically:NO]; continue;
            }
            if (b->kind != NVN_RB_SBUFU || !u->addr || u->size < b->off + 8u) continue;
            pend_read(u->addr + b->off, &addr, 8, 0); size = 2u << 20; suffix = [NSString stringWithFormat:@"storage%d", b->aid];
        }
        void *mb; uint64_t mo; if (!addr || !mtl_addr(addr, 1, &mb, &mo)) continue;
        id<MTLBuffer> source = (__bridge id<MTLBuffer>)mb;
        size = MIN(size, source.length - mo); if (!size || size > (2u << 20)) continue;
        upd_guard(addr, addr + size);
        id<MTLBuffer> rb; size_t ro = arena_alloc((size_t)size, &rb);
        enc_b_copy_buffer(blit(), source, mo, rb, ro, (NSUInteger)size); enc_b_end(g_blit); g_blit = nil;
        NSString *path = [base stringByAppendingFormat:@"-%@.bin", suffix];
        const uint8_t *bytes = (const uint8_t *)rb.contents + ro;
        [g_commit_blocks addObject:^{ FILE *f = fopen(path.UTF8String, "wb"); if (f) { fwrite(bytes, 1, (size_t)size, f); fclose(f); } }];
        info[suffix] = @{@"address":@(addr), @"size":@(size)};
    }
    [[NSJSONSerialization dataWithJSONObject:info options:NSJSONWritingPrettyPrinted error:nil] writeToFile:[base stringByAppendingString:@".json"] atomically:NO];
}
/* The production vertex descriptor only reads streams referenced by stage_in.
   Old bindings from earlier programs may occupy all 16 slots; resolving them on
   every draw adds pool locks and Metal argument updates without a GPU reader.
   Route A / unknown shaders retain the conservative original path. */
static uint16_t vertex_stream_mask(NvShader *vs) {
    if(!vs || !vs->rf.route_b)return UINT16_MAX;
    uint16_t mask=0;
    for(int i=0;i<vs->rf.nv;i++) {
        unsigned a=vs->rf.v[i].nvn;
        if(a<16 && a<(unsigned)S.nattr && vfmt(S.attr[a].fmt)!=MTLVertexFormatInvalid && S.attr[a].stream<16)
            mask|=(uint16_t)(1u<<S.attr[a].stream);
    }
    return mask;
}
static void bind_vertex_streams(id<MTL4RenderCommandEncoder> en, NvShader *vs) {
    uint16_t mask=vertex_streams_on()?vertex_stream_mask(vs):UINT16_MAX;
    uint64_t bound=0,omitted=0;
    if(g_vertex_streams_negative)mask&=~(1u<<5);
    for(unsigned stream=0;stream<16;stream++) {
        if(!S.vbuf[stream].address)continue;
        if(!(mask&(1u<<stream))){omitted++;continue;}
        void *mb;uint64_t mo,gpu;
        if(mtl_gpu_addr(S.vbuf[stream].address,1,&mb,&mo,&gpu)) {
            enc_r_vbuf(en,16+stream,gpu);use_rec("buf",S.vbuf[stream].address,S.vbuf[stream].size,0,NULL);bound++;
        }
    }
    g_vertex_streams_bound+=bound;g_vertex_streams_omitted+=omitted;
    static id<MTLBuffer> zero;static uint64_t zero_gpu;
    if(!zero){zero=[g_dev newBufferWithLength:64 options:MTLResourceStorageModeShared];res_add(zero);zero_gpu=zero.gpuAddress;}
    enc_r_vbuf(en,30,zero_gpu);
}
static void mfxup_queue_residency(void) {   // the upscaler's own output textures, once they exist
    static int added; if (added) return;
    id<MTLResidencySet> r = mfxup_residency(); if (r) { [g_q4 addResidencySet:r]; added = 1; }
}
/* TOTK_MFX_UPSCALE, docked (Mac, user's choice 2026-10-16 night: keep the native render size, let MetalFX do the upscale):
   after tone mapping and FXAA at 1600x900 the game's own upscale a3a570eb (FSR 1 EASU-style: twelve texel reads per pixel)
   draws the picture into the 1920x1080 canvas, before the UI. MetalFX writes the canvas instead and the draw is not
   encoded. The shader's constants (fragment cbuf slot 0 = Maxwell c3; FSR con0: c3[0].xy = viewport / output size,
   c3[0].zw = 0.5 * c3[0].xy - 0.5; c3[4].xy = output size as integers) give the valid region of the picture, so a smaller
   region (dynamic resolution) is upscaled as the game would. Only when the draw would overwrite the whole canvas (no
   blending, full write mask, viewport and scissor over the whole target, no depth / stencil test) and the region starts
   at the origin; anything else keeps the game's own upscale. Perceptual colour processing: the canvas holds display
   values (dd3d1e0e copies it to the swapchain unchanged). Returns 1 when the draw was replaced.
   This is controlled only by the manual picture mode, not by the window's output resolution. */
static uint64_t g_mfxup_dock_n, g_mfxup_dock_no[6];
static int mfxup_dock(__unsafe_unretained NvShader *fs) {
    /* 1.4.3's EASU variant normalizes samples by 1/3.5, then multiplies its
     * clamped result by 3.5. Those operations cancel for flat colors; adding
     * another gain would brighten the picture incorrectly. Its source retains
     * highlights above 1, so request HDR processing. Exact-shader GPU probe:
     * scripts/upscale_color_probe.m (RGB/HDR + deliberately missing gain).
     * Both variants use the same con0 and integer output-size layout below. */
    int hdr = !strncmp(fs->key, "ac58fc18eda5f3878b16074dc28cd441644a66710f1b73954d39e57ac357561b-", 65);
    if (!hdr && strncmp(fs->key, "a3a570eb159202f14bc86479b0d3a012ecd830b8550505536eb70bf82372e4bb-", 65)) return 0;
    int color_mode = hdr ? MFXUP_HDR : MFXUP_PERCEPTUAL;
    const view_t *cv = &R.rt.cview[0]; int hv = R.rt.has_cview[0];
    int why = 0;
    if (R.rt.n < 1 || !R.rt.color[0]) why = 1;
    for (int i = 1; !why && i < R.rt.n; i++) if (R.rt.color[i]) why = 1;
    if (!why && ((S.color.blend_en & 1) || (S.chmask.mask & 0xf) != 0xf || S.ms.a2c || (R.rt.depth && (S.ds.dt || S.ds.st)))) why = 2;
    if (!why && hv && ((cv->set_levels && cv->base_level) || (cv->set_layers && cv->min_layer))) why = 1;   // the canvas's own level 0 / layer 0 (its view names layer 0: still the whole texture)
    int tid = (int)(S.tex[1][0] & 0xfffff);
    id<MTLTexture> src = !why && tid > 0 ? (__bridge id<MTLTexture>)tex_fast(tid) : nil, dst = src ? attachment_texture(R.rt.color[0], cv, hv) : nil;
    if (!why && (!src || !dst || src.textureType != MTLTextureType2D || dst.textureType != MTLTextureType2D || src.width >= dst.width || src.height >= dst.height)) why = 3;
    if (!why && ((S.has_vp && !(S.vp[0][0] == 0 && S.vp[0][1] == 0 && S.vp[0][2] == (float)dst.width && S.vp[0][3] == (float)dst.height)) ||
                 !scissor_full(R.rt.color[0], cv, hv))) why = 4;
    NSUInteger cw = 0, ch = 0; float k[20] = {0}; uint32_t ow = 0, oh = 0;
    if (!why) {
        const p_bind_buf *u = &S.ubo[1][0];
        if (!u->addr || u->size < sizeof k) why = 5;
        else {
            pend_materialize(u->addr, (uint8_t *)k, sizeof k, 0);   // the bytes this draw sees (the submit-time copy plus pending inline updates)
            memcpy(&ow, &k[16], 4); memcpy(&oh, &k[17], 4);
            double fw = (double)k[0] * ow, fh = (double)k[1] * oh;
            cw = (NSUInteger)llround(fw); ch = (NSUInteger)llround(fh);
            if (ow != dst.width || oh != dst.height || !cw || !ch || cw > src.width || ch > src.height || fabs(fw - (double)cw) > 0.02 || fabs(fh - (double)ch) > 0.02 ||
                fabsf(k[2] - (0.5f * k[0] - 0.5f)) > 1e-4f || fabsf(k[3] - (0.5f * k[1] - 0.5f)) > 1e-4f) why = 5;
        }
    }
    { static int once; if (!once && (!why || why == 5)) { once = 1;
        MLOG("[mfx-up] 主机模式放大点 %.8s（%s）：源 %lux%lu（格式 %lu）→ 画布 %lux%lu（格式 %lu）；常量 con0 %g %g %g %g、输出 %ux%u → 有效区域 %lux%lu%s",
             fs->key, hdr ? "HDR" : "perceptual", (unsigned long)src.width, (unsigned long)src.height, (unsigned long)src.pixelFormat, (unsigned long)dst.width, (unsigned long)dst.height, (unsigned long)dst.pixelFormat,
             k[0], k[1], k[2], k[3], ow, oh, (unsigned long)cw, (unsigned long)ch, why ? "：常量不符，保留游戏自己的放大" : ""); } }
    if (why) {
        static int told[6]; if (!told[why]++)   // the first refusal of each kind, with what was seen
            MLOG("[mfx-up] 主机模式放大点 %.8s 不替换（原因 %d：1 目标 / 2 混合掩码深度 / 3 贴图 / 4 视口剪裁 / 5 常量）：附件 %d 个，视图 %d（级 %d:%u 层 %d:%u），"
                 "源 %s %lux%lu 类型 %lu，画布 %s %lux%lu 类型 %lu 用途 %#lx；混合 %#x 掩码 %#x a2c %d 深度 %d/%d；视口 %d %g,%g %gx%g 剪裁 %d %d,%d %dx%d；常量 %g %g %g %g 输出 %ux%u",
                 fs->key, why, R.rt.n, hv, hv ? cv->set_levels : 0, hv ? cv->base_level : 0, hv ? cv->set_layers : 0, hv ? cv->min_layer : 0,
                 src ? "有" : "无", (unsigned long)src.width, (unsigned long)src.height, (unsigned long)src.textureType,
                 dst ? "有" : "无", (unsigned long)dst.width, (unsigned long)dst.height, (unsigned long)dst.textureType, (unsigned long)dst.usage,
                 (unsigned)S.color.blend_en, S.chmask.mask, (int)S.ms.a2c, (int)S.ds.dt, (int)S.ds.st,
                 S.has_vp, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3], R.has_sc, R.sc.x, R.sc.y, R.sc.w, R.sc.h, k[0], k[1], k[2], k[3], ow, oh);
        g_mfxup_dock_no[why]++; return 0; }
    if (!mfxup_prepare(MFXUP_DOCKED, src, dst, dst.width, dst.height, color_mode)) { g_mfxup_dock_no[0]++; return 0; }   // scaler still being built (background): the game's own upscale this frame
    rp_end(); blit_close(); rs_sync();   // MetalFX encodes on the command buffer itself: the encode thread must be idle first
    pt_mark(g_cb, 'X', nil);   // TOTK_PASS_GPU: the MetalFX upscale as its own entry
    if (!mfxup_encode_ex(MFXUP_DOCKED, g_cb, src, cw, ch, dst, dst.width, dst.height, color_mode)) { g_mfxup_dock_no[0]++; return 0; }
    mfxup_queue_residency();
    R.clear_c[0] = 0;   // MetalFX wrote every pixel of the canvas: a clear still pending for it would wipe the picture when the UI opens the pass
    g_mfxup_dock_n++; atomic_store(&g_mfxup_last, atomic_load(&g_cur_frame));
    return 1;
}
/* TOTK_MFX_UPSCALE: the game's compose shader 0ac9a549 (tone mapping and bloom into the canvas) samples the scene at
   texture slot 0; when that scene is smaller than the canvas (TOTK_RENDER_RES below the canvas size), upscale it with
   MetalFX first and let encode_args bind the result. Runs before the draw opens its render encoder. Docked: mfxup_dock.
   Returns 1 when the draw was replaced (not to be encoded). */
static int mfxup_draw(NvProg *pr) {
    g_mfxup_src = nil; g_mfxup_out = nil; g_mfxup_sh = nil;
    if (!pr || !mfxup_enabled()) return 0;
    __unsafe_unretained NvShader *fs = pr->sh[1];
    if (!fs) return 0;
    if (mfxup_dock(fs)) {
        static unsigned n; if (++n % 1800 == 1) { char b[200]; mfxup_stats(MFXUP_DOCKED, b, sizeof b);
            MLOG("[mfx-up] 至帧 %llu：%s；替换游戏放大 %llu 次，保留 %llu 次（建不出或在后台建 %llu、目标不符 %llu、混合/掩码/深度 %llu、尺寸 %llu、视口/剪裁 %llu、常量 %llu）",
                 (unsigned long long)atomic_load(&g_cur_frame), b, (unsigned long long)g_mfxup_dock_n,
                 (unsigned long long)(g_mfxup_dock_no[0] + g_mfxup_dock_no[1] + g_mfxup_dock_no[2] + g_mfxup_dock_no[3] + g_mfxup_dock_no[4] + g_mfxup_dock_no[5]),
                 (unsigned long long)g_mfxup_dock_no[0], (unsigned long long)g_mfxup_dock_no[1], (unsigned long long)g_mfxup_dock_no[2],
                 (unsigned long long)g_mfxup_dock_no[3], (unsigned long long)g_mfxup_dock_no[4], (unsigned long long)g_mfxup_dock_no[5]); }
        return 1;
    }
    if (strncmp(fs->key, "0ac9a54938974396", 16) || !R.rt.n || !R.rt.color[0]) return 0;
    int tid = (int)(S.tex[1][0] & 0xfffff); if (tid <= 0) return 0;
    id<MTLTexture> src = (__bridge id<MTLTexture>)tex_fast(tid), dst = tex_mtl(R.rt.color[0]);
    if (!src || !dst || src.width >= dst.width || src.height >= dst.height) return 0;
    rp_end(); blit_close(); rs_sync();
    pt_mark(g_cb, 'X', nil);   // TOTK_PASS_GPU: the MetalFX upscale as its own entry
    id<MTLTexture> out = mfxup_encode(g_cb, src, dst.width, dst.height);
    if (!out) return 0;
    mfxup_queue_residency();
    g_mfxup_src = src; g_mfxup_out = out; g_mfxup_sh = fs; atomic_store(&g_mfxup_last, atomic_load(&g_cur_frame));
    static unsigned n; if (++n % 1800 == 1) { char b[200]; mfxup_stats(MFXUP_SCENE, b, sizeof b); MLOG("[mfx-up] 至帧 %llu：%s", (unsigned long long)atomic_load(&g_cur_frame), b); }
    return 0;
}
#include "nvn_cube_group.h"
static void draw_inner(const p_draw *dw, const p_draw_idx *di) {
    if (!S.prog) { RED_ONCE(18, "绘制前没有绑定程序"); g_draw_skip[0]++; return; }
    __unsafe_unretained NvProg *pr = prog_get(S.prog);
    if (pr && pr->missing) { g_draw_skip[15]++; return; }   // TOTK (iOS): a stage is not translated yet
    if (g_nupd) upd_guard_draw(pr, di);   // TOTK: before any encoder or texture preparation of this draw
    { int fit = pr ? prog_fit(pr) : 0; if (fit < 0) { g_draw_skip[15]++; return; } if (fit) g_pso_dirty = 1; }
    if (skv_on()) { g_skv_draws++; if (skv_check((__bridge const void *)pr, dw, di)) return; }   // measurement experiment: skip draws invisible in the last frames
    geometry_dump(pr, dw, di);
    if (di && di->itype == 0 && di->count && pend_hits(di->ibuf, (uint64_t)di->count)) widen_indices_gpu(di);
    /* Merge/skip checks can resolve sampling views too; prepare before any of those checks. */
    if (!S.render_disabled && !S.raster_discard) { prepare_draw_textures(pr); if (mfxup_draw(pr)) return; }   // docked: MetalFX wrote the canvas instead of this draw
    if (g_ruler_truth == 0) {   // 尺子负对照（DM_NVN_RULER_NEG=<片元键>）：被测模式下丢掉该着色器的绘制，尺子必须报不等价 | Ruler negative control (DM_NVN_RULER_NEG=<fragment key>): in the mode under test drop this shader's draws, and the ruler must report non-equivalence
        static const char *neg; static int ng = -1; if (ng < 0) { neg = getenv("DM_NVN_RULER_NEG"); ng = neg && *neg; }
        if (ng && pr) for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (sh && sh->stage == 1 && !strncmp(sh->key, neg, strlen(neg))) { static int once; if (!once++) MLOG("DM_NVN_RULER_NEG：被测模式丢弃着色器 %s 的绘制", sh->key); return; } }
    }
    if (zskip_can_draw(pr)) {   // 零源跳过：输出恒为 0、写 0 盖 0，这次绘制不改变任何结果（见 zskip_can_draw） | Zero-source skip: the output is always 0, writing 0 over 0, so this draw changes no result (see zskip_can_draw)
        g_zskip_count++; { static int once; if (!once++) MLOG("零源跳过：首例 通道 %llu", g_stat_pass + 1); }
        PD("N %llu 零源跳过", g_stat_pass + 1); g_draw_skip[1]++; return; }
    if (!g_defer_replaying) {
        if (g_defer.active && g_defer.in_p && !g_defer.at_q) { if (defer_take(dw, di)) { g_draw_skip[2]++; return; } }
    }
    /* Diagnostic: sample the HDR texture immediately before tone mapping.
       The same allocation is reused by FXAA later, so reading it at present
       cannot establish which pass first introduced a false-colour frame. */
    { static int on = -1; static FILE *f; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
      if (on < 0) { const char *path = getenv("DM_NVN_TONEIN_LOG"); on = path && (f = fopen(path, "w")) != NULL; }
      if (on && pr && pr->sh[1] && !strncmp(pr->sh[1]->key, "0ac9a54938974396", 16)) {
          int tid = (int)(S.tex[1][0] & 0xfffff);
          int btid = (int)(S.tex[1][1] & 0xfffff);
          id<MTLTexture> src = tid > 0 && tid < g_ntreg && g_treg[tid].ok ? tex_mtl(&g_treg[tid].t) : nil;
          id<MTLTexture> bloom = btid > 0 && btid < g_ntreg && g_treg[btid].ok ? tex_mtl(&g_treg[btid].t) : nil;
          if (src && src.pixelFormat == MTLPixelFormatRG11B10Float && src.width >= 64 && src.height >= 64) {
              BOOL have_bloom = bloom && bloom.pixelFormat == MTLPixelFormatRG11B10Float && bloom.mipmapLevelCount > 1 && bloom.width >= 128 && bloom.height >= 128;
              id<MTLBuffer> rb; size_t ro = arena_alloc(64 * 64 * 4 * (have_bloom ? 2 : 1), &rb);
              id<MTL4ComputeCommandEncoder> be = blit();
              enc_b_copy_texture_buffer(be, src, 0, 0, MTLOriginMake((src.width - 64) / 2, (src.height - 64) / 2, 0), MTLSizeMake(64, 64, 1), rb, ro, 64 * 4, 64 * 64 * 4);
              if (have_bloom) enc_b_copy_texture_buffer(be, bloom, 0, 1, MTLOriginMake((bloom.width / 2 - 64) / 2, (bloom.height / 2 - 64) / 2, 0), MTLSizeMake(64, 64, 1), rb, ro + 64 * 64 * 4, 64 * 4, 64 * 64 * 4);
              enc_b_end(g_blit); g_blit = nil;
              uint64_t fr = atomic_load(&g_cur_frame);
              [g_commit_blocks addObject:^{ const uint32_t *v = (const uint32_t *)((const uint8_t *)rb.contents + ro); double avg[6] = {0};
                  for (unsigned part = 0; part < (have_bloom ? 2u : 1u); part++) {
                      uint64_t r = 0, g = 0, b = 0; const uint32_t *p = v + part * 4096;
                      for (unsigned i = 0; i < 4096; i++) { r += p[i] & 0x7ff; g += (p[i] >> 11) & 0x7ff; b += (p[i] >> 22) & 0x3ff; }
                      avg[part * 3] = r / 4096.0; avg[part * 3 + 1] = g / 4096.0; avg[part * 3 + 2] = b / 4096.0; }
                  pthread_mutex_lock(&mu); fprintf(f, "%llu\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", (unsigned long long)fr, avg[0], avg[1], avg[2], avg[3], avg[4], avg[5]); fflush(f);
                  { static unsigned saved; const char *dir = getenv("DM_NVN_TONEIN_DUMP");
                    if (dir && avg[0] > 1100 && saved++ < 16) { char path[1024]; snprintf(path, sizeof path, "%s/F%llu.raw", dir, (unsigned long long)fr);
                        FILE *out = fopen(path, "wb"); if (out) { fprintf(out, "RG11B10F 64 64 %d\n", have_bloom ? 2 : 1);
                            fwrite(v, 4, 4096 * (have_bloom ? 2 : 1), out); fclose(out); } } }
                  pthread_mutex_unlock(&mu); }];
          }
      } }
    // 什么都画不出的绘制（两面全剔、剪裁为空）在开通道之前就返回：否则通道已开，TBDR 上整张附件白读白写
    // （Mac 选角页：3 个通道只因此类绘制而开，全部读入、零绘制）
    // 量上界用的诊断开关 DM_NVN_SKIP_FS=<片元键>[,<片元键>…]（缺省无）：丢掉这些片元着色器的全部绘制（画面会错，只用来在真机 ABA 里量「这类绘制总共值多少 GPU 时间」）
    // Draws that cannot draw anything (both faces culled, empty scissor) return before the render pass is opened: otherwise the render pass is already open and on TBDR the whole attachment is loaded and stored for nothing
    // (Mac character select page: 3 render passes are opened only because of such draws, all loaded, zero draws)
    // Diagnostic switch for measuring an upper bound DM_NVN_SKIP_FS=<fragment key>[,<fragment key>...] (none by default): drops all draws of these fragment shaders (the picture will be wrong; only used in real-device ABA to measure "how much GPU time this class of draws is worth in total")
    { uint64_t skip = atomic_load(&g_diag_skip_fs);
      if (skip && pr->sh[1] && !g_defer_replaying) { char prefix[17]; snprintf(prefix, sizeof prefix, "%016llx", (unsigned long long)skip);
          if (!strncmp(pr->sh[1]->key, prefix, 16)) return; }
      static char sk[16][20]; static int nsk = -1;
      if (nsk < 0) { nsk = 0; for (const char *q = getenv("DM_NVN_SKIP_FS"); q && *q && nsk < 16; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) { size_t n = strcspn(q, ","); if (n > 16) n = 16; memcpy(sk[nsk], q, n); sk[nsk++][n] = 0; }
                     if (nsk) MLOG("DM_NVN_SKIP_FS：丢弃 %d 个片元着色器的绘制（诊断，画面不对）", nsk); }
      if (g_ub_n > 0 && pr->sh[1] && !g_defer_replaying && ub_skip(pr->sh[1]->key)) return;   // 同进程交替测量：本相丢这一类 | In-process alternating measurement: this phase drops this class
      if (nsk && pr->sh[1] && !g_defer_replaying) for (int i = 0; i < nsk; i++) if (!strncmp(pr->sh[1]->key, sk[i], strlen(sk[i]))) return; }
    // 同类诊断 DM_NVN_SKIP_SMALL=<宽>：丢掉目标宽度（按所挂的级）≤ 该值的全部绘制——量「极小通道整批丢掉」在并行流水下值多少（每通道固定开销）
    // Similar diagnostic DM_NVN_SKIP_SMALL=<width>: drop all draws whose target width (by the attached level) is <= this value -- measures how much "dropping tiny render passes in bulk" is worth under the parallel pipeline (fixed per-pass overhead)
    { static int small = -1; if (small < 0) { const char *e = getenv("DM_NVN_SKIP_SMALL"); small = e ? atoi(e) : 0; if (small) MLOG("DM_NVN_SKIP_SMALL：丢弃目标宽度 ≤ %d 的绘制（诊断，画面不对）", small); }
      if (small > 0 && R.has_rt && !g_defer_replaying) { int k = R.rt.n && R.rt.color[0] ? 0 : R.rt.depth ? 8 : -1;
        if (k >= 0) { uint64_t s_; int lv, sl; rt_sub(&R.rt, k, &s_, &lv, &sl); const tex_t *t = k == 8 ? R.rt.depth : R.rt.color[0]; if ((int)MAX(t->d.width >> lv, 1u) <= small) return; } } }
    // 试验开关 DM_NVN_NOFXAA=1（缺省关，去留按视觉尺子 + 实机看图定）：跳过 FXAA 那次全屏绘制，后面的锐化 / 放大改读 FXAA 的输入（色调映射结果）
    // Experimental switch DM_NVN_NOFXAA=1 (off by default; keep or drop decided by the visual ruler + looking at the picture on a real machine): skips the FXAA full-screen draw, and the later sharpening / upscale reads FXAA's input instead (the tone mapping result)
    { static int nofxaa = -1; static uint64_t fx_from, fx_to; if (nofxaa < 0) { const char *e = getenv("DM_NVN_NOFXAA"); nofxaa = e && !strcmp(e, "1"); if (nofxaa) MLOG("DM_NVN_NOFXAA=1：跳过 FXAA"); }
      if (nofxaa && tbdr_on() && !g_defer_replaying && pr->sh[1]) { NvShader *f = pr->sh[1];
        if (!strncmp(f->key, "ebcfcd1fa0a77dbc", 16) && !g_enc && R.rt.n == 1 && R.rt.color[0] && !R.clear_c[0]) {
            uint64_t h = 0; for (int i = 0; i < f->rf.nb && !h; i++) if (f->rf.b[i].kind == NVN_RB_TEX && f->rf.b[i].reg < 128) h = S.tex[1][f->rf.b[i].reg];
            if (h) { fx_from = R.rt.color[0]->serial; fx_to = h; PD("N %llu 跳过 FXAA：ser=%llu 的读者改读 ser=%llu", g_stat_pass, (unsigned long long)fx_from, (unsigned long long)h_ser(h)); return; } }
        else if (fx_from && !strncmp(f->key, "52aac1f2306b8fc5", 16)) { for (int r = 0; r < 128; r++) if (S.tex[1][r] && h_ser(S.tex[1][r]) == fx_from) S.tex[1][r] = fx_to; fx_from = 0; } } }
    { static uint64_t bloomq_frame; if (bloomq_frame != atomic_load(&g_cur_frame)) { bloomq_frame = atomic_load(&g_cur_frame); bloomq_reset(); }
      if (bloomq_enabled() && tbdr_on() && !g_defer_replaying && pr->sh[1]) {
        NvShader *f = pr->sh[1]; uint64_t tgt = 0; int lv, sl;
        if (!strncmp(f->key, "2938a5b71b8e21b2", 16)) {
            if (g_bloomq.state) bloomq_reset();
            if (bloomq_target(&tgt, &lv, &sl) && R.rt.color[0]) {
                uint64_t h = 0; const tex_t *st = NULL; int cbuf = 0;
                for (int i = 0; i < f->rf.nb; i++) { nvn_rbind *b = &f->rf.b[i];
                    if (b->kind == NVN_RB_TEX && b->reg < 128 && !h) { h = S.tex[1][b->reg]; int tid = (int)(h & 0xfffff); if (tid > 0 && tid < g_ntreg && g_treg[tid].ok) st = &g_treg[tid].t; }
                    if (b->kind == NVN_RB_SAMPT && b->reg < 128) g_bloomq.samp_h = S.tex[1][b->reg] >> 20;
                    if (b->kind == NVN_RB_SAMP && b->reg < 128) g_bloomq.samp_h = S.samp[1][b->reg] >> 20;
                    if (b->kind == NVN_RB_CBUF && !cbuf) cbuf = ubo_float(1, (uint16_t)(b->reg << 8), &g_bloomq.threshold);   // c7[0].x
                }
                if (h && st && cbuf && f->rf.c1_len >= sizeof(float) * 2) {
                    float c1[2]; memcpy(c1, f->rf.c1, sizeof c1); g_bloomq.s_tex = st; g_bloomq.s_h = h; g_bloomq.s_ser = st->serial; g_bloomq.b0_ser = tgt; g_bloomq.coeff_g = c1[0]; g_bloomq.coeff_b = c1[1]; g_bloomq.state = 1;
                    { static int once; if (!once++) MLOG("泛光 1/4 起点：生效（S ser=%llu B0 ser=%llu）", (unsigned long long)g_bloomq.s_ser, (unsigned long long)g_bloomq.b0_ser); } PD("A %llu 泛光 1/4 起点 S=%llu B0=%llu", g_stat_pass, (unsigned long long)g_bloomq.s_ser, (unsigned long long)g_bloomq.b0_ser); R.clear_c[0] = 0;   // B0 整级不要了：挂起的清除也丢掉，否则会为它单开一个只清除的通道 | The whole B0 level is no longer needed: drop the pending clear too, otherwise a clear-only render pass would be opened just for it
                    return;
                }
            }
        } else if (g_bloomq.state == 1 && !strncmp(f->key, "019ce2f82edbda84", 16)) {
            uint64_t src = 0; for (int i = 0; i < f->rf.nb; i++) if (f->rf.b[i].kind == NVN_RB_TEX && f->rf.b[i].reg < 128) { src = h_ser(S.tex[1][f->rf.b[i].reg]); break; }
            if (src == g_bloomq.b0_ser && bloomq_target(&tgt, &lv, &sl) && bloomq_capture()) { R.clear_c[0] = 0; g_bloomq.b1_ser = tgt; g_bloomq.state = 2; PD("B %llu 泛光 1/4 原生截取 B0=%llu B1=%llu", g_stat_pass, (unsigned long long)g_bloomq.b0_ser, (unsigned long long)g_bloomq.b1_ser); return; }
            RED_ONCE(62, "泛光 1/4 起点：截取已跳过但原生截取没画成，本帧缺泛光"); bloomq_reset();
        } else if (g_bloomq.state == 2 && !strncmp(f->key, "019ce2f82edbda84", 16)) {
            for (int i = 0; i < f->rf.nb; i++) if (f->rf.b[i].kind == NVN_RB_TEX && f->rf.b[i].reg < 128 && h_ser(S.tex[1][f->rf.b[i].reg]) == g_bloomq.b1_ser) { g_bloomq.b1_h = S.tex[1][f->rf.b[i].reg]; break; }
        } else if (g_bloomq.state == 2 && !strncmp(f->key, "cc264583f2000c81", 16)) {
            if (bloomq_target(&tgt, &lv, &sl) && tgt == g_bloomq.b0_ser) { g_bloomq.state = 3; PD("C %llu 泛光 1/4 跳过 B0 上采样", g_stat_pass); return; }
        } else if (g_bloomq.state == 3 && !strncmp(f->key, "2e2d6700a2569795", 16)) {
            if (g_bloomq.b1_h) { for (int r = 0; r < 128; r++) if (S.tex[1][r] && h_ser(S.tex[1][r]) == g_bloomq.b0_ser) S.tex[1][r] = g_bloomq.b1_h; PD("D %llu 泛光 1/4 叠加改读 B1", g_stat_pass); g_bloomq_last_b0 = g_bloomq.b0_ser; g_bloomq_last_b1h = g_bloomq.b1_h; g_bloomq_last_b1s = g_bloomq.b1_ser; bloomq_reset(); }
            else { RED_ONCE(63, "泛光 1/4 状态缺少 B1 句柄"); bloomq_reset(); }
        } else if (!g_bloomq.state && g_bloomq_last_b1h && h_ser(g_bloomq_last_b1h) == g_bloomq_last_b1s && !strncmp(f->key, "2e2d6700a2569795", 16)) {   // B1 句柄仍指同一张贴图（换分辨率重建后不认） | The B1 handle still points to the same texture (not recognized after a rebuild on resolution change)
            // 本帧泛光链没跑（运行时外推的外推帧跳过了它：输入 469 区域外是上一帧 LDR，不能跑）就到了泛光叠加：照最近一次改读 B1，叠上一帧的泛光。
            // 不改读就读 B0——开了泛光 1/4 的话 B0 的上采样一直被跳过、从没写过，读到的是未初始化显存（M4 Pro 上区域里的单位整个发绿，别处多半是 0、缺泛光）
            // The bloom overlay is reached without the bloom chain having run this frame (the extrapolated frame of runtime extrapolation skipped it: outside the region the input 469 is the previous frame's LDR, so it cannot run): redirect the read to B1 as last time, overlaying the previous frame's bloom.
            // Without the redirect it would read B0 -- with bloom 1/4 on, B0's upsample is always skipped and never written, so what is read is uninitialized video memory (on M4 Pro the units in the region turn entirely green; elsewhere it is mostly 0, i.e. missing bloom)
            int hit = 0; for (int r = 0; r < 128; r++) if (S.tex[1][r] && h_ser(S.tex[1][r]) == g_bloomq_last_b0) { S.tex[1][r] = g_bloomq_last_b1h; hit = 1; }
            if (hit) { PD("D %llu 泛光链没跑：叠加沿用上一帧 B1", g_stat_pass); static uint64_t n; if (n++ % 600 == 0) MLOG("泛光 1/4：泛光链没跑的帧叠加沿用上一帧 B1（第 %llu 次）", (unsigned long long)n); }
        } else if (g_bloomq.state) { RED_ONCE(61, "泛光 1/4 起点：链中途出现别的绘制，本帧泛光不完整"); bloomq_reset(); }
      }
    }
    { static int dl = -1, nd; if (dl < 0) dl = getenv("TOTK_DRAW_LOG") != NULL;
      if (dl && nd < 40) { nd++;
        MLOG("[draw] #%d vs=%s ps=%s rt0=%p %ux%u fmt=%#x", nd, pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[1] ? pr->sh[1]->key : "-", (void *)(R.has_rt ? R.rt.color[0] : NULL),
             R.has_rt && R.rt.color[0] ? R.rt.color[0]->d.width : 0, R.has_rt && R.rt.color[0] ? R.rt.color[0]->d.height : 0, R.has_rt && R.rt.color[0] ? R.rt.color[0]->d.format : 0);
        for (int r = 0; r < 4; r++) if (S.tex[1][r]) { int tid = (int)(S.tex[1][r] & 0xfffff); const treg_t *t = tid < g_ntreg && g_treg[tid].ok ? &g_treg[tid] : NULL;
            MLOG("[draw]   ps slot %d handle %#llx tid %d %s %ux%u fmt=%#x tgt=%d layout=%d mem=%p ser=%llu", r, (unsigned long long)S.tex[1][r], tid, t ? "ok" : "UNREGISTERED",
                 t ? t->t.d.width : 0, t ? t->t.d.height : 0, t ? t->t.d.format : 0, t ? t->t.d.target : 0, t ? t->t.has_layout : 0, t ? (void *)t->t.mem : NULL, t ? (unsigned long long)t->t.serial : 0ull); } } }
    if (S.color.logic_op != NVN_LOGIC_OP_COPY || (S.color.alpha_func && S.color.alpha_func != NVN_ALPHA_FUNC_ALWAYS))
        RED_ONCE(70, "TOTK: logic op %u / alpha test %u not implemented on Metal (drawn without it)", S.color.logic_op, S.color.alpha_func);
    if (S.render_disabled || S.raster_discard) { g_draw_skip[S.render_disabled ? 3 : 4]++; return; }   // TOTK: SetRenderEnable(false) / SetRasterizerDiscard(true)
    if (S.poly.cull == NVN_FACE_FRONT_AND_BACK) { g_draw_skip[5]++; return; }
    if (exp_skip_fs(pr)) return;   // TOTK_EXP_SKIP_FS (measurement only)
    // 续用的通道可能是少挂深度 / 模板开的（开法 2 / 3 / 4）：这次绘制要的平面没挂，就在它之前切开（重开时按新前瞻挂）
    // The continued render pass may have been opened with fewer depth / stencil attachments (open modes 2 / 3 / 4): if the plane this draw needs is not attached, split before it (on reopen, attach per the new lookahead)
    if (g_rt_merged && g_enc_mode != 0 && (((S.ds.dt || S.ds.dw) && !g_pass.hd) || (S.ds.st && !g_pass.hs))) {
        static int once2; if (!once2++) MLOG("同目标续用：续用通道内绘制要的深度 / 模板平面没挂，在此切开（首例 通道 %llu）", g_stat_pass);
        PD("M %llu 续用通道内绘制要的深度 / 模板平面没挂，在此切开", g_stat_pass); rp_end(); }
    if (g_rt_range_active && (!rt_range_pr_ro(pr) || !S.ds.dt || S.ds.dw || S.ds.st)) rp_end();   // 原encoder边界的外部写可见性不删 | The external write visibility at the original encoder boundary is not removed
    if (g_rt_merged && draw_samples_rt(pr)) { static int once; if (!once++) MLOG("同目标续用：续用通道内绘制要采样附件，在此切开（首例 通道 %llu）", g_stat_pass);
        PD("M %llu 续用通道内绘制要采样附件，在此切开", g_stat_pass); rp_end(); }
    static id<MTLFunction> fz_fn;   // 合并通道后一个通道本次绘制的特化片元函数（nil = 用原版本） | Specialized fragment function for this draw of the later pass in a merged render pass (nil = use the original version)
    if (g_fz.active && g_enc && g_fz.phase == 2 && !fz_p2_draw(pr, dw, di, g_fz.lmap2, &fz_fn)) {   // 证不出：结束合并通道（前一个通道的结果照常写回），按后一个通道的原目标重开 | Cannot be proven: end the merged render pass (the earlier pass's result is stored as usual) and reopen with the later pass's original targets
        static int once3; if (!once3++) MLOG("可读 tile 通道合并：后一个通道的绘制证不出，退回分开（首例 通道 %llu）", g_stat_pass);
        g_fz_fallback++; rp_end(); }
    if (R.has_sc && !g_enc) {
        NSUInteger tw0 = 0, th0 = 0; rt_dims(&tw0, &th0);
        long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y < 0 ? 0 : R.sc.y, x1 = (long)R.sc.x + R.sc.w, y1 = (long)R.sc.y + R.sc.h;
        if (x1 > (long)tw0) x1 = tw0; if (y1 > (long)th0) y1 = th0;
        if (x1 <= x0 || y1 <= y0) { g_draw_skip[6]++; return; }
    }

#ifdef NVN_RT_RANGE_TEST
    if (!g_enc) rr_begin();
#endif
    int defer_q_draw = !g_enc && !g_defer_replaying && g_defer_natq > 0;   // Q 开通道的这次绘制：通道照常经 hoist_try / fuse_try 开好后，先回放 P 再画本次 | The draw that opens Q's render pass: after the render pass is opened as usual through hoist_try / fuse_try, replay P first and then draw this one
    if (g_frz_state) goto frz_plain;   // 世界冻结的完整帧 / 冻结帧：不前提、不推迟、不合并（跳通道的状态机要看到每一条设目标） | Full frame / frozen frame of world freeze: no hoisting, no deferral, no merging (the pass-skipping state machine must see every set-target)
    if (!g_enc && tbdr_on()) hoist_try();   // 先把夹在本通道与其续用之间的纯清除通道提前做掉（rp_begin_ex 之前，不占本通道） | First hoist and execute the pure-clear render passes sandwiched between this render pass and its continuation (before rp_begin_ex, not occupying this render pass)
    // 推迟判定在 hoist_try 之后：被提前的纯清除通道在前瞻里跳过，P 的同目标续段才认得出来
    // The deferral decision comes after hoist_try: hoisted pure-clear render passes are skipped in the lookahead, and only then can P's same-target continuation segment be recognized
    if (!g_enc && tbdr_on() && !defer_has_in_p() && !g_defer_replaying && !g_fz.active && !g_fz_checking && g_exec_next) { defer_try(dw, di); if (g_defer_cur->active && g_defer_cur->in_p) { g_draw_skip[7]++; return; } }
    if (!g_enc && tbdr_on() && !g_fz_checking) fuse_try();   // 可读 tile 通道合并缺省开；结构 / 本像素证明不成立仍退回普通通道，CHECK 重画时不重入 | Readable tile render pass merge is on by default; when the structure / same-pixel proof does not hold it still falls back to a normal render pass, and it is not re-entered during the CHECK redraw
frz_plain:;
    int need = g_enc || !tbdr_on() ? 3 : pass_ds_need(), mode = defer_q_draw ? 0 : need == 0 ? 2 : need == 2 ? 3 : need == 1 ? 4 : 0;   // 2 都不挂 3 只挂模板 4 只挂深度 | 2 attach neither, 3 attach stencil only, 4 attach depth only
    int skip_d = mode != 0;
    if (!g_enc && !g_cg.replaying && cg_on()) {   // TOTK: cube-face-set pass groups (nvn_cube_group.h)
        if (cg_take(pr, dw, di)) return;
        if (g_cg.ng) cg_flush();   // every other draw runs after the pending groups (order, and reads of their cubes)
    }
    if (!g_enc) { g_fd_dw = dw; g_fd_di = di; }
    if (g_enc && g_lay.active && !lay_draw_ok(pr, dw, di)) {   // TOTK: this draw cannot write its layer: end the layered pass, draw in a plain one
        PD("L %llu 分层通道内的绘制不能写层号，在此切开", g_stat_pass); rp_end(); g_lay.refuse = 1; g_lay.refused++; }
    if (!g_mfx.scene_done && g_mfx.canvas && R.rt.n > 0 && R.rt.color[0] && R.rt.color[0]->serial == g_mfx.canvas && pr && pr->sh[0] && mfx_ui_vs(pr->sh[0]->key) && mfx_enabled())
        mfx_snapshot_scene();   // TOTK_MFX: first UI draw into the canvas: copy the finished scene first
    id<MTLRenderCommandEncoder> en = rp_begin_ex(mode);
    g_fd_dw = NULL; g_fd_di = NULL;
    if (!en) { g_draw_skip[8]++; return; }
    if (g_pass.mfx || g_mfx.role) {   // TOTK_MFX: every draw into the canvas after the scene copy goes to the UI layer too
        int r = 0;
        if (g_pass.mfx) {
            r = S.color.blend_en & 1 ? MFX_ROLE_UI : MFX_ROLE_UI_OPAQUE;
            /* TOTK_MFX_ROLELOG=<first>-<last frame>: every canvas draw with its role (diagnostic) */
            static long r0 = -2, r1; if (r0 == -2) { const char *e = getenv("TOTK_MFX_ROLELOG"); r0 = e ? atol(e) : -1; r1 = e && strchr(e, '-') ? atol(strchr(e, '-') + 1) : r0; }
            long f = (long)atomic_load(&g_cur_frame);
            if (r0 >= 0 && f >= r0 && f <= r1) MLOG("[mfx] role F%ld pass %llu draw role %d vs %s fs %s blend %#x count %u", f, g_stat_pass, r, pr && pr->sh[0] ? pr->sh[0]->key : "-", pr && pr->sh[1] ? pr->sh[1]->key : "-",
                                                     S.color.blend_en, dw ? dw->count : di ? di->count : 0);
        }
        if (r != g_mfx.role) { g_mfx.role = r; g_pso_dirty = 1; }
    }
    if (g_mfx.want_cam) mfx_take_camera();
    if (defer_q_draw) defer_replay();
    if (skip_d) { g_stat_ds_omit++; if (mode == 3) g_stat_s_only++; else if (mode == 4) g_stat_d_only++; }
    // 管线只依赖（程序, 通道目标, 混合 / 颜色 / 写掩码 / 多重采样状态, 顶点布局）：这些没变就沿用上次的
    // The pipeline depends only on (program, render pass targets, blend / color / write mask / multisample state, vertex layout): if these did not change, reuse the previous one
    static id<MTLRenderPipelineState> cur_ps; static NvProg *cur_pr; static uint64_t cur_ps_pass;
    if (g_fz.active && !g_defer_replaying) { g_pso_lmap = g_fz.phase == 1 ? g_fz.lmap1 : g_fz.lmap2; g_pso_fsfn = g_fz.phase == 1 ? nil : fz_fn; g_pso_fz = g_fz.phase == 1 ? (pso_fz_desc){0} : g_fz_desc; g_pso_dirty = 1; }   // 可读 tile 合并：按物理 → 逻辑映射与特化取管线 | Readable tile merge: fetch the pipeline by the physical -> logical mapping and the specialization
    if (g_pso_dirty || pr != cur_pr || cur_ps_pass != g_pass.serial) { cur_ps = pso_get(pr, nil); cur_pr = pr; cur_ps_pass = g_pass.serial; g_pso_dirty = !cur_ps; }
    if (g_pso_lmap) { g_pso_lmap = NULL; g_pso_fsfn = nil; memset(&g_pso_fz, 0, sizeof g_pso_fz); g_pso_dirty = 1; }   // 下一次绘制重新按自身情况取 | The next draw fetches again according to its own situation
    __unsafe_unretained id<MTLRenderPipelineState> ps = cur_ps;
    if (!ps) { g_draw_skip[9]++; static int failed_n; if (failed_n++ < 3) MLOG("[pipeline-draw] rt-depth=%p serial=%llu fmt=%#x pass-depth=%u dt=%u dw=%u fs-depth=%d", R.rt.depth, (unsigned long long)(R.rt.depth ? R.rt.depth->serial : 0), R.rt.depth ? R.rt.depth->d.format : 0, g_pass.dfmt, S.ds.dt, S.ds.dw, pr->sh[1] ? pr->sh[1]->writes_depth : 0); return; }
    if (ps != g_enc_pso) { enc_r_pso(en, ps); g_enc_pso = ps; }
    static id<MTLDepthStencilState> cur_ds; static uint64_t cur_ds_pass;
    if (g_dss_dirty || cur_ds_pass != g_pass.serial) { cur_ds = dss_get(g_pass.hd, g_pass.hs); cur_ds_pass = g_pass.serial; g_dss_dirty = 0; }
    /* TOTK diagnostic TOTK_GPU_SKIP_VC=<n>: skip every draw whose vertex count is n and whose colour mask is 0 (isolates a single depth-only draw). */
    { static long skip_vc = -2; if (skip_vc == -2) { const char *e = getenv("TOTK_GPU_SKIP_VC"); skip_vc = e ? atol(e) : -1; }
      if (skip_vc >= 0 && S.chmask.mask == 0 && (long)(dw ? dw->count : di->count) == skip_vc) { static int said; if (said++ < 3) MLOG("TOTK_GPU_SKIP_VC: skipped a %ld-vertex depth-only draw (pass %llu)", skip_vc, g_stat_pass); return; } }
    /* TOTK diagnostic TOTK_GPU_NODEPTH_FROM=<frame>: from that frame on, no depth test / write for any draw. */
    { static long nodepth_from = -2; static id<MTLDepthStencilState> nodss; if (nodepth_from == -2) { const char *e = getenv("TOTK_GPU_NODEPTH_FROM"); nodepth_from = e ? atol(e) : -1; }
      if (nodepth_from >= 0 && (long)atomic_load(&g_cur_frame) >= nodepth_from) { if (!nodss) { MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new]; dd.depthCompareFunction = MTLCompareFunctionAlways; dd.depthWriteEnabled = NO; nodss = [g_dev newDepthStencilStateWithDescriptor:dd]; } cur_ds = nodss; } }
    if (cur_ds != g_enc_dss) { enc_r_dss(en, cur_ds); g_enc_dss = cur_ds; }
    if (g_rt_ovr && g_pass.hs) enc_r_stencil_ref(en, g_rt_ovr == 3 ? 0x40 : 0, 0); else enc_r_stencil_ref(en, S.st_ref[0], S.st_ref[1]);
    // 视口 / 剪裁（夹到目标尺寸内）
    // Viewport / scissor (clamped within the target size)
    NSUInteger tw = g_pass.tw, th = g_pass.th;
    // 界面矩形不贴画面顶边时（iPad 真全屏：界面按 16:9 摆在画面底部，见 dm/host/dm_uiscreen.c），界面绘制整体下移：投影矩阵加纵向偏移（ui_mat_patch），剪裁跟着下移。
    // 界面绘制 = 4 顶点、2 个属性、顶点 20 字节，独占一趟、帧末合成（量过：启动读取、选角、建角、进局读取、局内、开面板、全屏地图都不混别的绘制）。
    // 矩阵这次读不到（本提交里还有没落地的写）时退回挪视口：位置一样，只是贴顶上提的控件这一次会被视口顶边裁掉。
    // 从顶边起的剪裁（y ≤ 0）不下移顶边：宿主把贴顶的控件（小地图、地名等）上提回真实画面顶部时，它们画在界面矩形上方，不能被裁掉。
    // 例外：不带贴图、正好盖满整个界面矩形的那一块（开面板时的压暗、转场的黑幕）改成盖满整个画面——视口纵向放大 1/h、不下移，否则界面矩形外的 3D 留一条明暗分界。
    // 没人登记入口或界面矩形就是整个画面（iPhone、Mac 缺省）时 vo = so = 0、vs = 1、不改矩阵，下面与原来逐字相同。
    // When the UI rectangle does not touch the top edge of the picture (iPad true full screen: the UI is laid out at 16:9 at the bottom of the picture, see dm/host/dm_uiscreen.c), UI draws are shifted down as a whole: a vertical offset is added to the projection matrix (ui_mat_patch), and the scissor moves down with it.
    // A UI draw = 4 vertices, 2 attributes, 20-byte vertices, in a pass of its own and composited at end of frame (measured: startup loading, character select, character creation, entering-game loading, in-game, open panel, and full-screen map never mix in other draws).
    // When the matrix cannot be read this time (there are still unlanded writes in this submit), fall back to moving the viewport: same position, except that widgets lifted to the top get clipped by the viewport's top edge this one time.
    // A scissor starting from the top edge (y <= 0) does not have its top edge moved down: when the host lifts top-anchored widgets (minimap, area name etc.) back to the real top of the picture, they are drawn above the UI rectangle and must not be clipped.
    // Exception: the untextured block that exactly covers the whole UI rectangle (the dimming when a panel opens, the black curtain of a transition) is changed to cover the whole picture -- the viewport is scaled vertically by 1/h and not moved down, otherwise the 3D outside the UI rectangle is left with a light/dark dividing line.
    // When nobody registered the entry or the UI rectangle is the whole picture (iPhone, Mac default), vo = so = 0, vs = 1, the matrix is not changed, and what follows is word-for-word the same as before.
    long vo = 0, so = 0; double vs = 1, moff = 0;
    if (g_ui_rect_fn && dw && dw->count == 4 && S.nattr == 2 && S.stream[0].stride == 20) { double uy = 0, uh = 1; g_ui_rect_fn(&uy, &uh);
        if (uh > 0 && uh < 1) { if (ui_cover_all(uh)) vs = 1 / uh; else if (uy > 0) { so = lround(uy * th); if (ui_mat_ok()) moff = 2 * uy; else { vo = so; static int n; if (n++ < 4) MLOG("界面下移：投影矩阵这次读不到，退回挪视口（通道 %llu）", g_stat_pass); } } } }
    if (moff != g_ui_moff) { g_ui_moff = moff; args_dirty_stage(0); }
    if (S.has_vp) enc_r_viewport(en, (MTLViewport){S.vp[0][0], S.vp[0][1] + vo, S.vp[0][2], S.vp[0][3] * vs, S.dr[0], S.dr[1]});
    else enc_r_viewport(en, (MTLViewport){0, vo, tw, th * vs, 0, 1});
    MTLScissorRect sc = {0, 0, tw, th};
    if (R.has_sc) {
        long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y <= 0 ? 0 : R.sc.y + so, x1 = (long)R.sc.x + R.sc.w, y1 = vs != 1 ? (long)th : (long)R.sc.y + R.sc.h + so;
        if (x1 > (long)tw) x1 = tw; if (y1 > (long)th) y1 = th;
        if (x1 <= x0 || y1 <= y0) { g_draw_skip[10]++; return; }
        sc = (MTLScissorRect){(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0)};
    }
    enc_r_scissor(en, sc);
    if (S.color.logic_op != NVN_LOGIC_OP_COPY || (S.color.alpha_func && S.color.alpha_func != NVN_ALPHA_FUNC_ALWAYS))
        RED_ONCE(70, "TOTK: logic op %u / alpha test %u not implemented on Metal (drawn without it)", S.color.logic_op, S.color.alpha_func);
    if (S.poly.cull == NVN_FACE_FRONT_AND_BACK) return;
    /* TOTK diagnostic TOTK_GPU_NOCULL_FROM=<frame>: from that frame on, face culling is disabled for every draw. */
    { static long nocull_from = -2; if (nocull_from == -2) { const char *e = getenv("TOTK_GPU_NOCULL_FROM"); nocull_from = e ? atol(e) : -1; }
      if (nocull_from >= 0 && (long)atomic_load(&g_cur_frame) >= nocull_from) enc_r_cull(en, MTLCullModeNone);
      else enc_r_cull(en, S.poly.cull == NVN_FACE_FRONT ? MTLCullModeFront : S.poly.cull == NVN_FACE_BACK ? MTLCullModeBack : MTLCullModeNone); }
    /* TOTK diagnostic TOTK_GPU_WINDING_FLIP=1: invert the NVN -> Metal front-face mapping for every draw. */
    { static int flip = -1; if (flip < 0) { const char *e = getenv("TOTK_GPU_WINDING_FLIP"); flip = e && atoi(e); }
      int ccw = (S.poly.front == NVN_FRONT_FACE_CCW) != (flip != 0);
      /* TOTK diagnostic: TOTK_GPU_HILITE_CULL=0/1/2 (none/front/back) and TOTK_GPU_HILITE_WIND=0/1 (cw/ccw) override the state of the highlighted draws only */
      { static int hc = -2, hw = -2; if (hc == -2) { const char *a = getenv("TOTK_GPU_HILITE_CULL"), *b = getenv("TOTK_GPU_HILITE_WIND"); hc = a ? atoi(a) : -1; hw = b ? atoi(b) : -1; }
        const char *hl = hilite1_env();
        if ((hc >= 0 || hw >= 0) && hl && pr->sh[1] && strstr(hl, [[NSString stringWithUTF8String:pr->sh[1]->key] substringToIndex:8].UTF8String)) {
            if (hc >= 0) enc_r_cull(en, hc == 1 ? MTLCullModeFront : hc == 2 ? MTLCullModeBack : MTLCullModeNone);
            if (hw >= 0) ccw = hw; } }
      enc_r_winding(en, ccw ? MTLWindingCounterClockwise : MTLWindingClockwise);
      /* TOTK diagnostic: for the fragment shaders named in TOTK_GPU_HILITE1, log the encoder state of their first 3 draws */
      static int logged; const char *hl = hilite1_env();
      if (hl && logged < 3 && pr->sh[1] && strstr(hl, [[NSString stringWithUTF8String:pr->sh[1]->key] substringToIndex:8].UTF8String)) { logged++;
          MLOG("HILITE draw: fs %.8s pass %llu front=%d(ccw=%d) cull=%d vp=%g,%g,%g,%g dr=%g,%g sc=%lu,%lu,%lu,%lu defer=%d idx=%s inst=%d count=%d wide=%d", pr->sh[1]->key, g_stat_pass, (int)S.poly.front, ccw, (int)S.poly.cull,
               S.has_vp ? S.vp[0][0] : -1, S.has_vp ? S.vp[0][1] : -1, S.has_vp ? S.vp[0][2] : -1, S.has_vp ? S.vp[0][3] : -1, S.dr[0], S.dr[1], (unsigned long)sc.x, (unsigned long)sc.y, (unsigned long)sc.width, (unsigned long)sc.height,
               (int)g_defer_replaying, di ? (di->itype == 0 ? "u8" : di->itype == 1 ? "u16" : "u32") : "none", dw ? dw->instances : (di ? di->instances : 0), dw ? dw->count : (di ? di->count : 0), di && di->itype == 0);
          { void *mb; uint64_t mo, g0 = 0, g1 = 0; if (S.ubo[0][0].addr && mtl_gpu_addr(S.ubo[0][0].addr, 1, &mb, &mo, &g0)) {} if (S.ssbo[0][0].addr && mtl_gpu_addr(S.ssbo[0][0].addr, 1, &mb, &mo, &g1)) {}
            MLOG("HILITE draw: blend_en=%#x target0 src_rgb=%u dst_rgb=%u eq_rgb=%u src_a=%u dst_a=%u eq_a=%u chmask=%#x a2c=%d", S.color.blend_en, S.blend[0].src_rgb, S.blend[0].dst_rgb, S.blend[0].eq_rgb, S.blend[0].src_a, S.blend[0].dst_a, S.blend[0].eq_a, S.chmask.mask, (int)S.ms.a2c);
            MLOG("HILITE draw: base_vertex=%d base_instance=%d vs ubo0 addr=%#llx gpu=%#llx size=%llu ssbo0 addr=%#llx gpu=%#llx size=%llu", di ? di->base_vertex : -1, di ? di->base_instance : (dw ? dw->base_instance : -1),
                 (unsigned long long)S.ubo[0][0].addr, (unsigned long long)g0, (unsigned long long)S.ubo[0][0].size, (unsigned long long)S.ssbo[0][0].addr, (unsigned long long)g1, (unsigned long long)S.ssbo[0][0].size); } } }
    enc_r_fill(en, S.poly.mode == NVN_POLYGON_MODE_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill);
    if (S.poly.offset_en & 4) enc_r_depth_bias(en, S.po.units, S.po.factor, S.po.clamp); else enc_r_depth_bias(en, 0, 0, 0);
    { static int noclamp = -1; if (noclamp < 0) noclamp = getenv("TOTK_GPU_NOCLAMP") != NULL;   // TOTK diagnostic: TOTK_GPU_NOCLAMP=1 ignores SetDepthClamp (always clip)
      enc_r_clip(en, S.depth_clamp && !noclamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip); }   // TOTK: SetDepthClamp
    enc_r_blend_color(en, S.blend_color[0], S.blend_color[1], S.blend_color[2], S.blend_color[3]);
    bind_vertex_streams((id<MTL4RenderCommandEncoder>)en,pr->sh[0]);
    if (g_lay.active) enc_r_vbuf(en, LAY_BUF_INDEX, lay_gpu(g_lay_probe_neg == 1 ? 0 : g_lay.cur - g_lay.base));   // TOTK: this draw's layer within the layered pass
    if (pr->sh[0]) encode_args(en, pr->sh[0], 0, 1);
    if (pr->sh[1]) encode_args(en, pr->sh[1], 1, 0);
    // 参数绑定在登记表锁内捕获实际资源；资格判断到绑定期间有版本变化，则在任何GPU draw前切开重试。
    // Parameter binding captures the actual resources under the registry lock; if the version changed between the eligibility check and the binding, split and retry before any GPU draw.
    if (g_rt_range_active && (g_pass.reg_gen != g_reg_gen || g_pass.ser_gen != g_ser_gen || g_pass.range_dirty != nvn_dirty_now())) { rp_end(); draw(dw, di); return; }
    int prim = dw ? dw->prim : di->prim;
    MTLPrimitiveType mp;
    switch (prim) {
        case 0: mp = MTLPrimitiveTypePoint; break; case 1: mp = MTLPrimitiveTypeLine; break; case 3: mp = MTLPrimitiveTypeLineStrip; break;
        case 4: case 7: mp = MTLPrimitiveTypeTriangle; break; case 5: mp = MTLPrimitiveTypeTriangleStrip; break;
        default: RED_ONCE(19, "图元类型 %d 未实现", prim); g_draw_skip[11]++; return;
    }
    if (g_vis_counting && perdraw_vis_on() && g_enc_has_vis && g_vis && g_vis_n < VIS_SLOTS - 256) {
        /* TOTK_ANOMALY: this draw gets its own slot, which also counts towards the game's samples-passed counter (a report
           sums every slot since the reset): consecutive slots of one buffer extend the list's last entry. */
        uint32_t slot = g_vis_n++;
        if (g_nvs && g_vs[g_nvs - 1].buf == (__bridge void *)g_vis && g_vs[g_nvs - 1].slot + g_vs[g_nvs - 1].n + 1 == slot) g_vs[g_nvs - 1].n++;
        else { if (g_nvs == g_vscap) { g_vscap = g_vscap ? g_vscap * 2 : 64; g_vs = realloc(g_vs, g_vscap * sizeof *g_vs); }
               g_vs[g_nvs++] = (vis_slot){(void *)CFBridgingRetain(g_vis), slot, 0}; }
        enc_r_visibility(en, MTLVisibilityResultModeCounting, (NSUInteger)slot * 8); g_enc_vis = (int)slot + 1; g_vis_cur = -1; g_anom_vis_slot = (int)slot;
    } else if (g_vis_counting) {   // 遮挡查询计数：本通道新槽（报告 / 清零后也换新槽） | Occlusion query counting: a new slot for this render pass (also switch to a new slot after a report / reset)
        if (!g_enc_has_vis) RED_ONCE(34, "遮挡查询计数开始时所在通道没挂可见性缓冲（首次清零发生在通道中途），该通道的采样不计");
        else {
            if (g_vis_cur < 0 || g_vis_cur_pass != g_pass.serial) {
                if (g_vis_n >= VIS_SLOTS) RED_ONCE(35, "可见性槽用尽");
                else { g_vis_cur = (int)g_vis_n++; g_vis_cur_pass = g_pass.serial;
                    if (g_nvs == g_vscap) { g_vscap = g_vscap ? g_vscap * 2 : 64; g_vs = realloc(g_vs, g_vscap * sizeof *g_vs); }
                    g_vs[g_nvs++] = (vis_slot){(void *)CFBridgingRetain(g_vis), (uint32_t)g_vis_cur}; }
            }
            if (g_vis_cur >= 0 && g_enc_vis != g_vis_cur + 1) { enc_r_visibility(en, MTLVisibilityResultModeCounting, (NSUInteger)g_vis_cur * 8); g_enc_vis = g_vis_cur + 1; }
        }
    } else if (perdraw_vis_on() && g_enc_has_vis && g_vis && g_vis_n < VIS_SLOTS - 256) {   // TOTK_ANOMALY: this draw's own sample counter
        uint32_t slot = g_vis_n++; enc_r_visibility(en, MTLVisibilityResultModeCounting, (NSUInteger)slot * 8); g_enc_vis = (int)slot + 1; g_anom_vis_slot = (int)slot;
    }
    if (g_skv_cur >= 0) { if (g_anom_vis_slot >= 0 && g_vis) skv_note((uint32_t)g_anom_vis_slot); else g_skv_cur = -1; if (!anom_vis_on()) g_anom_vis_slot = -1; }
    if (pr && nan_trace_match(pr->sh[1]) && !pr->sh[1]->writes_depth) nan_trace_bind(en);   // diagnostic TOTK_NAN_TRACE
    g_nt_pass_draws++;
    g_stat_draw++;
    { static int tg; if (!tg) { tg = 1; const char *e = getenv("DM_TEST_GHOST"); float v[8];   // 测试开关 DM_TEST_GHOST=px,py,gx,gy,gw,gh,cx,cy（归一化；按下点、格网、跟手图中心）：不经前端直接开跟手图（Mac 宿主截图验证用） | Test switch DM_TEST_GHOST=px,py,gx,gy,gw,gh,cx,cy (normalized; press point, grid, center of the drag-follow image): turns on the drag-follow image directly without going through the front end (for Mac host screenshot verification)
        if (e && sscanf(e, "%f,%f,%f,%f,%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) == 8) mtl_ui_ghost(1, v[0], v[1], &v[2], v[6], v[7]); } }
    if (dw && atomic_load(&g_ghost_on)) ghost_note(dw);
    if (enc_label_on() && !g_el_draws++ && pr->sh[1]) { memcpy(g_el_key, pr->sh[1]->key, 8); g_el_key[8] = 0; }
    if (g_pt_key_pending && pr->sh[1]) pt_note_key(pr->sh[1]->key);
    if (anom_on() && g_anom_rec) anom_note(pr, dw, di);   // TOTK_ANOMALY
    if (passdump()) {   // 通道首次绘制的特征（判读入是否可省：全屏、无混合、写满通道、不依赖深度）；DM_NVN_PASSDUMP_ALL 时每次绘制都记 | Characteristics of a render pass's first draw (to judge whether the load can be omitted: full screen, no blending, writes all channels, does not depend on depth); with DM_NVN_PASSDUMP_ALL every draw is recorded
        static uint64_t last_pass = ~0ull; static int all = -1; if (all < 0) all = getenv("DM_NVN_PASSDUMP_ALL") != NULL || getenv("TOTK_INPUT_FILE") != NULL;
        if (all) PD("d %llu bv=%d bi=%d first=%d vs=%s fs=%s vc=%d inst=%d blend=%#x b0=%#llx dt=%d dw=%d df=%d cull=%d st=%d sf=%d,%d sop=%d,%d,%d/%d,%d,%d smask=%#x,%#x/%#x,%#x sref=%#x,%#x ib=%#llx it=%d cmask=%#x front=%d dr=%g,%g clamp=%d", g_stat_pass, di ? di->base_vertex : 0, di ? di->base_instance : 0, dw ? dw->first : 0, pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[1] ? pr->sh[1]->key : "-",
                    dw ? dw->count : di->count, dw ? 1 : di->instances, S.color.blend_en, (unsigned long long)*(uint64_t *)&S.blend[0], (int)S.ds.dt, (int)S.ds.dw, (int)S.ds.df, (int)S.poly.cull,
                    (int)S.ds.st, (int)S.ds.sf_f, (int)S.ds.sf_b, (int)S.ds.fail_f, (int)S.ds.zfail_f, (int)S.ds.zpass_f,
                    (int)S.ds.fail_b, (int)S.ds.zfail_b, (int)S.ds.zpass_b, S.st_vmask[0], S.st_vmask[1], S.st_wmask[0], S.st_wmask[1], S.st_ref[0], S.st_ref[1],
                    di ? (unsigned long long)di->ibuf : 0ull, di ? di->itype : 0, S.chmask.mask, (int)S.poly.front, S.dr[0], S.dr[1], (int)S.depth_clamp);
        if (all && pr->sh[0]) {   // 顶点阶段各常量缓冲的内容指纹（判某次绘制跨帧是否完全相同：阴影缓存等用）；读不到（本提交待执行的写）记 P | Content fingerprint of each constant buffer of the vertex stage (to judge whether a draw is completely identical across frames: used by the shadow cache etc.); when unreadable (a write pending execution in this submit) record P
            char ub[400]; size_t o = 0; NvShader *v = pr->sh[0];
            for (int i = 0; i < v->rf.nb && o < sizeof ub - 40; i++) if (v->rf.b[i].kind == NVN_RB_CBUF && v->rf.b[i].reg < 16) {
                const p_bind_buf *u = &S.ubo[0][v->rf.b[i].reg]; void *mb; uint64_t mo; const uint8_t *base; uint64_t h = 1469598103934665603ull;
                if (!u->addr || !u->size) { o += snprintf(ub + o, sizeof ub - o, " r%d=-", v->rf.b[i].reg); continue; }
                if (pend_hits(u->addr, u->size) || !mtl_addr(u->addr, u->size, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) { o += snprintf(ub + o, sizeof ub - o, " r%d=P/%llu", v->rf.b[i].reg, (unsigned long long)u->size); continue; }
                uint64_t h64 = 0; for (uint64_t q = 0; q < u->size; q++) { h = (h ^ base[mo + q]) * 1099511628211ull; if (q == 63) h64 = h; }   // h64：前 64 字节（物体缓冲的世界矩阵 vec0–3） | h64: the first 64 bytes (world matrix vec0–3 of the object buffer)
                { static uint64_t hp = ~0ull; static int hn; if (hp != g_stat_pass) { hp = g_stat_pass; hn = 0; }   // 每通道前 6 次绘制另把 ≤ 608 字节的缓冲按十六进制打出来（逐字节比跨帧变了哪几个字段） | For the first 6 draws of each render pass, additionally print buffers of <= 608 bytes in hex (to compare byte by byte which fields changed across frames)
                  if (hn < 12 && u->size <= 608 && v->rf.b[i].reg == 3) { hn++; char hx[1300]; for (uint64_t q = 0; q < u->size; q++) snprintf(hx + 2 * q, 3, "%02x", base[mo + q]); PD("x %llu r%d %s", g_stat_pass, v->rf.b[i].reg, hx); } }
                o += snprintf(ub + o, sizeof ub - o, " r%d=%016llx/%llu/%016llx", v->rf.b[i].reg, (unsigned long long)h, (unsigned long long)u->size, (unsigned long long)h64); }
            PD("u %llu%s", g_stat_pass, ub); }
        if (all) { char vb[1400]; size_t o = 0; o += snprintf(vb + o, sizeof vb - o, "nattr=%d", S.nattr);
            for (int i = 0; i < S.nattr && i < 16; i++) { int st = S.attr[i].stream; if (!S.attr[i].fmt && i >= 6) continue;
                o += snprintf(vb + o, sizeof vb - o, " a%d{fmt=%#x off=%d s=%d stride=%u div=%u vb=%#llx/%llu}", i, S.attr[i].fmt, S.attr[i].offset, st,
                              st < 16 ? S.stream[st].stride : 0, st < 16 ? S.stream[st].divisor : 0, st < 16 ? (unsigned long long)S.vbuf[st].address : 0ull, st < 16 ? (unsigned long long)S.vbuf[st].size : 0ull); }
            PD("v %llu %s", g_stat_pass, vb);
            /* TOTK: first 64 bytes of vertex stream 0 and the first 16 index bytes, read from the CPU-visible pool (what the GPU will fetch). */
            { char hx[200]; size_t o2 = 0; void *mb; uint64_t mo; const uint8_t *b;
              if (S.vbuf[0].address && mtl_addr(S.vbuf[0].address, 64, &mb, &mo) && (b = [(__bridge id<MTLBuffer>)mb contents])) { for (int q = 0; q < 64; q++) o2 += snprintf(hx + o2, sizeof hx - o2, "%02x", b[mo + q]); PD("w %llu vb0=%s", g_stat_pass, hx); }
              if (di && di->ibuf && mtl_addr(di->ibuf, 16, &mb, &mo) && (b = [(__bridge id<MTLBuffer>)mb contents])) { o2 = 0; for (int q = 0; q < 16; q++) o2 += snprintf(hx + o2, sizeof hx - o2, "%02x", b[mo + q]); PD("w %llu ib=%s", g_stat_pass, hx); } } }
        if (last_pass != g_stat_pass) {
            last_pass = g_stat_pass;
            PD("D %llu prim=%d vc=%d inst=%d blend=%#x wm=%#x dt=%d dw=%d df=%d st=%d sc=%lu,%lu,%lu,%lu tgt=%lux%lu vp=%.0f,%.0f,%.0f,%.0f", g_stat_pass, prim,
               dw ? dw->count : di->count, dw ? 1 : di->instances, S.color.blend_en, S.chmask.mask, (int)S.ds.dt, (int)S.ds.dw, (int)S.ds.df, (int)S.ds.st,
               (unsigned long)sc.x, (unsigned long)sc.y, (unsigned long)sc.width, (unsigned long)sc.height, (unsigned long)tw, (unsigned long)th,
               S.has_vp ? S.vp[0][0] : 0, S.has_vp ? S.vp[0][1] : 0, S.has_vp ? S.vp[0][2] : (float)tw, S.has_vp ? S.vp[0][3] : (float)th);
        }
    }
    if (rt_range_on() && !rt_range_pr_ro(pr)) g_pass.range_external=1;
    rt_range_note_depth();
#ifdef NVN_RT_RANGE_TEST
    rr_snap(dw,di);
#endif
    udc_note_draw(pr);
    // 0 个图元的绘制（游戏会发 count=0）不画任何像素，跳过编码；Metal 规定计数必须非零，校验层否则当场中止（营地 / 野外 / 战斗页实测）
    // A draw with 0 primitives (the game does send count=0) draws no pixels, so skip encoding; Metal requires the count to be non-zero, otherwise the validation layer aborts on the spot (measured on the camp / wilderness / combat pages)
    if (dw && prim == 7) {   // 四边形：共享索引 0,1,2,0,2,3 | Quads: shared indices 0,1,2,0,2,3
        if (!g_quad_ib) {
            uint32_t n = 65536; uint32_t *ix = malloc(n * 6 * 4);
            for (uint32_t q = 0; q < n; q++) { uint32_t b = q * 4; uint32_t v[6] = {b, b + 1, b + 2, b, b + 2, b + 3}; memcpy(ix + q * 6, v, 24); }
            g_quad_ib = [g_dev newBufferWithBytes:ix length:n * 24 options:MTLResourceStorageModeShared]; g_quad_gpu = g_quad_ib.gpuAddress; res_add(g_quad_ib); free(ix);
        }
        if (dw->count >= 4) enc_r_draw_indexed(en, mp, (dw->count / 4) * 6, MTLIndexTypeUInt32, g_quad_gpu, g_quad_ib.length, dw->instances > 1 ? dw->instances : 1, dw->first, dw->base_instance);
    } else if (dw) {
        if (dw->count && (dw->instances > 1 || dw->base_instance)) enc_r_draw_instanced(en, mp, dw->first, dw->count, dw->instances, dw->base_instance);
        else if (dw->count && dw->instances >= 0) enc_r_draw(en, mp, dw->first, dw->count);
    } else {
        if (prim == 7) { RED_ONCE(20, "带索引的四边形未实现"); return; }
        void *mb; uint64_t mo;
        uint64_t gpu;
        if (!mtl_gpu_addr(di->ibuf, 1, &mb, &mo, &gpu)) { RED_ONCE(22, "索引缓冲不在内存池里"); return; }
        if (di->itype == 0) {   // TOTK: Metal has no 8-bit indices; widen to 16 bits in this submit's arena (restart 0xFF -> 0xFFFF)
            id<MTLBuffer> wide; size_t woff;
            int wi = wide_find(di);
            if (wi >= 0) { wide = g_wide[wi].buf; woff = g_wide[wi].off; }   // widened on the GPU before this draw's pass
            else {
                if (pend_hits(di->ibuf, (uint64_t)di->count)) MDIE("8-bit index buffer written by pending GPU work was not widened on the GPU");
                const uint8_t *src = (const uint8_t *)[(__bridge id<MTLBuffer>)mb contents] + mo;
                woff = arena_alloc((size_t)di->count * 2 + 2, &wide);
                uint16_t *dst = (uint16_t *)((uint8_t *)wide.contents + woff);
                for (int i = 0; i < di->count; i++) dst[i] = src[i] == 0xff ? 0xffff : src[i];
            }
            { static int logged; const char *hl = hilite1_env();   /* TOTK diagnostic: index path of highlighted draws */
              if (hl && logged < 3 && pr->sh[1] && strstr(hl, [[NSString stringWithUTF8String:pr->sh[1]->key] substringToIndex:8].UTF8String)) { logged++;
                  const uint16_t *w16 = (const uint16_t *)((const uint8_t *)wide.contents + woff);
                  MLOG("HILITE draw: 8-bit index path %s, count %d, first 6 widened indices %u %u %u %u %u %u, source %#llx", wi >= 0 ? "GPU-widened (cached)" : "CPU-widened", di->count,
                       w16[0], w16[1], w16[2], w16[3], w16[4], w16[5], (unsigned long long)di->ibuf); } }
            if (di->count) enc_r_draw_indexed(en, mp, di->count, MTLIndexTypeUInt16, wide.gpuAddress + woff, (size_t)di->count * 2, di->instances ? di->instances : 1, di->base_vertex, di->base_instance);
            return;
        }
        use_rec("buf", di->ibuf, (uint64_t)di->count * (di->itype == 1 ? 2 : 4), 0, NULL);
        if (di->count) enc_r_draw_indexed(en, mp, di->count, di->itype == 1 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32, gpu, ((__bridge id<MTLBuffer>)mb).length - mo,
                di->instances ? di->instances : 1, di->base_vertex, di->base_instance);
    }
    if (g_rt_range_pending) { PD("j rt_level_range pass=%llu vc=%u", g_stat_pass, (unsigned)(dw ? dw->count : di->count)); g_rt_range_pending=0; }
}
static nvn_hmap g_cps = {.klen = sizeof(void *)};
static void warm_claim(void) {
    if (!atomic_exchange_explicit(&g_warm_ready, 0, memory_order_acq_rel)) return;
    os_unfair_lock_lock(&g_warm_lock);
    NSArray<WarmItem *> *items = [g_warm_items copy];
    [g_warm_items removeAllObjects];
    os_unfair_lock_unlock(&g_warm_lock);
    for (WarmItem *item in items) {
        if (item->kind == 0) {
            if (!g_shcache) g_shcache = [NSMutableDictionary dictionary];
            if (!g_shcache[item->key]) { g_shcache[item->key] = item->shader; g_warm_claim_shaders++; }
        } else if (item->kind == 1) {
            if (!g_shcache) g_shcache = [NSMutableDictionary dictionary];
            NvShader *vs = g_shcache[item->vs_key], *fs = item->fs_key.length ? g_shcache[item->fs_key] : nil;
            if (!vs) { vs = item->vs; g_shcache[item->vs_key] = vs; }
            if (item->fs_key.length && !fs) { fs = item->fs; g_shcache[item->fs_key] = fs; }
            item->pso.vs = (__bridge void *)vs; item->pso.fs = (__bridge void *)fs;
            if (!hm_get_id(&g_pso, &item->pso)) { hm_put_id(&g_pso, &item->pso, item->object); g_warm_claim_rps++; }
        } else {
            NvShader *cs = g_shcache[item->key];
            if (!cs) { cs = item->shader; if (!g_shcache) g_shcache = [NSMutableDictionary dictionary]; g_shcache[item->key] = cs; }
            const void *key = (__bridge const void *)cs;
            if (!hm_get_id(&g_cps, &key)) { hm_put_id(&g_cps, &key, item->object); g_warm_claim_cps++; }
        }
    }
}
// ---------- 3D 画面加亮 ----------
// 色调映射（片元 a1b1aab66933fa8f）= 把合成后的 HDR ×0.008 做 PQ 编码、再查 32³ 调色表（每帧由计算着色器 552f45c348a61b86 按区域重算，只被色调映射读）。
// HDR 先乘曝光 k 等价于把表换成 L'(u) = L(PQ(k · PQ⁻¹(u)))：游戏算完表的那次派发之后，拷一份再按此原地重写（32³ 一次小派发）；表每帧重算，不会越乘越亮。
// 只动 3D 世界：界面、文字、血球在色调映射之后才合；亮部经 PQ 压缩，不会像最后整屏乘系数那样糊成一片白。普通 / 可读 tile 合并 / 冻结各条路径都只是读这张表，一处生效。
// k 每帧由宿主 dm_bright_k() 给（dm/host/dm_bright.c：局外 1、迷宫类 4、原版画面不随昼夜 4、其余按原版环境亮度推，平滑过渡）；k ≈ 1 时不重写。
// ---------- 3D picture brightening ----------
// Tone mapping (fragment a1b1aab66933fa8f) = PQ-encode the composited HDR x0.008, then look up a 32^3 color lookup table (recomputed per region every frame by compute shader 552f45c348a61b86, read only by tone mapping).
// Multiplying HDR by exposure k first is equivalent to replacing the table with L'(u) = L(PQ(k * PQ^-1(u))): after the dispatch in which the game finishes computing the table, make a copy and rewrite it in place accordingly (one small 32^3 dispatch); the table is recomputed every frame, so it does not get brighter with each multiplication.
// Only the 3D world is touched: UI, text and the health orbs are composited after tone mapping; highlights are compressed by PQ and do not blur into a sheet of white as a final whole-screen multiply would. The normal / readable tile merge / freeze paths all just read this table, so it takes effect in one place.
// k is given every frame by the host's dm_bright_k() (dm/host/dm_bright.c: 1 out of game, 4 for maze-type areas, 4 where the original game's picture does not follow day and night, the rest derived from the original game's ambient brightness, with smooth transitions); no rewrite when k ~= 1.
__attribute__((weak)) float dm_bright_k(void) { return 1.0f; }   // 宿主 dm_bright.c 提供强定义；图形单测不链宿主时不加亮 | The host's dm_bright.c provides the strong definition; no brightening when graphics unit tests do not link the host
static const char k_lut_msl[] =
"#include <metal_stdlib>\nusing namespace metal;\n"
"constant float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;\n"   // 与色调映射着色器里的 PQ 常数逐位相同 | Bit-identical to the PQ constants in the tone mapping shader
"static float3 pq(float3 v) { float3 p = pow(abs(v), m1); return pow((c1 + c2 * p) / (1.0 + c3 * p), m2); }\n"
"static float3 pq_inv(float3 u) { float3 q = pow(u, 1.0 / m2); return pow(max(q - c1, 0.0) / max(c2 - c3 * q, 1e-6), 1.0 / m1); }\n"
"kernel void lutk(texture3d<float> src [[texture(0)]], texture3d<float, access::write> dst [[texture(1)]], constant float &k [[buffer(0)]], uint3 g [[thread_position_in_grid]]) {\n"
"  uint3 n = uint3(dst.get_width(), dst.get_height(), dst.get_depth()); if (any(g >= n)) return;\n"
"  constexpr sampler ls(filter::linear, address::clamp_to_edge, coord::normalized);\n"
"  float3 u = (float3(g) + 0.5) / float3(n); float3 u2 = clamp(pq(k * pq_inv(u)), 0.0, 1.0);\n"
"  float4 b = src.sample(ls, u2), o = src.read(g);\n"   // b：按曝光 k 提亮后的输出；o：原版这一格 | b: output after brightening by exposure k; o: this cell in the original game
"  float s = smoothstep(0.65, 1.0, dot(b.rgb, float3(0.2126, 0.7152, 0.0722)));\n"   // 高光保护：提亮后越接近白，越收回原值（雪、天光不糊成一片白）；暗处 s = 0 照常提亮 | Highlight protection: after brightening, the closer to white, the more it is pulled back to the original value (snow and skylight do not blur into a sheet of white); in dark areas s = 0 and brightening proceeds as usual
"  dst.write(float4(max(mix(b.rgb, o.rgb, s), o.rgb), b.a), g); }\n";
static void bright_lut(NvShader *cs) {
    static int failed; static id<MTLComputePipelineState> ps; static id<MTL4ArgumentTable> at; static id<MTLTexture> tmp; static id<MTLBuffer> kb[3]; static int ki;
    if (failed || strncmp(cs->key, "552f45c348a61b86", 16)) return;
    float k = dm_bright_k(); if (!(k > 1.005f)) return;   // k ≈ 1（局外 / 正午）：表保持原版 | k ~= 1 (out of game / noon): the table stays as in the original game
    if (k > 4.0f) k = 4.0f;
    int tid = 0; for (int i = 0; i < cs->rf.nb && !tid; i++) if (cs->rf.b[i].kind == NVN_RB_IMG && cs->rf.b[i].reg < 32) tid = (int)(S.img[5][cs->rf.b[i].reg] & 0xfffff);
    registry_lock(); id<MTLTexture> lut = tid ? tex_by_id(tid) : nil; registry_unlock();
    if (!lut || lut.textureType != MTLTextureType3D || !(lut.usage & MTLTextureUsageShaderWrite)) { static int n; if (!n++) MLOG("【红】3D 画面加亮：调色表认不出（贴图号 %d），不加亮", tid); return; }
    if (!ps) { NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(k_lut_msl) options:nil error:&e]; id<MTLFunction> f = lib ? [lib newFunctionWithName:@"lutk"] : nil;
        ps = f ? [g_dev newComputePipelineStateWithFunction:f error:&e] : nil;
        MTL4ArgumentTableDescriptor *ad = [MTL4ArgumentTableDescriptor new]; ad.maxTextureBindCount = 2; ad.maxBufferBindCount = 1; at = [M3ArgTable new];
        for (int i = 0; i < 3; i++) { kb[i] = [g_dev newBufferWithLength:16 options:MTLResourceStorageModeShared]; if (kb[i]) res_add(kb[i]); }
        if (!ps || !at || !kb[0] || !kb[1] || !kb[2]) { failed = 1; MLOG("【红】3D 画面加亮：计算管线建不出 %s（退回原版亮度）", e.localizedDescription.UTF8String); return; }
        MLOG("3D 画面加亮：建好（调色表 %lux%lux%lu，曝光每帧由 dm_bright_k 给）", (unsigned long)lut.width, (unsigned long)lut.height, (unsigned long)lut.depth); }
    if (!tmp || tmp.width != lut.width || tmp.height != lut.height || tmp.depth != lut.depth || tmp.pixelFormat != lut.pixelFormat) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor new]; d.textureType = MTLTextureType3D; d.pixelFormat = lut.pixelFormat; d.width = lut.width; d.height = lut.height; d.depth = lut.depth;
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate; if (tmp) defer_obj_release(tmp); tmp = [g_dev newTextureWithDescriptor:d]; if (!tmp) { failed = 1; return; } res_add(tmp); }
    id<MTLBuffer> b0 = kb[ki]; ki = (ki + 1) % 3; *(float *)b0.contents = k;   // 三份轮换：在途的帧还在读上一份 | Three copies in rotation: in-flight frames are still reading the previous copy
    id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_texture(b, lut, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(lut.width, lut.height, lut.depth), tmp, 0, 0, MTLOriginMake(0, 0, 0)); enc_b_end(g_blit); g_blit = nil;
    id<MTL4ComputeCommandEncoder> ce = (id<MTL4ComputeCommandEncoder>)enc_c_begin(NULL); m3_attach(at, ce, M3_STAGE_COMPUTE); [at setTexture:tmp atIndex:0]; [at setTexture:lut atIndex:1]; [at setAddress:b0.gpuAddress atIndex:0];
    enc_c_pso(ce, ps); enc_c_dispatch(ce, MTLSizeMake((lut.width + 3) / 4, (lut.height + 3) / 4, (lut.depth + 3) / 4), MTLSizeMake(4, 4, 4)); enc_c_end(ce);
    args_dirty_all();
}
/* TOTK: back-to-back dispatches share one compute encoder. Each dispatch used to open and end a compute encoder of its own
   (about 24 per world frame, 13 of them right after another dispatch). On Metal 4 the copy encoder (g_blit) is a compute
   encoder already kept open across copies, closed wherever another encoder starts or the command buffer is committed; a
   dispatch now uses it too. Ordering stays as before: between two commands of the encoder a barrier orders blit and dispatch
   stages with device visibility (blit()), as the end / start barriers of two encoders did. The argument table is set again
   for every dispatch (copies use none, other code may set its own). Off with DM_NVN_PASSGPU (one timing slot per encoder)
   and with TOTK_DISPATCH_REUSE=0. */
static int disp_reuse_on(void) { static int v = -1; if (v < 0) {
#if DM_NVN_M3
    /* Metal 3 的 g_blit 是真正的 blit 编码器（Metal 4 的 MTL4ComputeCommandEncoder 才兼做拷贝），
       不能在上面设计算管线或派发：M3 下每次派发单开一个 compute 编码器。
       In Metal 3 g_blit is a real blit encoder (only Metal 4's MTL4ComputeCommandEncoder doubles as the copy
       encoder), so it cannot take a compute pipeline or a dispatch: every dispatch opens its own compute encoder. */
    v = 0;
#else
    const char *e = getenv("TOTK_DISPATCH_REUSE"); v = !(e && !strcmp(e, "0")); if (!v) MLOG("TOTK_DISPATCH_REUSE=0：每次派发单开计算编码器（旧行为）");
#endif
    } return v; }
static uint64_t g_disp_n, g_disp_shared;
static int g_disp_probe_nobar;   // probe-gpu-dispatch-chain negative control: no barrier between the commands of the shared encoder
static id<MTL4ComputeCommandEncoder> disp_enc(void) {
    int open = g_blit != nil;
    g_disp_n++; g_disp_shared += open;
    if (!open) { rs_sync(); ser_gate(); pt_mark(g_cb, 'C', nil); g_blit = enc_b_begin(g_cb); ((void)0); }
    else if (!g_disp_probe_nobar) ((void)0);
    m3_attach(g_ct, g_blit, M3_STAGE_COMPUTE); args_dirty_all();
    if (g_disp_n % 20000 == 0) MLOG("[nvn] 计算派发 至 %llu 次：其中 %llu 次（%.0f%%）接在已开的计算编码器里，没有另开编码器", (unsigned long long)g_disp_n, (unsigned long long)g_disp_shared, 100.0 * (double)g_disp_shared / (double)g_disp_n);
    return g_blit;
}
static void dispatch(const p_dispatch *dp) {
    zskip_clear();
    if (!S.prog) { RED_ONCE(27, "派发前没有绑定程序"); return; }
    __unsafe_unretained NvProg *cpr = prog_get(S.prog);
    if (cpr->missing) { static uint64_t n; if (!n++) MLOG("[shader-miss] 计算着色器还没翻译：跳过这个派发（以后同类不再提示）"); return; }
    NvShader *cs = cpr->sh[5];
    if (!cs) { RED_ONCE(28, "派发的程序没有计算着色器（或未映射）"); return; }
    const void *k = (__bridge const void *)cs;
    id<MTLComputePipelineState> ps = hm_get_id(&g_cps, &k);
    if (!ps) {
        NSError *e = nil; uint64_t bt = bp_now(); ps = [g_dev newComputePipelineStateWithFunction:cs->fn error:&e];
        g_bp_pso_n++; g_bp_pso_ns += bp_now() - bt;
        if (!ps) { MLOG("【红】建计算管线失败：%s", e.localizedDescription.UTF8String); return; }
        if (survey_shader_log()) MLOG("[survey-shader] frame %llu compute pipeline %s in %.2f ms", (unsigned long long)atomic_load(&g_cur_frame), cs->key, (bp_now() - bt) / 1e6);
        pso_record("C\t%s\n", cs->key);
        if (cs->key[0]) pso_user_record("C\t%s\n", cs->key);
        hm_put_id(&g_cps, &k, ps);
    }
    if (g_nupd) upd_guard_stage(cs, 5);   // TOTK: renamed uniform updates touched by this dispatch's storage / texture buffers
    if (cs->builtin) {   // 内嵌计算核：n = uniform 缓冲 0，src = 存储缓冲 0，dst = 存储缓冲 1 | Built-in compute kernel: n = uniform buffer 0, src = storage buffer 0, dst = storage buffer 1
        const p_bind_buf *u = &S.ubo[5][0], *a0 = &S.ssbo[5][0], *a1 = &S.ssbo[5][1];
        if (g_nupd) { upd_guard(u->addr, u->addr + 16); upd_guard(a0->addr, a0->addr + (a0->size ? a0->size : 8)); upd_guard(a1->addr, a1->addr + (a1->size ? a1->size : 8)); }   // the built-in kernel reads its uniform from memory
        void *ub, *sb, *db; uint64_t uo, so, dof;
        if (!mtl_addr(u->addr, 4, &ub, &uo) || !mtl_addr(a0->addr, 8, &sb, &so) || !mtl_addr(a1->addr, 8, &db, &dof)) { RED_ONCE(30, "内嵌计算核 %s 的缓冲没绑全", cs->builtin); return; }
        if (!cs->logged++) {
            module_t *m = module_of((void *)dp->caller);
            MLOG("内嵌计算核 %s 首次派发：调用点 %s+%#llx，线程组 %dx%dx%d，元素数（CPU 侧当前值）%u", cs->builtin, m ? m->name : "?",
                 m ? (unsigned long long)(dp->caller - (uint64_t)m->base) : (unsigned long long)dp->caller, dp->x, dp->y, dp->z, *(uint32_t *)u->addr);
        }
        rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
        id<MTLComputeCommandEncoder> ce = pg_compute();
        enc_c_pso(ce, ps);
        enc_c_args(ce, ((__bridge id<MTLBuffer>)ub).gpuAddress + uo, 0); enc_c_args(ce, ((__bridge id<MTLBuffer>)sb).gpuAddress + so, 1); enc_c_args(ce, ((__bridge id<MTLBuffer>)db).gpuAddress + dof, 2);   // 内嵌计算核：冷路径，直接取地址 | Built-in compute kernel: cold path, take the addresses directly
        enc_c_dispatch(ce, MTLSizeMake(dp->x, dp->y, dp->z), MTLSizeMake(32, 1, 1));
        enc_c_end(ce);
        return;
    }
    const uint16_t *tg = cs->rf.tg;
    if (!tg[0]) { RED_ONCE(29, "计算着色器缺线程组大小反射"); return; }
    rp_end();
    int reuse = disp_reuse_on() && !pg_on();
    id<MTLComputeCommandEncoder> ce;
    if (reuse) ce = disp_enc();
    else { if (g_blit) { enc_b_end(g_blit); g_blit = nil; } ce = pg_compute(); }
    enc_c_pso(ce, ps);
    g_dispatch_grid[0] = (uint32_t)dp->x; g_dispatch_grid[1] = (uint32_t)dp->y; g_dispatch_grid[2] = (uint32_t)dp->z;
    encode_args(ce, cs, 5, 0);
    enc_c_dispatch(ce, MTLSizeMake(dp->x, dp->y, dp->z), MTLSizeMake(tg[0], tg[1], tg[2]));
    if (!reuse) enc_c_end(ce);
    bright_lut(cs);   // 3D 画面加亮：游戏算完调色表之后重写 | 3D picture brightening: rewrite after the game finishes computing the color lookup table
    /* TOTK_EXPO_LOG=<file> (diagnostic): after each dispatch of the exposure kernel (ae725063), copy floats 96..127 of its
       storage buffer 0 (statistics; exposure at 108..111) into a shared ring and write them with the frame number once
       the GPU is done. */
    { static int on = -1; static FILE *f; if (on < 0) { const char *e = getenv("TOTK_EXPO_LOG"); on = e && (f = fopen(e, "w")) != NULL; }
      if (on && !strncmp(cs->key, "ae725063", 8)) {
          const p_bind_buf *sb = &S.ssbo[5][0]; void *mb; uint64_t mo;
          if (sb->addr && mtl_addr(sb->addr + 384, 128, &mb, &mo)) {
              id<MTLBuffer> rb; size_t ro = arena_alloc(128, &rb);
              id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_buffer(b, (__bridge id<MTLBuffer>)mb, mo, rb, ro, 128); enc_b_end(g_blit); g_blit = nil;
              const float *v = (const float *)((const uint8_t *)rb.contents + ro); uint64_t fr = atomic_load(&g_cur_frame);
              [g_commit_blocks addObject:^{ fprintf(f, "%llu", (unsigned long long)fr); for (int i = 0; i < 32; i++) fprintf(f, "\t%g", v[i]); fputc('\n', f); fflush(f); }];
          }
      } }
}
// GPU：贴图 → 缓冲（行距取 SetCopyRowStride，0 = 紧排）
// GPU: texture -> buffer (row stride taken from SetCopyRowStride, 0 = tightly packed)
static void copy_t2b(const p_copy_b2t *c) {
    if (c->tex) xv_check(c->tex->serial, c->has_view && c->view.set_levels ? c->view.base_level : 0, c->has_view && c->view.set_layers ? c->view.min_layer : 0, "贴图读缓冲");
    id<MTLTexture> t = tex_mtl(c->tex);
    if (!t) return;
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh);
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((c->r.width + bw - 1) / bw) * bpb;
    size_t rows = (c->r.height + bh - 1) / bh, img = row * rows;
    int is3d = c->tex->d.target == 2, nz = c->r.depth ? c->r.depth : 1;
    void *mb; uint64_t mo;
    upd_guard(c->src, c->src + img * nz);   // TOTK: renamed uniform updates before a GPU write to the range
    if (!mtl_addr(c->src, img * nz, &mb, &mo)) { RED_ONCE(31, "贴图→缓冲拷贝的目标不在内存池里"); return; }
    usnap_drop(c->src, img * nz);   // TOTK: a GPU write: no uniform copy of these bytes from here on
    id<MTL4ComputeCommandEncoder> b = blit();
    if (is3d)
        enc_b_copy_texture_buffer(b, t, 0, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, c->r.zoffset), MTLSizeMake(c->r.width, c->r.height, nz), (__bridge id<MTLBuffer>)mb, mo, row, img);
    else
        for (int z = 0; z < nz; z++)
            enc_b_copy_texture_buffer(b, t, layer0 + c->r.zoffset + z, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, 0), MTLSizeMake(c->r.width, c->r.height, 1), (__bridge id<MTLBuffer>)mb, mo + z * img, row, img);
}
// CPU 读贴图（ReadTexels）：预排布贴图以池内存为真值，解排布直读；其余从 MTLTexture blit 读回（调用方已按 NVN 规矩等过 GPU）
// CPU texture read (ReadTexels): for packaged textures the pool memory is the ground truth, de-layout and read directly; the rest are read back from the MTLTexture by blit (the caller has already waited for the GPU per NVN rules)
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    xv_check(t->serial, level, layer0, "CPU读回");
    PD("U cpu_read ser=%llu lv%u sl%u xyz=%d,%d,%d size=%d,%d,%d dst=%p row=%lu img=%lu", t->serial, level, layer0, r->xoffset, r->yoffset, r->zoffset, r->width, r->height, r->depth, dst, (unsigned long)row, (unsigned long)img);
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, hb = (r->height + bh - 1) / bh, x0 = (size_t)(r->xoffset / bw) * bpb, y0 = r->yoffset / bh;
    int nz = r->depth ? r->depth : 1, is3d = t->d.target == 2;
    /* TOTK: a mirrored texture the GPU has rendered into keeps its content only in the MTLTexture (its pool bytes are
       older, and it is never re-uploaded from them, see tex_mtl): read that back like any GPU texture. The game bakes
       terrain tile normals on the GPU into CPU-visible memory and reads them back; reading the pool bytes gave it stale
       normals, which it wrote into new tiles: the ground turned black block by block after moving (2026-10-15). */
    uint32_t upf_; int ubpp_;
    int gpu_content = tex_mirrored(t) && gpu_owned(t->serial) && !bc_unpack(t->d.format, &upf_, &ubpp_);
    TILELOG(t, "ReadTexels lv%u xy=%d,%d %dx%d (%s)", level, r->xoffset, r->yoffset, r->width, r->height, gpu_content ? "from the GPU texture" : tex_mirrored(t) ? "from the pool bytes" : "GPU texture");
    if (gpu_content) { static int n; if (++n <= 8) MLOG("[texels] ReadTexels ser=%llu %ux%u nvnfmt=%#x lv%u: rendered by the GPU, read back from the Metal texture (%d so far)", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, level, n); }
    if (tex_mirrored(t) && !gpu_content) {
        uint32_t lw = MAX(t->d.width >> level, 1u), lh = MAX(t->d.height >> level, 1u), lz = is3d ? MAX(t->d.depth >> level, 1u) : 1;
        size_t lp = (size_t)((lw + bw - 1) / bw) * bpb, lr = (lh + bh - 1) / bh;
        uint8_t *lin = malloc(lp * lr * lz);
        for (int z = 0; z < nz; z++) {
            unsigned layer = is3d ? 0 : layer0 + r->zoffset + z;
            if ((is3d && z == 0) || !is3d) if (nvn_deswizzle(&t->d, level, layer, t->mem, lin, lp)) { RED_ONCE(32, "ReadTexels 解排布失败"); free(lin); return; }
            const uint8_t *src = lin + (is3d ? (size_t)(r->zoffset + z) * lp * lr : 0);
            for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, src + (y0 + y) * lp + x0, wb);
        }
        free(lin);
        return;
    }
    id<MTLTexture> mt = tex_mtl(t);
    if (!mt) { memset(dst, 0, img * nz); return; }
    mtl_finish_gpu();
    id<MTLBuffer> buf = [g_dev newBufferWithLength:wb * hb * nz options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = enc_b_begin(cb);
    for (int z = 0; z < nz; z++)
        [b copyFromTexture:mt sourceSlice:is3d ? 0 : layer0 + r->zoffset + z sourceLevel:level sourceOrigin:MTLOriginMake(r->xoffset, r->yoffset, is3d ? r->zoffset + z : 0)
                sourceSize:MTLSizeMake(r->width, r->height, 1) toBuffer:buf destinationOffset:z * wb * hb destinationBytesPerRow:wb destinationBytesPerImage:wb * hb];
    enc_b_end(b); cb_commit(cb); cb_wait(cb);
    for (int z = 0; z < nz; z++) for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, (uint8_t *)buf.contents + z * wb * hb + y * wb, wb);
}
// TOTK: CPU texture write (WriteTexels), mirroring mtl_read_texels: packaged textures update pool memory in
// block-linear layout (dirty tracking re-uploads it); other textures are written into the MTLTexture by blit.
void mtl_write_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, const uint8_t *src, size_t row, size_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!bpb) die("WriteTexels: unknown format %d", t->d.format);
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, hb = (r->height + bh - 1) / bh, x0 = (size_t)(r->xoffset / bw) * bpb, y0 = r->yoffset / bh;
    int nz = r->depth ? r->depth : 1, is3d = t->d.target == 2;
    if (!r->width || !r->height || !nz) return;   // empty region writes nothing
    if (r->xoffset < 0 || r->yoffset < 0 || r->zoffset < 0 || r->width < 0 || r->height < 0 || nz < 0 || r->xoffset % bw || r->yoffset % bh) die("WriteTexels: invalid region xyz=%d,%d,%d size=%d,%d,%d fmt=%#x tgt=%d %ux%ux%u", r->xoffset, r->yoffset, r->zoffset, r->width, r->height, r->depth, t->d.format, t->d.target, t->d.width, t->d.height, t->d.depth);
    uint32_t lw = MAX(t->d.width >> level, 1u), lh = MAX(t->d.height >> level, 1u), lz = is3d ? MAX(t->d.depth >> level, 1u) : 1;
    if ((uint32_t)(r->xoffset + r->width) > lw || (uint32_t)(r->yoffset + r->height) > lh || (is3d && (uint32_t)(r->zoffset + nz) > lz)) die("WriteTexels: region outside level %u", level);
    TILELOG(t, "WriteTexels lv%u xy=%d,%d %dx%d owned %d mirrored %d flags=%#x pbh=%d tileH=%u size=%llu", level, r->xoffset, r->yoffset, r->width, r->height, gpu_owned(t->serial), tex_mirrored(t), t->d.flags, t->d.pkg_block_h, t->L.tileH, (unsigned long long)t->L.size);
    if (tex_mirrored(t)) {
        size_t lp = (size_t)((lw + bw - 1) / bw) * bpb, lr = (lh + bh - 1) / bh;
        uint8_t *lin = malloc(lp * lr * lz);
        if (!lin) die("WriteTexels: out of memory");
        dm_guest_write_prepare(t->mem, t->L.size);
        for (int z = 0; z < nz; z++) {
            unsigned layer = is3d ? 0 : layer0 + r->zoffset + z;
            if ((is3d && z == 0) || !is3d) if (nvn_deswizzle(&t->d, level, layer, t->mem, lin, lp)) die("WriteTexels: unsupported layout");
            uint8_t *dst = lin + (is3d ? (size_t)(r->zoffset + z) * lp * lr : 0);
            for (size_t y = 0; y < hb; y++) memcpy(dst + (y0 + y) * lp + x0, src + z * img + y * row, wb);
            if (!is3d || z == nz - 1) if (nvn_swizzle(&t->d, level, layer, lin, lp, t->mem)) die("WriteTexels: unsupported layout");
        }
        free(lin);
        /* TOTK: a mirrored texture the GPU has rendered into is never re-uploaded from its pool bytes (tex_mtl), so the
           write goes into the MTLTexture as well (below); the pool bytes stay in step for texture objects created later. */
        if (!gpu_owned(t->serial)) return;
    }
    uint32_t upf; int ubpp;
    if (tex_astc(t)) {   // TOTK: BC1 stored as ASTC 4x4: transcode the written blocks on the CPU
        id<MTLTexture> mt = tex_mtl(t);
        if (!mt) die("WriteTexels: texture %llu has no Metal storage", (unsigned long long)t->serial);
        mtl_finish_gpu();
        size_t bw4 = wb / 8, arow = bw4 * 16;
        id<MTLBuffer> buf = [g_dev newBufferWithLength:arow * hb * nz options:MTLResourceStorageModeShared];
        for (int z = 0; z < nz; z++) nvn_bc1_to_astc_image(src + z * img, row, (uint32_t)bw4, (uint32_t)hb, (uint8_t *)buf.contents + z * arow * hb, bc1_rgba(t->d.format));
        id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = enc_b_begin(cb);
        for (int z = 0; z < nz; z++)
            [b copyFromBuffer:buf sourceOffset:z * arow * hb sourceBytesPerRow:arow sourceBytesPerImage:arow * hb sourceSize:MTLSizeMake(r->width, r->height, 1)
                    toTexture:mt destinationSlice:layer0 + r->zoffset + z destinationLevel:level destinationOrigin:MTLOriginMake(r->xoffset, r->yoffset, 0)];
        enc_b_end(b); cb_commit(cb); cb_wait(cb);
        if (!nogpu() && cb.status != MTLCommandBufferStatusCompleted) die("WriteTexels: GPU upload failed");
        return;
    }
    if (bc_unpack(t->d.format, &upf, &ubpp)) die("WriteTexels: BC fallback textures not supported");
    id<MTLTexture> mt = tex_mtl(t);
    if (!mt) die("WriteTexels: texture %llu has no Metal storage", (unsigned long long)t->serial);
    mtl_finish_gpu();
    id<MTLBuffer> buf = [g_dev newBufferWithLength:wb * hb * nz options:MTLResourceStorageModeShared];
    for (int z = 0; z < nz; z++) for (size_t y = 0; y < hb; y++) memcpy((uint8_t *)buf.contents + z * wb * hb + y * wb, src + z * img + y * row, wb);
    id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = enc_b_begin(cb);
    for (int z = 0; z < nz; z++)
        [b copyFromBuffer:buf sourceOffset:z * wb * hb sourceBytesPerRow:wb sourceBytesPerImage:wb * hb sourceSize:MTLSizeMake(r->width, r->height, 1)
                toTexture:mt destinationSlice:is3d ? 0 : layer0 + r->zoffset + z destinationLevel:level destinationOrigin:MTLOriginMake(r->xoffset, r->yoffset, is3d ? r->zoffset + z : 0)];
    enc_b_end(b); cb_commit(cb); cb_wait(cb);
    if (!nogpu() && cb.status != MTLCommandBufferStatusCompleted) die("WriteTexels: GPU upload failed");
}
static void copy_b2t(const p_copy_b2t *c) {
    if (c->tex) zskip_del(c->tex->serial);
    id<MTLTexture> t = tex_mtl(c->tex);
    PD("C ser=%llu", c->tex->serial);
    if (!t) return;
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh);
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((c->r.width + bw - 1) / bw) * bpb;
    size_t rows = (c->r.height + bh - 1) / bh, img = row * rows;
    int is3d = c->tex->d.target == 2, nz = c->r.depth ? c->r.depth : 1;
    void *mb; uint64_t mo;
    upd_guard(c->src, c->src + img * nz);   // TOTK: renamed uniform updates before a GPU read of the range
    if (!mtl_addr(c->src, img * nz, &mb, &mo)) { RED_ONCE(23, "缓冲→贴图拷贝的源不在内存池里"); return; }
    uint32_t upf; int ubpp;
    if (pf_astc4(t.pixelFormat) && bc1_fmt(c->tex->d.format)) {   // TOTK: BC1 stored as ASTC 4x4: the source blocks are transcoded on the CPU
        const uint8_t *src = [(__bridge id<MTLBuffer>)mb contents];
        if (!src) { RED_ONCE(46, "无 BC 解压：缓冲→贴图拷贝的源缓冲 CPU 读不到"); return; }
        size_t bw4 = (c->r.width + 3) / 4, arow = bw4 * 16, aimg = arow * rows;
        id<MTLBuffer> tb = [g_dev newBufferWithLength:aimg * nz ?: 16 options:MTLResourceStorageModeShared];
        for (int z = 0; z < nz; z++) nvn_bc1_to_astc_image(src + mo + z * img, row, (uint32_t)bw4, (uint32_t)rows, (uint8_t *)tb.contents + z * aimg, bc1_rgba(c->tex->d.format));
        res_add(tb);
        id<MTL4ComputeCommandEncoder> b = blit();
        for (int z = 0; z < nz; z++) enc_b_copy_buffer_texture(b, tb, z * aimg, arow, aimg, MTLSizeMake(c->r.width, c->r.height, 1), t, layer0 + c->r.zoffset + z, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, 0));
        defer_obj_release(tb);
        return;
    }
    if (bc_unpack(c->tex->d.format, &upf, &ubpp)) {   // 无 BC：源里的压缩块在 CPU 上解压进临时缓冲，再按像素拷进解压格式的贴图 | No BC: the compressed blocks in the source are decompressed on the CPU into a temporary buffer, then copied pixel-wise into the decompressed-format texture
        const uint8_t *src = [(__bridge id<MTLBuffer>)mb contents];
        if (!src) { RED_ONCE(46, "无 BC 解压：缓冲→贴图拷贝的源缓冲 CPU 读不到"); return; }
        size_t up = (size_t)c->r.width * ubpp, uimg = up * c->r.height;
        id<MTLBuffer> tb = [g_dev newBufferWithLength:uimg * nz ?: 16 options:MTLResourceStorageModeShared];
        for (int z = 0; z < nz; z++) nvn_bc_decode(c->tex->d.format, src + mo + z * img, row, c->r.width, c->r.height, (uint8_t *)tb.contents + z * uimg, up);
        res_add(tb);
        id<MTL4ComputeCommandEncoder> b = blit();
        if (is3d) enc_b_copy_buffer_texture(b, tb, 0, up, uimg, MTLSizeMake(c->r.width, c->r.height, nz), t, 0, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, c->r.zoffset));
        else for (int z = 0; z < nz; z++) enc_b_copy_buffer_texture(b, tb, z * uimg, up, uimg, MTLSizeMake(c->r.width, c->r.height, 1), t, layer0 + c->r.zoffset + z, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, 0));
        defer_obj_release(tb);   // 本次提交跑完再放（并撤驻留） | Release after this submit has finished running (and remove from residency)
        return;
    }
    id<MTL4ComputeCommandEncoder> b = blit();
    if (is3d)
        enc_b_copy_buffer_texture(b, (__bridge id<MTLBuffer>)mb, mo, row, img, MTLSizeMake(c->r.width, c->r.height, nz), t, 0, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, c->r.zoffset));
    else
        for (int z = 0; z < nz; z++)
            enc_b_copy_buffer_texture(b, (__bridge id<MTLBuffer>)mb, mo + z * img, row, img, MTLSizeMake(c->r.width, c->r.height, 1), t, layer0 + c->r.zoffset + z, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, 0));
}

// 无 BC + 老机型补充包：游戏把文件来的贴图（已是 ASTC 4x4）整块拷进自己建的 BC3 纹理数组（无 BC 时建成 RGBA8）。Switch 上两种格式块大小相同、
// 拷贝是按原始块搬；这里改成计算核逐像素解出来写进去，sRGB 源先读出线性值再编码回 sRGB 值，结果 = 原样搬块后按目标格式读到的值
// No BC + Legacy Device Pack: the game copies textures from files (already ASTC 4x4) whole into a BC3 texture array it creates itself (created as RGBA8 when there is no BC). On Switch the two formats have the same block size and
// the copy moves raw blocks; here it is changed to a compute kernel that decodes pixel by pixel and writes them in, with sRGB sources first read out as linear values and then encoded back to sRGB values, so the result = the values read in the destination format after moving the blocks as-is
static const char k_a2u_msl[] =
"#include <metal_stdlib>\nusing namespace metal;\n"
"static float3 enc(float3 c) { return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308); }\n"
"kernel void a2u(texture2d<float> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]], constant uint4 *p [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {\n"
"  if (g.x >= p[1].x || g.y >= p[1].y) return; float4 v = src.read(g + p[0].xy); if (p[1].z) v.rgb = enc(v.rgb); dst.write(v, g + p[0].zw); }\n";
static int copy_t2t_astc(id<MTLTexture> s, id<MTLTexture> d, unsigned sl, unsigned ss, unsigned dl, unsigned ds, int sx, int sy, int dx, int dy, int w, int h) {
    static int failed; static id<MTLComputePipelineState> ps; static id<MTL4ArgumentTable> at;
    if (failed) return 0;
    if (!ps) { NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(k_a2u_msl) options:nil error:&e]; id<MTLFunction> f = lib ? [lib newFunctionWithName:@"a2u"] : nil;
        ps = f ? [g_dev newComputePipelineStateWithFunction:f error:&e] : nil;
        MTL4ArgumentTableDescriptor *ad = [MTL4ArgumentTableDescriptor new]; ad.maxTextureBindCount = 2; ad.maxBufferBindCount = 1; at = [M3ArgTable new];
        if (!ps || !at) { failed = 1; MLOG("【红】无 BC：ASTC → 解压格式拷贝的计算管线建不出 %s", e.localizedDescription.UTF8String); return 0; } }
    MTLTextureType tt = MTLTextureType2D;
    id<MTLTexture> sv = [s newTextureViewWithPixelFormat:s.pixelFormat textureType:tt levels:NSMakeRange(sl, 1) slices:NSMakeRange(ss, 1)];
    id<MTLTexture> dv = [d newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm textureType:tt levels:NSMakeRange(dl, 1) slices:NSMakeRange(ds, 1)];
    id<MTLBuffer> pb = [g_dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
    if (!sv || !dv || !pb) { RED_ONCE(47, "无 BC：ASTC → 解压格式拷贝建不出视图 / 参数块"); return 0; }
    uint32_t *p = pb.contents; p[0] = sx; p[1] = sy; p[2] = dx; p[3] = dy; p[4] = w; p[5] = h; p[6] = s.pixelFormat == MTLPixelFormatASTC_4x4_sRGB; p[7] = 0;
    res_add(pb);
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    id<MTL4ComputeCommandEncoder> ce = (id<MTL4ComputeCommandEncoder>)enc_c_begin(NULL); m3_attach(at, ce, M3_STAGE_COMPUTE);
    [at setTexture:sv atIndex:0]; [at setTexture:dv atIndex:1]; [at setAddress:pb.gpuAddress atIndex:0];
    enc_c_pso(ce, ps); enc_c_dispatch(ce, MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), MTLSizeMake(8, 8, 1)); enc_c_end(ce);
    args_dirty_all();
    defer_obj_unref(sv); defer_obj_unref(dv); defer_obj_release(pb);   // 本次提交跑完再放（视图不单独驻留，参数块撤驻留） | Release after this submit has finished running (views are not made resident separately; the parameter block is removed from residency)
    return 1;
}
// GPU：贴图 → 贴图（原先误列为空操作，选角页每帧 3 次）。等尺寸、不镜像、同格式 → 逐层 blit；其余语义（缩放带过滤、镜像、换格式）blit 表达不了，报红待实现
// GPU: texture -> texture (previously mislisted as a no-op, 3 times per frame on the character select page). Same size, not mirrored, same format -> blit layer by layer; the other semantics (scaling with filtering, mirroring, format change) cannot be expressed by blit: report a failure, to be implemented
static void copy_t2t(const p_copy_t2t *c) {
    if (c->src) xv_check(c->src->serial, c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, "贴图拷贝源");
    rp_flush();   // 挂起的纯清除通道先落地（零源候选在通道结束时才进集合）；下面 blit() 本来也会结束通道 | Land the pending pure-clear render pass first (zero-source candidates only enter the set when the render pass ends); the blit() below would end the render pass anyway
    if (c->dst) zskip_del(c->dst->serial);
    if (c->src && c->dst && c->sr.xoffset == 0 && c->sr.yoffset == 0 && c->dr.xoffset == 0 && c->dr.yoffset == 0
        && c->sr.width == (int)c->src->d.width && c->sr.height == (int)c->src->d.height && c->dr.width == (int)c->dst->d.width && c->dr.height == (int)c->dst->d.height) {
        int sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
        int ss = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, ds = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
        if (zskip_has(c->src->serial, sl, ss)) zskip_add(c->dst->serial, dl, ds);
    }
    id<MTLTexture> s = tex_mtl(c->src), d = tex_mtl(c->dst);
    PD("T src=%llu dst=%llu %dx%dx%d→%dx%dx%d flags=%#x", c->src->serial, c->dst->serial, c->sr.width, c->sr.height, c->sr.depth, c->dr.width, c->dr.height, c->dr.depth, c->flags);
    if (!s || !d) return;
    unsigned sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
    unsigned sz0 = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, dz0 = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
    PD("t src=%llu lv%u sl%u xyz=%d,%d,%d dst=%llu lv%u sl%u xyz=%d,%d,%d size=%d,%d,%d", c->src->serial, sl, sz0, c->sr.xoffset, c->sr.yoffset, c->sr.zoffset,
       c->dst->serial, dl, dz0, c->dr.xoffset, c->dr.yoffset, c->dr.zoffset, c->sr.width, c->sr.height, c->sr.depth);
    int s3 = c->src->d.target == 2, d3 = c->dst->d.target == 2, nz = c->sr.depth ? c->sr.depth : 1;
#if DM_NVN_M3
    if (!s3 && !d3 && c->sr.depth==c->dr.depth &&
        (c->sr.width!=c->dr.width || c->sr.height!=c->dr.height) &&
        !(c->flags & ~(NVN_COPY_FLAGS_ENGINE_2D_BIT|NVN_COPY_FLAGS_LINEAR_FILTER_BIT))) {
        rp_end();blit_close();rs_sync();es_reset();
        int ok=1;
        for(int z=0;z<nz && ok;z++)ok=nvn_scaled_copy_encode(g_dev,g_cb,s,d,sl,sz0+c->sr.zoffset+z,dl,dz0+c->dr.zoffset+z,
            c->sr.xoffset,c->sr.yoffset,c->sr.width,c->sr.height,c->dr.xoffset,c->dr.yoffset,c->dr.width,c->dr.height,c->flags&NVN_COPY_FLAGS_LINEAR_FILTER_BIT);
        args_dirty_all();g_pso_dirty=g_dss_dirty=1;
        if(ok){gpu_owned_mark(c->dst->serial);return;}
    }
#endif
    if (no_bc() && (s.pixelFormat == MTLPixelFormatASTC_4x4_LDR || s.pixelFormat == MTLPixelFormatASTC_4x4_sRGB) && (d.pixelFormat == MTLPixelFormatRGBA8Unorm || d.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB)
        && !s3 && !d3 && c->sr.width == c->dr.width && c->sr.height == c->dr.height && (d.usage & MTLTextureUsageShaderWrite)) {   // 无 BC：ASTC 源 → 解压格式目标（见 copy_t2t_astc） | No BC: ASTC source -> decompressed-format destination (see copy_t2t_astc)
        int ok = 1;
        for (int z = 0; z < nz && ok; z++) ok = copy_t2t_astc(s, d, sl, sz0 + c->sr.zoffset + z, dl, dz0 + c->dr.zoffset + z, c->sr.xoffset, c->sr.yoffset, c->dr.xoffset, c->dr.yoffset, c->sr.width, c->sr.height);
        if (ok) return;
    }
    if (c->sr.width != c->dr.width || c->sr.height != c->dr.height || (c->sr.depth ? c->sr.depth : 1) != (c->dr.depth ? c->dr.depth : 1) || (c->flags & ~NVN_COPY_FLAGS_ENGINE_2D_BIT & ~NVN_COPY_FLAGS_LINEAR_FILTER_BIT)
        || s.pixelFormat != d.pixelFormat || s3 != d3) {
        RED_ONCE(33, "贴图→贴图拷贝未实现的形态：%dx%dx%d→%dx%dx%d 标志 %#x 格式 %lu→%lu 3D %d→%d", c->sr.width, c->sr.height, c->sr.depth, c->dr.width, c->dr.height, c->dr.depth,
                 c->flags, (unsigned long)s.pixelFormat, (unsigned long)d.pixelFormat, s3, d3);
        return;
    }
    id<MTL4ComputeCommandEncoder> b = blit();
    if (s3)
        enc_b_copy_texture(b, s, 0, sl, MTLOriginMake(c->sr.xoffset, c->sr.yoffset, c->sr.zoffset), MTLSizeMake(c->sr.width, c->sr.height, nz), d, 0, dl, MTLOriginMake(c->dr.xoffset, c->dr.yoffset, c->dr.zoffset));
    else
        for (int z = 0; z < nz; z++)
            enc_b_copy_texture(b, s, sz0 + c->sr.zoffset + z, sl, MTLOriginMake(c->sr.xoffset, c->sr.yoffset, 0), MTLSizeMake(c->sr.width, c->sr.height, 1), d, dz0 + c->dr.zoffset + z, dl, MTLOriginMake(c->dr.xoffset, c->dr.yoffset, 0));
}

// GPU：用给定值清贴图区域（原先误列为空操作，初始化调 5 次；Metal 私有贴图初值未定义，游戏依赖清出的初值）。
// 整级整面、四通道全写 → 每个层 / 深度切片开一个只清不画的通道（硬件按贴图格式换算清除值，含整数、深度）；
// 视图换格式：在该格式的视图上清；部分区域、部分通道、压缩 / 预排布贴图（无渲染目标用途）：报红待实现
// GPU: clear a texture region with the given value (previously mislisted as a no-op, called 5 times at initialization; the initial value of a Metal private texture is undefined, and the game relies on the cleared initial value).
// Whole level and whole face, all four channels written -> open one clear-only render pass per layer / depth slice (the hardware converts the clear value by the texture format, including integer and depth);
// view with changed format: clear on a view of that format; partial region, partial channels, compressed / packaged textures (no render target usage): report a failure, to be implemented
static void clear_tex(const p_clear_tex *c) {
    if (c->tex) zskip_del(c->tex->serial);
    id<MTLTexture> t = tex_mtl(c->tex);
    PD("X ser=%llu 视图格式 %#x 区域 %d,%d,%d %dx%dx%d 掩码 %#x %s %g,%g,%g,%g", c->tex->serial, c->has_view && c->view.set_format ? c->view.format : -1, c->r.xoffset, c->r.yoffset, c->r.zoffset, c->r.width, c->r.height, c->r.depth, c->mask,
       c->is_uint ? "ui" : "f", c->is_uint ? (double)c->c.u[0] : c->c.f[0], c->is_uint ? (double)c->c.u[1] : c->c.f[1], c->is_uint ? (double)c->c.u[2] : c->c.f[2], c->is_uint ? (double)c->c.u[3] : c->c.f[3]);
    if (!t) return;
    unsigned lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, l0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    NSUInteger w = MAX(t.width >> lv, 1u), h = MAX(t.height >> lv, 1u);
    int is3d = t.textureType == MTLTextureType3D, nz = c->r.depth ? c->r.depth : 1;
    if (c->has_view && c->view.set_format) {   // 视图换格式：清除值按视图格式解释（游戏 5 次全是 ui 版配整数视图），在同格式视图上清 | View with changed format: the clear value is interpreted in the view format (all 5 of the game's calls are the ui version with an integer view), and the clear is done on a view of the same format
        uint32_t vpf = 0; uint8_t swz[4];
        if (!nvn_fmt_to_mtl(c->view.format, &vpf, swz) || !vpf) { RED_ONCE(42, "清贴图视图格式 %#x 无对应", c->view.format); return; }
        if (tex_need_pfv(t.pixelFormat, (MTLPixelFormat)vpf) && !(t.usage & MTLTextureUsagePixelFormatView)) t = tex_upgrade_pfv(c->tex, t, (MTLPixelFormat)vpf);
        if (t.pixelFormat != (MTLPixelFormat)vpf) {
            id<MTLTexture> v = [t newTextureViewWithPixelFormat:(MTLPixelFormat)vpf];
            if (!v) { RED_ONCE(43, "清贴图建不出视图：格式 %lu → %u", (unsigned long)t.pixelFormat, vpf); return; }
            res_add(v); defer_obj_release(v); t = v;
        }
    }
    MTLPixelFormat pf = t.pixelFormat;
    int depth = pf == MTLPixelFormatDepth32Float || pf == MTLPixelFormatDepth16Unorm || pf == MTLPixelFormatDepth32Float_Stencil8, stencil = pf == MTLPixelFormatStencil8 || pf == MTLPixelFormatDepth32Float_Stencil8;
    if (c->r.xoffset || c->r.yoffset || (NSUInteger)c->r.width != w || (NSUInteger)c->r.height != h || (c->mask & 0xf) != 0xf
        || !(t.usage & MTLTextureUsageRenderTarget)) {
        RED_ONCE(44, "清贴图未实现的形态：区域 %d,%d %dx%d（本级 %lux%lu）掩码 %#x 可作渲染目标 %d", c->r.xoffset, c->r.yoffset, c->r.width, c->r.height,
                 (unsigned long)w, (unsigned long)h, c->mask, !!(t.usage & MTLTextureUsageRenderTarget));
        return;
    }
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    for (int z = 0; z < nz; z++) {
        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
        MTLRenderPassAttachmentDescriptor *a = depth ? rp.depthAttachment : stencil ? rp.stencilAttachment : rp.colorAttachments[0];
        a.texture = t; a.level = lv; a.loadAction = MTLLoadActionClear; a.storeAction = MTLStoreActionStore;
        if (is3d) a.depthPlane = c->r.zoffset + z; else a.slice = l0 + c->r.zoffset + z;
        if (depth) { rp.depthAttachment.clearDepth = pf == MTLPixelFormatDepth16Unorm ? nvn_depth16_clear_value(c->c.f[0]) : c->c.f[0];
            if (stencil) { rp.stencilAttachment.texture = t; rp.stencilAttachment.level = lv; rp.stencilAttachment.slice = a.slice; rp.stencilAttachment.loadAction = MTLLoadActionClear;
                           rp.stencilAttachment.storeAction = MTLStoreActionStore; rp.stencilAttachment.clearStencil = c->c.u[1]; } }
        else if (stencil) rp.stencilAttachment.clearStencil = c->c.u[0];
        else rp.colorAttachments[0].clearColor = c->is_uint ? MTLClearColorMake(c->c.u[0], c->c.u[1], c->c.u[2], c->c.u[3]) : MTLClearColorMake(c->c.f[0], c->c.f[1], c->c.f[2], c->c.f[3]);
        ser_gate(); enc_r_end(enc_r_begin(rp));
    }
}

// 一次提交的现场描述（mtl_sub_commit 每次提交取一份，命令缓冲出错时打进红行）：进崔凡克首帧的 GPU 内存故障，红行里只有序号与错误码，看不出那次提交里有什么
// Context description of one submit (mtl_sub_commit takes one per submit and prints it into the failure line when the command buffer errors): for the GPU memory fault on the first frame entering Travincal, the failure line had only the sequence number and error code, and it was impossible to tell what was in that submit
static const char *cctx_snap(void) {
    static char b[320]; static uint64_t p_pass, p_up, p_reup, p_ser; uint64_t up = 0, reup = 0; mtl_pkg_stats(&up, &reup);
    pthread_mutex_lock(&g_deferred_mu); unsigned long nd = (unsigned long)g_deferred.count; pthread_mutex_unlock(&g_deferred_mu);
    snprintf(b, sizeof b, "帧 %llu；自上次提交：通道 +%llu、贴图首传 +%llu、重传 +%llu、序号表变动 +%llu；延迟释放队列 %lu 个；世界冻结状态 %d",
             (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)(g_stat_pass - p_pass), (unsigned long long)(up - p_up), (unsigned long long)(reup - p_reup),
             (unsigned long long)(g_ser_gen - p_ser), nd, g_frz_state);
    p_pass = g_stat_pass; p_up = up; p_reup = reup; p_ser = g_ser_gen; return b;
}
// ---------- 世界冻结时不画世界（面板开着时整个游戏可以冻结） ----------
// 「世界已冻结」状态位 g_frz_on 由 mtl_world_frozen() 置（宿主的冻结握手调它；测试时可用 DM_NVN_TEST_FRZ=1 顶替，也可经 DM_NVN_UB 的 @set:g_frz_on 同进程切）。
// 这里不自己判面板开没开。一帧的结构（3.3.94012 局内，逐通道转储）：界面趟 →［界面模板趟］→ 阴影 / 深度 / 光照 / 透明 / 后处理约 40 趟 → 末尾合成趟（写窗口贴图）。
// 置位后的第一帧 = 完整帧：照常全画，只丢界面模板趟的绘制（模板全 0 = 没有被面板挡住的像素）——后处理平时不写被面板挡住的像素，面板缩小会露出旧内容；
// 第二帧起 = 冻结帧：从第一个不带颜色附件的通道起、到写窗口贴图的通道之前，设目标 / 清除 / 绘制 / 贴图间拷贝 / 计算全不执行，场景结果留着完整帧那一份，界面趟与末尾合成趟照画。
// 帧的界线按通道认：写过窗口贴图之后第一条不带窗口贴图的设目标 = 新一帧开始，状态位在这里锁存。认不出结构（一帧里没有不带颜色的通道）= 什么都不跳。
// 状态常量 / 缓冲更新 / 事件 / 计数器 / 贴图上传照常执行（界面趟要用，游戏在等）。
// ---------- Do not draw the world while the world is frozen (the whole game can be frozen while a panel is open) ----------
// The "world frozen" status bit g_frz_on is set by mtl_world_frozen() (the host's freeze handshake calls it; in tests DM_NVN_TEST_FRZ=1 can stand in, or it can be switched in-process via @set:g_frz_on of DM_NVN_UB).
// This code does not judge by itself whether a panel is open. Structure of a frame (3.3.94012 in-game, per-pass dump): UI pass -> [UI stencil pass] -> about 40 passes of shadows / depth / lighting / transparency / post-processing -> final compositing pass (writes the window texture).
// The first frame after the bit is set = full frame: draw everything as usual, dropping only the draws of the UI stencil pass (stencil all 0 = no pixels blocked by the panel) -- post-processing normally does not write pixels blocked by the panel, so a shrinking panel would expose old content;
// from the second frame on = frozen frame: from the first render pass without a color attachment up to before the render pass that writes the window texture, set-target / clear / draw / texture-to-texture copy / compute are all not executed, the scene result stays as the full frame's copy, and the UI pass and the final compositing pass are drawn as usual.
// Frame boundaries are identified by render pass: the first set-target without the window texture after the window texture has been written = start of a new frame, and the status bit is latched here. Unrecognized structure (no render pass without color in a frame) = skip nothing.
// State constants / buffer updates / events / counters / texture uploads execute as usual (the UI pass needs them, the game is waiting).
int g_frz_on = -1;
static int g_frz_phase = 2, g_frz_drop, g_frz_seen_ds; static uint64_t g_frz_full_n, g_frz_skip_n;
void mtl_world_frozen(int on) { g_frz_on = on != 0; }
static int frz_wanted(void) { if (g_frz_on < 0) { const char *e = getenv("DM_NVN_TEST_FRZ"); g_frz_on = e && atoi(e); if (g_frz_on) MLOG("DM_NVN_TEST_FRZ：世界冻结状态位由测试开关顶替（常开）"); } tbdr_on(); return g_frz_on > 0 && tbdr_on() && g_ruler_truth <= 0; }
// ---------- 运行时外推（省电画法；不给玩家开关；线上出过卡顿缺陷，现缺省关，A17 上有不卡的证据再开）----------
// 3D 隔帧：全画帧照常画，并在附件被挪作他用之前拷下不透明 HDR（468）、主深度模板、色调映射后的 468、FXAA 后的 469；
// 外推帧：界面模板趟照跑；阴影图 / 深度预渲染 / 主通道只留会动单位（蒙皮）的绘制；深度预渲染第一个蒙皮绘制前跑一趟外推（nvn_rtex.inc：按 cb3 +464 → +336
// 把上一帧 HDR 与深度搬到本帧，旧单位位置 0x10 / 界面挖空 0x2 当洞补，洞与屏边新露出的像素导出模板 0x80）；半透明照画并写特效位 0x40；
// 合成 / 加泛光 / 加雾 / 色调映射 / FXAA 只在 0xD0 区域里真跑，区域外是上一帧 LDR 结果按整幅平移（正交相机，取整）拷进来；泛光链等全屏项跳过；锐化放大与合界面照跑。
// 退回原画法：版本文件 render_safe = true（dm_update.m）、外推管线建不出（整局）、测试 DM_RT_AB=0。面板开着世界冻结时让给冻结路径。
// 镜头跳变（朝向变或平移超 64 像素）与换 3D 渲染比例前后 4 帧整帧全画。地面上的单位影子沿用上一全画帧。
// ---------- Runtime extrapolation (power-saving rendering method; no switch for the player; it caused a stutter defect in released builds, now off by default, to be re-enabled once there is evidence of no stutter on A17) ----------
// 3D every other frame: a fully drawn frame draws as usual and, before the attachments are repurposed, copies the opaque HDR (468), the main depth-stencil, the post-tone-mapping 468 and the post-FXAA 469;
// extrapolated frame: the UI stencil pass runs as usual; shadow map / depth prepass / main pass keep only the draws of moving units (skinned); before the first skinned draw of the depth prepass run one extrapolation pass (nvn_rtex.inc: by cb3 +464 -> +336
// move the previous frame's HDR and depth to this frame, treat old unit positions 0x10 / UI cutout 0x2 as holes to fill, and export stencil 0x80 for holes and pixels newly exposed at the screen edge); translucents are drawn as usual and write effect bit 0x40;
// compositing / add bloom / add fog / tone mapping / FXAA really run only in the 0xD0 region, and outside the region the previous frame's LDR result is copied in with a whole-image shift (orthographic camera, rounded); full-screen items such as the bloom chain are skipped; sharpening upscale and UI compositing run as usual.
// Fallback to the original rendering method: version file render_safe = true (dm_update.m), the extrapolation pipeline cannot be built (whole run), test DM_RT_AB=0. When a panel is open and the world is frozen, yield to the freeze path.
// On a camera jump (orientation change or translation over 64 pixels) and for 4 frames before and after switching the 3D render scale, whole frames are fully drawn. Unit shadows on the ground keep using the previous fully drawn frame.
static int rt_is_skin(void) {   // 会动单位 = 顶点着色器按骨骼下标读常量的那 10 个（键表取自 Mac 着色器缓存 b-424a072084e6，按源码里 vp_c5 动态下标认） | Moving units = the 10 whose vertex shader reads constants by bone index (the key table is taken from the Mac shader cache b-424a072084e6, identified by the dynamic vp_c5 index in the source)
    __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; if (!pr || !pr->sh[0]) return 0;
    static const char *k[] = { "0fa4eb7559f1637a", "1d97d33ab66abcba", "3deaf68fd65b5298", "4dffa355f2fbcdaf", "54c93fa22abff9ee",
                               "64dd7d022f3fae39", "67e2ebc28c830f44", "add86ee8e20f89b9", "b3ae9be5cc670f35", "fbf6dc1873a449e9" };
    for (unsigned i = 0; i < sizeof k / sizeof *k; i++) if (!strncmp(pr->sh[0]->key, k[i], 16)) return 1;
    return 0;
}
int g_dm_rt_ab = 0;   // 缺省原画法（新画法在 A17 iPhone 15 Pro 上局内卡、选角页人物重影，先关）；DM_RT_AB=1 测试时开新画法。版本文件 render_safe 只会置 0，不影响 | Default is the original rendering method (the new method stutters in-game on A17 iPhone 15 Pro and ghosts characters on the character select page, so off for now); DM_RT_AB=1 turns on the new method for testing. The version file render_safe can only set it to 0, no effect
static struct { int env, frame, filt, first, hdrp, mainp_seen, skipall, oit_seen, tm_seen, fx_state, comp_seen; uint64_t ldr_ser; float vpc[16], vpp[16]; id<MTLTexture> ldr, pl468, pl469; int ui_seen, main_seen, copied, ex_done, shadow, have_prev, have_ldr, sdx, sdy, cam_checked, force_full, disabled, was_on; uint64_t n_jump; uint64_t ds_ser, hdr_ser, n_ex, n_full, n_noex;
    id<MTLTexture> ds, hdr, pds, phdr; struct { uint64_t key; int skin, other; } sh[16]; int nsh, cur_sh; } g_rt = { .env = -1 };
static int rt_want(void);
static void rt_toggle_poll(void) { static int tg = -1; if (tg < 0) { const char *e = getenv("DM_RT_AB_TOGGLE"); tg = e ? atoi(e) : 0; (void)rt_want(); }   // 首次顺带读 DM_RT_AB（真机上世界冻结位早早置 0，frz_op 不一定被调到）   // 测试：每 tg 秒按墙钟在 A / B 之间来回切 | On first call also read DM_RT_AB (on a real device the world-freeze bit is set to 0 very early, so frz_op is not necessarily called)   // test: switch back and forth between A / B every tg seconds by wall clock
    if (tg > 0) { int v = (int)((time(NULL) / tg) & 1); if (v != g_dm_rt_ab) { g_dm_rt_ab = v; MLOG("DM_RT_AB_TOGGLE：画法切到 %s", v ? "B" : "A"); } } }
void mtl_rt_force_full(int n) { g_rt.force_full = n; }   // 宿主调：要换 3D 渲染比例等画面整体变化前，之后 n 帧一律全画 | Called by the host: before an overall picture change such as switching the 3D render scale, the following n frames are all fully drawn
// ---------- 省电模式（30 帧）----------
// 开：显示节拍对象改 30 Hz（帧率范围 30–30，ProMotion 屏跟着降刷新率；front_ios.m 的帧率提示一起降），游戏每拍出一帧；运行时外推关（每帧真画）。
// 游戏逻辑本来就按 25 拍 / 秒走、单位位置按时间插值，30 帧下没有重复帧（真机实测：每帧镜头位移的变异系数 30 帧 0.292、60 帧 0.296）。
// 开关在「自定义」页（dm_custom.c eco_fps，缺省关，应用即生效）；测试开关 DM_FPS=30/60 钉住。
// ---------- Power-saving mode (30 frames) ----------
// On: the display link object changes to 30 Hz (frame rate range 30–30, a ProMotion screen lowers its refresh rate accordingly; the frame rate hint in front_ios.m is lowered together), the game presents one frame per display-link tick; runtime extrapolation off (really draws every frame).
// Game logic already runs at 25 ticks / second and unit positions are interpolated by time, so there are no repeated frames at 30 frames (measured on a real device: coefficient of variation of per-frame camera displacement 0.292 at 30 frames, 0.296 at 60 frames).
// The switch is on the 「自定义」(Custom) page (dm_custom.c eco_fps, off by default, takes effect on apply); the test switch DM_FPS=30/60 pins it.
__attribute__((weak)) int dm_fps_setting(void) { return 60; }   // 宿主 dm_custom.c 提供强定义；图形单测不链宿主时恒 60 | The host's dm_custom.c provides the strong definition; always 60 when graphics unit tests do not link the host
int mtl_fps_target(void) { static int env = -2; if (env == -2) { const char *e = getenv("DM_FPS"); env = e ? atoi(e) : -1; if (env != 30 && env != 60) env = -1; else MLOG("DM_FPS=%d：帧数由测试开关钉住", env); }
    return env > 0 ? env : dm_fps_setting() == 30 ? 30 : 60; }
// 外推启用状态一变（省电模式切换、世界冻结接手 / 交回）：先让「跨帧不写回」账本重学（试用期里照常写回），外推晚一帧再真正切换——
// 保证切换前最后一次提交是全量写回的。不然 30 帧时每帧先清空的附件被学成不写回，切回 60 帧的第一个外推帧读到旧内容、闪一帧（测试版实测，xv 自检报红后才撤销）
// Whenever the extrapolation enabled state changes (power-saving mode toggled, world freeze takes over / hands back): first let the "no store across frames" ledger relearn (during the trial period it stores as usual), and the extrapolation really switches one frame later --
// guaranteeing the last submit before the switch is a full store. Otherwise attachments cleared first every frame at 30 frames are learned as no-store, and the first extrapolated frame after switching back to 60 frames reads old content and flashes for one frame (measured on a test build; only reverted after the xv self-check reported a failure)
static void xv_relearn(const char *why) { g_xv_n = 0; g_xv_ntrial = 0; g_xv_ntouch = 0; MLOG("跨帧不写回：%s，账本清空重学（试用期内照常写回）", why); }
static int rt_want_raw(void);
static int rt_want(void) {
    static int eff = -1; static uint64_t pend; int want = rt_want_raw(); uint64_t f = atomic_load(&g_cur_frame);
    if (eff < 0) eff = want;
    if (want == eff) { pend = 0; return eff; }
    if (!pend) { pend = f + 1; xv_relearn(want ? "外推将启用" : "外推将停用"); return eff; }
    if (f > pend) { eff = want; pend = 0; }
    return eff;
}
static int rt_want_raw(void) { static int forced; if (g_rt.env < 0) { const char *e = getenv("DM_RT_AB"); g_rt.env = e ? atoi(e) : 0; forced = e != NULL; if (e) MLOG("DM_RT_AB=%d：画法由环境变量钉住（测试用，自定义页的开关不再起作用）", g_rt.env); }
    if (forced) g_dm_rt_ab = g_rt.env;
    return g_dm_rt_ab == 1 && !g_rt.disabled && g_frz_on <= 0 && tbdr_on() && g_ruler_truth <= 0 && mtl_fps_target() == 60; }   // 面板开着世界冻结时让给冻结那条（更省）；省电模式（30 帧）每帧真画 | When a panel is open and the world is frozen, yield to the freeze path (saves more); power-saving mode (30 frames) really draws every frame
#include "nvn_rtex.inc"
static uint64_t rt_key(const p_set_rt *r) { uint64_t s_; int lv, sl; rt_sub(r, 8, &s_, &lv, &sl); return s_ * 64 + (uint64_t)lv * 8 + (uint64_t)sl; }
static int rt_is_shadow(const p_set_rt *r) { return !r->n && r->depth && r->depth->d.width == r->depth->d.height; }
static id<MTLTexture> rt_cp(id<MTLTexture> src, id<MTLTexture> dst0) {
    id<MTLTexture> dst = dst0;
    if (!dst || dst.width != src.width || dst.height != src.height || dst.pixelFormat != src.pixelFormat) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat width:src.width height:src.height mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView; d.storageMode = MTLStorageModePrivate;
        if (dst0) defer_obj_release(dst0);   // 换尺寸：旧副本可能还被在途的命令缓冲读着（MTL4 命令缓冲不保留资源），延迟放 | Size change: the old copy may still be read by an in-flight command buffer (MTL4 command buffers do not retain resources), so release it deferred
        dst = [g_dev newTextureWithDescriptor:d]; if (!dst) return nil; res_add(dst); }
    id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_texture(b, src, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(src.width, src.height, 1), dst, 0, 0, MTLOriginMake(0, 0, 0)); return dst; }
static void rt_blit_shift(id<MTLTexture> src, id<MTLTexture> dst, int dx, int dy) {   // 整数偏移拷贝：dst(x + dx, y + dy) = src(x, y)；搬不到的屏边一条留给区域重算（外推趟已标 0x80） | Integer-offset copy: dst(x + dx, y + dy) = src(x, y); the strip at the screen edge that cannot be moved is left for the region recompute (the extrapolation pass has already marked it 0x80)
    if (!src || !dst || src.width != dst.width || src.height != dst.height) return;
    int W = (int)src.width, H = (int)src.height, w = W - abs(dx), h = H - abs(dy); if (w <= 0 || h <= 0) return;
    id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_texture(b, src, 0, 0, MTLOriginMake(MAX(0, -dx), MAX(0, -dy), 0), MTLSizeMake(w, h, 1), dst, 0, 0, MTLOriginMake(MAX(0, dx), MAX(0, dy), 0)); }
static void rt_copy_prev(void) {   // 全画帧：不透明 HDR 与主深度模板各拷一份（GPU 内按序，在本命令缓冲里） | Fully drawn frame: copy the opaque HDR and the main depth-stencil once each (in order within the GPU, in this command buffer)
    if (g_rt.copied || !g_rt.main_seen || !g_rt.hdr || !g_rt.ds) return; g_rt.copied = 1;   // 本帧没认出主通道（前台 / 读取画面）不拷：上次认的贴图可能已被游戏放掉、不在驻留集里 | Do not copy when the main pass was not recognized this frame (foreground / loading screen): the textures recognized last time may already have been released by the game and not be in the residency set
    if (!rx_setup(g_rt.hdr.pixelFormat, g_rt.ds.pixelFormat)) { if (!g_rt.disabled) { g_rt.disabled = 1; MLOG("【红】运行时外推：外推管线在这台设备上建不出，整局退回原画法"); } return; }   // 建不出（老 GPU 不支持模板导出等）：整局退回 | Cannot be built (old GPU does not support stencil export etc.): fall back for the whole run
    g_rt.phdr = rt_cp(g_rt.hdr, g_rt.phdr); g_rt.pds = rt_cp(g_rt.ds, g_rt.pds); g_rt.have_prev = g_rt.phdr && g_rt.pds; }
static void rt_frame_begin(void) { g_rt.filt = g_rt.ui_seen = g_rt.main_seen = g_rt.copied = g_rt.ex_done = g_rt.mainp_seen = 0; g_rt.cur_sh = -1;
    g_rt.oit_seen = g_rt.tm_seen = g_rt.fx_state = g_rt.comp_seen = g_rt.skipall = g_rt.cam_checked = 0; g_rt_ovr = 0; g_dss_dirty = 1;
    if (g_frz_state == 0) { g_rt.have_prev = g_rt.have_ldr = 0; for (int i = 0; i < g_rt.nsh; i++) g_rt.sh[i].skin = g_rt.sh[i].other = 0; } }   // 全画帧：本帧拷成了才许下一帧外推 | Fully drawn frame: the next frame may extrapolate only if this frame's copies succeeded
static void rt_set_rt(const p_set_rt *r, int win) {   // 每条设目标（全画帧与外推帧都走）：认角色、全画帧找拷贝点、外推帧定本通道是否只留蒙皮 | Every set-target (both fully drawn and extrapolated frames go through here): identify roles, find the copy points in fully drawn frames, and in extrapolated frames decide whether this render pass keeps only skinned draws
    g_rt.cur_sh = -1; g_rt.shadow = 0; g_rt.first = 1; g_rt.hdrp = 0;
    if (win) { if (g_frz_state == 0) { rt_copy_prev(); if (g_rt.fx_state == 2 && g_rt.ldr) { g_rt.pl469 = rt_cp(g_rt.ldr, g_rt.pl469); g_rt.have_ldr = g_rt.pl469 != nil; g_rt.fx_state = 3; } }
        g_rt.filt = 0; g_rt.skipall = 0; if (g_rt_ovr) { g_rt_ovr = 0; g_dss_dirty = 1; } return; }
    id<MTLTexture> c0 = r->n > 0 && r->color[0] ? tex_mtl(r->color[0]) : nil, dt = r->depth ? tex_mtl(r->depth) : nil;
    int hdr_here = 0; for (int i = 0; i < r->n && i < 8; i++) if (r->color[i] && g_rt.hdr_ser && r->color[i]->serial == g_rt.hdr_ser) hdr_here = 1;
    if (g_frz_state == 0) {
        if (!g_rt.main_seen && r->n == 1 && c0 && c0.pixelFormat == MTLPixelFormatRG11B10Float && dt && dt.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 && dt.width >= 500 && dt.width != dt.height) {
            { static uint64_t lh, ld; if (lh != r->color[0]->serial || ld != r->depth->serial) { lh = r->color[0]->serial; ld = r->depth->serial; MLOG("运行时外推：认出主通道 不透明 HDR 序号 %llu %lux%lu、主深度模板序号 %llu %lux%lu（通道 %llu）", (unsigned long long)lh, (unsigned long)c0.width, (unsigned long)c0.height, (unsigned long long)ld, (unsigned long)dt.width, (unsigned long)dt.height, g_stat_pass + 1); } }
            g_rt.main_seen = 1; g_rt.hdr_ser = r->color[0]->serial; g_rt.ds_ser = r->depth->serial; g_rt.hdr = c0; g_rt.ds = dt; }
        else if (g_rt.main_seen && hdr_here && !g_rt.copied) rt_copy_prev();   // 主通道之后再挂不透明 HDR 的第一个设目标（半透明那趟，468 占 0 号槽但不写）：之前拷 HDR 与主深度模板 | The first set-target after the main pass that attaches the opaque HDR again (the translucent pass; 468 occupies slot 0 but is not written): before it, copy the HDR and the main depth-stencil
        else if (g_rt.main_seen && hdr_here && r->n == 1 && r->depth && r->depth->serial == g_rt.ds_ser) g_rt.fx_state = 1;   // 只挂 468 + 主深度 = 色调映射（把 468 挪作 LDR 输出） | Only 468 + main depth attached = tone mapping (repurposes 468 as the LDR output)
        else if (g_rt.fx_state == 1) {   // 色调映射之后第一个通道 = FXAA（读 468、写 469）：之前拷色调映射后的 468（LDR），记下 469 | The first render pass after tone mapping = FXAA (reads 468, writes 469): before it, copy the post-tone-mapping 468 (LDR) and note down 469
            if (r->n >= 1 && r->color[0] && c0) { g_rt.ldr_ser = r->color[0]->serial; g_rt.ldr = c0; g_rt.pl468 = rt_cp(g_rt.hdr, g_rt.pl468); g_rt.fx_state = 2; } else g_rt.fx_state = 0; }
        else if (g_rt.fx_state == 2 && g_rt.ldr) { g_rt.pl469 = rt_cp(g_rt.ldr, g_rt.pl469); g_rt.have_ldr = g_rt.pl468 && g_rt.pl469; g_rt.fx_state = 3; }   // FXAA 之后第一个通道之前：拷 FXAA 后的 469 | Before the first render pass after FXAA: copy the post-FXAA 469
    }
    if (rt_is_shadow(r)) { uint64_t k = rt_key(r); int i; for (i = 0; i < g_rt.nsh && g_rt.sh[i].key != k; i++) {}
        if (i == g_rt.nsh && i < 16) { g_rt.sh[i].key = k; g_rt.sh[i].skin = g_rt.sh[i].other = 0; g_rt.nsh++; } g_rt.cur_sh = i < 16 ? i : -1; g_rt.shadow = 1; }
    int ds_here = r->depth && g_rt.ds_ser && r->depth->serial == g_rt.ds_ser;
    if (ds_here && !r->n && !g_rt.ui_seen) { g_rt.ui_seen = 1; g_rt.filt = 0; return; }   // 本帧第一个带主深度、无颜色的通道 = 界面模板趟：照跑 | The first render pass in this frame with the main depth and no color = the UI stencil pass: run as usual
    g_rt.hdrp = hdr_here && ds_here && r->n == 1 && !g_rt.mainp_seen; if (g_rt.hdrp) g_rt.mainp_seen = 1;   // 本帧第一个「只挂不透明 HDR + 主深度」的通道 = 主通道 | The first render pass in this frame with "only opaque HDR + main depth attached" = the main pass
    g_rt.filt = g_rt.shadow || (ds_here && !r->n) || g_rt.hdrp;
    if (g_frz_state == 2 && g_rt.mainp_seen && !g_rt.hdrp) {   // 外推帧、主通道之后：后处理只在单位 / 特效区域跑（乙方案） | Extrapolated frame, after the main pass: post-processing runs only in the unit / effect region (plan B)
        int c0ldr = r->n >= 1 && r->color[0] && g_rt.ldr_ser && r->color[0]->serial == g_rt.ldr_ser, ovr = 0; g_rt.skipall = 0;
        if (g_rt.fx_state == 2) {}   // FXAA 之后（锐化放大 + 暗角、合界面）：全跑 | After FXAA (sharpening upscale + vignette, UI compositing): run everything
        else if (g_rt.fx_state == 1) g_rt.fx_state = 2;
        else if (!g_rt.oit_seen && ds_here && r->n >= 2) { g_rt.oit_seen = 1; ovr = 3; }   // 半透明（加权混合双目标）：照画，通过处写特效位 0x40 | Translucent (weighted blended, two targets): draw as usual, writing effect bit 0x40 where it passes
        else if (!g_rt.comp_seen) { if (ds_here && c0ldr) { g_rt.comp_seen = 1; ovr = 1; rt_blit_shift(g_rt.pl469, g_rt.ldr, g_rt.sdx, g_rt.sdy); } else g_rt.skipall = 1; }   // 合成之前：上一帧 FXAA 后的 469 整数平移拷进来（区域里随后重算）   // 合成之前的其余各趟（mip 链、511 折射层等）跳过 | Before compositing: the previous frame's post-FXAA 469 is copied in with an integer shift (then recomputed in the region)   // the other passes before compositing (mip chain, 511 refraction layer etc.) are skipped
        else if (ds_here && c0ldr && g_rt.tm_seen) {   // FXAA 本趟（在区域里重算；区域外的 469 合成前已平移拷好） | The FXAA pass itself (recomputed in the region; the 469 outside the region was already shift-copied before compositing)
            g_rt.fx_state = 1; ovr = 1; }
        else if (ds_here) { if (hdr_here && r->n == 1) { g_rt.tm_seen = 1; rt_blit_shift(g_rt.pl468, g_rt.hdr, g_rt.sdx, g_rt.sdy); } ovr = 1; }   // 色调映射之前：上一帧色调映射后的 468（LDR）整数平移拷进来   // 加泛光、加雾、色调映射：只在区域里跑 | Before tone mapping: the previous frame's post-tone-mapping 468 (LDR) is copied in with an integer shift   // add bloom, add fog, tone mapping: run only in the region
        else g_rt.skipall = 1;   // 泛光链等不挂主深度的：跳过（区域里沿用上一帧的泛光） | Those without the main depth attached, such as the bloom chain: skip (the region keeps using the previous frame's bloom)
        if (ovr != g_rt_ovr) { g_rt_ovr = ovr; g_dss_dirty = 1; }
    }
}
static int rt_op(unsigned op) {   // 外推帧里非设目标的操作：返回 1 = 不执行 | Non-set-target operations in an extrapolated frame: returns 1 = not executed
    if (g_frz_state == 2 && g_rt_ovr == 1 && op == OP_CLEAR_COLOR) return 1;   // 区域后处理各趟不清：区域外是平移拷进来的上一帧结果 | The region post-processing passes do not clear: outside the region is the previous frame's result copied in with a shift
    if (g_rt.skipall && (op == OP_DRAW || op == OP_DRAW_IDX || op == OP_CLEAR_COLOR || op == OP_CLEAR_DS || op == OP_COPY_T2T || op == OP_CLEAR_TEX || op == OP_DISPATCH)) return 1;
    if (op == OP_DRAW || op == OP_DRAW_IDX) {
        if (g_frz_state == 2 && !g_rt.cam_checked) {   // 外推帧的第一个绘制（界面模板趟，阴影 / 预渲染之前）：核本帧相机，跳变就整帧改全画 | First draw of an extrapolated frame (UI stencil pass, before shadows / prepass): check this frame's camera, and on a jump switch the whole frame to fully drawn
            g_rt.cam_checked = 1; const p_bind_buf *u = &S.ubo[0][3]; void *mb; uint64_t mo; const uint8_t *base; int ok = 0;
            if (u->addr && u->size >= 528 && mtl_addr(u->addr, u->size, &mb, &mo) && (base = [(__bridge id<MTLBuffer>)mb contents])) {
                const float *cur = (const float *)(base + mo + 336), *prev = (const float *)(base + mo + 464); int dx, dy; float md = 0;
                for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) md = MAX(md, fabsf(cur[r * 4 + c] - prev[r * 4 + c]));   // 前三行（朝向 / 缩放）应不变：正交相机只平移 | The first three rows (orientation / scale) should be unchanged: an orthographic camera only translates
                rt_shift_px(prev, cur, g_rt.hdr.width, g_rt.hdr.height, &dx, &dy);
                ok = md < 1e-4f && abs(dx) <= 64 && abs(dy) <= 64; if (!ok) { static int n; if (n++ < 20) MLOG("运行时外推：镜头跳变（平移 %d,%d 像素、朝向差 %g），本帧改全画", dx, dy, md); } }
            if (!ok) { g_frz_state = 0; g_rt.n_jump++; rt_frame_begin(); g_rt.cam_checked = 1; return 0; }
        }
        if (g_frz_state == 0) { if (g_rt.cur_sh >= 0) { if (rt_is_skin()) g_rt.sh[g_rt.cur_sh].skin++; else g_rt.sh[g_rt.cur_sh].other++; } return 0; }
        if (!g_rt.filt) return 0;
        if (!rt_is_skin()) return 1;
        if (!g_rt.ex_done && !g_rt.shadow) {   // 深度预渲染第一个蒙皮绘制之前：外推趟（读本绘制绑定的 cb3：+336 本帧、+464 上一帧视图投影） | Before the first skinned draw of the depth prepass: the extrapolation pass (reads the cb3 bound by this draw: +336 this frame's, +464 the previous frame's view-projection)
            g_rt.ex_done = 1; const p_bind_buf *u = &S.ubo[0][3]; void *mb; uint64_t mo; const uint8_t *base;
            if (u->addr && u->size >= 528 && mtl_addr(u->addr, u->size, &mb, &mo) && (base = [(__bridge id<MTLBuffer>)mb contents])) {
                float cur[16], prev[16]; memcpy(cur, base + mo + 336, 64); memcpy(prev, base + mo + 464, 64); memcpy(g_rt.vpc, cur, 64); memcpy(g_rt.vpp, prev, 64); rt_shift_px(prev, cur, g_rt.hdr.width, g_rt.hdr.height, &g_rt.sdx, &g_rt.sdy);
                rp_end(); if (rt_ex_encode(g_rt.hdr, g_rt.ds, g_rt.phdr, g_rt.pds, prev, cur, 0x10, 0x2)) g_rt.n_ex++; else g_rt.n_noex++;
            } else g_rt.n_noex++;
        }
        return 0;
    }
    if (g_frz_state == 2 && g_rt.filt) {
        if (op == OP_CLEAR_COLOR || op == OP_CLEAR_DS) {
            if (g_rt.shadow && g_rt.cur_sh >= 0 && !g_rt.sh[g_rt.cur_sh].other) return 0;   // 上一全画帧只有蒙皮的阴影级（人物专用级）：照清照画 | A shadow level that had only skinned draws in the previous fully drawn frame (character-only level): clear and draw as usual
            return 1; }
    }
    return 0;
}
static int frz_op(unsigned op, const void *a) {   // 返回 1 = 这条操作不执行 | Returns 1 = this operation is not executed
    if (op == OP_SET_RT) {
        const p_set_rt *r = a; int win = 0;
        for (int i = 0; i < r->n; i++) { uint64_t s_; int lv, sl; rt_sub(r, i, &s_, &lv, &sl); if (s_ && xv_present(s_)) win = 1; }
        if (g_frz_phase == 2 && !win && rt_want()) {   // 运行时外推：新一帧，全画帧与外推帧交替 | Runtime extrapolation: new frame, fully drawn frames and extrapolated frames alternate
            int was = g_frz_state; g_frz_phase = 0; g_rt.was_on = 1;
            { static int dbg = -1, n; if (dbg < 0) dbg = getenv("DM_RT_DBG") != NULL; if (dbg && n < 4000 && g_rt.ds_ser) { n++; MLOG("[rtdbg] 帧界 通道 %llu 上一段状态 %d 认出主通道 %d 拷了 %d 外推 %d fx %d ldr %d hoist %d fz %d defer %d/%d", g_stat_pass, was, g_rt.main_seen, g_rt.copied, g_rt.ex_done, g_rt.fx_state, g_rt.have_ldr, g_hoist_n, g_fz.active, defer_has_active(), g_defer_replaying); } }
            if (g_rt.force_full > 0) g_rt.force_full--;
            g_frz_state = !was && g_rt.have_prev && g_rt.have_ldr && !g_rt.force_full && !g_hoist_n && !g_fz.active && !defer_has_active() && !g_defer_replaying ? 2 : 0;
            if (g_frz_state == 2) g_rt.frame++; else g_rt.n_full++;
            if (g_frz_state != was && (g_rt.frame + g_rt.n_full) % 600 < 2) MLOG("运行时外推：全画帧 %llu、外推帧 %d（外推趟编了 %llu 次、没编 %llu 次；镜头跳变改全画 %llu 次）", g_rt.n_full, g_rt.frame, g_rt.n_ex, g_rt.n_noex, g_rt.n_jump);
            rt_frame_begin();
        }
        if (g_frz_phase != 2 && rt_want()) { if (win) g_frz_phase = 2; rt_set_rt(r, win); return g_rt.skipall; }
        if (g_frz_phase == 2 && win && rt_want()) { rt_set_rt(r, win); return 0; }
        { int on = rt_want(); if (g_rt.was_on && !on) {   // 新画法刚关（版本文件退回 / 面板冻结接手 / 整局退回）：状态归零，交回原路径 | The new rendering method was just turned off (version file fallback / panel freeze takes over / fallback for the whole run): reset state to zero and hand back to the original path
              g_frz_state = 0; g_frz_phase = 2; g_rt.have_prev = g_rt.have_ldr = 0; if (g_rt_ovr) { g_rt_ovr = 0; g_dss_dirty = 1; } g_rt.skipall = 0;
              MLOG("运行时外推：新画法停用（画法 %d、世界冻结 %d、整局退回 %d）", g_dm_rt_ab, g_frz_on, g_rt.disabled); }
          g_rt.was_on = on; }
        if (g_frz_phase == 2 && !win) {   // 新一帧 | New frame
            int on = frz_wanted(), was = g_frz_state; g_frz_phase = 0; g_frz_seen_ds = 0;
            { static uint64_t fr, p0, k0; if (was == 2 && (++fr == 4 || fr % 600 == 0)) MLOG("世界冻结：第 %llu 个冻结帧开了 %llu 个通道、跳过 %llu 条设目标", fr, g_stat_pass - p0, g_frz_skip_n - k0); if (was != 2) fr = 0; p0 = g_stat_pass; k0 = g_frz_skip_n; }
            if (!on) g_frz_state = 0;
            else if (was) g_frz_state = 2;
            else if (!g_hoist_n && !g_fz.active && !defer_has_active() && !g_defer_replaying) g_frz_state = 1;   // 上一帧留下的前提 / 合并 / 推迟还没走完：下一帧再进 | The premises / merges / deferrals left by the previous frame have not finished yet: enter next frame
            if (g_frz_state != was) MLOG("世界冻结：通道 %llu 起 %s（完整帧累计 %llu，已跳过通道 %llu）", g_stat_pass + 1, g_frz_state == 1 ? "完整帧（不带界面模板）" : g_frz_state == 2 ? "冻结帧（只画界面与末尾合成）" : "恢复正常", g_frz_full_n += g_frz_state == 1, g_frz_skip_n);
        }
        g_frz_drop = 0;
        if (win) { g_frz_phase = 2; return 0; }
        if (!g_frz_state || g_frz_phase == 2) return 0;
        if (g_frz_phase == 0 && !r->n && r->depth && !g_frz_seen_ds) {   // 本帧第一个不带颜色的通道 | The first render pass without color in this frame
            g_frz_seen_ds = 1;
            if (g_frz_state == 2) g_frz_phase = 1;
            else { uint64_t s_; int lv, sl; rt_sub(r, 8, &s_, &lv, &sl); static int neg = -1; if (neg < 0) neg = getenv("DM_NVN_TEST_FRZ_NEG") != NULL;   // 负对照：完整帧照带界面模板画，面板缩小时必须露出旧像素 | Negative control: the full frame is drawn with the UI stencil as usual, and old pixels must be exposed when the panel shrinks
                   g_frz_drop = !neg && MAX(r->depth->d.width >> lv, 1u) != MAX(r->depth->d.height >> lv, 1u); }   // 完整帧：不是正方形（阴影图）的那个就是界面模板趟 | Full frame: the one that is not square (shadow map) is the UI stencil pass
        }
        if (g_frz_phase == 1) { g_frz_skip_n++; return 1; }
        return 0;
    }
    if (rt_want() && g_frz_phase != 2) return rt_op(op);
    if (g_frz_drop) return op == OP_DRAW || op == OP_DRAW_IDX;
    if (g_frz_state == 2 && g_frz_phase == 1) return op == OP_CLEAR_COLOR || op == OP_CLEAR_DS || op == OP_DRAW || op == OP_DRAW_IDX || op == OP_COPY_T2T || op == OP_CLEAR_TEX || op == OP_DISPATCH;
    return 0;
}

void mtl_exec(const uint8_t *p) {
    static int inited; if (!inited) { inited = 1; state_reset(); }
    static int dump_all = -1; if (dump_all < 0) dump_all = getenv("DM_NVN_PASSDUMP_ALL") != NULL;
    { static long st_frame = -2, st_ms; if (st_frame == -2) { const char *e = getenv("DM_NVN_STALL"); st_frame = e ? atol(e) : -1; st_ms = e && strchr(e, ':') ? atol(strchr(e, ':') + 1) : 1000; }   // TOTK diagnostic: DM_NVN_STALL=frame:ms sleeps the replay thread once at the start of that frame's replay
      static long st_done = -1; long cf = (long)atomic_load(&g_cur_frame); if (st_frame >= 0 && cf == st_frame && st_done != cf) { st_done = cf; MLOG("[stall] frame %ld: sleeping %ld ms on the replay thread", cf, st_ms); usleep((useconds_t)(st_ms * 1000)); } }
    int trace = dump_all && passdump();   // 普通模式不逐操作查帧范围，完整命令位置仅用于指定诊断帧 | Normal mode does not check the frame range per operation; the full command position is only used for the designated diagnostic frames
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (h.size < 8 || (h.size & 7) || h.op >= OP_COUNT_) MDIE("corrupt command record at %p: op %u size %u (call depth %d)", (const void *)p, (unsigned)h.op, (unsigned)h.size, g_ret_n);
        g_exec_cur = p;   // 写回前瞻的扫描起点：结束通道的那条操作（含） | Scan start point of the store lookahead: the operation that ends the render pass (inclusive)
        if (p == g_defer_pend) { for (int gi = 0; gi < 4; gi++) if (g_defer_slots[gi].active && g_defer_slots[gi].p_end_op == p) g_defer_slots[gi].in_p = 0; g_defer_pend = NULL; }
        sdx_seq(p);
        if (h.op == OP_SET_RT) rt_toggle_poll();
        if ((g_frz_on || g_frz_state || g_dm_rt_ab == 1) && frz_op(h.op, p + 8)) { p += h.size; continue; }
        if (trace) PD("o cmd=%p pass=%llu op=%u size=%u", p, g_stat_pass, (unsigned)h.op, (unsigned)h.size);
        if (g_cg_pending && !cg_op_safe(h.op)) { g_cg.reason[1]++; cg_flush(); }   // TOTK: cube-face-set groups pend only across state / binding ops (nvn_cube_group.h)
        if (g_hoist_n && hoist_take(p)) { if (g_fz.active && p == g_fz.rt2_op) { g_fz.phase = 2; g_rt_merged = 0; PD("F %llu 合并通道进入后一个通道的绘制", g_stat_pass); } p += h.size; continue; }   // 已提前执行的纯清除通道 / 已并入的后一个通道开头   // 进入后一个通道：前一个通道的同目标续用到此为止，后一个通道读前一个通道附件由 fz_p2_draw 证本像素，不再按续用规则切开 | A pure-clear render pass already executed early / the start of a later pass already merged in   // Entering the later pass: the same-target continuation of the earlier pass ends here; the later pass reading the earlier pass's attachment is proven to be same-pixel by fz_p2_draw, and is no longer split by the continuation rule
        const void *a = p + 8;
        switch (h.op) {
            case OP_END: g_exec_cur = NULL;
                if (g_hoist_n && !g_ret_n && !g_nrest) {
                    if (g_fz.active) fz_unskip();
                    if (g_hoist_n) { MLOG("【红】纯清除通道前提：提交结束时还有 %d 条已前提的操作没走到，清空（不应发生）", g_hoist_n); g_hoist_n = 0; }
                }
                return;   // 列表外（出图 / CPU 读回等）触发的结束通道不得扫已过期的命令内存：起点为空即全部写回 | An end-of-pass triggered outside a list (present / CPU readback etc.) must not scan expired command memory: a null start point means store everything
            case OP_BEGIN: break;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8);
                if (g_ret_n < 16) g_ret[g_ret_n] = p + h.size; g_ret_n++;   // 前瞻用的返回栈：子列表结束后接着看调用方 | Return stack for lookahead: after a sub-list ends, continue looking at the caller
                mtl_exec((const uint8_t *)t); g_ret_n--; break; }
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, (const uint8_t *)a + 8, 4); memcpy(&v, (const uint8_t *)a + 12, 4);
                PD("Y signal_event cmd=%p dst=%p mode=%u value=%u", p, val, m, v); mtl_sub_event(val, m, v); break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, (const uint8_t *)a + 8, 8); g_cnt_report[t & 31]++;
                PD("Q report type=%llu cmd=%p dst=%#llx", (unsigned long long)t, p, (unsigned long long)ad);
                cpend_push(ad, g_fence + 1, 0, t != NVN_COUNTER_TYPE_SAMPLES_PASSED);   // TOTK: value the completion handler will write (counter word)
                if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) {
                    post_op q = {3, NULL, 0, 0, ad, (uint32_t)t, vis_snapshot()};
                    if ((long)atomic_load(&g_cur_frame) <= atomic_load(&g_ql_until)) {   // "querylog": the proxy draw's sphere (vertex uniform block 1)
                        const p_bind_buf *u = &S.ubo[0][1]; q.qlog = 1;
                        if (u->addr && u->size >= 16) pend_read(u->addr, q.sph, sizeof q.sph, 0); else q.sph[3] = -1;
                    }
                    post(q); g_vis_cur = -1; }
                else mtl_sub_counter(ad, (uint32_t)t);
                break; }
            case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); g_cnt_reset[t & 31]++; PD("Q reset type=%llu cmd=%p", (unsigned long long)t, p);
                if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) {
#ifdef NVN_RT_RANGE_TEST
                    rr_reset();
#endif
                    g_vis_on = 1; g_vis_counting = 1; vis_clear(); }
                break; }
            case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, (const uint8_t *)a + 8, 8); memcpy(&n, (const uint8_t *)a + 16, 8);
                PD("Y copy_b2b cmd=%p src=%#llx dst=%#llx bytes=%llu", p, (unsigned long long)s_, (unsigned long long)d, (unsigned long long)n); mtl_sub_copy(s_, d, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, (const uint8_t *)a + 8, 8);
                PD("Y update_ub cmd=%p src=%#llx dst=%#llx bytes=%llu", p, (unsigned long long)((const uint8_t *)a + 16), (unsigned long long)d, (unsigned long long)n); mtl_sub_update((const uint8_t *)a + 16, d, n); break; }
            case OP_SET_RT: {
                const p_set_rt *r = a;
                g_cg.rt_seq++; if (g_cg_pending && !cg_rt_match(r)) { g_cg.reason[0]++; cg_flush(); }   // TOTK: a target that is not a cube face set lands the pending groups
                if (defer_begin_q(p)) break;
                int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
                if (g_enc && g_lay.active && !pend && lay_continue(r)) break;   // TOTK: another layer of the layered pass
                if (g_enc && !pend && R.has_rt && tbdr_on() && rt_range_same(&R.rt,r)) {
                    g_rt_merged = g_rt_range_active = g_rt_range_pending = 1; R.rt = *r; break; }
                if (g_enc && g_fz.active && g_fz.phase == 1 && fz_reuse_op(p) && !pend && g_enc_mode != 1 && R.has_rt && tbdr_on() && rt_same(&g_fz.rt1, r)) {
                    g_rt_merged = 1; if (!g_rt_merge_n++) MLOG("同目标续用：首例 通道 %llu", g_stat_pass); PD("M %llu 同目标续用通道", g_stat_pass); break; }
                // 不续用只为清除开的通道；少挂深度 / 模板的可续用（绘制前核平面）
                // Do not continue a render pass opened only for a clear; one with fewer depth / stencil attachments can be continued (check the planes before drawing)
                /* TOTK: continuing the pass on a same-target SetRenderTargets no longer needs the rest of the TBDR rewrites
                   (DM_NVN_TBDR stays off): TOTK re-sets identical targets ~14 times per frame, and each new Metal render pass
                   costs ~50 us of GPU time on the M4. The checks in draw_inner (missing depth/stencil plane, sampling an
                   attachment) and in the clear ops still end the continued pass where needed. TOTK_RT_CONTINUE=0 turns it off. */
                static int rt_cont = -1; if (rt_cont < 0) { const char *e = getenv("TOTK_RT_CONTINUE"); rt_cont = !(e && !strcmp(e, "0")); }
                if (g_enc && g_enc_mode != 1 && !pend && R.has_rt && (tbdr_on() || rt_cont) && rt_same(&R.rt, r)) { g_rt_merged = 1; if (!g_rt_merge_n++) MLOG("同目标续用：首例 通道 %llu", g_stat_pass); PD("M %llu 同目标续用通道", g_stat_pass); R.rt = *r; break; }
                rp_flush(); R.rt = *r; R.has_rt = 1; break;
            }
            case OP_CLEAR_COLOR: {
                const p_clear_color *c = a;
                if (S.render_disabled) { g_draw_skip[12]++; PD("E clear c%d skipped: rendering disabled", c->index); break; }   // TOTK: SetRenderEnable(false) / conditional
                if (g_rt_merged) rp_end();   // 续用的通道里来了清除：在此结束，清除照旧挂到下一个通道开头 | A clear arrived in a continued render pass: end it here, and the clear is attached to the start of the next render pass as usual
                if (c->index < 0 || c->index >= R.rt.n || !R.rt.color[c->index]) { RED_ONCE(5, "清色目标 %d 无效", c->index); break; }
                if (g_enc || c->mask != 0xf || !scissor_full(R.rt.color[c->index], &R.rt.cview[c->index], R.rt.has_cview[c->index])) { if ((c->mask & 0xf) && !S.render_disabled) clear_quad(c); break; }
                R.clear_c[c->index] = 1; R.cc[c->index] = MTLClearColorMake(c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]);
                break;
            }
            case OP_CLEAR_DS: {
                const p_clear_ds *c = a;
                if (S.render_disabled) { g_draw_skip[12]++; PD("E clear ds skipped: rendering disabled"); break; }   // TOTK
                if (g_rt_merged) rp_end();
                if (!R.rt.depth || (!c->depth_mask && !(c->stencil_mask & 255))) break;
                if (g_enc || ((c->stencil_mask & 255) && (c->stencil_mask & 255) != 255) || !scissor_full(R.rt.depth, &R.rt.dview, R.rt.has_dview)) { clear_ds_quad(c); break; }
                if (c->depth_mask) { R.clear_d = 1; R.cd = c->depth; }
                if (c->stencil_mask) { R.clear_s = 1; R.cs = (uint32_t)c->stencil; }
                break;
            }
            case OP_DRAW: g_exec_next = p + h.size; draw(a, NULL); g_exec_next = NULL; break;   // 前瞻起点 | Lookahead start point
            case OP_DRAW_IDX: g_exec_next = p + h.size; draw(NULL, a); g_exec_next = NULL; break;
            // 状态没变就不置脏（游戏每次绘制前常重发同值的程序 / 状态块 / 顶点格式；脏标志只决定是否重取管线 / 深度模板对象，值相同则对象相同）
            // Do not mark dirty if the state did not change (the game often re-sends the same program / state block / vertex format before each draw; the dirty flag only decides whether to re-fetch the pipeline / depth-stencil object, and equal values mean the same object)
            case OP_SCISSOR: case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
            case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_STENCIL:
            case OP_POLY_OFFSET: case OP_VIEWPORTS: case OP_DEPTH_RANGE: case OP_BLEND_COLOR: case OP_COPY_STRIDE: case OP_DEPTH_CLAMP: case OP_RENDER_ENABLE: case OP_RASTER_DISCARD: case OP_RENDER_COND:
                state_op(h.op, a); break;
            case OP_COPY_B2T: copy_b2t(a); break;
            case OP_COPY_T2B: copy_t2b(a); break;
            case OP_COPY_T2T: copy_t2t(a); break;
            case OP_CLEAR_TEX: clear_tex(a); break;
            case OP_DISPATCH: dispatch(a); break;
            case OP_BARRIER: {
#if DM_NVN_M3
                int32_t bits;memcpy(&bits,a,sizeof bits);
                /* Automatic Metal 3 hazard tracking orders separate encoders;
                   it does not make attachment writes visible to texture reads
                   inside a continued render pass. Preserve explicit NVN edges. */
                if(bits&(NVN_BARRIER_ORDER_PRIMITIVES_BIT|NVN_BARRIER_ORDER_FRAGMENTS_BIT|
                         NVN_BARRIER_ORDER_INDIRECT_DATA_BIT|NVN_BARRIER_INVALIDATE_TEXTURE_BIT|
                         NVN_BARRIER_INVALIDATE_SHADER_BIT)){
                    rp_flush();blit_close();rs_sync();args_dirty_all();
                }
#endif
                break;
            }
            case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: break;
            default: break;
        }
        p += h.size;
    }
}

// ---------- 出图：交换链贴图画到 CAMetalLayer；可选截图 ----------
// ---------- Present: draw the swapchain texture to the CAMetalLayer; optional screenshot ----------
static id<MTLRenderPipelineState> g_blit_pso; static MTLPixelFormat g_blit_fmt;
static id<MTLSamplerState> g_blit_samp;
static uint64_t g_present_n;
uint64_t mtl_present_count(void) { return g_present_n; }   // 实际出图次数（发热归因 DM_HEAT 算出图帧率） | Actual present count (heat attribution DM_HEAT computes the present frame rate)
void mtl_res_snapshot(mtl_res_count *out) {
    memset(out, 0, sizeof *out);
    out->pso = g_pso.n;
    out->cps = g_cps.n;
    out->dss = g_dss.n;
    out->shaders = (uint32_t)(g_shcache ? g_shcache.count : 0);
    os_unfair_lock_lock(&g_c1_lock);
    out->c1_bufs = (uint32_t)(g_c1_banks.count + g_c1_exclusive); out->c1_alloc = g_c1_alloc_bytes; out->c1_data = g_c1_data_bytes;
    os_unfair_lock_unlock(&g_c1_lock);
    pthread_mutex_lock(&g_smu);
    out->live_tex = g_by_serial.n;
    pthread_mutex_unlock(&g_smu);
    pthread_mutex_lock(&g_mu);
    out->pools = (uint32_t)g_pools.n;
    out->pool_adds = g_pool_adds;
    pthread_mutex_unlock(&g_mu);
    registry_lock();
    out->view_slots = (uint32_t)g_ntreg;
    for (int i = 0; i < g_ntreg; i++) if (g_treg[i].view) out->view_nonnull++;
    for (int i = 0; i < g_nsreg; i++) if (g_sview[i]) out->sview_nonnull++;
    registry_unlock();
}
// 资源计数（长时间稳定性 L3 读）：宿主按游戏拍 / 事件调用（每 250 拍、读取画面已撤、退出），不按墙钟 / 出图数取样。
// 行格式：[nvn] 资源计数 拍 <拍号> <触发>：<名> <数值> …（名不含空格，单空格分隔）；加字段就往这一行后面追加
// Resource counts (read by long-run stability L3): the host calls it per game tick / event (every 250 ticks, loading screen dismissed, exit), not sampled by wall clock / present count.
// Line format: [nvn] 资源计数 拍 <拍号> <触发>：<名> <数值> … ([nvn] resource counts tick <tick number> <trigger>: <name> <value> ...) (names contain no spaces, single-space separated); new fields are appended to the end of this line
void mtl_res_counts(uint32_t tick, const char *why) {
    mtl_res_count c;
    mtl_res_snapshot(&c);
    int dl, du; uint64_t da; nvn_dirty_slot_stats(&dl, &du, &da);
    LOG("[nvn] 资源计数 拍 %u %s：渲染管线 %u 计算管线 %u 着色器 %u 深度模板 %u view非空 %u view槽 %u 存活贴图 %u 内存池存活 %u 内存池累计 %llu 采样器view非空 %u 脏页区间存活 %d 脏页区间累计 %llu c1缓冲 %u c1字节 %llu c1实际 %llu",
        tick, why, c.pso, c.cps, c.shaders, c.dss, c.view_nonnull, c.view_slots, c.live_tex,
        c.pools, (unsigned long long)c.pool_adds, c.sview_nonnull, dl, (unsigned long long)da, c.c1_bufs, (unsigned long long)c.c1_alloc, (unsigned long long)c.c1_data);
}
// 按请求截图（DM_NVN_SHOT_UI、TOTK_INPUT_FILE 的 shot 用）：dm_gfx_shot_at(k, 路径) = 在当前游戏帧之后第 k 帧截图，在游戏出图调用
// （nvnQueuePresentTexture，mtl_shot_at_present）里检查到期请求。不按上屏计数：测试窗口被挡住时上屏停住，游戏照跑，请求会永远等不到（经验第 41 条）
// Screenshot on request (used by DM_NVN_SHOT_UI and the TOTK_INPUT_FILE shot command): dm_gfx_shot_at(k, path) = take a screenshot k game frames
// after the current one; due requests are checked at the game's present call (nvnQueuePresentTexture, mtl_shot_at_present). Not keyed on the
// present count: an occluded test window stalls presents while the game keeps running, so such a request would never come due (lesson 41)
#define SHOTQ_MAX 64
static struct { uint64_t frame; char path[1024]; } g_shotq[SHOTQ_MAX]; static int g_nshotq;
static pthread_mutex_t g_shotq_mu = PTHREAD_MUTEX_INITIALIZER;
static int present_encode(id<MTLCommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst, const int crop[4], int fps_overlay);
static int present_encode_m4(id<MTL4CommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst, const int crop[4], int uv_i);
// 整窗截图要的游戏画面（前端 window_shot，不再对挂着显示节拍的 Metal 图层做 drawViewHierarchy）：下一次上屏时，出图线程把同一帧
// 按上屏同一套编码（present_encode）画进一张与交换链同尺寸的离屏贴图，GPU 做完后在完成回调里交出 CGImage（回调不阻塞上屏）。g_shotq_mu 护
// Game picture needed by the whole-window screenshot (front end window_shot; no longer does drawViewHierarchy on a Metal layer with the display link attached): at the next present, the present thread draws the same frame
// with the same encoding as the present (present_encode) into an offscreen texture of the same size as the swapchain, and after the GPU finishes hands out a CGImage in the completion callback (the callback does not block the present). Guarded by g_shotq_mu
static void (^g_grab_cb)(CGImageRef img);
void mtl_grab_next_present(void (^cb)(CGImageRef img)) { pthread_mutex_lock(&g_shotq_mu); g_grab_cb = [cb copy]; pthread_mutex_unlock(&g_shotq_mu); }
/* The window picture re-rendered offscreen by the legacy encoder: only for presents that draw no drawable
   (TOTK_PRESENT_SKIP measurement runs); a real present copies its drawable (present_now). */
static void grab_present(void (^cb)(CGImageRef), id<MTLTexture> src, const int crop[4], CGSize sz) {
    if (!cb) return;
    size_t w = (size_t)sz.width, h = (size_t)sz.height, row = w * 4;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> out = w && h ? [g_dev newTextureWithDescriptor:d] : nil; id<MTLBuffer> buf = out ? [g_dev newBufferWithLength:row * h options:MTLResourceStorageModeShared] : nil;
    id<MTLCommandBuffer> cb2 = buf ? q_cb() : nil;
    if (!cb2 || !present_encode(cb2, src, out, crop, 1)) { MLOG("【红】整窗截图：游戏画面离屏编码失败（%zux%zu）", w, h); cb(NULL); return; }
    id<MTLBlitCommandEncoder> b = [cb2 blitCommandEncoder];
    [b copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1) toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:row * h];
    [b endEncoding];
    [cb2 addCompletedHandler:^(id<MTLCommandBuffer> c) { (void)c;
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef ctx = CGBitmapContextCreate(buf.contents, w, h, 8, row, cs, kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
        CGImageRef img = ctx ? CGBitmapContextCreateImage(ctx) : NULL; cb(img);
        CGImageRelease(img); CGContextRelease(ctx); CGColorSpaceRelease(cs); }];
    cb_commit(cb2);
}
/* TOTK_WINDOW_SHOT=n[,n...]:path (%d = present count): saves the picture as shown in the window, FPS counter included,
   rendered by the same encoder into an offscreen texture of the drawable's size. Diagnostic; independent of whether the
   window is visible on the current Space (screencapture returns an empty window then). */
static void window_shot_save(CGImageRef img, NSString *path) {
    CFURLRef url = img ? CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)path.UTF8String, (CFIndex)strlen(path.UTF8String), false) : NULL;
    CGImageDestinationRef dst = url ? CGImageDestinationCreateWithURL(url, (__bridge CFStringRef)UTTypePNG.identifier, 1, NULL) : NULL;
    int ok = 0;
    if (dst) { CGImageDestinationAddImage(dst, img, NULL); ok = CGImageDestinationFinalize(dst); }
    if (dst) CFRelease(dst);
    if (url) CFRelease(url);
    if (ok) MLOG("整窗截图 %s 成功", path.UTF8String); else MLOG("【红】整窗截图 %s 失败（取图或写文件）", path.UTF8String);
}
static void window_shot_tick(uint64_t present_n) {
    static long at[16]; static int n = -1; static const char *path;
    if (n < 0) {
        n = 0; const char *e = getenv("TOTK_WINDOW_SHOT"), *c = e ? strchr(e, ':') : NULL;
        if (c) { path = c + 1; for (const char *q = e; q < c && n < 16; q = strchr(q, ',') ? strchr(q, ',') + 1 : c) at[n++] = atol(q); }
        else if (e) MLOG("【红】TOTK_WINDOW_SHOT 需要 序号[,序号]:路径");
    }
    for (int i = 0; i < n; i++) if ((long)present_n == at[i]) {
        char out[1024]; snprintf(out, sizeof out, path, (int)at[i]);
        NSString *p = @(out);
        mtl_grab_next_present(^(CGImageRef img) { window_shot_save(img, p); });
    }
}
/* "wshot <path>" (TOTK_INPUT_FILE / the desktop command channel): the next game frame as the window shows it. With
   TOTK_WSHOT_SRC=1 (tests) the same present's source picture (the game's frame before the output upscale) goes to
   <path>.src.png as well, so the 2K output can be compared with what it was made from, pixel for pixel. */
static char g_grab_src_path[1024];
void dm_gfx_window_shot(const char *path) {
    NSString *p = @(path);
    if (getenv("TOTK_WSHOT_SRC")) { pthread_mutex_lock(&g_shotq_mu); snprintf(g_grab_src_path, sizeof g_grab_src_path, "%s.src.png", path); pthread_mutex_unlock(&g_shotq_mu); }
    mtl_grab_next_present(^(CGImageRef img) { window_shot_save(img, p); });
}
void dm_gfx_shot_at(long k, const char *path) {
    pthread_mutex_lock(&g_shotq_mu);
    if (g_nshotq < SHOTQ_MAX) { g_shotq[g_nshotq].frame = atomic_load(&g_cur_frame) + (uint64_t)(k > 0 ? k : 0); strlcpy(g_shotq[g_nshotq].path, path, sizeof g_shotq[0].path); g_nshotq++; }
    else MLOG("【红】截图请求超过 %d 个，丢弃 %s", SHOTQ_MAX, path);
    pthread_mutex_unlock(&g_shotq_mu);
}
// DM_NVN_SHOT_OUT=宽x高（画质度量用）：出图帧截图改截「经 present_encode 放大到该分辨率」的结果，即玩家在屏幕上看到的画面；screen=0（贴图转储）不受影响
// DM_NVN_SHOT_OUT=WIDTHxHEIGHT (for picture quality metrics): the present-frame screenshot instead captures the result "upscaled to this resolution by present_encode", i.e. the picture the player sees on screen; screen=0 (texture dump) is unaffected

/* TOTK: frame-keyed screenshots taken at the game's present call (nvnQueuePresentTexture), independent of the display
   link / drawable pacing (an asleep or locked display stalls presents; the game keeps running). DM_NVN_SHOT_FRAME=1 with
   DM_NVN_SHOT=frame,...:path. */
void mtl_shot_at_present(const tex_t *t) {
    static long shots[16]; static int nshot = -1; static const char *shot_path; static char done[16]; static int by_frame;
    if (nshot < 0) {
        nshot = 0; const char *e = getenv("DM_NVN_SHOT"), *c = e ? strchr(e, ':') : NULL; const char *bf = getenv("DM_NVN_SHOT_FRAME"); by_frame = bf && atoi(bf);
        if (c && by_frame) { shot_path = c + 1; for (const char *q = e; q < c && nshot < 16; q = strchr(q, ',') ? strchr(q, ',') + 1 : c) shots[nshot++] = atol(q); }
    }
    long cur = (long)atomic_load(&g_cur_frame);
    pthread_mutex_lock(&g_shotq_mu);   // dm_gfx_shot_at requests (game-frame keyed, see there)
    for (int i = 0; i < g_nshotq;) if ((uint64_t)cur >= g_shotq[i].frame) {
        id<MTLTexture> src = tex_mtl(t);
        if (src) { mtl_finish_gpu(); shot(src, g_shotq[i].path, 1, NULL); MLOG("[shot] F%ld %s", cur, g_shotq[i].path); } else MLOG("【红】截图：出图纹理不可用，丢弃 %s", g_shotq[i].path);
        g_shotq[i] = g_shotq[--g_nshotq];
    } else i++;
    pthread_mutex_unlock(&g_shotq_mu);
    if (!nshot) return;
    for (int i = 0; i < nshot; i++) if (!done[i] && cur >= shots[i]) {
        done[i] = 1; id<MTLTexture> src = tex_mtl(t); if (!src) { MLOG("【红】截图：出图纹理不可用"); continue; }
        char path[1024]; snprintf(path, sizeof path, shot_path, (int)shots[i]); mtl_finish_gpu(); shot(src, path, 1, NULL);
    }
}
static void shot_body(id<MTLTexture> src, const char *path, int screen, const int crop[4]);
/* The command buffers and the read-back buffer are autoreleased objects: screenshots are taken on the game's present
   thread, which has no autorelease pool, so without this pool every shot kept its ~8 MB buffer (TOTK, 2026-10-15:
   255 shots in one session reached the 5000 MiB GPU budget). */
static void shot(id<MTLTexture> src, const char *path, int screen, const int crop[4]) { @autoreleasepool { shot_body(src, path, screen, crop); } }
static void shot_body(id<MTLTexture> src, const char *path, int screen, const int crop[4]) {
    static int out_w = -1, out_h;
    if (out_w < 0) {
        out_w = 0;
        const char *e = getenv("DM_NVN_SHOT_OUT"); int w = 0, h = 0; char sep = 0, extra = 0;
        if (e && sscanf(e, "%d%c%d%c", &w, &sep, &h, &extra) == 3 && sep == 'x' && w > 0 && h > 0) { out_w = w; out_h = h; }
    }
    id<MTLTexture> read = src;
    if (out_w && screen) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:out_w height:out_h mipmapped:NO];
        d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
        id<MTLTexture> out = [g_dev newTextureWithDescriptor:d];
        if (!out) { MLOG("【红】截图输出贴图建不出 %dx%d", out_w, out_h); return; }
        id<MTLCommandBuffer> cb = q_cb();
        if (!present_encode(cb, src, out, crop, 0)) return;
        cb_commit(cb); cb_wait(cb);
        read = out;
    }
    if (read.pixelFormat != MTLPixelFormatRGBA8Unorm && read.pixelFormat != MTLPixelFormatRGBA8Unorm_sRGB &&
        read.pixelFormat != MTLPixelFormatBGRA8Unorm && read.pixelFormat != MTLPixelFormatBGRA8Unorm_sRGB) { MLOG("【红】截图不支持格式 %lu", (unsigned long)read.pixelFormat); return; }
    size_t w = read.width, h = read.height, row = w * 4;
    id<MTLBuffer> buf = [g_dev newBufferWithLength:row * h options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = q_cb();
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromTexture:read sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
              toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:row * h];
    [b endEncoding]; cb_commit(cb); cb_wait(cb);
    int bgra = read.pixelFormat == MTLPixelFormatBGRA8Unorm || read.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB;
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGBitmapInfo bi = bgra ? (kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little) : (CGBitmapInfo)kCGImageAlphaNoneSkipLast;
    CGContextRef ctx = CGBitmapContextCreate(buf.contents, w, h, 8, row, cs, bi);
    CGImageRef img = ctx ? CGBitmapContextCreateImage(ctx) : NULL;
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)path, strlen(path), false);
    CGImageDestinationRef dst = url ? CGImageDestinationCreateWithURL(url, (__bridge CFStringRef)UTTypePNG.identifier, 1, NULL) : NULL;
    // 任一步为空（目录不存在 / 不可写、位图上下文建不出）报红返回；原先直接 CFRelease(NULL) 崩在 CFRelease.cold（1952 宽 TEXDUMP 时撞到）
    // If any step yields null (directory does not exist / not writable, bitmap context cannot be created), report a failure and return; previously it went straight to CFRelease(NULL) and crashed in CFRelease.cold (hit during a 1952-wide TEXDUMP)
    int ok = 0;
    if (!ctx || !img || !dst) MLOG("【红】截图 %s 失败：%s", path, !ctx ? "位图上下文建不出" : !img ? "取图失败" : "写不出（目录不存在或不可写）");
    else { CGImageDestinationAddImage(dst, img, NULL); ok = CGImageDestinationFinalize(dst); }
    if (dst) CFRelease(dst); if (url) CFRelease(url); CGImageRelease(img); CGContextRelease(ctx); CGColorSpaceRelease(cs);
    dm_shot_written(path, ok);   // 先让宿主写「冻结拍」行，再写「成功」行：截图脚本见到「成功」就停宿主，顺序反了满载时日志会缺「冻结拍」行（闸门 require_log 误红） | Let the host write the 「冻结拍」(frozen tick) line first, then write the 「成功」(success) line: the screenshot script stops the host as soon as it sees 「成功」(success), and with the order reversed the log would lack the 「冻结拍」(frozen tick) line under full load (false failure in the gate check's require_log)
    MLOG("截图 %zux%zu 格式 %lu → %s %s", w, h, (unsigned long)read.pixelFormat, path, ok ? "成功" : "失败");
    if (g_ruler_truth && getenv("DM_NVN_RULER") && !strcmp(getenv("DM_NVN_RULER"), "1") && ok) { static int shots; if (++shots == 2) g_ruler_pending = 1; }
    if (g_defer_ab >= 0 && getenv("DM_NVN_DEFER_AB") && !strcmp(getenv("DM_NVN_DEFER_AB"), "1") && ok) { static int shots; if (++shots == 2) { g_defer_ab_on = 1; MLOG("DM_NVN_DEFER_AB：第二张截图写盘完成，打开推迟合并通道"); } }
    if (g_udc_ab >= 0 && getenv("DM_NVN_UDC_AB") && !strcmp(getenv("DM_NVN_UDC_AB"), "1") && ok) {
        static int shots;
        if (++shots == 2) { g_udc_ab_on = 1; MLOG("DM_NVN_UDC_AB：第二张截图写盘完成，打开 DM_NVN_UNWRITTEN_DC"); }
    }
}
// ---------- DM_NVN_TEXDUMP=帧号:路径（路径里 %d 换成「序号*100+mip 级」）：把该帧用过的颜色渲染目标逐张存图 ----------
// 诊断用：在下一帧第一次开通道前等 GPU 做完，再把每张（取该帧用到的 mip 级）全屏采样画进 RGBA8 存 PNG（浮点格式按 [0,1] 截断）
// ---------- DM_NVN_TEXDUMP=frame number:path (%d in the path is replaced with "index*100+mip level"): save each color render target used by that frame as an image ----------
// Diagnostic: before the first render pass of the next frame is opened, wait for the GPU to finish, then draw each one (the mip level used by that frame) by full-screen sampling into RGBA8 and save as PNG (float formats are clamped to [0,1])
static _Atomic long g_td_frame = -2; static const char *g_td_path; static NSMutableArray *g_td_list; static int g_td_done;
static _Atomic int g_td_auto_armed;
// TOTK: DM_NVN_TEXPNG_FS=片元着色器键前缀：记下该着色器实际绑定过的贴图序号，帧转储时连同各级 mip 一起存成 PNG（序号每次运行都变，按着色器选更稳）
// TOTK: DM_NVN_TEXPNG_FS=fragment shader key prefix: remember the serials of textures that shader actually bound, and dump them with every mip level at frame dump time (serials change per run; selecting by shader is stable)
static uint64_t g_tp_ser[64]; static tex_t g_tp_tex[64]; static int g_tp_n; static const char *g_tp_fs; static int g_tp_init;
static long g_tp_only = -1;   // TOTK: an on-demand dump ("texdump") records the shader's bindings of the dump frame only
static int tp_fs_match(const char *key) {   // DM_NVN_TEXPNG_FS may list several key prefixes: a,b,c
    uint64_t fs = atomic_load(&g_diag_texpng_fs);
    if (fs && diag_shader_prefix_matches(key, fs)) return 1;
    for (const char *q = g_tp_fs; q && *q; q = strchr(q, ',') ? strchr(q, ',') + 1 : NULL) {
        size_t n = strchr(q, ',') ? (size_t)(strchr(q, ',') - q) : strlen(q);
        if (n && !strncmp(key, q, n)) return 1;
    }
    return 0;
}
static void tp_note_fs(const char *key, uint64_t ser, const tex_t *t) {
    if (!g_tp_init) { g_tp_init = 1; g_tp_fs = getenv("DM_NVN_TEXPNG_FS"); }
    if (!key || (!g_tp_fs && !atomic_load(&g_diag_texpng_fs)) || !tp_fs_match(key)) return;
    if (g_tp_only >= 0 && (long)atomic_load(&g_cur_frame) != g_tp_only) return;
    for (int i = 0; i < g_tp_n; i++) if (g_tp_ser[i] == ser) return;
    if (g_tp_n < 64) { g_tp_tex[g_tp_n] = *t; g_tp_ser[g_tp_n++] = ser; }
}
static void td_dump(void);
/* TOTK: TOTK_INPUT_FILE command "texdump <path>" (%d as in DM_NVN_TEXDUMP): the same dump for the frame after the current
   one, requested while the game runs (the place to look at is found by playing). The request is armed by the replay
   thread at its next render pass; DM_NVN_TEXPNG_FS then records that frame's bindings only, and DM_NVN_TEXSNAP(_MIN)
   snapshots apply to that frame. */
static char *_Atomic g_td_req;
void dm_gfx_texdump_next(const char *path) {
    char *p = strdup(path); if (!p) return;
    free(atomic_exchange(&g_td_req, p));
    MLOG("[texdump] 按输入文件转储下一帧 → %s", p);
}
void td_note(MTL4RenderPassDescriptor *pd, NSUInteger nc) {
    if (g_td_frame == -2) { const char *e = getenv("DM_NVN_TEXDUMP"), *c = e ? strchr(e, ':') : NULL; g_td_frame = c ? atol(e) : -1; g_td_path = c ? c + 1 : NULL; g_td_list = [NSMutableArray array]; }
    { char *req = atomic_exchange(&g_td_req, NULL);
      if (req) { long f1 = (long)atomic_load(&g_cur_frame) + 1; g_td_path = req; [g_td_list removeAllObjects]; g_tp_n = 0; g_tp_only = f1; g_td_done = 0;
                 atomic_store(&g_td_frame, f1); MLOG("[texdump] 已布置：帧 %ld", f1); } }
    if (g_td_frame < 0 || g_td_done) return;
    long f = (long)atomic_load(&g_cur_frame);
    if (f > g_td_frame) { td_dump(); return; }
    if (f != g_td_frame) return;
    for (NSUInteger i = 0; i < nc && i < 8; i++) {
        id<MTLTexture> t = pd.colorAttachments[i].texture; if (!t) continue;
        uint64_t ser = i < (NSUInteger)R.rt.n && R.rt.color[i] ? R.rt.color[i]->serial : 0;
        NSUInteger lv = pd.colorAttachments[i].level;
        BOOL seen = NO; for (NSArray *e in g_td_list) if ([e[1] unsignedLongLongValue] == ser && [e[2] unsignedIntegerValue] == lv) seen = YES;
        if (!seen) [g_td_list addObject:@[t, @(ser), @(lv)]];
    }
}
static NSMutableArray *g_ts_made;   // DM_NVN_TEXSNAP copies made for this dump (resident until saved)
static void td_dump_body(void);
/* The dump runs on the replay thread, which has no autorelease pool of its own: without one the temporaries of a dump
   (descriptors, command buffers, images) stayed alive, and the snapshot copies stayed in the residency set, so every
   on-demand dump kept several hundred MB and a second one ran into the GPU budget. */
static void td_dump(void) {
    @autoreleasepool {
        td_dump_body();
        for (id<MTLTexture> c in g_ts_made) res_remove(c);
        [g_ts_made removeAllObjects];
    }
}
static void td_dump_body(void) {
    g_td_done = 1;
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    mtl_finish_gpu();
    static id<MTLRenderPipelineState> ps; static id<MTLSamplerState> sp;
    if (!ps) {
        static const char *src_msl =
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "struct V { float4 p [[position]]; float2 uv; };\n"
            "vertex V vs(uint id [[vertex_id]]) { float2 uv = float2((id << 1) & 2, id & 2); V o; o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = uv; return o; }\n"
            "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) { float4 c = t.sample(s, in.uv); return float4(c.rgb, 1); }\n"
            "fragment float4 fsa(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) { float4 c = t.sample(s, in.uv); return float4(c.a, c.a, c.a, 1); }\n";   // TOTK: DM_NVN_TEXDUMP_ALPHA=1 writes the alpha channel as grey
        NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src_msl) options:nil error:&e];
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [lib newFunctionWithName:@"vs"]; d.fragmentFunction = [lib newFunctionWithName:getenv("DM_NVN_TEXDUMP_ALPHA") ? @"fsa" : @"fs"];
        d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        ps = [g_dev newRenderPipelineStateWithDescriptor:d error:&e];
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sp = [g_dev newSamplerStateWithDescriptor:sd];
        if (!ps) { MLOG("【红】TEXDUMP 管线失败 %s", e.localizedDescription.UTF8String); return; }
    }
    MLOG("[texdump] 帧 %ld 共 %lu 张颜色目标", g_td_frame, (unsigned long)g_td_list.count);
    // DM_NVN_TEXRAW=序号,…：按贴图序号存原始字节（含 3D；文件 路径%d 换成序号、后缀 .raw，首行文本头：宽 高 深 格式 每行字节）
    // DM_NVN_TEXRAW=index,...: save raw bytes by texture index (including 3D; file: %d in the path is replaced with the index, suffix .raw, first-line text header: width height depth format bytes-per-row)
    for (const char *q = getenv("DM_NVN_TEXRAW"); q && *q; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) {
        uint64_t ser = (uint64_t)atoll(q); id o = serial_get(ser);
        id<MTLTexture> t = [o isKindOfClass:[NvPkg class]] ? ((NvPkg *)o)->tex : o;
        if (!t) { MLOG("[texraw] ser=%llu 不在序号表", ser); continue; }
        int bw, bh; NSUInteger w = t.width, h = t.height, dz = t.depth;
        size_t bpp = t.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : t.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : 4; (void)bw; (void)bh;
        size_t row = w * bpp, img = row * h;
        id<MTLBuffer> buf = [g_dev newBufferWithLength:img * dz options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, dz) toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:img];
        [b endEncoding]; cb_commit(cb); cb_wait(cb);
        char path[1024]; snprintf(path, sizeof path, g_td_path, (int)ser); strlcat(path, ".raw", sizeof path);
        FILE *f = fopen(path, "wb"); if (f) { fprintf(f, "%lu %lu %lu %lu %zu\n", (unsigned long)w, (unsigned long)h, (unsigned long)dz, (unsigned long)t.pixelFormat, row); fwrite(buf.contents, 1, img * dz, f); fclose(f); }
        MLOG("[texraw] ser=%llu %lux%lux%lu 格式 %lu → %s", ser, (unsigned long)w, (unsigned long)h, (unsigned long)dz, (unsigned long)t.pixelFormat, path);
    }
    // TOTK: DM_NVN_TEXPNG=序号,…：按贴图序号把被采样的贴图（含压缩格式）每一级 mip 用采样画成 PNG（文件序号 = 序号*100+mip 级）
    // TOTK: DM_NVN_TEXPNG=serial,...: render every mip level of the given sampled textures (compressed formats included) to PNG by sampling (file index = serial*100+mip level)
    uint64_t tp_list[128]; int tp_n = 0;
    for (const char *q = getenv("DM_NVN_TEXPNG"); q && *q && tp_n < 64; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) tp_list[tp_n++] = (uint64_t)atoll(q);
    for (int i = 0; i < g_tp_n; i++) tp_list[tp_n++] = g_tp_ser[i];
    for (int i = 0; i < tp_n; i++) {
        uint64_t ser = tp_list[i]; id o = serial_get(ser);
        id<MTLTexture> t = [o isKindOfClass:[NvPkg class]] ? ((NvPkg *)o)->tex : o;
        if (!t) { MLOG("[texpng] ser=%llu 不在序号表", ser); continue; }
        MLOG("[texpng] ser=%llu %lux%lu 格式 %lu mips=%lu 类型 %lu", ser, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.pixelFormat, (unsigned long)t.mipmapLevelCount, (unsigned long)t.textureType);
        if (t.textureType == MTLTextureType3D) {   // TOTK: 3D tables are dumped slice by slice (index = serial*100 + z, z < 100)
            for (NSUInteger z = 0; z < t.depth && z < 100; z++) {
                MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat width:t.width height:t.height mipmapped:NO];
                d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate; id<MTLTexture> c2 = [g_dev newTextureWithDescriptor:d];
                id<MTLCommandBuffer> cb2 = q_cb(); id<MTLBlitCommandEncoder> bb = [cb2 blitCommandEncoder];
                [bb copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, z) sourceSize:MTLSizeMake(t.width, t.height, 1) toTexture:c2 destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                [bb endEncoding]; cb_commit(cb2); cb_wait(cb2);
                [g_td_list addObject:@[c2, @(ser * 100 + z), @0]]; }   // file index = (serial*100+z)*100
            continue; }
        for (int j = 0; j < g_tp_n; j++) if (g_tp_ser[j] == ser) { const tex_t *x = &g_tp_tex[j]; uint64_t nz = 0; if (x->mem) for (uint64_t k = 0; k < x->L.size; k += 8) if (*(const uint64_t *)(x->mem + k)) nz++;
            MLOG("[texpng]   nvnfmt=%#x flags=%#x has_layout=%d mem=%p size=%llu nz8=%llu/%llu pool=%p pflags=%#x phost=%p off=%lld", x->d.format, x->d.flags, x->has_layout, (void *)x->mem, (unsigned long long)x->L.size, (unsigned long long)nz, (unsigned long long)(x->L.size / 8), (void *)x->pool, x->pool ? x->pool->flags : 0, x->pool ? (void *)x->pool->host : NULL, (long long)x->off); }
        for (NSUInteger lv = 0; lv < t.mipmapLevelCount; lv++) [g_td_list addObject:@[t, @(ser), @(lv)]];
    }
    for (NSArray *e in g_td_list) {
        id<MTLTexture> t = e[0]; uint64_t ser = [e[1] unsignedLongLongValue]; NSUInteger lv = [e[2] unsignedIntegerValue];
        if (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureType2DArray) {   // TOTK: cube faces / array slices as separate 2D images, file index = (ser + slice*100000)*100 + level
            NSUInteger nsl = t.textureType == MTLTextureTypeCube ? 6 : MIN(t.arrayLength, 8u);
            for (NSUInteger sl = 0; sl < nsl; sl++) {
                id<MTLTexture> v = [t newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(lv, 1) slices:NSMakeRange(sl, 1)];
                if (!v) continue;
                NSUInteger w = MAX(t.width >> lv, 1u), h = MAX(t.height >> lv, 1u);
                MTLTextureDescriptor *od = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
                od.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; od.storageMode = MTLStorageModePrivate;
                id<MTLTexture> o = [g_dev newTextureWithDescriptor:od];
                id<MTLCommandBuffer> cb = q_cb();
                MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
                rp.colorAttachments[0].texture = o; rp.colorAttachments[0].loadAction = MTLLoadActionDontCare; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                id<MTLRenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp]; es_reset();
                [en setRenderPipelineState:ps]; [en setFragmentTexture:v atIndex:0]; [en setFragmentSamplerState:sp atIndex:0];
                [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [en endEncoding]; cb_commit(cb); cb_wait(cb);
                char path[1024]; snprintf(path, sizeof path, g_td_path, (int)((ser + sl * 100000) * 100 + lv));
                MLOG("[texdump] ser=%llu slice=%lu lv=%lu 格式 %lu %lux%lu（立方体/数组切片）", ser, (unsigned long)sl, (unsigned long)lv, (unsigned long)t.pixelFormat, (unsigned long)w, (unsigned long)h);
                shot(o, path, 0, NULL);
            }
            continue;
        }
        if (t.textureType != MTLTextureType2D) { MLOG("[texdump] ser=%llu 类型 %lu 跳过", ser, (unsigned long)t.textureType); continue; }
        if (t.pixelFormat == MTLPixelFormatDepth32Float || t.pixelFormat == MTLPixelFormatDepth16Unorm || t.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) {   // TOTK: depth: raw only
            NSUInteger rw = MAX(t.width >> lv, 1u), rh = MAX(t.height >> lv, 1u); size_t bpp = t.pixelFormat == MTLPixelFormatDepth16Unorm ? 2 : 4;
            id<MTLBuffer> rb = [g_dev newBufferWithLength:rw * rh * bpp options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> rc = q_cb(); id<MTLBlitCommandEncoder> bb = [rc blitCommandEncoder];
            MTLBlitOption opt = t.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone;
            [bb copyFromTexture:t sourceSlice:0 sourceLevel:lv sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(rw, rh, 1) toBuffer:rb destinationOffset:0 destinationBytesPerRow:rw * bpp destinationBytesPerImage:rw * rh * bpp options:opt];
            [bb endEncoding]; cb_commit(rc); cb_wait(rc);
            char rp_[1024]; snprintf(rp_, sizeof rp_, g_td_path, (int)(ser * 100 + lv)); strlcat(rp_, ".depth.raw", sizeof rp_);
            FILE *f = fopen(rp_, "wb"); if (f) { fprintf(f, "%lu %lu %lu %zu\n", (unsigned long)rw, (unsigned long)rh, (unsigned long)t.pixelFormat, rw * bpp); fwrite(rb.contents, 1, rw * rh * bpp, f); fclose(f); }
            MLOG("[texdump] depth ser=%llu %lux%lu 格式 %lu → %s", ser, (unsigned long)rw, (unsigned long)rh, (unsigned long)t.pixelFormat, rp_);
            continue;
        }
        if (getenv("DM_NVN_SNAPRAW")) {   // 同时存原始字节（HDR 目标转 8 位图会截断；首行文本头：宽 高 格式 每行字节） | Also save the raw bytes (converting an HDR target to an 8-bit image truncates; first-line text header: width height format bytes-per-row)
            NSUInteger rw = MAX(t.width >> lv, 1u), rh = MAX(t.height >> lv, 1u);
            size_t bpp = t.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : t.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : t.pixelFormat == MTLPixelFormatR16Float || t.pixelFormat == MTLPixelFormatR16Unorm ? 2 : 4;
            id<MTLBuffer> rb = [g_dev newBufferWithLength:rw * rh * bpp options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> rc = q_cb(); id<MTLBlitCommandEncoder> bb = [rc blitCommandEncoder];
            [bb copyFromTexture:t sourceSlice:0 sourceLevel:lv sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(rw, rh, 1) toBuffer:rb destinationOffset:0 destinationBytesPerRow:rw * bpp destinationBytesPerImage:rw * rh * bpp];
            [bb endEncoding]; cb_commit(rc); cb_wait(rc);
            char rp_[1024]; snprintf(rp_, sizeof rp_, g_td_path, (int)(ser * 100 + lv)); strlcat(rp_, ".raw", sizeof rp_);
            FILE *f = fopen(rp_, "wb"); if (f) { fprintf(f, "%lu %lu %lu %zu\n", (unsigned long)rw, (unsigned long)rh, (unsigned long)t.pixelFormat, rw * bpp); fwrite(rb.contents, 1, rw * rh * bpp, f); fclose(f); }
        }
        id<MTLTexture> v = [t newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(lv, 1) slices:NSMakeRange(0, 1)];
        NSUInteger w = MAX(t.width >> lv, 1u), h = MAX(t.height >> lv, 1u);
        MTLTextureDescriptor *od = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
        od.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; od.storageMode = MTLStorageModePrivate;
        id<MTLTexture> o = [g_dev newTextureWithDescriptor:od];
        id<MTLCommandBuffer> cb = q_cb();
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = o; rp.colorAttachments[0].loadAction = MTLLoadActionDontCare; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp]; es_reset();
        [en setRenderPipelineState:ps]; [en setFragmentTexture:v atIndex:0]; [en setFragmentSamplerState:sp atIndex:0];
        [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [en endEncoding]; cb_commit(cb); cb_wait(cb);
        char path[1024]; snprintf(path, sizeof path, g_td_path, (int)(ser * 100 + lv));
        MLOG("[texdump] ser=%llu lv=%lu 格式 %lu %lux%lu", ser, (unsigned long)lv, (unsigned long)t.pixelFormat, (unsigned long)w, (unsigned long)h);
        shot(o, path, 0, NULL);
    }
    [g_td_list removeAllObjects];
}
// DM_NVN_TEXSNAP=帧内序号,…（配合 DM_NVN_TEXDUMP 的帧号与路径）：这些通道一结束就把其颜色附件拷一份，与帧末存图一起存
// （同一张目标常在帧内被清掉复用，帧末状态反映不了中间通道的输出）；文件名序号 = 帧内序号*100000 + 贴图序号*100 + mip 级
// DM_NVN_TEXSNAP=in-frame index,... (together with the frame number and path of DM_NVN_TEXDUMP): as soon as these render passes end, copy their color attachments and save them together with the end-of-frame dump
// (the same target is often cleared and reused within a frame, so the end-of-frame state cannot reflect the output of intermediate passes); file name index = in-frame index*100000 + texture index*100 + mip level
static int g_ts_idx[64], g_ts_n = -1, g_ts_on; static uint64_t g_ts_first, g_ts_curf = ~0ull;
static __strong id<MTLTexture> g_ts_att[8]; static uint64_t g_ts_ser[8]; static NSUInteger g_ts_lv[8]; static int g_ts_na;
void ts_note(MTL4RenderPassDescriptor *pd, NSUInteger nc) {
    if (g_ts_n < 0) { g_ts_n = 0; const char *e = getenv("DM_NVN_TEXSNAP");
        for (const char *q = e; q && *q && g_ts_n < 64; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) g_ts_idx[g_ts_n++] = atoi(q); }
    static long minw = -1; if (minw < 0) { const char *e = getenv("DM_NVN_TEXSNAP_MIN"); minw = e ? atol(e) : 0; }
    static int dsw = -1, dsh; if (dsw < 0) { const char *e = getenv("DM_NVN_TEXSNAP_DEPTH"); if (!e || sscanf(e, "%dx%d", &dsw, &dsh) != 2) dsw = 0; }   // TOTK: depth-only passes of exactly this size (e.g. 256x144, the occluder depth of the occlusion culling)
    g_ts_on = 0; if ((!g_ts_n && !minw && !dsw) || g_td_frame < 0 || g_td_done) return;
    uint64_t f = atomic_load(&g_cur_frame); if (f != g_ts_curf) { g_ts_curf = f; g_ts_first = g_stat_pass; }
    if ((long)f != g_td_frame) return;
    int idx = (int)(g_stat_pass - g_ts_first); for (int i = 0; i < g_ts_n; i++) if (g_ts_idx[i] == idx) g_ts_on = idx + 1;
    /* TOTK: DM_NVN_TEXSNAP_MIN=<width>: also snapshot every pass whose colour attachment 0 is at least that wide (pass indices
       shift between runs, target sizes do not). */
    { id<MTLTexture> t0 = nc ? pd.colorAttachments[0].texture : nil; if (minw > 0 && t0 && (long)t0.width >= minw) g_ts_on = idx + 1;
      id<MTLTexture> td = pd.depthAttachment.texture; if (minw > 0 && !t0 && td && (long)td.width >= minw) g_ts_on = idx + 1;   // TOTK: depth-only passes (shadow maps) too
      if (dsw > 0 && !t0 && td && (int)td.width == dsw && (int)td.height == dsh) g_ts_on = idx + 1; }
    g_ts_na = 0; if (!g_ts_on) return;
    for (NSUInteger i = 0; i < nc && i < 8; i++) { id<MTLTexture> t = pd.colorAttachments[i].texture; if (!t) continue;
        g_ts_att[g_ts_na] = t; g_ts_ser[g_ts_na] = R.rt.color[i] ? R.rt.color[i]->serial : 0; g_ts_lv[g_ts_na] = pd.colorAttachments[i].level; g_ts_na++; }
    /* TOTK: the depth attachment is snapshotted too (raw float dump only; see td_dump) */
    if (pd.depthAttachment.texture && R.rt.depth && g_ts_na < 8) { g_ts_att[g_ts_na] = pd.depthAttachment.texture; g_ts_ser[g_ts_na] = R.rt.depth->serial; g_ts_lv[g_ts_na] = pd.depthAttachment.level; g_ts_na++; }
}
void ts_after(void) {
    if (!g_ts_on) return;
    /* TOTK: the copy is encoded in the frame's own (Metal 4) command buffer right after the pass, so it sees the pass's
       output; a separate command buffer committed here would run before the frame's work. File index = pass index * 20000 + serial. */
    for (int i = 0; i < g_ts_na; i++) {
        id<MTLTexture> t = g_ts_att[i]; if (t.textureType != MTLTextureType2D && t.textureType != MTLTextureType2DArray) continue;
        /* TOTK: array textures (shadow cascades) are snapshotted slice by slice as 2D copies; the file index adds slice*2000 */
        NSUInteger nsl = t.textureType == MTLTextureType2DArray ? MIN(t.arrayLength, 4u) : 1;
        for (NSUInteger sl = 0; sl < nsl; sl++) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat width:t.width height:t.height mipmapped:t.mipmapLevelCount > 1];
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate; d.mipmapLevelCount = t.mipmapLevelCount;
        id<MTLTexture> c = [g_dev newTextureWithDescriptor:d]; res_add(c);
        if (!g_ts_made) g_ts_made = [NSMutableArray array];
        [g_ts_made addObject:c];   // leaves the residency set once td_dump has saved it
        id<MTL4ComputeCommandEncoder> b = blit();
        [b copyFromTexture:t sourceSlice:sl sourceLevel:g_ts_lv[i] toTexture:c destinationSlice:0 destinationLevel:g_ts_lv[i] sliceCount:1 levelCount:1];
        [g_td_list addObject:@[c, @((uint64_t)(g_ts_on - 1) * 20000 + (g_ts_ser[i] % 20000) + sl * 2000), @(g_ts_lv[i])]];
        }
    }
    g_ts_on = 0;
}
// 真实呈现时刻环：出图序号 → presentedTime（秒；−1 = 被系统丢弃未呈现）
// Ring of actual presentation times: present sequence number -> presentedTime (seconds; -1 = dropped by the system, not presented)
#define PRING 1024
typedef struct { uint64_t n; double t; } pres_ent;
static pres_ent g_pres[PRING];
// 取第 n 次出图的实际呈现时刻（未就绪返回 0，被丢弃返回 −1）
// Get the actual presentation time of the nth present (returns 0 if not ready, -1 if dropped)
double mtl_presented_time(uint64_t n) { pres_ent e = g_pres[n % PRING]; return e.n == n ? e.t : 0; }
static _Atomic uint64_t g_drawable_ns;   // 出图里等 nextDrawable 的累计墙钟（计时埋点） | Accumulated wall clock waiting for nextDrawable in present (timing instrumentation)
uint64_t mtl_drawable_ns_take(void) { return atomic_exchange(&g_drawable_ns, 0); }
// 有窗口时屏幕的最高刷新率（Hz）；无窗口返回 0。Mac 用 CoreGraphics 显示模式（线程安全的 C 接口，出图配置在游戏线程上读）
// Maximum refresh rate (Hz) of the screen when there is a window; returns 0 without a window. Mac uses CoreGraphics display modes (thread-safe C interface; the present configuration is read on the game thread)
int mtl_display_hz(void) {
    if (!front_metal_layer()) return 0;
#if TARGET_OS_OSX
    CGDisplayModeRef m = CGDisplayCopyDisplayMode(CGMainDisplayID());
    double hz = m ? CGDisplayModeGetRefreshRate(m) : 0; if (m) CGDisplayModeRelease(m);
    return (int)(hz + 0.5);
#else
    Class c = NSClassFromString(@"UIScreen"); id s = c ? [c valueForKey:@"mainScreen"] : nil;
    return s ? [[s valueForKey:@"maximumFramesPerSecond"] intValue] : 0;
#endif
}
static _Atomic uint64_t g_present_stall_ns;   // 单测用：出图线程每次出图前额外停顿（模拟 drawable 迟迟不可用） | For unit tests: an extra pause before each present on the present thread (simulating a drawable that stays unavailable)
static CAMetalLayer *g_present_layer_resident;
void mtl_present_test_stall(uint64_t ns) { g_present_stall_ns = ns; }
static void present_pipeline(MTLPixelFormat format) {
    if (!g_blit_pso || g_blit_fmt != format) {
        static const char *src_msl =
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "struct V { float4 p [[position]]; float2 uv; };\n"
            "vertex V vs(uint id [[vertex_id]]) { float2 uv = float2((id << 1) & 2, id & 2); V o; o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = uv; return o; }\n"
            "struct Crop { float4 rect; float4 edge; };\n"
            "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant Crop& c [[buffer(0)]]) { return t.sample(s, clamp(in.uv * c.rect.zw + c.rect.xy, c.edge.xy, c.edge.zw)); }\n";
        NSError *e = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src_msl) options:nil error:&e];
        if (!lib) MDIE("出图着色器编译失败 %s", e.localizedDescription.UTF8String);
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
        pd.colorAttachments[0].pixelFormat = format;
        g_blit_pso = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e];
        if (!g_blit_pso) MDIE("出图管线失败 %s", e.localizedDescription.UTF8String);
        g_blit_fmt = format;
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear; sd.supportArgumentBuffers = YES;
        g_blit_samp = [g_dev newSamplerStateWithDescriptor:sd];
    }
}
// 出图仅采NVN声明的有效源矩形；与原始交换链读回截图分开。边缘钳到有效区纹素中心，双线性不读外侧黑纹素。
// Present samples only the valid source rectangle declared by NVN; kept separate from the raw swapchain readback screenshot. Edges are clamped to the texel centers of the valid area, so bilinear does not read the black texels outside.
// TOTK: FPS and on-screen picture size in the top-right corner of the game picture (pixels come from fps_overlay.c). Rasterises into
// the slot's texture only when what is shown changes, and returns the viewport and source rectangle to draw.
static int fps_overlay_prepare(int slot, MTLViewport vp, MTLViewport *ovp, float uv[8]) {
    if (!fps_overlay_hints_enabled() || !fps_overlay_enabled() || slot < 0 || slot > 3 || !g_fps_tex[slot]) return 0;
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    static uint8_t px[FPS_OVERLAY_TEX_W * FPS_OVERLAY_TEX_H * 4];
    int scale = fps_overlay_scale(vp.height);
    uint64_t mfx_at = atomic_load(&g_mfxup_last), gen_at = atomic_load(&g_mfxi_last), cur = atomic_load(&g_cur_frame);   // the last game frame MetalFX did the upscale (" MFX") / an interpolated frame was shown ("+50")
    int gen = gen_at && cur - gen_at < 30 ? fps_overlay_gen_value() : 0;
    fps_overlay_info info = {fps_overlay_value(), (int)lround(vp.width), (int)lround(vp.height), mfx_at && cur - mfx_at < 30, gen > 0 ? gen : 0, atomic_load(&g_pic_big)};
    pthread_mutex_lock(&mu);
    if (g_fps_key[slot].scale != scale || memcmp(&g_fps_key[slot].info, &info, sizeof info)) {
        fps_overlay_raster(&info, scale, px, &g_fps_box[slot][0], &g_fps_box[slot][1]);
        [g_fps_tex[slot] replaceRegion:MTLRegionMake2D(0, 0, FPS_OVERLAY_TEX_W, FPS_OVERLAY_TEX_H) mipmapLevel:0 withBytes:px bytesPerRow:FPS_OVERLAY_TEX_W * 4];
        g_fps_key[slot].info = info; g_fps_key[slot].scale = scale;
    }
    float bw = g_fps_box[slot][0], bh = g_fps_box[slot][1], W = FPS_OVERLAY_TEX_W, H = FPS_OVERLAY_TEX_H;
    pthread_mutex_unlock(&mu);
    double m = 3.0 * scale;
    *ovp = (MTLViewport){vp.originX + vp.width - m - bw, vp.originY + m, bw, bh, 0, 1};   // top-right corner
    const float u[8] = {0, 0, bw / W, bh / H, 0.5f / W, 0.5f / H, (bw - 0.5f) / W, (bh - 0.5f) / H};
    memcpy(uv, u, sizeof u);
    return 1;
}
/* TOTK automatic shader collection, bottom-left box (user, 2026-10-16 night: "is anything new being found and collected"):
   "SHADER +<variants new to the translation cache this run>/<variants in the cache> PSO +<pipelines added to the warm
   lists this run>", yellow for 3 s after something new was collected. Same texture size, font and scale as the FPS box. */
static int shader_overlay_prepare(int slot, MTLViewport vp, MTLViewport *ovp, float uv[8]) {
    if (!fps_overlay_hints_enabled() || !shader_overlay_enabled() || slot < 0 || slot > 3 || !g_shd_tex[slot]) return 0;
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    static uint8_t px[FPS_OVERLAY_TEX_W * FPS_OVERLAY_TEX_H * 4];
    uint64_t fresh = 0, last = 0; int64_t total = -1; totk_rt_counts(&fresh, &total, &last);
    uint64_t pso = atomic_load(&g_collect_new[1]) + atomic_load(&g_collect_new[2]), plast = atomic_load(&g_collect_last_ns);
    uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW), newest = last > plast ? last : plast;
    char text[48]; shader_overlay_text(text, sizeof text, fresh, total, pso);
    int hl = newest && now - newest < 3000000000ull, scale = fps_overlay_scale(vp.height);
    pthread_mutex_lock(&mu);
    if (g_shd_key[slot].scale != scale || g_shd_key[slot].hl != hl || strcmp(g_shd_key[slot].text, text)) {
        fps_overlay_raster_line(text, hl, scale, px, &g_shd_box[slot][0], &g_shd_box[slot][1]);
        [g_shd_tex[slot] replaceRegion:MTLRegionMake2D(0, 0, FPS_OVERLAY_TEX_W, FPS_OVERLAY_TEX_H) mipmapLevel:0 withBytes:px bytesPerRow:FPS_OVERLAY_TEX_W * 4];
        snprintf(g_shd_key[slot].text, sizeof g_shd_key[slot].text, "%s", text); g_shd_key[slot].hl = hl; g_shd_key[slot].scale = scale;
    }
    float bw = g_shd_box[slot][0], bh = g_shd_box[slot][1], W = FPS_OVERLAY_TEX_W, H = FPS_OVERLAY_TEX_H;
    pthread_mutex_unlock(&mu);
    double m = 3.0 * scale;
    *ovp = (MTLViewport){vp.originX + m, vp.originY + vp.height - m - bh, bw, bh, 0, 1};   // bottom-left corner
    const float u[8] = {0, 0, bw / W, bh / H, 0.5f / W, 0.5f / H, (bw - 0.5f) / W, (bh - 0.5f) / H};
    memcpy(uv, u, sizeof u);
    return 1;
}
/* TOTK load line, bottom-right (user, 2026-10-16 night: every CPU core and the GPU, one line like the FPS counter):
   "P 55% 56% 55% 53%  E 43% 29% 20% 12%  9%  7%  T 47%  GPU 99% (70%)" -- performance / efficiency cores, the busiest thread, the GPU as macOS reports it (our
   keep-alive keeps it near 100 %) and in brackets the GPU time of the game's own work. Values from totk_sysload.c (0.5 s). */
static int load_overlay_prepare(int slot, MTLViewport vp, MTLViewport *ovp, float uv[8]) {
    if (!fps_overlay_hints_enabled() || !sysload_enabled() || slot < 0 || slot > 3 || !g_load_tex[slot]) return 0;
    sysload_start();
    sysload_info li; if (!sysload_get(&li)) return 0;
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    static uint8_t px[LOAD_OVERLAY_TEX_W * FPS_OVERLAY_TEX_H * 4];
    char text[96]; sysload_text(text, sizeof text, &li);
    int scale = fps_overlay_scale(vp.height);
    pthread_mutex_lock(&mu);
    if (g_load_key[slot].scale != scale || strcmp(g_load_key[slot].text, text)) {
        fps_overlay_raster_line_tex(text, 0, scale, px, LOAD_OVERLAY_TEX_W, FPS_OVERLAY_TEX_H, &g_load_box[slot][0], &g_load_box[slot][1]);
        [g_load_tex[slot] replaceRegion:MTLRegionMake2D(0, 0, LOAD_OVERLAY_TEX_W, FPS_OVERLAY_TEX_H) mipmapLevel:0 withBytes:px bytesPerRow:LOAD_OVERLAY_TEX_W * 4];
        snprintf(g_load_key[slot].text, sizeof g_load_key[slot].text, "%s", text); g_load_key[slot].scale = scale;
    }
    float bw = g_load_box[slot][0], bh = g_load_box[slot][1], W = LOAD_OVERLAY_TEX_W, H = FPS_OVERLAY_TEX_H;
    pthread_mutex_unlock(&mu);
    double m = 3.0 * scale;
    *ovp = (MTLViewport){vp.originX + vp.width - m - bw, vp.originY + vp.height - m - bh, bw, bh, 0, 1};   // bottom-right corner
    const float u[8] = {0, 0, bw / W, bh / H, 0.5f / W, 0.5f / H, (bw - 0.5f) / W, (bh - 0.5f) / H};
    memcpy(uv, u, sizeof u);
    return 1;
}
/* TOTK 2K output (user, 2026-10-16 night: "2K output with MetalFX"): when the window or the full screen shows the game
   picture bigger than it is (the 1920x1080 picture at 2560x1440 full screen on the user's 3440x1440 display), MetalFX
   upscales it to the viewport (perceptual: display values) and the present draws the result 1:1, instead of the bilinear
   stretch. Key 3 enables both MetalFX stages in its manual mode; TOTK_MFX_OUTPUT=0 disables the output stage. Only an enlargement
   by a margin and a picture region at the texture's origin (MetalFX reads the top-left content region); while the window
   is being resized the size must hold for 15 pictures first (building a scaler per size would stall every present). */
static uint64_t g_outup_n, g_outup_skip[3];   // done; skipped: off, not an enlargement / region not at the origin / not built, size settling
static id<MTL4CounterHeap> g_outup_heap; static double g_outup_hz;
static int g_outup_timed[3];                  // per present slot: this command buffer carries the upscale's two timestamps
static double g_outup_ms[512]; static _Atomic int g_outup_nms;
static NSUInteger g_outup_size[4];            // in and out size of the last upscale
static id<MTLTexture> output_upscale(id<MTL4CommandBuffer> cb, id<MTLTexture> src, NSUInteger cx, NSUInteger cy, NSUInteger cw, NSUInteger ch, MTLViewport vp, int slot) {
    static NSUInteger sw, sh; static int same;
    if (slot >= 0 && slot < 3) g_outup_timed[slot] = 0;
    if (!mfxup_enabled() || !mfxup_output_enabled()) { g_outup_skip[0]++; return nil; }
    NSUInteger ow = (NSUInteger)vp.width, oh = (NSUInteger)vp.height;
    if (cx || cy || ow < cw + 16 || oh < ch + 9) { g_outup_skip[1]++; return nil; }
    if (ow != sw || oh != sh) { sw = ow; sh = oh; same = 0; }
    if (same < 15) { same++; g_outup_skip[2]++; return nil; }
    if (!g_outup_heap) { MTL4CounterHeapDescriptor *hd = [MTL4CounterHeapDescriptor new]; hd.type = MTL4CounterHeapTypeTimestamp; hd.count = 6;
                         NSError *e = nil; g_outup_heap = [[M3CounterHeap alloc] initWithDevice:g_dev count:hd.count]; g_outup_hz = (double)1e9; }
    int timed = g_outup_heap && slot >= 0 && slot < 3;
    if (timed) m3_timestamp(g_outup_heap, slot * 2);
    uint64_t t0 = now_ns();
    id<MTLTexture> out = mfxup_encode_ex(MFXUP_OUTPUT, cb, src, cw, ch, nil, ow, oh, MFXUP_PERCEPTUAL);
    uint64_t t1 = now_ns();
    if (timed) m3_timestamp(g_outup_heap, slot * 2 + 1);
    if (t1 - t0 > 2000000) MLOG("[mfx-up] output: %lux%lu -> %lux%lu, encoding took %.1f ms on the present thread (scaler built)", (unsigned long)cw, (unsigned long)ch, (unsigned long)ow, (unsigned long)oh, (t1 - t0) / 1e6);
    if (!out) { g_outup_skip[1]++; return nil; }
    if (timed) g_outup_timed[slot] = 1;
    g_outup_size[0] = cw; g_outup_size[1] = ch; g_outup_size[2] = ow; g_outup_size[3] = oh;
    mfxup_queue_residency(); g_outup_n++; atomic_store(&g_mfxup_last, atomic_load(&g_cur_frame));
    return out;
}
static int dcmp_out(const void *x, const void *y) { double a = *(const double *)x, b = *(const double *)y; return a < b ? -1 : a > b; }
static void outup_note_gpu(int slot) {   // the present command buffer of this slot completed: the upscale's GPU time
    if (!g_outup_heap || g_outup_hz <= 0) return;
    NSData *d = [g_outup_heap resolveCounterRange:NSMakeRange((NSUInteger)slot * 2, 2)]; const MTL4TimestampHeapEntry *t = d.bytes;
    if (!t || d.length < 2 * sizeof *t || t[1].timestamp <= t[0].timestamp) return;
    int n = atomic_fetch_add(&g_outup_nms, 1); g_outup_ms[n % 512] = (t[1].timestamp - t[0].timestamp) / g_outup_hz * 1e3;
    if ((n + 1) % 1200 == 0) {
        double a[512]; memcpy(a, g_outup_ms, sizeof a); qsort(a, 512, sizeof *a, dcmp_out);
        MLOG("[mfx-up] 出图放大（2K 输出）至帧 %llu：MetalFX %lux%lu → %lux%lu 共 %llu 次，每次 GPU p50 %.2f / p95 %.2f ms；没放大：关 %llu、不是放大或区域不在原点 %llu、窗口尺寸在变 %llu",
             (unsigned long long)atomic_load(&g_cur_frame), (unsigned long)g_outup_size[0], (unsigned long)g_outup_size[1], (unsigned long)g_outup_size[2], (unsigned long)g_outup_size[3],
             (unsigned long long)g_outup_n, a[256], a[486], (unsigned long long)g_outup_skip[0], (unsigned long long)g_outup_skip[1], (unsigned long long)g_outup_skip[2]);
    }
}
/* TOTK_PICTURE_FILL: how the game picture fills the drawable. fit (default) keeps 16:9 and letterboxes;
   stretch covers the whole drawable with non-uniform scaling; crop covers it uniformly and cuts the overflowing edges. */
static int present_fill_mode(void) {
    static int m = -2;
    if (m == -2) {
        const char *e = getenv("TOTK_PICTURE_FILL");
        m = (e && !strcmp(e, "stretch")) ? 1 : (e && !strcmp(e, "crop")) ? 2 : (e && !strcmp(e, "1")) ? 1 : (e && !strcmp(e, "2")) ? 2 : 0;
        MLOG("[present] picture fill: %s", m == 1 ? "stretch (full, non-uniform)" : m == 2 ? "crop (full, uniform, edges cut)" : "fit (16:9 letterbox)");
    }
    return m;
}
static int present_encode(id<MTLCommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst, const int crop[4], int fps_overlay) {
    NSUInteger cx = 0, cy = 0, cw = src.width, ch = src.height;
    if (crop && (crop[2] || crop[3])) {
        if (crop[0] < 0 || crop[1] < 0 || crop[2] <= 0 || crop[3] <= 0 ||
            (uint64_t)crop[0] + crop[2] > src.width || (uint64_t)crop[1] + crop[3] > src.height) {
            MLOG("【红】出图裁切越界：%d,%d %dx%d / 原纹理 %lux%lu", crop[0], crop[1], crop[2], crop[3], (unsigned long)src.width, (unsigned long)src.height); return 0;
        }
        cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3];
    }
    static uint64_t last_crop[8]; uint64_t meta[8] = {src.width, src.height, cx, cy, cw, ch, dst.width, dst.height};
    if (memcmp(meta, last_crop, sizeof meta)) { memcpy(last_crop, meta, sizeof meta);
        MLOG("出图裁切：原纹理 %lux%lu，有效 %lu,%lu %lux%lu → 目标 %lux%lu", (unsigned long)src.width, (unsigned long)src.height,
             (unsigned long)cx, (unsigned long)cy, (unsigned long)cw, (unsigned long)ch, (unsigned long)dst.width, (unsigned long)dst.height); }
    present_pipeline(dst.pixelFormat);
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    // 保持游戏画面宽高比居中，余下留黑边（屏幕比例与游戏不同时不拉伸，如 iPhone 约 19.5:9 对游戏 16:9）；
    // 比例差 < 1%（全屏口径下游戏宽度取 8 的倍数带来的零头，如 1952x900 对 2868x1320 差 0.17%）直接铺满，不留几像素的边
    // Keep the game picture's aspect ratio and center it, leaving black bars in the rest (no stretching when the screen ratio differs from the game's, e.g. iPhone about 19.5:9 versus the game's 16:9);
    // when the ratio difference is < 1% (the remainder from the game width being a multiple of 8 in the full-screen setup, e.g. 1952x900 versus 2868x1320 differs by 0.17%) fill directly, leaving no few-pixel border
    double dw = dst.width, dh = dst.height;
    int fill = present_fill_mode();
    MTLViewport vp;
    if (fill) {
        vp = (MTLViewport){0, 0, dw, dh, 0, 1};
        if (fill == 2) {   /* uniform cover: crop the source to the drawable aspect */
            double da = dw / dh, sa = (double)cw / ch;
            if (sa > da) { NSUInteger nw = (NSUInteger)llround(ch * da); if (nw < 1) nw = 1; cx += (cw - nw) / 2; cw = nw; }
            else { NSUInteger nh = (NSUInteger)llround(cw / da); if (nh < 1) nh = 1; cy += (ch - nh) / 2; ch = nh; }
        }
    } else {
        double s = fmin(dw / cw, dh / ch);
        vp = (MTLViewport){floor((dw - cw * s) / 2), floor((dh - ch * s) / 2), round(cw * s), round(ch * s), 0, 1};
        if (fabs((dw / dh) / ((double)cw / ch) - 1) < 0.01) vp = (MTLViewport){0, 0, dw, dh, 0, 1};
    }
    int full = vp.originX == 0 && vp.originY == 0 && vp.width == dw && vp.height == dh;
    static double last_vp[4];   // 视口变化时记一行（真机屏幕级验收 screen_check.py 核宽高比用） | Log one line when the viewport changes (used by the real-device screen-level acceptance screen_check.py to check the aspect ratio)
    if (memcmp(last_vp, &vp, sizeof last_vp)) { memcpy(last_vp, &vp, sizeof last_vp);
        MLOG("出图视口 %.0f,%.0f %.0fx%.0f / 画布 %.0fx%.0f（游戏画面 %lux%lu）", vp.originX, vp.originY, vp.width, vp.height, dw, dh, (unsigned long)src.width, (unsigned long)src.height); }
    rp.colorAttachments[0].texture = dst; rp.colorAttachments[0].loadAction = full ? MTLLoadActionDontCare : MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    /* 出图线程不碰回放线程的渲染状态缓存：present 路径直接 setViewport/setRenderPipelineState，
       es_reset() 会把游戏线程正在用的全局缓存清掉（跨线程数据竞争）。
       The present thread must not touch the replay thread's render-state cache: this path sets the viewport and
       pipeline directly, and es_reset() would clear the global cache the game thread is using (a data race). */
    id<MTLRenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp];
    [en setViewport:vp];
    [en setRenderPipelineState:g_blit_pso]; [en setFragmentTexture:src atIndex:0]; [en setFragmentSamplerState:g_blit_samp atIndex:0];
    float uv[8] = {(float)cx / src.width, (float)cy / src.height, (float)cw / src.width, (float)ch / src.height,
                   (cx + 0.5f) / src.width, (cy + 0.5f) / src.height, (cx + cw - 0.5f) / src.width, (cy + ch - 0.5f) / src.height};
    [en setFragmentBytes:uv length:sizeof uv atIndex:0];
    [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    MTLViewport ovp; float fuv[8];
    if (fps_overlay && fps_overlay_prepare(3, vp, &ovp, fuv)) {
        [en setViewport:ovp]; [en setFragmentTexture:g_fps_tex[3] atIndex:0]; [en setFragmentBytes:fuv length:sizeof fuv atIndex:0];
        [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    if (fps_overlay && shader_overlay_prepare(3, vp, &ovp, fuv)) {
        [en setViewport:ovp]; [en setFragmentTexture:g_shd_tex[3] atIndex:0]; [en setFragmentBytes:fuv length:sizeof fuv atIndex:0];
        [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    if (fps_overlay && load_overlay_prepare(3, vp, &ovp, fuv)) {
        [en setViewport:ovp]; [en setFragmentTexture:g_load_tex[3] atIndex:0]; [en setFragmentBytes:fuv length:sizeof fuv atIndex:0];
        [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [en endEncoding];
    return 1;
}
static int present_encode_m4(id<MTL4CommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst, const int crop[4], int uv_i) {
    NSUInteger cx = 0, cy = 0, cw = src.width, ch = src.height;
    if (crop && (crop[2] || crop[3])) {
        if (crop[0] < 0 || crop[1] < 0 || crop[2] <= 0 || crop[3] <= 0 ||
            (uint64_t)crop[0] + crop[2] > src.width || (uint64_t)crop[1] + crop[3] > src.height) {
            MLOG("【红】出图裁切越界：%d,%d %dx%d / 原纹理 %lux%lu", crop[0], crop[1], crop[2], crop[3], (unsigned long)src.width, (unsigned long)src.height); return 0;
        }
        cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3];
    }
    if (crop) atomic_store(&g_pic_big, cw > 1920 || ch > 1080);   // a game frame (interpolated frames come without the crop)
    present_pipeline(dst.pixelFormat);
    double dw = dst.width, dh = dst.height;
    int fill = present_fill_mode();
    MTLViewport vp;
    if (fill) {
        vp = (MTLViewport){0, 0, dw, dh, 0, 1};
        if (fill == 2) {
            double da = dw / dh, sa = (double)cw / ch;
            if (sa > da) { NSUInteger nw = (NSUInteger)llround(ch * da); if (nw < 1) nw = 1; cx += (cw - nw) / 2; cw = nw; }
            else { NSUInteger nh = (NSUInteger)llround(cw / da); if (nh < 1) nh = 1; cy += (ch - nh) / 2; ch = nh; }
        }
    } else {
        double s = fmin(dw / cw, dh / ch);
        vp = (MTLViewport){floor((dw - cw * s) / 2), floor((dh - ch * s) / 2), round(cw * s), round(ch * s), 0, 1};
        if (fabs((dw / dh) / ((double)cw / ch) - 1) < 0.01) vp = (MTLViewport){0, 0, dw, dh, 0, 1};
    }
    int full = vp.originX == 0 && vp.originY == 0 && vp.width == dw && vp.height == dh;
    id<MTLTexture> up = output_upscale(cb, src, cx, cy, cw, ch, vp, uv_i);   // TOTK 2K output: MetalFX to the viewport, drawn 1:1 below
    id<MTLTexture> pic = up ? up : src;
    if (up) { cx = 0; cy = 0; cw = up.width; ch = up.height; }
    MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = full ? MTLLoadActionDontCare : MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    /* 同上：出图线程不重置回放线程的全局状态缓存 | Same: the present thread does not reset the replay thread's global state cache */
    id<MTL4RenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp];
    if (!en) { MLOG("【红】M4 出图渲染编码器创建失败"); return 0; }
    ((void)0);
    id<MTL4ArgumentTable> args = g_present_args[uv_i];
    /* M3 的参数表是立即绑定：必须先挂到编码器上，后面的 setTexture/setSamplerState/setAddress 才落得进去。
       Metal 4 里参数表是独立对象，先填、再 setArgumentTable: 挂上；移植时把 m3_attach 留在了原来的位置
       （所有 set 之后），于是每次出图的纹理/采样器/UV 都没绑上，blit 绘制拿着空资源去执行。
       The Metal 3 argument table binds immediately, so it must be attached to the encoder before the
       setTexture/setSamplerState/setAddress calls. In Metal 4 the table is a standalone object that is
       filled first and attached with setArgumentTable: afterwards; the port kept m3_attach where the
       original attach was (after every set), so the present blit drew with nothing bound. */
    m3_attach(args, en, M3_STAGE_FRAGMENT);
    [args setTexture:pic atIndex:0];
    [args setSamplerState:g_blit_samp atIndex:0];
    float *uv = g_present_uv[uv_i].contents;
    uv[0] = (float)cx / pic.width; uv[1] = (float)cy / pic.height; uv[2] = (float)cw / pic.width; uv[3] = (float)ch / pic.height;
    uv[4] = (cx + 0.5f) / pic.width; uv[5] = (cy + 0.5f) / pic.height; uv[6] = (cx + cw - 0.5f) / pic.width; uv[7] = (cy + ch - 0.5f) / pic.height;
    [args setAddress:g_present_uv[uv_i].gpuAddress atIndex:0];
    [en setViewport:vp]; [en setRenderPipelineState:g_blit_pso];
    [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    MTLViewport ovp; float fuv[8];
    if (fps_overlay_prepare(uv_i, vp, &ovp, fuv)) {   // argument-table bindings are captured per draw, so rebinding here leaves the game picture's draw intact
        memcpy(g_fps_uv[uv_i].contents, fuv, sizeof fuv);
        [args setTexture:g_fps_tex[uv_i] atIndex:0];
        [args setAddress:g_fps_uv[uv_i].gpuAddress atIndex:0];
        [en setViewport:ovp];
        [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    if (shader_overlay_prepare(uv_i, vp, &ovp, fuv)) {   // bottom-left: automatic shader collection
        memcpy(g_shd_uv[uv_i].contents, fuv, sizeof fuv);
        [args setTexture:g_shd_tex[uv_i] atIndex:0];
        [args setAddress:g_shd_uv[uv_i].gpuAddress atIndex:0];
        [en setViewport:ovp];
        [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    if (load_overlay_prepare(uv_i, vp, &ovp, fuv)) {   // bottom-right: CPU cores and GPU load
        memcpy(g_load_uv[uv_i].contents, fuv, sizeof fuv);
        [args setTexture:g_load_tex[uv_i] atIndex:0];
        [args setAddress:g_load_uv[uv_i].gpuAddress atIndex:0];
        [en setViewport:ovp];
        [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    m4_end(en);
    return 1;
}
static double g_link_it_tin;   // 节拍回调正在上屏的那一帧的入队时刻（只在节拍线程读写） | Enqueue time of the frame the display-link callback is presenting (read and written only on the display-link thread)
static id<MTL4CommandAllocator> g_flash_guard_alloc;
static id<MTL4CommandBuffer> g_flash_guard_cb;
static id<MTLBuffer> g_flash_guard_sample;
static id<MTLTexture> g_flash_guard_previous;
static double g_flash_guard_good[3];
static int g_flash_guard_ready, g_flash_guard_rejected;
/* TOTK_ANOMALY detector (see anom_note): four horizontal strips of each presented picture (at 15 / 35 / 50 / 65 % of the
   height, 4 rows, whole width) are read back on the presenting queue and reduced to 4 x 32 cell luminances. Frame N-1 is
   an anomaly when it differs from both N-2 and N by more than ANOM_T (mean absolute cell difference, 0..255) while N-2
   and N are much closer. Then the draw ring for frames N-4..N and the pictures of N-2..N go to <dir>/a<k>-*. */
enum { ANOM_CELLS = 128, ANOM_CAPTURES = 6 };
static const double ANOM_T = 12.0;
static id<MTL4CommandAllocator> g_anom_alloc; static id<MTL4CommandBuffer> g_anom_cb; static id<MTLBuffer> g_anom_strip;
static id<MTLTexture> g_anom_img[3]; static uint64_t g_anom_img_f[3];
static double g_anom_v[3][ANOM_CELLS]; static uint64_t g_anom_vf[3];
static int g_anom_captures; static uint64_t g_anom_last, g_anom_win0, g_anom_win_hits, g_anom_total; static double g_anom_win_max; static uint64_t g_anom_win_f;
static uint64_t g_anom_readbacks, g_anom_triplets;
static double anom_mad(const double *a, const double *b) { double s = 0; for (int i = 0; i < ANOM_CELLS; i++) s += fabs(a[i] - b[i]); return s / ANOM_CELLS; }
static void anom_dump(uint64_t fa, double d1, double d2, double d0);
/* TOTK_INPUT_FILE "anomdump": the draw ring of the last frames (frames N-5..N-1, N = current) with each draw's GPU sample
   count, without an anomaly. Needs TOTK_ANOMALY=<dir> with per-draw records on. For "which draws produce no fragment". */
void dm_gfx_anomdump(void) {
    if (!anom_on() || !g_anom_rec) { MLOG("[anom] anomdump: TOTK_ANOMALY=<目录> 且未设 TOTK_ANOMALY_RECORD=0 时才有逐绘制记录"); return; }
    uint64_t f = atomic_load(&g_cur_frame); if (f < 6) return;
    MLOG("[anom] anomdump: 写出第 %llu–%llu 帧的逐绘制记录（含 GPU 通过样本数）", (unsigned long long)(f - 5), (unsigned long long)(f - 1));
    anom_dump(f - 2, 0, 0, 0);
}
static void anom_dump(uint64_t fa, double d1, double d2, double d0) {
    int k = ++g_anom_captures; char path[1024];
    snprintf(path, sizeof path, "%s/a%d-draws.txt", g_anom_dir, k);
    FILE *o = fopen(path, "w"); if (!o) { MLOG("【红】[anom] cannot write %s", path); return; }
    fprintf(o, "# anomaly frame %llu: differs from previous %.1f, next %.1f; previous vs next %.1f\n", (unsigned long long)fa, d1, d2, d0);
    for (uint64_t f = fa >= 3 ? fa - 3 : 0; f <= fa + 1; f++) {
        int s_ = (int)(f % ANOM_FRAMES); pthread_mutex_lock(&g_anom[s_].mu);
        if (g_anom[s_].frame == f) for (uint32_t i = 0; i < g_anom[s_].n; i++) {
            const anom_draw *a = &g_anom[s_].d[i];
            if (a->kind != 'D') {
                fprintf(o, "F%llu %u p%llu %c dst=%#llx n=%d src=%#llx f=[", (unsigned long long)f, i, (unsigned long long)a->pass, a->kind, (unsigned long long)a->ib, a->count, (unsigned long long)a->vb0);
                for (int q = 0; q < 16; q++) fprintf(o, q ? ",%g" : "%g", a->b[0].f[q]);
                fprintf(o, "]\n"); continue;
            }
            fprintf(o, "F%llu %u p%llu vs=%.16s fs=%.16s vc=%d inst=%d first=%d bv=%d ib=%#llx vb0=%#llx", (unsigned long long)f, i, (unsigned long long)a->pass,
                    a->vs, a->fs, a->count, a->inst, a->first, a->bv, (unsigned long long)a->ib, (unsigned long long)a->vb0);
            for (int t = 0; t < a->ntex; t++) fprintf(o, " t%d=%llu", t, (unsigned long long)a->tex[t]);
            for (int t = 0; t < 3; t++) if (a->rt[t].ser) fprintf(o, " %s=%llu:%#llx:%ux%u:%#x", t == 2 ? "ds" : t ? "rt1" : "rt0", (unsigned long long)a->rt[t].ser, (unsigned long long)a->rt[t].mem, a->rt[t].w, a->rt[t].h, a->rt[t].fmt);
            for (int t = 0; t < a->nft; t++) if (a->ft[t].ser) fprintf(o, " ft%d=%llu:%#llx:%ux%u:%#x:%p", t, (unsigned long long)a->ft[t].ser, (unsigned long long)a->ft[t].mem, a->ft[t].w, a->ft[t].h, a->ft[t].fmt, a->ft[t].mtl);
            if (a->smp >= 0) fprintf(o, " smp=%lld", (long long)a->smp); else if (a->smp == -2) fprintf(o, " smp=?");
            if (a->ilen) fprintf(o, " ih=%016llx/%u", (unsigned long long)a->ih, a->ilen);
            for (int t = 0; t < a->nvs; t++) fprintf(o, " vh%d=%016llx/%u", t, (unsigned long long)a->vh[t], a->vlen[t]);
            for (int b = 0; b < a->nb; b++) {
                const anom_buf *u = &a->b[b];
                fprintf(o, " %c%d=%#llx/%u/%016llx[", u->kind, u->reg, (unsigned long long)u->addr, u->size, (unsigned long long)u->hash);
                for (int q = 0; q < 16; q++) fprintf(o, q ? ",%g" : "%g", u->f[q]);
                fputc(']', o);
            }
            fputc('\n', o);
        }
        pthread_mutex_unlock(&g_anom[s_].mu);
    }
    fclose(o);
    for (int i = 0; i < 3; i++) if (g_anom_img[i] && g_anom_img_f[i] + 2 >= fa && g_anom_img_f[i] <= fa + 1) {
        char pp[1024]; snprintf(pp, sizeof pp, "%s/a%d-F%llu.png", g_anom_dir, k, (unsigned long long)g_anom_img_f[i]); shot(g_anom_img[i], pp, 0, NULL);
    }
    MLOG("[anom] F%llu differs from previous %.1f, next %.1f (previous vs next %.1f): capture %d -> %s", (unsigned long long)fa, d1, d2, d0, k, path);
}
static void anom_present(id<MTLTexture> src, uint64_t frame) {
    if (!anom_on() || nogpu() || src.width < 64 || src.height < 64 || (src.pixelFormat != MTLPixelFormatBGRA8Unorm && src.pixelFormat != MTLPixelFormatRGBA8Unorm)) return;
    const NSUInteger W = src.width, H = src.height, ROWS = 4; const double ys[4] = {0.15, 0.35, 0.5, 0.65};
    if (!g_anom_alloc) {
        MTL4CommandAllocatorDescriptor *d = [MTL4CommandAllocatorDescriptor new]; d.label = @"TOTK anomaly probe"; NSError *e = nil;
        g_anom_alloc = [M3CommandAllocator new]; g_anom_cb = [g_q commandBuffer];
        if (!g_anom_alloc || !g_anom_cb) MDIE("anomaly probe allocation failed: %s", e.localizedDescription.UTF8String);
    }
    if (!g_anom_strip || g_anom_strip.length < 4 * W * ROWS * 4) {   // the last readback has completed (waited below): replace at once
        if (g_anom_strip) res_remove(g_anom_strip);
        g_anom_strip = [g_dev newBufferWithLength:4 * W * ROWS * 4 options:MTLResourceStorageModeShared];
        if (!g_anom_strip) MDIE("anomaly strip allocation failed");
        res_add(g_anom_strip);
    }
    int slot = (int)(frame % 3);
    if (!g_anom_img[slot] || g_anom_img[slot].width != W || g_anom_img[slot].height != H || g_anom_img[slot].pixelFormat != src.pixelFormat) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat width:W height:H mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
        if (g_anom_img[slot]) res_remove(g_anom_img[slot]);
        g_anom_img[slot] = [g_dev newTextureWithDescriptor:d]; if (!g_anom_img[slot]) MDIE("anomaly picture allocation failed");
        res_add(g_anom_img[slot]);
    }
    res_commit();   // new readback resources must be resident before this command buffer runs (a GPU fault otherwise: 2-4 GPU restarts at boot)
    ((void)0); g_anom_cb = [g_q commandBuffer];
#if DM_NVN_M3
    /* Metal 3 的 compute 编码器没有 copyFromTexture:toBuffer:/toTexture:；异常检测的读回要用真正的 blit 编码器。
       Metal 3 compute encoders have no copyFromTexture:toBuffer:/toTexture:; the anomaly readback needs a blit encoder. */
    id<MTLBlitCommandEncoder> be = [g_anom_cb blitCommandEncoder];
#else
    id<MTL4ComputeCommandEncoder> be = [g_anom_cb computeCommandEncoder];
#endif
    ((void)0);
    for (int r = 0; r < 4; r++)
        [be copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, (NSUInteger)(ys[r] * H), 0) sourceSize:MTLSizeMake(W, ROWS, 1)
                   toBuffer:g_anom_strip destinationOffset:(NSUInteger)r * W * ROWS * 4 destinationBytesPerRow:W * 4 destinationBytesPerImage:W * ROWS * 4];
    [be copyFromTexture:src toTexture:g_anom_img[slot]];
    m4_end(be); ((void)0);
    dispatch_semaphore_t done = dispatch_semaphore_create(0); __block int failed = 0;
    TotkCommitOptions *o = [TotkCommitOptions new]; [o addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
        failed = fb.error != nil; if (failed) MLOG("【红】[anom] GPU readback failed: %s", fb.error.localizedDescription.UTF8String);
        dispatch_semaphore_signal(done); }];
    q4_commit(g_anom_cb, o); dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    if (failed) return;
    g_anom_readbacks++;
    g_anom_img_f[slot] = frame;
    double v[ANOM_CELLS] = {0}; const uint8_t *px = g_anom_strip.contents; const NSUInteger cw = W / 32;
    for (int r = 0; r < 4; r++) for (int c = 0; c < 32; c++) {
        double sum = 0;
        for (NSUInteger y = 0; y < ROWS; y++) for (NSUInteger x = c * cw; x < (NSUInteger)(c + 1) * cw; x++) {
            const uint8_t *q = px + ((NSUInteger)r * ROWS + y) * W * 4 + x * 4; sum += 0.3 * q[0] + 0.59 * q[1] + 0.11 * q[2];
        }
        v[r * 32 + c] = sum / (double)(ROWS * cw);
    }
    memmove(g_anom_v[0], g_anom_v[1], sizeof g_anom_v[0] * 2); memcpy(g_anom_v[2], v, sizeof v);
    g_anom_vf[0] = g_anom_vf[1]; g_anom_vf[1] = g_anom_vf[2]; g_anom_vf[2] = frame;
    if (!g_anom_win0) g_anom_win0 = frame;
    if (g_anom_vf[0] && g_anom_vf[1] == g_anom_vf[0] + 1 && g_anom_vf[2] == g_anom_vf[1] + 1) {
        g_anom_triplets++;
        double d1 = anom_mad(g_anom_v[1], g_anom_v[0]), d2 = anom_mad(g_anom_v[1], g_anom_v[2]), d0 = anom_mad(g_anom_v[0], g_anom_v[2]);
        double score = fmin(d1, d2) - d0;
        if (score > g_anom_win_max) { g_anom_win_max = score; g_anom_win_f = g_anom_vf[1]; }
        if (d1 > ANOM_T && d2 > ANOM_T && d0 < 0.4 * fmin(d1, d2)) {
            g_anom_win_hits++; g_anom_total++;
            if (g_anom_captures < ANOM_CAPTURES && (!g_anom_last || g_anom_vf[1] > g_anom_last + 120)) { g_anom_last = g_anom_vf[1]; anom_dump(g_anom_vf[1], d1, d2, d0); }
            else MLOG("[anom] F%llu one-frame anomaly (previous %.1f, next %.1f, previous vs next %.1f); %llu so far", (unsigned long long)g_anom_vf[1], d1, d2, d0, (unsigned long long)g_anom_total);
        }
    }
    if (frame >= g_anom_win0 + 600) {
        MLOG("[anom] frames %llu-%llu: %llu anomalies (%llu in all); highest one-frame score %.1f at F%llu (threshold %.0f); successful readbacks %llu, consecutive triplets %llu", (unsigned long long)g_anom_win0, (unsigned long long)frame,
             (unsigned long long)g_anom_win_hits, (unsigned long long)g_anom_total, g_anom_win_max, (unsigned long long)g_anom_win_f, ANOM_T,
             (unsigned long long)g_anom_readbacks, (unsigned long long)g_anom_triplets);
        g_anom_win0 = frame; g_anom_win_max = 0; g_anom_win_hits = 0;
    }
}
int mtl_probe_anomaly(void) {
    mtl_init(); (void)anom_on();
    if (!g_anom_dir || !*g_anom_dir) MDIE("anomaly probe requires a fresh TOTK_ANOMALY directory");
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:64 height:64 mipmapped:NO];
    d.storageMode = MTLStorageModeShared; d.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> src = [g_dev newTextureWithDescriptor:d]; if (!src) MDIE("anomaly probe texture allocation"); res_add(src);
    uint8_t pixels[64 * 64 * 4]; memset(pixels, 0, sizeof pixels);
    [src replaceRegion:MTLRegionMake2D(0, 0, 64, 64) mipmapLevel:0 withBytes:pixels bytesPerRow:64 * 4];
    int saved = atomic_load(&g_anom_requested), checked = 0;
    atomic_store(&g_anom_requested, 0); anom_present(src, 10);
    if (g_anom_readbacks) MDIE("anomaly probe disabled detector read the GPU"); checked++;
    atomic_store(&g_anom_requested, 1);
    for (uint64_t frame = 11; frame <= 13; frame++) {
        memset(pixels, frame == 12 ? 200 : 0, sizeof pixels);
        [src replaceRegion:MTLRegionMake2D(0, 0, 64, 64) mipmapLevel:0 withBytes:pixels bytesPerRow:64 * 4];
        if (!(frame == 12 && getenv("TOTK_PROBE_NEGATIVE") && atoi(getenv("TOTK_PROBE_NEGATIVE")) == 13)) anom_present(src, frame);
    }
    if (g_anom_total != 1 || g_anom_last != 12) MDIE("anomaly probe failed to detect known single-frame flash"); checked++;
    for (int frame = 11; frame <= 13; frame++) { char path[1024]; struct stat st;
        snprintf(path, sizeof path, "%s/a1-F%d.png", g_anom_dir, frame);
        if (stat(path, &st) || st.st_size <= 0) MDIE("anomaly probe missing source/flash/next-frame capture"); checked++;
    }
    atomic_store(&g_anom_requested, 0); anom_present(src, 14);
    atomic_store(&g_anom_requested, 1);
    for (uint64_t frame = 15; frame <= 17; frame++) anom_present(src, frame);
    if (g_anom_total != 1 || g_anom_readbacks != 6 || g_anom_triplets != 2) MDIE("anomaly probe toggle/gap/readback coverage mismatch"); checked++;
    atomic_store(&g_anom_requested, saved); res_remove(src);
    return checked;
}
/* TOTK: what the screen actually showed: the interval between the presentedTime of consecutive drawables, game frames and
   MetalFX interpolated ones alike. On a display of hz, an interval longer than 1.5 refreshes means the previous picture
   stayed up for an extra refresh (a judder: uncapped play at 85 fps on a 100 Hz display repeats about every seventh
   refresh; interpolation should leave none while the game keeps up). One line per 600 pictures (6 s at 100 Hz). */
static int dcmp_mtl(const void *x, const void *y) { double a = *(const double *)x, b = *(const double *)y; return a < b ? -1 : a > b; }
static void cadence_note(double tt, int interp) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    enum { N = 600 };
    static double last, iv[N]; static int n, ni, hz; static uint64_t dropped;
    pthread_mutex_lock(&mu);
    if (tt <= 0) dropped++;
    else { if (last > 0 && tt > last && n < N) { iv[n++] = (tt - last) * 1e3; ni += interp; } if (tt > last) last = tt; }
    if (n == N) {
        if (!hz) { hz = mtl_display_hz(); if (hz <= 0) hz = 60; }
        double a[N]; memcpy(a, iv, sizeof a); qsort(a, N, sizeof *a, dcmp_mtl);
        double tick = 1000.0 / hz; int held = 0; for (int i = 0; i < N; i++) held += a[i] > 1.5 * tick;
        MLOG("上屏节奏（近 %d 张，至帧 %llu，显示器 %d Hz）：间隔 p50 %.2f / p95 %.2f / 最大 %.2f ms；多停一次刷新以上 %d 次（%.1f%%）；其中补帧 %d 张；系统丢弃 %llu 张",
             N, (unsigned long long)atomic_load(&g_cur_frame), hz, a[N / 2], a[N * 95 / 100], a[N - 1], held, 100.0 * held / N, ni, (unsigned long long)dropped);
        n = ni = 0; dropped = 0;
    }
    pthread_mutex_unlock(&mu);
}
static int present_trace_on(void) { static int on = -1; if (on < 0) on = getenv("TOTK_PRESENT_TRACE") != NULL; return on; }
static int present_now(const tex_t *t, double period, uint64_t frame, uint64_t ticket, const int crop[4], id<CAMetalDrawable> supplied, double target_ts, int interp) {
    if (g_present_stall_ns) { struct timespec ts = {0, (long)g_present_stall_ns}; nanosleep(&ts, NULL); }
    g_present_n++;
    /* TOTK: the FPS box counts game presents per second and, while MetalFX interpolates, the generated frames shown as
       well ("FPS 50+50"): the user found the plain displayed count misleading on the phone ("the number looks good, it
       feels slow"), so both are shown. */
    if (interp >= 0) { fps_overlay_note_gen(++g_gen_shown, CACurrentMediaTime()); atomic_store(&g_mfxi_last, frame ? frame : 1); }
    else fps_overlay_note(ticket, CACurrentMediaTime());
    if (g_use != USE_OFF && g_use != USE_DONE) use_tick();
    if (bootprof() && (g_present_n == 1 || g_present_n % 120 == 0)) {
        if (g_present_n == 1) g_bp_first_present = bp_now() - g_bp_t0;
        MLOG("[bootprof] t=%.3f s 出图第 %llu 次：metallib 载入 %llu 个 / %.1f ms；管线创建 %llu 个 / %.1f ms（均在提交线程同步）；预热领用 着色器 %llu / 渲染管线 %llu / 计算管线 %llu；"
             "贴图上传 %.1f MB / %llu 张首传 + %llu 次重传 / %.1f ms；首帧出图 t=%.3f s", (bp_now() - g_bp_t0) / 1e9, g_present_n,
             g_bp_lib_n, g_bp_lib_ns / 1e6, g_bp_pso_n, g_bp_pso_ns / 1e6, g_warm_claim_shaders, g_warm_claim_rps, g_warm_claim_cps,
             g_bp_up_bytes / 1048576.0, g_pkg_uploads, g_pkg_reuploads,
             g_up_total_ns / 1e6, g_bp_first_present / 1e9);
    }
    PD("X ser=%llu%s", t->serial, interp >= 0 ? " (MetalFX interpolated frame before it)" : "");
    id<MTLTexture> src = interp >= 0 ? mfx_output(interp) : tex_mtl(t);   // TOTK_MFX: the interpolated frame shown one tick before this game frame
    if (!src) { RED_ONCE(8, "出图贴图建不出 MTLTexture"); return 0; }
    if (interp < 0) anom_present(src, frame);   // TOTK_ANOMALY
    /* One-frame temporal fallback (TOTK_FLASH_GUARD=1, off by default): a same-queue readback of the picture's centre
       before the drawable is encoded; on an abrupt colour jump the previous good picture is shown again, at most twice
       in a row. It was a stopgap against the frame-wide flashes; their cause (compressible storage shared across formats,
       see stor_compressed) is fixed, and the guard also holds back genuine sudden changes (lightning, cuts) and waits for
       the GPU every frame. Offscreen benchmarks have no picture. */
    static int flash_guard = -1; if (flash_guard < 0) {
        const char *e = getenv("TOTK_FLASH_GUARD");
        flash_guard = e && atoi(e) != 0 && !getenv("TOTK_PRESENT_SKIP");
        MLOG("闪屏保护：%s（TOTK_FLASH_GUARD=1 打开；爆闪根因已修，见 stor_compressed）", flash_guard ? "开" : "关");
    }
    int flash_guard_update = 0;
    if (flash_guard && interp < 0 && !nogpu() && src.width >= 128 && src.height >= 128 &&
        (src.pixelFormat == MTLPixelFormatBGRA8Unorm || src.pixelFormat == MTLPixelFormatRGBA8Unorm)) {
        if (!g_flash_guard_alloc) { MTL4CommandAllocatorDescriptor *d = [MTL4CommandAllocatorDescriptor new]; d.label = @"TOTK flash guard probe";
            NSError *e = nil; g_flash_guard_alloc = [M3CommandAllocator new];
            g_flash_guard_cb = [g_q commandBuffer];
            g_flash_guard_sample = [g_dev newBufferWithLength:128 * 128 * 4 options:MTLResourceStorageModeShared];
            if (!g_flash_guard_alloc || !g_flash_guard_cb || !g_flash_guard_sample) MDIE("flash guard probe allocation failed: %s", e.localizedDescription.UTF8String);
            res_add(g_flash_guard_sample); }
        if (!g_flash_guard_previous || g_flash_guard_previous.width != src.width || g_flash_guard_previous.height != src.height || g_flash_guard_previous.pixelFormat != src.pixelFormat) {
            MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat width:src.width height:src.height mipmapped:NO];
            d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
            if (g_flash_guard_previous) defer_obj_release(g_flash_guard_previous);   // a present still in flight may read it
            g_flash_guard_previous = [g_dev newTextureWithDescriptor:d];
            if (!g_flash_guard_previous) MDIE("flash guard previous texture allocation failed");
            res_add(g_flash_guard_previous); g_flash_guard_ready = g_flash_guard_rejected = 0;
        }
        res_commit();   // TOTK: new readback resources must be resident before this command buffer runs
        ((void)0); g_flash_guard_cb = [g_q commandBuffer];
        id<MTL4ComputeCommandEncoder> be = [g_flash_guard_cb computeCommandEncoder];
        ((void)0);
        [be copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(src.width/2-64, src.height/2-64, 0)
            sourceSize:MTLSizeMake(128, 128, 1) toBuffer:g_flash_guard_sample destinationOffset:0 destinationBytesPerRow:512 destinationBytesPerImage:65536];
        m4_end(be); ((void)0);
        dispatch_semaphore_t done = dispatch_semaphore_create(0); __block int failed = 0;
        TotkCommitOptions *o = [TotkCommitOptions new]; [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
            failed = f.error != nil; dispatch_semaphore_signal(done); }];
        q4_commit(g_flash_guard_cb, o); dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        if (!failed) {
            const uint8_t *px = g_flash_guard_sample.contents; uint64_t sums[3] = {0};
            for (int q = 0; q < 128*128; q++) for (int c = 0; c < 3; c++) sums[c] += px[4*q+c];
            double now[3] = {sums[0]/16384.0, sums[1]/16384.0, sums[2]/16384.0};
            double max_delta = 0, total_delta = 0;
            for (int c = 0; c < 3; c++) { double delta = fabs(now[c] - g_flash_guard_good[c]); max_delta = fmax(max_delta, delta); total_delta += delta; }
            int reject = g_flash_guard_ready && g_flash_guard_rejected < 2 && max_delta > 90 && total_delta > 150;
            if (reject) { src = g_flash_guard_previous; g_flash_guard_rejected++;
                MLOG("[flash-guard] F%llu ticket=%llu replaced abrupt RGB %.1f,%.1f,%.1f (prior %.1f,%.1f,%.1f)",
                    (unsigned long long)frame, (unsigned long long)ticket, now[0], now[1], now[2],
                    g_flash_guard_good[0], g_flash_guard_good[1], g_flash_guard_good[2]); }
            else { memcpy(g_flash_guard_good, now, sizeof now); g_flash_guard_ready = 1; g_flash_guard_rejected = 0; flash_guard_update = 1; }
        }
    }
    // DM_NVN_REGLUM=x0,y0,x1,y1:文件（归一化坐标）：每次出图回读该区域，记「出图序号、出图时刻 ns、区域平均亮度、隔点灰度小图（十六进制）」（诊断：核动画速度与帧率无关）。
    // 旧版诊断队列 g_q 与实际上屏的 g_q4 不同；该读回只作粗略参考，精确对帧请用 DM_PRESENT_PROBE。
    // DM_NVN_REGLUM=x0,y0,x1,y1:file (normalized coordinates): read back this region on every present and record "present sequence number, present time ns, region mean luminance, subsampled grayscale thumbnail (hex)" (diagnostic: verify that animation speed is independent of frame rate).
    // This legacy readback uses g_q, while rendering and presenting use g_q4; use DM_PRESENT_PROBE for frame-accurate evidence.
    static int rl = -1; static float rr[4]; static FILE *rlf; static pthread_mutex_t rlmu = PTHREAD_MUTEX_INITIALIZER;
    if (rl < 0) { const char *e = getenv("DM_NVN_REGLUM"), *c = e ? strchr(e, ':') : NULL;
        rl = c && sscanf(e, "%f,%f,%f,%f", &rr[0], &rr[1], &rr[2], &rr[3]) == 4 && (rlf = fopen(c + 1, "w")) != NULL;
        if (e && !rl) MLOG("【红】DM_NVN_REGLUM 参数或文件无效：%s", e); }
    if (rl && (src.pixelFormat == MTLPixelFormatRGBA8Unorm || src.pixelFormat == MTLPixelFormatBGRA8Unorm || src.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB || src.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB)) {
        NSUInteger x0 = (NSUInteger)(rr[0] * src.width), y0 = (NSUInteger)(rr[1] * src.height), w = (NSUInteger)(rr[2] * src.width) - x0, h = (NSUInteger)(rr[3] * src.height) - y0;
        id<MTLBuffer> rb = [g_dev newBufferWithLength:w * h * 4 options:MTLResourceStorageModeShared];
        const char *hdr_dir = getenv("DM_NVN_FLASHHDR");
        id<MTLTexture> ht = hdr_dir && g_flash_hdr_source && g_flash_hdr_source.pixelFormat == MTLPixelFormatRG11B10Float ? g_flash_hdr_source : nil;
        NSUInteger hw = ht ? ht.width : 0, hh = ht ? ht.height : 0;
        id<MTLBuffer> hb = ht ? [g_dev newBufferWithLength:hw * hh * 4 options:MTLResourceStorageModeShared] : nil;
        id<MTLTexture> tt = hdr_dir && g_flash_tonemap_target && g_flash_tonemap_target.pixelFormat == MTLPixelFormatRG11B10Float ? g_flash_tonemap_target : nil;
        NSUInteger tw = tt ? tt.width : 0, th = tt ? tt.height : 0;
        id<MTLBuffer> tb = tt ? [g_dev newBufferWithLength:tw * th * 4 options:MTLResourceStorageModeShared] : nil;
        id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x0, y0, 0) sourceSize:MTLSizeMake(w, h, 1) toBuffer:rb destinationOffset:0 destinationBytesPerRow:w * 4 destinationBytesPerImage:w * h * 4];
        if (hb) [b copyFromTexture:ht sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(hw, hh, 1) toBuffer:hb destinationOffset:0 destinationBytesPerRow:hw * 4 destinationBytesPerImage:hw * hh * 4];
        if (tb) [b copyFromTexture:tt sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(tw, th, 1) toBuffer:tb destinationOffset:0 destinationBytesPerRow:tw * 4 destinationBytesPerImage:tw * th * 4];
        [b endEncoding];
        unsigned long long n = g_present_n, tns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { (void)c; const uint8_t *q = rb.contents; uint64_t sum = 0;
            uint8_t g[4096]; NSUInteger ng = 0;   // 64×64 以内的全区域缩略图，供逐帧图像相关分析 | Whole-region thumbnail, at most 64x64, for frame correlation
            NSUInteger sx = MAX((w + 63) / 64, 1u), sy = MAX((h + 63) / 64, 1u);
            for (NSUInteger y = 0; y < h; y += sy) for (NSUInteger x = 0; x < w && ng < sizeof g; x += sx) { const uint8_t *px = q + 4 * (y * w + x); g[ng++] = (uint8_t)((px[0] + px[1] + px[2]) / 3); }
            uint64_t channel[3] = {0,0,0}, magenta = 0;
            for (NSUInteger k = 0; k < w * h; k++) {
                uint8_t blue=q[4*k],green=q[4*k+1],red=q[4*k+2];
                channel[0]+=red;channel[1]+=green;channel[2]+=blue;
                magenta += red>50&&blue>50&&((int)red+(int)blue)/2-(int)green>18;
                sum += red+green+blue;
            }
            double mean = sum / (3.0 * w * h), magfrac = magenta / (double)(w * h);
            pthread_mutex_lock(&rlmu); fprintf(rlf, "%llu\t%llu\t%.4f\t", n, tns, mean);
            for (NSUInteger k = 0; k < ng; k++) fprintf(rlf, "%02x", g[k]);
            fprintf(rlf,"\t%.3f\t%.3f\t%.3f\t%.5f",channel[0]/(double)(w*h),channel[1]/(double)(w*h),channel[2]/(double)(w*h),magfrac);
            if (hb) { const uint32_t *v = hb.contents; uint64_t hs[3] = {0,0,0}, hn = 0;
                NSUInteger dx = MAX(hw / 64, 1u), dy = MAX(hh / 64, 1u);
                for (NSUInteger y = 0; y < hh; y += dy) for (NSUInteger x = 0; x < hw; x += dx) {
                    uint32_t z = v[y * hw + x]; hs[0] += z & 0x7ff; hs[1] += (z >> 11) & 0x7ff; hs[2] += (z >> 22) & 0x3ff; hn++; }
                fprintf(rlf,"\t%.2f\t%.2f\t%.2f",hs[0]/(double)hn,hs[1]/(double)hn,hs[2]/(double)hn); }
            fputc('\n',rlf);
            /* Diagnostic only: preserve a colour crop of a sudden bright or
               purple frame. The small history is updated in completion order
               under the same lock as the luminance record. */
            { static double history[8]; static unsigned count, next, saved_bright, saved_pink;
              static uint8_t *previous_hdr; static NSUInteger previous_hdr_size, previous_hw, previous_hh;
              static uint8_t *previous_tone; static NSUInteger previous_tone_size, previous_tw, previous_th;
              double baseline = 0; for (unsigned k = 0; k < count; k++) baseline += history[k];
              if (count) baseline /= count;
              const char *dir = getenv("DM_NVN_FLASHCAP");
              bool pink = magfrac > 0.30;
              bool bright = count == 8 && mean - baseline > 40;
              if (dir && ((pink && saved_pink < 20) || (!pink && bright && saved_bright < 20))) {
                  char name[1024]; snprintf(name, sizeof name, "%s/%s-present-%llu.ppm", dir, pink ? "pink" : "bright", n);
                  FILE *out = fopen(name, "wb");
                  if (out) { fprintf(out, "P6\n%lu %lu\n255\n", (unsigned long)w, (unsigned long)h);
                      uint8_t *rgb = malloc(w * h * 3);
                      if (rgb) { for (NSUInteger k = 0; k < w*h; k++) { rgb[3*k] = q[4*k+2]; rgb[3*k+1] = q[4*k+1]; rgb[3*k+2] = q[4*k]; }
                          fwrite(rgb, 3, w*h, out); free(rgb); if (pink) saved_pink++; else saved_bright++; }
                      fclose(out); }
                  if (hb && hdr_dir) { char hp[1024]; snprintf(hp, sizeof hp, "%s/hdr-%s-present-%llu.raw", hdr_dir, pink ? "pink" : "bright", n);
                      FILE *hf = fopen(hp, "wb"); if (hf) { fprintf(hf, "RG11B10F %lu %lu\n", (unsigned long)hw, (unsigned long)hh);
                          fwrite(hb.contents, 4, hw * hh, hf); fclose(hf); }
                      if (previous_hdr && previous_hdr_size == previous_hw * previous_hh * 4) {
                          snprintf(hp, sizeof hp, "%s/hdr-before-%s-present-%llu.raw", hdr_dir, pink ? "pink" : "bright", n);
                          hf = fopen(hp, "wb"); if (hf) { fprintf(hf, "RG11B10F %lu %lu\n", (unsigned long)previous_hw, (unsigned long)previous_hh);
                              fwrite(previous_hdr, 1, previous_hdr_size, hf); fclose(hf); } } }
                  if (tb && hdr_dir) { char tp[1024]; snprintf(tp, sizeof tp, "%s/tone-%s-present-%llu.raw", hdr_dir, pink ? "pink" : "bright", n);
                      FILE *tf = fopen(tp, "wb"); if (tf) { fprintf(tf, "RG11B10F %lu %lu\n", (unsigned long)tw, (unsigned long)th);
                          fwrite(tb.contents, 4, tw * th, tf); fclose(tf); }
                      if (previous_tone && previous_tone_size == previous_tw * previous_th * 4) {
                          snprintf(tp, sizeof tp, "%s/tone-before-%s-present-%llu.raw", hdr_dir, pink ? "pink" : "bright", n);
                          tf = fopen(tp, "wb"); if (tf) { fprintf(tf, "RG11B10F %lu %lu\n", (unsigned long)previous_tw, (unsigned long)previous_th);
                              fwrite(previous_tone, 1, previous_tone_size, tf); fclose(tf); } } }
              }
              if (hb && hdr_dir) { NSUInteger sz = hw * hh * 4;
                  if (sz != previous_hdr_size) { free(previous_hdr); previous_hdr = malloc(sz); previous_hdr_size = previous_hdr ? sz : 0; }
                  if (previous_hdr) { memcpy(previous_hdr, hb.contents, sz); previous_hw = hw; previous_hh = hh; } }
              if (tb && hdr_dir) { NSUInteger sz = tw * th * 4;
                  if (sz != previous_tone_size) { free(previous_tone); previous_tone = malloc(sz); previous_tone_size = previous_tone ? sz : 0; }
                  if (previous_tone) { memcpy(previous_tone, tb.contents, sz); previous_tw = tw; previous_th = th; } }
              history[next++ % 8] = mean; if (count < 8) count++;
            }
            fflush(rlf); pthread_mutex_unlock(&rlmu); }];
        cb_commit(cb);
    }
    // DM_NVN_SHOT=帧号[,帧号…]:路径（路径里的 %d 换成帧号）
    // DM_NVN_SHOT=frame number[,frame number...]:path (%d in the path is replaced with the frame number)
    static long shots[16]; static int nshot = -1; static const char *shot_path;
    if (nshot < 0) {
        nshot = 0; const char *e = getenv("DM_NVN_SHOT"), *c = e ? strchr(e, ':') : NULL;
        if (c) { shot_path = c + 1; for (const char *q = e; q < c && nshot < 16; q = strchr(q, ',') ? strchr(q, ',') + 1 : c) shots[nshot++] = atol(q); }
    }
    /* TOTK: DM_NVN_SHOT_FRAME=1 keys the shots on the game frame counter instead of the present count (an occluded window
       stalls presents, the game keeps running); the first present at or after the frame number takes the shot. */
    { static int by_frame = -1; static char done[16]; if (by_frame < 0) { const char *e = getenv("DM_NVN_SHOT_FRAME"); by_frame = e && atoi(e); }
      long cur = (long)g_present_n;
      for (int i = 0; i < nshot && interp < 0; i++) if (!by_frame && !done[i] && cur == shots[i]) { done[i] = 1; char path[1024]; snprintf(path, sizeof path, shot_path, (int)shots[i]); mtl_finish_gpu(); shot(src, path, 1, crop); } }

    CAMetalLayer *layer = (__bridge CAMetalLayer *)front_metal_layer();
    if (!layer) return 0;
    #if DM_NVN_M3
    /* CAMetalLayer.residencySet 是 macOS 26 的属性；macOS 15 上没有，可绘制对象的驻留由命令队列的驻留集覆盖。
       CAMetalLayer.residencySet is a macOS 26 property; macOS 15 has it not, and the drawable's residency
       is covered by the command queue's residency set. */
    (void)g_present_layer_resident;
#else
    if (g_present_layer_resident != layer) { if (!layer.residencySet) MDIE("present layer has no residency set (layer device %s)", layer.device ? "set" : "missing"); [g_q4 addResidencySet:layer.residencySet]; g_present_layer_resident = layer; }
#endif
#if TARGET_OS_OSX
    if (!layer.displaySyncEnabled) layer.displaySyncEnabled = YES;   // 窗口模式按显示器垂直同步，防撕裂 | Windowed mode follows the display's vsync to prevent tearing
#endif
    present_pipeline(layer.pixelFormat);   // 和原路径一样，首帧编管线在取drawable之前 | Same as the original path: the first frame compiles the pipeline before taking the drawable
    window_shot_tick(g_present_n);
    /* TOTK: whole-window shots (wshot, TOTK_WINDOW_SHOT) copy the drawable itself after the present has drawn into it: what
       the screen shows, MetalFX output upscale and boxes included (grab_present re-renders only for TOTK_PRESENT_SKIP). */
    void (^grab)(CGImageRef) = nil; NSString *grab_src = nil;
    if (g_grab_cb && interp < 0) { pthread_mutex_lock(&g_shotq_mu); grab = g_grab_cb; g_grab_cb = nil;
        if (g_grab_src_path[0]) { grab_src = @(g_grab_src_path); g_grab_src_path[0] = 0; } pthread_mutex_unlock(&g_shotq_mu); }
    /* TOTK_PRESENT_SKIP=1 (measurement only): never take a drawable, so the render pipeline's own frame rate shows (the window
       does not update). */
    static int present_skip = -1; if (present_skip < 0) { present_skip = getenv("TOTK_PRESENT_SKIP") != NULL; if (present_skip) MLOG("【实验】TOTK_PRESENT_SKIP=1：不出图，只测渲染流水线"); }
    if (nogpu() || present_skip) { if (grab && present_skip && !nogpu()) grab_present(grab, src, crop, layer.drawableSize);
        pthread_mutex_lock(&g_pq_mu); if (ticket > g_pq_done) g_pq_done = ticket; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu); return 1; }
    uint64_t dt0 = now_ns();
    id<CAMetalDrawable> dr = supplied ? supplied : [layer nextDrawable];
    if (!supplied) g_drawable_ns += now_ns() - dt0;
    if (!dr) return 0;
    if (present_trace_on()) MLOG("[present-trace] ticket=%llu %s：取到 drawable %lux%lu 格式 %lu",
        (unsigned long long)ticket, supplied ? "节拍" : "自取", (unsigned long)dr.texture.width, (unsigned long)dr.texture.height, (unsigned long)dr.texture.pixelFormat);
#if DM_NVN_M3
    /* 判定用：TOTK_ACQUIRE_SKIP_DRAW=1 仍然取 drawable，但跳过 present_encode_m4（不往 dr.texture 里画），
       只把这张未画的 drawable present 出去让图层回收；用来把「取 drawable」与「渲染进 drawable」分开。
       同时给 TOTK_ACQUIRE_ONLY=1 就连 present 也不做（纯 nextDrawable 测试；自己取 drawable 的模式下三张用完后会停住）。
       Diagnostic: TOTK_ACQUIRE_SKIP_DRAW=1 still takes the drawable but skips present_encode_m4 (nothing is drawn
       into dr.texture), presenting the untouched drawable only so the layer recycles it; this splits "take the
       drawable" from "render into it". With TOTK_ACQUIRE_ONLY=1 it does not present either (a pure nextDrawable
       test; in the self-acquire mode it stalls once the three drawables are used up). */
    static int acquire_skip_draw = -1;
    if (acquire_skip_draw < 0) { acquire_skip_draw = getenv("TOTK_ACQUIRE_SKIP_DRAW") != NULL;
        if (acquire_skip_draw) MLOG("【实验】TOTK_ACQUIRE_SKIP_DRAW=1：照常取 drawable，但不渲染进它的纹理（present 一张未画的图，除非同时有 TOTK_ACQUIRE_ONLY=1）"); }
    if (acquire_skip_draw) {
        if (grab) grab(NULL);
        static int acq_only = -1; if (acq_only < 0) acq_only = getenv("TOTK_ACQUIRE_ONLY") != NULL;
        TotkCommitOptions *so = [TotkCommitOptions new];
        [so addFeedbackHandler:^(id<MTL4CommitFeedback> f) { (void)f;
            pthread_mutex_lock(&g_pq_mu); g_pq_inflight--; if (ticket > g_pq_done) g_pq_done = ticket;
            pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu); }];
        id<MTL4CommandBuffer> scb = [g_q commandBuffer];
        if (!acq_only) [scb presentDrawable:dr];
        pthread_mutex_lock(&g_pq_mu); g_pq_inflight++; pthread_mutex_unlock(&g_pq_mu);
        q4_commit(scb, so);
        if (present_trace_on()) MLOG("[present-trace] ticket=%llu：跳过渲染，%s", (unsigned long long)ticket, acq_only ? "也不 present" : "已 present 未画的 drawable");
        return 1;
    }
#endif
    int alloc_i = (int)((ticket * 2 + (interp >= 0 ? 0 : 1)) % 3);   // the interpolated frame of a ticket and the ticket's own frame use different allocators
    pthread_mutex_lock(&g_pq_mu);
    while (g_present_alloc_busy[alloc_i]) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
    pthread_mutex_unlock(&g_pq_mu);
    ((void)0);
    ((void)0);
    id<MTL4CommandBuffer> cb = g_present_cb[alloc_i];
    cb = [g_q commandBuffer];
    if (!present_encode_m4(cb, src, dr.texture, interp >= 0 ? NULL : crop, alloc_i)) { ((void)0); ((void)0); if (grab) grab(NULL); return 0; }   // the interpolated frame has the canvas size: no swapchain crop
    if (present_trace_on()) MLOG("[present-trace] ticket=%llu：已编码进 drawable", (unsigned long long)ticket);
    id<MTLBuffer> grab_buf = nil; size_t grab_w = dr.texture.width, grab_h = dr.texture.height;
    if (grab) {
        grab_buf = grab_w && grab_h ? [g_dev newBufferWithLength:grab_w * grab_h * 4 options:MTLResourceStorageModeShared] : nil;
        if (grab_buf) {
            res_add(grab_buf); res_commit();   // resident before this command buffer runs (lesson 55)
            id<MTL4ComputeCommandEncoder> ge = enc_b_begin(cb);
            ((void)0);
            [ge copyFromTexture:dr.texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(grab_w, grab_h, 1)
                       toBuffer:grab_buf destinationOffset:0 destinationBytesPerRow:grab_w * 4 destinationBytesPerImage:grab_w * grab_h * 4];
            m4_end(ge);
        } else { MLOG("【红】整窗截图：%zux%zu 读回缓冲建不出", grab_w, grab_h); grab(NULL); grab = nil; }
    }
    id<MTLBuffer> src_buf = nil; size_t src_w = src.width, src_h = src.height;   // TOTK_WSHOT_SRC: the source picture of this present too
    if (grab && grab_src && src.pixelFormat == MTLPixelFormatRGBA8Unorm) {
        src_buf = [g_dev newBufferWithLength:src_w * src_h * 4 options:MTLResourceStorageModeShared];
        if (src_buf) { res_add(src_buf); res_commit();
            id<MTL4ComputeCommandEncoder> ge = enc_b_begin(cb);
            ((void)0);
            [ge copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(src_w, src_h, 1)
                       toBuffer:src_buf destinationOffset:0 destinationBytesPerRow:src_w * 4 destinationBytesPerImage:src_w * src_h * 4];
            m4_end(ge); }
    }
    if (flash_guard_update) { id<MTL4ComputeCommandEncoder> be = enc_b_begin(cb);
        ((void)0);
        [be copyFromTexture:src sourceSlice:0 sourceLevel:0 toTexture:g_flash_guard_previous destinationSlice:0 destinationLevel:0 sliceCount:1 levelCount:1];
        m4_end(be); }
    int probe = interp < 0 && g_present_probe && src.width >= 128 && src.height >= 128 && dr.texture.width >= 128 && dr.texture.height >= 128 &&
        (src.pixelFormat == MTLPixelFormatBGRA8Unorm || src.pixelFormat == MTLPixelFormatRGBA8Unorm) && dr.texture.pixelFormat == MTLPixelFormatBGRA8Unorm;
    if (probe) {
        id<MTL4ComputeCommandEncoder> b = enc_b_begin(cb);
        ((void)0);
        [b copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(src.width / 2 - 64, src.height / 2 - 64, 0)
            sourceSize:MTLSizeMake(128, 128, 1) toBuffer:g_present_probe_src[alloc_i] destinationOffset:0 destinationBytesPerRow:512 destinationBytesPerImage:65536];
        [b copyFromTexture:dr.texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(dr.texture.width / 2 - 64, dr.texture.height / 2 - 64, 0)
            sourceSize:MTLSizeMake(128, 128, 1) toBuffer:g_present_probe_dst[alloc_i] destinationOffset:0 destinationBytesPerRow:512 destinationBytesPerImage:65536];
        m4_end(b);
    }
    ((void)0);
    TotkCommitOptions *o = [TotkCommitOptions new];
    { int timed = alloc_i >= 0 && alloc_i < 3 && g_outup_timed[alloc_i], slot = alloc_i;
      [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
          sysload_note_gpu(f.GPUStartTime, f.GPUEndTime);   // the load line's game GPU share includes the present (and its MetalFX upscale)
          if (timed && !f.error) outup_note_gpu(slot); }]; }
    if (src_buf) { id<MTLBuffer> buf = src_buf; size_t w = src_w, h = src_h; NSString *path = grab_src;
        [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
            if (!f.error) { CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
                CGContextRef ctx = CGBitmapContextCreate(buf.contents, w, h, 8, w * 4, cs, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
                CGImageRef img = ctx ? CGBitmapContextCreateImage(ctx) : NULL; window_shot_save(img, path);
                if (img) CGImageRelease(img); if (ctx) CGContextRelease(ctx); CGColorSpaceRelease(cs); }
            res_remove(buf); }]; }
    if (grab) { void (^g)(CGImageRef) = grab; id<MTLBuffer> buf = grab_buf; size_t w = grab_w, h = grab_h;
        [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
            CGImageRef img = NULL; CGContextRef ctx = NULL; CGColorSpaceRef cs = NULL;
            if (!f.error) { cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
                ctx = CGBitmapContextCreate(buf.contents, w, h, 8, w * 4, cs, kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
                img = ctx ? CGBitmapContextCreateImage(ctx) : NULL; }
            g(img);
            if (img) CGImageRelease(img); if (ctx) CGContextRelease(ctx); if (cs) CGColorSpaceRelease(cs);
            res_remove(buf); }]; }
    [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
        if (f.error) MLOG("【红】上屏命令缓冲错误：%s", f.error.description.UTF8String);
        if (probe && !f.error) {
            const unsigned char *s = g_present_probe_src[alloc_i].contents, *d = g_present_probe_dst[alloc_i].contents;
            unsigned long long sv = 0, dv = 0, srgb[3] = {0}, drgb[3] = {0}; int sb = 0, db = 0, sm = 0, dm = 0;
            for (int k = 0; k < 128 * 128; k++) {
                const unsigned char *sp = s + 4*k, *dp = d + 4*k;
                sv += sp[0] + sp[1] + sp[2]; dv += dp[0] + dp[1] + dp[2];
                sb += sp[0] > 230 && sp[1] > 220 && sp[2] > 230;
                db += dp[0] > 230 && dp[1] > 220 && dp[2] > 230;
                for (int c = 0; c < 3; c++) { srgb[c] += sp[c]; drgb[c] += dp[c]; }
                sm += sp[0] > 50 && sp[2] > 50 && ((int)sp[0] + (int)sp[2]) / 2 - (int)sp[1] > 18;
                dm += dp[0] > 50 && dp[2] > 50 && ((int)dp[0] + (int)dp[2]) / 2 - (int)dp[1] > 18;
            }
            if (getenv("DM_PRESENT_PROBE_LOG") && ticket > 2500)
                MLOG("[present-probe-rgb] ticket=%llu frame=%llu source=%.1f,%.1f,%.1f pink=%d drawable=%.1f,%.1f,%.1f pink=%d",
                    (unsigned long long)ticket, (unsigned long long)frame, srgb[0]/16384.0, srgb[1]/16384.0, srgb[2]/16384.0, sm,
                    drgb[0]/16384.0, drgb[1]/16384.0, drgb[2]/16384.0, dm);
            if (db > 1024 || sb > 1024 || llabs((long long)dv - (long long)sv) > 128 * 128 * 100)
                MLOG("[present-probe] ticket=%llu source avg=%.1f bright=%d drawable avg=%.1f bright=%d", (unsigned long long)ticket,
                    (double)sv / (128 * 128 * 3), sb, (double)dv / (128 * 128 * 3), db);
            if (getenv("DM_PRESENT_PROBE_DUMP") && ticket > 3000 && sv > 128ull * 128 * 3 * 220 && sb > 4000 && g_td_frame == -1) {
                int expected = 0;
                if (atomic_compare_exchange_strong(&g_td_auto_armed, &expected, 1)) {
                    long target = (long)atomic_load(&g_cur_frame) + 3;
                    atomic_store(&g_td_frame, target);
                    MLOG("[present-probe] triggered render-target dump at game frame %ld (ticket %llu)", target, (unsigned long long)ticket);
                }
            }
        }
        pthread_mutex_lock(&g_pq_mu); g_present_alloc_busy[alloc_i] = 0; if (interp < 0 && ticket > g_pq_done) g_pq_done = ticket; g_pq_inflight--; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
    }];
    pthread_mutex_lock(&g_pq_mu); g_present_alloc_busy[alloc_i] = 1; g_pq_inflight++; pthread_mutex_unlock(&g_pq_mu);
#if DM_NVN_M3
    /* Metal 3 的呈现必须挂在命令缓冲上（presentDrawable:），系统才会等 GPU 完成这一帧再换屏。
       老接口下 CAMetalDrawable 的 present / presentAfterMinimumDuration: 是不等 GPU 的旧式用法；
       两者同时走会让同一张 drawable 被呈现两次、并可能在 GPU 还在使用时被回收 —— 实测表现是
       “Each CAMetalLayerDrawable can only be presented once!” 之后崩在 AGX 驱动内部。
       Metal 3 present must be attached to the command buffer (presentDrawable:) so the system waits for the
       GPU to finish the frame. Under the older API CAMetalDrawable's present / presentAfterMinimumDuration:
       are the legacy form that does not wait; doing both presents the same drawable twice and can recycle it
       while the GPU still uses it — measured as "Each CAMetalLayerDrawable can only be presented once!"
       followed by a crash inside the AGX driver. */
    /* 判定用：TOTK_ACQUIRE_ONLY=1 只取 drawable、不呈现，用来把「取图」与「上屏」分开 */
    /* Diagnostic: TOTK_ACQUIRE_ONLY=1 acquires the drawable but never presents, splitting acquisition from present. */
    static int acquire_only = -1;
    if (acquire_only < 0) acquire_only = getenv("TOTK_ACQUIRE_ONLY") != NULL;
    if (!acquire_only) [cb presentDrawable:dr];
#endif
    q4_commit(cb, o);
#if DM_NVN_M3
    if (present_trace_on()) MLOG("[present-trace] ticket=%llu：已提交（%s）", (unsigned long long)ticket, acquire_only ? "未 present" : "已 present");
#else
    if (present_trace_on()) MLOG("[present-trace] ticket=%llu：已提交", (unsigned long long)ticket);
#endif
    if (target_ts > 0) {
        static double leads[300], e2e[300];   // e2e：游戏入队 → 系统预定上屏（含在队列里等节拍回调的时间），输入延迟里我们这一侧能读到的全程 | e2e: game enqueue -> system-scheduled present (including the time waiting in the queue for the display-link callback), the full span of input latency that our side can read
        static uint64_t lead_n, e2e_n;
        leads[lead_n % 300] = (target_ts - CACurrentMediaTime()) * 1000.0;
        if (interp < 0) e2e[e2e_n++ % 300] = g_link_it_tin > 0 ? (target_ts - g_link_it_tin) * 1000.0 : 0;   // TOTK: game frames only (with MetalFX interpolation the game frame is shown one refresh after the interpolated one)
        lead_n++;
        if (lead_n % 300 == 0) {
            double a[300]; memcpy(a, leads, sizeof a);
            for (int i = 1; i < 300; i++) { double v = a[i]; int j = i - 1; while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; } a[j + 1] = v; }
            double b[300]; memcpy(b, e2e, sizeof b); int nb = e2e_n < 300 ? (int)e2e_n : 300; if (nb < 1) nb = 1;
            for (int i = 1; i < nb; i++) { double v = b[i]; int j = i - 1; while (j >= 0 && b[j] > v) { b[j + 1] = b[j]; j--; } b[j + 1] = v; }
            MLOG("显示节拍 至 %llu：节拍 300、出图 %llu、空节拍 %llu、合并丢弃旧帧 %llu；提交领先目标上屏 p50 %.3f ms / 最小 %.3f ms；游戏帧入队到目标上屏 p50 %.3f ms / 最小 %.3f / 最大 %.3f ms",
                 (unsigned long long)g_link_tick_n, (unsigned long long)g_present_n, (unsigned long long)(g_link_tick_n - g_present_n),
                 (unsigned long long)g_link_dropped, a[149], a[0], b[nb / 2], b[0], b[nb - 1]);
        }
    }
    ((void)0);
#if TARGET_OS_SIMULATOR   // iOS 模拟器 SDK 没有呈现时刻回调与最短显示时长（只作画面验收用，不计节拍） | The iOS simulator SDK has no presentation-time callback and no minimum display duration (used only for picture acceptance, display-link ticks are not counted)
    (void)period; [dr present];
#else
    uint64_t pn = ticket;   // 按游戏出图序号（票号）记上屏时刻：hle 的上屏统计按「游戏第 n 次出图」取；显示节拍路径里实际出图次数 ≠ 游戏出图次数（旧帧会被合并丢弃） | Record the present time by the game's present sequence number (ticket number): hle's present statistics are taken by "the game's nth present"; in the display link path the actual present count != the game's present count (old frames get merged and dropped)
#if DM_NVN_M3
    /* The M3 display link already presented this drawable before the ticket reaches here; adding a presented
       handler now is rejected by Metal ("cannot be called after drawable has been presented") and only spams the
       log. The frontend still needs the first-frame notification; call it directly. */
    (void)pn;
    front_frame_presented();
#else
    [dr addPresentedHandler:^(id<MTLDrawable> d) {   // 真实呈现时刻（屏幕上真正换上这一帧的时间，CACurrentMediaTime 基准） | Actual presentation time (the time this frame is really put on screen, CACurrentMediaTime base)
        CFTimeInterval tt = d.presentedTime;
        if (interp < 0) g_pres[pn % PRING] = (pres_ent){pn, tt > 0 ? tt : -1};   // statistics per game frame; the interpolated one is not a game frame
        cadence_note(tt, interp >= 0);
        front_frame_presented();
    }];
#endif
    // 每张至少显示一个出图周期：GPU 完成时刻在 120 Hz 刷新边界附近抖动时，不再出现「25 ms + 8.33 ms」成对的顿挫
    // Each frame is displayed for at least one present period: when the GPU completion time jitters around a 120 Hz refresh boundary, the paired "25 ms + 8.33 ms" stutter no longer appears
#if !DM_NVN_M3
    if (target_ts > 0) [dr present];
    else if (period > 0) [dr presentAfterMinimumDuration:period]; else [dr present];
#endif
#endif
    (void)frame;
    return 1;
}
// ---------- 出图线程 ----------
// NVN 语义：QueuePresentTexture 只把出图排进队列、立即返回（游戏常持锁调用它；iOS 真机：在里面同步等 nextDrawable
// 约 30 ms，锁被占约 90%，装载线程饿死）。取 drawable、blit、present 都在这条线程上按入队顺序做。
// 票号 = 第几次入队（从 1 起）；g_pq_done = 已处理完（blit 已提交到 GPU 队列）的票号。
// 交换链贴图归还：AcquireTexture 拿到某张前等它上一次出图的票号处理完（mtl_present_wait），之后游戏再往里画的命令
// 在 GPU 队列上必排在那次 blit 之后。
// ---------- Present thread ----------
// NVN semantics: QueuePresentTexture only queues the present and returns immediately (the game often calls it while holding a lock; iOS real device: synchronously waiting for nextDrawable inside it
// takes about 30 ms, the lock is held about 90% of the time, and the loading thread starves). Taking the drawable, blit and present are all done on this thread in enqueue order.
// Ticket number = which enqueue it is (starting from 1); g_pq_done = the ticket number that has been processed (blit committed to the GPU queue).
// Swapchain texture return: before AcquireTexture gets a texture it waits for the ticket of that texture's previous present to be processed (mtl_present_wait), after which the commands the game draws into it
// are necessarily ordered after that blit on the GPU queue.
#define PQ 8
typedef struct { tex_t t; double period; uint64_t frame; int crop[4]; double t_in; int interp, interp_shown; double interp_ts; } pq_item;   // interp (TOTK_MFX): output slot of the frame shown before this one, or -1; interp_ts: the refresh it was shown at   // t_in：游戏把这一帧交进出图队列的时刻（CACurrentMediaTime） | t_in: the time the game handed this frame into the present queue (CACurrentMediaTime)
static pq_item g_pq[PQ];
static uint64_t g_pq_in;
static int present_now(const tex_t *, double, uint64_t, uint64_t, const int[4], id<CAMetalDrawable>, double, int);

@interface NvMetalDisplayLinkDelegate : NSObject <CAMetalDisplayLinkDelegate>
@end

// 前台节拍停了的兜底（两次合入冒烟卡死：前台时节拍不再回调，排队帧没人取，游戏在「取下一张交换链贴图」前等上一次出图处理完，永远等下去）：
// mtl_present_wait 等超过 STALL_NS 且节拍这么久都没回调，就照进后台那样把等着的这张及更早的排队帧丢掉（按掉帧记账），放游戏继续跑；节拍回来自动照常。
// 一段停顿记一行线索、恢复时记一行（都不是红）；同一段停顿连着超过 STALL_RED_NS 才红。全在 g_pq_mu 下
// Fallback for the display link going silent while in the foreground (two smoke-test hangs before merge: the display link stopped calling back while the app was in the foreground, nobody takes the queued frames, and the game waits forever for its previous present to be processed before "acquiring the next swapchain texture"):
// when mtl_present_wait has waited longer than STALL_NS and the display link has not called back for that long, drop the waiting frame and earlier queued frames as when entering the background (accounted as dropped frames) and let the game keep running; when the display link comes back things return to normal automatically.
// One stall logs one hint line, and one line on recovery (neither is a failure); only when the same stall lasts longer than STALL_RED_NS is it a failure. All under g_pq_mu
#define STALL_NS 100000000ull
#define STALL_RED_NS 2000000000ull
static uint64_t g_stall_t0, g_stall_drop, g_stall_n; static int g_stall_red, g_stall_logged;
// 显示节拍看门狗只认「激活」状态：App 失去活跃（下拉控制中心 / 通知、来电、系统弹窗）和后台里系统本来就不回调，画面也被盖住——丢帧兜底照旧，
// 但不算卡住、不报红；停顿从「最近一次回调 / 最近一次激活」较晚的那个算起（A16 日志：失活 8–17 秒、后台 17 秒都被报成「前台卡住」）
// The display link watchdog only recognizes the "active" state: when the App loses active status (pulling down Control Center / notifications, incoming call, system popup) and in the background the system does not call back anyway, and the picture is covered too -- the frame-drop fallback stays as before,
// but it is not counted as stuck and no failure is reported; the stall is measured from the later of "the most recent callback / the most recent activation" (A16 logs: 8–17 seconds inactive and 17 seconds in the background were both reported as 「前台卡住」(stuck in foreground))
__attribute__((weak)) int runtime_front_window_visible(void) { return -1; }   // runtime/frontend.m (macOS window occlusion); unknown in the graphics unit tests and on iOS
static const char *window_seen(void) { int v = runtime_front_window_visible(); return v > 0 ? "窗口可见" : v == 0 ? "窗口被挡住或在别的桌面：这段的出图与上屏数据无效" : "窗口可见性未知"; }
__attribute__((weak)) int dm_focus_now(void) { return 1; }   // 宿主 hle_oe.c 提供强定义；图形单测不链宿主时视为一直激活 | The host's hle_oe.c provides the strong definition; when graphics unit tests do not link the host it is treated as always active
__attribute__((weak)) uint64_t dm_focus_in_since(void) { return 0; }
@implementation NvMetalDisplayLinkDelegate
- (void)metalDisplayLink:(CAMetalDisplayLink *)link needsUpdate:(CAMetalDisplayLinkUpdate *)update {
    (void)link;
    pthread_mutex_lock(&g_pq_mu);
    if (g_stall_t0) { uint64_t since = dm_focus_in_since(), t = now_ns();
        MLOG("显示节拍恢复：这段停了 %.0f ms（其中激活中 %.0f ms），兜底丢了 %llu 帧（第 %llu 段）", (t - g_stall_t0) / 1e6, (t - (g_stall_t0 > since ? g_stall_t0 : since)) / 1e6, (unsigned long long)g_stall_drop, (unsigned long long)g_stall_n); g_stall_t0 = 0; }
    pthread_mutex_unlock(&g_pq_mu);
    pthread_mutex_lock(&g_link_mu);
    g_link_tick_ns = now_ns();
    g_link_tick_n++;
    pthread_cond_broadcast(&g_link_cv);
    pthread_mutex_unlock(&g_link_mu);
    pthread_mutex_lock(&g_pq_mu);
    if (!g_dm_present_link || g_pq_read == g_pq_in) { pthread_mutex_unlock(&g_pq_mu); return; }
    uint64_t first = g_pq_read + 1, ticket = g_pq_in;
    pq_item *ip = &g_pq[(g_pq_in - 1) % PQ];
    int interp = ip->interp >= 0 && !ip->interp_shown ? ip->interp : -1;   // TOTK_MFX: this tick shows the interpolated frame, a later one the game frame
    double tts = update.targetPresentationTimestamp;
    if (interp < 0 && ip->interp >= 0 && tts > 0 && ip->interp_ts > 0) {
        /* TOTK_MFX: the game frame goes up at the refresh nearest to half a game frame after its interpolated one, so the
           two pictures share the frame's time evenly (30+30 on a 100 Hz display: two refreshes later, not one; half the
           refresh rate and uncapped: the next refresh). Until then the interpolated picture stays up. */
        static double tick; if (!tick) { int hz = mtl_display_hz(); tick = 1.0 / (hz > 0 ? hz : 60); }
        if (tts < ip->interp_ts + ip->period / 2 - tick / 2) { pthread_mutex_unlock(&g_pq_mu); return; }
    }
    if (interp >= 0) { ip->interp_shown = 1; ip->interp_ts = tts; }
    pq_item it = *ip; g_link_it_tin = it.t_in;
    g_pq_read = interp >= 0 ? g_pq_in - 1 : g_pq_in;
    if (ticket > first) { g_link_dropped += ticket - first; for (uint64_t k = first; k < ticket; k++) g_pres[k % PRING] = (pres_ent){k, -1}; }   // 被合并丢弃的旧帧记「未上屏」，上屏统计如实算掉帧 | Old frames merged and dropped are recorded as "not presented", so the present statistics honestly count them as dropped frames
    if (ticket > 1 && g_pq_done < ticket - 1) g_pq_done = ticket - 1;
    pthread_cond_broadcast(&g_pq_cv);
    pthread_mutex_unlock(&g_pq_mu);
    int submitted;
    @autoreleasepool { submitted = present_now(&it.t, it.period, it.frame, ticket, it.crop, update.drawable, update.targetPresentationTimestamp, interp); }
    if (!submitted && interp < 0) {
        pthread_mutex_lock(&g_pq_mu);
        if (ticket > g_pq_done) g_pq_done = ticket;
        pthread_cond_broadcast(&g_pq_cv);
        pthread_mutex_unlock(&g_pq_mu);
    }
}
@end

static CAMetalDisplayLink *g_link; static CFRunLoopRef g_link_rl; static int g_link_started;
static int g_link_owns_layer, g_classic_busy;   // 持 g_pq_mu：图层挂着节拍对象期间 nextDrawable 会抛异常，自取 drawable 的出图线程与节拍对象互斥 | Under g_pq_mu: nextDrawable throws while the layer has the display link object attached, so the present thread that takes its own drawable and the display link object are mutually exclusive
static int link_on(void) { return g_dm_present_link && g_link != nil; }
static CAFrameRateRange link_range(int mode) {
    if (mode == 3) return CAFrameRateRangeMake(30, 30, 30);
    if (mode == 2) return CAFrameRateRangeMake(60, 120, 120);
    const char *explicit_hz = getenv("DM_PRESENT_HZ"), *rt = getenv("TOTK_REALTIME");
    int hz = explicit_hz ? atoi(explicit_hz) : (rt && !strcmp(rt, "0")) ? 60 : mtl_display_hz();
    if (hz < 30 || hz > 120) hz = 60;
    return CAFrameRateRangeMake(hz, hz, hz);
}
static int g_link_mode = 1;   // 节拍对象当前的模式：1 / 2 同 g_dm_present_link，3 = 省电模式 30 Hz | Current mode of the display link object: 1 / 2 same as g_dm_present_link, 3 = power-saving mode 30 Hz
static void *link_main(void *arg) {   // 显示节拍线程：切到节拍模式时起，切回 0 时退出 | Display link thread: started when switching to display-link mode, exits when switching back to 0
    (void)arg;
    pthread_setname_np("nvn-present-link");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    CAMetalLayer *layer = (__bridge CAMetalLayer *)front_metal_layer();
    pthread_mutex_lock(&g_pq_mu); g_link_owns_layer = 1; while (g_classic_busy) pthread_cond_wait(&g_pq_cv, &g_pq_mu); pthread_mutex_unlock(&g_pq_mu);   // 等出图线程手上那一帧交完 | Wait for the present thread to finish handing over the frame it holds
    NvMetalDisplayLinkDelegate *delegate = [NvMetalDisplayLinkDelegate new];
    CAMetalDisplayLink *link = [[CAMetalDisplayLink alloc] initWithMetalLayer:layer];
    link.preferredFrameRateRange = link_range(g_link_mode);
    link.preferredFrameLatency = g_present_latency;
    link.delegate = delegate;
    [link addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
    g_link_rl = CFRunLoopGetCurrent(); g_link = link;
    CFRunLoopRun();   // 切回模式 0 时节拍对象作废、本线程退出（图层还挂着节拍对象时 nextDrawable 取不到图，不能只暂停） | When switching back to mode 0 the display link object is invalidated and this thread exits (nextDrawable cannot get a drawable while the layer still has the display link object attached, so pausing alone is not enough)
    g_link_started = 0;
    return NULL;
}
static void link_sync(void) {   // 游戏出图线程每帧调：把节拍对象的状态对齐到 g_dm_present_link 与省电模式（运行中可改） | Called every frame by the game's present thread: aligns the display link object's state to g_dm_present_link and the power-saving mode (can change at run time)
    static int applied; int want = g_dm_present_link;
    if (want < 0 || want > 2) { g_dm_present_link = applied == 3 ? 1 : applied; return; }
    if (want && mtl_fps_target() == 30) want = 3;   // 省电模式：显示节拍 30 Hz | Power-saving mode: display link at 30 Hz
    if (want == applied) return;
    g_link_mode = want;
    if (!g_link_started) {
        if (!want) { applied = 0; return; }
        if (!front_metal_layer()) return;
        g_link_started = 1; pthread_t th; if (pthread_create(&th, NULL, link_main, NULL)) MDIE("显示节拍线程起不来"); pthread_detach(th);
        return;   // 节拍对象在新线程里建，下一帧再对齐 | The display link object is created on a new thread; align again next frame
    }
    CAMetalDisplayLink *link = g_link; if (!link) return;
    if (want) link.preferredFrameRateRange = link_range(want);
    else { CFRunLoopRef rl = g_link_rl; g_link = nil; CFRunLoopPerformBlock(rl, kCFRunLoopDefaultMode, ^{ [link invalidate]; pthread_mutex_lock(&g_pq_mu); g_link_owns_layer = 0; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu); CFRunLoopStop(rl); }); CFRunLoopWakeUp(rl); }
    MLOG("出图节拍模式 %d → %d（帧 %llu）", applied, want, (unsigned long long)atomic_load(&g_cur_frame));
    applied = want;
    pthread_mutex_lock(&g_pq_mu); pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
}
// 进后台前（mtl_drain）：显示节拍驱动出图时，排着队还没上屏的帧只能由节拍回调取走，而后台里系统不再回调——
// 原样等「出图队列清空」会永远等下去，进后台的回调不返回，系统把进程杀掉（切到别的 App 后 22 秒内进程消失）。
// 这些帧反正不会被看到：直接丢弃（按掉帧记账），只等已经提交给 GPU 的那一次上屏做完。
// Before going to the background (mtl_drain): when the display link drives presents, frames queued but not yet presented can only be taken by the display-link callback, and in the background the system no longer calls back --
// waiting as-is for "the present queue to empty" would wait forever, the enter-background callback would not return, and the system would kill the process (the process disappears within 22 seconds after switching to another App).
// These frames will not be seen anyway: discard them directly (accounted as dropped frames) and only wait for the one present already submitted to the GPU to finish.
static void present_drop_for_suspend(void) {
    pthread_mutex_lock(&g_pq_mu);
    if (g_link_owns_layer) {
        uint64_t n = g_pq_in - g_pq_read;
        for (uint64_t k = g_pq_read + 1; k <= g_pq_in; k++) g_pres[k % PRING] = (pres_ent){k, -1};
        g_link_dropped += n; g_pq_read = g_pq_in;
        while (g_pq_inflight) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        if (g_pq_done < g_pq_in) g_pq_done = g_pq_in;
        pthread_cond_broadcast(&g_pq_cv);
        if (n) MLOG("进后台：丢弃排队中未上屏的 %llu 帧（显示节拍在后台不回调）", (unsigned long long)n);
    }
    pthread_mutex_unlock(&g_pq_mu);
}
static void *present_main(void *arg) {
    (void)arg;
    pthread_setname_np("nvn-present");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    qos_class_t qc = QOS_CLASS_UNSPECIFIED; int rp = 0; pthread_get_qos_class_np(pthread_self(), &qc, &rp); atomic_store(&g_present_qos, (uint32_t)qc);
    for (;;) {
        pthread_mutex_lock(&g_pq_mu);
        while (g_pq_read == g_pq_in || g_link_owns_layer) pthread_cond_wait(&g_pq_cv, &g_pq_mu);   // 节拍模式下由节拍回调取；空队列看「已取出」；「已完成」要等上屏提交的 feedback，落后于取出 | In display-link mode the display-link callback takes them; an empty queue is judged by "taken"; "done" has to wait for the feedback of the present submit and lags behind taken
        pq_item it = g_pq[g_pq_read % PQ];
        uint64_t ticket = ++g_pq_read; g_classic_busy = 1;
        pthread_mutex_unlock(&g_pq_mu);
        int submitted;
        /* TOTK_MFX without the display link: the interpolated picture for at least one refresh (at most half a game frame),
           the game frame for at least half a game frame (30+30 at 100 Hz: 1 + 2 refreshes within the 3.3 a frame lasts). When
           a newer game frame is already queued (uncapped interpolation above half the refresh rate) the interpolated
           picture is skipped: two pictures per game frame would fall further and further behind. */
        static double tick; if (!tick) { int hz = mtl_display_hz(); tick = 1.0 / (hz > 0 ? hz : 60); }
        pthread_mutex_lock(&g_pq_mu); int behind = g_pq_read < g_pq_in; pthread_mutex_unlock(&g_pq_mu);
        int show_interp = it.interp >= 0 && !behind;
        if (show_interp) { @autoreleasepool { (void)present_now(&it.t, fmin(it.period / 2, tick), it.frame, ticket, it.crop, nil, 0, it.interp); } }
        @autoreleasepool { submitted = present_now(&it.t, show_interp ? it.period / 2 : it.period, it.frame, ticket, it.crop, nil, 0, -1); }
        pthread_mutex_lock(&g_pq_mu); g_classic_busy = 0; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
        if (!submitted) {
            pthread_mutex_lock(&g_pq_mu);
            if (ticket > g_pq_done) g_pq_done = ticket;
            pthread_cond_broadcast(&g_pq_cv);
            pthread_mutex_unlock(&g_pq_mu);
        }
    }
    return NULL;
}
uint32_t mtl_present_qos(void) { return atomic_load(&g_present_qos); }
static void present_start(void) { pthread_t th; if (pthread_create(&th, NULL, present_main, NULL)) MDIE("出图线程起不来"); pthread_detach(th); }
static _Atomic int g_present_discard;
void mtl_link_pause(int on);
// 前端进后台、等游戏存盘那一小段：出图直接丢（显示节拍后台不回调，入队会把游戏线程堵住；画面也没人看）。
// 打开时把已经排着队的帧一并清掉：系统是先停节拍回调、之后才通知进后台，这段空档里游戏已把交换链画满，正卡在「取下一张贴图前等它上一次出图处理完」
// （hle_nvn.c nvnWindowAcquireTexture → mtl_present_wait）。不清的话等存盘那 2.5 秒游戏一拍都走不了，盘没存成（玩家日志里三次同形：排队 3 帧、提交 0 次）
// The short period when the front end goes to the background and waits for the game to save: presents are dropped directly (the display link does not call back in the background, enqueueing would block the game thread; nobody sees the picture anyway).
// When turned on, also clear the frames already queued: the system first stops the display-link callback and only afterwards notifies about going to the background, and in that gap the game has already filled the swapchain and is stuck at "before acquiring the next texture, wait for its previous present to be processed"
// (hle_nvn.c nvnWindowAcquireTexture -> mtl_present_wait). Without clearing, the game cannot advance a single tick during those 2.5 seconds of waiting for the save, and the save is not written (the same pattern three times in player logs: 3 frames queued, 0 submits)
void mtl_present_discard(int on) { atomic_store(&g_present_discard, on != 0); if (on) present_drop_for_suspend(); }
/* TOTK_MFX: at the game's present (game thread; this frame's submission is committed, the next one not yet started):
   remember the presented texture for the canvas search, encode motion vectors and the interpolated frame between the
   previous and this frame in a command buffer of its own (GPU order: after this frame, before the next), reset the
   per-frame state. Returns the output slot or -1. */
static id<MTL4CommandAllocator> g_mfx_alloc[3]; static id<MTL4CommandBuffer> g_mfx_cb[3]; static _Atomic int g_mfx_busy[3];
static id<MTLTexture> g_mfx_swap_prev;   // TOTK_MFX_DUMP: copy of the previous presented image
static int mfx_present_frame_body(const tex_t *t, double period, int discard) {
    if (t) {
        int known = 0; for (int i = 0; i < g_mfx.nswap; i++) known |= g_mfx.swap[i] == t->serial;
        if (!known) g_mfx.swap[g_mfx.nswap < 4 ? g_mfx.nswap++ : (int)(t->serial % 4)] = t->serial;
        g_mfx.swap_w = t->d.width; g_mfx.swap_h = t->d.height;
    }
    int slot = mfx_frame_prepare(period);
    if (slot >= 0) {
        int a = -1; for (int i = 0; i < 3 && a < 0; i++) if (!atomic_load(&g_mfx_busy[i])) a = i;
        if (a < 0 || discard || nogpu()) { mfx_frame_abort(); slot = -1; }
        else {
#if DM_NVN_M3
            if (!g_mfx_alloc[a]) { g_mfx_alloc[a] = [M3CommandAllocator new]; g_mfx_cb[a] = [g_q commandBuffer]; }
            id<MTL4CommandBuffer> cb = g_mfx_cb[a];
            cb = [g_q commandBuffer];
            static id<MTL4CounterHeap> heap; static double hz;
            if (!heap) { MTL4CounterHeapDescriptor *hd = [MTL4CounterHeapDescriptor new]; hd.type = MTL4CounterHeapTypeTimestamp; hd.count = 6;
                         NSError *e = nil; heap = [[M3CounterHeap alloc] initWithDevice:g_dev count:hd.count]; hz = (double)1e9; }
            NSArray *frame_resources = nil;
            (void)mfx_frame_encode(cb, slot, heap, (NSUInteger)a * 2);
#else
            if (!g_mfx_alloc[a]) { g_mfx_alloc[a] = [g_dev newCommandAllocator]; g_mfx_cb[a] = [g_dev newCommandBuffer]; }
            [g_mfx_alloc[a] reset];
            id<MTL4CommandBuffer> cb = g_mfx_cb[a];
            [cb beginCommandBufferWithAllocator:g_mfx_alloc[a]];
            static id<MTL4CounterHeap> heap; static double hz;
            if (!heap) { MTL4CounterHeapDescriptor *hd = [MTL4CounterHeapDescriptor new]; hd.type = MTL4CounterHeapTypeTimestamp; hd.count = 6;
                         NSError *e = nil; heap = [g_dev newCounterHeapWithDescriptor:hd error:&e]; hz = (double)[g_dev queryTimestampFrequency]; }
            NSArray *frame_resources = mfx_frame_encode(cb, slot, heap, (NSUInteger)a * 2);
            [cb endCommandBuffer];
#endif
            atomic_store(&g_mfx_busy[a], 1);
            TotkCommitOptions *o = [TotkCommitOptions new];
            id<MTL4CounterHeap> h = heap; double fq = hz;
            [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
                // Capturing the payload keeps this frame's inputs/output and
                // interpolator alive even across a canvas/configuration change.
                if (frame_resources && frame_resources.count != 8) MDIE("invalid interpolation resource lifetime payload");
                sysload_note_gpu(f.GPUStartTime, f.GPUEndTime);   // the load line's game GPU share includes the interpolation
                if (f.error) MLOG("【红】[mfx] 插值命令缓冲错误：%s", f.error.description.UTF8String);
                else if (h && fq > 0) { NSData *d = [h resolveCounterRange:NSMakeRange((NSUInteger)a * 2, 2)]; const MTL4TimestampHeapEntry *t = d.bytes;
                    if (t && d.length >= 2 * sizeof *t && t[1].timestamp > t[0].timestamp) mfx_note_gpu((t[1].timestamp - t[0].timestamp) / fq * 1e3); }
                atomic_store(&g_mfx_busy[a], 0); }];
            q4_commit(cb, o);
            /* TOTK_MFX_SHOT=<game frame>[,<frame>...]:<path with %d>: save the interpolated frame shown before that game frame
               (compare with DM_NVN_SHOT_FRAME=1 DM_NVN_SHOT shots of the frames around it) */
            static long shots[16]; static int nshot = -1; static const char *shot_path;
            if (nshot < 0) { nshot = 0; const char *e = getenv("TOTK_MFX_SHOT"), *c = e ? strchr(e, ':') : NULL;
                             if (c) { shot_path = c + 1; for (const char *q = e; q < c && nshot < 16; q = strchr(q, ',') && strchr(q, ',') < c ? strchr(q, ',') + 1 : c) shots[nshot++] = atol(q); } }
            mfx_mv_log();
            long f = (long)atomic_load(&g_cur_frame);
            if (mfx_dump_due()) {   // TOTK_MFX_DUMP (diagnostic): previous and this presented image as well
                mtl_finish_gpu(); while (atomic_load(&g_mfx_busy[a])) usleep(200);
                id<MTLTexture> sc = t ? tex_mtl(t) : nil; mfx_dump(f, g_mfx_swap_prev, sc);
            }
            for (int i = 0; i < nshot; i++) if (shots[i] == f) { char path[1024]; snprintf(path, sizeof path, shot_path, (int)f); mtl_finish_gpu(); shot(mfx_output(slot), path, 1, NULL); }
        }
    }
    if (getenv("TOTK_MFX_DUMP") && t) {   // keep a copy of this presented image for the next frame's dump (diagnostic)
        id<MTLTexture> sc = tex_mtl(t);
        if (sc && (!g_mfx_swap_prev || g_mfx_swap_prev.width != sc.width || g_mfx_swap_prev.height != sc.height)) {
            MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:sc.pixelFormat width:sc.width height:sc.height mipmapped:NO];
            d.storageMode = MTLStorageModeShared; g_mfx_swap_prev = [g_dev newTextureWithDescriptor:d]; }
        if (sc && g_mfx_swap_prev) { mtl_finish_gpu(); id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder]; [b copyFromTexture:sc toTexture:g_mfx_swap_prev]; [b endEncoding]; cb_commit(cb); cb_wait(cb); }
    }
    g_mfx.scene_done = g_mfx.ui_cleared = g_mfx.depth_done = g_mfx.want_cam = 0;
    if (++g_mfx.frames % 300 == 0) { char b[256]; mfx_stats(b, sizeof b); MLOG("[mfx] 至帧 %llu：%s", (unsigned long long)atomic_load(&g_cur_frame), b); }
    return slot;
}
static int mfx_present_frame(const tex_t *t, double period, int discard) {
    @autoreleasepool { return mfx_present_frame_body(t, period, discard); }
}
uint64_t mtl_present(const tex_t *t, double period, const int crop[4]) {
    int discard = atomic_load(&g_present_discard), islot = mfx_enabled() && t ? mfx_present_frame(t, period, discard) : -1;
    if (mfx_latch() && !mfx_enabled()) g_mfx.scene_done = g_mfx.ui_cleared = g_mfx.depth_done = g_mfx.want_cam = 0;   // key 2 / flag mfx_interp: the next frame is fully on or off
    if (discard) return 0;   // 票号 0 = 没有待完成的上屏，取交换链贴图处不等 | Ticket number 0 = no pending present; the swapchain texture acquire does not wait
    if (t) { xv_check(t->serial, 0, 0, "出图"); if (g_xv_npresent < 128) g_xv_present[g_xv_npresent++] = t->serial; }
    static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, present_start);
    link_sync();
    {   // 测试开关 DM_TEST_LINK_STALL=起始秒,时长秒：从第一次出图起算到点暂停显示节拍（模仿前台时节拍不回调），时长到了放开。验兜底用，产品不设 | Test switch DM_TEST_LINK_STALL=start seconds,duration seconds: counting from the first present, pause the display link at that point (imitating the display link not calling back while in the foreground) and release it when the duration is up. For verifying the fallback; not set in the product
        static double t0, at = -1, dur; static int st; double now = CACurrentMediaTime();
        if (at < 0) { at = 0; const char *e = getenv("DM_TEST_LINK_STALL"); if (e && sscanf(e, "%lf,%lf", &at, &dur) == 2 && at > 0 && dur > 0) { t0 = now; st = 1; } else at = 0; }
        if (st == 1 && link_on() && now - t0 >= at) { st = 2; mtl_link_pause(1); MLOG("DM_TEST_LINK_STALL：暂停显示节拍 %.1f 秒（帧 %llu）", dur, (unsigned long long)atomic_load(&g_cur_frame)); }
        else if (st == 2 && now - t0 >= at + dur) { st = 3; mtl_link_pause(0); MLOG("DM_TEST_LINK_STALL：放开显示节拍（帧 %llu）", (unsigned long long)atomic_load(&g_cur_frame)); }
    }
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_in - g_pq_done >= PQ) pthread_cond_wait(&g_pq_cv, &g_pq_mu);   // 队满才等；正常由 AcquireTexture 先挡住 | Wait only when the queue is full; normally AcquireTexture blocks first
    pq_item *it = &g_pq[g_pq_in % PQ]; *it = (pq_item){*t, period, atomic_load(&g_cur_frame), {0}, CACurrentMediaTime(), islot, 0, 0};
    if (crop) memcpy(it->crop, crop, sizeof it->crop);   // 当帧状态随票号入队，不在出图线程晚读可变window | The current frame's state is enqueued with the ticket number, rather than reading the mutable window late on the present thread
    uint64_t tk = ++g_pq_in;
    pthread_cond_broadcast(&g_pq_cv);
    pthread_mutex_unlock(&g_pq_mu);
    return tk;
}
// 丢弃期间不等：标志置上之后才入队的那一帧（出图线程正好走过了标志检查）没人取，等它就又卡住；它留给随后的 mtl_drain 清
// 节拍模式下不无限等：见 STALL_NS 处的说明。只丢「还没被节拍取走」的帧，已在上屏途中的那张等它交给 GPU（同进后台的丢法）
// Do not wait during discard: the frame enqueued only after the flag was set (the present thread had just passed the flag check) has nobody to take it, and waiting for it would stall again; it is left for the following mtl_drain to clear
// Do not wait indefinitely in display-link mode: see the note at STALL_NS. Only frames "not yet taken by the display link" are dropped; the one already on its way to the screen is waited for until handed to the GPU (same dropping as when entering the background)
void mtl_present_wait(uint64_t tk) {
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_done < tk && !atomic_load(&g_present_discard)) {
        if (!g_link_owns_layer) { pthread_cond_wait(&g_pq_cv, &g_pq_mu); continue; }
        if (g_pq_read >= tk) { pthread_cond_wait(&g_pq_cv, &g_pq_mu); continue; }   // 这张已被节拍取走、正在上屏：等它交给 GPU（GPU 完成回调会叫醒） | This one has already been taken by the display link and is being presented: wait for it to be handed to the GPU (the GPU completion callback will wake us)
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_nsec += (long)STALL_NS; if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        // 已认定在停顿里（节拍回来时回调先清 g_stall_t0、再记节拍时刻）：不再每帧白等 100 ms，直接丢，游戏按自己的节拍等待超时（约两拍）走
        // Already determined to be in a stall (when the display link comes back the callback first clears g_stall_t0, then records the display-link tick time): no longer wait 100 ms in vain every frame, drop directly, and the game proceeds by its own display-link wait timeout (about two ticks)
        if (!g_stall_t0 && (pthread_cond_timedwait(&g_pq_cv, &g_pq_mu, &ts) != ETIMEDOUT || g_pq_done >= tk)) continue;
        uint64_t now = now_ns(), last = __atomic_load_n(&g_link_tick_ns, __ATOMIC_RELAXED);
        if (g_pq_read >= tk) continue;
        if (now - last < STALL_NS) { pthread_cond_timedwait(&g_pq_cv, &g_pq_mu, &ts); continue; }   // 节拍还在走，只是这张还没轮到：接着等（有上限，不空转） | The display link is still running, it is just not this one's turn yet: keep waiting (bounded, no busy spinning)
        uint64_t n = tk - g_pq_read;
        for (uint64_t k = g_pq_read + 1; k <= tk; k++) g_pres[k % PRING] = (pres_ent){k, -1};
        g_link_dropped += n; g_pq_read = tk;
        while (g_pq_inflight) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        if (g_pq_done < tk) g_pq_done = tk;
        pthread_cond_broadcast(&g_pq_cv);
        if (!g_stall_t0) { g_stall_t0 = last ? last : now; g_stall_drop = 0; g_stall_red = 0; g_stall_logged = 0; g_stall_n++; }
        g_stall_drop += n;
        int active = dm_focus_now() == 1; uint64_t since = dm_focus_in_since(), from = g_stall_t0 > since ? g_stall_t0 : since;
        if (active && !g_stall_logged && now - from >= STALL_NS) { g_stall_logged = 1;
            MLOG("显示节拍 %.0f ms 没回调（激活中，%s）：丢掉排队未上屏的 %llu 帧，游戏继续（第 %llu 段；节拍回来自动照常）", (now - from) / 1e6, window_seen(), (unsigned long long)n, (unsigned long long)g_stall_n); }
        if (active && !g_stall_red && now - from >= STALL_RED_NS) { g_stall_red = 1; MLOG("【红】显示节拍激活中连续 %.1f 秒没回调，出图全靠兜底丢帧，画面不动（%s）", (now - from) / 1e9, window_seen()); }
    }
    pthread_mutex_unlock(&g_pq_mu);
}
// 等已入队的出图全部处理完（mtl_finish = nvnQueueFinish 等用）：走 mtl_present_wait，节拍停了一样有兜底（真机：停节拍期间游戏调了一次 QueueFinish，
// 原先这里不带超时地等，排队的 3 帧没人取，游戏停住；Mac 上那两局没撞到这个调用）
// Wait until all queued presents have been processed (used by mtl_finish = nvnQueueFinish etc.): goes through mtl_present_wait, so there is still a fallback when the display link stops (real device: the game called QueueFinish once while the display link was stopped,
// and this used to wait here without a timeout; nobody took the 3 queued frames and the game stalled; those two runs on Mac did not hit this call)
void mtl_present_flush(void) { pthread_mutex_lock(&g_pq_mu); uint64_t h = g_pq_in; pthread_mutex_unlock(&g_pq_mu); mtl_present_wait(h); }
void mtl_stats(uint64_t *passes, uint64_t *draws) { *passes = g_stat_pass; *draws = g_stat_draw; }
void mtl_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = g_stat_args_enc; *reuse = g_stat_args_reuse;
    static uint64_t logged; uint64_t frame = atomic_load(&g_cur_frame);
    if (frame && !(frame % 300) && frame != logged) { logged = frame;
        MLOG("[drv-cache] hit=%llu miss=%llu verified=%llu", (unsigned long long)g_driver_hit, (unsigned long long)g_driver_miss, (unsigned long long)g_driver_check);
        MLOG("[texture-plan] enabled=%d total=%llu selected=%llu verified=%llu", texture_plan_on(), (unsigned long long)g_texture_plan_total, (unsigned long long)g_texture_plan_selected, (unsigned long long)g_texture_plan_verified);
    }
}
void mtl_use_stats(uint64_t *call, uint64_t *skip) { *call = 0; *skip = 0; }
uint64_t mtl_ds_omit(void) { return g_stat_ds_omit; }
void mtl_bw(uint64_t *load, uint64_t *store, uint64_t *full) { *load = g_bw_load; *store = g_bw_store; *full = g_pass_full_rw; }
uint64_t mtl_udc_saved_bytes(void) { return g_udc_saved_bytes; }
int mtl_has_window(void) { return front_metal_layer() != NULL; }
int mtl_link_active(void) { return link_on(); }
void mtl_link_pause(int on) { CAMetalDisplayLink *l = g_link; if (l) l.paused = on != 0; }   // 测试用（DM_TEST_BG）：模仿系统在后台停掉节拍回调 | For testing (DM_TEST_BG): imitates the system stopping the display-link callback in the background
int mtl_link_wait_tick(uint64_t timeout_ns, unsigned frame_ticks) {
    if (!mtl_link_active()) return 0;
    // 只在跑得比显示快时才等节拍：上一帧开工之后若已经过了节拍（这一帧已落后），立刻开工，不再白等到下一个节拍——
    // 否则帧时间略大于一个节拍时会「错过节拍 → 再等一整拍」，帧率直接锁到一半（Mac 有窗口负载下实测成串掉帧）
    // Wait for the display link only when running faster than the display: if a display-link tick has already passed since the previous frame started work (this frame is already behind), start immediately instead of waiting in vain for the next display-link tick --
    // otherwise when the frame time is slightly more than one display-link period it would "miss the tick -> wait another whole tick", locking the frame rate straight to half (measured on Mac under windowed load: runs of dropped frames)
    static uint64_t started_tick;   // 上一帧开工时的节拍号（只有游戏出图线程调本函数） | Display-link tick number when the previous frame started work (only the game's present thread calls this function)
    /* TOTK: a display link that has stopped calling back (window covered, minimised or on another Space: macOS stops the
       callbacks; test windows hit it) is not waited for: the caller paces by its own clock instead. Waiting out the timeout
       every frame turned MetalFX frame interpolation's 50 fps into 18 (2026-10-16 night, mfiA1), and the presents are
       dropped by mtl_present_wait's fallback anyway. The wait resumes with the first callback. */
    if (g_link_tick_ns && now_ns() - __atomic_load_n(&g_link_tick_ns, __ATOMIC_RELAXED) > STALL_NS) return 0;
    /* TOTK: a frame occupies frame_ticks 60 Hz ticks (the game's present interval); the variable-frame-rate step
       advances game time by exactly that interval, so presenting faster would speed the game up. */
    const uint64_t step = (uint64_t)(g_dm_present_link == 2 ? 2 : 1) * (frame_ticks ? frame_ticks : 1);   // 一帧占几拍：游戏出图间隔 × 每个 60 Hz 节拍对应的节拍数（120 Hz 节拍为 2） | Ticks per frame: the game's present interval times display-link ticks per 60 Hz tick (2 at 120 Hz)
    pthread_mutex_lock(&g_link_mu);
    if (g_link_tick_n >= started_tick + step) { started_tick = g_link_tick_n; pthread_mutex_unlock(&g_link_mu); return 1; }
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ns / 1000000000ull; ts.tv_nsec += timeout_ns % 1000000000ull;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    while (g_link_tick_n < started_tick + step) if (pthread_cond_timedwait(&g_link_cv, &g_link_mu, &ts) == ETIMEDOUT) { pthread_mutex_unlock(&g_link_mu); return 0; }
    started_tick = g_link_tick_n;
    pthread_mutex_unlock(&g_link_mu); return 1;
}

static id<CAMetalDrawable> g_prefetched_drawable; // 合成直画 drawable 用 | Used for compositing straight onto the drawable
id<CAMetalDrawable> mtl_present_take_prefetched(void) { return g_prefetched_drawable; }

#ifdef NVN_RT_RANGE_TEST
#include "rt_range_check.inc"
#endif

// 看门狗（无进展）时调用：把图形侧所有阻塞等待点的状态写进日志——谁在等谁一眼可见（换 Metal 4 提交模型后出现过一次「跑到 900 帧后停住、无红行」）
// Called by the watchdog (no progress): writes the state of all blocking wait points on the graphics side to the log -- who is waiting for whom is visible at a glance (after switching to the Metal 4 submit model there was once a case of "ran to 900 frames then stopped, no failure line")
void mtl_wait_state_dump(void) {
    MLOG("等待点状态：主栅栏 已提交 %llu / 已完成 %llu；主分配器栅栏 [%llu %llu %llu]；上传 未完成 %d、分配器占用 [%d %d %d]、序号 %u、有未冲刷批次 %d；出图队列 入 %llu / 完成 %llu；主命令缓冲在编 %d、提交线程已设 %d",
         (unsigned long long)g_fence, (unsigned long long)g_done, (unsigned long long)g_alloc_fence[0], (unsigned long long)g_alloc_fence[1], (unsigned long long)g_alloc_fence[2],
         atomic_load(&g_up_pending), atomic_load(&g_up_alloc_busy[0]), atomic_load(&g_up_alloc_busy[1]), atomic_load(&g_up_alloc_busy[2]), g_up_seq, g_up_cb != nil,
         (unsigned long long)g_pq_in, (unsigned long long)g_pq_done, g_cb != nil, atomic_load(&g_sub_thread_set));
}
void mtl_alloc_stats(uint64_t *waits, uint64_t *grown, int *n) { *waits = g_alloc_waits; *grown = g_alloc_grown; *n = g_alloc_n; }   // 主提交分配器池：等过几次、扩过几次、现有几个 | Main submit allocator pool: how many times it waited, how many times it grew, how many exist now
#pragma clang attribute pop

// ---------- TOTK standalone probes for the isolated GPU paths (runtime/gpu_path_probe.m) ----------
static id<MTLBuffer> probe_buf(NSUInteger n) { id<MTLBuffer> b = [g_dev newBufferWithLength:n options:MTLResourceStorageModeShared]; if (!b) MDIE("probe buffer"); res_add(b); return b; }
/* widen_u8_gpu over sources at several byte offsets; returns the number of checked values, dies on a mismatch. */
int mtl_probe_widen(void) {
    mtl_init();
    int checked = 0;
    const uint32_t counts[] = {1, 63, 64, 65, 1000, 4099}; const uint32_t shifts[] = {0, 1, 5, 15};
    id<MTLBuffer> src = probe_buf(8192); uint8_t *sp = src.contents;
    for (int i = 0; i < 8192; i++) sp[i] = i % 7 == 3 ? 0xff : (uint8_t)(i * 13 + 1);
    for (unsigned c = 0; c < sizeof counts / sizeof *counts; c++) for (unsigned k = 0; k < sizeof shifts / sizeof *shifts; k++) {
        uint32_t n = counts[c], sh = shifts[k] + 32 * c;
        mtl_sub_begin();
        id<MTLBuffer> out; size_t off; widen_u8_gpu(src.gpuAddress + sh, n, &out, &off);
        mtl_sub_commit(); mtl_finish();
        const uint16_t *o = (const uint16_t *)((const uint8_t *)out.contents + off);
        for (uint32_t i = 0; i < n; i++) {
            uint8_t v = sp[sh + i]; uint16_t want = v == 0xff && !getenv("TOTK_PROBE_NEGATIVE") ? 0xffff : v;   /* negative control: no restart mapping */
            if (o[i] != want) MDIE("widen probe: count %u shift %u index %u got %#x want %#x", n, sh, i, o[i], want);
            checked++;
        }
    }
    defer_obj_release(src);
    return checked;
}
/* blk_copy both ways (block texture <-> compressed texture) and rendering into the block texture, for 2D and 3D
   BC4 (8-byte blocks) and 2D BC3 (16-byte blocks). Returns the number of compared bytes; dies on a mismatch. */
int mtl_probe_blockrt(void) {
    mtl_init();
    int checked = 0;
    struct { MTLTextureType type; MTLPixelFormat bc, view; NSUInteger w, h, d, bpb, level, plane; } cases[] = {
        {MTLTextureType2D, MTLPixelFormatBC4_RUnorm, MTLPixelFormatRGBA16Uint, 32, 32, 1, 8, 0, 0},
        {MTLTextureType2D, MTLPixelFormatBC4_RUnorm, MTLPixelFormatRGBA16Uint, 64, 32, 1, 8, 1, 0},
        {MTLTextureType3D, MTLPixelFormatBC4_RUnorm, MTLPixelFormatRGBA16Uint, 32, 32, 4, 8, 0, 2},
        {MTLTextureType2D, MTLPixelFormatBC3_RGBA, MTLPixelFormatRGBA32Uint, 32, 16, 1, 16, 0, 0},
        {MTLTextureType2D, MTLPixelFormatBC4_RUnorm, MTLPixelFormatRGBA16Uint, 256, 256, 1, 8, 7, 0},
        {MTLTextureType2D, MTLPixelFormatBC4_RUnorm, MTLPixelFormatRGBA16Uint, 256, 256, 1, 8, 8, 0},
    };
    for (unsigned c = 0; c < sizeof cases / sizeof *cases; c++) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor new];
        d.textureType = cases[c].type; d.pixelFormat = cases[c].bc; d.width = cases[c].w; d.height = cases[c].h; d.depth = cases[c].d;
        d.mipmapLevelCount = cases[c].level + 1; d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> bc = [g_dev newTextureWithDescriptor:d]; if (!bc) MDIE("probe compressed texture case %u", c); res_add(bc);
        tex_t t={0};t.serial=0xfffff900+c;t.d.format=cases[c].bpb==8?NVN_FORMAT_RGTC1_UNORM:NVN_FORMAT_RGBA_DXT5;
        t.d.target=cases[c].type==MTLTextureType3D?2:1;t.d.width=cases[c].w;t.d.height=cases[c].h;t.d.depth=cases[c].d;t.d.levels=cases[c].level+1;
        serial_put(t.serial,bc);
        view_t view={0};view.set_format=view.set_levels=view.set_layers=1;view.format=cases[c].bpb==8?NVN_FORMAT_RGBA16UI:NVN_FORMAT_RGBA32UI;view.base_level=cases[c].level;view.min_layer=cases[c].plane;
        setenv("TOTK_GPU_BLOCKRT","1",1);
        blk_wb w;id<MTLTexture> aux=blk_aux(&t,&view,1,&w);
        d.pixelFormat=cases[c].view;d.width=w.bx;d.height=w.by;d.depth=MAX(cases[c].d>>cases[c].level,1u);d.mipmapLevelCount=1;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
        id<MTLTexture> aux2=[g_dev newTextureWithDescriptor:d];if(!aux||!aux2)MDIE("probe auxiliary texture");res_add(aux2);
        NSUInteger lv=cases[c].level,lw=w.lw,lh=w.lh,al=0;
        NSUInteger row = w.bx * w.bpb, size = row * w.by;
        id<MTLBuffer> ref = probe_buf(size), rb1 = probe_buf(size), rb2 = probe_buf(size), rb3 = probe_buf(size);
        uint8_t *r = ref.contents; for (NSUInteger i = 0; i < size; i++) r[i] = (uint8_t)(i * 31 + c * 7 + 5);
        // 1. reference blocks -> block texture -> compressed texture -> buffer, and back into a second block texture
        mtl_sub_begin();
        enc_b_copy_buffer_texture(blit(), ref, 0, row, size, MTLSizeMake(w.bx, w.by, 1), aux, 0, al, MTLOriginMake(0, 0, w.plane));
        ((void)0);
        if (!getenv("TOTK_PROBE_NEGATIVE")) blk_copy(aux, 1, bc, &w);   /* negative control: the copy is left out */
        ((void)0);
        enc_b_copy_texture_buffer(blit(), bc, 0, lv, MTLOriginMake(0, 0, w.plane), MTLSizeMake(lw, lh, 1), rb1, 0, row, size);
        blk_wb w2 = w; w2.aux = aux2;
        blk_copy(bc, 0, aux2, &w2);
        ((void)0);
        enc_b_copy_texture_buffer(blit(), aux2, 0, al, MTLOriginMake(0, 0, w.plane), MTLSizeMake(w.bx, w.by, 1), rb2, 0, row, size);
        // 2. a render pass clears the block texture (as a render target of the view format), then it goes to the compressed texture
        enc_b_end(g_blit); g_blit = nil;
        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
        rp.colorAttachments[0].texture = aux; rp.colorAttachments[0].level = al;
        if (cases[c].type == MTLTextureType3D) rp.colorAttachments[0].depthPlane = w.plane;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0x1234, 0x5678, 0x9abc, 0xdef0);
        rp.renderTargetWidth = MAX(aux.width >> al, 1u); rp.renderTargetHeight = MAX(aux.height >> al, 1u);
        id<MTL4RenderCommandEncoder> re = enc_r_begin(rp); enc_r_end(re);
        blk_copy(aux, 1, bc, &w);
        ((void)0);
        enc_b_copy_texture_buffer(blit(), bc, 0, lv, MTLOriginMake(0, 0, w.plane), MTLSizeMake(lw, lh, 1), rb3, 0, row, size);
        mtl_sub_commit(); mtl_finish();
        if (memcmp(rb1.contents, r, size)) MDIE("block probe case %u: compressed texture differs from the rendered blocks", c);
        if (memcmp(rb2.contents, r, size)) MDIE("block probe case %u: copy back to a block texture differs", c);
        const uint8_t *c3 = rb3.contents; uint8_t want[16];
        if (cases[c].bpb == 8) { uint16_t v[4] = {0x1234, 0x5678, 0x9abc, 0xdef0}; memcpy(want, v, 8); }
        else { uint32_t v[4] = {0x1234, 0x5678, 0x9abc, 0xdef0}; memcpy(want, v, 16); }
        for (NSUInteger i = 0; i < size; i += cases[c].bpb) if (memcmp(c3 + i, want, cases[c].bpb)) MDIE("block probe case %u: cleared block %lu differs", c, (unsigned long)(i / cases[c].bpb));
        checked += (int)(3 * size);
        for (id o in @[bc, aux, aux2, ref, rb1, rb2, rb3]) defer_obj_release(o);
    }
    return checked;
}
/* Exercise the production descriptor with the same fragment shader, with and without a depth attachment. */
int mtl_probe_depth_output(void) {
    mtl_init();
    NSString *vsrc = @"#include <metal_stdlib>\nusing namespace metal;\nstruct V { float4 p [[position]]; };\nvertex V vs(uint id [[vertex_id]]) { float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)}; return {float4(p[id],0.75,1)}; }";
    NSString *fsrc = @"#include <metal_stdlib>\nusing namespace metal;\nstruct main0_out { float4 color [[color(0)]]; float gl_FragDepth [[depth(any)]]; };\nfragment main0_out main0() { main0_out out = {}; out.gl_FragDepth=0.125; out.color=float4(out.gl_FragDepth*2,0.5,0.75,1); return out; }";
    NSError *err = nil; id<MTLLibrary> vl = [g_dev newLibraryWithSource:vsrc options:totk_precise_opts() error:&err];
    id<MTLLibrary> fl = [g_dev newLibraryWithSource:fsrc options:totk_precise_opts() error:&err];
    if (!vl || !fl) MDIE("depth output probe compile: %s", err.localizedDescription.UTF8String);
    NvShader *vs = [NvShader new], *fs = [NvShader new]; vs->fn = [vl newFunctionWithName:@"vs"]; vs->io = @{};
    fs->fn = [fl newFunctionWithName:@"main0"]; fs->src = fsrc; fs->io = @{}; fs->writes_depth = 1; strcpy(fs->key, "probe-depth-output");
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
    td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> color = [g_dev newTextureWithDescriptor:td]; res_add(color);
    td.pixelFormat = MTLPixelFormatDepth32Float; id<MTLTexture> depth = [g_dev newTextureWithDescriptor:td]; res_add(depth);
    id<MTLBuffer> colorRead = probe_buf(1024), depthRead = probe_buf(1024);
    for (int hasDepth = 0; hasDepth < 2; hasDepth++) {
        pso_key k = {0}; k.cfmt[0] = MTLPixelFormatRGBA8Unorm; k.samples = 1; k.chmask = 15;
        k.dfmt = hasDepth ? MTLPixelFormatDepth32Float : 0;
        MTLRenderPipelineDescriptor *pd = pso_desc(&k, vs, fs, nil, 0);
        if (!hasDepth && getenv("TOTK_PROBE_NEGATIVE")) pd.fragmentFunction = fs->fn;
        id<MTLRenderPipelineState> ps = [g_dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!ps) MDIE("depth output probe pipeline: %s", err.localizedDescription.UTF8String);
        if (hasDepth && pd.fragmentFunction != fs->fn) MDIE("depth output probe lost attached depth export");
        mtl_sub_begin(); MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
        rp.colorAttachments[0].texture = color; rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        if (hasDepth) { rp.depthAttachment.texture = depth; rp.depthAttachment.loadAction = MTLLoadActionClear; rp.depthAttachment.storeAction = MTLStoreActionStore; rp.depthAttachment.clearDepth = 1; }
        rp.renderTargetWidth = rp.renderTargetHeight = 16;
        id<MTL4RenderCommandEncoder> en = enc_r_begin(rp); enc_r_pso(en, ps);
        MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new]; dd.depthCompareFunction = MTLCompareFunctionAlways; dd.depthWriteEnabled = hasDepth;
        enc_r_dss(en, [g_dev newDepthStencilStateWithDescriptor:dd]); enc_r_viewport(en, (MTLViewport){0,0,16,16,0,1}); enc_r_scissor(en, (MTLScissorRect){0,0,16,16});
        enc_r_draw(en, MTLPrimitiveTypeTriangle, 0, 3); enc_r_end(en);
        enc_b_copy_texture_buffer(blit(), color, 0, 0, MTLOriginMake(0,0,0), MTLSizeMake(16,16,1), colorRead, 0, 64, 1024);
        if (hasDepth) enc_b_copy_texture_buffer(blit(), depth, 0, 0, MTLOriginMake(0,0,0), MTLSizeMake(16,16,1), depthRead, 0, 64, 1024);
        mtl_sub_commit(); mtl_finish();
        for (int i = 0; i < 256; i++) {
            const uint8_t *p = (const uint8_t *)colorRead.contents + 4*i;
            if (abs((int)p[0]-64)>1 || abs((int)p[1]-128)>1 || abs((int)p[2]-191)>1 || p[3]!=255) MDIE("depth output probe color mismatch pixel %d", i);
            if (hasDepth && ((const float *)depthRead.contents)[i] != 0.125f) MDIE("depth output probe depth mismatch pixel %d", i);
        }
    }
    for (id r in @[color, depth, colorRead, depthRead]) defer_obj_release(r);
    return 768;
}

/* TOTK probe: front-face winding through the production Metal 4 render path. A TL-BL-TR triangle (counter-clockwise
   in y-up NDC) is drawn with cull back under both windings; standalone Metal 3 measurements: CCW front draws it, CW culls it. */
int mtl_probe_winding(void) {
    mtl_init();
    const NSUInteger W = 64, H = 64;
    static const char *msl = "#include <metal_stdlib>\nusing namespace metal;\nstruct V { float4 p [[position]]; };\n"
        "vertex V vs(uint id [[vertex_id]], constant float &z [[buffer(0)]]) { float2 q[3] = {float2(-1, 1), float2(-1, -1), float2(1, 1)}; V o; o.p = float4(q[id], z, 1); return o; }\n"
        "fragment float4 fs() { return float4(1, 1, 1, 1); }\n";
    NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(msl) options:nil error:&e]; if (!lib) MDIE("winding probe library: %s", e.localizedDescription.UTF8String);
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new]; pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm; id<MTLRenderPipelineState> ps = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e]; if (!ps) MDIE("winding probe pso: %s", e.localizedDescription.UTF8String);
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget; d.storageMode = MTLStorageModePrivate;
    int checked = 0; long lit[6] = {0, 0, 0, 0, 0, 0}; MTLWinding wd[6] = {MTLWindingCounterClockwise, MTLWindingClockwise, MTLWindingClockwise, MTLWindingCounterClockwise, MTLWindingClockwise, MTLWindingClockwise};
    MTLCullMode cm[6] = {MTLCullModeBack, MTLCullModeBack, MTLCullModeNone, MTLCullModeBack, MTLCullModeBack, MTLCullModeNone}; float zs[6] = {0, 0, 0, 1, 1, 1};
    MTL4ArgumentTableDescriptor *atd = [MTL4ArgumentTableDescriptor new]; atd.maxBufferBindCount = 1; id<MTL4ArgumentTable> at = [M3ArgTable new]; if (!at) MDIE("winding probe argument table");
    for (int k = 0; k < 6; k++) {
        id<MTLBuffer> zb = probe_buf(16); memcpy(zb.contents, &zs[k], 4);
        id<MTLTexture> t = [g_dev newTextureWithDescriptor:d]; if (!t) MDIE("winding probe texture"); res_add(t);
        id<MTLBuffer> rb = probe_buf(W * H * 4);
        mtl_sub_begin();
        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
        rp.colorAttachments[0].texture = t; rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1); rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.renderTargetWidth = W; rp.renderTargetHeight = H;
        id<MTL4RenderCommandEncoder> re = enc_r_begin(rp);
        enc_r_pso(re, ps); enc_r_viewport(re, (MTLViewport){0, 0, W, H, 0, 1}); enc_r_scissor(re, (MTLScissorRect){0, 0, W, H});
        m3_attach(at, re, M3_STAGE_VERTEX); [at setAddress:zb.gpuAddress atIndex:0];
        enc_r_cull(re, cm[k]); enc_r_winding(re, wd[k]);
        enc_r_draw(re, MTLPrimitiveTypeTriangle, 0, 3);
        enc_r_end(re);
        enc_b_copy_texture_buffer(blit(), t, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), rb, 0, W * 4, W * H * 4);
        enc_b_end(g_blit); g_blit = nil;
        mtl_sub_commit(); mtl_finish();
        const uint8_t *px = rb.contents; for (NSUInteger i = 0; i < W * H; i++) lit[k] += px[i * 4] > 0;
        checked++;
    }
    MLOG("winding probe (Metal 4 path): z=0: CCW+back %ld, CW+back %ld, none %ld; z=1 (far plane): CCW+back %ld, CW+back %ld, none %ld (of %lu)", lit[0], lit[1], lit[2], lit[3], lit[4], lit[5], (unsigned long)(W * H));
    if (!(lit[0] > 0 && lit[1] == 0 && lit[2] > 0 && lit[3] > 0 && lit[4] == 0 && lit[5] > 0)) MDIE("winding probe: Metal 4 path does not match the Metal 3 measurement (CCW drawn, CW culled, far plane kept)");
    return checked;
}

/* TOTK probe: replay the tile-light quad draw (375 instances, 8-bit indices, VS with instance_id and no vertex inputs) through
   the production Metal 4 path with the real translated VS (TOTK_PROBE_TILE_VS), real constant buffer (TOTK_PROBE_TILE_CB) and
   real storage buffer (TOTK_PROBE_TILE_SB). In the game these quads are culled with CCW front + cull back; offline with Metal 3
   they are counter-clockwise and drawn. Reports lit pixels for: CCW+back, CW+back, none, CCW+back instances=1. */
int mtl_probe_tile(void) {
    mtl_init();
    const char *vsp = getenv("TOTK_PROBE_TILE_VS"), *cbp = getenv("TOTK_PROBE_TILE_CB"), *sbp = getenv("TOTK_PROBE_TILE_SB");
    if (!vsp || !cbp || !sbp) MDIE("tile probe needs TOTK_PROBE_TILE_VS/_CB/_SB");
    NSString *vsrc = [NSString stringWithContentsOfFile:@(vsp) encoding:NSUTF8StringEncoding error:nil]; if (!vsrc) MDIE("tile probe: cannot read %s", vsp);
    NSData *cbd = [NSData dataWithContentsOfFile:@(cbp)], *sbd = [NSData dataWithContentsOfFile:@(sbp)]; if (!cbd || !sbd) MDIE("tile probe: cannot read buffers");
    NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:[vsrc stringByAppendingString:@"\nfragment float4 fs_solid() { return float4(1, 1, 1, 1); }\n"] options:nil error:&e];
    if (!lib) MDIE("tile probe library: %s", e.localizedDescription.UTF8String);
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new]; pd.vertexFunction = [lib newFunctionWithName:@"main0"]; pd.fragmentFunction = [lib newFunctionWithName:@"fs_solid"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> ps = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e]; if (!ps) MDIE("tile probe pso: %s", e.localizedDescription.UTF8String);
    /* variant with the game's vertex descriptor (the VS has no stage_in; the game still attaches attribute 0 = Half4 at buffer 16) */
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor]; vd.attributes[0].format = MTLVertexFormatHalf4; vd.attributes[0].offset = 0; vd.attributes[0].bufferIndex = 16; vd.layouts[16].stride = 16;
    pd.vertexDescriptor = vd; id<MTLRenderPipelineState> ps2 = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e]; if (!ps2) MLOG("tile probe: pso with vertex descriptor failed: %s", e.localizedDescription.UTF8String);
    const NSUInteger W = 1600, H = 900;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:NO]; d.usage = MTLTextureUsageRenderTarget; d.storageMode = MTLStorageModePrivate;
    id<MTLBuffer> cb = probe_buf(cbd.length), sb = probe_buf(sbd.length); memcpy(cb.contents, cbd.bytes, cbd.length); memcpy(sb.contents, sbd.bytes, sbd.length);
    int checked = 0; long lit[5] = {0, 0, 0, 0, 0}; MTLWinding wd[5] = {MTLWindingCounterClockwise, MTLWindingClockwise, MTLWindingClockwise, MTLWindingCounterClockwise, MTLWindingCounterClockwise};
    MTLCullMode cm[5] = {MTLCullModeBack, MTLCullModeBack, MTLCullModeNone, MTLCullModeBack, MTLCullModeBack}; NSUInteger inst[5] = {375, 375, 375, 1, 375};
    for (int k = 0; k < 5; k++) {
        id<MTLTexture> t = [g_dev newTextureWithDescriptor:d]; if (!t) MDIE("tile probe texture"); res_add(t);
        id<MTLBuffer> rb = probe_buf(W * H * 4);
        mtl_sub_begin();
        id<MTLBuffer> ab; size_t ao = arena_alloc(16, &ab); uint64_t *slots = (uint64_t *)((uint8_t *)ab.contents + ao); slots[0] = cb.gpuAddress; slots[1] = sb.gpuAddress;
        id<MTLBuffer> ib; size_t io = arena_alloc(16, &ib); uint16_t *ix = (uint16_t *)((uint8_t *)ib.contents + io); uint16_t src[6] = {0, 2, 1, 1, 2, 3}; memcpy(ix, src, 12);
        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
        rp.colorAttachments[0].texture = t; rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1); rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.renderTargetWidth = W; rp.renderTargetHeight = H;
        id<MTL4RenderCommandEncoder> re = enc_r_begin(rp);
        enc_r_pso(re, k == 4 && ps2 ? ps2 : ps); enc_r_viewport(re, (MTLViewport){0, 0, W, H, 0, 1}); enc_r_scissor(re, (MTLScissorRect){0, 0, W, H});
        enc_r_args(re, MTLRenderStageVertex, ab.gpuAddress + ao, 0); if (k == 4) enc_r_vbuf(re, 16, cb.gpuAddress); m3_attach(g_vt, re, M3_STAGE_VERTEX);
        enc_r_cull(re, cm[k]); enc_r_winding(re, wd[k]);
        enc_r_draw_indexed(re, MTLPrimitiveTypeTriangle, 6, MTLIndexTypeUInt16, ib.gpuAddress + io, 12, inst[k], 0, 0);
        enc_r_end(re);
        enc_b_copy_texture_buffer(blit(), t, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), rb, 0, W * 4, W * H * 4);
        enc_b_end(g_blit); g_blit = nil;
        mtl_sub_commit(); mtl_finish();
        const uint8_t *px = rb.contents; for (NSUInteger i = 0; i < W * H; i++) lit[k] += px[i * 4] > 0;
        checked++;
    }
    MLOG("tile probe (Metal 4 path, real VS/CB/SB): CCW+back %ld, CW+back %ld, none %ld, CCW+back 1 instance %ld, CCW+back with vertex descriptor %ld (of %lu)", lit[0], lit[1], lit[2], lit[3], lit[4], (unsigned long)(W * H));
    return checked;
}
/* clear_quad_core on two color targets plus depth: masked, scissored clear of attachment 1 and a half-target clear of
   attachment 0 inside one render pass; everything else (other channels, other pixels, depth) must stay. */
int mtl_probe_midclear(void) {
    mtl_init();
    const NSUInteger W = 32, H = 32;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget; d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> c0 = [g_dev newTextureWithDescriptor:d]; d.pixelFormat = MTLPixelFormatRGBA16Float; id<MTLTexture> c1 = [g_dev newTextureWithDescriptor:d];
    d.pixelFormat = MTLPixelFormatDepth32Float; id<MTLTexture> dt = [g_dev newTextureWithDescriptor:d];
    if (!c0 || !c1 || !dt) MDIE("midclear probe textures"); res_add(c0); res_add(c1); res_add(dt);
    id<MTLBuffer> r0 = probe_buf(W * H * 4), r1 = probe_buf(W * H * 8), rd = probe_buf(W * H * 4);
    mtl_sub_begin();
    MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
    rp.colorAttachments[0].texture = c0; rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0); rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[1].texture = c1; rp.colorAttachments[1].loadAction = MTLLoadActionClear; rp.colorAttachments[1].clearColor = MTLClearColorMake(0.25, 0.25, 0.25, 0.25); rp.colorAttachments[1].storeAction = MTLStoreActionStore;
    rp.depthAttachment.texture = dt; rp.depthAttachment.loadAction = MTLLoadActionClear; rp.depthAttachment.clearDepth = 1.0; rp.depthAttachment.storeAction = MTLStoreActionStore;
    id<MTL4RenderCommandEncoder> en = enc_r_begin(rp);
    uint32_t cf[8] = {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA16Float};
    const float v1[4] = {1, 0.5f, 0.75f, 1}, v0[4] = {1, 1, 0, 1};
    clear_quad_core(en, cf, MTLPixelFormatDepth32Float, 0, 1, W, H, 1, 0x5, v1, (MTLScissorRect){4, 4, 8, 8});
    clear_quad_core(en, cf, MTLPixelFormatDepth32Float, 0, 1, W, H, 0, getenv("TOTK_PROBE_NEGATIVE") ? 0x7 : 0xf, v0, (MTLScissorRect){0, 0, W, H / 2});
    enc_r_end(en);
    id e = blit();
    enc_b_copy_texture_buffer(e, c0, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), r0, 0, W * 4, W * H * 4);
    enc_b_copy_texture_buffer(e, c1, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), r1, 0, W * 8, W * H * 8);
    enc_b_copy_texture_buffer(e, dt, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), rd, 0, W * 4, W * H * 4);
    mtl_sub_commit(); mtl_finish();
    int checked = 0; const uint8_t *p0 = r0.contents; const __fp16 *p1 = r1.contents; const float *pd = rd.contents;
    for (NSUInteger y = 0; y < H; y++) for (NSUInteger x = 0; x < W; x++) {
        const uint8_t *a = p0 + (y * W + x) * 4; int top = y < H / 2;
        uint8_t want0[4] = {top ? 255 : 0, top ? 255 : 0, 0, top ? 255 : 0};
        if (memcmp(a, want0, 4)) MDIE("midclear probe: attachment 0 pixel (%lu,%lu) = %u,%u,%u,%u", (unsigned long)x, (unsigned long)y, a[0], a[1], a[2], a[3]);
        const __fp16 *b = p1 + (y * W + x) * 4; int in = x >= 4 && x < 12 && y >= 4 && y < 12;
        float want1[4] = {in ? 1.0f : 0.25f, 0.25f, in ? 0.75f : 0.25f, 0.25f};
        for (int k = 0; k < 4; k++) if ((float)b[k] != want1[k]) MDIE("midclear probe: attachment 1 pixel (%lu,%lu) channel %d = %g want %g", (unsigned long)x, (unsigned long)y, k, (double)b[k], want1[k]);
        if (pd[y * W + x] != 1.0f) MDIE("midclear probe: depth changed at (%lu,%lu)", (unsigned long)x, (unsigned long)y);
        checked += 9;
    }
    for (id o in @[c0, c1, dt, r0, r1, rd]) defer_obj_release(o);
    return checked;
}

/* Known linear shader outputs through production attachment binding, read back as raw UNORM bytes.
   Cover ClearTexture and mip/layer selection too. Alpha must remain linear. */
int mtl_probe_srgb_view(void) {
    mtl_init(); const NSUInteger W = 8, H = 8;
    MTLTextureDescriptor *d = [MTLTextureDescriptor new]; d.textureType = MTLTextureType2DArray;
    d.pixelFormat = MTLPixelFormatRGBA8Unorm; d.width = W; d.height = H; d.arrayLength = 2; d.mipmapLevelCount = 2;
    d.usage = MTLTextureUsageRenderTarget; d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> base = [g_dev newTextureWithDescriptor:d]; if (!base) MDIE("sRGB probe texture"); res_add(base);
    tex_t t = {0}; t.serial = 0xfffffff0; t.d.format = 0x25; t.d.target = 4; serial_put(t.serial, base);
    const float rgba[4] = {0.0031308f, 0.18f, 0.5f, 0.25f}; const uint8_t encoded[4] = {10, 118, 188, 64}, linear[4] = {1, 46, 128, 64};
    int checked = 0;
    for (int mode = 0; mode < 3; mode++) {
        view_t v = {0}; v.set_format = 1; v.format = mode ? 0x38 : 0x25;
        v.set_levels = v.set_layers = 1; v.base_level = 1; v.min_layer = 1;
        id<MTLBuffer> rb = probe_buf(W * H * 4); mtl_sub_begin();
        if (mode == 2) {
            p_clear_tex c = {0}; c.tex = &t; c.view = v; c.has_view = 1; c.r.width = W / 2; c.r.height = H / 2; c.r.depth = 1; c.mask = 0xf;
            memcpy(c.c.f, rgba, sizeof rgba); clear_tex(&c);
        } else {
            MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
            attach(rp.colorAttachments[0], &t, &v, !getenv("TOTK_PROBE_NEGATIVE"));
            if (getenv("TOTK_PROBE_NEGATIVE")) { rp.colorAttachments[0].level = 1; rp.colorAttachments[0].slice = 1; }
            rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            uint32_t cf[8] = {(uint32_t)rp.colorAttachments[0].texture.pixelFormat};
            id en = enc_r_begin(rp); clear_quad_core(en, cf, 0, 0, 1, W / 2, H / 2, 0, 0xf, rgba, (MTLScissorRect){0, 0, W / 2, H / 2}); enc_r_end(en);
        }
        enc_b_copy_texture_buffer(blit(), base, 1, 1, MTLOriginMake(0, 0, 0), MTLSizeMake(W / 2, H / 2, 1), rb, 0, W * 2, W * H);
        mtl_sub_commit(); mtl_finish();
        const uint8_t *actual = rb.contents, *want = mode ? encoded : linear;
        for (NSUInteger i = 0; i < W * H; i++) { if (abs((int)actual[i] - want[i % 4]) > 1) MDIE("sRGB view probe mode %d byte %lu: %u != %u", mode, (unsigned long)i, actual[i], want[i % 4]); checked++; }
        defer_obj_release(rb);
    }
    serial_del(t.serial); defer_obj_release(base); return checked;
}

/* Replay actual depth/stencil clear records across active and restarted passes. Preserve colors, depth
   outside the scissor, stencil bits outside the mask, and depth during stencil-only clears. */
int mtl_probe_ds_clear(void) {
    mtl_init(); setenv("TOTK_GPU_MIDCLEAR", "1", 1); const NSUInteger W = 32, H = 32;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget; d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> color = [g_dev newTextureWithDescriptor:d]; d.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    id<MTLTexture> depth = [g_dev newTextureWithDescriptor:d]; if (!color || !depth) MDIE("depth/stencil clear probe textures"); res_add(color); res_add(depth);
    tex_t ct = {0}, dt = {0}; ct.serial = 0xffffffd0; dt.serial = 0xffffffd1;
    ct.d.width = dt.d.width = W; ct.d.height = dt.d.height = H; ct.d.format = 0x25; dt.d.format = 0x36;
    serial_put(ct.serial, color); serial_put(dt.serial, depth);
    id<MTLBuffer> rc = probe_buf(W * H * 4), rd = probe_buf(W * H * 4), rs = probe_buf(W * H);
    op_hdr end = {OP_END, 8}; mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
    R.rt.n = 1; R.rt.color[0] = &ct; R.rt.depth = &dt; R.has_rt = 1;
    R.clear_c[0] = R.clear_d = R.clear_s = 1; R.cc[0] = MTLClearColorMake(0.25, 0.5, 0.75, 1); R.cd = 0.8; R.cs = 0xa5; rp_begin();
    struct { op_hdr h; p_clear_ds c; op_hdr end; } op = {{OP_CLEAR_DS, 8 + sizeof(p_clear_ds)}, {0.25f, 1, 0x3c, 0x0f}, {OP_END, 8}};
    R.has_sc = 1; R.sc = (p_scissor){4, 8, 8, 16};
    if (getenv("TOTK_PROBE_NEGATIVE")) op.c.stencil_mask = 255;
    mtl_exec((const uint8_t *)&op); rp_end(); /* next partial clear starts its own pass */
    R.sc = (p_scissor){8, 8, 16, 16}; op.c = (p_clear_ds){0.9f, 0, 0x50, 0xf0}; mtl_exec((const uint8_t *)&op);
    R.sc = (p_scissor){0, 0, W, 8}; op.c = (p_clear_ds){0.5f, 1, 0xff, 0}; mtl_exec((const uint8_t *)&op);
    op.c = (p_clear_ds){1, 0, 0, 0}; mtl_exec((const uint8_t *)&op); rp_end();
    id<MTL4ComputeCommandEncoder> e = blit(); enc_b_copy_texture_buffer(e, color, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), rc, 0, W * 4, W * H * 4);
    [e copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1) toBuffer:rd destinationOffset:0 destinationBytesPerRow:W * 4 destinationBytesPerImage:W * H * 4 options:MTLBlitOptionDepthFromDepthStencil];
    [e copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1) toBuffer:rs destinationOffset:0 destinationBytesPerRow:W destinationBytesPerImage:W * H options:MTLBlitOptionStencilFromDepthStencil];
    mtl_sub_commit(); mtl_finish();
    const uint8_t *pc = rc.contents, *ps = rs.contents; const float *pd = rd.contents; int checked = 0;
    for (NSUInteger y = 0; y < H; y++) for (NSUInteger x = 0; x < W; x++) {
        NSUInteger i = y * W + x; int a = x >= 4 && x < 12 && y >= 8 && y < 24, b = x >= 8 && x < 24 && y >= 8 && y < 24;
        const uint8_t want[4] = {64, 128, 191, 255};
        for (int k = 0; k < 4; k++) if (abs(pc[i * 4 + k] - want[k]) > 1) MDIE("depth/stencil clear changed color at %lu,%lu", x, y);
        float z = y < 8 ? 0.5f : a ? 0.25f : 0.8f; uint8_t st = a ? 0xac : 0xa5; if (b) st = (st & 15) | 0x50;
        if (fabsf(pd[i] - z) > 0.000001f || ps[i] != st) MDIE("depth/stencil clear mismatch at %lu,%lu: depth %g != %g, stencil %#x != %#x", x, y, pd[i], z, ps[i], st);
        checked += 6;
    }
    memset(&R, 0, sizeof R); serial_del(ct.serial); serial_del(dt.serial);
    for (id o in @[color, depth, rc, rd, rs]) defer_obj_release(o); return checked;
}

/* Keep a game submission open while more than one staging batch is uploaded. Check both payload
   ordering and bounded batch memory, including retirement before the game fence is committed. */
int mtl_probe_upload_batch(void) {
    mtl_init(); const int N = 12; const size_t cap = 2u * UP_BLOCK_SIZE; int checked = 0;
    g_up_batch_limit = getenv("TOTK_PROBE_NEGATIVE") ? SIZE_MAX : cap;
    id<MTLBuffer> src = [g_dev newBufferWithLength:N * 256 options:MTLResourceStorageModeShared]; res_add(src);
    id<MTLBuffer> result = [g_dev newBufferWithLength:N * 256 options:MTLResourceStorageModeShared]; res_add(result);
    id<MTLTexture> dst[N];
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:1 height:1 mipmapped:NO];
    d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageShaderRead;
    mtl_sub_begin();
    for (int i = 0; i < N; i++) {
        uint8_t *p = (uint8_t *)src.contents + i * 256; for (int c = 0; c < 4; c++) p[c] = (uint8_t)(i * 17 + c);
        dst[i] = [g_dev newTextureWithDescriptor:d]; res_add(dst[i]);
        up_begin(); size_t off; id<MTLBuffer> b = up_alloc(UP_BLOCK_SIZE, &off);
        if (g_up_batch_bytes > cap) MDIE("upload batch probe: %zu staging bytes exceed %zu", g_up_batch_bytes, cap);
        enc_b_copy_buffer(g_up_ce, src, i * 256, b, off, 4);
        up_blit_add(b, off, 4, 4, MTLSizeMake(1, 1, 1), dst[i], 0, 0); [g_up_texs addObject:dst[i]];
        checked++;
    }
    up_flush(); while (atomic_load(&g_up_pending)) usleep(1000);
    /* Staging must be gone even though this NVN submission has not reached its fence. */
    pthread_mutex_lock(&g_res_mu);
    for (id obj in g_up_old.allObjects) if ([g_res_objects containsObject:obj]) MDIE("upload batch probe: completed staging still resident");
    pthread_mutex_unlock(&g_res_mu); checked++;
    for (int i = 0; i < N; i++) enc_b_copy_texture_buffer(blit(), dst[i], 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(1, 1, 1), result, i * 256, 256, 256);
    mtl_sub_commit(); mtl_finish();
    for (int i = 0; i < N; i++) for (int c = 0; c < 4; c++) {
        uint8_t want = (uint8_t)(i * 17 + c), actual = ((uint8_t *)result.contents)[i * 256 + c];
        if (actual != want) MDIE("upload batch probe: texture %d channel %d: %u != %u", i, c, actual, want);
        checked++;
    }
    for (int i = 0; i < N; i++) res_remove(dst[i]); res_remove(src); res_remove(result);
    return checked;
}

/* A sampling descriptor gains image/format-view usage while another render pass is open. The
   resolved view must retain every mip/layer written through the original storage descriptor. */
int mtl_probe_storage_view(void) {
    mtl_init(); const NSUInteger W=16,H=16;int checked=0;
    for(int test=0;test<10;test++) {
        int mode=test%2,kind=test/2,stage=test%2,reg=kind==2?31:127;
        NvProg *pr=[NvProg new];NvShader *sh=[NvShader new];pr->sh[stage]=sh;
        /* Exercise texture, typed-buffer and image dependency slots on both
           vertex/fragment preparation paths. GPU readback checks the actual
           storage upgrade and encoder split, not just the metadata list. */
        sh->rf.nb=2;sh->rf.b[0].kind=NVN_RB_CBUF;
        sh->rf.b[1].kind=kind==2?NVN_RB_IMG:kind==1?NVN_RB_TBUF:NVN_RB_TEX;sh->rf.b[1].reg=reg;
        uint64_t handles[32]={0};
        if(kind==3){sh->rf.b[1].kind=NVN_RB_TEXU;sh->rf.b[1].reg=0;sh->rf.b[1].count=8;sh->rf.b[1].off=8;handles[8]=1;}
        void *mem=calloc(1,4096);tex_t t={0};t.serial=0xffffffa0+mode*2;t.mem=mem;t.L.size=4096;
        t.d.width=W;t.d.height=H;t.d.depth=2;t.d.levels=2;t.d.target=4;t.d.format=0x25;
        tex_t alias=t;alias.serial++;if(!mode){alias.d.format=0x27;alias.d.flags=NVN_TEXTURE_FLAGS_IMAGE;}
        view_t view={0};view.set_format=1;view.format=0x27;
        mtl_reg_tex(1,&alias,mode?&view:NULL);
        id<MTLBuffer> rb=probe_buf(4096);op_hdr end={OP_END,8};mtl_sub_begin();mtl_exec((const uint8_t *)&end);
        if(kind==3)S.ubo[stage][0]=(p_bind_buf){.stage=stage,.index=0,.addr=(uint64_t)(uintptr_t)handles,.size=sizeof handles};
        else if(kind==2)S.img[stage][reg]=1;else S.tex[stage][reg]=1;
        for(int layer=0;layer<2;layer++)for(int level=0;level<2;level++) {
            memset(&R,0,sizeof R);R.has_rt=1;R.rt.n=1;R.rt.color[0]=&t;R.rt.has_cview[0]=1;
            view_t *v=&R.rt.cview[0];v->set_levels=v->set_layers=1;v->base_level=level;v->nlevels=1;v->min_layer=layer;v->nlayers=1;
            R.clear_c[0]=1;R.cc[0]=MTLClearColorMake((1+layer+2*level)/8.0,0.5,0.75,1);rp_begin();
            if(layer!=1||level!=1)rp_end();
        }
        if(kind==4){
            struct {op_hdr barrier;int32_t bits,pad;op_hdr end;} ordered={{OP_BARRIER,16},NVN_BARRIER_ORDER_FRAGMENTS_BIT|NVN_BARRIER_INVALIDATE_TEXTURE_BIT,0,{OP_END,8}};
            mtl_exec((const uint8_t *)&ordered);
#if DM_NVN_M3
            if(rp_active())MDIE("explicit texture barrier retained an active render encoder");
#endif
            checked++;
        }
        struct {op_hdr h;p_set_rt rt;op_hdr end;} future={{OP_SET_RT,8+sizeof(p_set_rt)},{0},{OP_END,8}};
        future.rt.n=1;future.rt.color[0]=&alias;
        const uint8_t *saved_next=g_exec_next;g_exec_next=(const uint8_t *)&future;uint64_t gen=g_ser_gen;
        pso_predict_rest();g_exec_next=saved_next;
        if(g_ser_gen!=gen || serial_get(alias.serial))MDIE("pipeline prediction changed GPU texture storage");checked++;
        if(getenv("TOTK_PROBE_NEGATIVE")) {registry_lock();(void)tex_fast(1);registry_unlock();}
        else prepare_draw_textures(pr);
        registry_lock();id<MTLTexture> sampled=tex_by_id(1);registry_unlock();
        if(!sampled)MDIE("storage-view probe missing sampling view");
        size_t off=0;for(int layer=0;layer<2;layer++)for(int level=0;level<2;level++) {
            NSUInteger w=W>>level,h=H>>level;
            enc_b_copy_texture_buffer(blit(),sampled,layer,level,MTLOriginMake(0,0,0),MTLSizeMake(w,h,1),rb,off,w*4,w*h*4);off+=w*h*4;
        }
        mtl_sub_commit();mtl_finish();off=0;
        for(int layer=0;layer<2;layer++)for(int level=0;level<2;level++) {
            NSUInteger w=W>>level,h=H>>level;uint8_t want[4]={(uint8_t)lround((1+layer+2*level)*255/8.0),128,191,255};
            const uint8_t *got=(const uint8_t *)rb.contents+off;
            for(NSUInteger i=0;i<w*h*4;i++){if(abs((int)got[i]-want[i%4])>1)MDIE("storage-view probe mode%d layer%d level%d byte%lu: %u != %u",mode,layer,level,i,got[i],want[i%4]);checked++;}
            off+=w*h*4;
        }
        memset(&R,0,sizeof R);mtl_reg_tex(1,NULL,NULL);serial_del(t.serial);serial_del(alias.serial);stor_drop_range((uintptr_t)mem,(uintptr_t)mem+4096);defer_obj_release(rb);free(mem);
    }
    return checked;
}

/* Compressible storage changing hands (stor_compressed): an R32F depth and an RGBA8 target on the same compressible pool
   memory. Clearing the RGBA8 one to white must leave the depth readable as before (on hardware the fast clear does not
   write the bytes; sharing one Metal texture turned the depth into NaN: the 2026-10-15 flashes); a new RGBA8 object on
   that storage sees the white (same-format content persists), a new R32F object after the depth was finalised gets the
   old depth back, and non-compressible textures still alias as plain memory. TOTK_PROBE_NEGATIVE: share across formats. */
static int probe_rb_check(id<MTLTexture> tx, NSUInteger w, NSUInteger h, uint32_t want, const char *what) {
    id<MTLBuffer> rb = probe_buf(w * h * 4); op_hdr end = {OP_END, 8};
    mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
    enc_b_copy_texture_buffer(blit(), tx, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(w, h, 1), rb, 0, w * 4, w * h * 4);
    mtl_sub_commit(); mtl_finish();
    const uint32_t *v = rb.contents; int n = 0;
    for (NSUInteger i = 0; i < w * h; i++) { if (v[i] != want) MDIE("compressed-alias probe, %s: texel %lu is %#x, want %#x", what, (unsigned long)i, v[i], want); n++; }
    defer_obj_release(rb);
    return n;
}
static void probe_clear(tex_t *t, MTLClearColor c) {
    op_hdr end = {OP_END, 8}; mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
    memset(&R, 0, sizeof R); R.has_rt = 1; R.rt.n = 1; R.rt.color[0] = t; R.clear_c[0] = 1; R.cc[0] = c; rp_begin(); rp_end();
    memset(&R, 0, sizeof R); mtl_sub_commit(); mtl_finish();
}
int mtl_probe_compressed_alias(void) {
    if (getenv("TOTK_PROBE_NEGATIVE")) setenv("TOTK_COMPRESSED_ALIAS", "1", 1);
    mtl_init(); enum { W = 16, H = 16 }; int checked = 0;
    for (int compressible = 1; compressible >= 0; compressible--) {
        tex_t depth = {0}; depth.serial = 0xffffff80 + compressible * 8;
        depth.d = (nvn_texdesc){.width = W, .height = H, .depth = 1, .levels = 1, .target = 1, .format = 0xa, .flags = compressible ? NVN_TEXTURE_FLAGS_COMPRESSIBLE : 0};
        nvn_tex_layout(&depth.d, &depth.L); size_t size = (depth.L.size + getpagesize() - 1) & ~((size_t)getpagesize() - 1);
        void *mem = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0); if (mem == MAP_FAILED) MDIE("compressed-alias probe mmap");
        pool_t pool = {.mem = mem, .size = size, .flags = NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | (compressible ? NVN_MEMORY_POOL_FLAGS_COMPRESSIBLE : 0)};
        depth.mem = mem; depth.pool = &pool;
        tex_t color = depth; color.serial++; color.d.format = 0x25; nvn_tex_layout(&color.d, &color.L);
        if (color.L.size != depth.L.size) MDIE("compressed-alias probe: layouts differ (%llu / %llu)", (unsigned long long)color.L.size, (unsigned long long)depth.L.size);
        probe_clear(&depth, MTLClearColorMake(0.25, 0, 0, 0));                 /* the linear depth, 0.25 = 0x3e800000 */
        probe_clear(&color, MTLClearColorMake(1, 1, 1, 1));                    /* the memory handed to an RGBA8 target, cleared to white */
        if (compressible) {
            checked += probe_rb_check(tex_mtl(&depth), W, H, 0x3e800000, "depth after the RGBA8 clear");       /* the stale descriptor still reads depth */
            checked += probe_rb_check(tex_mtl(&color), W, H, 0xffffffff, "RGBA8 target after its clear");
            tex_t again = color; again.serial = color.serial + 1;                                                /* the target re-initialised next frame */
            checked += probe_rb_check(tex_mtl(&again), W, H, 0xffffffff, "new RGBA8 object on the storage");
            mtl_tex_release(&depth);                                                                              /* the game finalises the depth object */
            tex_t back = depth; back.serial = color.serial + 2;                                                  /* and later reads that storage as R32F again */
            checked += probe_rb_check(tex_mtl(&back), W, H, 0x3e800000, "new R32F object after the depth was finalised");
            mtl_tex_release(&again); mtl_tex_release(&back);
        } else {
            checked += probe_rb_check(tex_mtl(&depth), W, H, 0xffffffff, "non-compressible depth after the RGBA8 clear (plain aliasing)");
            mtl_tex_release(&depth);
        }
        mtl_tex_release(&color);
        stor_drop_range((uintptr_t)mem, (uintptr_t)mem + size); mtl_finish(); munmap(mem, size);
    }
    return checked;
}

/* CPU-visible linear-render-target textures contain ordinary pool data until the GPU writes them.
   Exercise both water-height formats, cold sampling, a CPU texel update, and subsequent GPU rendering. */
int mtl_probe_linear_rt_upload(void) {
    mtl_init();int checked=0;
    for(int mode=0;mode<2;mode++) {
        const NSUInteger W=mode?68:260,H=W;
        tex_t t={0};t.serial=0xffffffe0+mode;t.d=(nvn_texdesc){.width=W,.height=H,.depth=1,.levels=1,.target=1,.format=mode?0x2a:0x6,.flags=NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET};
        nvn_tex_layout(&t.d,&t.L);size_t size=(t.L.size+getpagesize()-1)&~((size_t)getpagesize()-1);
        void *mem=mmap(NULL,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);if(mem==MAP_FAILED)MDIE("linear-rt probe mmap");
        pool_t pool={0};pool.mem=mem;pool.size=size;pool.flags=0x24;t.pool=&pool;t.mem=mem;
        const NSUInteger bpp=mode?8:2,row=W*bpp;uint8_t *pixels=malloc(row*H);
        for(NSUInteger i=0;i<row*H;i++)pixels[i]=(uint8_t)(i*29+i/13+17);
        if(nvn_swizzle(&t.d,0,0,pixels,row,mem))MDIE("linear-rt probe swizzle");
        mtl_pool_add(mem,size);nvn_dirty_add_range(mem,size);
        id<MTLBuffer> rb=probe_buf(row*H);op_hdr end={OP_END,8};mtl_sub_begin();mtl_exec((const uint8_t *)&end);
        id<MTLTexture> tex=tex_mtl(&t);if(!tex)MDIE("linear-rt probe missing texture");
        enc_b_copy_texture_buffer(blit(),tex,0,0,MTLOriginMake(0,0,0),MTLSizeMake(W,H,1),rb,0,row,row*H);
        mtl_sub_commit();mtl_finish();
        for(NSUInteger i=0;i<row*H;i++){if(((uint8_t *)rb.contents)[i]!=pixels[i])MDIE("linear-rt cold upload mode%d byte%lu: %u != %u",mode,i,((uint8_t *)rb.contents)[i],pixels[i]);checked++;}
        uint8_t change[8]={31,47,89,103,127,151,193,239};NVNcopyRegion region={.xoffset=17,.yoffset=23,.width=1,.height=1,.depth=1};
        mtl_write_texels(&t,0,0,&region,change,bpp,bpp);memcpy(pixels+23*row+17*bpp,change,bpp);
        mtl_sub_begin();mtl_exec((const uint8_t *)&end);tex=tex_mtl(&t);
        enc_b_copy_texture_buffer(blit(),tex,0,0,MTLOriginMake(0,0,0),MTLSizeMake(W,H,1),rb,0,row,row*H);
        mtl_sub_commit();mtl_finish();
        for(NSUInteger i=0;i<row*H;i++){if(((uint8_t *)rb.contents)[i]!=pixels[i])MDIE("linear-rt CPU update mode%d byte%lu",mode,i);checked++;}
        /* Once rendered, unrelated writes in the same CPU pool must not restore the old pixels. */
        mtl_sub_begin();mtl_exec((const uint8_t *)&end);memset(&R,0,sizeof R);R.has_rt=1;R.rt.n=1;R.rt.color[0]=&t;R.clear_c[0]=1;R.cc[0]=MTLClearColorMake(.25,.5,.75,1);rp_begin();rp_end();
        dm_guest_write_prepare(mem,size);((uint8_t *)mem)[size-1]^=0x80;tex=tex_mtl(&t);
        enc_b_copy_texture_buffer(blit(),tex,0,0,MTLOriginMake(0,0,0),MTLSizeMake(W,H,1),rb,0,row,row*H);mtl_sub_commit();mtl_finish();
        const uint16_t expected[4]={16384,32768,49151,65535};const uint16_t *got=rb.contents;
        for(NSUInteger i=0;i<W*H*bpp/2;i++){int want=expected[mode?i%4:0];if(abs((int)got[i]-want)>1)MDIE("linear-rt GPU render mode%d component%lu: %u != %d",mode,i,got[i],want);checked++;}
        memset(&R,0,sizeof R);mtl_tex_release(&t);mtl_pool_del(mem);nvn_dirty_remove_range(mem);mtl_finish();res_remove(rb);munmap(mem,size);free(pixels);
    }
    return checked;
}

/* Reproduce unused descriptor slots holding the old texture after streaming replaces its storage. */
int mtl_probe_texture_retire(void) {
    mtl_init(); enum { W=256,H=256,N=16 }; int checked=0;
    tex_t t={0}; t.serial=0xffffffb0;t.d=(nvn_texdesc){.width=W,.height=H,.depth=1,.levels=1,.target=1,.format=0x2a,.flags=0x40,.swizzle_r=2,.swizzle_g=3,.swizzle_b=4,.swizzle_a=5};
    nvn_tex_layout(&t.d,&t.L);size_t size=(t.L.size+getpagesize()-1)&~((size_t)getpagesize()-1);
    void *mem=mmap(NULL,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);if(mem==MAP_FAILED)MDIE("texture retire mmap");
    pool_t pool={.mem=mem,.size=size,.flags=0x24};t.mem=mem;t.pool=&pool;memset(mem,0,size);
    mtl_pool_add(mem,size);nvn_dirty_add_range(mem,size);
    view_t v={.set_levels=1,.nlevels=1};id __weak old[N]={nil};
    op_hdr end={OP_END,8};id<MTLBuffer> rb=probe_buf(128);uint8_t expected[128];
    for(unsigned round=0;round<N;round++) {
        mtl_sub_begin();mtl_exec((const uint8_t *)&end);
        dm_guest_write_prepare(mem,size);for(unsigned i=0;i<128;i++)((uint8_t *)mem)[i]=expected[i]=(uint8_t)(round*17+i*11+3);
        mtl_reg_tex(1+round*2,&t,NULL);mtl_reg_tex(2+round*2,&t,&v);
        registry_lock();id<MTLTexture> base=tex_by_id(1+round*2),view=tex_by_id(2+round*2),typed=tex_type_view(base,MTLTextureType2DArray);registry_unlock();
        if(!base||!view||view.parentTexture!=base||!typed||typed.textureType!=MTLTextureType2DArray)MDIE("texture retire fixture needs base, sampling and type views");
        old[round]=base;
        enc_b_copy_texture_buffer(blit(),typed,0,0,MTLOriginMake(0,0,0),MTLSizeMake(16,1,1),rb,0,128,128);
        mtl_sub_commit();mtl_finish();
        for(unsigned i=0;i<128;i++){if(((uint8_t *)rb.contents)[i]!=expected[i])MDIE("texture retire upload round%u byte%u",round,i);checked++;}
    }
    for(unsigned i=0;i<66;i++){mtl_sub_begin();mtl_exec((const uint8_t *)&end);mtl_sub_commit();mtl_finish();}
    /* With re-upload recycling on, a replaced texture is kept in the recycle pool (or already reused as a later round's
       base): only those may still be alive. TOTK_REUP_RECYCLE=0 keeps the strict check. */
    for(unsigned i=0;i<N-1;i++){id o=old[i];if(o&&!reup_holds(o)&&o!=old[N-1])MDIE("texture retire: unused descriptor still retains replaced texture round%u",i);checked++;}
    MLOG("texture retire: re-upload recycling %s, %llu reuses", reup_on()?"on":"off", (unsigned long long)g_reup_reused);
    registry_lock();
    if(!g_treg[2*N-1].view||!g_treg[2*N].view)MDIE("texture retire: current descriptor was evicted");checked+=2;
    registry_unlock();
    mtl_sub_begin();mtl_exec((const uint8_t *)&end);
    registry_lock();id<MTLTexture> again=tex_by_id(1);registry_unlock();
    if(again!=old[N-1])MDIE("texture retire: old descriptor failed to rebuild with current storage");checked++;
    id __weak replaced_type=[g_type_views objectForKey:again];
    registry_lock();(void)tex_type_view(again,MTLTextureType2D);registry_unlock();
    if(!replaced_type)MDIE("texture retire: changing type released a view before its submission finished");checked++;
    enc_b_copy_texture_buffer(blit(),replaced_type,0,0,MTLOriginMake(0,0,0),MTLSizeMake(16,1,1),rb,0,128,128);
    mtl_sub_commit();mtl_finish();for(unsigned i=0;i<128;i++){if(((uint8_t *)rb.contents)[i]!=expected[i])MDIE("texture retire rebuilt view pixel%u",i);checked++;}
    mtl_sub_begin();mtl_exec((const uint8_t *)&end);mtl_sub_commit();mtl_finish();
    if(replaced_type)MDIE("texture retire: replaced type view retained after GPU completion");checked++;
    mtl_tex_release(&t);
    registry_lock();
    for(unsigned i=1;i<=N*2;i++) { if(g_treg[i].view||g_treg[i].indexed_serial)MDIE("texture retire: finalized serial kept descriptor %u",i);checked++; }
    registry_unlock();
    for(unsigned i=1;i<=N*2;i++)mtl_reg_tex(i,NULL,NULL);
    /* Finalized CPU-upload wrappers use the same bounded recycler. Keep an old
       draw queued at finalization, then ensure all new pixels replace its content
       and that the allocation can be reused only after the earlier fence. */
    int saved_final = atomic_load(&g_pkg_final_cache);
    const char *ne = getenv("TOTK_PROBE_NEGATIVE"); int no_final = ne && atoi(ne) == 7;
    atomic_store(&g_pkg_final_cache, !no_final);
    id<MTLBuffer> final_rb = probe_buf(4096); id<MTLTexture> prior = nil;
    uint64_t kept_before = atomic_load(&g_pkg_final_kept), reused_before = g_reup_reused;
    for (unsigned round=0; round<8; round++) {
        @autoreleasepool {
            tex_t fresh=t; fresh.serial=0xffffa000+round;
            fresh.d=(nvn_texdesc){.width=32,.height=32,.depth=1,.levels=1,.target=1,.format=0x25,.flags=0x40,
                .swizzle_r=2,.swizzle_g=3,.swizzle_b=4,.swizzle_a=5}; nvn_tex_layout(&fresh.d,&fresh.L);
            dm_guest_write_prepare(mem,size); memset(mem,(int)(round*19+13),4096);
            mtl_sub_begin();mtl_exec((const uint8_t *)&end);
            id<MTLTexture> current=tex_mtl(&fresh);
            if(round && current!=prior)MDIE("texture final cache: completed allocation was not reused");
            enc_b_copy_texture_buffer(blit(),current,0,0,MTLOriginMake(0,0,0),MTLSizeMake(32,32,1),final_rb,0,128,4096);
            uint64_t pending_kept=atomic_load(&g_pkg_final_kept);
            mtl_tex_release(&fresh);
            if(atomic_load(&g_pkg_final_kept)!=pending_kept)MDIE("texture final cache: wrapper retired before GPU completion");
            mtl_sub_commit();mtl_finish();
            for(unsigned i=0;i<4096;i++){if(((uint8_t *)final_rb.contents)[i]!=(uint8_t)(round*19+13))MDIE("texture final cache: changed queued pixels round%u byte%u",round,i);checked++;}
            prior=current;
        }
        for(unsigned i=0;i<3;i++){mtl_sub_begin();mtl_exec((const uint8_t *)&end);mtl_sub_commit();mtl_finish();}
        if(atomic_load(&g_pkg_final_kept)<=kept_before+round)MDIE("texture final cache: finalized wrapper was not retained");
        checked+=3;
    }
    if(g_reup_reused<reused_before+7)MDIE("texture final cache: missing allocation reuse");checked++;
    atomic_store(&g_pkg_final_cache,saved_final);res_remove(final_rb);
    mtl_pool_del(mem);nvn_dirty_remove_range(mem);mtl_finish();res_remove(rb);munmap(mem,size);
    return checked;
}

/* Renamed inline uniform updates (upd_shadow / upd_flush). In one submission a uniform range is read before an update, after
   it, and after a second overlapping update: the reads must see the old, new and newest bytes; a buffer copy of the range
   in between must see the first update (the copy flushes the logged updates first); memory holds the newest bytes after the
   submission; an update marks the argument buffers dirty. Reads are compute dispatches from the address a draw would bind.
   TOTK_PROBE_NEGATIVE=1 binds memory instead of the private copy, =2 skips the check before the copy; both must fail. */
int mtl_probe_ub_rename(void) {
    mtl_init(); setenv("TOTK_UB_RENAME", "1", 1);
    const char *ne = getenv("TOTK_PROBE_NEGATIVE"); int neg = ne ? atoi(ne) : 0, checked = 0;
    if (!ub_rename_on()) MDIE("ub rename probe: renaming is off");
    const size_t PG = 16384; uint8_t *mem = NULL;
    if (posix_memalign((void **)&mem, PG, PG) || !mem) MDIE("ub rename probe memory");
    memset(mem, 0x11, PG); mtl_pool_add(mem, PG);
    NSError *e = nil;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]], uint i [[thread_position_in_grid]]) { dst[i] = src[i]; }\n" options:nil error:&e];
    id<MTLComputePipelineState> ps = lib ? [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"] error:&e] : nil;
    if (!ps) MDIE("ub rename probe pipeline: %s", e.localizedDescription.UTF8String);
    id<MTLBuffer> out = probe_buf(4 * 128);
    const uint64_t base = (uint64_t)(uintptr_t)mem + 256, copy_dst = (uint64_t)(uintptr_t)mem + 4096, size = 96;
    uint8_t want[3][96], u1[64], u2[16]; memset(want, 0x11, sizeof want); memset(u1, 0x22, sizeof u1); memset(u2, 0x33, sizeof u2);
    memset(want[1] + 16, 0x22, 64); memset(want[2] + 16, 0x22, 64); memset(want[2] + 32, 0x33, 16);
    g_upd_guard_off = neg == 2;
    op_hdr end = {OP_END, 8}; mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
    memset(&S.ubo, 0, sizeof S.ubo); S.ubo[1][3] = (p_bind_buf){1, 3, base, size}; S.ubo[0][0] = (p_bind_buf){0, 0, base + 2048, 256};
    for (int step = 0; step < 3; step++) {
        if (step) {
            memset(g_args_dirty, 0, sizeof g_args_dirty);
            if (step == 1) mtl_sub_update(u1, base + 16, 64); else mtl_sub_update(u2, base + 32, 16);
            for (int st = 0; st < 6; st++) if (g_args_dirty[st] != (st == 1)) MDIE("ub rename probe: update %d left stage %d argument buffers %s", step, st, g_args_dirty[st] ? "dirty" : "clean");
            checked += 6;
        }
        uint64_t gpu = neg == 1 ? 0 : upd_shadow(base, size);
        if (!gpu) { void *mb; uint64_t mo; if (!mtl_gpu_addr(base, size, &mb, &mo, &gpu)) MDIE("ub rename probe: range not in the pool"); }
        rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }   // as dispatch() does before its encoder
        id<MTL4ComputeCommandEncoder> ce = enc_c_begin(NULL); enc_c_pso(ce, ps); enc_c_args(ce, gpu, 0); enc_c_args(ce, out.gpuAddress + (uint64_t)step * 128, 1);
        [ce dispatchThreads:MTLSizeMake(size, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; enc_c_end(ce);
        if (step == 1) {   // a copy reads the renamed range: the logged update must reach memory before it
            if (g_nupd != 1) MDIE("ub rename probe: %d logged updates before the copy, expected 1", g_nupd);
            mtl_sub_copy(base, copy_dst, size); checked++;
        }
    }
    if (neg != 2 && g_nupd != 1) MDIE("ub rename probe: %d logged updates at the end, expected 1", g_nupd);
    mtl_sub_commit(); mtl_finish(); g_upd_guard_off = 0; memset(&S.ubo, 0, sizeof S.ubo);
    const uint8_t *o = out.contents;
    for (int step = 0; step < 3; step++) for (int i = 0; i < 96; i++) {
        if (o[step * 128 + i] != want[step][i]) MDIE("ub rename probe: read %d byte %d = %#x, expected %#x", step, i, o[step * 128 + i], want[step][i]);
        checked++;
    }
    for (int i = 0; i < 96; i++) { if (mem[4096 + i] != want[1][i]) MDIE("ub rename probe: copied byte %d = %#x, expected %#x", i, mem[4096 + i], want[1][i]); checked++; }
    for (int i = 0; i < 96; i++) { if (mem[256 + i] != want[2][i]) MDIE("ub rename probe: memory byte %d = %#x after the submission, expected %#x", i, mem[256 + i], want[2][i]); checked++; }
    for (int i = 0; i < 256; i++) { if (mem[i] != 0x11 || mem[352 + i] != 0x11) MDIE("ub rename probe: bytes outside the updates changed"); checked++; }
    /* Two reads separated by an overlapping update in one unflushed submission.
       An index keyed only by address/range would return the first immutable copy.
       Also force index collisions and the 512-entry authoritative-list rollover. */
    int saved_index=atomic_load(&g_shadow_index_on); atomic_store(&g_shadow_index_on,1);
    g_shadow_index_stale_probe=neg==8;
    mtl_sub_begin();mtl_exec((const uint8_t *)&end);
    mtl_sub_update(u1,base+16,64);
    uint64_t first=upd_shadow(base,size);
    if(!first || upd_shadow(base,size)!=first)MDIE("shadow index: same version not reused");checked++;
    mtl_sub_update(u2,base+32,16);
    uint64_t second=upd_shadow(base,size);
    if(second==first)MDIE("shadow index: stale version after overlapping update");checked++;
    id<MTL4ComputeCommandEncoder> se=enc_c_begin(NULL);enc_c_pso(se,ps);
    enc_c_args(se,first,0);enc_c_args(se,out.gpuAddress,1);
    [se dispatchThreads:MTLSizeMake(size,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];enc_c_end(se);
    se=enc_c_begin(NULL);enc_c_pso(se,ps);enc_c_args(se,second,0);enc_c_args(se,out.gpuAddress+128,1);
    [se dispatchThreads:MTLSizeMake(size,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];enc_c_end(se);
    /* Multiple containing ranges preserve the same update but occupy different
       list entries. Full list rollover must never expose an unrelated address. */
    for(unsigned n=0;n<UBO_SHADOW_MAX+19;n++) {
        uint64_t span=size+n, cached=upd_shadow(base,span);
        int previous=atomic_load(&g_shadow_index_on);atomic_store(&g_shadow_index_on,0);
        uint64_t reference=upd_shadow(base,span);atomic_store(&g_shadow_index_on,previous);
        if(!cached || cached!=reference)MDIE("shadow index: collision/rollover mismatch %u",n);checked++;
    }
    mtl_sub_commit();mtl_finish();
    for(int step=0;step<2;step++)for(int i=0;i<96;i++) {
        uint8_t expected=(i>=16 && i<80)?0x22:0x11;
        if(step && i>=32 && i<48)expected=0x33;
        if(((uint8_t *)out.contents)[step*128+i]!=expected)MDIE("shadow index: GPU bytes differ step%d offset%d",step,i);checked++;
    }
    g_shadow_index_stale_probe=0;atomic_store(&g_shadow_index_on,saved_index);
    mtl_pool_del(mem); return checked;
}

/* TOTK probe: uniform contents as of the submit call (mtl_sub_snapshot_ubos). A submission binds three uniform ranges, two
   close together (merged into one copy) and one far away behind an OP_CALL; after the snapshot the CPU rewrites all of
   them, as the game's threads do while a submission is replayed. Reads (compute dispatches from the address a draw would
   bind) and a pointer read (pend_read) must see the bytes of the submit call. A GPU buffer copy into the far range within
   the submission drops its copy: a later read sees the copied bytes. A range a writable storage buffer of the previous,
   still executing submission writes (held back by an event) is not copied: the read sees that write. Overlapping ranges
   past the 1 MiB cap give disjoint, sorted copies that hold each binding whole or not at all.
   TOTK_PROBE_NEGATIVE=1 turns the snapshot off, =2 keeps copies after a GPU write, =3 forgets the previous submission's
   GPU writes; each must fail. */
static size_t usp_bind(uint8_t *p, int st, int idx, uint64_t addr, uint64_t size) {
    op_hdr h = {OP_BIND_UBO, (uint32_t)(8 + sizeof(p_bind_buf))}; p_bind_buf b = {st, idx, addr, size};
    memcpy(p, &h, 8); memcpy(p + 8, &b, sizeof b); return h.size;
}
static void usp_read(id<MTLComputePipelineState> ps, uint64_t a, uint64_t n, id<MTLBuffer> out, uint64_t off) {
    const usnap_t *sc = usnap_find(a, n); void *mb; uint64_t mo, gpu;   // as encode_args chooses for a uniform binding
    if (sc) gpu = sc->gpu + (a - sc->lo); else if (!mtl_gpu_addr(a, n, &mb, &mo, &gpu)) MDIE("ubo snapshot probe: range not in the pool");
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    id<MTL4ComputeCommandEncoder> ce = enc_c_begin(NULL); enc_c_pso(ce, ps); enc_c_args(ce, gpu, 0); enc_c_args(ce, out.gpuAddress + off, 1);
    [ce dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; enc_c_end(ce);
}
static int usp_expect(const uint8_t *o, uint64_t n, uint8_t v, const char *what) {
    for (uint64_t i = 0; i < n; i++) if (o[i] != v) MDIE("ubo snapshot probe: %s byte %llu = %#x, expected %#x", what, (unsigned long long)i, o[i], v);
    return (int)n;
}
/* The probe rewrites guest memory right after the snapshot call, standing in for the game's threads that rebuild their
   constant pools milliseconds later. Since 2026-10-16 the copies are made by helper threads that may still be running when
   the call returns (usnap_take phase 2), so the probe waits for them first; the game never writes that early (the copies
   are done within ~0.15 ms, earlier than the single-threaded version finished them). */
static void usnap_wait_copies(void) { while (atomic_load_explicit(&g_ush_busy, memory_order_acquire)) sched_yield(); }
#define mtl_sub_snapshot_ubos(h, n) (mtl_sub_snapshot_ubos(h, n), usnap_wait_copies())
#define mtl_sub_snapshot_ubos_cap(h, n, c) (mtl_sub_snapshot_ubos_cap(h, n, c), usnap_wait_copies())
int mtl_probe_ubo_snapshot(void) {
    mtl_init();
    const char *ne = getenv("TOTK_PROBE_NEGATIVE"); int neg = ne ? atoi(ne) : 0, checked = 0;
    g_usnap_index_negative=neg==10;
    if(g_usnap_index_negative)atomic_store(&g_usnap_index_on,1);
    atomic_store(&g_usnap_on, neg == 1 ? 0 : 1); g_usnap_drop_off = neg == 2;
    const size_t PG = 16384; uint8_t *mem = NULL;
    if (posix_memalign((void **)&mem, PG, PG) || !mem) MDIE("ubo snapshot probe memory");
    memset(mem, 0x11, PG); memset(mem + 4096, 0x66, 128); mtl_pool_add(mem, PG);
    NSError *e = nil;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]], uint i [[thread_position_in_grid]]) { dst[i] = src[i]; }\n" options:nil error:&e];
    id<MTLComputePipelineState> ps = lib ? [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"] error:&e] : nil;
    if (!ps) MDIE("ubo snapshot probe pipeline: %s", e.localizedDescription.UTF8String);
    id<MTLBuffer> out = probe_buf(6 * 256), fill = probe_buf(256); memset(fill.contents, 0x77, 256);
    const uint64_t m = (uint64_t)(uintptr_t)mem, a0 = m + 256, a1 = m + 512, a2 = m + 8192, src = m + 4096;
    uint64_t r1[32], r2[16]; uint8_t *p = (uint8_t *)r1;   // R1: bind a0, bind a1, call R2, end; R2: bind a2, end
    p += usp_bind(p, 1, 3, a0, 96); p += usp_bind(p, 0, 2, a1, 64);
    { op_hdr h = {OP_CALL, 16}; uint64_t t = (uint64_t)(uintptr_t)r2; memcpy(p, &h, 8); memcpy(p + 8, &t, 8); p += 16; }
    { op_hdr h = {OP_END, 8}; memcpy(p, &h, 8); }
    p = (uint8_t *)r2; p += usp_bind(p, 4, 1, a2, 128); { op_hdr h = {OP_END, 8}; memcpy(p, &h, 8); }
    const uint64_t h1[1] = {(uint64_t)(uintptr_t)r1};
    op_hdr end = {OP_END, 8};

    /* 1. CPU rewrites after the submit call; a GPU copy within the submission */
    mtl_sub_begin(); mtl_sub_snapshot_ubos(h1, 1); mtl_exec((const uint8_t *)&end);
    if (neg != 1) {
        if (g_nusnap != 2) MDIE("ubo snapshot probe: %d copies, expected 2 (two close ranges merged, one far)", g_nusnap);
        if (!usnap_find(a0, 96) || usnap_find(a0, 96) != usnap_find(a1, 64) || !usnap_find(a2, 128) || usnap_find(a2, 128) == usnap_find(a0, 96)) MDIE("ubo snapshot probe: copies do not cover the bound ranges");
        checked += 2;
    }
    memset(mem + 256, 0x44, 96); memset(mem + 512, 0x45, 64); memset(mem + 8192, 0x55, 128);   // the game's threads, during the replay
    uint64_t ptr = 0; pend_read(a0 + 8, &ptr, 8, 0);
    if (ptr != 0x1111111111111111ull) MDIE("ubo snapshot probe: pointer read %#llx, expected the bytes of the submit call", (unsigned long long)ptr); checked++;
    usp_read(ps, a0, 96, out, 0); usp_read(ps, a1, 64, out, 256);
    mtl_sub_copy(src, a2, 128);   // a GPU write into a copied range: later readers must see it
    usp_read(ps, a2, 128, out, 512);
    mtl_sub_commit(); mtl_finish();
    const uint8_t *o = out.contents;
    checked += usp_expect(o, 96, 0x11, "read of the first range (bytes of the submit call)");
    checked += usp_expect(o + 256, 64, 0x11, "read of the merged neighbour");
    checked += usp_expect(o + 512, 128, 0x66, "read after the GPU copy");
    checked += usp_expect(mem + 8192, 128, 0x66, "memory after the GPU copy");
    if (g_nusnap != 0) MDIE("ubo snapshot probe: copies still listed after the commit"); checked++;

    /* 2. a writable storage buffer of the previous submission, still executing, writes a bound range */
    memset(mem + 256, 0x11, 96); memset(mem + 512, 0x11, 64);
    id<MTLSharedEvent> hold = [g_dev newSharedEvent]; hold.signaledValue = 0;
    mtl_sub_begin(); mtl_sub_snapshot_ubos(NULL, 0); mtl_exec((const uint8_t *)&end);
    { void *mb; uint64_t mo, gpu; if (!mtl_gpu_addr(a0, 96, &mb, &mo, &gpu)) MDIE("ubo snapshot probe: range not in the pool");
      usnap_drop(a0, 96);   // as encode_args does for a writable storage-buffer binding
      rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
      id<MTL4ComputeCommandEncoder> ce = enc_c_begin(NULL); enc_c_pso(ce, ps); enc_c_args(ce, fill.gpuAddress, 0); enc_c_args(ce, gpu, 1);
      [ce dispatchThreads:MTLSizeMake(96, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; enc_c_end(ce); }
    ((void)0);   // hold the writer back on the GPU
    uint64_t writer = mtl_sub_commit();
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 300 * NSEC_PER_MSEC), dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{ hold.signaledValue = 1; });   // release it even if something below waited
    if (neg == 3) g_ngw = 0;
    uint64_t skip0 = g_usnap_skip_gpu;
    mtl_sub_begin(); mtl_sub_snapshot_ubos(h1, 1); mtl_exec((const uint8_t *)&end);
    uint64_t done = __atomic_load_n(&g_done, __ATOMIC_ACQUIRE);
    if (done >= writer) MDIE("ubo snapshot probe: the held-back submission completed early (done %llu, writer %llu)", (unsigned long long)done, (unsigned long long)writer); checked++;
    if (neg != 3) {
        if (usnap_find(a0, 96) || usnap_find(a1, 64) || !usnap_find(a2, 128) || g_usnap_skip_gpu != skip0 + 1) MDIE("ubo snapshot probe: the range of the executing GPU write was copied (or the far range was not)");
        checked++;
    }
    usp_read(ps, a0, 96, out, 768);
    mtl_sub_commit(); hold.signaledValue = 1; mtl_finish();
    checked += usp_expect(o + 768, 96, 0x77, "read of a range the previous submission's storage write produces");

    /* 3. overlapping ranges past the 1 MiB cap: copies stay disjoint, sorted, at most 1 MiB each; a binding is held whole by
       one copy or read from memory. 24 ranges of 64 KiB, 60 KiB apart: 0..16 fill the first copy exactly, 17 straddles it,
       18..23 lie in the second copy. */
    if (neg != 1) {
        const size_t BIG = 2u << 20; uint8_t *big = NULL;
        if (posix_memalign((void **)&big, PG, BIG) || !big) MDIE("ubo snapshot probe big memory");
        memset(big, 0x22, BIG); mtl_pool_add(big, BIG);
        uint64_t rb[24 * 4 + 1]; uint8_t *q = (uint8_t *)rb; const uint64_t bb = (uint64_t)(uintptr_t)big;
        for (int i = 0; i < 24; i++) q += usp_bind(q, 1, i & 15, bb + (uint64_t)i * 60 * 1024, 64 * 1024);
        { op_hdr h = {OP_END, 8}; memcpy(q, &h, 8); }
        const uint64_t hb[1] = {(uint64_t)(uintptr_t)rb};
        mtl_sub_begin(); mtl_sub_snapshot_ubos(hb, 1); mtl_exec((const uint8_t *)&end);
        if (g_nusnap != 2) MDIE("ubo snapshot probe: %d copies of the overlapping ranges, expected 2", g_nusnap); checked++;
        for (int i = 0; i < g_nusnap; i++) {
            if (g_usnap[i].hi - g_usnap[i].lo > (1u << 20)) MDIE("ubo snapshot probe: copy %d holds %llu bytes", i, (unsigned long long)(g_usnap[i].hi - g_usnap[i].lo));
            if (i && g_usnap[i].lo < g_usnap[i - 1].hi) MDIE("ubo snapshot probe: copies %d and %d overlap", i - 1, i);
            checked += 2;
        }
        for (int i = 0; i < 24; i++) {
            uint64_t a = bb + (uint64_t)i * 60 * 1024; const usnap_t *c = usnap_find(a, 64 * 1024);
            if ((c != NULL) != (i != 17)) MDIE("ubo snapshot probe: range %d %s", i, c ? "found in a copy although it straddles two" : "not held by a copy");
            if (c && !(c->lo <= a && a + 64 * 1024 <= c->hi)) MDIE("ubo snapshot probe: copy does not hold range %d", i);
            checked++;
        }
        mtl_sub_commit(); mtl_finish(); mtl_pool_del(big);
    }
    /* 4. translation thread: the bytes are captured in the submit call and the replay starts later. (a) the game rewrites the
       ranges in between: the replay reads the captured bytes. (b) an earlier submission, replayed and completed after the
       capture, writes a bound range: that range is read on the GPU timeline (the capture predates the write). */
    if (neg != 1) {
        memset(mem + 256, 0x11, 96); memset(mem + 512, 0x11, 64); memset(mem + 8192, 0x11, 128);
        void *cap = neg == 4 ? NULL : mtl_sub_capture_ubos(h1, 1);
        memset(mem + 256, 0x44, 96); memset(mem + 512, 0x45, 64);   // the game, before the translation thread gets to it
        mtl_sub_begin(); mtl_sub_snapshot_ubos_cap(h1, 1, cap); mtl_exec((const uint8_t *)&end);
        usp_read(ps, a0, 96, out, 1024); usp_read(ps, a1, 64, out, 1024 + 128);
        mtl_sub_commit(); mtl_finish();
        checked += usp_expect(o + 1024, 96, 0x11, "read of a captured range rewritten before the replay");
        checked += usp_expect(o + 1024 + 128, 64, 0x11, "read of the merged neighbour, captured");
        memset(mem + 256, 0x11, 96);
        cap = mtl_sub_capture_ubos(h1, 1);
        mtl_sub_begin(); mtl_sub_snapshot_ubos(NULL, 0); mtl_exec((const uint8_t *)&end);   // the earlier submission, replayed after the capture
        { void *mb; uint64_t mo, gpu; if (!mtl_gpu_addr(a0, 96, &mb, &mo, &gpu)) MDIE("ubo snapshot probe: range not in the pool");
          usnap_drop(a0, 96);
          rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
          id<MTL4ComputeCommandEncoder> ce = enc_c_begin(NULL); enc_c_pso(ce, ps); enc_c_args(ce, fill.gpuAddress, 0); enc_c_args(ce, gpu, 1);
          [ce dispatchThreads:MTLSizeMake(96, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)]; enc_c_end(ce); }
        mtl_sub_commit(); mtl_finish();   // completed before the replay below starts
        g_ucap_probe_now = neg == 5;
        mtl_sub_begin(); mtl_sub_snapshot_ubos_cap(h1, 1, cap); mtl_exec((const uint8_t *)&end);
        g_ucap_probe_now = 0;
        if (neg != 5) { if (usnap_find(a0, 96) || !usnap_find(a2, 128)) MDIE("ubo snapshot probe: the range written after the capture was copied (or the far range was not)"); checked++; }
        usp_read(ps, a0, 96, out, 1280);
        mtl_sub_commit(); mtl_finish();
        checked += usp_expect(o + 1280, 96, 0x77, "read of a range an earlier submission wrote after the capture");
    }
    /* 5. A prior inline update finishes after pend_begin kept its CPU source pointer.
       Force arena churn during the new replay: GPU completion alone must not retire
       memory still referenced by pend_read. Negative 6 restores the former lifetime. */
    {
        uint8_t value[96]; memset(value, 0x5a, sizeof value);
        id<MTLSharedEvent> hold = [g_dev newSharedEvent]; hold.signaledValue = 0;
        mtl_sub_begin(); mtl_exec((const uint8_t *)&end); mtl_sub_update(value, a0, sizeof value);
        ((void)0); uint64_t writer = mtl_sub_commit();
        mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
        if (g_pend_reuse_done >= writer || !g_npend) MDIE("ubo snapshot probe: pending arena writer completed too early");
        hold.signaledValue = 1; mtl_wait(writer, UINT64_MAX);
        atomic_store(&g_pend_arena_pin, neg != 6);
        NSUInteger old_arenas = g_arenas.count;
        for (NSUInteger i = 0; i <= old_arenas; i++) { id<MTLBuffer> churn; size_t off = arena_alloc(4u << 20, &churn); memset((uint8_t *)churn.contents + off, 0xa7, 4u << 20); }
        uint64_t pointer = 0; pend_read(a0, &pointer, sizeof pointer, 0);
        if (pointer != 0x5a5a5a5a5a5a5a5aull) MDIE("ubo snapshot probe: pending arena source reused: %#llx, expected 0x5a5a5a5a5a5a5a5a", (unsigned long long)pointer);
        checked++; mtl_sub_commit(); mtl_finish(); atomic_store(&g_pend_arena_pin, 1);
    }
    /* 6. More disjoint ranges than hint slots; rebuild with half the entries.
       Poison live bytes after capture and verify the GPU still reads each range's
       distinct captured value. Hints must survive collisions/list reuse safely. */
    if(neg==0) {
        const unsigned NR=650;const size_t bytes=4u<<20;uint8_t *many=NULL;
        if(posix_memalign((void **)&many,16384,bytes)||!many)MDIE("snapshot index probe memory");
        memset(many,0,bytes);mtl_pool_add(many,bytes);
        uint8_t *cmd=calloc(NR,8+sizeof(p_bind_buf)+8);if(!cmd)MDIE("snapshot index probe commands");
        id<MTLBuffer> result=probe_buf(NR*16);int saved=atomic_load(&g_usnap_index_on);
        atomic_store(&g_usnap_index_on,1);atomic_store(&g_usnap_index_check,1);
        for(unsigned round=0;round<2;round++) {
            unsigned count=round?NR/2:NR;uint8_t *at=cmd;
            for(unsigned i=0;i<count;i++) {
                unsigned offset=(round?2*i:i)*4096;memset(many+offset,(uint8_t)(i*17+round*53),16);
                at+=usp_bind(at,0,0,(uint64_t)(uintptr_t)(many+offset),16);
            }
            memcpy(at,&end,8);uint64_t handles[]={(uint64_t)(uintptr_t)cmd};
            mtl_sub_begin();mtl_sub_snapshot_ubos(handles,1);mtl_exec((const uint8_t *)&end);
            if(g_nusnap!=(int)count)MDIE("snapshot index probe unexpected range count");checked++;
            memset(many,0xe7,bytes);
            for(unsigned i=0;i<count;i++) {
                uint64_t a=(uint64_t)(uintptr_t)(many+(round?2*i:i)*4096);
                if(!usnap_find(a,16))MDIE("snapshot index probe missing captured range");
                usp_read(ps,a,16,result,i*16);
            }
            mtl_sub_commit();mtl_finish();
            for(unsigned i=0;i<count;i++)checked+=usp_expect((uint8_t *)result.contents+i*16,16,(uint8_t)(i*17+round*53),"indexed range after list rebuild");
        }
        atomic_store(&g_usnap_index_check,0);atomic_store(&g_usnap_index_on,saved);
        mtl_pool_del(many);res_remove(result);free(many);free(cmd);
    }
    {   // the fault guard: a snapshot copy whose range is not readable (the game unmapped it) drops the entry instead of killing the process
        void *pg = mmap(NULL, 16384, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0); if (pg == MAP_FAILED) MDIE("ubo snapshot probe: mmap");
        uint8_t *buf = malloc(256); usnap_t e = {.lo = (uint64_t)(uintptr_t)pg, .hi = (uint64_t)(uintptr_t)pg + 256, .cpu = buf}; atomic_store(&e.st, US_COPYING);
        void *cap0 = g_usnap_cap; g_usnap_cap = NULL; uint64_t f0 = atomic_load(&g_usnap_faults);
        usnap_copy_one(&e);
        g_usnap_cap = cap0;
        if (atomic_load(&e.st) != US_DROPPED || atomic_load(&g_usnap_faults) != f0 + 1) MDIE("ubo snapshot probe: a copy of an unreadable range was not dropped (state %d, faults %llu -> %llu)", atomic_load(&e.st), (unsigned long long)f0, (unsigned long long)atomic_load(&g_usnap_faults));
        if (runtime_fault_guard) MDIE("ubo snapshot probe: the fault guard was left set");
        munmap(pg, 16384); free(buf); checked++;
        MLOG("ubo snapshot probe: a copy from an unreadable page was dropped by the fault guard (faults %llu)", (unsigned long long)atomic_load(&g_usnap_faults));
    }
    atomic_store(&g_usnap_on, 1); g_usnap_drop_off = 0; g_usnap_index_negative=0;
    mtl_pool_del(mem); return checked;
}
#undef mtl_sub_snapshot_ubos
#undef mtl_sub_snapshot_ubos_cap

/* TOTK probe: the driver constant segment c0 of a compute shader through the production dispatch (dispatch() ->
   encode_args). The kernel has the translated shaders' interface (cp_c0 / cp_s0 in argument buffer 0) and, like the
   occlusion-culling kernel ce4993a0, reads its thread-group counts from c0+0x200 and stores cell gy * nx + gx; the load-time
   scan (drv_scan) must find the read. A 5x3x1 dispatch must write all 15 cells with {5, 3, 1, tag}. Also checks the scan of
   vertex reads (storage table, point-size limit) and that it reports unfillable words. TOTK_DRV_FILL=0 (the old all-zero
   segment) must fail: every cell lands in the first row and the counts read 0. */
int mtl_probe_drv_c0(void) {
    driver_cache_set(1); // exercise the opt-in cache as well as the driver-bank contents
    mtl_init(); int checked = 0;
    static const char *msl =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct _16 { float4 _m0[4096]; };\nstruct _23 { uint _m0[1]; };\n"
        "struct spvDescriptorSetBuffer0 { constant _16* cp_c0 [[id(0)]]; device _23* cp_s0 [[id(1)]]; };\n"
        "kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])\n"
        "{\n    if (gl_LocalInvocationID.x != 0u || gl_LocalInvocationID.y != 0u) return;\n"
        "    uint nx = as_type<uint>((*spvDescriptorSet0.cp_c0)._m0[32].x), ny = as_type<uint>((*spvDescriptorSet0.cp_c0)._m0[32].y), nz = as_type<uint>((*spvDescriptorSet0.cp_c0)._m0[32].z);\n"
        "    uint t = gl_WorkGroupID.y * nx + gl_WorkGroupID.x;\n"
        "    (*spvDescriptorSet0.cp_s0)._m0[t * 4u] = nx; (*spvDescriptorSet0.cp_s0)._m0[t * 4u + 1u] = ny;\n"
        "    (*spvDescriptorSet0.cp_s0)._m0[t * 4u + 2u] = nz; (*spvDescriptorSet0.cp_s0)._m0[t * 4u + 3u] = 0xc0ffee00u + t;\n}\n";
    NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(msl) options:totk_precise_opts() error:&e];
    if (!lib) MDIE("drv-c0 probe: kernel does not compile: %s", e.localizedDescription.UTF8String);
    NvShader *cs = [NvShader new]; cs->fn = [lib newFunctionWithName:@"main0"]; cs->stage = 5; cs->src = @(msl);
    strlcpy(cs->key, "probe-drv-c0", sizeof cs->key);
    cs->rf.nb = 2; cs->rf.b[0].kind = NVN_RB_DRV0; cs->rf.b[0].aid = 0; cs->rf.b[1].kind = NVN_RB_SBUFP; cs->rf.b[1].aid = 1; cs->rf.b[1].reg = 0; cs->rf.b[1].rw = 1;
    cs->rf.tg[0] = 4; cs->rf.tg[1] = 2; cs->rf.tg[2] = 1;
    drv_scan(cs, cs->src);
    if (!cs->rf.drv_read[0] || !cs->drv_scanned[0]) MDIE("drv-c0 probe: the load-time scan missed the c0 read"); checked++;
    cs->ae = [cs->fn newArgumentEncoderWithBufferIndex:0]; cs->arg_len = cs->ae.encodedLength;
    if (cs->arg_len != 16) MDIE("drv-c0 probe: argument block %lu bytes, expected 16", (unsigned long)cs->arg_len);
    {   // vertex reads: storage table and point-size limit are filled, other words are reported
        NvShader *vs = [NvShader new]; vs->stage = 0; strlcpy(vs->key, "probe-drv-c0-vs", sizeof vs->key);
        vs->src = @"float a = (*spvDescriptorSet0.vp_c0)._m0[21].x; float b = (*spvDescriptorSet0.vp_c0)._m0[97].x;";
        if (drv_scan(vs, vs->src) != 0 || !vs->rf.drv_read[0]) MDIE("drv-c0 probe: vertex storage-table / point-size reads not accepted"); checked++;
        NvShader *bad = [NvShader new]; bad->stage = 0; strlcpy(bad->key, "probe-drv-c0-bad", sizeof bad->key);
        bad->src = @"float a = (*spvDescriptorSet0.vp_c0)._m0[200].x; float b = (*spvDescriptorSet0.fp_c2)._m0[i].y;";
        if (drv_scan(bad, bad->src) != 2) MDIE("drv-c0 probe: unfillable words were not reported"); checked++;
    }
    {   // c1 bank sharing: only plain indices inside the data (256 bytes -> 16 float4s, rounded + 256 zero bytes -> 32) share
        if (!c1_reads_bounded(@"float a = (*spvDescriptorSet0.fp_c1)._m0[9].x; float b = (*spvDescriptorSet0.vp_c1)._m0[31u].y; float c = (*spvDescriptorSet0.fp_c11)._m0[900].x;", 256))
            MDIE("c1 probe: plain in-range indices (and another buffer's c11) were not accepted"); checked++;
        if (c1_reads_bounded(@"float a = (*spvDescriptorSet0.fp_c1)._m0[int(_997)].x;", 256)) MDIE("c1 probe: a computed index was accepted"); checked++;
        if (c1_reads_bounded(@"float a = (*spvDescriptorSet0.cp_c1)._m0[32].x;", 256)) MDIE("c1 probe: an index past the data and its zero pad was accepted"); checked++;
        if (c1_reads_bounded(@"float a = (*spvDescriptorSet0.fp_c1)._m0[ 3].x;", 256)) MDIE("c1 probe: an index that is not a plain integer was accepted"); checked++;
        if (c1_reads_bounded(nil, 256)) MDIE("c1 probe: a shader without source was accepted"); checked++;
    }
    NvProg *pr = [NvProg new]; pr->sh[5] = cs;
    prog_t p = {0}; p.serial = 0xfffffd00; serial_put(p.serial, pr);
    const size_t PG = 16384; uint32_t *mem = NULL;
    if (posix_memalign((void **)&mem, PG, PG) || !mem) MDIE("drv-c0 probe memory");
    memset(mem, 0, PG); mtl_pool_add(mem, PG);
    op_hdr end = {OP_END, 8}; mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
    memset(&S.ssbo, 0, sizeof S.ssbo); S.ssbo[5][0] = (p_bind_buf){5, 0, (uint64_t)(uintptr_t)mem, PG};
    const prog_t *saved = S.prog; S.prog = &p;
    p_dispatch dp = {5, 3, 1, 0}; dispatch(&dp);
    S.prog = saved; mtl_sub_commit(); mtl_finish(); memset(&S.ssbo, 0, sizeof S.ssbo);
    for (uint32_t t = 0; t < 15; t++) {
        const uint32_t *c = mem + 4 * t;
        if (c[0] != 5 || c[1] != 3 || c[2] != 1 || c[3] != 0xc0ffee00u + t)
            MDIE("drv-c0 probe: cell %u = {%u, %u, %u, %#x}, expected {5, 3, 1, %#x}", t, c[0], c[1], c[2], c[3], 0xc0ffee00u + t);
        checked += 4;
    }
    /* Same submission: the second dispatch changes BOTH output binding and group counts.
       A stale bank reuses {5,3,1} and writes the wrong rows into the second output. */
    memset(mem, 0, PG); mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
    S.prog = &p; S.ssbo[5][0] = (p_bind_buf){5, 0, (uint64_t)(uintptr_t)mem, PG};
    dispatch(&dp);
    const char *negative = getenv("TOTK_PROBE_NEGATIVE");
    g_driver_probe_stale = negative && !strcmp(negative, "2");
    S.ssbo[5][0] = (p_bind_buf){5, 0, (uint64_t)(uintptr_t)(mem + 1024), PG - 4096};
    p_dispatch dp2 = {3, 2, 1, 0}; dispatch(&dp2);
    g_driver_probe_stale = 0;
    S.prog = saved; mtl_sub_commit(); mtl_finish(); memset(&S.ssbo, 0, sizeof S.ssbo);
    for (uint32_t t = 0; t < 6; t++) {
        const uint32_t *c = mem + 1024 + 4 * t;
        if (c[0] != 3 || c[1] != 2 || c[2] != 1 || c[3] != 0xc0ffee00u + t)
            MDIE("drv-cache probe: changed dispatch cell %u = {%u,%u,%u,%#x}, expected {3,2,1,%#x}", t,c[0],c[1],c[2],c[3],0xc0ffee00u+t);
        checked += 4;
    }
    /* The production cache builder's other dependencies: vertex SSBO size/address,
       point-size limit, c2 combined handles, separate samplers, compute uniform table. */
    mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
    id<MTLBuffer> bank_buffer;
    driver_bank_gpu(0,0,&bank_buffer);
    S.ssbo[0][3] = (p_bind_buf){0,3,0x123456789abcdef0ull,12345};
    driver_bank_gpu(0,0,&bank_buffer);
    driver_bank_cache *bc = &g_driver_bank[0][0];
    const uint8_t *bytes = (const uint8_t *)bank_buffer.contents + bc->offset;
    uint64_t value; memcpy(&value,bytes+0x140,8);
    if (value != 0x123456789abcdef0ull || *(const uint32_t *)(bytes+0x148)!=12345 || *(const float *)(bytes+0x610)!=2047.9375f) MDIE("drv-cache probe: changed vertex storage binding");
    checked += 3;
    driver_bank_gpu(2,0,&bank_buffer); S.tex[0][3]=0x80000042; S.samp[0][3]=0x123;
    driver_bank_gpu(2,0,&bank_buffer); bc=&g_driver_bank[0][1]; bytes=(const uint8_t *)bank_buffer.contents+bc->offset;
    if (*(const uint32_t *)(bytes+0x180)!=0x80000042 || *(const uint32_t *)(bytes+0x580)!=0x123 || *(const uint32_t *)(bytes+0x38)!=0x80000042) MDIE("drv-cache probe: changed texture/sampler handle");
    checked += 3;
    driver_bank_gpu(0,5,&bank_buffer); S.ubo[5][4]=(p_bind_buf){5,4,0x1122334455667788ull,456};
    driver_bank_gpu(0,5,&bank_buffer); bc=&g_driver_bank[5][0]; bytes=(const uint8_t *)bank_buffer.contents+bc->offset;
    memcpy(&value,bytes+0x250,8);
    if(value!=0x1122334455667788ull || *(const uint32_t *)(bytes+0x258)!=456) MDIE("drv-cache probe: changed compute uniform binding");
    checked += 2;
    uint64_t misses=g_driver_miss; driver_bank_gpu(0,5,&bank_buffer);
    if(driver_cache_on() && g_driver_miss!=misses) MDIE("drv-cache probe: identical bank was not reused"); checked++;
    mtl_sub_commit(); mtl_finish();
    mtl_sub_begin(); mtl_exec((const uint8_t *)&end); misses=g_driver_miss; driver_bank_gpu(0,5,&bank_buffer);
    if(g_driver_miss!=misses+1) MDIE("drv-cache probe: bank survived arena epoch change"); checked++;
    mtl_sub_commit(); mtl_finish();
    memset(&S.ssbo,0,sizeof S.ssbo); memset(&S.ubo,0,sizeof S.ubo); memset(&S.tex,0,sizeof S.tex); memset(&S.samp,0,sizeof S.samp);
    serial_put(p.serial, nil); mtl_pool_del(mem); free(mem);
    return checked;
}

/* TOTK probe: back-to-back dispatches through the production dispatch() share one compute encoder and stay ordered. Two
   kernels with the translated shaders' interface alternate 16 times in one submission over 2^20 elements: K1 does
   s0[i] += i + 1, K2 does s1[i] += s0[i], so s1[i] ends at (i + 1) * 16 * 17 / 2 only if every dispatch sees the previous
   one complete. Also counts the dispatches that ran in an already open encoder (31 of 32). Negative controls:
   TOTK_PROBE_NEGATIVE=1 drops the barrier between the shared encoder's commands (values must fail; a race, so it is
   reported if it does not show), TOTK_DISPATCH_REUSE=0 opens an encoder per dispatch (the sharing count must fail). */
int mtl_probe_dispatch_chain(void) {
    mtl_init(); int checked = 0;
    const char *ne = getenv("TOTK_PROBE_NEGATIVE"); int neg = ne ? atoi(ne) : 0; g_disp_probe_nobar = neg == 1;
    static const char *head = "#include <metal_stdlib>\nusing namespace metal;\nstruct _23 { uint _m0[1]; };\n"
        "struct spvDescriptorSetBuffer0 { device _23* cp_s0 [[id(0)]]; device _23* cp_s1 [[id(1)]]; };\n"
        "kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint3 g [[thread_position_in_grid]])\n{\n    uint i = g.x;\n";
    const char *body[2] = {"    (*spvDescriptorSet0.cp_s0)._m0[i] += i + 1u;\n}\n", "    (*spvDescriptorSet0.cp_s1)._m0[i] += (*spvDescriptorSet0.cp_s0)._m0[i];\n}\n"};
    NvProg *pr[2]; prog_t pg[2] = {{0}, {0}};
    for (int k = 0; k < 2; k++) {
        NSError *e = nil; NSString *msl = [NSString stringWithFormat:@"%s%s", head, body[k]];
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:msl options:totk_precise_opts() error:&e];
        if (!lib) MDIE("dispatch-chain probe: kernel %d does not compile: %s", k, e.localizedDescription.UTF8String);
        NvShader *cs = [NvShader new]; cs->fn = [lib newFunctionWithName:@"main0"]; cs->stage = 5; cs->src = msl;
        snprintf(cs->key, sizeof cs->key, "probe-dispatch-chain-%d", k);
        cs->rf.nb = 2; for (int b = 0; b < 2; b++) { cs->rf.b[b].kind = NVN_RB_SBUFP; cs->rf.b[b].aid = b; cs->rf.b[b].reg = b; cs->rf.b[b].rw = 1; }
        cs->rf.tg[0] = 64; cs->rf.tg[1] = 1; cs->rf.tg[2] = 1;
        cs->ae = [cs->fn newArgumentEncoderWithBufferIndex:0]; cs->arg_len = cs->ae.encodedLength;
        pr[k] = [NvProg new]; pr[k]->sh[5] = cs; pg[k].serial = 0xfffffd10 + (uint64_t)k; serial_put(pg[k].serial, pr[k]);
    }
    enum { N = 1 << 20, R = 16 };
    const size_t SZ = (size_t)N * 4; uint32_t *s0 = NULL, *s1 = NULL;
    if (posix_memalign((void **)&s0, 16384, SZ) || posix_memalign((void **)&s1, 16384, SZ) || !s0 || !s1) MDIE("dispatch-chain probe memory");
    memset(s0, 0, SZ); memset(s1, 0, SZ); mtl_pool_add(s0, SZ); mtl_pool_add(s1, SZ);
    op_hdr end = {OP_END, 8}; mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
    memset(&S.ssbo, 0, sizeof S.ssbo);
    S.ssbo[5][0] = (p_bind_buf){5, 0, (uint64_t)(uintptr_t)s0, SZ}; S.ssbo[5][1] = (p_bind_buf){5, 1, (uint64_t)(uintptr_t)s1, SZ};
    const prog_t *saved = S.prog; uint64_t shared0 = g_disp_shared;
    rp_end(); blit_close();
    for (int r = 0; r < R; r++) for (int k = 0; k < 2; k++) { S.prog = &pg[k]; p_dispatch dp = {N / 64, 1, 1, 0}; dispatch(&dp); }
    uint64_t shared = g_disp_shared - shared0;
    S.prog = saved; mtl_sub_commit(); mtl_finish(); memset(&S.ssbo, 0, sizeof S.ssbo);
    if (neg != 1 && shared != 2 * R - 1) MDIE("dispatch-chain probe: %llu of %d dispatches ran in an open encoder, expected %d", (unsigned long long)shared, 2 * R, 2 * R - 1);
    checked++;
    uint64_t bad = 0, first = N;
    for (uint32_t i = 0; i < N; i++) { uint32_t want = (i + 1u) * (uint32_t)(R * (R + 1) / 2); if (s1[i] != want || s0[i] != (i + 1u) * R) { if (!bad++) first = i; } }
    if (bad) MDIE("dispatch-chain probe: %llu elements wrong, first %llu: s0 %u s1 %u, expected %u %u", (unsigned long long)bad, (unsigned long long)first,
                  s0[first], s1[first], (uint32_t)(first + 1) * R, (uint32_t)(first + 1) * (uint32_t)(R * (R + 1) / 2));
    checked += 2;
    if (neg == 1) MLOG("dispatch-chain probe: without the barrier the values still came out right (the race did not show this time)");
    for (int k = 0; k < 2; k++) serial_put(pg[k].serial, nil);
    mtl_pool_del(s0); mtl_pool_del(s1); free(s0); free(s1); g_disp_probe_nobar = 0;
    return checked;
}

/* TOTK probe: layered passes through the production replay (mtl_exec). Command records select single layers of a cube
   and of 2D-array levels; each draw's colour encodes its draw number (vertex id / 3). Scenarios: a whole face chain; a
   range starting at layer 2 ended by a full clear and by a layer below the range; a pass opened by a partial clear (the
   clear quad lands on the first layer only); a draw whose vertex function cannot write the layer; a layer of the range
   that no draw touches (loaded and stored unchanged). Every layer and level is read back, and the Metal render passes,
   merged SetRenderTargets and refusals are counted. Negative controls: TOTK_PROBE_NEGATIVE=1 makes every draw write layer 0
   of its pass (pixels must fail), TOTK_PROBE_NEGATIVE=2 merges no SetRenderTargets (pass counts must fail). */
static uint8_t *lp_rec(uint8_t *w, uint32_t op, const void *payload, size_t n) {
    op_hdr h = {op, (uint32_t)((8 + n + 7) & ~(size_t)7)}; memset(w, 0, h.size); memcpy(w, &h, sizeof h); if (n) memcpy(w + 8, payload, n); return w + h.size;
}
static uint8_t *lp_rt(uint8_t *w, const tex_t *t, int level, int layer) {
    p_set_rt r; memset(&r, 0, sizeof r); r.n = 1; r.color[0] = t; r.has_cview[0] = 1;
    r.cview[0].set_levels = r.cview[0].set_layers = 1; r.cview[0].base_level = (uint8_t)level; r.cview[0].nlevels = 1;
    r.cview[0].min_layer = (uint16_t)layer; r.cview[0].nlayers = 1;
    return lp_rec(w, OP_SET_RT, &r, sizeof r);
}
static uint8_t *lp_draw(uint8_t *w, int k) { p_draw d = {4, 3 * k, 3, 0, 0}; return lp_rec(w, OP_DRAW, &d, sizeof d); }
static uint8_t *lp_sc(uint8_t *w, int x, int y, int sw, int sh) { p_scissor s = {x, y, sw, sh}; return lp_rec(w, OP_SCISSOR, &s, sizeof s); }
static uint8_t *lp_prog(uint8_t *w, const prog_t *p) { p_bind_program b = {p, 0x1f}; return lp_rec(w, OP_BIND_PROGRAM, &b, sizeof b); }
static uint8_t *lp_clear(uint8_t *w, float r, float g, float b, float a) { p_clear_color c = {0, 0xf, {r, g, b, a}}; return lp_rec(w, OP_CLEAR_COLOR, &c, sizeof c); }
static uint8_t *lp_end(uint8_t *w) { return lp_rec(w, OP_END, NULL, 0); }
static uint32_t lp_px(int k) { return (uint32_t)lroundf(255.f * (float)(k + 1) / 64.f) | 64u << 8 | 128u << 16 | 255u << 24; }   // the draw's colour (R from the draw number)
typedef struct { __unsafe_unretained id<MTLTexture> tex; int level, layers, size; uint32_t want[6][64]; } lp_img;   // tex: held by the probe's locals
static void lp_fill(lp_img *im, int layer, int x0, int x1, uint32_t v) { for (int y = 0; y < im->size; y++) for (int x = x0; x < x1 && x < im->size; x++) im->want[layer][y * im->size + x] = v; }
// Replays one scenario and checks the passes it opened, the SetRenderTargets merged into layered passes and the refusals.
static void lp_run(const char *name, const uint8_t *ops, uint64_t passes, uint64_t layered, uint64_t merges, uint64_t refused) {
    uint64_t p0 = g_stat_pass, l0 = g_lay.passes, m0 = g_lay.merges, r0 = g_lay.refused;
    mtl_exec(ops); rp_end();
    uint64_t dp = g_stat_pass - p0, dl = g_lay.passes - l0, dm = g_lay.merges - m0, dr = g_lay.refused - r0;
    MLOG("layer probe %s: %llu passes (%llu layered), %llu SetRenderTargets merged, %llu refused", name, (unsigned long long)dp, (unsigned long long)dl, (unsigned long long)dm, (unsigned long long)dr);
    if (dp != passes || dl != layered || dm != merges || dr != refused)
        MDIE("layer probe %s: %llu passes / %llu layered / %llu merged / %llu refused, expected %llu / %llu / %llu / %llu", name, (unsigned long long)dp, (unsigned long long)dl,
             (unsigned long long)dm, (unsigned long long)dr, (unsigned long long)passes, (unsigned long long)layered, (unsigned long long)merges, (unsigned long long)refused);
}
int mtl_probe_layer_merge(void) {
    mtl_init(); setenv("TOTK_GPU_MIDCLEAR", "1", 1); g_lay_probe_neg = getenv("TOTK_PROBE_NEGATIVE") ? atoi(getenv("TOTK_PROBE_NEGATIVE")) : 0;
    if (!lay_on()) MDIE("layer probe: layered passes are off (TOTK_LAYER_MERGE=0 or DM_NVN_TBDR=1)");
    NSString *vsrc = @"#include <metal_stdlib>\nusing namespace metal;\n\nstruct main0_out\n{\n    float4 m_0 [[user(locn0)]];\n    float4 gl_Position [[position, invariant]];\n    float gl_PointSize [[point_size]];\n};\n\n"
        "vertex main0_out main0(uint gl_VertexIndex [[vertex_id]])\n{\n    main0_out out = {};\n    out.gl_PointSize = 1.0;\n    uint k = gl_VertexIndex / 3u, i = gl_VertexIndex % 3u;\n"
        "    out.gl_Position = float4(float((i << 1u) & 2u) * 2.0 - 1.0, float(i & 2u) * 2.0 - 1.0, 0.0, 1.0);\n"
        "    out.m_0 = float4(float(k + 1u) / 64.0, 0.25, 0.5, 1.0);\n    if (k >= 64u)\n    {\n        return out;\n    }\n    return out;\n}\n";
    NSString *fsrc = @"#include <metal_stdlib>\nusing namespace metal;\n\nstruct main0_out\n{\n    float4 c [[color(0)]];\n};\n\nstruct main0_in\n{\n    float4 m_0 [[user(locn0)]];\n};\n\n"
        "fragment main0_out main0(main0_in in [[stage_in]])\n{\n    main0_out out = {};\n    out.c = in.m_0;\n    return out;\n}\n";
    NSError *e = nil; id<MTLLibrary> vl = [g_dev newLibraryWithSource:vsrc options:totk_precise_opts() error:&e], fl = vl ? [g_dev newLibraryWithSource:fsrc options:totk_precise_opts() error:&e] : nil;
    if (!vl || !fl) MDIE("layer probe shaders: %s", e.localizedDescription.UTF8String);
    NvShader *vs = [NvShader new], *vs_plain = [NvShader new], *fs = [NvShader new];
    vs->fn = vs_plain->fn = [vl newFunctionWithName:@"main0"]; vs->src = vsrc; strcpy(vs->key, "probe-layer-vs"); strcpy(vs_plain->key, "probe-layer-vs-without-msl");
    fs->fn = [fl newFunctionWithName:@"main0"]; fs->src = fsrc; fs->stage = 1; strcpy(fs->key, "probe-layer-fs");
    NvProg *pr = [NvProg new], *pr_plain = [NvProg new]; pr->sh[0] = vs; pr_plain->sh[0] = vs_plain; pr->sh[1] = pr_plain->sh[1] = fs;
    prog_t p1, p2; memset(&p1, 0, sizeof p1); memset(&p2, 0, sizeof p2); p1.serial = 0xffffffb0; p2.serial = 0xffffffb1; serial_put(p1.serial, pr); serial_put(p2.serial, pr_plain);
    MTLTextureDescriptor *d = [MTLTextureDescriptor new]; d.pixelFormat = MTLPixelFormatRGBA8Unorm; d.usage = MTLTextureUsageRenderTarget; d.storageMode = MTLStorageModePrivate;
    d.textureType = MTLTextureTypeCube; d.width = d.height = 8; d.mipmapLevelCount = 2; id<MTLTexture> cube = [g_dev newTextureWithDescriptor:d];
    d.textureType = MTLTextureType2DArray; d.mipmapLevelCount = 1; d.arrayLength = 6; id<MTLTexture> arr = [g_dev newTextureWithDescriptor:d];
    d.width = d.height = 4; d.arrayLength = 4; id<MTLTexture> arr2 = [g_dev newTextureWithDescriptor:d];
    if (!cube || !arr || !arr2) MDIE("layer probe textures"); res_add(cube); res_add(arr); res_add(arr2);
    tex_t ct, at, bt; memset(&ct, 0, sizeof ct); memset(&at, 0, sizeof at); memset(&bt, 0, sizeof bt);
    ct.serial = 0xffffffb2; at.serial = 0xffffffb3; bt.serial = 0xffffffb4; ct.d.format = at.d.format = bt.d.format = 0x25;
    ct.d.width = ct.d.height = at.d.width = at.d.height = 8; bt.d.width = bt.d.height = 4; ct.d.depth = 1; at.d.depth = 6; bt.d.depth = 4;
    ct.d.levels = 2; at.d.levels = bt.d.levels = 1; ct.d.target = 8; at.d.target = bt.d.target = 4;
    serial_put(ct.serial, cube); serial_put(at.serial, arr); serial_put(bt.serial, arr2);
    lp_img im[4]; memset(im, 0, sizeof im);
    im[0] = (lp_img){cube, 0, 6, 8}; im[1] = (lp_img){cube, 1, 6, 4}; im[2] = (lp_img){arr, 0, 6, 8}; im[3] = (lp_img){arr2, 0, 4, 4};
    mtl_sub_begin(); { op_hdr end = {OP_END, 8}; mtl_exec((const uint8_t *)&end); }   // replay state and an open submission
    for (int i = 0; i < 4; i++) {   // every layer of every level starts transparent black
        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new]; rp.colorAttachments[0].texture = im[i].tex; rp.colorAttachments[0].level = (NSUInteger)im[i].level;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0); rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.renderTargetArrayLength = (NSUInteger)im[i].layers; ser_gate(); enc_r_end(enc_r_begin(rp));
    }
    static uint8_t ops[65536]; uint8_t *w;
    // A: the six faces of cube level 1 in order: one layered pass, five merged SetRenderTargets
    w = lp_prog(ops, &p1); w = lp_sc(w, 0, 0, 8, 8);
    for (int f = 0; f < 6; f++) { w = lp_rt(w, &ct, 1, f); w = lp_draw(w, f); lp_fill(&im[1], f, 0, 4, lp_px(f)); }
    lp_end(w); lp_run("A cube faces", ops, 1, 1, 5, 0);
    // B: array layers 2-5 (lookahead stops at layer 1); the full clear on layer 4 ends the layered pass and clears layer 4 only,
    // layer 5 and layer 1 get passes of their own
    w = lp_sc(ops, 0, 0, 8, 8);
    w = lp_rt(w, &at, 0, 2); w = lp_draw(w, 10); w = lp_rt(w, &at, 0, 3); w = lp_draw(w, 11);
    w = lp_rt(w, &at, 0, 4); w = lp_clear(w, 1, 0, 0, 1); w = lp_sc(w, 0, 0, 4, 8); w = lp_draw(w, 12); w = lp_sc(w, 0, 0, 8, 8);
    w = lp_rt(w, &at, 0, 5); w = lp_draw(w, 13); w = lp_rt(w, &at, 0, 1); w = lp_draw(w, 14); lp_end(w);
    lp_fill(&im[2], 2, 0, 8, lp_px(10)); lp_fill(&im[2], 3, 0, 8, lp_px(11)); lp_fill(&im[2], 4, 0, 4, lp_px(12)); lp_fill(&im[2], 4, 4, 8, 0xff0000ffu);
    lp_fill(&im[2], 5, 0, 8, lp_px(13)); lp_fill(&im[2], 1, 0, 8, lp_px(14));
    lp_run("B array range, clear, lower layer", ops, 4, 1, 2, 0);
    // C: cube level 0, the pass opened by a partial clear of face 0 (clear quad on the first layer), draws on the right half
    w = lp_rt(ops, &ct, 0, 0); w = lp_sc(w, 0, 0, 4, 8); w = lp_clear(w, 0, 1, 0, 1); w = lp_sc(w, 4, 0, 4, 8); w = lp_draw(w, 20);
    w = lp_rt(w, &ct, 0, 1); w = lp_draw(w, 21); w = lp_rt(w, &ct, 0, 2); w = lp_draw(w, 22); lp_end(w);
    lp_fill(&im[0], 0, 0, 4, 0xff00ff00u); lp_fill(&im[0], 0, 4, 8, lp_px(20)); lp_fill(&im[0], 1, 4, 8, lp_px(21)); lp_fill(&im[0], 2, 4, 8, lp_px(22));
    lp_run("C opened by a partial clear", ops, 1, 1, 2, 0);
    // D: cube level 0 faces 3-5; the draw on face 4 cannot write the layer (no MSL): the layered pass ends, face 4 and 5 are plain
    w = lp_sc(ops, 0, 0, 8, 8); w = lp_rt(w, &ct, 0, 3); w = lp_draw(w, 30); w = lp_rt(w, &ct, 0, 4); w = lp_prog(w, &p2); w = lp_draw(w, 31);
    w = lp_prog(w, &p1); w = lp_rt(w, &ct, 0, 5); w = lp_draw(w, 32); lp_end(w);
    lp_fill(&im[0], 3, 0, 8, lp_px(30)); lp_fill(&im[0], 4, 0, 8, lp_px(31)); lp_fill(&im[0], 5, 0, 8, lp_px(32));
    lp_run("D draw without a layer output", ops, 3, 1, 1, 1);
    // E: array layer 1 drawn in a plain pass, then a layered pass over 0-2 that skips layer 1 (loaded and stored unchanged)
    w = lp_rt(ops, &bt, 0, 1); w = lp_draw(w, 40); w = lp_rt(w, &bt, 0, 0); w = lp_draw(w, 41); w = lp_rt(w, &bt, 0, 2); w = lp_draw(w, 42); lp_end(w);
    lp_fill(&im[3], 1, 0, 4, lp_px(40)); lp_fill(&im[3], 0, 0, 4, lp_px(41)); lp_fill(&im[3], 2, 0, 4, lp_px(42));
    lp_run("E untouched layer in the range", ops, 2, 1, 1, 0);
    id<MTLBuffer> rb[4]; for (int i = 0; i < 4; i++) rb[i] = probe_buf(6 * 256);
    for (int i = 0; i < 4; i++) for (int l = 0; l < im[i].layers; l++)
        enc_b_copy_texture_buffer(blit(), im[i].tex, (NSUInteger)l, (NSUInteger)im[i].level, MTLOriginMake(0, 0, 0), MTLSizeMake((NSUInteger)im[i].size, (NSUInteger)im[i].size, 1),
                                  rb[i], (NSUInteger)l * 256, (NSUInteger)im[i].size * 4, (NSUInteger)(im[i].size * im[i].size * 4));
    mtl_sub_commit(); mtl_finish();
    int checked = 0;
    for (int i = 0; i < 4; i++) for (int l = 0; l < im[i].layers; l++) for (int p = 0; p < im[i].size * im[i].size; p++) {
        const uint8_t *got = (const uint8_t *)rb[i].contents + l * 256 + p * 4; uint32_t want = im[i].want[l][p];
        for (int c = 0; c < 4; c++) { int wc = (int)(want >> (8 * c) & 255);
            if (abs((int)got[c] - wc) > 1) MDIE("layer probe: texture %d level %d layer %d pixel %d,%d channel %d = %u, expected %d", i, im[i].level, l, p % im[i].size, p / im[i].size, c, got[c], wc);
            checked++; }
    }
    memset(&R, 0, sizeof R); state_reset(); g_lay_probe_neg = 0;
    for (int i = 0; i < 5; i++) serial_del((uint64_t[]){p1.serial, p2.serial, ct.serial, at.serial, bt.serial}[i]);
    for (int i = 0; i < 4; i++) defer_obj_release(rb[i]);
    for (id o in @[cube, arr, arr2]) defer_obj_release(o);
    return checked + 5 * 4;
}

/* Render through the same descriptor and stream-binding helper as draw_inner.
   Populate every unused stream with poison; change the used color stream; use
   the descriptor's zero-input fallback; compare both modes to expected pixels. */
int mtl_probe_vertex_streams(void) {
    mtl_init();int checked=0;
    const size_t size=16384;void *mem=NULL;if(posix_memalign(&mem,size,size)||!mem)MDIE("vertex stream probe memory");
    memset(mem,0,size);mtl_pool_add(mem,size);
    float positions[12]={-1,-1,0,1,3,-1,0,1,-1,3,0,1};memcpy(mem,positions,sizeof positions);
    float colors[12]={.25,.5,.75,1,.25,.5,.75,1,.25,.5,.75,1};memcpy((uint8_t *)mem+256,colors,sizeof colors);
    float poison[12]={1,0,1,1,1,0,1,1,1,0,1,1};memcpy((uint8_t *)mem+512,poison,sizeof poison);
    NSError *error=nil;
    NSString *source=@"#include <metal_stdlib>\nusing namespace metal;\nstruct In { float4 p [[attribute(0)]]; float4 c [[attribute(3)]]; }; struct Out { float4 p [[position]]; float4 c; }; vertex Out vs(In i [[stage_in]]) { return {i.p,i.c}; } fragment float4 fs(Out i [[stage_in]]) { return i.c; }\n";
    id<MTLLibrary> lib=[g_dev newLibraryWithSource:source options:nil error:&error];if(!lib)MDIE("vertex stream probe shader: %s",error.localizedDescription.UTF8String);
    NvShader *vs=[NvShader new],*fs=[NvShader new];vs->fn=[lib newFunctionWithName:@"vs"];fs->fn=[lib newFunctionWithName:@"fs"];
    vs->rf.route_b=1;vs->rf.nv=2;vs->rf.v[0].loc=0;vs->rf.v[0].nvn=0;vs->rf.v[1].loc=3;vs->rf.v[1].nvn=3;vs->io=fs->io=@{};
    MTLTextureDescriptor *td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
    td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget;
    id<MTLTexture> texture=[g_dev newTextureWithDescriptor:td];res_add(texture);id<MTLBuffer> rb=probe_buf(1024);
    op_hdr end={OP_END,8};int saved=atomic_load(&g_vertex_streams_on), saved_arg=atomic_load(&g_arg_bind_cache_on);
    for(int cache_mode=0;cache_mode<2;cache_mode++)for(int mode=0;mode<2;mode++)for(int scenario=0;scenario<3;scenario++) {
        atomic_store(&g_arg_bind_cache_on,cache_mode);
        g_arg_bind_cache_negative=cache_mode && getenv("TOTK_PROBE_NEGATIVE") && atoi(getenv("TOTK_PROBE_NEGATIVE"))==15;
        atomic_store(&g_vertex_streams_on,mode);g_vertex_streams_negative=mode && getenv("TOTK_PROBE_NEGATIVE") && atoi(getenv("TOTK_PROBE_NEGATIVE"))==9;
        mtl_sub_begin();mtl_exec((const uint8_t *)&end);
        S.nattr=4;memset(S.attr,0,sizeof S.attr);memset(S.stream,0,sizeof S.stream);
        S.attr[0]=(va_st){.fmt=0x2e,.stream=2};S.attr[3]=(va_st){.fmt=scenario==2?0:0x2e,.stream=scenario==1?7:5};
        memset(S.vbuf,0,sizeof S.vbuf);
        for(unsigned i=0;i<15;i++){S.stream[i].stride=16;S.vbuf[i]=(NVNbufferRange){.address=(uint64_t)(uintptr_t)mem+512,.size=48};}
        S.vbuf[2].address=(uint64_t)(uintptr_t)mem;S.vbuf[scenario==1?7:5].address=(uint64_t)(uintptr_t)mem+256;
        uint32_t formats[8]={MTLPixelFormatRGBA8Unorm};pso_key key;pso_key_fill(&key,vs,fs,formats,0,0,1,0);key.chmask=15;
        MTLRenderPipelineDescriptor *pd=pso_desc(&key,vs,fs,nil,0);
        id<MTLRenderPipelineState> ps=[g_dev newRenderPipelineStateWithDescriptor:pd error:&error];if(!ps)MDIE("vertex stream probe pipeline: %s",error.localizedDescription.UTF8String);
        MTL4RenderPassDescriptor *rp=[MTL4RenderPassDescriptor new];rp.colorAttachments[0].texture=texture;rp.colorAttachments[0].loadAction=MTLLoadActionClear;rp.colorAttachments[0].storeAction=MTLStoreActionStore;
        rp.renderTargetWidth=rp.renderTargetHeight=16;id<MTL4RenderCommandEncoder> en=enc_r_begin(rp);enc_r_pso(en,ps);enc_r_cull(en,MTLCullModeNone);
        enc_r_viewport(en,(MTLViewport){0,0,16,16,0,1});enc_r_scissor(en,(MTLScissorRect){0,0,16,16});
        void *mb;uint64_t off,pgpu;if(!mtl_gpu_addr((uint64_t)(uintptr_t)mem+512,48,&mb,&off,&pgpu))MDIE("vertex stream probe poison mapping");
        for(unsigned i=0;i<15;i++)enc_r_vbuf(en,16+i,pgpu);
        uint64_t bound=g_vertex_streams_bound;bind_vertex_streams(en,vs);
        unsigned expected_bind=mode?(scenario==2?1:2):15;
        if(g_vertex_streams_bound-bound!=expected_bind && !g_vertex_streams_negative)MDIE("vertex stream probe unexpected binding count");checked++;
        uint64_t skips=g_arg_bind_skips;bind_vertex_streams(en,vs);
        if(cache_mode && !g_arg_bind_cache_negative && g_arg_bind_skips==skips)MDIE("argument binding probe did not skip repeated addresses");checked++;
        enc_r_draw(en,MTLPrimitiveTypeTriangle,0,3);enc_r_end(en);
        enc_b_copy_texture_buffer(blit(),texture,0,0,MTLOriginMake(0,0,0),MTLSizeMake(16,16,1),rb,0,64,1024);mtl_sub_commit();mtl_finish();
        for(unsigned i=0;i<1024;i++){int expected=scenario==2?0:(int[]){64,128,191,255}[i%4];if(abs((int)((uint8_t *)rb.contents)[i]-expected)>1)MDIE("vertex stream probe GPU mismatch mode%d scenario%d byte%u",mode,scenario,i);checked++;}
    }
    /* The two shader stages both use buffer(0), with distinct addresses. Change
       both inside one pass, then start a new encoder using unchanged addresses. */
    NSString *arg_source=@"#include <metal_stdlib>\nusing namespace metal;\nstruct V { float4 p [[position]]; float4 c; }; vertex V va(uint i [[vertex_id]], constant float4 &c [[buffer(0)]]) { float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return {float4(p[i],0,1),c}; } fragment float4 fa(V i [[stage_in]], constant float4 &c [[buffer(0)]]) { return float4(i.c.r,c.g,i.c.b,1); }\n";
    id<MTLLibrary> arg_lib=[g_dev newLibraryWithSource:arg_source options:nil error:&error];if(!arg_lib)MDIE("argument binding probe shader: %s",error.localizedDescription.UTF8String);
    MTLRenderPipelineDescriptor *arg_pd=[MTLRenderPipelineDescriptor new];arg_pd.vertexFunction=[arg_lib newFunctionWithName:@"va"];arg_pd.fragmentFunction=[arg_lib newFunctionWithName:@"fa"];arg_pd.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> arg_ps=[g_dev newRenderPipelineStateWithDescriptor:arg_pd error:&error];if(!arg_ps)MDIE("argument binding probe pipeline: %s",error.localizedDescription.UTF8String);
    id<MTLBuffer> arg_data=probe_buf(64);float arg_colors[16]={.25,0,.75,1,0,.5,0,1,.75,0,.25,1,0,.25,0,1};memcpy(arg_data.contents,arg_colors,sizeof arg_colors);
    g_arg_bind_cache_negative=0;
    for(int cache_mode=0;cache_mode<2;cache_mode++) {
        atomic_store(&g_arg_bind_cache_on,cache_mode);mtl_sub_begin();mtl_exec((const uint8_t *)&end);
        for(int pass=0;pass<2;pass++) {
            MTL4RenderPassDescriptor *rp=[MTL4RenderPassDescriptor new];rp.colorAttachments[0].texture=texture;rp.colorAttachments[0].loadAction=pass?MTLLoadActionLoad:MTLLoadActionClear;rp.colorAttachments[0].storeAction=MTLStoreActionStore;rp.renderTargetWidth=rp.renderTargetHeight=16;
            id<MTL4RenderCommandEncoder> en=enc_r_begin(rp);enc_r_pso(en,arg_ps);enc_r_cull(en,MTLCullModeNone);enc_r_viewport(en,(MTLViewport){0,0,16,16,0,1});
            for(int side=pass?1:0;side<2;side++) {
                enc_r_scissor(en,(MTLScissorRect){side?8:0,0,8,16});uint64_t sets=g_arg_bind_sets,skips=g_arg_bind_skips;
                for(int repeat=0;repeat<2;repeat++) {
                    enc_r_args(en,MTLRenderStageVertex,arg_data.gpuAddress+side*32,0);
                    enc_r_args(en,MTLRenderStageFragment,arg_data.gpuAddress+side*32+16,0);
                }
                if(g_arg_bind_sets-sets!=(cache_mode?2:4) || g_arg_bind_skips-skips!=(cache_mode?2:0))MDIE("argument binding probe stage/reset count mismatch");checked++;
                enc_r_draw(en,MTLPrimitiveTypeTriangle,0,3);
            }
            enc_r_end(en);
        }
        enc_b_copy_texture_buffer(blit(),texture,0,0,MTLOriginMake(0,0,0),MTLSizeMake(16,16,1),rb,0,64,1024);mtl_sub_commit();mtl_finish();
        for(unsigned i=0;i<1024;i++){int right=(i/4)%16>=8;int expected=right?(int[]){191,64,64,255}[i%4]:(int[]){64,128,191,255}[i%4];if(abs((int)((uint8_t *)rb.contents)[i]-expected)>1)MDIE("argument binding probe stage GPU mismatch mode%d byte%u",cache_mode,i);checked++;}
    }
    if (rs_enabled() && !g_rs_passes) MDIE("render stream probe never exercised the worker; use threshold 1");
    if (rs_enabled()) MLOG("render stream probe: %llu passes, %llu packets completed", (unsigned long long)g_rs_passes, (unsigned long long)g_rs_packets);
    g_vertex_streams_negative=0;atomic_store(&g_vertex_streams_on,saved);atomic_store(&g_arg_bind_cache_on,saved_arg);memset(&S.vbuf,0,sizeof S.vbuf);mtl_pool_del(mem);res_remove(texture);res_remove(rb);res_remove(arg_data);free(mem);return checked;
}

/* Exercise production state_op + encode_args on real GPU draws. Distinct shader
   objects alternate with unchanged bindings, then UBO/storage/registry/serial
   changes invalidate them. Separate submissions/passes and snapshot retirement
   must also invalidate. Negative 18 ignores binding generations, 19 snapshots. */
#include "nvn_args_invalidation_probe.h"

int mtl_probe_arg_epoch(void) {
    mtl_init(); int checked = probe_args_invalidation();
    const char *ne = getenv("TOTK_PROBE_NEGATIVE"); int negative = ne ? atoi(ne) : 0;
    const size_t size = 16384; void *mem = NULL;
    if (posix_memalign(&mem, size, size) || !mem) MDIE("argument epoch probe memory");
    memset(mem, 0, size); mtl_pool_add(mem, size);
    const float colors[8] = {.25,.5,.75,1, .75,.25,.5,1};
    memcpy(mem, colors, sizeof colors);
    NSError *error = nil;
    NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\nstruct A { constant float4 *c [[id(0)]]; constant uint *d [[id(1)]]; }; vertex float4 vs(uint i [[vertex_id]]) { float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[i],0,1); } fragment float4 fa(constant A &a [[buffer(0)]]) { return float4(a.c->rgb,a.d[324]==0x1234u?1.0:0.0); } fragment float4 fb(constant A &a [[buffer(0)]]) { return float4(a.c->rgb,a.d[324]==0x1234u?1.0:0.0); }\n";
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:source options:nil error:&error];
    if (!lib) MDIE("argument epoch probe shader: %s", error.localizedDescription.UTF8String);
    NvShader *shaders[2]; id<MTLRenderPipelineState> pipelines[2]; prog_t programs[2] = {0};
    for (int i = 0; i < 2; i++) {
        NvShader *sh = [NvShader new]; shaders[i] = sh; sh->fn = [lib newFunctionWithName:i ? @"fb" : @"fa"]; sh->stage = 1;
        snprintf(sh->key, sizeof sh->key, "probe-arg-epoch-%d", i);
        sh->rf.nb = 2; sh->rf.b[0] = (nvn_rbind){.kind=NVN_RB_CBUF,.aid=0,.reg=0};
        sh->rf.b[1] = (nvn_rbind){.kind=NVN_RB_DRV0,.aid=1}; sh->rf.drv_read[0] = 1;
        sh->ae = [sh->fn newArgumentEncoderWithBufferIndex:0]; sh->arg_len = sh->ae.encodedLength;
        if (sh->arg_len != 16) MDIE("argument epoch probe layout");
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new]; pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = sh->fn;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        pipelines[i] = [g_dev newRenderPipelineStateWithDescriptor:pd error:&error];
        if (!pipelines[i]) MDIE("argument epoch probe pipeline: %s", error.localizedDescription.UTF8String);
    }
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:32 height:8 mipmapped:NO];
    td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> texture = [g_dev newTextureWithDescriptor:td]; res_add(texture); id<MTLBuffer> rb = probe_buf(2048);
    op_hdr end = {OP_END,8};
    for (int mode = 0; mode < 3; mode++) {
        atomic_store(&g_arg_epoch_on, mode != 0); atomic_store(&g_arg_epoch_check, mode == 2);
        g_arg_epoch_negative = mode && negative == 18;
        for (int submission = 0; submission < 2; submission++) {
            mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
            for (int pass = 0; pass < 2; pass++) {
                g_pass.serial++; args_dirty_all();
                p_bind_buf u = {1,0,(uint64_t)(uintptr_t)mem,16}, storage = {1,0,0x1234,16};
                state_op(OP_BIND_UBO, &u); state_op(OP_BIND_SSBO, &storage);
                MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new]; rp.colorAttachments[0].texture = texture;
                rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                rp.renderTargetWidth = 32; rp.renderTargetHeight = 8;
                id<MTL4RenderCommandEncoder> en = enc_r_begin(rp); enc_r_cull(en,MTLCullModeNone); enc_r_viewport(en,(MTLViewport){0,0,32,8,0,1});
                for (int stripe = 0; stripe < 16; stripe++) {
                    int shader = stripe == 1 || stripe == 4 ? 1 : 0;
                    p_bind_program program = {&programs[shader], 0x1f}; state_op(OP_BIND_PROGRAM, &program);
                    if (stripe == 3) { u.addr += 16; state_op(OP_BIND_UBO, &u); }
                    if (stripe == 6) args_dirty_all();
                    if (stripe == 7) g_reg_gen++;
                    if (stripe == 8) ser_bump(0);
                    if (stripe == 9 || stripe == 10) { storage.addr = stripe == 9 ? 0x5678 : 0x1234; state_op(OP_BIND_SSBO, &storage); }
                    if (stripe == 11) state_op(OP_BIND_SSBO, &storage); // identical binding must keep the epoch
                    uint64_t encodes = g_stat_args_enc, extra = g_arg_epoch_extra, hits = g_arg_epoch_hits;
                    enc_r_pso(en, pipelines[shader]); enc_r_scissor(en,(MTLScissorRect){stripe*2,0,2,8});
                    encode_args((id<MTLCommandEncoder>)en, shaders[shader], 1, 0); enc_r_draw(en,MTLPrimitiveTypeTriangle,0,3);
                    if (!negative && mode && (stripe == 2 || stripe == 5) && g_arg_epoch_extra != extra+1) MDIE("argument epoch probe did not reuse across programs");
                    if (!negative && mode && stripe >= 3 && stripe <= 10 && stripe != 5 && g_stat_args_enc != encodes+1) MDIE("argument epoch probe failed to rebuild stripe %d", stripe);
                    if (!negative && mode && stripe == 11 && g_arg_epoch_hits != hits+1) MDIE("argument epoch probe rebuilt identical binding");
                    checked++;
                }
                enc_r_end(en);
                enc_b_copy_texture_buffer(blit(),texture,0,0,MTLOriginMake(0,0,0),MTLSizeMake(32,8,1),rb,pass*1024,128,1024);
            }
            mtl_sub_commit(); mtl_finish();
            for (int byte = 0; byte < 2048; byte++) {
                int stripe = (byte/4)%32/2, channel = byte%4;
                int want = stripe < 3 ? (int[]){64,128,191,255}[channel] : (int[]){191,64,128,255}[channel];
                if (stripe == 9 && channel == 3) want = 0;
                if (abs((int)((uint8_t *)rb.contents)[byte]-want)>1) MDIE("argument epoch probe GPU mismatch mode %d stripe %d byte %d",mode,stripe,byte);
                checked++;
            }
        }
        // A real captured uniform is invalidated while a cached shader block still exists.
        memcpy(mem,colors,16); mtl_sub_begin(); mtl_exec((const uint8_t *)&end);
        p_bind_buf u = {1,0,(uint64_t)(uintptr_t)mem,16}, storage = {1,0,0x1234,16};
        uint8_t commands[128]; uint8_t *w = lp_rec(commands,OP_BIND_UBO,&u,sizeof u); lp_end(w);
        uint64_t handle = (uint64_t)(uintptr_t)commands; mtl_sub_snapshot_ubos(&handle,1);
        if (!usnap_find(u.addr,16)) MDIE("argument epoch probe missing snapshot");
        memcpy(mem,colors+4,16); state_op(OP_BIND_UBO,&u); state_op(OP_BIND_SSBO,&storage); args_dirty_all(); g_pass.serial++;
        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new]; rp.colorAttachments[0].texture = texture;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.renderTargetWidth = 32; rp.renderTargetHeight = 8;
        id<MTL4RenderCommandEncoder> en = enc_r_begin(rp); enc_r_cull(en,MTLCullModeNone); enc_r_viewport(en,(MTLViewport){0,0,32,8,0,1});
        for (int stripe = 0; stripe < 4; stripe++) {
            int shader = stripe == 1 ? 1 : 0;
            p_bind_program program = {&programs[shader],0x1f}; state_op(OP_BIND_PROGRAM,&program);
            if (stripe == 2) usnap_drop(u.addr,16);
            if (mode && negative == 19 && stripe == 2) shaders[0]->ab_epoch.snapshot_drops = g_usnap_drops;
            enc_r_pso(en,pipelines[shader]); enc_r_scissor(en,(MTLScissorRect){stripe*8,0,8,8});
            encode_args((id<MTLCommandEncoder>)en,shaders[shader],1,0); enc_r_draw(en,MTLPrimitiveTypeTriangle,0,3);
        }
        enc_r_end(en); enc_b_copy_texture_buffer(blit(),texture,0,0,MTLOriginMake(0,0,0),MTLSizeMake(32,8,1),rb,0,128,1024); mtl_sub_commit(); mtl_finish();
        for (int byte = 0; byte < 1024; byte++) {
            int stripe = (byte/4)%32/8, channel = byte%4;
            int want = stripe < 2 ? (int[]){64,128,191,255}[channel] : (int[]){191,64,128,255}[channel];
            if (abs((int)((uint8_t *)rb.contents)[byte]-want)>1) MDIE("argument epoch snapshot GPU mismatch mode %d stripe %d byte %d",mode,stripe,byte);
            checked++;
        }
        memcpy(mem,colors,sizeof colors);
    }
    if (!g_arg_epoch_extra || !g_arg_epoch_verified) MDIE("argument epoch probe missed reuse/check paths");
    NvShader *unsafe = [NvShader new]; unsafe->rf.nb=1; unsafe->rf.b[0].kind=NVN_RB_SBUFU;
    if (arg_epoch_safe(unsafe)) MDIE("argument epoch accepted indirect storage"); checked++;
    unsafe=[NvShader new]; unsafe->rf.nb=1; unsafe->rf.b[0].kind=NVN_RB_SBUFP;unsafe->rf.b[0].rw=1;
    if (arg_epoch_safe(unsafe)) MDIE("argument epoch accepted writable storage"); checked++;
    MLOG("argument epoch probe: %llu extra reuses, %llu shadow comparisons",(unsigned long long)g_arg_epoch_extra,(unsigned long long)g_arg_epoch_verified);
    g_arg_epoch_negative=0; atomic_store(&g_arg_epoch_on,0);atomic_store(&g_arg_epoch_check,0);
    memset(&S,0,sizeof S);args_dirty_all();mtl_pool_del(mem);res_remove(texture);res_remove(rb);free(mem);return checked;
}

#include "nvn_tex_slots_probe.h"

#include "nvn_pool_retire_probe.h"
#include "nvn_cube_group_probe.h"
