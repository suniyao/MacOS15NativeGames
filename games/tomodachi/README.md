# Tomodachi Life: Living the Dream — native port development

Experimental native ARM64 host for the **1.0.0** base game, title
`010051F0207B2000`, on Apple silicon and macOS 15. This is not yet a
verified playable port. No game data, keys, firmware or saves are included.

The host shares the `m3-macos15` branch's Metal 3 renderer and system-service
implementations, with a separate loader and title identity. Zelda game patches,
world inspection and automatic save/menu actions are excluded. Runtime service
coverage, graphics correctness and save compatibility require actual validation.

## Development commands

Use Python with the dependencies in `../totk/requirements.txt`, Xcode command-line
tools, and the existing Opus and libvpx libraries. All generated output must be
outside the checkout. Use your own game and system files.

```sh
python games/tomodachi/prepare.py /path/to/tomodachi.nsp \
  --keys /path/to/prod.keys --work /path/to/Tomodachi_build/prepared --romfs

python games/tomodachi/build_host.py \
  --prepared /path/to/Tomodachi_build/prepared \
  --work /path/to/Tomodachi_build/host \
  --shared-fonts /path/to/verified/shared-fonts \
  --spirv-cross /path/to/verified/spirv-cross

python games/tomodachi/run.py --work /path/to/Tomodachi_build/host --mode probe
python games/tomodachi/run.py --work /path/to/Tomodachi_build/host --mode startup
python games/tomodachi/run.py --work /path/to/Tomodachi_build/host --mode boot --seconds 120
```

Preparation verifies the title/version, pinned executable hashes, NSO segment
hashes, instruction rewriting, relocation tables and optional RomFS integrity.
`--resume` revalidates and reuses existing intermediate results. An unknown game
build fails rather than receiving patches for a different executable.

After a source-only change, rebuild with
`make -C games/tomodachi WORK=/path/to/Tomodachi_build/host`.

Each diagnostic uses a fresh profile under the host work directory and records
its output in `tests/`. It neither imports nor modifies Ryujinx saves. A successful
CPU probe or constructor checkpoint is not a gameplay test. The diagnostic time
limit deliberately terminates the process; these profiles are disposable and
must not be used for progress you want to keep.

## Provenance

Base: `suniyao/MacNativeGames`, branch `m3-macos15`, commit
`f71ea7b56eb1f7064d5e753428cc6b05a124241b`.

Preparation tools and host-table generation adapted from upstream
`ly05010419/MacNativeGames` commit `ff63a162b35d8b486b941191785d8e957c1f5df2`,
`games/acnh/`. Earlier tool provenance and the MIT license are retained in
`tools/UPSTREAM.json` and `tools/LICENSE`. The rewriter remains shared with TOTK;
Tomodachi's signed-table metadata is keyed to its exact executable build ID.

## Current validation and differences

On an M4 running macOS 15.6, the native CPU probe, all main/SDK constructors,
Metal depth-clear probe and first date/time setup screen have been verified.
The first setup screen renders at approximately 60 frames per second after
shader compilation. This does **not** establish complete gameplay or save/load
compatibility. Subsequent screens are still under active validation.

The title-specific host supplies mouse gestures/touch, local Mii defaults,
asynchronous macOS text entry, the host clock/time zone, application identity and
NACP-derived save limits. The local name filter deliberately uses an **empty
profanity dictionary**; Nintendo's filtering dictionary is not reproduced.
Several audio effects remain unsupported, although audio output is active.

A guarded executable patch increases the MiiBodyBaseModel child heap from
128 KiB to 256 KiB: original instruction `0x321b2fe0` at main+`0x580434` becomes
`0x321b33e0`. The smaller heap exhausted during native Mii model creation.
The copied-save test also exhausted a 32 KiB `ClothModel` heap on a
3616-byte allocation. The two 32 KiB clothing pools at main+`0x57b9fc` and
main+`0x57ba70` now receive 64 KiB (`0x52900003` → `0x52a00023`).
These patches check the original instructions and only run with the pinned game
build. `TOMODACHI_MII_HEAP_PATCH=0` disables them for diagnosis.

The shared renderer now resolves texture/sampler arrays from user uniform
buffers. Their eight-byte handle stride follows the vendored Ryujinx
`BindlessToArray.cs` index conversion. Reflection keeps one record per MSL
member, validates array ranges/overlap, and disables argument reuse for these
indirect bindings. `tests/test_uniform_texture_arrays.m` verifies all 8,192 entries
using a real Metal 3 compute shader and checks rejection of malformed ranges.

## Local Desktop app

```sh
python games/tomodachi/package.py --work /path/to/Tomodachi_build/host \
  --output '/path/to/MacNativeGames/Tomodachi Life.app'
```

The app uses the external host and RomFS folders through relative paths. Keep
those folders together when moving the Desktop collection. Required fonts and
shader tools are stored in the host folder. A source rebuild needs repackaging;
`--replace` preserves the previous app in the host's test archive.

The app's separate persistent profile is at
`~/Library/Application Support/MacNativeGames/Tomodachi/profile`.
Keyboard buttons: J = A, K/Space = B, I = X, U = Y, Return = Plus,
Tab/Escape = Minus. A connected controller uses the shared native input backend.
D-pad: F/T/H/G = left/up/right/down. Mouse/trackpad touch support is experimental. Key 4 toggles diagnostic overlays.
Ryujinx save data is not imported automatically. `--language zh-Hant` preserves
a Traditional Chinese game setting when packaging; replacements retain the
previous language unless explicitly overridden.

Shared-font payload decoding is backported from upstream `games/acnh/runtime/shared_font.c`
at the provenance commit above. File hashes are checked before in-memory BFTTF
XOR decoding; the decoded sfnt header is then validated. Returning encrypted font
payloads caused the missing Nintendo HOME/button symbols (black question-mark boxes).
`tests/test_shared_fonts.m` calls the host font API before any load-state request,
loads all six fonts through CoreText and checks that the Nintendo extension has
private-use symbols. Fonts remain external, supplied by the user's firmware.

The compressed-render-target probe additionally uses the production `blk_aux`
path for a 256x256 BC4 texture at mip levels 7 and 8. Auxiliary block surfaces
are allocated per mip, so the 1-block tail levels remain valid Metal textures.
The M4 readback checks pass for these tails and the existing 2D/3D/BC3 cases.

Text entry uses an asynchronous native field attached to the game window. It
does not activate the app, raise the window, or restore a minimized window.
The field receives first-responder status only while the game window is key;
returning to that window restores the existing field. A title-requested
`Disappear` finishes the display lifecycle without sending a user-cancel
callback. Guest `SetInputText` updates are not echoed as new user edits, and
duplicate `Appear` calls are ignored while a field or its completion is pending.
Cancellation is scoped to the requesting keyboard object so an older worker
cannot dismiss a newer field. The keyboard unit test covers Unicode, callback
threading, echoed text, duplicate appearance and repeated programmatic hide.
Actual typing, input-method composition and app switching require an interactive
check.

The renderer clamps fixed maximum mip counts to the usable physical chain when
creating sampling views, without treating a truncated 1x1 tail as a dropped top
mip. Metal 3 filtered RGBA8 texture copies now use a render pass to generate the
requested smaller mip, preserving alpha. `tests/test_mip_range.c` and
`tests/test_scaled_copy.m` cover the range mapping and GPU filtering. Resident
portraits were observed with facial features after these changes; the user
confirmed faces remain correct while zooming in and out. The subsequent full handle-table fix corrected the flat terrain and mosaic
edges in local visual checks. Some tree shadows improved, but the user
subsequently reported intermittent rectangular Mii shadows indoors as well.
Green horizontal lines and rectangular shadows remain under investigation.

Indirect uniform texture handles participate in draw-texture preflight and
same-target render-pass dependency checks, using the same submit-time bytes as
argument encoding. A direct-slot-only texture plan is bypassed for such shaders.
Metal 3 replay ends encoders for explicit NVN primitive/fragment/indirect ordering
and texture/shader invalidation barriers; automatic hazard tracking alone does
not establish an edge inside a continued encoder. The storage-view GPU probe
covers indirect handle arrays (both vertex and fragment), all mip/layer pixel
readbacks, and explicit barrier preservation. Its external logs are under the
host work directory's `tests/probe-gpu-storage-view-*` directories. The earlier barrier-only build did not correct the visible island artifacts.

## Full uniform texture tables (2026-10-29)

The native translator's `QueryTextureArrayLengthFromBuffer` returned 32. This
silently clamped the island's material texture lookups to entry 31: the generated
`bb62dd4d` shader's first dynamic lookup uses a byte offset of at least 272
(index 34), and the bound user buffer is 41,216 bytes. The translator now covers
all 8,192 eight-byte handles in a maximum-size 64 KiB user buffer. Reflection and
argument encoding allow both full texture and sampler arrays, plus ordinary
bindings. Cached 32-entry resource-array shaders regenerate automatically; other
cached shaders remain usable. The formal NativeAOT translator and its SHA-256
were rebuilt from the checked-in source with Microsoft's verified .NET 10 SDK.
The external build is under `Tomodachi_build/tooling/translator-full-handles`.

The Metal 3 GPU test verifies all 8,192 entries, including high indices that the
old 32-entry test could not cover. Local game inspection with normal GPU texture
uploads and default cross-format sharing showed detailed grass, continuous path
edges, shaped tree shadows, and resident faces in indoor scenes. Green lines
still occur at regularly spaced rows in the captured game image; they are not
fixed by this change. CPU texture upload and cross-format sharing comparisons
did not correct the earlier terrain defects. Captures and previous translator /
shader backups are external, under the host work directory's `tests`, including
`full-bindless-table-*`, `physical-pool-only-*`, `compressed-alias-*`, and
`translator-before-full-handles`.

A separate pinned instruction patch at main+`0x32a5b0` strengthens alignment only
for PHYSICAL pool allocations. It branches through verified zero padding after
the main text segment, calls the title's original heap allocator, and preserves
signed alignment. This fixes the rejected 128 MiB CPU-visible physical pool on
16 KiB Darwin pages. Increasing alignment for every pool exhausted a small
Mii-resource heap; that broad experiment was replaced by the physical-only
patch. `tests/test_tomodachi_pool_alignment.py` exercises 35 actual allocation
prefix cases with physical and ordinary flags. `TOMODACHI_POOL_ALIGNMENT_PATCH=0`
disables the patch for diagnosis.

The sparse-table encoder avoids repeated Metal resource declarations for
consecutive identical/dummy textures and does not declare samplers as resources.
An additional validation-only run exposed repeated attachment bindings to the
same 8x8 placeholder texture (`attachment-validation-*`); the relation to the
remaining green lines has not been established. Normal app runs do not enable
the temporary Metal validation environment used to identify that issue.

Finder/LaunchServices relaunch testing also stalled in filesystem reads before
the game window appeared. Samples are at `/tmp/tomodachi-normal-startup.txt` and
`/tmp/tomodachi-normal-startup2.txt`; direct process launches from the development
runner work. A proposed diagnostic-path-only workaround did not resolve the
underlying file access stall and was removed. Finder relaunch remains a separate
unresolved check; do not report it as fixed.
