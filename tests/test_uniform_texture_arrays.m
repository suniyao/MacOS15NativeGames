/* Verify translated reflection and the Metal 3 resource-ID array layout. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nvn_refl.h"
#include <assert.h>
#include <stdio.h>

static NSDictionary *binding(NSString *name, NSString *kind, NSString *type, int aid) {
    return @{ @"msl_name":name,@"kind":kind,@"msl_type":type,@"arg_id":@(aid),@"reg":[NSNull null] };
}
int main(void) { @autoreleasepool {
    NSDictionary *texture=binding(@"fp_t2d_cb3_0",@"tex",@"array<texture2d<float>, 8192>",5);
    NSDictionary *sampler=binding(@"fp_t2d_cb3_0Smplr",@"samp_tex",@"array<sampler, 8192>",8197);
    nvn_refl rf; char error[256];
    assert(!nvn_refl_from_json(@{@"route":@"B",@"bindings":@[texture,sampler]},&rf,error,sizeof error));
    assert(rf.nb==2 && rf.b[0].kind==NVN_RB_TEXU && rf.b[1].kind==NVN_RB_SAMPU);
    assert(rf.b[0].reg==0 && rf.b[0].off==0 && rf.b[0].count==8192 && rf.b[1].aid==8197);
    assert(nvn_refl_from_json(@{@"route":@"B",@"bindings":@[texture,texture]},&rf,error,sizeof error));
    assert(nvn_refl_from_json(@{@"route":@"B",@"bindings":@[binding(@"fp_t2d_cb3_0",@"tex",@"array<texture2d<float>, 16512>",5)]},&rf,error,sizeof error));
    assert(nvn_refl_from_json(@{@"route":@"B",@"bindings":@[binding(@"fp_t2d_cb2_0",@"tex",@"texture2d<float>",5)]},&rf,error,sizeof error));

    id<MTLDevice> device=MTLCreateSystemDefaultDevice(); assert(device);
    NSString *source=@"#include <metal_stdlib>\nusing namespace metal;\n"
      "struct Args { device float4 *out [[id(0)]]; array<device float*,4> padding [[id(1)]]; array<texture2d<float>,8192> tex [[id(5)]]; array<sampler,8192> smp [[id(8197)]]; };\n"
      "kernel void test(constant Args& a [[buffer(0)]],uint i [[thread_position_in_grid]]) { a.out[i]=a.tex[i].sample(a.smp[i],float2(.5),level(0))+a.padding[i%4][0]; }";
    NSError *err=nil; id<MTLLibrary> lib=[device newLibraryWithSource:source options:nil error:&err];
    if (!lib) fprintf(stderr,"%s\n",err.localizedDescription.UTF8String); assert(lib);
    id<MTLFunction> fn=[lib newFunctionWithName:@"test"];
    id<MTLComputePipelineState> pipe=[device newComputePipelineStateWithFunction:fn error:&err]; assert(pipe);
    id<MTLArgumentEncoder> encoder=[fn newArgumentEncoderWithBufferIndex:0]; fprintf(stderr,"Metal array argument length: %lu\n",(unsigned long)encoder.encodedLength); assert(encoder.encodedLength==16389*8);
    id<MTLBuffer> args=[device newBufferWithLength:encoder.encodedLength options:MTLResourceStorageModeShared];
    id<MTLBuffer> out=[device newBufferWithLength:8192*16 options:MTLResourceStorageModeShared];
    uint64_t *slots=args.contents; slots[0]=out.gpuAddress;
    id<MTLBuffer> zero=[device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    memset(zero.contents,0,16);for(int i=1;i<=4;i++)slots[i]=zero.gpuAddress;
    MTLSamplerDescriptor *sd=[MTLSamplerDescriptor new];sd.supportArgumentBuffers=YES;
    id<MTLSamplerState> smp=[device newSamplerStateWithDescriptor:sd];
    NSMutableArray *textures=[NSMutableArray array];
    for (unsigned i=0;i<8192;i++) {
        MTLTextureDescriptor *td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:1 height:1 mipmapped:NO];
        td.storageMode=MTLStorageModeShared;td.usage=MTLTextureUsageShaderRead;
        id<MTLTexture> tex=[device newTextureWithDescriptor:td];assert(tex);
        float value=i+.25f;[tex replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&value bytesPerRow:4];
        [textures addObject:tex];slots[5+i]=tex.gpuResourceID._impl;slots[8197+i]=smp.gpuResourceID._impl;
    }
    id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];[ce setComputePipelineState:pipe];[ce setBuffer:args offset:0 atIndex:0];
    [ce useResource:out usage:MTLResourceUsageWrite];[ce useResource:zero usage:MTLResourceUsageRead];
    for(id<MTLTexture> tex in textures)[ce useResource:tex usage:MTLResourceUsageRead];
    [ce dispatchThreads:MTLSizeMake(8192,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[ce endEncoding];[cb commit];[cb waitUntilCompleted];
    assert(cb.status==MTLCommandBufferStatusCompleted);
    for(unsigned i=0;i<8192;i++)assert(((float *)out.contents)[4*i]==i+.25f);
    puts("PASS: reflection rejects invalid ranges; all 8192 Metal texture/sampler array entries match");
} return 0; }
