#pragma once
#import <Metal/Metal.h>
#include <stdio.h>
/* Metal blits cannot filter. Render a single selected source mip into the
 * destination mip, preserving alpha and the requested source/destination rects.
 * Caller ends earlier encoders and invalidates its cached render state. */
static int nvn_scaled_copy_encode(id<MTLDevice> device, id<MTLCommandBuffer> cb,
 id<MTLTexture> src, id<MTLTexture> dst, unsigned sl, unsigned ss, unsigned dl, unsigned ds,
 unsigned sx,unsigned sy,unsigned sw,unsigned sh,unsigned dx,unsigned dy,unsigned dw,unsigned dh,int linear) {
 if(!sw||!sh||!dw||!dh||sl>=src.mipmapLevelCount||dl>=dst.mipmapLevelCount)return 0;
 if(!(dst.usage&MTLTextureUsageRenderTarget)||src.pixelFormat!=dst.pixelFormat)return 0;
 if(src.pixelFormat!=MTLPixelFormatRGBA8Unorm && src.pixelFormat!=MTLPixelFormatRGBA8Unorm_sRGB)return 0;
 if(sx+sw>MAX(src.width>>sl,1u)||sy+sh>MAX(src.height>>sl,1u)||dx+dw>MAX(dst.width>>dl,1u)||dy+dh>MAX(dst.height>>dl,1u))return 0;
 static id<MTLLibrary> lib;static NSMutableDictionary *pipelines;
 if(!lib){
  NSString *msl=@"#include <metal_stdlib>\nusing namespace metal;\n"
   "struct V { float4 p [[position]]; float2 uv; };\n"
   "vertex V vs(uint i [[vertex_id]]) { float2 u=float2((i<<1)&2,i&2); return {float4(u*float2(2,-2)+float2(-1,1),0,1),u}; }\n"
   "fragment float4 fs(V v [[stage_in]],texture2d<float> t [[texture(0)]],sampler s [[sampler(0)]],constant float4& r [[buffer(0)]]) { return t.sample(s,r.xy+v.uv*r.zw,level(0)); }\n";
  NSError *err=nil;lib=[device newLibraryWithSource:msl options:nil error:&err];
  if(!lib){fprintf(stderr,"scaled copy shader: %s\n",err.localizedDescription.UTF8String);return 0;}
  pipelines=[NSMutableDictionary dictionary];
 }
 NSNumber *key=@(dst.pixelFormat);id<MTLRenderPipelineState> ps=pipelines[key];
 if(!ps){MTLRenderPipelineDescriptor *pd=[MTLRenderPipelineDescriptor new];pd.vertexFunction=[lib newFunctionWithName:@"vs"];pd.fragmentFunction=[lib newFunctionWithName:@"fs"];pd.colorAttachments[0].pixelFormat=dst.pixelFormat;ps=[device newRenderPipelineStateWithDescriptor:pd error:nil];if(!ps)return 0;pipelines[key]=ps;}
 id<MTLTexture> view=[src newTextureViewWithPixelFormat:src.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(sl,1) slices:NSMakeRange(ss,1)];
 if(!view)return 0;
 MTLSamplerDescriptor *sd=[MTLSamplerDescriptor new];sd.minFilter=sd.magFilter=linear?MTLSamplerMinMagFilterLinear:MTLSamplerMinMagFilterNearest;sd.sAddressMode=sd.tAddressMode=MTLSamplerAddressModeClampToEdge;
 id<MTLSamplerState> sampler=[device newSamplerStateWithDescriptor:sd];
 MTLRenderPassDescriptor *rp=[MTLRenderPassDescriptor renderPassDescriptor];rp.colorAttachments[0].texture=dst;rp.colorAttachments[0].level=dl;rp.colorAttachments[0].slice=ds;rp.colorAttachments[0].loadAction=MTLLoadActionLoad;rp.colorAttachments[0].storeAction=MTLStoreActionStore;
 id<MTLRenderCommandEncoder> en=[cb renderCommandEncoderWithDescriptor:rp];
 if(!en)return 0;
 float rect[4]={(float)sx/view.width,(float)sy/view.height,(float)sw/view.width,(float)sh/view.height};
 [en setRenderPipelineState:ps];[en setViewport:(MTLViewport){dx,dy,dw,dh,0,1}];[en setCullMode:MTLCullModeNone];
 [en setFragmentTexture:view atIndex:0];[en setFragmentSamplerState:sampler atIndex:0];[en setFragmentBytes:rect length:sizeof rect atIndex:0];[en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];[en endEncoding];
 return 1;
}
