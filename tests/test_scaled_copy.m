#import <Foundation/Foundation.h>
#include <assert.h>
#include <stdio.h>
#include "nvn_scaled_copy.h"
int main(void){@autoreleasepool{
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();assert(dev);
 MTLTextureDescriptor *td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:48 height:96 mipmapped:YES];td.storageMode=MTLStorageModeShared;td.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
 id<MTLTexture> t=[dev newTextureWithDescriptor:td];assert(t);
 unsigned char pixels[48*96*4];
 for(unsigned y=0;y<96;y++)for(unsigned x=0;x<48;x++){unsigned p=(y*48+x)*4;pixels[p]=x%2?200:0;pixels[p+1]=y;pixels[p+2]=80;pixels[p+3]=x%2?240:0;}
 [t replaceRegion:MTLRegionMake2D(0,0,48,96) mipmapLevel:0 withBytes:pixels bytesPerRow:48*4];
 id<MTLCommandQueue> q=[dev newCommandQueue];id<MTLCommandBuffer> cb=[q commandBuffer];
 assert(nvn_scaled_copy_encode(dev,cb,t,t,0,0,1,0,0,0,48,96,0,0,24,48,1));
 assert(nvn_scaled_copy_encode(dev,cb,t,t,1,0,2,0,0,0,24,48,0,0,12,24,1));
 [cb commit];[cb waitUntilCompleted];assert(cb.status==MTLCommandBufferStatusCompleted);
 unsigned char result[24*48*4];[t getBytes:result bytesPerRow:24*4 fromRegion:MTLRegionMake2D(0,0,24,48) mipmapLevel:1];
 for(unsigned y=0;y<48;y++)for(unsigned x=0;x<24;x++){unsigned p=(y*24+x)*4;assert(abs(result[p]-100)<=1);assert(abs(result[p+1]-(2*(int)y+1))<=1);assert(result[p+2]==80);assert(abs(result[p+3]-120)<=1);}
 unsigned char tail[12*24*4];[t getBytes:tail bytesPerRow:12*4 fromRegion:MTLRegionMake2D(0,0,12,24) mipmapLevel:2];
 for(unsigned i=0;i<12*24;i++){assert(abs(tail[i*4]-100)<=1);assert(abs(tail[i*4+3]-120)<=1);}
 puts("PASS: production scaled copy generates two filtered mips, preserving color, orientation and alpha");
}return 0;}
