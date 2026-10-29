/* Read verified external fonts through the same host API used by the game. */
#import <Foundation/Foundation.h>
#import <CoreText/CoreText.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
void *runtime_font_override(const char *);
_Noreturn void runtime_fail(const char *message){fprintf(stderr,"%s\n",message);abort();}
int main(void){@autoreleasepool{
 const void *(*address)(int)=runtime_font_override("_ZN2nn2pl20GetSharedFontAddressENS0_14SharedFontTypeE");
 size_t(*size)(int)=runtime_font_override("_ZN2nn2pl17GetSharedFontSizeENS0_14SharedFontTypeE");
 /* Address must work as the first call, before any load-state query. */
 for(int type=0;type<6;type++){
  const uint8_t *bytes=address(type);size_t length=size(type);
  assert(bytes&&length>12);
  assert((bytes[0]==0&&bytes[1]==1&&bytes[2]==0&&bytes[3]==0)||!memcmp(bytes,"OTTO",4));
  CGDataProviderRef provider=CGDataProviderCreateWithData(NULL,bytes,length,NULL);
  CGFontRef graphics=CGFontCreateWithDataProvider(provider);assert(graphics);
  CTFontRef font=CTFontCreateWithGraphicsFont(graphics,32,NULL,NULL);assert(font);
  if(type==5){
   unsigned icons=0;for(UniChar code=0xe000;code<0xf000;code++){
    CGGlyph glyph=0;if(CTFontGetGlyphsForCharacters(font,&code,&glyph,1)&&glyph)icons++;
   }
   assert(icons>32);printf("Nintendo extension: %u private-use symbols available\n",icons);
  }
  CFRelease(font);CGFontRelease(graphics);CGDataProviderRelease(provider);
 }
 puts("PASS: all six decoded shared fonts load through CoreText");
}return 0;}
