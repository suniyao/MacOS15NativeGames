#include "nvn_mip_range.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
 unsigned first,count;
 assert(nvn_physical_mips(48,96,1,15)==7);
 assert(nvn_physical_mips(128,128,1,15)==8);
 assert(nvn_physical_mips(384,384,1,15)==9);
 assert(nvn_mip_view_range(8,8,0,15,&first,&count) && first==0 && count==8);
 assert(nvn_mip_view_range(7,7,2,15,&first,&count) && first==2 && count==5);
 assert(nvn_mip_view_range(10,9,2,4,&first,&count) && first==1 && count==4);
 assert(nvn_mip_view_range(10,9,0,10,&first,&count) && first==0 && count==9);
 assert(!nvn_mip_view_range(8,8,8,1,&first,&count));
 assert(!nvn_mip_view_range(8,0,0,1,&first,&count));
 puts("PASS: fixed mip tails preserve top levels; real top-mip drops shift views");
}
