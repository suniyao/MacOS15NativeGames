"""Run the pinned game's pool-allocation prefix against a mock heap in Unicorn.

Usage: python test_tomodachi_pool_alignment.py /external/rewritten/main/main.patched.bin
Requires a user-prepared module and Unicorn; no game bytes are stored here.
"""
import struct
import sys
from pathlib import Path
from unicorn import Uc, UC_ARCH_ARM64, UC_MODE_ARM, UC_HOOK_CODE
from unicorn.arm64_const import *

main = Path(sys.argv[1]).read_bytes()
start, end, site = 0x32A580, 0x32A5E0, 0x32A5B0
assert struct.unpack_from("<I", main, site)[0] == 0x2A0103E2

def allocate(alignment, flags, patched):
    code = bytearray(main[start:end])
    cave = 0x25CE6A0
    assert main[cave:cave+20] == bytes(20)
    if patched:
        struct.pack_into("<I", code, site-start, 0x14000000 | ((cave-site)//4))
    u = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
    u.mem_map(start & ~4095, 4096)
    u.mem_write(start, bytes(code))
    u.mem_map(cave & ~4095, 4096)
    stub = [0x2A0103E2, 0xB94000A9, 0x36400049, 0x531E7422,
            0x14000000 | (((site+4-cave-16)//4) & 0x3FFFFFF)]
    u.mem_write(cave, struct.pack("<5I", *stub))
    u.mem_map(0x800000, 0x10000)
    u.mem_write(0x803000,struct.pack("<I",flags))
    u.mem_map(0x900000, 4096)
    # x7 -> pool owner -> heap -> vtable -> allocate(size, signed alignment).
    for address,value in [(0x800008,0x801000),(0x801000,0x802000),(0x802030,0x900000)]:
        u.mem_write(address,struct.pack("<Q",value))
    u.reg_write(UC_ARM64_REG_SP,0x80F000)
    for reg,value in [(UC_ARM64_REG_X0,0x8000000),(UC_ARM64_REG_X1,alignment&0xffffffff),
                      (UC_ARM64_REG_X2,0),(UC_ARM64_REG_X4,0x100000),
                      (UC_ARM64_REG_X5,0x803000),(UC_ARM64_REG_X7,0x800000)]:
        u.reg_write(reg,value)
    seen=[]
    def hook(uc,pc,size,data):
        if pc==0x900000:
            seen.append((uc.reg_read(UC_ARM64_REG_X0),uc.reg_read(UC_ARM64_REG_X1),
                         uc.reg_read(UC_ARM64_REG_W2)))
            uc.emu_stop()
    u.hook_add(UC_HOOK_CODE,hook)
    u.emu_start(start,0,count=100)
    assert len(seen)==1
    return seen[0]

for flags in [0x18C, 0x104, 0x21, 0x62, 0x88]:
    for alignment in [4096,8192,16384,65536,1048576,-4096,-65536]:
        before=allocate(alignment,flags,False)
        after=allocate(alignment,flags,True)
        assert after[:2]==before[:2]==(0x801000,0x8000000)
        value=after[2] if after[2]<0x80000000 else after[2]-0x100000000
        assert value==alignment*(4 if flags & 0x100 else 1)
        if flags & 0x100:
            assert abs(value)%16384==0
print("PASS: 35 actual allocation cases preserve heap/size and ordinary alignment; physical pools use host pages")
