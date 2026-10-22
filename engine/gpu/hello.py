"""Smoke tests: (1) one-instruction-per-lane kernel; (2) QPU ids of an 8-batch dispatch:
each batch stores its tidx into slot (tidx >> 2) & 15 of a 16-slot buffer."""
import numpy as np
from videocore6.assembler import *  # noqa: F403
from videocore6.driver import Driver


@qpu
def k_add1(asm):
    eidx(r0, sig=ldunif)          # r5 = address of X
    shl(r0, r0, 2)
    add(r1, r5, r0)               # per-lane address
    mov(tmua, r1, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r2))
    add(r2, r2, 1)
    mov(tmud, r2)
    mov(tmua, r1)
    tmuwt()
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def k_ids(asm):
    tidx(r0, sig=ldunif)          # r5 = base of the 16 x 16 slot buffer
    shr(r1, r0, 2)
    band(r1, r1, 15)              # candidate qpu id
    shl(r1, r1, 6)                # slot * 64 bytes
    eidx(r2)
    shl(r2, r2, 2)
    add(r1, r1, r2)
    add(r1, r1, r5)
    mov(tmud, r0)
    mov(tmua, r1)
    tmuwt()
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


with Driver() as drv:
    code = drv.program(k_add1)
    x = drv.alloc(16, dtype=np.uint32)
    x[:] = np.arange(16)
    u = drv.alloc(1, dtype=np.uint32)
    u[0] = x.addresses()[0]
    drv.execute(code, local_invocation=(16, 1, 1), uniforms=u.addresses()[0], thread=1)
    print("add1:", "OK" if (x == np.arange(16) + 1).all() else f"FAIL {x.tolist()}")

    code2 = drv.program(k_ids)
    slots = drv.alloc((16, 16), dtype=np.uint32)
    slots[:] = 0xFFFFFFFF
    u2 = drv.alloc(1, dtype=np.uint32)
    u2[0] = slots.addresses()[0, 0]
    drv.execute(code2, local_invocation=(16, 1, 1), uniforms=u2.addresses()[0], thread=8)
    used = [i for i in range(16) if slots[i, 0] != 0xFFFFFFFF]
    print("ids: slots written", used, "tidx lane0", [hex(int(slots[i, 0])) for i in used])
    print("ids:", "OK" if used == list(range(8)) else "UNEXPECTED")
