"""Hardware diagnostics for the constructs of the Q4 kernel, each checked against numpy
(Pi side; seconds). A test that takes > 0.3 s has hung the GPU (driver timeout + reset).

  unif   per tile: branch with unif_addr resets the uniform stream; nb ldunif in a loop
  fifo   N back-to-back vec4 TMU reads (4N words outstanding) then N x 4 ldtmu: probes the
         TMU output FIFO depth (N = 1..4 -> 4, 8, 12, 16 words)
  math   band / shr by register / smul24 / asr by register (-16) / itof / fmul with .l / .h
         unpack of an fp16 pair, on the kernel's formula for one column
usage: bash tmp/gpu/gpurun.sh diag.py [unif fifo math]
"""
import sys
import time

import numpy as np
from videocore6.assembler import *  # noqa: F403
from videocore6.driver import Driver


def end(asm=None):
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def k_unif(asm):
    nop(sig=ldunifrf(rf0))   # act (stream address)
    nop(sig=ldunifrf(rf1))   # nt
    nop(sig=ldunifrf(rf2))   # nb
    nop(sig=ldunifrf(rf3))   # out
    eidx(r0)
    shl(r0, r0, 2)
    add(rf3, rf3, r0)
    shl(rf6, 4, 4)           # 64
    mov(rf4, rf1)
    b(R.tile, cond="always").unif_addr(rf0)
    nop()
    nop()
    nop()
    L.tile
    mov(rf5, rf2)
    mov(r1, 0)
    L.blk
    nop(sig=ldunif)
    add(r1, r1, r5)
    sub(rf5, rf5, 1, cond="pushz")
    b(R.blk, cond="anyna")
    nop()
    nop()
    nop()
    mov(tmud, r1)
    mov(tmua, rf3)
    add(rf3, rf3, rf6)
    sub(rf4, rf4, 1, cond="pushz")
    b(R.tile, cond="anyna").unif_addr(rf0)
    nop()
    nop()
    nop()
    tmuwt()
    end()


@qpu
def k_fifo(asm, nreq):
    nop(sig=ldunifrf(rf0))   # src
    nop(sig=ldunifrf(rf1))   # out
    nop(sig=ldunifrf(rf2))   # iterations
    eidx(r0)
    shl(r1, r0, 4)
    add(rf0, rf0, r1)        # + 16 * lane
    shl(r1, r0, 2)
    add(rf1, rf1, r1)        # + 4 * lane
    shl(rf3, 4, 4)           # 64
    shl(rf3, rf3, 2)         # 256 = 16 lanes x 16 B
    mov(r1, 0)
    L.loop
    for _ in range(nreq):
        bnot(tmuc, 3)
        mov(tmua, rf0)
        add(rf0, rf0, rf3)
    for _ in range(4 * nreq):
        nop(sig=ldtmu(r2))
        add(r1, r1, r2)
    sub(rf2, rf2, 1, cond="pushz")
    b(R.loop, cond="anyna")
    nop()
    nop()
    nop()
    mov(tmud, r1)
    mov(tmua, rf1)
    tmuwt()
    end()


@qpu
def k_math(asm):
    nop(sig=ldunifrf(rf0))   # w words (per lane)
    nop(sig=ldunifrf(rf1))   # sc words (per lane)
    nop(sig=ldunifrf(rf2))   # out: 4 words per lane
    nop(sig=ldunifrf(rf3))   # mask
    nop(sig=ldunifrf(rf4))   # 65536
    nop(sig=ldunifrf(rf5))   # x
    nop(sig=ldunifrf(rf6))   # negc
    nop(sig=ldunifrf(rf7))   # dx (float bits)
    eidx(r0)
    shl(r0, r0, 2)
    add(rf0, rf0, r0)
    add(rf1, rf1, r0)
    shl(r1, r0, 2)
    add(rf2, rf2, r1)
    mov(tmua, rf0)
    mov(tmua, rf1)
    nop()
    nop()
    nop(sig=ldtmu(rf10))
    nop(sig=ldtmu(rf11))
    mov(rf12, -16)
    mov(rf13, 4)
    shl(rf14, 4, 4)          # 64
    shr(r4, rf10, rf13)      # column m = 1: shift by register
    band(r2, r4, rf3)
    smul24(r3, r2, rf5)
    add(r3, r3, rf6)         # S
    smul24(r0, r3, rf4)
    asr(r0, r0, rf12)        # L
    sub(r1, r3, r0)
    asr(r1, r1, rf12)        # H
    mov(tmud, r0)
    mov(tmua, rf2)
    add(rf2, rf2, rf14)
    mov(tmud, r1)
    mov(tmua, rf2)
    add(rf2, rf2, rf14)
    itof(r0, r0)
    itof(r1, r1)
    fmul(r2, rf11.unpack("l"), rf7)
    fmul(r3, rf11.unpack("h"), rf7)
    fmul(r0, r0, r2)
    fmul(r1, r1, r3)
    mov(tmud, r0)
    mov(tmua, rf2)
    add(rf2, rf2, rf14)
    mov(tmud, r1)
    mov(tmua, rf2)
    tmuwt()
    end()


def run(drv, code, unif, label):
    t0 = time.perf_counter()
    drv.execute(code, local_invocation=(16, 1, 1), uniforms=unif.addresses()[0], thread=1)
    dt = time.perf_counter() - t0
    hang = " (HANG-like: > 0.3 s)" if dt > 0.3 else ""
    print(f"{label}: {dt * 1e3:.2f} ms{hang}")
    return dt


def test_unif(drv):
    nt, nb = 3, 10
    act = drv.alloc(nb, dtype=np.uint32)
    act[:] = np.arange(1, nb + 1) * 3
    out = drv.alloc((nt, 16), dtype=np.uint32)
    out[:] = 0xDEAD
    unif = drv.alloc(4, dtype=np.uint32)
    unif[:] = [act.addresses()[0], nt, nb, out.addresses()[0, 0]]
    code = drv.program(k_unif)
    run(drv, code, unif, "unif")
    want = int(act.sum())
    ok = bool((out == want).all())
    print(f"unif: {'OK' if ok else 'FAIL'} got {out[:, 0].tolist()} want {want}")


def test_fifo(drv, nreq):
    iters = 8
    src = drv.alloc((iters * nreq, 16, 4), dtype=np.uint32)
    src[:] = np.arange(src.size, dtype=np.uint32).reshape(src.shape)
    out = drv.alloc(16, dtype=np.uint32)
    out[:] = 0xDEAD
    unif = drv.alloc(3, dtype=np.uint32)
    unif[:] = [src.addresses()[0, 0, 0], out.addresses()[0], iters]
    code = drv.program(k_fifo, nreq)
    dt = run(drv, code, unif, f"fifo nreq={nreq} ({4 * nreq} words outstanding)")
    want = np.array(src).sum(axis=(0, 2)).astype(np.uint32)
    ok = bool((out == want).all())
    print(f"fifo nreq={nreq}: {'OK' if ok else 'FAIL'}{' hang' if dt > 0.3 else ''}")
    return ok and dt <= 0.3


def test_math(drv):
    rng = np.random.default_rng(3)
    w = rng.integers(0, 2 ** 32, 16, dtype=np.uint64).astype(np.uint32)
    sc = (rng.standard_normal(32) * 0.02).astype(np.float16).view(np.uint16).astype(np.uint32)
    scw = (sc[:16] | (sc[16:] << 16)).astype(np.uint32)
    x = int(rng.integers(-127, 128))
    negc = (-(8 * 65537 * 5 * x)) & 0xFFFFFFFF      # as if sum(x) over 5 columns were 5x
    dx = np.float32(rng.standard_normal() * 0.01)
    wb = drv.alloc(16, dtype=np.uint32)
    wb[:] = w
    sb = drv.alloc(16, dtype=np.uint32)
    sb[:] = scw
    out = drv.alloc((4, 16), dtype=np.uint32)
    out[:] = 0xDEAD
    unif = drv.alloc(8, dtype=np.uint32)
    unif[:] = [wb.addresses()[0], sb.addresses()[0], out.addresses()[0, 0], 0x000F000F, 65536,
               x & 0xFFFFFFFF, negc, dx.view(np.uint32)]
    code = drv.program(k_math)
    run(drv, code, unif, "math")
    p = (w >> 4) & 0x000F000F
    ua, ub = (p & 0xFFFF).astype(np.int64), (p >> 16).astype(np.int64)
    S = (ua * x + 65536 * ub * x + int(np.int64(negc).astype(np.int32) if negc < 2**31 else negc - 2**32)) & 0xFFFFFFFF
    S = S.astype(np.uint32).view(np.int32).astype(np.int64)
    L = ((S << 16) & 0xFFFFFFFF).astype(np.uint32).view(np.int32) >> 16
    H = ((S - L) >> 16).astype(np.int32)
    Lf = (sc[:16].astype(np.uint16).view(np.float16).astype(np.float32) * dx).astype(np.float32) * L.astype(np.float32)
    Hf = (sc[16:].astype(np.uint16).view(np.float16).astype(np.float32) * dx).astype(np.float32) * H.astype(np.float32)
    got = np.array(out)
    okL = bool((got[0].view(np.int32) == L).all())
    okH = bool((got[1].view(np.int32) == H).all())
    okLf = bool((got[2] == Lf.astype(np.float32).view(np.uint32)).all())
    okHf = bool((got[3] == Hf.astype(np.float32).view(np.uint32)).all())
    print(f"math: L {okL} H {okH} Lf {okLf} Hf {okHf}")
    if not (okL and okH and okLf and okHf):
        print("  got L", got[0].view(np.int32)[:4].tolist(), "want", L[:4].tolist())
        print("  got H", got[1].view(np.int32)[:4].tolist(), "want", H[:4].tolist())
        print("  got Lf", got[2].view(np.float32)[:4].tolist(), "want", Lf[:4].tolist())


def main():
    tests = sys.argv[1:] or ["unif", "fifo", "math"]
    with Driver() as drv:
        if "unif" in tests:
            test_unif(drv)
        if "math" in tests:
            test_math(drv)
        if "fifo" in tests:
            for nreq in (1, 2, 3, 4):
                if not test_fifo(drv, nreq):
                    break


if __name__ == "__main__":
    main()
