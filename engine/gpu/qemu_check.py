"""PC-side check of the generated Q4 kernel: a numpy emulator of the scheduled instructions
(16 lanes; reads at instruction start, writes at the end; TMU FIFO; uniform stream) runs
the tile / block / tile-end regions for every tile of a small matrix and must reproduce
q4gpu.reference_gpu bit for bit. Also assembles the full program (encoding check).

usage: python qemu_check.py PYVC6_SRC [n ...]
"""
from __future__ import annotations

import os
import sys

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), sys.argv[1]]
import numpy as np  # noqa: E402

import q4gpu  # noqa: E402
from _videocore6.assembler import assemble  # noqa: E402

M32 = np.uint64(0xFFFFFFFF)


def f32(a):
    return a.view(np.float32)


class Emu:
    def __init__(self, mem: np.ndarray, unif: np.ndarray):
        self.mem = mem            # uint32 words, byte address = 4 * index
        self.unif = unif
        self.up = 0
        self.reg: dict[str, np.ndarray] = {}
        self.fifo: list[np.ndarray] = []
        self.tmuc = None
        self.tmud: list[np.ndarray] = []

    def val(self, s):
        if isinstance(s, float):
            return np.full(16, np.float32(s)).view(np.uint32)
        if isinstance(s, int):
            return np.full(16, s & 0xFFFFFFFF, np.uint32)
        v = self.reg.get(s.name, np.zeros(16, np.uint32))
        ub = s.unpack_bits[0] if hasattr(s, "unpack_bits") else 1
        if ub == 2:   # .l : low f16 -> f32
            return (v & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32).view(np.uint32)
        if ub == 3:   # .h
            return (v >> 16).astype(np.uint16).view(np.float16).astype(np.float32).view(np.uint32)
        return v.copy()

    def alu(self, op):
        a = [self.val(s) for s in op.srcs]
        nm = op.name
        i32 = lambda x: x.view(np.int32)  # noqa: E731
        if nm in ("mov", "bor"):
            return a[0] if nm == "mov" or len(a) == 1 else a[0] | a[1]
        if nm == "add":
            return (a[0] + a[1]).astype(np.uint32)
        if nm == "sub":
            return (a[0] - a[1]).astype(np.uint32)
        if nm == "band":
            return a[0] & a[1]
        if nm == "bnot":
            return ~a[0]
        if nm == "shr":
            return a[0] >> (a[1] & 31)
        if nm == "shl":
            return (a[0].astype(np.uint64) << (a[1] & 31).astype(np.uint64) & M32).astype(np.uint32)
        if nm == "asr":
            return (i32(a[0]) >> (a[1] & 31).astype(np.int32)).view(np.uint32)
        if nm == "smul24":
            s = lambda x: ((x & 0xFFFFFF).astype(np.int64) ^ 0x800000) - 0x800000  # noqa: E731
            return ((s(a[0]) * s(a[1])) & 0xFFFFFFFF).astype(np.uint32)
        if nm == "itof":
            return i32(a[0]).astype(np.float32).view(np.uint32)
        if nm == "fmul":
            return (f32(a[0]) * f32(a[1])).astype(np.float32).view(np.uint32)
        if nm == "fadd":
            return (f32(a[0]) + f32(a[1])).astype(np.float32).view(np.uint32)
        raise NotImplementedError(nm)

    def write(self, dst, v):
        nm = dst.name
        if nm == "null":
            return
        if nm == "tmuc":
            self.tmuc = v[0]
        elif nm == "tmud":
            self.tmud.append(v)
        elif nm == "tmua":
            if self.tmud:   # store
                for i in range(16):
                    self.mem[v[i] // 4] = self.tmud[0][i]
                assert len(self.tmud) == 1
                self.tmud = []
            else:           # load (vec4 if configured)
                cnt = 4 if self.tmuc == 0xFFFFFFFC else 1
                for k in range(cnt):
                    self.fifo.append(self.mem[(v + 4 * k) // 4].copy())
                # the TMU output FIFO holds 16 words per QPU thread on V3D 4.x
                assert len(self.fifo) <= 16, f"TMU output FIFO overflow: {len(self.fifo)}"
                self.max_fifo = max(getattr(self, "max_fifo", 0), len(self.fifo))
            self.tmuc = None
        else:
            self.reg[nm] = v

    def run(self, slots):
        for s in slots:
            outs = []
            for op in (s.add, s.mul):
                if op is not None:
                    outs.append((op.dst, self.alu(op)))
            for sg in s.sigs:
                if sg.name == "ldunif":
                    outs.append((q4gpu.r5, np.full(16, self.unif[self.up], np.uint32)))
                    self.up += 1
                elif sg.name == "ldtmu":
                    outs.append((sg.dst, self.fifo.pop(0)))
            for dst, v in outs:
                self.write(dst, v)


def check(n: int, R: int = 64, K: int = 256, seed: int = 0) -> bool:
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 16, (R, K), dtype=np.uint8)
    d = (rng.standard_normal((R, K // 32)) * 0.01).astype(np.float16)
    x = rng.standard_normal((n, K)).astype(np.float32)
    q, dx = q4gpu.quant_act(x)
    kern = q4gpu.Kernel(n)
    codes, scales = q4gpu.layout_weights(u, d)
    T, B = R // 32, K // 32
    # memory map (byte addresses)
    cbase, sbase = 0, codes.nbytes + 512 * 3
    ybase = sbase + scales.nbytes + 64 * 3
    words = (ybase + n * R * 4) // 4 + 64
    mem = np.zeros(words, np.uint32)
    mem[cbase // 4: cbase // 4 + codes.size] = codes.reshape(-1)
    mem[sbase // 4: sbase // 4 + scales.size] = scales.reshape(-1)
    unif = q4gpu.build_uniforms(kern, q, dx)
    e = Emu(mem, unif)
    g = kern.g
    lane = np.arange(16, dtype=np.uint32)
    c = lambda v: np.full(16, v & 0xFFFFFFFF, np.uint32)  # noqa: E731
    e.reg.update({g.mask.name: c(q4gpu.MASK_PAIR), g.sh4.name: c(4), g.sh8.name: c(8),
                  g.sh12.name: c(12), g.k16.name: c(-16), g.k65536.name: c(65536),
                  g.k512.name: c(512), g.k64.name: c(64), g.ystride.name: c(4 * R),
                  g.nb.name: c(B), g.yaddr.name: c(ybase) + 4 * lane})
    ca = c(cbase) + 32 * lane
    sa = c(sbase) + 4 * lane
    # prologue: block 0 in registers, block 1 in the TMU FIFO, CA / SA at block 2
    w = [mem[(ca + 4 * k) // 4] for k in range(8)]
    for k in range(8):
        e.reg[g.w[k].name] = w[k]
    e.reg[g.sc.name] = mem[sa // 4]
    e.fifo = [mem[(ca + 512 + 4 * k) // 4].copy() for k in range(8)] + [mem[(sa + 64) // 4].copy()]
    e.reg[g.ca.name], e.reg[g.sa.name] = ca + 1024, sa + 128
    for t in range(T):
        e.up = 0
        e.run(kern.start_slots)
        for b in range(B):
            e.run(kern.body_slots)
        assert e.up == len(unif), (e.up, len(unif))
        e.run(kern.end_slots)
    assert len(e.fifo) == 9, len(e.fifo)   # the epilogue drains exactly one block
    y = mem[ybase // 4: ybase // 4 + n * R].view(np.float32).reshape(n, R)
    ref = q4gpu.reference_gpu(u, d, q, dx)
    cpu = q4gpu.reference_cpu_order(u, d, q, dx)
    exact = bool(np.array_equal(y.view(np.uint32), ref.view(np.uint32)))
    rel = float(np.abs(y - cpu).max() / np.abs(cpu).max())
    ndiff = int((y != cpu).sum())
    print(f"n={n}: emulator == reference_gpu: {exact}; vs CPU formula: {ndiff}/{y.size} differ, "
          f"max |diff| / max|y| = {rel:.2e}; stats {kern.stats()}")
    code = assemble(kern.program())
    print(f"   assembled {len(code)} instructions")
    return exact


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[2:]] or [1, 2, 7, 8, 11]
    ok = all(check(n) for n in ns)
    print("ALL OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)
