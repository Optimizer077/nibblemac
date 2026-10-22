"""Q4_0 x int8 batched matmul on the Raspberry Pi 4 V3D 4.2 QPUs (engine/GPU.md).

Layout (prepared on the host, `layout_weights`):
  lane i of a 32-row tile t holds rows a = 32t + i (low 16-bit field) and b = 32t + 16 + i
  (high field); code word k (0..7) of block b holds the u4 codes of columns 32b + 4k + m,
  m = 0..3: row a at bits 4m, row b at bits 16 + 4m. Codes as [tile][block][lane][8 words]
  (two 16-byte TMU vec4 reads per lane and block); scales [tile][block][lane] = fp16 d_a |
  fp16 d_b << 16.
Math per block and vector v (one smul24 = 2 MACs):
  S = -524296 * sum(x) + sum_c x_c * ((W >> 4m) & 0x000F000F)        (int32, wrapping)
    = L + 65536 * H, L = sum x (u_a - 8), H = sum x (u_b - 8), |L|, |H| <= 32512
  y_a += float(L) * (d_a * dx_v);  y_b += float(H) * (d_b * dx_v)      (fp32 mul, add)
Activations are a uniform stream (one 32-bit uniform per smul24), built per call by
`build_uniforms` from the int8 codes, scales and block sums (same quantization as the CPU).
"""
from __future__ import annotations

import numpy as np

import qsched as Q
from _videocore6.assembler import Instruction, qpu  # MIT-licensed part of py-videocore6

REG = Instruction.REGISTERS
r0, r1, r2, r3, r4, r5 = (REG[f"r{i}"] for i in range(6))
tmua, tmud, tmuc, null = REG["tmua"], REG["tmud"], REG["tmuc"], REG["null"]
RF = [REG[f"rf{i}"] for i in range(64)]
MASK_PAIR = 0x000F000F
OFFS = 8 * 65537  # u - 8 in both fields
EITHER = ("add", "mul")
# MAC-section schedule knobs (tuned with sweep_sched.py): product registers in rotation,
# accumulate adds moved to the mul ALU per column (index m; m = 0 only for odd words),
# ALUs allowed for the other accumulate adds.
NPROD = 2
MUL_ADDS = [1, 1, 1, 1]
ACC_ALUS = ("add", "mul")
REQUEST_AFTER_WORD = 4   # TMU request for block b+2 after word 4 of b+1 was consumed
PARAMS = ["codes", "scales", "y", "act", "nt", "nb", "tsc", "tss", "tsy", "r4", "mask", "k65536",
          "k512", "k64"]


class Regs:
    """rf allocation for a pass of n vectors."""

    def __init__(self, n: int):
        it = iter(RF)
        take = lambda k: [next(it) for _ in range(k)]  # noqa: E731
        self.acc, self.ya, self.yb = take(n), take(n), take(n)
        self.w = take(8)
        (self.sc, self.mask, self.sh4, self.sh8, self.sh12, self.k65536, self.k16, self.k512,
         self.k64, self.ca, self.sa, self.yaddr, self.bcnt, self.tcnt, self.act, self.nb,
         self.ystride) = take(17)
        self.tmp_rf = list(it)
        if len(self.tmp_rf) < 3:
            raise ValueError(f"n = {n} needs too many registers")
        self.sh = [None, self.sh4, self.sh8, self.sh12]


def block_ops(n: int, g: Regs) -> list[Q.Op]:
    """One 32-column block for n vectors (the loop body)."""
    A, S = Q.alu, Q.sig
    ops: list[Q.Op] = []
    # (acc[v] = -524296 * sum(x) of this block was set by the previous block / tile start;
    #  block b+1's words were requested at the end of block b-1 and arrive by ldtmu here)
    prod = [r0, r1] + g.tmp_rf[:max(0, NPROD - 2)]
    j = 0
    for k in range(8):
        for m in range(4):
            c = 4 * k + m
            p = r2 if c % 2 == 0 else r3
            if m == 0:
                ops.append(A("band", p, g.w[k], g.mask))
            else:
                ops += [A("shr", r4, g.w[k], g.sh[m]), A("band", p, r4, g.mask)]
            if m == 3:
                ops.append(S("ldtmu", g.w[k]))  # next block's word k
                if k == REQUEST_AFTER_WORD:
                    # request block b+2 once k+1 of the 9 words of b+1 left the TMU output
                    # FIFO (16 words on V3D 4.x): (8 - k) + 9 <= 16 outstanding
                    ops += tmu_request_ops(g)
            # accumulate adds: add ALU, except MUL_ADDS[m] of them (the last ones) on the mul ALU
            nmul = MUL_ADDS[m] if (m or k % 2) else 0
            for v in range(n):
                pr = prod[j % len(prod)]
                j += 1
                unit = ("mul",) if v >= n - nmul else ACC_ALUS
                ops += [S("ldunif", tag=("x", c, v)), A("smul24", pr, p, r5, alus=("mul",)),
                        A("add", g.acc[v], g.acc[v], pr, alus=unit)]
    # extraction + float scaling; temps rotate over sets of 4 registers
    pool = [r0, r1, r2, r3, r4] + g.tmp_rf
    sets = [pool[i:i + 4] for i in range(0, len(pool) - 3, 4)]
    for v in range(n):
        t1, t2, s1, s2 = sets[v % len(sets)]
        ops += [A("smul24", t1, g.acc[v], g.k65536, alus=("mul",)),   # (S & 0xffff) << 16
                A("asr", t1, t1, g.k16),                               # L
                A("sub", t2, g.acc[v], t1, alus=EITHER),
                A("asr", t2, t2, g.k16),                               # H
                A("itof", t1, t1), A("itof", t2, t2),
                S("ldunif", tag=("dx", v)),
                A("fmul", s1, g.sc.unpack("l"), r5, alus=("mul",)),
                A("fmul", s2, g.sc.unpack("h"), r5, alus=("mul",)),
                A("fmul", t1, t1, s1, alus=("mul",)), A("fmul", t2, t2, s2, alus=("mul",)),
                A("fadd", g.ya[v], g.ya[v], t1), A("fadd", g.yb[v], g.yb[v], t2),
                # start value of the next block (a dummy after the tile's last block)
                S("ldunif", tag=("negc_next", v)), A("mov", g.acc[v], r5, alus=EITHER)]
    ops.append(S("ldtmu", g.sc))  # next block's scales
    ops.append(A("sub", g.bcnt, g.bcnt, 1, alus=("add",), cond="pushz"))
    return ops


def tmu_request_ops(g: Regs) -> list[Q.Op]:
    """Request the block at CA / SA (two vec4 code reads + one scale read), advance CA / SA."""
    A, M = Q.alu, ("mul",)
    return [A("mov", tmuc, -4, alus=M), A("mov", tmua, g.ca, alus=EITHER),    # -4 = vec4 config
            A("mov", tmuc, -4, alus=M), A("sub", tmua, g.ca, -16, alus=EITHER),
            A("mov", tmua, g.sa, alus=EITHER),
            A("add", g.ca, g.ca, g.k512, alus=EITHER), A("add", g.sa, g.sa, g.k64, alus=EITHER)]


def tile_end_ops(n: int, g: Regs) -> list[Q.Op]:
    A = Q.alu
    cur = g.tmp_rf[0]
    ops = [A("mov", cur, g.yaddr, alus=EITHER)]
    for v in range(n):
        ops += [A("mov", tmud, g.ya[v], alus=EITHER), A("mov", tmua, cur, alus=EITHER),
                A("mov", tmud, g.yb[v], alus=EITHER), A("add", tmua, cur, g.k64, alus=EITHER),
                A("add", cur, cur, g.ystride, alus=EITHER)]
    ops += [A("add", g.yaddr, g.yaddr, g.k64, alus=EITHER),
            A("add", g.yaddr, g.yaddr, g.k64, alus=EITHER),
            A("sub", g.tcnt, g.tcnt, 1, alus=("add",), cond="pushz")]
    return ops


def tile_start_ops(n: int, g: Regs) -> list[Q.Op]:
    A, S = Q.alu, Q.sig
    ops = []
    for v in range(n):
        ops += [A("mov", g.ya[v], 0.0, alus=EITHER), A("mov", g.yb[v], 0.0, alus=EITHER),
                S("ldunif", tag=("negc", v)), A("mov", g.acc[v], r5, alus=EITHER)]
    ops.append(A("mov", g.bcnt, g.nb, alus=EITHER))
    return ops


class Kernel:
    """Generated program for passes of n vectors; `block_tags` = uniform order per block."""

    def __init__(self, n: int):
        self.n = n
        self.g = Regs(n)
        self.body = block_ops(n, self.g)
        self.block_tags = Q.uniform_tags(self.body)
        self.body_slots = Q.schedule(self.body)
        start = tile_start_ops(n, self.g)
        self.start_tags = Q.uniform_tags(start)
        self.start_slots = Q.schedule(start)
        self.end_slots = Q.schedule(tile_end_ops(n, self.g))

    def stats(self) -> dict:
        st = Q.stats(self.body_slots)
        st["macs_per_lane"] = 64 * self.n
        st["mac_per_instr"] = round(64 * self.n / (st["instr"] + 4), 3)  # + branch + 3 slots
        return st

    def program(self):
        k = self

        @qpu
        def prog(asm):
            G = globals()
            g = k.g
            # params -> registers (the main uniform stream at dispatch)
            dst = {"codes": g.ca, "scales": g.sa, "y": g.yaddr, "act": g.act, "nt": g.tcnt,
                   "nb": g.nb, "tsc": g.tmp_rf[0], "tss": g.tmp_rf[1], "tsy": g.tmp_rf[2],
                   "r4": g.ystride, "mask": g.mask, "k65536": g.k65536, "k512": g.k512, "k64": g.k64}
            for name in PARAMS:
                G["nop"](sig=G["ldunifrf"](dst[name]))
            # qpu index q = (tidx >> 2) & 15 (one thread per QPU)
            G["tidx"](r0)
            G["shr"](r0, r0, 2)
            G["band"](r0, r0, 15)
            G["eidx"](r1)
            G["umul24"](r2, r0, g.tmp_rf[0])          # q * tile-range stride (codes)
            G["shl"](r3, r1, 5)
            G["add"](g.ca, g.ca, r2)
            G["add"](g.ca, g.ca, r3)                   # + 32 * lane
            G["umul24"](r2, r0, g.tmp_rf[1])
            G["shl"](r3, r1, 2)
            G["add"](g.sa, g.sa, r2)
            G["add"](g.sa, g.sa, r3)                   # + 4 * lane
            G["umul24"](r2, r0, g.tmp_rf[2])
            G["add"](g.yaddr, g.yaddr, r2)
            G["add"](g.yaddr, g.yaddr, r3)             # + 4 * lane
            G["mov"](g.sh4, 4)
            G["mov"](g.sh8, 8)
            G["mov"](g.sh12, 12)
            G["mov"](g.k16, -16)                       # shift amounts use the low 5 bits
            # block 0 of the first tile into registers, block 1 requested (2 blocks in flight)
            Q.emit(G, Q.schedule(tmu_request_ops(g)))
            for w in g.w:
                G["nop"](sig=G["ldtmu"](w))
            G["nop"](sig=G["ldtmu"](g.sc))
            Q.emit(G, Q.schedule(tmu_request_ops(g)))
            G["b"](G["R"].tile, cond="always").unif_addr(g.act)
            G["nop"]()
            G["nop"]()
            G["nop"]()
            G["L"].tile
            Q.emit(G, k.start_slots)
            G["L"].block
            Q.emit(G, k.body_slots)
            G["b"](G["R"].block, cond="anyna")
            G["nop"]()
            G["nop"]()
            G["nop"]()
            Q.emit(G, k.end_slots)
            G["b"](G["R"].tile, cond="anyna").unif_addr(g.act)
            G["nop"]()
            G["nop"]()
            G["nop"]()
            for _ in range(9):   # drain the request for the block after the last one
                G["nop"](sig=G["ldtmu"](r0))
            G["tmuwt"]()
            G["nop"](sig=G["thrsw"])
            G["nop"](sig=G["thrsw"])
            G["nop"]()
            G["nop"]()
            G["nop"](sig=G["thrsw"])
            G["nop"]()
            G["nop"]()
            G["nop"]()

        return prog


# ------------------------------------------------------------------ host side

def layout_weights(u: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u: (R, K) uint8 codes 0..15 (value = (u - 8) * d), d: (R, K/32) float16.
    Returns (codes uint32 (T, B, 16, 8), scales uint32 (T, B, 16)), each + 1 padding block."""
    R, K = u.shape
    assert R % 32 == 0 and K % 32 == 0
    T, B = R // 32, K // 32
    uu = u.reshape(T, 2, 16, B, 8, 4).astype(np.uint32)
    sh = (4 * np.arange(4, dtype=np.uint32))
    words = (uu[:, 0] << sh).sum(axis=-1, dtype=np.uint32) | (uu[:, 1] << (sh + 16)).sum(axis=-1, dtype=np.uint32)
    codes = np.ascontiguousarray(words.transpose(0, 2, 1, 3))         # (T, B, 16, 8)
    dh = d.astype(np.float16).view(np.uint16).astype(np.uint32).reshape(T, 2, 16, B)
    scales = np.ascontiguousarray((dh[:, 0] | (dh[:, 1] << 16)).transpose(0, 2, 1))  # (T, B, 16)
    return codes, scales


def quant_act(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CPU-identical int8 per-32 quantization (kern_act.c): q = rint(x * 127/amax), d = amax/127."""
    n, K = x.shape
    xb = x.reshape(n, K // 32, 32).astype(np.float32)
    amax = np.abs(xb).max(axis=-1)
    inv = np.where(amax > 0, np.float32(127.0) / np.where(amax > 0, amax, 1), 0).astype(np.float32)
    q = np.rint(xb * inv[..., None]).astype(np.int8)
    return q.reshape(n, K), (amax / np.float32(127.0)).astype(np.float32)


def build_uniforms(kern: Kernel, q: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Uniform stream for one pass: q (n, K) int8, dx (n, K/32) float32 -> uint32 (B * per_block)."""
    n, K = q.shape
    B = K // 32
    assert n == kern.n
    qb = q.reshape(n, B, 32).astype(np.int64)
    negc = (-(OFFS * qb.sum(axis=-1))).astype(np.int64) & 0xFFFFFFFF          # (n, B)
    negc_next = np.concatenate([negc[:, 1:], np.zeros((n, 1), np.int64)], axis=1)
    xs = qb.astype(np.int64) & 0xFFFFFFFF                                       # (n, B, 32)
    dxs = dx.astype(np.float32).view(np.uint32).astype(np.int64)                # (n, B)
    cols = []
    for tag in kern.block_tags:
        if tag[0] == "negc_next":
            cols.append(negc_next[tag[1]])
        elif tag[0] == "x":
            cols.append(xs[tag[2], :, tag[1]])
        else:
            cols.append(dxs[tag[1]])
    head = [negc[tag[1], 0] for tag in kern.start_tags]
    body = np.stack(cols, axis=1).astype(np.uint32).reshape(-1)
    return np.concatenate([np.array(head, np.uint32), body])


def reference_gpu(u: np.ndarray, d: np.ndarray, q: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """What the kernel computes, in fp32 with round-to-nearest mul and add: (n, R)."""
    R, K = u.shape
    n = q.shape[0]
    B = K // 32
    w = (u.astype(np.int32) - 8).reshape(R, B, 32)
    qb = q.astype(np.int32).reshape(n, B, 32)
    S = np.einsum("rbc,nbc->nrb", w, qb)                       # exact block sums (n, R, B)
    df = d.astype(np.float32)                                   # (R, B)
    y = np.zeros((n, R), np.float32)
    for b in range(B):
        s = (df[None, :, b] * dx[:, None, b]).astype(np.float32)   # fp32 product
        y = (y + (S[:, :, b].astype(np.float32) * s).astype(np.float32)).astype(np.float32)
    return y


def reference_cpu_order(u, d, q, dx) -> np.ndarray:
    """The CPU kernel's formula (kern_quant.c q4_block), FMA emulated in float64 then rounded."""
    R, K = u.shape
    n = q.shape[0]
    B = K // 32
    uu = u.astype(np.int64).reshape(R, B, 32)
    qb = q.astype(np.int64).reshape(n, B, 32)
    T = np.einsum("rbc,nbc->nrb", uu, qb).astype(np.float64)
    fs = (dx * qb.sum(-1).astype(np.float32)).astype(np.float32)        # fsum = d * sum(q)
    df = d.astype(np.float32)
    y = np.zeros((n, R), np.float32)
    for b in range(B):
        p = (df[None, :, b] * dx[:, None, b]).astype(np.float32).astype(np.float64)
        y = (y.astype(np.float64) + T[:, :, b] * p).astype(np.float32)          # fma
        o = (df[:, b] * np.float32(8)).astype(np.float64)
        y = (y.astype(np.float64) - o[None, :] * fs[:, None, b].astype(np.float64)).astype(np.float32)
    return y
