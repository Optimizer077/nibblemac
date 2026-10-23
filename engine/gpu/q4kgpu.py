"""Q4K x int8 QPU pilot, preserving the original Q4K codes and coefficients.

At load, D*s6 and M*m6 become exact fp32 per-32 scale/offsets (17 significant
bits at most). No fp16 re-quantization. Packed integer MACs still use centered
u-8 codes so both signed 16-bit sums fit; adding 8*sum(q) after extraction
recovers the raw u dot q. The float order is explicitly modeled by reference_gpu.

Each lane loads d_a,d_b,o_a,o_b in one vec4. Next-block coefficients have their
own four registers: draining them after code word3 keeps the two-thread TMU
output queue <=8 words (Mesa v3d_compiler.h:16/threads). All load requests carry
thrsw and the scheduler keeps its following two instructions free of signals.
Primary FIFO definition: https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/91ab53b4b904a515324390908f652909ff501344/src/broadcom/compiler/v3d_compiler.h#49
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
PARAMS = ["codes", "scales", "y", "act", "nt", "nb", "tsc", "tss", "tsy", "r4", "mask", "k65536",
          "k512", "k64", "k256"]


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
        self.scales = [self.sc] + self.tmp_rf[:3]
        self.next_scales = self.tmp_rf[3:7]
        self.k256 = self.tmp_rf[7]
        self.tmp_rf = self.tmp_rf[8:]
        if len(self.tmp_rf) < 3:
            raise ValueError('Q4K supports passes of at most 8 vectors')
        self.sh = [None, self.sh4, self.sh8, self.sh12]


def block_ops(n: int, g: Regs) -> list[Q.Op]:
    A, S = Q.alu, Q.sig
    ops = []
    prod = [r0, r1] + g.tmp_rf[:max(0, NPROD - 2)]
    j = 0
    for k in range(8):
        for m in range(4):
            c = 4*k+m
            p = r2 if c % 2 == 0 else r3
            if m == 0:
                ops.append(A('band',p,g.w[k],g.mask))
            else:
                ops += [A('shr',r4,g.w[k],g.sh[m]),A('band',p,r4,g.mask)]
            if m == 3:
                ops.append(S('ldtmu',g.w[k]))
                if k == 3:
                    ops += [S('ldtmu',r) for r in g.next_scales]
                    ops += segment_ops(g,['hi'])
                if k == 5:
                    ops += segment_ops(g,['lo'])
            nmul = MUL_ADDS[m] if (m or k%2) else 0
            for v in range(n):
                pr = prod[j % len(prod)];j += 1
                unit = ('mul',) if v >= n-nmul else ACC_ALUS
                ops += [S('ldunif',tag=('x',c,v)),
                        A('smul24',pr,p,r5,alus=('mul',)),
                        A('add',g.acc[v],g.acc[v],pr,alus=unit)]
    pool = [r0,r1,r2,r3,r4]+g.tmp_rf
    sets = [pool[i:i+4] for i in range(0,len(pool)-3,4)]
    da,db,oa,ob = g.scales
    for v in range(n):
        t1,t2,s1,s2 = sets[v % len(sets)]
        ops += [A('smul24',t1,g.acc[v],g.k65536,alus=('mul',)),
                A('asr',t1,t1,g.k16),A('sub',t2,g.acc[v],t1,alus=EITHER),
                A('asr',t2,t2,g.k16),S('ldunif',tag=('sum8',v)),
                A('add',t1,t1,r5,alus=EITHER),A('add',t2,t2,r5,alus=EITHER),
                A('itof',t1,t1),A('itof',t2,t2),S('ldunif',tag=('dx',v)),
                A('fmul',s1,da,r5,alus=('mul',)),A('fmul',s2,db,r5,alus=('mul',)),
                A('fmul',t1,t1,s1,alus=('mul',)),A('fmul',t2,t2,s2,alus=('mul',)),
                A('fadd',g.ya[v],g.ya[v],t1),A('fadd',g.yb[v],g.yb[v],t2),
                S('ldunif',tag=('fsum',v)),
                A('fmul',s1,oa,r5,alus=('mul',)),A('fmul',s2,ob,r5,alus=('mul',)),
                A('fsub',g.ya[v],g.ya[v],s1),A('fsub',g.yb[v],g.yb[v],s2),
                S('ldunif',tag=('negc_next',v)),A('mov',g.acc[v],r5,alus=EITHER)]
    ops += [A('mov',dst,src,alus=EITHER) for dst,src in zip(g.scales,g.next_scales)]
    ops += segment_ops(g,['sc'])
    ops.append(A('sub',g.bcnt,g.bcnt,1,alus=('add',),cond='pushz'))
    return ops


# The block starts with next-block lo4 + coefficients4 in the FIFO. Word3 drains
# the coefficients into their second register bank, then requests hi4. Word5
# requests the following lo4; the float epilogue requests coefficients4. This
# keeps at most eight words outstanding, including tile transitions and drain.
PLANS = {'split': {'req': [], 'pending': ['lo','sc']}}
PLAN = "split"
SEG_WORDS = {"lo": 4, "hi": 4, "sc": 4}


def segment_ops(g: Regs, segs: list[str]) -> list[Q.Op]:
    """TMU load requests. Every load request carries the thrsw signal: without it the
    V3D 4.2 TMU never returns the data and the GPU hangs (gpu-diag_20261022_220321)."""
    A, M = Q.alu, ("mul",)
    ops = []
    for s in segs:
        if s == "lo":
            ops += [A("mov", tmuc, -4, alus=M),                                 # -4: vec4
                    A("mov", tmua, g.ca, alus=EITHER, thrsw=True)]
        elif s == "hi":
            ops += [A("mov", tmuc, -4, alus=M),
                    A("sub", tmua, g.ca, g.k16, alus=EITHER, thrsw=True),      # ca + 16
                    A("add", g.ca, g.ca, g.k512, alus=EITHER)]
        else:
            ops += [A("mov", tmuc, -4, alus=M),
                    A("mov", tmua, g.sa, alus=EITHER, thrsw=True),
                    A("add", g.sa, g.sa, g.k256, alus=EITHER)]
    return ops


def pending_words() -> int:
    return sum(SEG_WORDS[s] for s in PLANS[PLAN]["pending"])


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
                   "r4": g.ystride, "mask": g.mask, "k65536": g.k65536, "k512": g.k512, "k64": g.k64, "k256": g.k256}
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
            G["shl"](r3, r1, 4)
            G["add"](g.sa, g.sa, r2)
            G["add"](g.sa, g.sa, r3)                   # + 16 * lane
            G["shl"](r3, r1, 2)
            G["umul24"](r2, r0, g.tmp_rf[2])
            G["add"](g.yaddr, g.yaddr, r2)
            G["add"](g.yaddr, g.yaddr, r3)             # + 4 * lane
            G["mov"](g.sh4, 4)
            G["mov"](g.sh8, 8)
            G["mov"](g.sh12, 12)
            G["mov"](g.k16, -16)                       # shift amounts use the low 5 bits
            # block 0 of the first tile into registers, one segment at a time (<= 4 words
            # outstanding), then the plan's pending part of block 1
            for seg, dsts in (("lo", g.w[:4]), ("hi", g.w[4:]), ("sc", g.scales)):
                Q.emit(G, Q.schedule(segment_ops(g, [seg])))
                for d in dsts:
                    G["nop"](sig=G["ldtmu"](d))
            Q.emit(G, Q.schedule(segment_ops(g, PLANS[PLAN]["pending"])))
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
            for _ in range(pending_words()):   # drain what was requested past the last block
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

def layout_weights(u, d, o):
    """Codes (R,K) u4; exact fp32 coefficients d/o (R,K/32)."""
    R,K = u.shape;T,B = R//32,K//32
    assert R%32==0 and K%32==0
    uu=u.reshape(T,2,16,B,8,4).astype(np.uint32)
    sh=4*np.arange(4,dtype=np.uint32)
    words=(uu[:,0]<<sh).sum(-1,dtype=np.uint32)|(uu[:,1]<<(sh+16)).sum(-1,dtype=np.uint32)
    codes=np.ascontiguousarray(words.transpose(0,2,1,3))
    ds=d.astype(np.float32).reshape(T,2,16,B)
    os=o.astype(np.float32).reshape(T,2,16,B)
    scales=np.ascontiguousarray(np.stack((ds[:,0],ds[:,1],os[:,0],os[:,1]),axis=-1).transpose(0,2,1,3)).view(np.uint32)
    return codes,scales


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
        elif tag[0] == 'sum8':
            cols.append((8*qb.sum(-1)[tag[1]]).astype(np.int32).view(np.uint32))
        elif tag[0] == 'fsum':
            cols.append((dx*qb.sum(-1).astype(np.float32))[tag[1]].astype(np.float32).view(np.uint32))
        else:
            cols.append(dxs[tag[1]])
    head = [negc[tag[1], 0] for tag in kern.start_tags]
    body = np.stack(cols, axis=1).astype(np.uint32).reshape(-1)
    return np.concatenate([np.array(head, np.uint32), body])


def reference_gpu(u,d,o,q,dx):
    """Independent fp32 order model; intentionally separate from the NEON FMA order."""
    R,K=u.shape;n=q.shape[0];B=K//32
    raw=np.einsum('rbc,nbc->nrb',u.astype(np.int32).reshape(R,B,32),q.astype(np.int32).reshape(n,B,32))
    sums=q.astype(np.int32).reshape(n,B,32).sum(-1).astype(np.float32)
    fs=(dx*sums).astype(np.float32)
    y=np.zeros((n,R),np.float32)
    for b in range(B):
        scale=(d[None,:,b]*dx[:,None,b]).astype(np.float32)
        y=(y+(raw[:,:,b].astype(np.float32)*scale).astype(np.float32)).astype(np.float32)
        y=(y-(o[None,:,b]*fs[:,None,b]).astype(np.float32)).astype(np.float32)
    return y


def reference_cpu_order(u,d,o,q,dx):
    R,K=u.shape;n=q.shape[0];B=K//32
    raw=np.einsum('rbc,nbc->nrb',u.astype(np.int32).reshape(R,B,32),q.astype(np.int32).reshape(n,B,32)).astype(np.float64)
    sums=q.astype(np.int32).reshape(n,B,32).sum(-1).astype(np.float32)
    fs=(dx*sums).astype(np.float32)
    y=np.zeros((n,R),np.float32)
    for b in range(B):
        scale=(d[None,:,b]*dx[:,None,b]).astype(np.float32).astype(np.float64)
        y=(y.astype(np.float64)+raw[:,:,b]*scale).astype(np.float32)
        y=(y.astype(np.float64)-o[None,:,b].astype(np.float64)*fs[:,None,b].astype(np.float64)).astype(np.float32)
    return y
