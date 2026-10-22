"""Generate the dispatch-ordered Q4_G32 / Q4K_G32 batched kernels (KERNELS.md Part 5).

usage: python tools/gen_kern_q4s.py OUT.h [--magic] [--early] [--prefix NAME]
  engine/bench/q4s_gen.h  = --prefix q4s          (tbench variant q4s)
  engine/bench/q4e_gen.h  = --early --prefix q4e  (q4e: per-vector loads at the step start)
  engine/bench/q4m_gen.h  = --magic --prefix q4m  (q4m: magic-number int->float, NV <= 6)
Measured and NOT integrated: -9 to -13 % cycles per block-vector on Q4 with all data in L1,
+2.1 / +3.5 % on the 4B prefill (64 / 256 tokens), below the 5 % bar (KERNELS.md Part 5).

The Cortex-A72 dispatches at most 3 uops per cycle in program order, at most one to each of
the F0 and F1 pipes (Arm Cortex-A72 Software Optimization Guide UAN0016A, section 4.1), and
SMULL / SMLAL execute only on F0 (p. 25). So a run of multiplies dispatches one per cycle and
blocks every op behind it. This generator writes one asm loop per pass size NV (2..7) in which
every multiply of vector v is followed by one op of the reduction of vector v-1 or of the
float update of vector v-2 (a software pipeline over the vectors of a block that continues
across blocks), so F1 works while F0 multiplies.

Bit-exactness: the integer products and the pairwise-add tree are those of kern_quant.c (same
int16 lanes, same order); the float part per (row, block, vector) is the same operations on
the same values: m = d4 * d[v] (FMUL), acc = fma(acc, float(t), m) (FMLA), acc = fms(acc, o4,
fsum[v]) (FMLS). float(t) is SCVTF, or with --magic the exact equivalent (t + 0x4B400000) as
float - 1.5 x 2^23 (|t| <= 16 x 4 x 15 x 127 < 2^22).

Registers: v0-v3 low nibbles (columns 0-15) of the 4 rows, v4-v7 high nibbles (16-31),
v8 d4, v9 o4, v10 / v11 the vector's scale / fsum (then d4 x scale), v12-v15 and v16-v19 the
int16 accumulators of two vectors in flight, v20 / v21 the int32 block sums of two vectors,
v22 / v23 the activations (columns 0-15 / 16-31), v24.. the fp32 accumulators, v31 the 0x0F
mask (v30 the magic constant with --magic, NV <= 6).
"""
import sys

MAGIC = "--magic" in sys.argv
EARLY = "--early" in sys.argv  # per-vector scale / sum loads at the start of the step
PREFIX = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "q4s"
OUT = sys.argv[1]


class Step:
    """Ops of one pipeline step: 16 multiply slots, X ops attached after slot i (-1 = before)."""

    def __init__(self):
        self.m = []          # 16 multiply ops (or empty in drain steps)
        self.x = {}          # slot -> list of ops

    def at(self, slot, *ops):
        self.x.setdefault(slot, []).extend(ops)

    def emit(self):
        out = list(self.x.get(-1, []))
        if self.m:
            for i, op in enumerate(self.m):
                out.append(op)
                out += self.x.get(i, [])
        else:
            for i in range(16):
                out += self.x.get(i, [])
        out += self.x.get(16, [])
        return out


def sets(g):
    a = ["v12", "v13", "v14", "v15"]
    b = ["v16", "v17", "v18", "v19"]
    return (a, b) if g % 2 == 0 else (b, a)


def tregs(g):
    """(T written by R in step g = t of vector g-1, T read by F = t of vector g-2)"""
    return ("v21", "v20") if g % 2 == 0 else ("v20", "v21")


def mul_ops(S):
    ops = []
    for r in range(4):
        ops.append("smull %s.8h, v%d.8b, v22.8b" % (S[r], r))
    for r in range(4):
        ops.append("smlal2 %s.8h, v%d.16b, v22.16b" % (S[r], r))
    for r in range(4):
        ops.append("smlal %s.8h, v%d.8b, v23.8b" % (S[r], 4 + r))
    for r in range(4):
        ops.append("smlal2 %s.8h, v%d.16b, v23.16b" % (S[r], 4 + r))
    return ops


def block_steps(NV, b_parity, kq, has_m=True, first=False, last=False, drain=False):
    """Steps of one block (or the 2 drain steps). b_parity: parity of the global step of v = 0."""
    steps = []
    nsteps = 2 if drain else NV
    for v in range(nsteps):
        g = b_parity + v
        S, Sp = sets(g)
        Tr, Tf = tregs(g)
        st = Step()
        M = has_m and not drain
        R = not (first and v < 1)
        F = not (first and v < 2)
        if drain:
            R = v == 0
            F = True
        w = (v - 2) if (v >= 2 and not drain) else (NV - 2 + v)  # acc index of the F vector
        acc = "v%d" % (24 + w)
        if M:
            st.m = mul_ops(S)
            st.at(-1, "ldr q23, [%%[xq], #%d]" % (32 * v + 16))
        if M and v == 0 and not first:  # high nibbles of this block (packed loaded last block)
            st.at(-1, "ushr v4.16b, v4.16b, #4")
            st.at(0, "ushr v5.16b, v5.16b, #4")
            st.at(2, "ushr v6.16b, v6.16b, #4")
            st.at(3, "ushr v7.16b, v7.16b, #4")
        if R:
            st.at(0, "addp %s.8h, %s.8h, %s.8h" % (Sp[0], Sp[0], Sp[1]))
            st.at(1, "addp %s.8h, %s.8h, %s.8h" % (Sp[2], Sp[2], Sp[3]))
            st.at(6, "addp %s.8h, %s.8h, %s.8h" % (Sp[0], Sp[0], Sp[2]))
            st.at(10, "saddlp %s.4s, %s.8h" % (Tr, Sp[0]))
            if MAGIC:
                st.at(11, "add %s.4s, %s.4s, v30.4s" % (Tr, Tr))
        if F:
            if MAGIC:
                st.at(2, "fsub %s.4s, %s.4s, v30.4s" % (Tf, Tf))
            else:
                st.at(2, "scvtf %s.4s, %s.4s" % (Tf, Tf))
            st.at(-1 if EARLY else 3, "ldr s10, [%%[fd], #%d]" % (4 * w))
            st.at(4, "fmul v10.4s, v8.4s, v10.s[0]")
            st.at(-1 if EARLY else 7, "ldr s11, [%%[ff], #%d]" % (4 * w))
            st.at(8, "fmla %s.4s, %s.4s, v10.4s" % (acc, Tf))
            st.at(12, "fmls %s.4s, v9.4s, v11.s[0]" % acc)
        if v == 1 and not drain:
            # F has finished the previous block: advance its pointers, load this block's d4/o4
            if F:
                st.at(13, "add %[fd], %[fd], %[fst]", "add %[ff], %[ff], %[fst]")
            if kq:
                st.at(13, "ldp q8, q9, [%[sp]]")
            else:
                st.at(13, "ldr d8, [%[sp]]", "fcvtl v8.4s, v8.4h")
                st.at(14, "fmov v9.4s, #8.0")
                st.at(15, "fmul v9.4s, v8.4s, v9.4s")
        if M and v + 1 < NV:
            st.at(7, "ldr q22, [%%[xq], #%d]" % (32 * (v + 1)))
        if M and v == NV - 1 and not last:
            # next block: x0 of its vector 0, packed codes (lo in place now, hi next block)
            st.at(7, "ldr q22, [%[xqn]]", "ldp q0, q1, [%[cpn]]", "ldp q2, q3, [%[cpn], #32]")
            st.at(9, "and v0.16b, v0.16b, v31.16b")
            st.at(11, "and v1.16b, v1.16b, v31.16b")
            st.at(13, "and v2.16b, v2.16b, v31.16b")
            st.at(14, "and v3.16b, v3.16b, v31.16b")
            st.at(15, "ldp q4, q5, [%[cpn]]", "ldp q6, q7, [%[cpn], #32]")
        steps.append(st)
    return steps


def block_addr_ops(kq, first):
    """Scalar ops at the end of block b: b += 1, then the pointers of block b (now current) and
    b + 1 (next): sp = scales of the current block, cpn = codes of the next block."""
    ops = ["add %[b], %[b], #1",
           "add %[xq], %[xq], %[qs]",
           "add %[xqn], %[xq], %[qs]",
           # current block b: superblock base = p + (b >> 3) * 576
           "lsr %[t1], %[b], #3",
           "madd %[t1], %[t1], %[c576], %[p]",
           "and %[t2], %[b], #7"]
    if kq:
        ops.append("add %[sp], %[ksc], %[b], lsl #5")
    else:
        ops.append("add %[sp], %[t1], %[t2], lsl #3")
    # next block b + 1
    ops += ["add %[t2], %[b], #1",
            "lsr %[t1], %[t2], #3",
            "madd %[t1], %[t1], %[c576], %[p]",
            "and %[t2], %[t2], #7",
            "add %[cpn], %[t1], %[t2], lsl #6",
            "add %[cpn], %[cpn], #64"]
    return ops


def emit_body(steps):
    """one source line per pipeline step (ops joined with ';' inside one asm string line)"""
    return [GROUP + "; ".join(st.emit()) for st in steps]


GROUP = "\x00"  # marks a line that holds a whole step


def gen_tile(NV, kq):
    name = "%s_%s_tile%d" % (PREFIX, "q4k" if kq else "q4", NV)
    acc = ["v%d" % (24 + i) for i in range(NV)]
    lines = []
    # prologue: constants, zero accumulators, block 0 codes, x0 of vector 0, d4 of block 0 at
    # step 1 (inside the block body)
    pro = ["movi v31.16b, #0x0f"]
    if MAGIC:
        pro.append("movi v30.4s, #0x4b, lsl #24")
        pro.append("orr v30.4s, #0x40, lsl #16")
    pro += ["movi %s.4s, #0" % a for a in acc]
    pro += ["mov %[b], #0",
            "add %[xqn], %[xq], %[qs]",
            "add %[cpn], %[p], #64",
            "ldp q0, q1, [%[cpn]]", "ldp q2, q3, [%[cpn], #32]",
            "ldp q4, q5, [%[cpn]]", "ldp q6, q7, [%[cpn], #32]",
            "ldr q22, [%[xq]]",
            "and v0.16b, v0.16b, v31.16b", "ushr v4.16b, v4.16b, #4",
            "and v1.16b, v1.16b, v31.16b", "ushr v5.16b, v5.16b, #4",
            "and v2.16b, v2.16b, v31.16b", "ushr v6.16b, v6.16b, #4",
            "and v3.16b, v3.16b, v31.16b", "ushr v7.16b, v7.16b, #4",
            "mov %[sp], %[p]" if not kq else "mov %[sp], %[ksc]",
            "add %[cpn], %[p], #128"]  # codes of block 1 (same superblock)
    lines += pro
    # block 0 (no R / F in its first steps)
    lines += emit_body(block_steps(NV, 0, kq, first=True))
    lines += block_addr_ops(kq, True)
    # loop over blocks 1 .. nblk-2
    odd = NV % 2 == 1
    lines += ["cmp %[b], %[nlast]", "b.ge 2f", "1:"]
    lines += emit_body(block_steps(NV, NV % 2, kq))          # block b (parity of b * NV)
    lines += block_addr_ops(kq, False)
    if odd:  # blocks come in pairs so that the register parity repeats
        lines += emit_body(block_steps(NV, 0, kq))
        lines += block_addr_ops(kq, False)
    lines += ["cmp %[b], %[nlast]", "b.lt 1b", "2:"]
    # final block nblk-1 (parity of (nblk-1) * NV = NV % 2), no next-block loads
    lines += emit_body(block_steps(NV, NV % 2, kq, last=True))
    # drain: global steps continue at parity (nblk * NV) % 2 = 0
    lines += emit_body(block_steps(NV, 0, kq, drain=True))
    for a in acc:
        lines += ["str q%s, [%%[y]]" % a[1:], "add %[y], %[y], %[yst]"]
    out, run = [], []
    for l in lines:  # join runs of plain ops; labels, compares and branches on their own line
        if l.startswith(GROUP) or l.endswith(":") or l.split()[0] in ("cmp", "b.ge", "b.lt"):
            if run:
                out.append("; ".join(run))
                run = []
            out.append(l.lstrip(GROUP))
        else:
            run.append(l)
    if run:
        out.append("; ".join(run))
    body = "".join('        "%s\\n\\t"\n' % l for l in out)
    clob = ", ".join('"v%d"' % i for i in range(32))
    return r'''
static void %(name)s(const uint8_t *p, long nblk, const int8_t *xq, const float *xd,
                     const float *xf, long qs, long fst, float *y, long yst, const float *ksc) {
    long b, t1, t2, nlast = nblk - 1;
    const int8_t *xqn;
    const uint8_t *cpn;
    const void *sp;
    const long c576 = 576;
    (void)ksc;
    __asm__ volatile(
%(body)s        : [b] "=&r"(b), [t1] "=&r"(t1), [t2] "=&r"(t2), [xqn] "=&r"(xqn), [cpn] "=&r"(cpn),
          [sp] "=&r"(sp), [xq] "+r"(xq), [fd] "+r"(xd), [ff] "+r"(xf), [y] "+r"(y)
        : [p] "r"(p), [nlast] "r"(nlast), [qs] "r"(qs), [fst] "r"(fst), [yst] "r"(yst),
          [ksc] "r"(ksc), [c576] "r"(c576)
        : %(clob)s, "cc", "memory");
}
''' % {"name": name, "body": body, "clob": clob}


HEAD = r'''/* GENERATED by tools/gen_kern_q4s.py%(magic)s --prefix %(prefix)s - do not edit; see that file
 * and KERNELS.md Part 5. Dispatch-ordered Q4_G32 / Q4K_G32 register passes of NV = 2..%(nvmax)d
 * vectors over one 4-row tile and all its blocks, bit-identical to the C loops of kern_quant.c.
 * %(prefix)s_<dtype>_tiles[nv](p, nblk, xq, xd, xf, qs, fst, y, yst, ksc): p = the tile's
 * superblocks, nblk = 8 x superblocks, xq / xd / xf = activations / scales / sums of the pass's
 * first vector in block 0, qs / fst = their byte strides per block, y / yst = output of the
 * first vector and the byte stride between vectors, ksc (Q4K only) = the tile's decoded
 * [block][d0..d3, o0..o3]. One pipeline step (16 multiplies + the interleaved ops) per line. */
#ifndef %(guard)s
#define %(guard)s
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
''' % {"magic": " --magic" if MAGIC else "", "prefix": PREFIX, "nvmax": 6 if MAGIC else 7,
       "guard": "PF_%s_GEN_H" % PREFIX.upper()}

with open(OUT, "w", newline="\n") as f:
    f.write(HEAD)
    nvs = range(2, 7) if MAGIC else range(2, 8)
    for kq in (0, 1):
        for nv in nvs:
            f.write(gen_tile(nv, kq))
    f.write("\ntypedef void (*%s_fn)(const uint8_t *, long, const int8_t *, const float *, const float *,\n"
            "                       long, long, float *, long, const float *);\n" % PREFIX)
    for kq in (0, 1):
        t = "q4k" if kq else "q4"
        f.write("static const %s_fn %s_%s_tiles[8] = {NULL, NULL" % (PREFIX, PREFIX, t))
        for nv in range(2, 8):
            f.write(", %s" % ("%s_%s_tile%d" % (PREFIX, t, nv) if nv in nvs else "NULL"))
        f.write("};\n")
    f.write("\n#endif\n")
print("wrote", OUT)
