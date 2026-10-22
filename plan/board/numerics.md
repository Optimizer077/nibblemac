# numerics (Winograd inner product, integer superblock accumulation, int16 chains, fast
transcendentals, fp16/bf16 DeltaNet state; notebook `engine/NUMERICS.md` on branch
`worktree-agent-ab578c0376849057a`, base 69b31a3)

(Kept in my worktree: the isolation guard of this agent refuses writes to the main checkout.
Path: `.claude/worktrees/agent-ab578c0376849057a/plan/board/numerics.md`.)

- 17:30 [idea] **Winograd overflow analysis (paper, before the microbench).** Pair column j with
  j + 16 (the Q4 nibble layout already holds them in the low / high nibble of one byte, so no
  shuffles): sum_j (x_j + w_{j+16})(x_{j+16} + w_j) = x.w + eta(x) + xi(w), eta = sum x_j x_{j+16}
  per vector-block, xi = sum w_j w_{j+16} per row-block. Per 4 rows x 32 cols x 1 vector:
  8 ADD.16b + 8 SMULL/SMLAL2 instead of 16 SMULL/SMLAL. Two constraints on the A72 int8 path:
  (1) x + w in int8: x +-127 with w in [-8, 7] needs 9 bits -> clamp x to +-119 (signed codes)
  or +-112 (u4 codes); (2) int16 lane headroom: |P.Q| <= 127^2 = 16129, so a lane holds 2
  products. Today's tree keeps 4 products per lane per row and 3 pairwise levels to 16 per lane
  (4 ops per block-vector). Winograd keeps that 4-op tree only if |P.Q| <= 4095 (8 per lane),
  i.e. activations ~+-48..55 (6.6 bits); <= 8191 gives a 5-op tree (+-82, 7.4 bits); full range
  needs SADDLP x4 + ADDP32 x3 = 7 ops. Cost model (ubench: SMULL 1.0 cycle, ADD.16b in an
  SMULL mix ~0.79): MAC part 16 -> ~12.6 cycles, reduction 4 -> 7, + 1 op for xi (eta folds into
  the existing offset FMLS for free), + xi itself (on the fly ~1.7 per bv, or a stored int16 table
  = +0.5 bit/weight, ~250 MB for the 4B). => full-range Winograd ~29 vs ~28 cycles per
  block-vector today (no gain); 6.6-bit activations ~26 (-6 %, with a quality cost).
  Microbench on the Pi follows.
- 17:30 [idea] **Integer superblock accumulation** (candidate 2). Q4K: with one activation scale
  per 256 columns, a block's reduced int16x8 [r0 r0 r1 r1 r2 r2 r3 r3] (<= 30480) is multiplied by
  the 6-bit sub-scales with SMLAL/SMLAL2 .4s into int32 (bound 30480 x 63 x 8 = 15.4 M per
  superblock; a whole 9216-column row fits too), the mins term is one SMLAL-by-element per
  block-vector; SADDLP + SCVTF + FMUL + FMLA + FMLS (~6 cycles) -> 3 cycles per block-vector, the
  float work moves to once per superblock. Estimate 28 -> ~24 cycles per bv (-12..15 %).
  Q4_0 (Cerno): fp16 scales per 32 -> needs integer sub-scales (8-bit multipliers of a per-256
  float in a side table; weights change by <= 0.4 % of a block scale) plus per-256 activation
  scales; same estimate. Both class (b).
- 17:30 [idea] candidate 4 (fast exp / SiLU / rsqrt) is mostly done already (core2's Cephes expf
  in forward_ops.c); all non-matmul elementwise work is ~1 ms of a 127 ms 0.8B step and ~0.6 s
  of a 37 s 4B prefill (~2 %), so its ceiling is < 1 % end to end. I will measure the cycle cost
  and report it as a bound.
- 17:30 [idea] candidate 5 (fp16 / bf16 DeltaNet state) targets 0.8B decode and the speculative
  verify / commit passes (state copy + replay were ~37 ms of a ~160 ms pass in CORE.md's profile);
  0.8B plain-decode ceiling ~4-5 ms of 127 (state r+w 37.7 MB/token incl. the dirty write-back
  that slows the next out_proj).
- 17:30 [for:compress] activation precision: int8 per-32 activations add ~0.7-0.9 % of max|y| output
  error (KERNELS.md); my int7 / per-256 / mixed-radix variants will report PPL deltas here.
- 18:05 [ack] after lead 17:5x / owner: focus on Cerno (Q4_0 batched kernel at n = 7 / 5, 4B shapes,
  decision time). Dropped for now: fp16 DeltaNet state (0.8B decode item; at 4B n = 64 the recurrence is
  NEON-bound per speed4b, fp16 storage would add conversions) and fast transcendentals: on 4B prefill
  every elementwise phase together is ~3 ms of 150.7 ms per token (audit 17:27), and mlp_act is
  DRAM-bound at n = 64 (4.7 MB of gate/up outputs per layer call read back, ~3.7 GB/s), so a cheaper exp
  saves nothing measurable. lut owns the exact F0/F1-reordered kernel; my class-(b) variants go on top.
- 18:05 [measured] (static, objdump of nbench, GCC 15 -O3, no Pi yet) instructions in the nv = 7 block
  loop (7 block-vectors): engine kern_q4 232 (33.1 per bv, 12 stack ops); `sgn` (signed codes, no FMLS)
  227 (19 stack ops: GCC spills more); Winograd `wf` 319 / `w7` 307 / `w55` 299 (45.6 / 43.9 / 42.7 per bv:
  8 SMULL + 8 ADD + 7/5/4-op tree + 2 corrections per bv as designed, but 46-48 stack ops: P, Q and xi
  push it over 32 registers); integer superblock `isb` (Q4_0, per-256 activations, exact 14-bit integer
  sub-scales from the fp16 scales on the fly, SMLAL .4s into int32, one SCVTF per 256 columns) 193
  (27.6 per bv, -17 %); `isbk` (Q4K) 184 vs kern_q4k 220. Under lut's F0 model isb has 18 F0 uops per bv
  (16 SMULL + 2 SMLAL .4s), like today's 16 SMULL + SCVTF Q: no F0 saving, only fewer total uops. All
  variants pass the double-reference check under qemu (rel_err_q <= 1.8e-5; isb's is the rounding of
  multipliers of blocks < 1/16 of the superblock's largest scale). Pi timing queued (numerics-nb1-hot).
- 18:15 [measured] (PC) **Cerno 4B Q4_0 block scales as integers**: my integer-superblock kernel writes each
  fp16 block scale as D_r x M (D_r = 2^(E_r - 28) per row and superblock, |M| <= 16376); over all 111,534,080
  block scales of cerno-4b-q4_0.pq every one is reproduced exactly (1491 needed a right shift, their low
  mantissa bits were zero) (tools/numerics/q4_scale_stats.py). So `isb` on Cerno changes ONLY the activation
  granularity (int8 per 256 columns instead of per 32); its integer sums are more accurate than today's float
  accumulation (rel. error vs double 4e-8..1e-7 vs 1.6e-7..4.5e-7 in nbench).
- 18:15 [integrating] (my branch only) opt-in `pinfer_params.act_quant` (0 default = today, 1 = per-256 +
  integer superblock kernels for Q4_G32 / Q4K_G32): new `engine/src/kern_qsb.c` (kern_q4_sb, kern_q4k_sb,
  kern_quant_act_sb), `pf_act.xs` (kern.h), `pf_pool_set_act_quant` (ops.h, matvec_ref.c no-op), pf_matmul
  picks the sb kernel per pool; pinfer.h / pinfer.c field, `--act-quant` in pinfer_run / pinfer_ppl, Python
  `Model.load(act_quant=)`. kern_quant.c untouched (for:lut: no conflict with your asm passes). Default path
  bit-identical by construction; Pi gates and Cerno agreement pending (Pi paused).
- 18:15 [idea] for:lut cost models: under your F0 model the integer-superblock multiply (2 SMLAL .4s per
  block-vector) is F0 work, so on top of a pipelined base isb has 18 F0 uops per bv vs 16; under your
  write-unit model it needs ~43 units vs ~50 (base with magic conversion). A hybrid that keeps one float
  FMLA per block but by the weight scale only (per-256 activations remove the per-vector FMUL and the FMLS)
  has 16 F0 uops and ~46 units. I will measure isb (C) against your asm base when both run.
- 18:15 [dead-end] (analysis, Pi numbers pending) **Winograd on the A72**: (a) +-119 activations fit int8
  but 2 products per int16 lane force a 7-op reduction; (b) objdump: 8 SMULL + 8 ADD.16b replace 16 SMULL 1:1
  in instruction count, so Winograd is 36-39 instructions per bv vs 31 (no-spill builds); (c) F0 model:
  8 F0 + ~22 other uops = ~15 cycles at perfect dual issue vs 16 (+6 % best case); (d) write-unit model:
  ~56 units vs ~50 -> slower. No regime in which it wins clearly; int7 (w7) and +-55 (w55) only shorten the
  tree and cost activation precision (nbench rel_err_fp 2x of int8).
