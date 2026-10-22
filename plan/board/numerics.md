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
