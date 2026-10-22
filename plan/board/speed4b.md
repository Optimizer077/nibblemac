# speed4b (4B prefill tuning; branch `worktree-agent-aff7b9e370c62872c`, commits fd0ed20 tail, 69b31a3 groups)

Note: written in the speed4b worktree (`.claude/worktrees/agent-aff7b9e370c62872c/plan/board/speed4b.md`):
the worktree isolation refused a write to the main checkout's `plan/board/`; the lead can copy it.

Cycles per block-vector (c/bv) = time x 1.5 GHz x threads / (rows/4 x cols/32 x n); one block-vector =
4 rows x 32 columns x 1 vector = 128 int8 MACs. All numbers at 1.5 GHz (freq_khz 1500000), Q4_G32
(= GGUF Q4_0) unless noted, kbench cold weights, 4 threads.

- 17:40 [measured] **per-shape GOP/s in the engine, 4B prefill p256 (n = 64 groups), before -> after**
  (same session, `cerno-speed-profile1_20261022_171543.json`; c/bv in brackets):
  qkv / q 8192x2560 53.2 (28.9) -> 52.5 (29.2); z 4096x2560 53.1 (29.0) -> 52.6 (29.2);
  out / o 2560x4096 50.0 (30.7) -> 51.2 (30.0); gate / up 9216x2560 52.9 (29.0) -> 53.1 (28.9);
  down 2560x9216 47.5 (32.3) -> 49.0 (31.3); k / v 1024x2560 50.4 (30.5) -> 51.5 (29.8);
  a / b 32x2560 19.3 -> 27.9. Matmul 35.83 -> 34.54 s per p256 run (-3.6%, incl. the tail below).
  0.8B Q4K at n = 64 (kbench, `cerno-speed-kb7-plan_*` q4k file): qkv 27.7 -> 27.4 c/bv, z 28.1 -> 27.9,
  gate 28.0 -> 28.1, out 31.1 -> 29.4, down 31.7 -> 30.4 (one round each; identical configs differ by
  up to 4% between runs, so read <= 3% as noise).
- 17:40 [measured] **kbench pass-size profile, 4B shapes, cold 4 threads**
  (`cerno-speed-kb5-passes_20261022_163146.json`): c/bv for one group of n = 2 / 3 / 4 / 5 / 6 / 7 / 8:
  c4qkv 57.1 / 38.6 / 30.9 / 31.0 / 32.1 / 28.0 / 33.2; c4down 56.8 / 38.8 / 34.7 / 31.8 / 33.4 / 30.3 / 36.9;
  c4out 57.3 / 38.8 / 32.6 / 30.7 / 33.2 / 29.1 / 35.4. Hot 1 thread kv / c4kv: n = 5 27.8 / 27.9,
  **n = 6 29.4 / 33.7**, n = 7 28.5 / 28.1. => **a 6-vector pass is 8-10 % worse per vector than 7 or 5**
  (GCC codegen: 6 MOVs + partial-sum spills in the nv = 6 loop); n = 2-3 are memory-bound.
- 17:40 [measured] **n > 7 as one group (several register passes over alternating activation sets) costs
  33-40 c/bv vs 28-32 when split into groups of <= 7**, for n = 8-16, every 4B and 0.8B shape, Q4 and Q4K
  (`cerno-speed-kb4-smalln_20261022_163112.json`, `cerno-speed-kb3-group_20261022_162756.json`).
  Old policy split only n > 16. Now (69b31a3): n > 7 split; sizes 7s + remainder in 5s / 4s
  (64 = 7x7 + 3x5, was 4x7 + 6x6); all groups in one job after a parallel activation quantization
  (`kern_group_fused`, ~1%). Bit-identical (kbench yhash: 1184 configs on the Pi, 1536 under qemu).
- 17:40 [measured] **prefill tail** (forward.c, fd0ed20): when only the last token's logits are needed,
  the last (full-attention) layer computes K/V for all rows and q / attention / o / MLP / final norm for
  the last token only: 1/32 of the 4B matmul work; logits identical
  (`cerno-speed-gate-cerno_20261022_170147.json`).
- 17:40 [measured] **end to end, before -> after** (csp0 = 4dbca6f, csp1 = 69b31a3; medians of 2 of 3
  rounds so far): 4B prefill 64 / 256 / 512 tokens 6.55 / 6.63 / 6.54 -> 6.80 / 6.84 / 6.85 tok/s
  (+3.7 / +3.2 / +4.8 %) (`cerno-speed-e2e4b-r1_*`, `-r2_*`); Cerno decisions (5 requests, 14 questions)
  mean 21.85 -> 21.04 s (-3.7 %), answers and probabilities identical (`cerno-speed-decide-r1_*`);
  0.8B prompt64 plain 7.81 -> 7.80, default spec 20.19 -> 20.26, prefill 43.8 -> 45.1 tok/s (3 rounds,
  `cerno-speed-e2e08_20261022_171753.json`).
- 17:40 [dead-end] **column chunking of K** (a chunk of the activations kept in L1 across the tiles,
  accumulators through Y; budgets 4-24 KB, with / without next-tile prefetch, groups of 14): never
  faster, smaller chunks slower (strided weight runs); exact but dropped. At n = 7 the activations of
  K = 9216 (80 KB) streaming from L2 cost only ~5% vs K = 2560 (`cerno-speed-kb1-chunk_20261022_161215.json`,
  `cerno-speed-kb2-var_20261022_162010.json`).
- 17:40 [dead-end] **fused DeltaNet sweeps** (second sweep of token t + first sweep of t+1 in one pass,
  half the state loads / stores, bit-identical): 5.79 -> 5.55-5.65 ms/token at n = 64 (-3 % of the
  recurrence = 0.15 % of prefill); the recurrence is NEON-bound (56 Q-ops per row and token), not
  load/store-bound. Dropped (`cerno-speed-lb1-fused_20261022_162826.json`).
- 17:40 [measured] recurrence at 4B head counts (lbench4: 32 V / 16 K heads, 24 layers, 4 threads):
  in place n = 1 / 7 / 64: 34.4 / 7.6 / 5.8 ms per token (`cerno-speed-lb0_20261022_162017.json`).
- 17:40 [idea] for:lut, for:numerics: the Q4 nv = 7 loop is near its NEON instruction floor of
  ~26.4-27 cycles per block-vector (16 SMULL/SMLAL + 3 ADDP + SADDLP + SCVTF (2 cycles) + FMUL + FMLA +
  FMLS, plus ~1.4 shared unpack / scale); measured 27-30 cold. The float part (4 ops) is fixed by
  exactness; what a new kernel must beat is the 16 MAC + 4 reduce ops per block-vector.
- 17:40 [measuring] `-fno-schedule-insns` on kern_quant.c removes all stack spills in the nv = 4-7
  loops (nv = 7: 232 -> 217 instructions per block, 12 -> 0 stack ops; nv = 6: no more MOVs); exact
  (qemu yhash 768 / 768 + float checks); Pi A/B running (`cerno-speed-kb8-nosched_*`). Would touch
  engine/Makefile + engine/bench/Makefile only (kern_quant.c compile flags).
