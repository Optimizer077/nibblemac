# lut (table-lookup kernels; compute ceiling of the Q4 kernels; branch worktree-agent-aa5998fa1cda1365d from 69b31a3)

- 17:35 [idea] Arithmetic bound for T-MAC-style LUT on the A72, before measuring: a TBL lane is
  one byte, and an exact table must hold its sums in 8 bits. With 4-bit indices (TBL 1 reg) the best
  split is 1 weight bit x 4 columns x 6 activation bits, but int8 activations need 2 slices (8 = 6 + 2
  is no better than 4 + 4), so every 16 rows x 4 columns x 4 weight bits cost 4 planes x 2 slices = 8 TBL
  = 8 MACs per TBL, the same as one SMLAL (8 MACs), before the adds that accumulate the bytes and
  the index unpacking. The guide gives TBL Q-form latency 3N+3 (no throughput; ubench measured 1.5
  cycles for Q 1-reg) vs SMLAL 1/cycle on F0 (Arm Cortex-A72 Software Optimization Guide UAN0016A,
  pp. 25, 30). So a pure LUT kernel cannot beat SMLAL; a hybrid only pays if TBL issues on F1 while
  SMLAL holds F0. Measuring that now (ubench2).
- 17:35 [idea] The real lever may be dispatch order: UAN0016A section 4.1 says the A72 dispatches up
  to 3 uops per cycle **in program order**, at most one F0 and one F1 uop per cycle. SMLAL is F0-only;
  ADDP / SADDLP / logical F0/F1; SADALP and shifts F1-only; FP mul/fma D-form F0/F1 (thr 2), Q-form
  thr 1; SCVTF Q-form F0-only, thr 1/2. ubench.c's mixes ran 8 SMULLs then 8 others, which in-order
  dispatch serializes whatever the pipe ("SMULL + USHR 0.94 cycles/insn"); interleaved, SMLAL and an
  F1 op might pair. Today's Q4 block-vector = 16 SMLAL (F0) + 4 reductions + SCVTF + FMUL + FMLA +
  FMLS + loads; if the float part moved to D-form ops on F1 and the int->float to a magic-number add
  (exact for |t| < 2^22, same float), F0 would carry only the 16 SMLAL: floor 16 cycles vs 28-30
  measured. ubench2 (`engine/bench/ubench2.c`) tests exactly these pairings; running on the Pi.
- 17:50 [measured] (static, no Pi yet: Pi paused for the overclock window) GCC's code for the shipping
  nv = 7 Q4 loop (`kern_q4`, 232 instructions per block = 33 per block-vector) clusters the multiplies:
  runs of 8-12 SMULL/SMLAL with no F1 op between them, so under in-order dispatch F1 idles while F0
  multiplies. `-fno-schedule-insns -fno-schedule-insns2` on an interleaved C version (tbench variant
  `isq`) does not keep the order either (GIMPLE reorders, 6 register MOVs + 1 spill per block).
- 17:50 [measuring] **asm generator `tools/gen_kern_q4s.py`**: one asm loop per pass size NV = 2..7 (Q4_G32
  and Q4K_G32), every multiply of vector v followed by one op of vector v-1's reduction (ADDP x3,
  SADDLP) or vector v-2's float update (SCVTF, FMUL, FMLA, FMLS: the same ops and operands as
  kern_quant.c), pipelined across blocks; fixed registers (32 used at NV = 7). Variant `q4m` = same
  with the int->float as ADD 0x4B400000 + FSUB 1.5*2^23 (exact for |t| < 2^22), NV <= 6.
  **Bit-exact under qemu**: Q4 and Q4K, n = 1-8, 13, 64, 1 and 3 threads, memcmp vs kern_q4/kern_q4k
  (tbench `--check`). Pi timing queued for when the Pi is back (tbench, lut-* bench names).
- 17:50 [for:speed4b] heads-up, not integrated yet: if the asm passes win, they replace the inner loop of
  `q4_tile` for nv >= 2 in kern_quant.c (same dispatch, same group plan); your group / pass-size tuning
  (7s + 5s) stays valid but the c/bv per pass size will change, so the 6-vs-5/7 choice may need a re-check.
- 18:05 [measured] **ubench2: A72 pairing rules with the dispatch order controlled** (cycles per loop of 16
  independent ops, 8 + 8 interleaved; `bench/results/lut-ubench2_20261022_175209.json`, `lut-ubench2b_20261022_175214.json`):
  SMLAL alone 16.0 (1/cycle, F0). SMLAL + any **D-form** op (USHR, ADD, FMLA/FMUL/FSUB .2s, TBL .8b,
  scalar MADD) **8.34 = perfect pairing**. SMLAL + Q-form ALU (ADD/ADDP/AND/MOVI/ZIP .16b/.8h) 11.95,
  + Q-form USHR / SADDLP / FMUL / FSUB 13.68, + FMLA .4s 12.39, + SADALP 11.01, + LDR q 11.02,
  + SCVTF .4s 24.0 (F0, 2 cycles each). At 3:1 (24 SMLAL + 8 X): FMLA/FMUL/FSUB .4s 25.0, ADDP 24.0,
  TBL .8b 24.1 = nearly free; SCVTF .4s 40. Grouped order (8 SMLAL then 8 X, as ubench.c did):
  SMLAL + USHR 15.6 vs 13.7 interleaved (ubench.c's "0.94 serialized" came from the order).
  One Q4 block-vector's op mix, hand-interleaved: **19.9 cycles** (16 SMLAL + 3 ADDP + SADDLP + SCVTF +
  FMUL + FMLA + FMLS + 2 loads); with D-form floats + magic int->float 19.7. The kernel runs 27-30.
  Model that fits most rows: the FP/ASIMD register file takes ~3 x 64-bit writes per cycle (SMLAL /
  Q-form / LDR q = 2, D-form = 1): 52-54 units per block-vector => ~18-20 cycles floor, not 16.
- 18:05 [measured] **TBL on the A72**: TBL .16b 1 reg 1.50 cycles, 2 regs 3.0, 4 regs 7.0; **TBL .8b 1 reg
  0.50 (2 per cycle, F0/F1)**, .8b 2 regs 1.0. (UAN0016A p. 30 gives latency only: 3N, Q-form 3N+3.)
- 18:05 [dead-end] **T-MAC LUT kernel on the A72** (analysis + ubench2; tbench numbers follow): an exact
  int8 table entry is a byte, so a TBL lane carries at most 16 bit-products = 0.5 MAC (1 weight bit x 4
  columns x a 4-bit activation slice); a TBL .8b = 4 MACs per 64-bit result write vs SMLAL 8 MACs per
  128-bit write = the same 4 MACs per write unit, and the LUT bytes still need accumulating adds
  (more writes) while SMLAL accumulates in place. Writes, not the multiplier, bound the kernel, so
  LUT cannot win; a hybrid has no room either (SMLAL alone uses 2 of the ~3 write units per cycle).
- 18:05 [for:numerics] The same write-port budget applies to Winograd: its pre-adds (ADD .16b, 2 units
  each) and the extra reductions cost writes; the multiply count is not the binding limit once the
  ops are interleaved (16 SMLAL = 32 of the ~54 write units of a block-vector).
- 18:05 [for:compress] TBL .8b is cheap (0.5 cycle, pairs with SMLAL): a 16-entry codebook decode is
  ~free next to SMLAL as long as its outputs are 8 bytes (D-form); Q-form TBL (1.5 cycles) is not.
- 18:08 [integrating] (in my branch only, not committed until the Pi numbers are in) `engine/src/kern_quant.c`:
  kern_q4 / kern_q4k with n >= 2 call the generated asm passes `engine/src/kern_q4s.h`
  (`tools/gen_kern_q4s.py`), same VECTOR_PASSES split (NV_MAX 7); Q4K decodes the tile's scales once
  per tile (not per pass); knob `kern_q4_asm` (default 1, kbench `--q4asm 0|1`); `kern.h` declares it.
  n = 1 (decode) unchanged. No change to matvec_fast.c, the group plan, formats or float math.
  qemu gate: 388 kbench configs (q4 / q4k / q8; kv, out, ab, c4kv, c4ab, c4down, down; n = 1-17, 64,
  `--group 0`) all exact, yhash identical to a 69b31a3 build. for:speed4b: this sits under your 69b31a3
  group plan; nothing of yours changes.
- 21:10 [integrating] WIP committed on branch `worktree-agent-aa5998fa1cda1365d` as d43cc18 (asm passes in
  kern_quant.c via `engine/src/kern_q4s.h`, knob `kern_q4_asm`; tbench / ubench2). **Not timed on the Pi
  yet** (Pi offline). Host: golden, state, dims PASS, malformed 43/43 (ASan + UBSan); pytest / spec running.
- 21:40 [for:toolchain] agreed, thanks: I take the Pi first when it answers for `lut-tbench-r1` (tbench + a
  kbench A/B, ~10 min, under the lock), then release; you run `toolchain-kb-r1`; I slot my e2e / decide /
  gate rounds between your rounds. My Pi files live in `tmp/lut/` and `engine/pinfer_run_lut*`.
- 21:40 [measured] host gates on d43cc18: golden PASS, state PASS, dims PASS, malformed 43/43 (ASan + UBSan),
  pytest 130 passed; test_spec killed twice (WSL restart at 21:27, then SIGTERM at 21:30), rerunning
  with --jobs 4. qemu yhash gate 2 (q4 c4qkv / c4z / c4out / c4gate n = 5, 7, 12 and q4k qkv / gate / q /
  z n = 2-7): 28 / 28 identical to 69b31a3.
