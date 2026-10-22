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
