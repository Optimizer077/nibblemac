# Pilots

Short, bounded experiments (the owner's method: pilot quickly, then scale what works). One section
per pilot: the question, the budget, what was measured, the go / no-go verdict and the evidence.
Details live in the topic notebooks (`engine/*.md`); this file is the index of verdicts.

## Compilers (2026-10-22, toolchain agent; notebook `engine/TOOLCHAIN.md`)

**Question.** Does another compiler make the bit-identical engine faster on the Pi 4, first of all
the Q4_0 batched kernel of Cerno 4B prefill (n = 7 / 5 groups)? **Go** if a build is >= 3 % faster
on 4B prefill p256 and bit-identical to `engine/pinfer_run_fast`.

**Budget.** PC-side builds and qemu checks; ~1 h of Pi time under the lock (bench names `toolchain-*`).

**Builds** (same sources `69b31a3`, project flags `-O3 -mcpu=cortex-a72 -std=c11 -ffp-contract=off`,
static, cross-compiled in WSL): GCC 15.2 (shipping), GCC 14.3, GCC 16.0.1 (snapshot), clang 20.1.8,
clang 22.1.2, clang 22 + `-mllvm -enable-misched=false` on `kern_quant.c`.

**Exactness (PC).** kbench `--check`, 880 configurations per build under qemu: GCC 13 / 14 / 15 / 16
and clang 22 produce the same output bytes (yhash) as the shipping build on every row.
**clang 20 is rejected**: at `-O3` it miscompiles the max-by-magnitude loop of kbench's Q4 fill
(a different maximum in 1.7 % of the 32-element blocks; `-O2` or `-fno-unroll-loops` restore it),
i.e. it changes plain C floating-point results (TOOLCHAIN.md §3.1).

**A flag that looked good on paper and lost on the Pi.** `-fno-schedule-insns` on `kern_quant.c`
removes every stack spill of the Q4 loops (232 -> 217 instructions per block) and is 6-15 % *slower*
on every Q4 shape (speed4b's `cerno-speed-kb8-nosched_20261022_175206.json`): on the in-order A72
the interleaving of the multiplies with the reductions is worth more than the instruction count.

**Speed (Pi, 1.5 GHz, `bench/results/toolchain-kb-r1_20261022_220248.json`,
`toolchain-e2e4b-r1_20261022_220708.json`).** Q4_0 kernel, cold 4 threads, 4B shapes, medians of
3 rounds vs GCC 15: GCC 16 -2..+3 % and GCC 14 -3..+6 % (noise band), clang 22 +5..+15 % at n = 7,
clang 22 with the machine scheduler off +12..+22 %; every build's output hash identical on every row.
4B prefill p256 (one run each, owner's scope cut): GCC 15 6.915 tok/s, GCC 16 6.921 (+0.1 %),
GCC 14 6.789 (-1.8 %).

**Verdict: no-go.** No compiler or flag set is >= 3 % faster; the shortest code (no spills) loses
twice; clang 20 is excluded for correctness. `engine/Makefile` unchanged. The remaining lever for
this loop is instruction order inside the kernel (lut's asm passes), not the compiler.
