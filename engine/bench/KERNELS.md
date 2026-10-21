# Kernels lab notebook (matvec / small-batch matmul on the Pi 4 Cortex-A72)

Owner: kernels/performance work. Files: `engine/src/matvec_fast.c` (pool + dispatch),
`engine/src/kern_float.c` (F32/BF16), `engine/src/kern_quant.c` (Q8/Q4/Q4K),
`engine/src/kern_act.c` (activation quantization, row dequant), `engine/src/kern.h`,
`engine/src/pq_quant.h`, `tools/quant.py`, `engine/bench/{kbench.c,kbench_util.c,ubench.c}`.

All numbers: Pi 4 (Cortex-A72 @ 1.5 GHz, governor ondemand, stock kernel 6.18.50, 4 KB
pages, no huge pages), static `aarch64-linux-gnu-gcc 15.2 -O3 -mcpu=cortex-a72`, run through
`harness/pi.py bench` (Pi-wide lock). GB/s = weight bytes of the matrix / time per call.
Reference ceiling: memband 4.20 GB/s (1 thread), 4.07 (4 threads)
(`bench/results/memband_20261021_210615.json`).

## How kbench measures
- Real Qwen3.5-0.8B shapes: qkv 6144x1024, z 2048x1024, out 1024x2048, q 4096x1024,
  kv 512x1024, gate/up 3584x1024, down 1024x3584, ab 16x1024 (in_proj_a/b), head 248320x1024.
- Cold weights: every call reads a different copy of the matrix from a >= 64 MB arena (L2 is
  1 MB), as in decode. `--hot`: one copy (cache-resident when it fits in L2) = compute ceiling.
- Correctness per configuration: `rel_err_kernel` = max |y - y_ref| / max |y_ref| against a
  double-precision reference built from `pf_get_row` rows and, for quantized dtypes, the
  exact int8-quantized activations the kernel sees (must be < 1e-4; observed 1e-7..2e-6).
  `rel_err_fp` = same against fp32 activations (shows activation-quantization error).
- Golden file from `tools/quant.py golden`: Python-packed matrices + expected outputs; checks
  that C `pf_get_row` equals the Python dequantization exactly and kernels match.
- Foreign-load guard (added after interference, see below): each timed repetition compares
  machine-wide busy CPU time (/proc/stat) with this process's (getrusage); repetitions where
  other processes used > ~5% of a core are retried and the result is flagged `noisy`.

## Machine facts measured here (ubench)
`engine/bench/ubench.c`: 16 independent copies of one instruction per loop, cycles/insn at
1.5 GHz. Results `bench/results/ubench-a72_20261021_215638.json`,
`bench/results/ubench-a72-mix_20261021_220124.json`.

| instruction | cycles | | pair (8 SMULL + 8 X) | cycles/insn |
|---|---|---|---|---|
| SMULL / SMLAL / SMLAL2 (8b->8h) | 1.00 | | + ADD/AND/ADDP | 0.79 (partial overlap) |
| SMLAL .4s (16-bit) | 1.00 | | + USHR | 0.94 (serialized) |
| MUL/MLA .8h, MLA .16b (non-widening) | **2.00** | | + SADDLP | 0.94 (serialized) |
| FMLA .4s | 1.00 | | + FMLA .4s | 1.00 (serialized) |
| FMLA .2s | 0.50 | | + SCVTF | 1.51 |
| ADD/AND/ADDP | 0.69 | | + LD1 (q) | 0.72 |
| USHR, SSHLL, SHLL, SADDLP, SADALP | 1.00 | | | |
| SCVTF .4s | **2.0-2.3** | | | |
| FCVTL (f16->f32) | **2.0** | | | |
| TBL (1 reg) | 1.5 | | | |

Consequences used throughout:
- Integer int8 MACs: 8/cycle (SMULL/SMLAL into int16). fp32 FMLA: 4 MACs/cycle (Q or D form).
  So int8 x int8 with activation quantization has 2x the MAC rate of dequant-to-fp32;
  non-widening MUL/MLA are half rate and not useful; int16 SMLAL .4s is 4 MACs/cycle.
- Almost nothing co-issues with SMULL: in practice the A72 retires ~1 128-bit NEON op/cycle.
  Cost model = count NEON instructions. A 4-row x 32-column Q4 block costs 16 multiplies +
  ~8 unpack + 4 reduce + ~6 scale/convert/accumulate => ~34 cycles for one vector; in a
  batch the unpack is shared and ~27 cycles per vector remain.
- SCVTF and FCVTL are expensive: convert once per 4-row block (lane = row), never per row.
- DRAM: one core streams 4.2 GB/s = 2.8 B/cycle. Q8 needs 128 B per ~38 cycles (3.4 B/cycle)
  => one core is enough; Q4 needs 64 B per ~38 cycles (1.7 B/cycle) => >= 2 cores.

## Variant log (chronological; losers included)
Cycle numbers are "cycles per 4-row x 32-column block per vector" (`--hot`, 1 thread, kv
shape unless noted) = time x 1.5 GHz / (rows/4 x cols/32 x n). A Q8 block is 128 B of codes,
a Q4 block 64 B.

### v1 — first complete version (2026-10-21 21:52)
Pool (spin then futex, workers pinned to CPUs 1..n-1, caller on CPU 0), contiguous slabs of
4-row tiles per thread. Float: 4 rows x up to 4 vectors per pass, BF16 via SHLL #16.
Quantized: int8 activations per 32 columns, SMULL/SMLAL(2) to int16, 4-row pairwise-add
reduction to one int32x4 (lane = row), one SCVTF + FMLA per block. First Q4 format variant
Q4A_G64 (asymmetric, fp16 scale+min per 64 columns) next to Q4_G32.
- `kbench-v1-n1_20261021_215259.json`: **invalid as a measurement** — another agent's job
  occupied a core during the first half: every 4-thread number was 1-2 GB/s while 3-thread
  numbers on CPUs 0-2 were fine. Led to the foreign-load guard.
- `kbench-v1-t4check_20261021_215346.json` (idle Pi), n=1 cold GB/s, 1 thread / 4 threads:
  F32 3.82-3.88 / 3.40-3.69; BF16 3.04-3.60 / 3.44-3.80; **Q8 4.14-4.25** / 3.77-4.04;
  Q4 2.28-2.31 / 3.59-3.93; Q4A 2.19-2.22 / 3.62-3.97.
- `kbench-v1-hot_20261021_215531.json`: hot = cold for Q8 (4.31) and Q4 (2.34) => both
  compute-bound on one core, at ~42-44 cycles/block, 2x the instruction-count estimate.
  (`kbench-v1-hot_20261021_215507/215518.json` are failed runs: RESULT format bug, fixed.)

### v2 — unroll the 8-block loop of a superblock
GCC kept the block loop rolled and spilled the scale line to the stack every block; full
unroll + moving the Q4 "-8" offset to a separate accumulator (`corr`, off the critical path).
`kbench-v2-hot_20261021_215816.json`: Q8 n=1 4.31 -> 5.20 GB/s (hot), Q4 2.34 -> 2.64,
Q4A 2.19 -> 2.79. But batches got worse: Q4 n=8 0.32 GB/s (x8 vectors) — heavy spills
(~2800 q-register stack ops in kern_q4 at nv=8).

### v3 — separate n=1 (unrolled) and batched (rolled) block loops
`kbench-v3-hot_20261021_220028.json`: n=1 unchanged (Q8 5.35, Q4 2.81 GB/s hot);
batched cycles/block/vector Q4 35.0 (n=2), 29.8 (n=4), 36.8 (n=8). Spills 2800 -> 139.

### ubench (see table above) — why ~40 cycles: SMULL does not co-issue with the rest.

### Prefetch sweep (v3 binary, cold, n=1)
`kbench-v3-pfsweep_20261021_220346.json`: **invalid** — another agent pushed 965 MB and
ran an unlocked sha256sum over it during the pf=0 pass.
`kbench-v3-pfsweep2_20261021_220614.json` (guarded, 0 noisy), GB/s for prefetch distance
0 / 256 / 512 / 1024 bytes (PRFM PLDL1STRM burst per superblock / per 32 columns):
- Q8 1 thread qkv: 4.31 / 3.38 / 3.39 / 3.36 — software prefetch **hurts** (load-port
  pressure; the hardware prefetcher already keeps up). 2 threads 4.13 / 4.11 / 4.11 / 4.10.
- F32 1 thread qkv 3.78 / 3.32 / 3.65 / 3.66; BF16 1 thread down 3.00 / 3.73 / 3.50 / 3.30
  (noisy pattern, no consistent winner); 2 threads all within 3.7-3.97.
- Q4 4 threads down 3.86 / 4.06 / 3.97 / 3.93; qkv 3.68 / 3.86 / 3.92 / 3.92.
Verdict: default `kern_prefetch = 0`; the knob stays for experiments (`--prefetch`).

### v4 — unroll the vector loop too
`kbench-v4-hot_20261021_221054.json`: Q4 n=8 36.8 -> 33.6 cycles; spills rose to 253.

### v5 — block-major activation layout
Activations stored as q[block][vector][32], d/fsum[block][vector] (internal `pf_act`), so a
block's data for all vectors is one pointer + immediate offsets; the Q4 offset term became
one FMLS with a precomputed per-block sum of x instead of an integer broadcast + subtract.
`kbench-v5-hot_20261021_222536.json`: Q4 n=8 33.6 -> 29.5, n=4 27.8; Q8 n=8 31.6.

## Quantization error on the real Qwen3.5-0.8B weights (`tools/quant.py report`)
Relative RMS error ||W - Q(W)|| / ||W|| aggregated per tensor type, in % (embed: every 4th
row). Full table: `logs/quant/quant_error_report_v2.txt` / `.json` (first run without Q4K:
`logs/quant/quant_error_report.txt`).

| tensor type | Q8_G32 (8.5 b) | q8 g128 (8.125) | Q4_G32 (4.5) | q4 g32 no search | q4 asym g64 (4.5, was Q4A) | q4 asym g32 (5.0) | **Q4K_G32 (4.5)** |
|---|---|---|---|---|---|---|---|
| embed_tokens (= head) | 0.541 | 0.683 | 8.453 | 8.903 | 8.655 | 7.380 | **7.452** |
| linear_attn.in_proj_qkv | 0.537 | 0.676 | 8.408 | 8.853 | 8.696 | 7.403 | **7.474** |
| linear_attn.in_proj_z | 0.532 | 0.667 | 8.332 | 8.777 | 8.630 | 7.364 | **7.433** |
| linear_attn.out_proj | 0.552 | 0.700 | 8.644 | 9.090 | 8.904 | 7.539 | **7.616** |
| linear_attn.in_proj_a / _b | 0.555 / 0.558 | 0.714 / 0.725 | 8.679 / 8.749 | | | | 7.639 / 7.672 |
| mlp.gate_proj / up_proj | 0.536 / 0.534 | 0.672 / 0.667 | 8.392 / 8.353 | 8.836 / 8.799 | 8.675 / 8.645 | 7.392 / 7.376 | **7.463 / 7.446** |
| mlp.down_proj | 0.539 | 0.680 | 8.435 | 8.880 | 8.713 | 7.410 | **7.481** |
| self_attn.q_proj | 0.539 | 0.679 | 8.443 | 8.888 | 8.728 | 7.427 | **7.498** |
| self_attn.k_proj | 0.555 | 0.712 | 8.693 | 9.137 | 8.995 | 7.583 | **7.659** |
| self_attn.v_proj | 0.603 | 0.804 | 9.415 | 9.844 | 9.636 | 7.964 | **8.051** |
| self_attn.o_proj | 0.569 | 0.734 | 8.914 | 9.362 | 9.173 | 7.699 | **7.778** |
| mtp.fc | 0.877 | 1.581 | 11.660 | 11.959 | 12.011 | 8.965 | **9.071** |
| mtp.self_attn.o_proj | 0.649 | 0.903 | 10.188 | 10.638 | 10.305 | 8.350 | **8.458** |

Findings:
- At 4.5 bits/weight the k-quant style Q4K_G32 is 11-22% lower error than symmetric Q4_G32
  and within 1% of a plain asymmetric g32 format that costs 5.0 bits. The asymmetric g64
  layout first implemented as Q4A_G64 (id 4) was **dominated** by Q4_G32 (8.7% vs 8.4% at
  the same size and speed) and was replaced by Q4K_G32 before any file used id 4.
- The scale search in Q4_G32 (try 7 scales + LS refit) is worth 0.45 points over plain
  Q4_0-style rounding.
- Q8 group size: g32 vs g128 is 0.54% vs 0.68% for 0.375 bit/weight (4.6% more bytes). Q8 is
  not the bandwidth-critical choice, so g32 was kept (one layout and scale granularity shared
  with the activation blocks).
- mtp.fc and v_proj/o_proj are the most sensitive tensors; embed/head is average.
- Activation quantization (int8 per 32 columns, done by the kernels) adds a max output error
  of ~0.7-0.9% of max|y| on Gaussian test vectors (kbench `rel_err_fp`); for Q4 this is small
  next to the weight error, for Q8 it is comparable to the weight error.

### v6 — Q4K_G32 replaces Q4A_G64 (id 4)
Same code layout as Q4_G32; per superblock the 6-bit sub-scales/mins are decoded once
(UNPACK6 + widen + UCVTF + FMUL by the per-row fp16 D/M) into a 256 B stack table.
`kbench-v6-q4k_20261021_231035.json`:
- hot cycles/block/vector, n = 1 / 2 / 4 / 8: Q4 40.4 / 33.5 / 27.8 / 29.3;
  **Q4K 46.7 / 38.8 / 30.8 / 30.4**; Q8 41.2 / 34.8 / 32.7 / 31.3.
- cold n=1 GB/s, threads 1 / 2 / 4: Q4 2.62 / 3.83 / 3.66-3.92; **Q4K 2.32 / 3.83-3.90 /
  3.70-3.88**; Q8 4.19-4.24 / 4.13 / 3.93-4.03. => Q4K's extra decode is invisible from 2
  threads on (DRAM-bound), and it buys 11-22% lower weight error.

### v7 — two accumulators per row in the F32/BF16 n<=2 path (LOSER, reverted)
Idea: halve the FMLA dependency chains. `kbench-v7-float_20261021_231123.json`: BF16 hot kv
n=1 3.47 GB/s (v1: 3.72), cold 1 thread 3.35-3.43 (v1/v3: 3.0-3.6), F32 unchanged
(3.80-3.90). No gain; reverted to keep the code simple. BF16 on one core stays ~3.5 GB/s
(SHLL serializes with FMLA: 16 NEON ops per 64 B); 2 threads reach 3.75-3.87.

### v8 — quantized passes capped at 4 vectors (n = 8 -> 2 passes of 4)
`kbench-v8-pass4_20261021_231157.json` hot cycles/block/vector at n=8: Q4 kv 29.3 -> 27.9,
Q4K 30.4 -> 30.3, Q8 31.3 -> 30.8 (n=3 now has its own pass: Q4 29.2). Kept.
Dispatch of an empty job: 11 ns with a 1-thread pool (no workers to wake).

### v9 — final full sweep (all dtypes x 9 real shapes x threads 1-4 x n = 1, 2, 4, 8)
`bench/results/kbench-v9-final_20261021_232530.json` (709 configurations, 0 noisy, all
correctness checks and golden cases passed; threads forced exactly, `kern_auto_threads=0`).
Text tables: `logs/kbench_v9_gbs.txt`, `logs/kbench_v9_us.txt`. Pool dispatch of an empty
job: 11 / 238 / 369 / 478 ns for 1 / 2 / 3 / 4 threads (requirement: << 10 us).

### n = 1: microseconds per call (GB/s) for 1 / 2 / 3 / 4 threads

| dtype | shape | MB | 1 thr | 2 thr | 3 thr | 4 thr |
|---|---|---|---|---|---|---|
| f32 | qkv | 25.17 | 6614 (3.81) | **6531 (3.85)** | 6646 (3.79) | 6767 (3.72) |
| f32 | z | 8.39 | 2200 (3.81) | **2175 (3.86)** | 2204 (3.81) | 2280 (3.68) |
| f32 | out | 8.39 | 2295 (3.66) | **2247 (3.73)** | 2262 (3.71) | 2369 (3.54) |
| f32 | q | 16.78 | 4489 (3.74) | **4361 (3.85)** | 4457 (3.76) | 4498 (3.73) |
| f32 | kv | 2.10 | 550 (3.81) | **545 (3.85)** | 550 (3.81) | 586 (3.58) |
| f32 | gate | 14.68 | 3848 (3.81) | **3828 (3.83)** | 3873 (3.79) | 3954 (3.71) |
| f32 | down | 14.68 | **3867 (3.80)** | 3881 (3.78) | 3911 (3.75) | 4002 (3.67) |
| f32 | ab | 0.07 | 17.5 (3.74) | **17.4 (3.77)** | 17.4 (3.77) | 17.7 (3.70) |
| bf16 | qkv | 12.58 | 3604 (3.49) | **3251 (3.87)** | **3251 (3.87)** | 3346 (3.76) |
| bf16 | z | 4.19 | 1210 (3.47) | **1066 (3.94)** | 1080 (3.88) | 1118 (3.75) |
| bf16 | out | 4.19 | 1175 (3.57) | **1106 (3.79)** | 1121 (3.74) | 1143 (3.67) |
| bf16 | q | 8.39 | 2413 (3.48) | **2141 (3.92)** | 2174 (3.86) | 2206 (3.80) |
| bf16 | kv | 1.05 | 299 (3.50) | **267 (3.92)** | 268 (3.92) | 296 (3.54) |
| bf16 | gate | 7.34 | 2112 (3.48) | **1869 (3.93)** | 1902 (3.86) | 1943 (3.78) |
| bf16 | down | 7.34 | 2065 (3.55) | **1965 (3.73)** | 1970 (3.73) | 2115 (3.47) |
| bf16 | ab | 0.03 | 9.8 (3.34) | 8.7 (3.76) | 8.6 (3.82) | **8.5 (3.84)** |
| bf16 | head | 508.56 | 145962 (3.48) | **129122 (3.94)** | 130803 (3.89) | 132113 (3.85) |
| q8 | qkv | 6.68 | **1545 (4.33)** | 1610 (4.15) | 1656 (4.04) | 1697 (3.94) |
| q8 | z | 2.23 | **518 (4.30)** | 536 (4.16) | 554 (4.02) | 582 (3.83) |
| q8 | out | 2.23 | **523 (4.26)** | 537 (4.15) | 557 (4.00) | 575 (3.88) |
| q8 | q | 4.46 | **1031 (4.32)** | 1068 (4.17) | 1105 (4.03) | 1148 (3.88) |
| q8 | kv | 0.56 | **132 (4.23)** | 135 (4.13) | 141 (3.94) | 140 (3.99) |
| q8 | gate | 3.90 | **903 (4.32)** | 935 (4.17) | 966 (4.04) | 967 (4.03) |
| q8 | down | 3.90 | **912 (4.28)** | 937 (4.16) | 965 (4.04) | 966 (4.04) |
| q8 | ab | 0.02 | **6.5 (2.68)** | 6.8 (2.58) | 6.6 (2.64) | 6.9 (2.53) |
| q8 | head | 270.17 | **62529 (4.32)** | 64559 (4.18) | 66616 (4.06) | 67060 (4.03) |
| q4 | qkv | 3.54 | 1299 (2.72) | 927 (3.82) | **911 (3.88)** | 938 (3.77) |
| q4 | z | 1.18 | 435 (2.71) | 308 (3.83) | **303 (3.89)** | 320 (3.69) |
| q4 | out | 1.18 | 436 (2.70) | 305 (3.86) | **302 (3.91)** | 312 (3.79) |
| q4 | q | 2.36 | 869 (2.71) | 619 (3.81) | **608 (3.88)** | 632 (3.73) |
| q4 | kv | 0.29 | 111 (2.66) | 77.7 (3.80) | **76.7 (3.85)** | 78.5 (3.75) |
| q4 | gate | 2.06 | 760 (2.72) | 542 (3.81) | **531 (3.89)** | 533 (3.88) |
| q4 | down | 2.06 | 772 (2.67) | 537 (3.85) | **531 (3.89)** | 534 (3.87) |
| q4 | ab | 0.01 | 6.0 (1.54) | 5.2 (1.76) | 5.1 (1.82) | **5.0 (1.85)** |
| q4 | head | 143.03 | 52457 (2.73) | 37279 (3.84) | **36749 (3.89)** | 36989 (3.87) |
| q4k | qkv | 3.54 | 1509 (2.35) | **913 (3.88)** | 913 (3.88) | 954 (3.71) |
| q4k | z | 1.18 | 508 (2.32) | 309 (3.82) | **303 (3.89)** | 324 (3.64) |
| q4k | out | 1.18 | 503 (2.35) | 303 (3.89) | **302 (3.91)** | 315 (3.74) |
| q4k | q | 2.36 | 1006 (2.35) | **610 (3.87)** | 611 (3.86) | 638 (3.70) |
| q4k | kv | 0.29 | 128 (2.30) | 77.0 (3.83) | **76.9 (3.84)** | 77.7 (3.79) |
| q4k | gate | 2.06 | 881 (2.34) | 534 (3.86) | **533 (3.87)** | 537 (3.85) |
| q4k | down | 2.06 | 898 (2.30) | **528 (3.91)** | 532 (3.88) | 538 (3.84) |
| q4k | ab | 0.01 | 6.5 (1.42) | 5.3 (1.75) | 5.3 (1.74) | **5.0 (1.84)** |
| q4k | head | 143.03 | 60833 (2.35) | **36696 (3.90)** | **36691 (3.90)** | 37359 (3.83) |

### Batch scaling, 4 threads: microseconds per call for n = 1 / 2 / 4 / 8 (cost of n tokens relative to 1)

| dtype | shape | n=1 | n=2 | n=4 | n=8 | n=8 / n=1 |
|---|---|---|---|---|---|---|
| f32 | qkv | 6767 | 6817 | 6832 | 9228 | 1.36x |
| f32 | gate | 3954 | 3952 | 3987 | 5326 | 1.35x |
| f32 | down | 4002 | 3949 | 4324 | 6747 | 1.69x |
| bf16 | qkv | 3346 | 3400 | 3654 | 6642 | 1.99x |
| bf16 | gate | 1943 | 1980 | 2112 | 4044 | 2.08x |
| bf16 | down | 2115 | 2039 | 2813 | 4516 | 2.14x |
| bf16 | head | 132113 | 135270 | 145816 | 270935 | 2.05x |
| q8 | qkv | 1697 | 1754 | 1804 | 2181 | 1.28x |
| q8 | gate | 967 | 1011 | 1066 | 1391 | 1.44x |
| q8 | down | 966 | 1006 | 1041 | 1581 | 1.64x |
| q8 | head | 67060 | 69692 | 71350 | 95076 | 1.42x |
| q4 | qkv | 938 | 957 | 996 | 2063 | 2.20x |
| q4 | gate | 533 | 555 | 598 | 1219 | 2.29x |
| q4 | down | 534 | 567 | 646 | 1412 | 2.65x |
| q4 | head | 36989 | 37799 | 39350 | 82747 | 2.24x |
| q4k | qkv | 954 | 964 | 1037 | 2307 | 2.42x |
| q4k | gate | 537 | 559 | 607 | 1339 | 2.50x |
| q4k | down | 538 | 564 | 700 | 1512 | 2.81x |
| q4k | head | 37359 | 38086 | 41048 | 92937 | 2.49x |

### v9-auto / v10 — library thread policy (`--auto`, pool of 4)
`kbench-v9-auto_20261021_232945.json` (n=1 caps F32 2, BF16 2, Q8 1, Q4/Q4K 4) and
`kbench-v10-auto-cap3_20261021_233247.json` (Q4/Q4K cap 3): cap 3 is 3-7% faster than 4
(Q4 qkv 920 vs 952 us, q 623 vs 663 us) => final caps F32 2, BF16 2, Q8 1, Q4 3, Q4K 3.
Batched calls (n >= 2) always use all threads. Matrices below 32 KB per thread use fewer.

## Best kernel per dtype (n = 1, cold weights, v9; ceiling = memband 4.20 GB/s)
| dtype | bits/w | best threads | GB/s (range over real shapes) | % of memband | 1-thread GB/s | limit on 1 core |
|---|---|---|---|---|---|---|
| F32 | 32 | 2 (1 within 1%) | 3.73-3.86 | 89-92% | 3.66-3.81 | DRAM |
| BF16 | 16 | 2 | 3.79-3.94 | 90-94% | 3.47-3.57 | SHLL+FMLA compute |
| Q8_G32 | 8.5 | **1** | 4.23-4.33 | 101-103% | same | DRAM |
| Q4_G32 | 4.5 | 3 | 3.85-3.91 | 92-93% | 2.66-2.73 | NEON issue (~40 cyc/64 B) |
| Q4K_G32 | 4.5 | 2-3 | 3.84-3.91 | 91-93% | 2.30-2.35 | NEON issue (~47 cyc/64 B) |
(Q8 slightly above memband: memband's 256 MB single stream vs kbench's 64 MB rotation; both
are DRAM-bound reads.) Output head 248320x1024: Q4/Q4K 36.7 ms, Q8 62.5 ms, BF16 129 ms.

Projected matvec time per decoded token (sum of v10/v9 n=1 times, 18 linear-attention + 6
full-attention layers): Q4K layers 74 ms + Q4K head 37 ms = **~111 ms (~9.0 tok/s)**;
with a Q8 head 74 + 63 = ~137 ms (~7.3 tok/s). llama.cpp Q4_0 measured 5.6 tok/s
(`bench/results/llamacpp-qwen35-0.8b-Q4_0_20261021_213343.json`).

## What did not work (or did not matter)
- Software prefetch (PRFM, 256-2048 B ahead): no gain anywhere, -22% for Q8 on 1 thread.
  The A72 hardware prefetcher keeps up; PRFM costs load-port slots. Knob kept, default off.
- More threads for bandwidth: one core already saturates DRAM; 4 streams cost 3-10% vs
  the best count (Q8 4.33 -> 3.94 GB/s on qkv). Hence the per-dtype n=1 caps.
- Full unroll of the vector loop at nv = 8 and passes of 8 vectors: GCC spilled
  accumulators/unpacked weights (up to ~2800 stack ops); 2 passes of 4 are faster.
- Asymmetric 4-bit with g64 (Q4A_G64): same speed as Q4_G32 but higher error; replaced.
- Two accumulators per row for F32/BF16 (v7): no measurable gain; reverted.
- dequantize-to-fp32 + FMLA was not built: ubench shows FMLA at 4 MACs/cycle vs 8 for
  SMULL/SMLAL, and the conversion (SXTL x2 + SCVTF at 2+ cycles) would cost more than
  the int8 path's activation quantization (done once per call).
- int16 activations (SMLAL .4s, 4 MACs/cycle) would remove activation-quantization error
  but halve the MAC rate; not pursued (Q8 is DRAM-bound on 1 core either way, so it could be
  an accuracy option later if Q8 tensors prove sensitive).
- Huge pages are unavailable (stock kernel). TLB: 4 KB pages => one walk per 4 KB of a
  sequential stream (page-table lines are cached; ~1-2% of the stream time by estimate).
  The hardware prefetcher stops at page boundaries, which is the likely reason streaming
  plateaus at ~4.2 GB/s for both memband and the kernels.

## Remaining ideas (not done)
- Q4 batched path is at the NEON-issue model (~27 cycles per block per vector); further
  gains need fewer NEON ops: integer->float via magic-number add instead of SCVTF (-0.6
  cycle), signed nibble codes to drop the offset FMLS in batches. Expected <= 5-10%.
- n > 8: works (passes of 4 per tile) but re-reads all activations for every 4-row tile;
  for large prefill batches, call in chunks of <= 8 or add row-group blocking.
- int16-activation Q8 variant for accuracy-sensitive tensors (see above).

## Recommended dtypes per tensor type (speed from v9, error from the table above)
- Layer matrices (in_proj_qkv, in_proj_z, out_proj, q/k/o_proj, gate/up/down): **Q4K_G32**
  (4.5 bits, 7.4-7.8% error, same decode speed as Q4_G32 from 2 threads on).
- self_attn.v_proj (8.1% Q4K, the most sensitive layer type; only 6 x 0.5 MB): Q8_G32 costs
  +1.6 ms/token in total - worth it if perplexity shows sensitivity.
- linear_attn.in_proj_a / in_proj_b (16 x 1024, 32 KB each): BF16 or Q8 - bytes are negligible.
- embed_tokens (tied output head): the single biggest cost per token. Q4K = 37 ms/token,
  Q8 = 63 ms/token. Start with Q4K and compare perplexity against Q8 (llama.cpp's Q4_K_M
  keeps the output at higher precision for a reason).
- mtp.* (speculative head): Q8 for mtp.fc (Q4K 9.1%, Q4 11.7%), Q4K for the rest.
- Avoid F32/BF16 for large matrices (2-4x the bytes; only for debugging/reference).

---
## Part 2 (2026-10-22): end-to-end profile, bit-exactness, Q4K, the recurrence

### Where a decode step goes (PF_KERN_STATS matmul profile + gprof)
`engine/src/kern_stats.c`: with `PF_KERN_STATS=1` pf_matmul logs wall time per (shape, dtype,
n) and prints the table at exit (stderr, saved in the bench JSON). Engine binaries built from
my sources: `engine/pinfer_run_kern*` on the Pi (never the shared `pinfer_run_fast`).

Q4K-headq4k (the shipping model), 4 threads, 64-token prompt, 64 generated
(`bench/results/pinfer-kstats-q4k-plain_20261022_003014.json`,
`pinfer-kstats-q4k-spec3_20261022_003024.json`):

| | per plain decode step | per spec k=3 verify pass (n = 4) |
|---|---|---|
| layer matmuls | 80.3 ms | 92.6 ms (x1.15) |
| output head (248320 x 1024) | 37.2 ms | 41.3 ms (x1.11) |
| **matmul total** | **117.5 ms** | **133.9 ms (x1.14)** |
| everything else (scalar forward.c / forward_lin.c) | 27.5 ms | ~105 ms (4 tokens x ~26 ms) |
| step / pass total | 145 ms (6.89 tok/s) | 239 ms verify + 28.6 MTP + 23.1 commit |

So the batched matmuls already cost 1.14x a single vector (kbench's 1.10 plus the dirty-L2
effect below); the verify-pass ratio of 1.65 comes from the per-token work outside the
kernels, which batching does not amortise. Same picture for Q4 (`pinfer-kstats-q4-*`): matmul
121 ms of 146 per step; verify 230 ms = 133 matmul + ~97 other. The Q4K vs Q4 spec gap
(11.40 vs 12.38 tok/s) is 3.7% verify time (Q4K n=4 matmuls are 4% slower, as in kbench) and
5.3% fewer accepted tokens per pass (3.32 vs 3.50: different drafts) - not a kernel defect.

gprof of a single-threaded run (`-pg` build `engine/pinfer_run_pg`, `-t 1`, 64 prompt + 24
generated; `logs/runs/20261022_002035_run.log`): kern_q4 81.3%, **pf_linear_attn 10.2%
(14.3 ms per token: delta rule, conv, norms)**, expf 1.6% (2.3 ms/token), pf_full_attn 1.3%
(1.8 ms/token), silu/rmsnorm/quant_act ~1%. The delta-rule recurrence is the one item big
enough to matter: 18 layers x 16 heads x 2 passes over a 64 KB state per token.

Prefill (n = 64 in one chunk): matmuls 1.50 s of 3.1 s (Q4K: qkv 19.0 ms, gate/up 11.2,
down 12.9, out 7.1 per call at n = 64, i.e. ~35 cycles per block-vector vs 27 at n = 4: the
64 x 1 KB int8 activations no longer fit L1 and stream from L2 for every tile); the other
1.6 s is the same per-token scalar work (26 ms/token). Target 30 tok/s = 2.1 s per 64
tokens: not reachable from the matmul side alone.

### Pool spin (v10b, kept but no effect on the numbers)
Workers now spin up to 50 ms before the futex sleep (was 65k polls ~0.5 ms). Hypothesis was
that out_proj (1024x2048, 554 us in the engine vs 303 us in kbench) paid a wake-up after the
recurrence; `pinfer-kstats-q4-plain-spin_20261022_002034.json` shows out_proj unchanged
(614 us; that run was ~8% slower overall: 54 C and other agents' unlocked jobs). The real
cause is the recurrence leaving 1 MB of dirty state lines in L2 that are written back while
out_proj streams its weights (kbench `--dirty 1024` reproduces it, see below). That traffic
(18 MB/token) is unavoidable with an fp32 state; it belongs to the decode floor
(437 MB weights + 36 MB state = 473 MB/token = 113 ms at 4.2 GB/s; we are at 145).

### v11 - bit-exactness of pf_matmul(n) rows vs pf_matvec (REQUIRED by speculation)
Root cause: for Q4_G32 / Q4K_G32 the single-vector pass (nv == 1) accumulated the offset
term `o * sum(x)` in a separate register (`corr`, v2) and subtracted it at the end, while the
batched passes folded it into `acc` after each block - two different rounding sequences, so
batched verify logits differed from plain-decode logits in the last bits and an argmax
near-tie flipped a token (lead's chat27 prompt, mix-headq4k model). F32, BF16 and Q8 were
already identical (same instruction sequence per vector for every nv; the int8 activation
quantization is per vector and the integer dot products are exact).
Fix: every nv uses the fold-in form (`acc = fma(cvt(t), d*dx, acc); acc = fms(acc, o, fs)`).
kbench now checks, for every dtype x shape x threads x n, that row v of pf_matmul(n) equals
pf_matvec on vector v with memcmp (`exact` in RESULT; a mismatch is a failing check);
`engine/bench/spec_exact.sh` checks on the Pi that `--spec k` generates exactly the tokens
of `--spec 0` for both models x two prompts x k = 0..4.

### v12 - Q4K scale decode without UCVTF
`u8_scaled`: the 6-bit sub-scales are widened to u32, OR-ed with the exponent of 2^23 and
scaled with one FMA against (D, -2^23 D) instead of UCVTF (2.3 cycles) + FMUL. Exact (6-bit
value x 11-bit fp16 mantissa fits a float mantissa). Saves ~20 cycles per superblock of the
~115 the decode cost; the batched Q4K block work itself (~27 cycles per block-vector:
16 SMULL/SMLAL, 4 reduction ops, SCVTF, FMUL, FMLA, FMLS, 2 loads) is at the instruction
floor of this format - per-32-column activation scales force a float conversion per block.
A k-quant style integer path (t * s6 accumulated in int32, one SCVTF per superblock) needs a
per-256-column activation scale (llama.cpp's Q8_K) and would gain ~6%; not done because it
changes the activation numerics the PPL was measured with.

### Recurrence kernel `engine/src/kern_lin.c` (not wired in: needs engine-core files)
`kern_lin_rule`: the delta rule of one layer over n tokens, NEON over the 128 value columns
in chunks of 32 (kv / o accumulators stay in registers across the 128 key rows), threaded
over the 16 heads through a new generic pool job (`kern_pool_run`). `kern_lin_apply`: the
commit replay, threaded the same way. Both keep the scalar code's exact per-element
operations (separate multiply and add, no FMA; per-column accumulation order over i), so the
state, outputs and rollback log are bit-identical to forward_lin.c - `engine/bench/lbench.c`
checks this (in-place, logged-on-scratch, and commit replay for n = 1 / 4 / 64; passed under
qemu and on the Pi) and times both. Integration = `engine/bench/lin_kernel.patch` (ops.h
gains `pf_lin_rule` / `pf_lin_apply`, matvec_ref.c receives the scalar code moved out of
forward_lin.c, forward_lin.c calls the contract); applied to a scratch copy to measure
(`engine/pinfer_run_lin` on the Pi); `tests/test_golden.py` and `tests/test_spec.py` run
against that copy's host build.

### Results (2026-10-22 00:40-00:50)
- **Exactness sweep** `bench/results/kbench-v11-exact_20261022_004156.json`: all 5 dtypes x
  10 shapes (incl. head 248320 and draft head 32768) x threads 1,4 x n = 1,2,3,4,5,8: every
  configuration `exact` (memcmp of pf_matmul rows vs pf_matvec), all error checks pass.
- **Cost of the fix** (hot, 1 thread, `kbench-v11-foldin-cost_20261022_004219.json`, old =
  kbench-v10 with `corr`): Q4 n=1 out 442 -> 424 us, kv 107 -> 100; Q4K n=1 out 514 -> 503,
  kv 125 -> 123; n=4 Q4 1214 -> 1209, Q4K out 1386 -> 1305 (v12 scale decode). The fold-in
  form is slightly *faster* (one register less in the unrolled n=1 loop). No speed cost.
- **Spec vs plain tokens on the Pi** (`engine/bench/spec_exact.sh`, 40 tokens, k = 0..4):
  old shared `pinfer_run_fast` (`spec-exact-old_20261022_004457.json`): mix-headq4k /
  chat27 mismatches for every k >= 1 (plain 0d45f4c6, spec c4644585), the other 3 model x
  prompt pairs match. Fixed kernels (`spec-exact-lin_20261022_004711.json`, binary with the
  recurrence patch): **20/20 match**; the plain output of mix/chat27 is now c4644585 (the
  n=1 path rounds like the batched path did, so the near-tie resolves the other way).
- **Dirty-L2 experiment** `kbench-v11-dirty_20261022_004759.json` (Q4K, 3 threads, n=1):
  writing 1 MB before every call (memset itself ~208 us, measured on the 16-row shape)
  turns out 1024x2048 from 337 us into 615 - 208 = ~407 us and down 1024x3584 from 534 into
  ~637 us: the write-back of 1 MB of dirty lines costs the next streaming matmul ~20-25%.
  In the engine out_proj follows the recurrence (1 MB dirty state) and measures 554-614 us.
- **Recurrence kernel, first version (token-outer)** `lbench-v1_20261022_004227.json`,
  ms per token for all 18 layers, cold state: scalar 1 thread n=1 20.7 / n=4 13.5 /
  n=64 11.7; NEON 1 thread 24.0 / 13.5 / 10.3; NEON 4 threads **13.7 / 6.2 / 4.4**.
  n=1 is DRAM-bound (36 MB of state traffic per token; 1 thread NEON is even slower than
  the scalar full-row sweep because the 128 B column-chunk reads stride through the rows);
  n >= 4 was L2-bandwidth-bound (256 KB of L2 traffic per head-token), hence v2 below.
- **Patched scalar build (host)**: `tests/test_golden.py` PASS (A vs B 0.0 error, greedy
  32/32), `tests/test_spec.py` PASS (all prompts, k = 0..4 and k3/v32768 exact);
  `logs/host_tests_lin_patch.txt`.

### Recurrence kernel v2 - column chunks outer, tokens inner (L1-resident chunk)
kv, delta and o are per value column, so S can be processed in 32-column chunks (16 KB)
with the chunk loop outside the token loop: for n > 1 a chunk stays in L1 for all tokens and
both passes. n = 1 keeps 64-column chunks (256 B runs per row for the DRAM stream). Still
bit-identical (per-column operation order unchanged). The commit replay is chunked the same
way. Numbers: lbench-v2 (below).

### Batched activations in groups of 16 (v13, prefill)
pf_matmul splits n > 16 into groups of 16 vectors; each group re-streams the weights but its
int8 activations (16 x cols B) stay in L1 across the tiles. At n = 64 the un-grouped kernel
re-read 64 KB of activations from L2 per 4-row tile (35 cycles per block-vector instead of
27). Per-vector arithmetic unchanged (exactness checked up to n = 33 under qemu, n = 64 on
the Pi). Numbers: kbench-v13-group (below).

### Results v2 / v13 (2026-10-22 00:50-00:56)
- **Grouping (v13)** `bench/results/kbench-v13-group_20261022_005221.json`, Q4K, 4 threads,
  ms per call at n = 64 for group 0 / 8 / 16 / 32: qkv 18.8 / 17.6 / **17.0** / 18.2;
  gate 11.0 / 10.4 / **9.9** / 11.0; down 13.0 / **11.0** / 11.9 / 11.8; out 6.9 / 6.6 / 6.7 /
  6.5. Group 16 kept (default): -9% on the big prefill matmuls, n = 4 unchanged (no
  grouping below 16). At n = 64 the kernel now runs at the n = 4 rate (~31-33 cycles per
  block-vector with 4 threads).
- **Recurrence v2 (column chunks, [i][j] layout)** `lbench-v2-chunk_20261022_005246.json`:
  4 threads n = 1 / 4 / 64: 12.6 / 5.7 / 4.6 ms per token (v1: 13.7 / 6.2 / 4.4). Still far
  from compute-bound at n >= 4 (should be ~1 ms/token at n = 64): the 512 B row stride puts
  a 16 KB chunk on 64 of the 256 L1 sets (2-way) -> L1 conflict misses, L2-bound. Hence the
  chunk-major state layout (v3, below).
- **End-to-end, Q4K-headq4k, 4 threads, 64 prompt / 64 generated**
  (`pinfer-kstats-q4k-*_20261022_0030*.json` before; `pinfer-kern4-q4k_20261022_005307.json`
  = kernels only (fix + v12 + v13); `pinfer-lin3-q4k_20261022_005*.json` = kernels +
  recurrence v2 patch):

| | before | kernels only | + recurrence v2 |
|---|---|---|---|
| plain decode tok/s | 6.89 | 6.91 | **7.13** |
| prefill tok/s (plain / spec run) | 20.7 / 19.5 | 18.9 / 20.7 | 21.8 / 24.6 |
| spec k=3 dv=32768 tok/s | 11.40 | 11.59 | **13.02** |
| verify ms per pass (n = 4) | 239.1 | 234.0 | **209.7** |
| MTP / commit ms per pass | 28.6 / 23.1 | 28.8 / 23.1 | 28.8 / **16.3** |
| tokens per pass | 3.316 | 3.316 | 3.316 |
| spec == plain tokens (20 configs) | 19/20 (old binary) | 20/20 | 20/20 |

### Recurrence v3 - chunk-major state layout (final) `lbench-v3-chunkmajor_20261022_005924.json`
ms per token for all 18 layers, cold state: 1 thread NEON n=1 / 4 / 64: 16.7 / 10.3 / 8.6
(scalar row-major: 17.6 / 13.1 / 11.8); **4 threads: 12.7 / 4.0 / 2.2** (v2: 12.6 / 5.7 /
4.6). n = 1 stays DRAM-bound (36 MB of state per token); n >= 4 is now compute/L2-bound at
the expected rate. Bit-exact against the scalar reference through a layout conversion
(lbench converts [i][j] <-> [j/32][i][j%32] before comparing).

### Final end-to-end (`pinfer_run_lin4` = fixed kernels + v12 + v13 + recurrence v3 patch)
`bench/results/pinfer-lin4-q4k_20261022_010*.json`, Q4K-headq4k, 4 threads, prompt64:

| | before (`pinfer-kstats-q4k-*_0030*`) | final | change |
|---|---|---|---|
| plain decode | 6.89 tok/s (145 ms) | **7.23 tok/s** (138 ms) | +5% |
| prefill, 64 tokens | 20.7 tok/s | **27.0 tok/s** | +30% |
| spec k=2 dv=32768 | (11.0 on Q4) | **12.12 tok/s** | |
| spec k=3 dv=32768 | 11.40 tok/s | **13.42 tok/s** | +18% |
| verify pass n=4 / n=3 | 239.1 ms / - | **204.4 / 183.8 ms** | |
| verify / plain-step ratio, k=3 / k=2 | 1.65 / - | **1.48 / 1.33** | |
| MTP / commit per pass (k=3) | 28.6 / 23.1 ms | 28.6 / **14.1** ms | |
| spec == plain tokens, 20 configs | 19/20 | 20/20 | |

Composition of the final n = 4 verify pass (204 ms): matmuls 134 ms (layers 92 + head 42;
1.13x the single-vector 119 ms), recurrence ~16 ms (4 x 4.0), **remaining scalar code
~55 ms (4 x ~14 ms: conv_step with 6144 expf per layer, l2norm, gated norm + silu, the MLP
silu*up (86k expf per token), full attention, argmax)**. The 1.15x target (159 ms) needs that
last part down to ~6 ms per token; it lives in forward.c / forward_lin.c (engine-core).
Prefill (27 tok/s = 2.37 s): matmuls 1.58 s (n = 64 at the n = 4 rate), recurrence 0.14 s,
remaining scalar ~0.65 s; 30 tok/s needs ~0.25 s more from the same scalar code.
