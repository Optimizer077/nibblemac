# Engine core lab notebook (forward pass outside the matmul kernels)

Owner: engine-core. Files: `engine/src/{forward.c,forward_lin.c,forward_ops.c,forward_stats.c,
forward.h,pinfer.c,model.c,matvec_ref.c}`, `engine/Makefile`, `engine/tools/pinfer_run.c`,
`engine/bench/core2_check.sh`, `tests/test_golden.py`.

## 1. Scalar reference (2026-10-21)
Correct forward pass against the fp32 transformers reference: worst per-layer hidden-state
relative error 2.87e-6, logits 1.65e-6, greedy 32/32 (`tests/test_golden.py`). Pi, scalar
`matvec_ref`, bf16: prefill 2.0 tok/s, decode 1.55 tok/s
(`bench/results/pinfer-ref-bf16_20261021_214710.json`).

## 2. Vectorizing and threading the per-token work (2026-10-22, "core2")

### Starting point (kernels agent, KERNELS.md Part 2)
Q4K-headq4k, 4 threads, `pinfer_run_fast`, prompt64, 64 generated
(`bench/results/pinfer-final-q4k_20261022_011230.json`): plain decode 7.26 tok/s (138 ms/step:
matmul ~119, recurrence kernel ~7, **~14 ms scalar forward code**); spec k=3 verify pass n=4
204 ms; verify / plain ratio 1.48; prefill 26.1 tok/s. Per token the scalar code called libm
`expf` ~245k times (conv SiLU 18 x 6144, gated-norm SiLU 18 x 2048, MLP SiLU 24 x 3584,
attention gate 6 x 2048, softmax), each with a scalar division.

### Rules kept throughout
- Every operation gives bit-identical results whether a call processes 1 or n rows, and the
  NEON body of every primitive is bit-identical to its scalar mirror (`-DPF_NO_NEON`):
  reductions use 4 or 16 interleaved partial sums combined as (s0+s1)+(s2+s3), no fused
  multiply-add anywhere (`-ffp-contract=off` is now explicit in the Makefile; `-std=c11`
  already implied it), threads only split independent rows / heads / channel ranges.
- Verified on the Pi after every round with `engine/bench/core2_check.sh`: md5 of the
  prompt64 hidden states and logits for batched prefill (`-b 64`), token-by-token (`-b 1`) and
  the no-NEON twin build, for the Q4K and the bf16 model: all six files identical per model,
  and identical across all three rounds (`fe836caf.../bb05dee8...` Q4K, `b1aaa505.../2f26639b...`
  bf16). Golden prompt on bf16: greedy 32/32 on the Pi.

### Round 1: NEON primitives (`forward_ops.c`)
| primitive | used by | note |
|---|---|---|
| `pf_expf` / `expq` | all SiLU, sigmoid, softmax | Cephes-style: n = rint(x log2 e), r = x - n ln2 (hi/lo), degree-6 polynomial, 2^n from exponent bits; clamp [-87, 88] |
| `pf_silu_mul`, `pf_mul_silu`, `pf_mul_sigmoid` | MLP, gated norm, attention gate | `vdivq_f32` (exact IEEE division); two exp chains per iteration since round 2 |
| `pf_rmsnorm`, `pf_l2norm_scale` | every norm; q/k L2 norm + 1/sqrt(dk) | sum of squares in 4 lanes, scalar rsqrt once per row |
| `pf_conv_silu` | conv1d (K = 4) + SiLU + state shift, channel range | `vld4q_f32` deinterleaves the [C][4] taps |
| `pf_rope`, `pf_dot`, `pf_softmax_exp`, `pf_attn_pv` | full attention | see round 3 for P.V |
| `pf_argmax` | every argmax | max via `vmaxq`, then first index equal to it |

The 16 per-layer `beta` / `decay` scalars keep libm (`expf`, `log1pf`).

exp accuracy (scratchpad `core2/exptest.c`, 1M points per range; identical numbers on the
host scalar mirror and on the Pi NEON path, 0 scalar-vs-vector bit mismatches):

| range | ours max rel (ulp) | libm expf max rel (ulp) | ours == libm bits |
|---|---|---|---|
| [-87, 88] | 7.80e-8 (0.95) | 5.95e-8 (0.50) | 90.1% |
| [-20, 20] | 8.07e-8 (0.97) | 5.95e-8 (0.50) | 90.1% |
| [-1, 1] | 6.81e-8 (0.82) | 5.96e-8 (0.50) | 90.6% |

SiLU max rel 1.68e-7, sigmoid 1.44e-7 on [-40, 40]. Golden (host): worst hidden rel max
err 3.19e-6 (was 2.87e-6), logits 2.30e-6 (was 1.65e-6), top-1 10/10, greedy 32/32.

Round-1 Pi numbers (`pinfer-core2-q4k_20261022_014330.json`): plain 7.58 tok/s, prefill 33.2,
spec k=3 15.00 tok/s, verify 179.3 ms, ratio 1.36.

### Profile tool: `PF_FWD_STATS=1` (`forward_stats.c`)
Per phase and call kind (n=1 / batched / logged verify pass) like the kernels' `PF_KERN_STATS`.
Round-1 profile, per plain decode step: matmul 118.3 ms, recurrence kernel 6.8, **mine 8.4**
(attention 3.8, conv+l2norm+gates 2.2, MLP SiLU 1.16, gated norm 0.6, norms/add 0.3, argmax
0.24). Per verify pass (n=4, 19 passes): matmul ~129, recurrence 23 (incl. its scratch copy
of the 18 MB state), log copies 1.3, **mine 28** (attention 11.7, conv 7.8, MLP 4.3, gated
norm 2.2). Two findings: the attention was DRAM-latency-bound (KV rows interleaved
`[pos][group*hd]`: 1 KB runs with 1 KB gaps, re-fetched every step because the weight stream
evicts L2) and the SiLU loops ran ~20 cycles per element (one exp + divide dependency chain).

### Round 2: threading, KV layout, unrolling
- `ops.h` (+ `matvec_ref.c` serial, `matvec_fast.c` -> `kern_pool_run`): **`pf_pool_run`**,
  a generic parallel job. Used for: attention heads (8 heads over 4 threads), conv channel
  ranges (bounds at multiples of lk_dim so each q/k head's L2 norm follows in the same
  thread), gated-norm heads, MLP activation slices. ~66 extra dispatches per step.
- KV cache layout `[n_kv][max_ctx][hd]` (one contiguous stream per group); `att` scratch per
  head. The layout is private to `pf_full_attn` (mtp.c only allocates its cache).
- exp loops two vectors per iteration.
Pi (`pinfer-core2-q4k_20261022_015645.json`): plain 7.87, prefill 37.3, k=3 16.40 tok/s,
verify 162.5 ms, ratio 1.28. Profile per plain step: mine 3.5 ms (attention 1.4, conv 1.1,
MLP 0.32, gated norm 0.23, argmax 0.24, norms 0.25); per verify pass mine 9.5 (attention 4.0,
conv 2.4, log copies 1.3, MLP 0.9, gated norm 0.65, norms 0.4).

### Round 3: attention inner loops measured in isolation (scratchpad `core2/abench2.c`)
One head, hd 256, 100 rows, Pi, per head (cold = a 4 MB sweep before the section):

| loop | warm | cold |
|---|---|---|
| QK, `pf_dot` row by row | 16.3 us | 24.0 us |
| QK, two rows per q load (`pf_dot2`, tried) | 19.0 | 47.3 (two interleaved streams defeat the prefetcher) — reverted |
| P.V column-chunk-outer (round 1/2: 16 sweeps over the rows at 1 KB stride) | 29.2 | 44.3 |
| P.V row blocks of 8 x 16 columns | 20.9 | 42.4 |
| P.V row blocks of 4 x 32 columns | 21.2 | 32.3 |
| **P.V row-outer, o (1 KB) in L1, v one sequential stream** | **17.1** | **23.8** |

All variants bit-identical (same per-element order over the rows). Row-outer P.V is the
final `pf_attn_pv`.

### Final (`pinfer_run_core2`, Q4K-headq4k, 4 threads, prompt64, 64 generated, warm-up on)
`bench/results/pinfer-core2-q4k_20261022_020959.json` (valid, 49-52 C):

| | baseline `pinfer-final-q4k_20261022_011230` | final core2 | target |
|---|---|---|---|
| plain decode | 7.26 tok/s (137.7 ms/step) | **7.81 tok/s** (128.0 ms) | >= 7.8 (met; run-to-run 7.81-7.87) |
| prefill 64 tokens | 26.1 tok/s | **37.4 tok/s** | >= 30 |
| spec k=2, dv 32768 | 12.13 tok/s | **14.35** | |
| spec k=3, dv 32768 | 13.35 tok/s | **16.74** | |
| spec k=4, dv 32768 | 11.98 tok/s | **15.10** | |
| verify pass n=4 (k=3) | 205.8 ms | **158.8 ms** | |
| verify / plain-step ratio, k=3 | 1.49 | **1.24** | <= 1.25 (met; previous run 1.28) |
| MTP per pass, k=3 | 28.5 ms | 25.1 ms | |
| commit per pass (kernels) | 14.1 ms | 14.2 ms | |
| peak RSS | 432.9 MB | 432.9 MB | |

Exactness on the Pi (`spec_exact.sh`, 2 models x 2 prompts x 5 configs): 20/20 after every
round (`spec-exact-core2_20261022_014236/015951/...json`). Host: `tests/test_golden.py` PASS,
`tests/test_spec.py` PASS (63/63 runs, MTP head rows <= 2.8e-6) after every round.

Final profile (`PF_FWD_STATS=1 PF_KERN_STATS=1`, same runs; the profiled runs are ~2% slower
than the clean bench):

| per plain decode step (66 steps) | ms | per verify pass n=4 (19 passes) | ms |
|---|---|---|---|
| matmul (KSTATS n=1) | 118.6 | matmul (KSTATS n=4, minus ~2.5 of MTP 4-row passes) | ~129 |
| recurrence kernel `pf_lin_rule` | 6.5 | recurrence kernel (incl. scratch copy of the state) | 22.8 |
| attention (6 layers) | 1.10 | attention | 2.87 |
| conv + L2 norm + gates (18) | 1.05 | conv + L2 norm + gates | 2.37 |
| argmax | 0.25 | log copies (conv state + raw inputs) | 1.29 |
| norms + residual adds | 0.27 | MLP SiLU | 0.86 |
| MLP SiLU (24) | 0.32 | gated norm | 0.65 |
| gated norm (18) | 0.21 | norms + residual adds | 0.38 |
| **step** | **128.0** (bench) | **pass** | **158.8** (bench) |

The recurrence kernel (22.8 ms per pass, roughly 9 of it the scratch copy of the 18 MB
state) and the 14 ms commit replay are now the largest non-matmul items of speculation and
belong to `kern_lin.c`; my remaining share is ~7 ms per pass and ~3 ms per plain step.
Ideas not done: avoid the conv-state copy in logged passes by convolving from
[state | raw inputs] without shifting (~0.4 ms per pass); fp16 KV cache (halves the 400 KB
per layer the attention re-fetches from DRAM every step); threading the argmax.

### `pinfer_run --bench` warm-up (lead's request)
`--warmup` / `--no-warmup` (default: on with `--bench`): an untimed prefill + 4 tokens in the
same mode, then `pinfer_reset`. The first prefill after a reboot pages the mmap'd weights in
from the SSD (20 vs 27 tok/s measured by the lead); the stats line now prints the warm-up
time (2.1-2.4 s on a cold cache).
