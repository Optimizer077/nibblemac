# Deeper Mac optimization: cooperative Metal and native I8MM

2026-10-28, same M2 Pro / 10 CPU cores / 16 GPU cores / 16 GiB as the
[public runtime comparison](PUBLIC_BASELINES.md). Official Qwen3.5-0.8B,
unchanged PQ Q4K/A8 weights, fp32 KV/state, no precision or fast-math change.

## Outcome

Two actual implementations, not just compiler flags:

1. **CPU + Metal row partitioning:** submit a small Q4K vocabulary-head prefix
   to Metal, compute the disjoint suffix with the NEON pool, then join. This
   removes most of the penalty of full head offload, but **does not establish
   a consistent win over CPU-only**.
2. **Native I8MM CPU build:** use integer `SMMLA` on pairs of token vectors for
   Q4/Q4K batched matmuls. Exact, but its current packing/layout implementation
   is **slower at prefill than SDOT**. It is a separately built experiment.

**Recommendation remains CPU SDOT.** Both experiments stay opt-in. Nothing here
establishes a new speedup over llama.cpp/MLX, whose baselines were measured in a
different session with different quantizers. No fully GPU-resident model claim.

## 1. Profile first

The fresh phase profile found the 248,320 x 1,024 vocabulary projection took
about **1.395 ms/call on CPU vs 6.077 ms/call with full Metal head offload**.
These counters include warmup and the final prefill projection, not just timed
decode. They identify a bottleneck, not an isolated laboratory estimate.

Evidence: [deep profile](../bench/results/metal/deep-profile-20261028T074823.054654Z.json).
Other small model matrices remained faster on CPU; broad per-operation GPU
offload had already failed in [the graph experiments](METAL_GRAPHS.md).

## 2. Cooperative vocabulary projection

For an eligible single-vector Q4K matmul:

1. Quantize the activation **once**, using the inherited quantizer.
2. Pick `G = floor(rows * percent / 100 / 32) * 32` GPU prefix rows.
3. Cache/upload only that immutable prefix and submit an asynchronous command.
4. The existing CPU pool computes `[G, rows)`, with its unchanged kernel.
5. Wait for Metal completion, copy the prefix output, and return synchronously.

For the real vocabulary head, 10% is **24,832 GPU rows** / 223,488 CPU rows.
GPU prefix weights are about **14.3 decimal MB**, not a second full head. CPU
and GPU outputs are disjoint. Every row retains its original integer dot and
chronological explicit FMA/correction order; no floating cross-device reduction.

The public Metal `begin`/`finish` pair enables CPU work between submission and
the join. Submission order allows overlap; no hardware timeline trace proving
the amount of actual overlap was captured. GPU faults are fatal/reported, never
silently completed on CPU. Scratch/input lifetimes extend through the join.
Reset/save/load/rewind see a completed call, not in-flight GPU state.

`NM_METAL_SPLIT_PERCENT=0` keeps the previous full-offload behavior. Nonzero
values must be 1..99. Only eligible `n=1` calls are partitioned; batching, an
empty aligned prefix and other dtypes retain their existing path. The ordinary
weight-size threshold still decides whether the matrix reaches Metal at all.

This adapts an established technique to this Mac backend. The inherited Pi
engine also had a row-splitting path; **no claim to inventing CPU/GPU row
partitioning**. The Mac implementation uses exact A8 kernels, Metal submission
lifetimes and the macOS worker pool, not the Pi device backend.

### Same-sweep comparison with full offload

Three rounds, 128 IDs / 127 timed steps, six CPU threads, head-only threshold
64 MiB, packed kernel, graph flags off. Decode tok/s: **median [min–max]**.

| Configuration | Prose 64 | Chat 36 | Repeated prose 512 |
|---|---:|---:|---:|
| CPU SDOT | 186.9 [140.9–192.0] | 195.7 [185.2–201.2] | 185.3 [158.1–186.7] |
| Previous full Metal head offload | 114.6 [103.2–116.6] | 115.5 [111.6–120.7] | 109.1 [108.0–112.8] |
| Cooperative 10% GPU / 90% CPU | 187.9 [98.1–194.3] | 176.7 [158.2–190.2] | 158.8 [134.6–168.4] |
| Cooperative 20% GPU / 80% CPU | 172.5 [149.8–179.7] | 157.8 [140.3–178.9] | 169.1 [168.7–169.4] |

10% partitioning improves observed medians **45–64% over the slower full-head
Metal baseline**, not over CPU. One severe outlier remains in the 10% prose
sample. All candidates, including slower splits, are retained. Source:
[36-run split sweep](../bench/results/metal/split-final-20261028T075335.498550Z.json).

Median peak RSS across prompts: CPU **435–448 MiB**, full head offload
**583–596 MiB**, cooperative 10% **460–472 MiB**, cooperative 20% **473–486 MiB**.
These are process high-water RSS values, not total GPU/system memory usage.

### Later repeated control, not substituted into the earlier sweep

The final 27-run sweep compared SDOT, the I8MM build, and the 10% cooperative
path. The controls ran faster and mostly more consistently than the earlier
sweep; cause unconfirmed. Do not combine the newer cooperative numbers with
the older full-offload baseline to manufacture a larger percentage gain.

| Configuration | Prose 64 decode | Chat 36 decode | Context 512 decode |
|---|---:|---:|---:|
| CPU SDOT | 223.0 [201.1–223.3] | 220.0 [216.8–224.3] | 210.8 [208.9–213.0] |
| CPU I8MM experiment | 218.5 [216.6–222.2] | 224.8 [223.5–226.0] | 210.5 [208.2–212.1] |
| Cooperative 10% GPU | 216.1 [212.2–222.1] | 221.5 [220.1–222.2] | 202.7 [198.2–205.3] |

No consistent CPU+GPU win: prose and the 512-token case remain slower than CPU.
Shared bandwidth/cache contention and submission costs are plausible
explanations, **not isolated causes established by these measurements**.
Source: [final controls](../bench/results/metal/deep-final-20261028T080635.587463Z.json).

## 3. Actual I8MM / SMMLA experiment

`sysctl hw.optional.arm.FEAT_I8MM` and `FEAT_DotProd` both report 1 on this M2 Pro.
The optional `cpu-i8mm` build requires compile-time Apple/I8MM support and emits
actual `SMMLA` instructions (verified in the built kernel object). It uses public
ARM Advanced SIMD intrinsics, **not undocumented Apple AMX APIs or Metal**.

Each instruction multiplies two 8-byte weight rows by two 8-byte activation
columns into four int32 dot sums. Four consecutive eight-column pieces complete
each 32-column quantization block. Lane rearrangement recovers four row values
per token; scale and offset FMAs run in the same order as the CPU oracle. Integer
dots cannot overflow here (`32 * 15 * 127 = 60,960`). Odd token tails and `n=1`
retain SDOT. Q8 also retains its prior path.

Initial grouping by four token vectors was slower, about 471–532 prefill tok/s;
its raw [tuning record](../bench/results/metal/i8mm-tuning-20261028T080045.012496Z.json)
and earlier quality gates remain. Restoring the same seven-vector grouping used
by SDOT improved it, but not enough. Final prefill tok/s, median [min–max]:

| Configuration | Prose 64 | Chat 36 | Context 512 |
|---|---:|---:|---:|
| CPU SDOT | 846.1 [607.7–846.3] | 815.9 [794.9–826.6] | 802.8 [796.3–803.9] |
| CPU I8MM, group 7 | 710.1 [706.5–710.2] | 690.4 [685.7–697.1] | 674.7 [659.2–681.4] |
| Cooperative Metal, CPU prefill unchanged | 849.1 [845.4–853.6] | 823.6 [822.3–826.5] | 803.5 [799.1–806.8] |

I8MM prefill is about **15–16% slower by these medians**. In-loop operand
rearrangement/packing, reuse and code generation need work; merely using a more
specialized instruction is not an optimization result. Small decode differences
must not be attributed to SMMLA: single-token decode still uses SDOT.

## Correctness and failure behavior

- **88 primitive I8MM/no-SDOT checks:** Q4/Q4K, batch 1/2/3/4/5/6/7/8/31/64,
  full/slab outputs, row/column shapes, zero inputs and saturated A8 values.
- All four Metal variants pass both synchronous **and asynchronous** exact A8
  tests, now including three additional invalid submission cases, plus the
  existing MLP and persistent recurrent-state graph tests.
- Final synthetic full-model checks: **126 cases** across Q4K/Q4/Q8 with the
  optional I8MM build and 10% cooperative mode; another **42 Q4K cases at 99%**
  exercise the small CPU suffix. Plain cases compare all dumped hidden states
  and logits; plain/speculative cases compare tokens. Eligible split fixtures
  must report nonzero `split_calls`, not silently skip the new path.
- Final real-model gates pass exact prefill logits/tokens, sequential/batched
  consistency, CPU/GPU saved-state bytes, both restore directions, rewind, fixed
  and adaptive speculation for CPU, cooperative Metal and the I8MM build.
- WikiText-2 **18.5061** for original no-SDOT, SDOT, cooperative Metal and I8MM:
  eight independent 512-token chunks / 2040 scored tokens. Batched PPL uses the
  pre-existing non-split GPU route; the split's single-token output is separately
  covered by decode/logit/state/speculation gates. This is parity with the same
  quantized oracle, not a new unquantized-teacher evaluation.

Actual-device evidence under `bench/results/metal/`:

- `graph-kernel-gates-20261028T080337.792714Z.json`: async/graph/I8MM primitive gates.
- `model-gates-20261028T081004.372451Z.json`: Q4K, 10%, final group-7 I8MM.
- `model-gates-20261028T081008.895357Z.json`: Q4, 10%.
- `model-gates-20261028T081016.425325Z.json`: Q8, 10%.
- `model-gates-20261028T081021.678560Z.json`: Q4K, 99%.
- `real-gates-20261028T080754.353751Z.json`: final state/quality gates, group 7.
- `real-gates-20261028T081919.481010Z.json`: split plus both existing graph
  features, exact state/logit/token/speculation integration checks; quality skipped.

The original four-vector I8MM run/gates and first two-chunk split quality run are
also retained. Measurement-time source/binary hashes identify each version.

## Next deeper optimization proposals

These are **proposals**, not implemented gains or claims of novel algorithms:

1. **Measure the GPU dot kernel and overlap boundary separately.** Capture CPU
   submit/suffix/join time and GPU timestamps before more partition tuning.
   The large head kernel itself remains slow; scheduling alone cannot fix that.
2. **Exact tiled GPU prefill.** Reuse weights and activations across more output
   rows/tokens. Evaluate SIMD-group matrix operations only if integer block sums
   can be proven exact; preserve the original per-32-column scale/FMA sequence.
   Do not silently dequantize the whole operation to a different fp16 recipe.
3. **Prepacked native CPU tiles.** Move I8MM operand rearrangement out of the
   inner loop, with a bounded immutable layout cache; compare against SDOT on
   identical hardware, weights and prompts. Account for memory and packing cost.
4. **Coarse GPU execution graphs.** Extend resident intermediates from the
   existing MLP/DeltaNet work to normalization, projection, attention/KV,
   residuals and complete layers. Preserve the CPU reduction/FMA order where
   exact speculative verification depends on it. Aim for fewer coarse submits,
   not more tiny GPU calls. This requires new operator/model/state quality gates.
5. **Only then retune CPU/GPU scheduling.** Choose CPU decode vs GPU prefill by
   observed shape-dependent cost; keep the fast CPU path when offload loses.
   Do not infer all-request latency or public-runtime superiority from a kernel
   microbenchmark, or from a different-quantization comparison.

Any changed precision or fast-math policy would be a separate explicit quality
experiment, not an unannounced optimization. Stop promoting candidates when
they fail performance gates; keep the negative evidence.

## Reproduce

```sh
make mac-cpu-baseline mac-cpu mac-infer mac-cpu-i8mm
python3 tools/check_graphs_macos.py --i8mm
NM_METAL_SPLIT_PERCENT=10 .venv/bin/python tests/test_macos_inference.py --i8mm
.venv/bin/python tests/test_real_macos.py --split-percent 10 --i8mm --quality-chunks 8
.venv/bin/python tools/bench_inference_macos.py --config cpu:6 \
  --config metal:6:67108864:packed --config metal:6:67108864:packed:none:10 \
  --config metal:6:67108864:packed:none:20 --rounds 3 --generate 128 --label split-repeat
.venv/bin/python tools/bench_inference_macos.py --config cpu:6 --config cpu-i8mm:6 \
  --config metal:6:67108864:packed:none:10 --rounds 3 --generate 128 --label deep-repeat
```

For direct CLI use, `NM_METAL_SPLIT_PERCENT=10` enables the cooperative strategy
in the **metal** binary only. `cpu-i8mm` is a separate **CPU-only** experiment;
this session does not provide an I8MM+Metal combined binary. Both retain the
normal model/state API, with all GPU work joined before each inference call returns.

Run jobs sequentially on the working Mac. The advisory benchmark lock cannot
prevent unrelated applications or enforce thermal/clock isolation. AC power and
no `pmset` thermal warning do not certify stable timings; publish all ranges.
