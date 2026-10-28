# Metal graph experiments: persistent state and fewer host waits

Follow-up to [the initial M2 Pro measurements](MAC_BENCHMARKS.md), 2026-10-28.
Hardware, model and quantization are unchanged: M2 Pro / 16 GiB, official Qwen3.5-0.8B,
436.5 MB `.pq` Q4K weights and upstream A8 activation quantization.

## Measured outcome

**The MLP graph improves the slow broad-offload GPU baseline by 20–23%, but CPU still wins.**
Persistent DeltaNet is correct but slower in this hybrid execution path, so both graph features
remain opt-in. The four-token tile did not improve the initial prefill sweep and is not selected
by default either.

Median generated tokens/sec over three rounds (128 generated IDs, 127 timed steps):

| Configuration | Prose 64 | Chat 36 | Context 512 |
|---|---:|---:|---:|
| CPU SDOT, six threads | 185.6 | 186.2 | 160.1 |
| Head-only Metal hybrid, four CPU threads | 120.9 | 118.8 | 108.7 |
| Broad Q4K GPU offload, separate operations | 17.16 | 17.03 | 16.73 |
| **Broad Q4K GPU offload + single-submit MLP** | **20.75** | **20.43** | **20.53** |
| Broad offload + MLP graph + GPU DeltaNet | 18.20 | 18.24 | 18.20 |

Corresponding median prefill tokens/sec:

| Configuration | Prose 64 | Chat 36 | Context 512 |
|---|---:|---:|---:|
| CPU SDOT, six threads | 710.3 | 673.2 | 625.5 |
| Head-only hybrid | 569.0 | 541.7 | 563.8 |
| Broad offload, separate operations | 209.1 | 181.2 | 220.1 |
| Broad offload + MLP graph | 220.7 | 192.6 | 228.2 |
| Broad offload + both graphs | 184.6 | 172.0 | 192.6 |

Broad offload uses a zero generic matrix-size threshold (all Q4K projections). Head-only uses
64 MiB. All hybrid rows use four CPU threads; CPU comparison uses six as before. Full attention,
normalization, residuals and other remaining operations still run on the CPU.

For the prose benchmark process including warmup, grouping MLPs reduces ordinary Metal
submissions from **24,871 to 18,487 (25.7%)**. There are 3,192 MLP graphs: each replaces three
submissions with one. Enabling GPU DeltaNet adds 2,394 recurrence submissions, showing why
eliminating state-matrix copies alone does not eliminate the hybrid-boundary cost.

Median peak RSS: CPU 435–448 MiB; head-only hybrid 583–596 MiB; broad offload with MLP graph
852–865 MiB; broad offload with both graphs 854–867 MiB. These are process RSS, not measured
GPU residency or total system memory pressure.

### Timing variability—do not hide the controls

This later session was noisier than the initial report. The three CPU samples span **161.7–194.1**
tok/s on prose, **164.3–193.5** on chat and **105.0–181.4** after 512 prompt tokens. Head-only prose
spans 76.7–125.3 tok/s. The cause is not established; `pmset` still reports AC power and no recorded
thermal warning, not a guarantee of stable clocks or absence of competing work.

MLP-graph decode ranges are 19.18–21.15 / 20.03–20.80 / 20.26–20.88 tok/s; the separate-operation
GPU baseline ranges are 15.17–17.23 / 17.00–17.59 / 16.21–16.91. Thus grouping helps this slow GPU
baseline in these samples, but precise ratios are observations, not isolated laboratory estimates.
Do not interpret lower CPU controls relative to the earlier session as a demonstrated code regression.
All runs are retained; `valid=true` indicates execution, finite logits and exactness checks passed,
**not** that timing noise or throttling has been ruled out.

Raw final record (45 runs, all generated IDs and last-prefill logits exact):
[`graph-final-20261028T064355.952875Z.json`](../bench/results/metal/graph-final-20261028T064355.952875Z.json).
Initial feature isolation and negative tile results are in
[`graph-tuning-20261028T064031.568756Z.json`](../bench/results/metal/graph-tuning-20261028T064031.568756Z.json)
and [`tile-tuning-20261028T064222.952653Z.json`](../bench/results/metal/tile-tuning-20261028T064222.952653Z.json).

## What is implemented

### Single-submit MLP

`NM_METAL_MLP=1` runs an eligible Q4K MLP as one command buffer:

```
fp32 input -> GPU A8 quantization -> gate + up projections
           -> GPU SiLU(gate) * up -> GPU A8 quantization -> down projection
```

The six GPU dispatches have explicit inter-stage buffer barriers and a single host wait.
Gate/up share the same quantized input. Intermediate gate, up and quantized activation buffers
stay on the GPU; the host uploads only the MLP input and reads only its output. This replaces
three separate host submissions/waits and CPU intermediate processing for broad GPU offload.

GPU activation quantization uses the CPU's per-32-column absmax, round-to-even codes, scales
and scaled integer sums. SiLU implements the same fp32 polynomial as `forward_ops.c` with
contraction disabled and precise division. Integer dots and explicit ordered FMAs retain the
quantized CPU kernel's arithmetic. This is not a precision-relaxation optimization.

Eligibility: all three weights Q4K, compatible projection shapes, hidden and FFN widths divisible
by 256, batch 1..64. Other MLPs retain the CPU/individual-matmul route. The explicit MLP flag
offloads eligible MLP weights regardless of the generic matrix offload threshold. It is off by default.

### Persistent shared DeltaNet state

`NM_METAL_RECURRENCE=1` allocates each recurrent layer's mutable state as a persistent shared
Metal buffer, owned by the inference pool. The engine's CPU pointer refers to that same buffer;
there is no mirrored CPU state matrix and no per-token upload/download of it.

The GPU recurrence uses the same chunk-major layout as `kern_lin.c` and the same scalar operation
order per value column. It supports ordinary recurrence, logged speculative verification,
deferred updates, and explicit commit replay. Speculative tokens after token 0 use private GPU
workspace; only token 0 updates the committed state until the accepted updates are replayed.

Each call completes before returning. CPU zero/reset, state serialization, restore and checkpoint
rewind therefore read/write the shared allocation directly, without cache invalidation bookkeeping.
Pool-owned states must be released through `pf_pool_state_free`, never plain `free`.

For the 0.8B model the 18 recurrent state matrices total 18,874,368 bytes (18 MiB). Input QKV,
gates, output vectors and speculative logs still cross the hybrid boundary. **Persistent GPU-addressable
state does not mean a GPU-resident full model, guaranteed GPU-cache residency, or no synchronization.**
There is still a per-layer wait. This path is off by default.

### Four-token packed-weight reuse

`NM_METAL_KERNEL=packed4` reuses a packed weight row across four activation vectors for batched
matmuls. It keeps chronological scale/correction order for each output and uses the ordinary
packed kernel for decode batches below four. No weights are requantized and the layout is unchanged.

It passed exactness checks, but was slower in the first prefill tuning sweep. It remains explicitly
experimental, not the default. It is a small register tile, **not** a SIMD-group matrix GEMM.

## Why fewer waits are not enough

Initial tuning found that grouping the MLP improves **broad GPU offload**, but offloading MLPs
that the CPU already handles efficiently slows the faster head-only hybrid. Separately offloading
DeltaNet also adds more submission latency than its shared-state benefit saves on this workload.
The four-token tile did not improve the first prefill sweep.

All variants, including slow ones, are preserved. Useful architectural work is not automatically
an end-to-end performance win. CPU SDOT remains the recommendation unless a repeated comparison
shows otherwise.

## Validation

`python3 tools/check_graphs_macos.py` records tests for `simd`, `row`, `packed` and `packed4`:

- Each kernel variant's 21 exact Q4K/A8 dot-product cases and 12 invalid-buffer/upload checks.
- Seven MLP cases, including zero input, 1/3/7/64-token batches and real Qwen MLP dimensions.
  Compare output and intermediate up/SiLU activations bit-for-bit against existing CPU primitives.
- Six DeltaNet dimension/batch cases. Each runs sequentially on the same nonzero shared state,
  checking ordinary recurrence, speculative logs, pending updates, CPU snapshot restoration and
  explicit replay against the original NEON implementation. Output padding and unused log row 0
  are checked too.
- Five invalid MLP and five invalid DeltaNet cases per shape test, rejected before dispatch.

`tests/test_real_macos.py --graphs` adds real-model tokens/logits, sequential-prefill consistency,
state/rewind/speculation and WikiText gates. Exported CPU/GPU state bytes and both restore directions
are compared. `--kernel packed4` also exercises the optional small tile.

This establishes parity on the checked inputs, not a proof for every possible float input or a
fresh evaluation against unquantized teacher weights. NaN/overflow/subnormal behavior is not a
general-purpose numerical API guarantee.

Evidence:

- [`graph-kernel-gates-20261028T064917.903778Z.json`](../bench/results/metal/graph-kernel-gates-20261028T064917.903778Z.json):
  all four kernel variants pass their 34 numerical shape cases and 77 invalid-input checks each.
- [`real-gates-20261028T064925.061432Z.json`](../bench/results/metal/real-gates-20261028T064925.061432Z.json):
  both graphs + optional `packed4`, 21 real-model runs; CPU/GPU saved-state bytes and both restore
  directions match; WikiText perplexity **18.5061** across all three backends (8 x 512, 2040 scored).
- 90 synthetic model cases with both graph flags and `packed4`, across Q4K / Q4 / Q8:
  `model-gates-20261028T065104.587209Z.json`, `model-gates-20261028T065106.505478Z.json`, and
  `model-gates-20261028T065107.684441Z.json` under `bench/results/metal/`. Q4/Q8 MLPs retain the CPU
  route; persistent DeltaNet still runs on the GPU. Plain hidden states/logits and speculative IDs match.
- Thirteen Python unit tests pass (benchmark failure recording and configuration parsing).
- **Optional ASan/UBSan check not completed:** the instrumented CPU tool stalled even on `--help`
  before any model loading. A short retry with ASan verbosity reached interceptor initialization and
  timed out too. Root cause is unconfirmed; no sanitizer pass is claimed. Diagnostic retained in
  [`sanitizer-diagnostic-20261028T065514.810408Z.json`](../bench/results/metal/sanitizer-diagnostic-20261028T065514.810408Z.json).

## Reproduce

```sh
make mac-cpu-baseline mac-cpu mac-infer
python3 tools/check_graphs_macos.py
NM_METAL_MLP=1 NM_METAL_RECURRENCE=1 NM_METAL_KERNEL=packed4 \
  .venv/bin/python tests/test_macos_inference.py
.venv/bin/python tests/test_real_macos.py --graphs --kernel packed4 --quality-chunks 8

.venv/bin/python tools/bench_inference_macos.py --config cpu:6 \
  --config metal:4:67108864:packed:none \
  --config metal:4:0:packed:none --config metal:4:0:packed:mlp \
  --config metal:4:0:packed:graph \
  --rounds 3 --cases prose64,chat,long512 --generate 128 --label graph-final
```

Graph mode is part of each recorded configuration: `none`, `mlp`, `rec`, or `graph` (both).
The benchmark clears inherited feature flags and requires GPU-work counters for requested features,
so an environment variable cannot silently enable a different experiment or skip the requested path.
Timing protocol, warmup exclusions, EOS behavior and power/thermal limits are the same as in the
initial report. Do not compare fixed-length tokens/sec to cold-start live-chat latency.

## Remaining work

Keep full-attention KV, inter-layer activations and residuals GPU-addressable; move larger contiguous
pieces of the layer onto one GPU execution graph; implement and profile real tiled prefill GEMM;
then repeat these gates and compare against CPU SDOT and matched llama.cpp Metal/MLX configurations.
Changing precision or reduction order is a separate quality decision, not a hidden shortcut.
