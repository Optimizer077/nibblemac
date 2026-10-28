# M2 Pro: real Qwen3.5-0.8B inference and optimization results

Measured on 2026-10-28. **These are full-model tokens/sec, not kernel estimates.**

## Outcome

- **Recommended measured configuration: six-thread CPU SDOT**, not the current Metal hybrid.
- CPU SDOT + measured thread selection: **196.5–210.0 generated tokens/sec**, with **769–791
  prompt tokens/sec** across the three tested workloads (medians of three rounds).
- Adaptive self-speculation: **228.3 tok/s on prose**, but **178.9 tok/s on chat**, slower than
  plain CPU there. It remains an explicit option, not a universal default.
- Optimized Metal hybrid: **118.9–124.1 tok/s**, **31–35% faster than the first Metal hybrid**,
  but still slower than CPU SDOT. Calling this a GPU win over CPU would be misleading.
- All 54 benchmark runs matched reference generated IDs; all 45 plain runs also matched last
  prefill logits bit-for-bit. Real state/replay/speculation gates pass. WikiText-2 perplexity is
  **18.5061** for the original NEON port, CPU SDOT, and optimized Metal hybrid.

Raw final measurements: [`final-20261028T061717.868349Z.json`](../bench/results/metal/final-20261028T061717.868349Z.json).
Quality/state evidence: [`real-gates-20261028T061914.533172Z.json`](../bench/results/metal/real-gates-20261028T061914.533172Z.json).

## Hardware and model

| Item | Measured configuration |
|---|---|
| Machine | MacBook Pro, Apple M2 Pro |
| CPU / GPU | 10 CPU cores (6 performance + 4 efficiency), 16 GPU cores |
| Unified memory | 16 GiB |
| OS | `sw_vers` reports 27.2, build 26B118 |
| Compiler | Apple clang 17.0.0 (`clang-1700.6.3.2`) |
| Native flags | `-O3 -mcpu=native -std=c11 -ffp-contract=off`; macOS 13 deployment target |
| Metal compilation | Runtime MSL 2.4, fast math disabled |
| Model | Official `Qwen/Qwen3.5-0.8B`, text inference, 24 layers, MTP head present |
| Source revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Converted model | 335 tensors, 436,507,904 bytes (436.5 MB / 416.3 MiB) |
| Quantization | Upstream `.pq` Q4K, 4.5 bits/weight including metadata; int8 activations in 32-column groups |
| Attention cache | fp32; max context allocation 2048 tokens |
| Sampling | Greedy; exact self-speculation where explicitly named |
| Power | AC power, battery fully charged during the final run |

Converted model SHA-256:
`3573deef290ff466f1295b023688ae3202c85d97600deadbe1e913c2a53ec350`.
The downloader verifies the official checkpoint/tokenizer against pinned Hugging Face LFS hashes.
Weights are ignored by git and are not published in this repository.

## Generation speed

**Median generated tokens/sec, three rounds each. Higher is better.**

| Configuration | Prose, 64 prompt tokens | Chat, 36 prompt tokens | Repeated prose, 512 prompt tokens |
|---|---:|---:|---:|
| Original NEON port, 4 threads (no SDOT) | 170.3 | 170.8 | 161.7 |
| CPU SDOT, 4 threads | 195.1 | 193.4 | 190.5 |
| **CPU SDOT, 6 threads** | **210.0** | **208.3** | **196.5** |
| CPU SDOT, 4 threads + adaptive MTP k<=4 | 228.3 | 178.9 | 220.3 |
| First Metal hybrid, 4 CPU threads | 90.4 | 91.7 | 90.4 |
| Optimized Metal hybrid, 4 CPU threads | 118.9 | 124.1 | 119.9 |

For both Metal rows, only matrices of at least 64 MiB are offloaded—in this model, the full
vocabulary projection. The rest remains CPU NEON/SDOT. The first GPU kernel uses eight lanes
per row with SIMD reductions; the optimized kernel uses one row per lane and coalesced code words.
This is **not fully GPU-resident inference**.

### Observed generation ranges

Min–max tokens/sec over the same three rounds:

| Configuration | Prose 64 | Chat 36 | Repeated prose 512 |
|---|---:|---:|---:|
| CPU SDOT, 6 threads | 207.9–212.0 | 205.7–208.6 | 193.8–200.9 |
| Adaptive MTP, 4 threads | 228.0–228.6 | 174.0–179.1 | 218.7–221.1 |
| Optimized Metal hybrid | 110.2–121.7 | 123.4–126.6 | 118.6–121.0 |

The synthetic repeated prose favors draft acceptance; it is not representative of every chat or
code workload. The chat prompt asks for an approximately 200-word explanation of unified memory.
Exact prompt text and token IDs, generated text, draft acceptance and per-run metrics are in the JSON.

## Prompt processing

**Median prompt tokens/sec, three rounds; includes the last-token vocabulary projection.**

| Configuration | Prose 64 | Chat 36 | Repeated prose 512 |
|---|---:|---:|---:|
| Original NEON port, 4 threads | 427.3 | 420.9 | 417.2 |
| CPU SDOT, 4 threads | 599.4 | 586.8 | 577.5 |
| **CPU SDOT, 6 threads** | **790.7** | **785.2** | **769.3** |
| CPU adaptive MTP, 4 threads | 548.5 | 529.2 | 532.1 |
| First Metal hybrid | 565.8 | 523.7 | 569.9 |
| Optimized Metal hybrid | 573.6 | 561.2 | 567.3 |

Speculative prefill also prepares the MTP state, so it costs more than plain prefill. GPU prefill
is still batched matvec, not a high-throughput tiled GEMM implementation.

## Process memory

Median peak resident set size (MiB; Darwin `ru_maxrss` converted from bytes):

| Configuration | Prose 64 | Chat 36 | Repeated prose 512 |
|---|---:|---:|---:|
| CPU SDOT, 6 threads | 437.3 | 435.2 | 447.8 |
| CPU adaptive MTP, 4 threads | 459.2 | 456.3 | 471.5 |
| Optimized Metal hybrid | 584.8 | 582.7 | 595.4 |

This is process peak RSS, not total system memory pressure, allocated virtual memory, or a complete
GPU residency/energy measurement. The hybrid keeps a separate immutable GPU-visible weight cache;
the vocabulary head is 143,032,320 bytes. Its repacked form retains the same size because the head's
row count is divisible by 32. Other shapes can need small row padding.

## What changed, and what did not help

1. **Native Mac thread pool:** replace Linux-only futex/affinity code with a condition-variable
   idle path and macOS worker QoS, retaining fast generation-counter dispatch and independent-row
   arithmetic. Do not assume Linux CPU pinning applies to performance/efficiency cores.
2. **CPU SDOT:** use Apple Silicon integer dot-product instructions instead of the Pi A72's
   SMULL/SMLAL sequences. Integer sums are exact and the existing floating-point scale/correction
   order is unchanged. Four-thread decode improves about **13–18%** over the no-SDOT port.
3. **Measured thread choice:** six threads improve plain decode **21–23%** and prefill **84–87%**
   versus the original four-thread port. This is a combined SDOT/thread-selection result, not an
   isolated SDOT claim, and not proof that six threads win on every Mac or workload.
4. **Metal Q4K/A8:** reuse the CPU's exact activation quantization, integer dot sums and ordered
   explicit FMA/correction. Avoid changing activation precision merely to get a faster GPU number.
5. **GPU row mapping:** remove cross-lane integer reductions and mostly inactive scale/FMA lanes.
6. **Coalesced weight layout:** transpose packed code words across 32 rows once at upload. A SIMD
   group can then load a contiguous span. No code/scale is requantized and no fp16 weight expansion
   is used. Combined with row mapping, full hybrid decode improves **31.4%, 35.4%, 32.6%** across
   the three workloads versus the first SIMD GPU kernel.
7. **Rejected broad GPU offload:** offloading matrices down to 1 MiB yielded only about **21–22
   tok/s** with the packed kernel, and slower prefill. Keep the cutoff explicit; do not silently
   describe those runs as fast GPU inference. All slow tuning records are retained.
8. **Speculation is workload-dependent:** fixed k=4 was especially slow on chat in tuning. Adaptive
   drafting helps there, but still trails plain six-thread CPU on this chat prompt.

The instrumented phase profile found the initial GPU head path substantially slower than CPU
(about 8.2 ms/call versus 1.44 ms/call in that profile, including host dispatch/copies/waits and
warmup). Those instrumented times are diagnostic, not the unprofiled final throughput figures.
Synchronous CPU/GPU boundaries and the absence of a full GPU execution graph remain major limits.

## Correctness and quality evidence

- Standalone F32/Q4K numerical, analytical-layout, zero/tail/batch tests and 14 original invalid cases.
- **63 exact Q4K/A8 kernel cases:** 21 shapes/batches for each of the three GPU variants; 12
  additional invalid upload/buffer/shape cases checked for each variant.
- **90 synthetic full-model cases:** Q4K, Q4 and Q8, each over three dimension fixtures and ten
  CPU/hybrid/plain/speculative configurations. Plain tokens, layer hidden states and logits are
  bit-identical to the no-SDOT CPU build; speculative k=1,2,3 emits the same IDs. GPU Q4/Q8
  currently use the CPU fallback by design; only Q4K is GPU-offloaded.
- **54 real benchmark runs:** 128 generated IDs per run agree with reference; 45 plain runs also
  agree on last-prefill logits. Speculative benchmark rows are token-gated, not mislabeled as
  having separately dumped prefill logits.
- Real full-model gates also force **all Q4K matrices onto the GPU**, not just the head, and check
  plain vs token-by-token prefill, CPU-written state restored by GPU, checkpoint/rewind, fixed
  speculation and adaptive speculation.
- **WikiText-2 raw test prefix:** 8 x 512-token chunks, reset state per chunk; score positions
  256..510 predicting 257..511, 2040 scored tokens. Tokenization matches inherited token IDs.
  Original NEON, CPU SDOT and optimized hybrid all report **18.5061 perplexity**.
- Eleven Python tests cover benchmark failure recording/gating and inference configuration parsing.

This establishes agreement with the existing quantized engine on these tests. It is **not** a fresh
unquantized teacher evaluation, a general model-quality benchmark, or a proof for every possible input.

## Measurement protocol and limits

- Six configurations x three prompts x three rounds, sequential under a local advisory lock.
- 128 generated IDs, **127 timed decode steps**: the first generated token is obtained from prefill.
- Decode rate = 127 / decode wall seconds. Prefill rate = prompt tokens / prefill wall seconds.
- 64-token prefill batches, fp32 KV cache, greedy output; no prefix reuse between benchmark runs.
- Untimed warmup processes the same prompt plus up to four decode steps, then resets state.
- Model loading, shader compilation, initial weight upload, tokenization, output decoding/rendering
  and warmup are excluded from tokens/sec. These numbers are not cold application TTFT or live-chat
  streaming latency. EOS is ignored for fixed-length measurement; output after EOS is not a chat answer.
- Configuration order reverses in round two. Report medians and ranges, not the fastest sample.
- AC/thermal status is recorded before and after. `pmset` reports no recorded warning; actual chip
  temperature, clock residency, watts and energy were not instrumented. Background desktop work and
  scheduler behavior are not isolated. Reproduce on an otherwise idle Mac for your own comparison.
- Source/binary/model hashes are recorded. Tuning ran from a dirty worktree; use those hashes and
  complete logs, not the pre-existing HEAD alone, to identify the measured code. Later additions to
  invalid-input tests do not change the benchmarked inference binaries.
- No apples-to-apples llama.cpp Metal / MLX baseline was run. **No superiority over either is claimed.**

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos.txt
python3 tools/setup_model_macos.py
.venv/bin/python tools/convert.py models/Qwen3.5-0.8B models/qwen35-0.8b-q4k.pq --quant q4k
make mac-cpu-baseline mac-cpu mac-infer
.venv/bin/python tools/bench_inference_macos.py \
  --config cpu-baseline:4 --config cpu:4 --config cpu:6 --config cpu-spec4:4 \
  --config metal:4:67108864:simd --config metal:4:67108864:packed \
  --adaptive --rounds 3 --cases prose64,chat,long512 --generate 128 --label final
.venv/bin/python tests/test_real_macos.py --quality-chunks 8
```

Other evidence in `bench/results/metal/` includes the first `tuning-*` sweep, `tuning-row-spec-*`,
`tuning-packed-*`, `profile-*`, `kernel-gates-*` and `model-gates-*` records. Nothing was deleted
because it was slow.

## Next optimization work

For this 0.8B model, use CPU SDOT today when throughput matters. To make Metal competitive, keep
activations, attention cache and DeltaNet state GPU-resident; encode/fuse larger portions of the
forward pass; implement a real tiled prefill GEMM; and profile bandwidth, occupancy and submission
cost separately. Re-run the same correctness/quality gates and CPU baseline after each change.
