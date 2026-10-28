# nibblemac

**An Apple Silicon / Metal inference project derived from
[nibblepi](https://github.com/Optimizer077/nibblepi).**

The goal is a highly optimized, native macOS engine for quantized Qwen3.5 inference:
packed 4-bit weights, unified memory, GPU-resident execution, and measured optimizations.
The first development machine is an **M2 Pro, 16 GPU cores, 16 GB unified memory**.

## Current status

This is an **experimental CPU / Metal hybrid inference engine**. It runs real Qwen3.5 `.pq`
models on macOS; it is not yet fully GPU-resident and is not packaged as a native chatbot.

- Working native C / Objective-C Metal runtime and C API.
- F32 and packed `.pq` Q4K matrix-vector / batched matrix-vector kernels.
- Weights stay packed; shared buffers expose unified memory to both CPU and GPU.
- Runtime shader compilation: full Xcode and the offline Metal compiler are not required.
- Deterministic numerical tests, invalid-input checks, and recorded microbenchmarks.
- The inherited scalar CPU inference tools now build on macOS.
- Native threaded NEON inference with Apple Silicon SDOT integer dot products.
- Experimental model execution using exact Q4K/A8 Metal kernels and a bounded immutable weight cache.
- GPU-friendly, coalesced packed-weight layout without requantizing or expanding to fp16.
- Opt-in single-submit GPU MLP graph with GPU activation quantization/SiLU and no CPU intermediate copies.
- Opt-in persistent shared GPU DeltaNet state, including speculative logs/replay and CPU-visible snapshots.
- Real-model tokens/sec recording with exact-output gates, separate prefill/decode and repeated rounds.

**Important:** the first full-model sweep found that CPU SDOT is faster than the current Metal
hybrid for this small model. Metal is opt-in; a GPU is not automatically an optimization.
Raw tuning runs are in `bench/results/metal/`; no llama.cpp/MLX speedup is claimed.

### Measured on the M2 Pro

Real Qwen3.5-0.8B Q4K, median of three rounds, 128 generated IDs (127 timed steps):

| Configuration | Prose 64: decode | Chat 36: decode | Context 512: decode |
|---|---:|---:|---:|
| Original NEON port, 4 threads | 170.3 tok/s | 170.8 tok/s | 161.7 tok/s |
| **CPU SDOT, 6 threads** | **210.0 tok/s** | **208.3 tok/s** | **196.5 tok/s** |
| CPU adaptive speculation, 4 threads | 228.3 tok/s | 178.9 tok/s | 220.3 tok/s |
| Optimized Metal hybrid, 4 CPU threads | 118.9 tok/s | 124.1 tok/s | 119.9 tok/s |

CPU SDOT prefill: **769–791 tok/s**. Metal hybrid improved **31–35%** over the first GPU
implementation but remains slower than CPU here. Plain CPU, SDOT and hybrid perplexity all
match at **18.5061** (WikiText-2, 8 x 512-token chunks).

**[Full results, ranges, memory, correctness evidence, limitations and reproduction](docs/MAC_BENCHMARKS.md).**

### Public runtime comparison

Same-Mac follow-up, three rounds, same prompt IDs and generation length:

| Runtime | Prose 64: decode | Chat 36: decode | Context 512: decode |
|---|---:|---:|---:|
| nibblemac CPU SDOT, 6 threads | 193.3 tok/s | 204.3 tok/s | 139.5 tok/s |
| nibblemac Metal head-only, 6 CPU threads | 122.1 tok/s | 120.4 tok/s | 112.2 tok/s |
| llama.cpp Metal, public Q4_K_M | 148.2 tok/s | 150.4 tok/s | 150.0 tok/s |
| MLX-LM Q4/group-32, synchronized adapter | 169.6 tok/s | 157.5 tok/s | 153.7 tok/s |

Public GPU runtimes lead prefill: **1019–2063 tok/s**, versus CPU **716–808 tok/s**.
WikiText PPL: **18.5061 nibblemac / 18.2005 llama.cpp / 20.5333 MLX**. Quantization,
cache precision and model provenance differ. Timings are noisy (CPU chat range
73.7–216.2 tok/s); this is not an equal-quality backend comparison or a blanket win.
The earlier table is a separate historical run, not a contemporaneous control.
**[Public baseline details, ranges, request time, memory and reproduction](docs/PUBLIC_BASELINES.md).**

**Not implemented yet:** GPU-resident full attention, efficient large-tile prefill GEMM, full-model
Metal execution, and macOS library/wheel packaging. The Pi engine's Python package does **not**
automatically use the new Metal runtime.

## Quickstart

Requirements: Apple Silicon Mac, macOS 13+, Xcode Command Line Tools (`xcode-select --install`),
and Python 3.10+ for benchmark recording. No model downloads are needed for these tests.

```sh
make                 # Build the Metal test/benchmark tool on macOS
make test-metal      # Numerical and validation tests on the actual GPU
make bench-metal     # Correctness-gated, three-round microbenchmark + JSON record
make cpu-reference   # Build the legacy scalar inference tools, not a Metal model runner
make mac-cpu         # Native NEON/SDOT model runner
make mac-infer       # Experimental CPU/Metal hybrid model runner
make test-a8         # Exact Q4K/A8 Metal vs NEON checks on Apple Silicon
make test-graphs     # Exact GPU MLP graph and persistent DeltaNet checks
```

### Real model setup and tokens/sec

The checkpoint download is about 1.75 GB; the converted Q4K model is about 436.5 MB.
Weights and build outputs are ignored by git. The source revision and LFS hashes are pinned.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos.txt
python3 tools/setup_model_macos.py
.venv/bin/python tools/convert.py models/Qwen3.5-0.8B models/qwen35-0.8b-q4k.pq --quant q4k
make mac-cpu-baseline mac-cpu mac-infer
.venv/bin/python tools/bench_inference_macos.py \
  --config cpu-baseline:4 --config cpu:4 --config metal:4:67108864:packed \
  --rounds 3 --cases prose64,chat,long512 --generate 128
```

This records **real model** throughput, generated text/IDs, peak RSS, raw logs, source and
binary hashes, model provenance and exactness against the quantized CPU reference. Prompt
processing and generated-token rates are separate; load, tokenizer and warmup are excluded.
The CLI intentionally ignores EOS for fixed-length benchmarks (not a live-chat latency claim).

`cpu-baseline` retains the original SMULL/SMLAL kernels; `cpu` enables SDOT. Append `-spec4`
to a configuration, e.g. `cpu-spec4:4`, and pass `--adaptive` to test exact adaptive speculation.
Metal configurations take a thread count, minimum matrix size in bytes, and kernel variant:
`simd` (first baseline), `row`, or `packed` (coalesced). Smaller offloads were slower in tuning;
the experimental default uses the packed kernel and a 64 MiB threshold (the vocabulary head).

### Experimental GPU graphs

These are architectural experiments, **not new recommended defaults**. A single MLP submission
keeps intermediate activations on the GPU, reducing three host waits to one. DeltaNet can use a
persistent shared Metal state buffer without per-token state-matrix upload/download. CPU state
save/load/reset/rewind still work because each call completes before returning.

```sh
# Both graph features (still a CPU/Metal hybrid, not a complete GPU execution graph):
NM_METAL_MLP=1 NM_METAL_RECURRENCE=1 \
  build/macos-metal/engine/pinfer_run -m models/qwen35-0.8b-q4k.pq --ids 1,2,3 -n 32

# Compare on identical model and prompts; flags are explicit in every result record:
.venv/bin/python tools/bench_inference_macos.py --config cpu:6 \
  --config metal:4:0:packed:none --config metal:4:0:packed:mlp \
  --config metal:4:0:packed:graph --rounds 3 --generate 128
python3 tools/check_graphs_macos.py
.venv/bin/python tests/test_real_macos.py --graphs --quality-chunks 8
```

Graph mode after the kernel name is `none` (default), `mlp`, `rec`, or `graph` (both).
`packed4` is an additional opt-in kernel that reuses packed weights over a four-token tile;
it preserves exact arithmetic but was slower in the first prefill sweep. Broad GPU offload and
per-layer DeltaNet submissions remain slower than CPU; fewer waits alone do not establish a win.
See [the GPU graph experiments report](docs/METAL_GRAPHS.md) for architecture, evidence and limitations.

Additional gates:

```sh
.venv/bin/python tests/test_macos_inference.py   # Synthetic logits/hidden/tokens/speculation
.venv/bin/python tests/test_real_macos.py --quality-chunks 8  # Real logits/tokens/state/PPL
```

Results go to `bench/results/metal/`. `python3 tools/bench_metal.py --rounds 1` runs a shorter
measurement. Test failures and failed benchmark runs are retained with `valid=false`.
The harness records source hashes, revision, compiler, chip, OS, and best-effort power/thermal
status without recording machine serial numbers or usernames.

`make bench-metal` produces **synthetic resident-buffer kernel measurements, not tokens/sec**. Repeated-dispatch
GPU throughput and single-submit wall-clock latency are reported separately. Small weights may
fit in cache. No speedup over llama.cpp, MLX, or the original Pi engine is claimed.

## Design direction

1. Establish correctness and native Metal measurement infrastructure. **Initial foundation done.**
2. Integrate a selectable backend with the existing loader and oracle. **Initial hybrid done:**
   Metal Q4K/A8, CPU BF16/Q8/Q4/F32 and non-matmul operations; real-model parity gates.
3. Keep activations, attention KV cache, and recurrent state on the GPU; avoid waiting after
   every small operation. **Partial:** persistent DeltaNet buffers and single-submit MLP graph
   are implemented as opt-in experiments; full attention and inter-layer residency remain future work.
4. Tune decode for packed-weight bandwidth and prefill for tiled SIMD-group matrix operations.
5. Benchmark against llama.cpp Metal and MLX. **Initial public-runtime comparison done:**
   same prompt IDs and quality protocol, but different quantizers/cache precision. Matched
   weight/precision comparisons and isolated timing remain open; retain only proven improvements.
6. Port speculative decoding, prefix reuse, and state checkpointing with explicit correctness gates.

The original standalone kernels use fp32 inputs and reductions and do **not** promise bit-identical
Pi output. The newer inference kernels instead consume the CPU's int8 activations, use exact integer
dots and the CPU's chronological FMA/correction order. Fast math stays disabled. Tests compare these
paths bit-for-bit, but this is agreement with the quantized engine—not a new fp32 teacher evaluation.

## Source map

| Path | Purpose |
|---|---|
| `engine/metal/` | New Metal C API, runtime, shaders, correctness/benchmark executable |
| `engine/macos/` | Native scalar-multiply baseline, SDOT CPU and hybrid model builds |
| `tools/bench_metal.py` | Local correctness-gated benchmark recorder |
| `tools/bench_inference_macos.py` | Real-model throughput, output agreement and raw measurement records |
| `engine/src/`, `engine/include/` | Inherited inference core, `.pq` loader, C API |
| `tools/convert.py`, `tools/quant.py` | Upstream model conversion and quantization |
| `engine/FORMAT.md` | Packed model/tensor format |
| `README.nibblepi.md` | Original Pi README and results (historical, not Mac results) |
| `task_plan.md` | Active Mac roadmap followed by archived Pi plans |
| `AGENTS.md` | Mac development and measurement rules |

Development checks without a GPU: `python3 tests/test_bench_metal.py`. The optional inherited
CPU regression suite needs NumPy: `.venv/bin/python tests/test_dims.py --bin build/host/engine/pinfer_run`.

The original `harness/`, `os/`, journals, and Pi benchmarks are retained as historical upstream
material. Mac development does not connect to or modify the original Raspberry Pi.

## License and attribution

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Derived from nibblepi by Optimizer077,
starting at commit `7a395e5`. Original history and attribution are preserved. This project is a
separate macOS/Metal derivative, not a claim that the upstream project supports Metal.
