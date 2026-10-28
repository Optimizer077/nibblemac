# nibblemac

**An Apple Silicon / Metal inference project derived from
[nibblepi](https://github.com/Optimizer077/nibblepi).**

The goal is a highly optimized, native macOS engine for quantized Qwen3.5 inference:
packed 4-bit weights, unified memory, GPU-resident execution, and measured optimizations.
The first development machine is an **M2 Pro, 16 GPU cores, 16 GB unified memory**.

## Current status

This is an **early Metal foundation**, not yet a full Metal-backed chatbot.

- Working native C / Objective-C Metal runtime and C API.
- F32 and packed `.pq` Q4K matrix-vector / batched matrix-vector kernels.
- Weights stay packed; shared buffers expose unified memory to both CPU and GPU.
- Runtime shader compilation: full Xcode and the offline Metal compiler are not required.
- Deterministic numerical tests, invalid-input checks, and recorded microbenchmarks.
- The inherited scalar CPU inference tools now build on macOS.

**Not implemented yet:** Metal integration into the model forward pass, GPU attention and
DeltaNet, tiled prefill GEMM, macOS library/wheel packaging, and end-to-end Mac performance
comparisons. The Pi engine's Python package does **not** use this new Metal runtime.

## Quickstart

Requirements: Apple Silicon Mac, macOS 13+, Xcode Command Line Tools (`xcode-select --install`),
and Python 3.10+ for benchmark recording. No model downloads are needed for these tests.

```sh
make                 # Build the Metal test/benchmark tool on macOS
make test-metal      # Numerical and validation tests on the actual GPU
make bench-metal     # Correctness-gated, three-round microbenchmark + JSON record
make cpu-reference   # Build the legacy scalar inference tools, not a Metal model runner
```

Results go to `bench/results/metal/`. `python3 tools/bench_metal.py --rounds 1` runs a shorter
measurement. Test failures and failed benchmark runs are retained with `valid=false`.
The harness records source hashes, revision, compiler, chip, OS, and best-effort power/thermal
status without recording machine serial numbers or usernames.

These are **synthetic resident-buffer kernel measurements, not tokens/sec**. Repeated-dispatch
GPU throughput and single-submit wall-clock latency are reported separately. Small weights may
fit in cache. No speedup over llama.cpp, MLX, or the original Pi engine is claimed.

## Design direction

1. Establish correctness and native Metal measurement infrastructure. **Initial foundation done.**
2. Integrate a selectable backend with the existing loader and scalar oracle; add BF16/Q8/Q4
   support where required by real models, and compare model logits and generated tokens.
3. Keep activations, attention KV cache, and recurrent state on the GPU; avoid waiting after
   every small operation. Fuse normalization, activation, and residual operations when measured.
4. Tune decode for packed-weight bandwidth and prefill for tiled SIMD-group matrix operations.
5. Benchmark against llama.cpp Metal and MLX using the same model, quantization, prompts,
   contexts, power conditions, and quality checks. Only retain proven improvements.
6. Port speculative decoding, prefix reuse, and state checkpointing with explicit correctness gates.

The baseline uses fp32 inputs and reductions with fast math disabled. It does **not** promise
bit-identical output to the Pi kernels: those quantize activations to int8 and use different
reduction ordering. Model-level agreement must be measured before enabling a Metal backend.

## Source map

| Path | Purpose |
|---|---|
| `engine/metal/` | New Metal C API, runtime, shaders, correctness/benchmark executable |
| `tools/bench_metal.py` | Local correctness-gated benchmark recorder |
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
