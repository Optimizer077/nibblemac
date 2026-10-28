# nibblemac development rules

## Scope
- This is a macOS / Apple Silicon derivative of nibblepi, not a Pi deployment.
- Active new backend: `engine/metal/`; native model builds: `engine/macos/`.
  Preserve the inherited kernels (including the no-SDOT baseline) as correctness oracles.
- Do not connect to the original Pi or run `harness/pi.py` for Mac work.
- The Pi instructions in `CLAUDE.md`, old journals and upstream plans apply only to legacy Pi work.
- Preserve upstream attribution, Apache-2.0 license, history, logs, and measurements.

## Build and validation
- `make metal`: native Metal build using Command Line Tools and runtime shader compilation.
- `make test-metal`: actual-device numerical and argument-validation tests; no silent CPU fallback.
- `make cpu-reference`: inherited scalar model tools. Linux-only V3D must stay guarded by OS AND ISA.
- `python3 tools/bench_metal.py`: correctness-gated local measurements with source hashes and metadata.
- `make mac-cpu-baseline mac-cpu mac-infer`: SMULL baseline, native SDOT, opt-in Metal hybrid model tools.
- `make test-a8`: exact Metal integer-activation kernels vs the inherited NEON implementation.
- `python3 tools/check_graphs_macos.py`: record exact MLP graph, persistent DeltaNet and invalid-input
  tests for every GPU kernel variant. `make test-graphs` runs the low-level graph suite directly.
- `.venv/bin/python tests/test_macos_inference.py`: synthetic full-model hidden/logit/token/speculation gates.
- `.venv/bin/python tests/test_real_macos.py`: real-model parity, state/rewind, speculation and WikiText PPL.
- Use `--graphs` for persistent DeltaNet + single-submit MLP gates; `--kernel packed4` also checks
  the experimental four-token weight-reuse tile. All graph features remain opt-in.
- `.venv/bin/python tools/bench_inference_macos.py`: locked real-model benchmarks, exact-output gated.
- Do not claim model-level parity, exact speculation, or token throughput from standalone kernel tests.
- Standalone Metal fp32 kernels differ from the upstream int8 CPU path; inference Q4K/A8 kernels
  instead use exact integer dots and ordered explicit FMA/correction matching the CPU kernel.
- New backend/precision paths stay experimental until end-to-end logits, greedy tokens, perplexity,
  batch/decode consistency, recurrent state and speculative decoding have been checked.
- Never enable fast math or change precision without explicit tests and recorded quality effects.

## Measurement and privacy
- Keep single-request wall time separate from batched GPU submission throughput.
- Do not label cache-resident microbenchmarks as DRAM bandwidth measurements.
- Compare runtimes on identical hardware/model/quantization/context; record power and thermal state.
- Keep failed results with `valid=false`; never delete results to improve reported numbers.
- Preserve slow configurations too. CPU SDOT currently beats the Metal hybrid on this 0.8B workload;
  do not describe a Metal-only improvement as an end-to-end win over CPU.
- Shared DeltaNet state is a persistent GPU-addressable MTLBuffer, not a CPU/GPU mirrored cache.
  Calls complete before CPU reset/save/load/rewind; free states through the pool, never `free` a
  Metal-owned pointer. Do not claim full-model residency just because DeltaNet state is shared.
- No concurrent heavy jobs during timing. Avoid long GPU runs on this person's working Mac.
- Never commit tokens, credentials, model weights, machine serials, usernames or network details.
- Log each session in `docs/journal/YYYY-MM-DD.md` and update `CHANGELOG.md` / the active roadmap.
