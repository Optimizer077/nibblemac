# nibblemac development rules

## Scope
- This is a macOS / Apple Silicon derivative of nibblepi, not a Pi deployment.
- Active new backend: `engine/metal/`. Preserve the inherited engine as a correctness oracle.
- Do not connect to the original Pi or run `harness/pi.py` for Mac work.
- The Pi instructions in `CLAUDE.md`, old journals and upstream plans apply only to legacy Pi work.
- Preserve upstream attribution, Apache-2.0 license, history, logs, and measurements.

## Build and validation
- `make metal`: native Metal build using Command Line Tools and runtime shader compilation.
- `make test-metal`: actual-device numerical and argument-validation tests; no silent CPU fallback.
- `make cpu-reference`: inherited scalar model tools. Linux-only V3D must stay guarded by OS AND ISA.
- `python3 tools/bench_metal.py`: correctness-gated local measurements with source hashes and metadata.
- Do not claim model-level parity, exact speculation, or token throughput from standalone kernel tests.
- Metal fp32 activations and reduction ordering currently differ from the upstream int8 CPU path.
- New backend/precision paths stay experimental until end-to-end logits, greedy tokens, perplexity,
  batch/decode consistency, recurrent state and speculative decoding have been checked.
- Never enable fast math or change precision without explicit tests and recorded quality effects.

## Measurement and privacy
- Keep single-request wall time separate from batched GPU submission throughput.
- Do not label cache-resident microbenchmarks as DRAM bandwidth measurements.
- Compare runtimes on identical hardware/model/quantization/context; record power and thermal state.
- Keep failed results with `valid=false`; never delete results to improve reported numbers.
- No concurrent heavy jobs during timing. Avoid long GPU runs on this person's working Mac.
- Never commit tokens, credentials, model weights, machine serials, usernames or network details.
- Log each session in `docs/journal/YYYY-MM-DD.md` and update `CHANGELOG.md` / the active roadmap.
