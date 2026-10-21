# Documentation index

One line per document. Paths are relative to this folder.

## Start here
- [../README.md](../README.md): what pinfer is, why the Pi 4 is hard, how it works, headline results, usage.
- [ARCHITECTURE.md](ARCHITECTURE.md): how the engine is built: forward pass, source map, `.pq` layout, threads, memory, speculation, bit-exactness (with diagrams).
- [RESULTS.md](RESULTS.md): every measured number in tables, each with its `bench/results/` file.
- [REPRODUCE.md](REPRODUCE.md): from a blank SSD to the scoreboard: hardware, flashing, cable, harness, builds, models, exact benchmark commands.
- [GLOSSARY.md](GLOSSARY.md): MTP, Gated DeltaNet, Q4K superblock, tryboot and other recurring terms.
- [../CONTRIBUTING.md](../CONTRIBUTING.md): working rules: harness only, bench lock, snapshots, journals, bit-exactness, commits, secrets.
- [library.md](library.md): install and use the C library, the Python package and the `pinfer` command.

## Design and specifications
- [design.md](design.md): the original design: intent, hardware facts, phases, constraints, target model.
- [../engine/FORMAT.md](../engine/FORMAT.md): `.pq` format v1, exact forward-pass math, quantized dtype layouts.
- [../engine/include/pinfer.h](../engine/include/pinfer.h): public C API; its comments are the API reference.
- [../engine/src/ops.h](../engine/src/ops.h): the contract between model code and matrix kernels.
- [release-0.1.0.md](release-0.1.0.md): draft release notes for 0.1.0 (not published).

## Lab notebooks (one per work area; measurements, failures, decisions)
- [../engine/bench/KERNELS.md](../engine/bench/KERNELS.md): NEON matvec/matmul kernels, A72 instruction costs, quantization error, the bit-exactness fix, the recurrence kernel.
- [../engine/SPEC.md](../engine/SPEC.md): MTP head semantics, speculative decoding design, validation, Pi measurements.
- [../engine/CORE.md](../engine/CORE.md): vectorizing and threading the per-token work, `exp` accuracy, step profiles.
- [../engine/CONTEXT.md](../engine/CONTEXT.md): context window: attention rewrite, KV dtypes, state save/restore, adaptive k.
- [../os/kernel/KERNEL.md](../os/kernel/KERNEL.md): custom 6.18.50-pinfer kernel: config, build, tryboot install, rollback, why no 16 KB pages.
- [../bench/BASELINES.md](../bench/BASELINES.md): llama.cpp, MNN, ONNX Runtime GenAI, LiteRT-LM and PyTorch on the Pi.

## Records
- [../notes.md](../notes.md): findings and the scoreboard tables with result-file paths.
- [../task_plan.md](../task_plan.md): phases, status, decisions, blockers.
- [../CHANGELOG.md](../CHANGELOG.md): every change to the project and to the Pi, newest first.
- [journal/2026-10-21.md](journal/2026-10-21.md): day 1: design, harness, flashing, first boot, baselines, scalar engine, MTP speculation, packaging.
- [journal/2026-10-22.md](journal/2026-10-22.md): day 2: formats and perplexity, the bit-exactness fix, custom kernel, vectorized engine, context work, GitHub repo.
- [pi-state.md](pi-state.md): every change from stock on the Pi and the PC, with undo commands, placed files and snapshots.
- [../CLAUDE.md](../CLAUDE.md): project rules for coding agents (same rules as CONTRIBUTING.md, in short form).
- `../logs/harness.jsonl`, `../logs/runs/`, `../logs/snapshots/`: audit log of every harness call, full `run` outputs, Pi configuration records.
- `../bench/results/`: one JSON file per benchmark run (format: [REPRODUCE.md §9](REPRODUCE.md#9-reading-benchresultsjson)).
- `figures/`: the result figures, each in four variants (desktop / phone × light / dark), rendered by `../tools/make_figures.py` and `../tools/make_figures2.py` (list: [RESULTS.md §13](RESULTS.md#13-figures)).
