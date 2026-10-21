# pinfer — Raspberry-Pi-4-native inference for Qwen3.5-0.8B

A C inference engine + Python package written for the Pi 4's Cortex-A72 (NEON only, no dot-product
instructions, ~4.2 GB/s DRAM), with its own 4-/8-bit weight formats and exact self-speculative
decoding through Qwen3.5's built-in MTP head. Everything is measured on the real board against
llama.cpp, MNN and ONNX Runtime, and every change to the Pi is logged with an undo command.

**Current result (2026-10-22, Qwen3.5-0.8B, 4 threads):** 7.9 tok/s plain greedy, **16.5 tok/s with
speculative decoding**, 37 tok/s prefill, at llama.cpp-Q4_K_M quality (perplexity 18.48 vs 18.59). llama.cpp Q4_K_M:
5.4 tok/s decode / 15 prefill; MNN int4: 6.7 / 31. Full table: [notes.md](notes.md) → "Scoreboard".

Design: [docs/design.md](docs/design.md) · Plan: [task_plan.md](task_plan.md) · Library guide:
[docs/library.md](docs/library.md) · Findings: [notes.md](notes.md)

## Layout
```
engine/include/pinfer.h   public C API              engine/src/          model loader, forward pass,
engine/FORMAT.md          .pq model format + math                        MTP/speculative, NEON kernels
engine/tools/             pinfer_run (CLI/bench), pinfer_ppl (perplexity)
engine/bench/             kernel microbenchmarks + KERNELS.md lab notebook
python/                   pip package `pinfer` (ctypes, tokenizer, chat, streaming, CLI)
tools/convert.py          HF safetensors -> .pq (bf16 / q8 / q4k)   tools/quant.py  quantizers
ref/                      fp32 golden reference (transformers)      tests/           golden + spec + pytest
bench/                    llama.cpp / MNN / ORT baselines, quality (perplexity), results/*.json
harness/pi.py             PC-side control of the Pi: check | push | run | bench | snapshot (auto-logged)
os/stock/                 flashing + headless first boot             os/kernel/  custom split-core kernel
docs/                     journal (per day), pi-state.md (every Pi change + undo), library guide
```

## Quick start (PC: Windows + WSL2 Ubuntu; Pi reachable over the harness)
```bash
# build (WSL):   engine + tools for the Pi, NEON kernels
make -C engine pi MATVEC=fast
# convert a model (PC, numpy only):
python tools/convert.py models/Qwen3.5-0.8B models/pq/qwen35-0.8b-q4k-headq4k.pq --quant q4k
# push + benchmark on the Pi (Windows):
python harness/pi.py push build/aarch64/engine/pinfer_run --dest engine
python harness/pi.py push models/pq/qwen35-0.8b-q4k-headq4k.pq --dest models
python harness/pi.py bench pinfer-q4k -- ./engine/pinfer_run -m models/qwen35-0.8b-q4k-headq4k.pq \
    -t 4 -n 64 --ids-file data/prompt64.ids --spec 3 --draft-vocab 32768 --bench
# Python on the Pi (wheel in build/dist/, install per docs/library.md):
pinfer chat -m models/qwen35-0.8b-q4k-headq4k.pq --tokenizer models/qwen35-0.8b-tokenizer \
    -p "Why is the sky blue?" --speculative --stats
```
Correctness: `uv run --no-project --with numpy python tests/test_golden.py` (engine vs fp32 reference,
per layer) and `tests/test_spec.py` (speculative == greedy).

## Records (where to look)
| Question | File |
|---|---|
| What changed, when? | `CHANGELOG.md` |
| What happened on a given day, and why? | `docs/journal/YYYY-MM-DD.md` |
| How does the Pi differ from stock, and how do I undo it? | `docs/pi-state.md` |
| Every command ever sent to the Pi | `logs/harness.jsonl` (+ full output in `logs/runs/`) |
| Full Pi configuration at a point in time | `logs/snapshots/<ts>_<label>/` |
| Benchmark numbers | `notes.md` scoreboard → `bench/results/*.json` |
| Kernel / speculative / OS lab notebooks | `engine/bench/KERNELS.md`, `engine/SPEC.md`, `os/kernel/KERNEL.md` |

## Setting up a Pi from scratch
`os/stock/render_firstboot.py` + `os/stock/flash_disk.ps1` write Raspberry Pi OS Lite (64-bit) with
headless Wi-Fi/SSH onto a disk; `docs/journal/2026-10-21.md` has the full story, including the
direct PC↔Pi Ethernet setup (`os/pc/eth_link.ps1`) used because `.local` names resolve wrongly on
this network.
