# piquant — Qwen3.5-0.8B at 16 tokens/s on a Raspberry Pi 4

**pinfer** is a C inference engine and Python package written for one board: the Raspberry Pi 4
(Cortex-A72, NEON only, 4 GB, ~4.2 GB/s of memory bandwidth). It runs the hybrid Qwen3.5-0.8B
model with its own 4-bit weight format and *exact* self-speculative decoding, and it is measured
on the real board against llama.cpp, MNN and ONNX Runtime, with every run's temperature and
throttle state recorded.

Repository: private, `github.com/Optimizer077/piquant` · engine and package: `pinfer`.

| | pinfer (Q4K) | llama.cpp Q4_K_M | MNN int4 |
|---|---|---|---|
| Decode, plain greedy | **7.9 tok/s** | 5.4 | 6.7 |
| Decode, speculative (exact) | **16.5 tok/s** | — | — |
| Prefill (prompt reading) | **37 tok/s** | 15 | 31 |
| Perplexity, wikitext-2 | 18.50 | 18.59 | n/a |

*Raspberry Pi 4 Model B Rev 1.1, stock Raspberry Pi OS kernel, 4 threads, 64-token prompt, 64 generated
tokens. Every number has a result file in `bench/results/` listed in [notes.md](notes.md).*

![Decode speed by engine](docs/figures/decode_speed.png)

## Contents
1. [Why the Pi 4 is hard, and what follows from it](#1-why-the-pi-4-is-hard-and-what-follows-from-it)
2. [How pinfer works](#2-how-pinfer-works)
3. [Results](#3-results)
4. [What did not help](#4-what-did-not-help)
5. [Correctness: how we know the output is right](#5-correctness-how-we-know-the-output-is-right)
6. [Using the library](#6-using-the-library)
7. [Methodology and records](#7-methodology-and-records)
8. [Repository layout](#8-repository-layout)
9. [Status and roadmap](#9-status-and-roadmap)

## 1. Why the Pi 4 is hard, and what follows from it

Three hardware facts decided the whole design.

- **One core saturates the memory bus.** A 256 MB streaming read runs at 4.20 GB/s on one core and
  4.07 GB/s on four (`bench/memband`). Generating a token means reading *every* weight once, so decode
  speed is simply `bandwidth ÷ bytes of weights per token`. With 437 MB of 4.5-bit weights that ceiling
  is about 8.9 tokens/s; nothing done on the cores can move it.
- **The Cortex-A72 has NEON but none of the newer instructions** (`asimddp` dot-product, `fp16`,
  `i8mm`). Every modern ARM inference library — llama.cpp, MNN, ONNX Runtime, LiteRT — keeps its fast
  paths behind those features and falls back to plain NEON on this chip. LiteRT-LM's prebuilt binaries
  do not run on it at all.
- **Qwen3.5-0.8B is a hybrid.** 18 of its 24 layers are Gated-DeltaNet linear attention with a fixed
  19 MB recurrent state; only 6 are full attention with a KV cache. Memory barely grows with context —
  but every token has to update that recurrent state, and the model ships a one-layer MTP draft head.

So: the only two ways past ~8 tokens/s are (a) fewer bytes per token, i.e. a quantization format that
loses no quality, and (b) producing more than one token per read of the weights — speculative decoding.
pinfer does both, and spends the rest of its effort making sure the core streams weights at the wall.

![Kernel bandwidth](docs/figures/kernel_bandwidth.png)

## 2. How pinfer works

### 2.1 A weight format laid out for the A72's loads
Weights are converted once (`tools/convert.py`) into a `.pq` file: rows are tiled four at a time into
256-column superblocks, each starting with one 64-byte cache line of scales, so a thread streams one
contiguous range and every scale line lands on a cache-line boundary. Three quantized formats exist;
their quality decided the default:

![Quantization](docs/figures/quantization.png)

- **Q4K** (4.5 bits/weight; asymmetric 32-column groups with 6-bit sub-scales under per-row fp16
  super-scales, llama.cpp "k-quant" style) matches Q4_K_M quality and is the shipped default.
- **Q4** (symmetric) is the same size and speed but loses 17% in perplexity — exactly like llama.cpp's
  Q4_0. The loss comes from the *layer* matrices, not the output head (measured by mixing formats).
- **Q8** is indistinguishable from bf16 (17.96 vs 17.93) at half the bytes.

Activations are quantized to int8 per 32 columns at run time; the kernels use `SMULL`/`SMLAL` into
int32 accumulators (one 128-bit NEON instruction per cycle on the A72, which is why int8 beats fp32 here).

### 2.2 Kernels that run at the memory wall
Per-dtype thread policy (Q8 on one thread already saturates DRAM; Q4K needs three to keep up with
the dequantization), a pinned thread pool with ~0.3 µs dispatch, and a batched path that reads the
weights once for up to 64 vectors. Things measured and rejected: software prefetch (22% *slower* on
one thread — the hardware prefetcher already keeps up), 8-wide passes (register spills), two
accumulators per row (no gain). Lab notebook: `engine/bench/KERNELS.md`.

### 2.3 Exact speculative decoding with the model's own draft head
![Speculative decoding](docs/figures/speculative_schematic.png)

Because the Pi is bandwidth-bound, scoring four positions in one pass costs 1.24× scoring one
(160 ms vs 128 ms). Qwen3.5's MTP head drafts 3 tokens for ~27 ms; the full model verifies all of them
in one batched pass and keeps the longest correct prefix plus one token of its own. Every kept token is
the full model's own greedy choice, so the output is token-for-token identical to plain decoding.

![MTP acceptance](docs/figures/mtp_acceptance.png)

Two details matter in practice. The draft step only considers the first 32,768 token ids (where
English and code live): acceptance is unchanged and the draft is 5× cheaper. And rejected drafts must
roll back the 19 MB recurrent state exactly — pinfer logs each rank-1 state update during the verify
pass and replays only the accepted ones, instead of copying state.

![Speculative k](docs/figures/speculative_k.png)

### 2.4 Where the remaining time goes
![Step breakdown](docs/figures/step_breakdown.png)

After vectorizing the recurrence (a NEON gated-delta-rule kernel, chunk-major so a 16 KB state chunk
stays in L1) and the per-token ops (norms, SiLU with a 1-ulp `exp`, RoPE, attention, argmax), a plain
step is 119 ms of weight streaming plus 10 ms of everything else — about 12% from the DRAM floor.

## 3. Results

![Progress](docs/figures/progress.png)

![Speed vs quality](docs/figures/speed_vs_quality.png)

![Prefill](docs/figures/prefill_speed.png)

**Standard runtimes on the same board** (`bench/BASELINES.md`): MNN 3.6.1 is the strongest, 6.67 tok/s
decode and 30.9 prefill; llama.cpp v0.7.0 reaches 5.4–5.6 decode; ONNX Runtime GenAI 0.18 int4 3.1;
LiteRT-LM's prebuilt wheels abort on the A72 (they require ARMv8.2 features); PyTorch fp32 runs but
needs 4.8 GB of RAM.

**Long contexts** are the current work (Phase 6). The first sweep showed decode falling 29% and the
speculative gain vanishing at 4096 tokens because the attention path, not the KV traffic, was the
bottleneck; the rewritten attention (in progress, bit-exact) holds 6.5 tok/s at 4096 and 32 tok/s
prefill at 2048.

![Context length](docs/figures/context_length.png)

## 4. What did not help

Measured, logged and kept for the record:

![OS experiment](docs/figures/os_experiment.png)

- **A custom kernel** (6.18.50-pinfer: `NO_HZ_FULL`, isolated tickless cores 1–3, 2 MB huge pages for
  the weights) — all within 1%. The engine is DRAM-bound and the A72 already hides the page-walk cost of
  a sequential stream. The kernel is built, tested through three one-time `tryboot`s and left installed
  side by side but inactive (`os/kernel/`).
- **16 KB pages**: impossible — the Cortex-A72 has no 16 KB translation granule (Arm TRM r0p3).
- **Software prefetch, 8-wide batched passes, Q8 overrides for sensitive layers** (no quality gain over
  plain Q4K), **a Q8 output head** (−0.2 perplexity for −11% speed).

## 5. Correctness: how we know the output is right

- **Golden reference.** The official `transformers` implementation run in fp32 on the PC produces every
  layer's hidden state, the logits and a greedy continuation; the engine reproduces them with a worst
  per-layer relative error of 3e-6 and 32/32 identical tokens (`tests/test_golden.py`).
- **Bit-exactness between paths.** The batched kernel used for prefill and verification is byte-identical
  to the single-token kernel for every format and batch size (checked by `memcmp` in `engine/bench/kbench`).
  An early version was not — the two paths rounded the Q4K offset term differently — and speculative output
  diverged from plain greedy at a near-tie on one chat prompt. That bug is fixed and guarded by
  `engine/bench/spec_exact.sh` (20/20 model × prompt × k cases on the Pi).
- **The MTP wiring is verified**, not assumed: feeding the pre-norm hidden state or swapping the
  concatenation order collapses acceptance to 55% and 0.3% (figure above).
- **Quality** is measured with llama.cpp's own perplexity protocol (`engine/tools/pinfer_ppl`, cross-checked
  against `llama-perplexity` on identical text: 15.1743 vs 15.1695).
- **Validity of every speed number.** `harness/pi.py bench` records temperature, clocks and the
  firmware throttle flags before and after each run and marks the run invalid if the board throttled or
  could not report; only valid runs are quoted.

## 6. Using the library

```bash
# PC (WSL): build the engine for the Pi (NEON kernels) and convert a model
make -C engine pi MATVEC=fast
python tools/convert.py models/Qwen3.5-0.8B models/pq/qwen35-0.8b-q4k-headq4k.pq --quant q4k

# Pi: C CLI
./engine/pinfer_run -m models/qwen35-0.8b-q4k-headq4k.pq -t 4 -n 64 --ids-file data/prompt64.ids \
    --spec 3 --draft-vocab 32768 --bench

# Pi: Python package (wheel in build/dist/, install per docs/library.md)
pinfer chat -m models/qwen35-0.8b-q4k-headq4k.pq --tokenizer models/qwen35-0.8b-tokenizer \
    -p "Why is the sky blue?" --speculative --stats
```

The C API (`engine/include/pinfer.h`) takes token ids and returns logits or generated ids; the Python
package adds the tokenizer, sampling, the Qwen chat template, streaming and `generate(speculative=True)`.
Full guide: [docs/library.md](docs/library.md). Not published yet; no license chosen.

## 7. Methodology and records

Everything that touched the Pi went through `harness/pi.py` (check / push / run / bench / snapshot),
which appends every command to `logs/harness.jsonl`, keeps full outputs in `logs/runs/`, and serializes
benchmarks with a Pi-wide lock so parallel work never measures on top of itself.

| Question | File |
|---|---|
| What changed, when? | `CHANGELOG.md` |
| What happened on a given day, and why (problems → root cause → fix)? | `docs/journal/YYYY-MM-DD.md` |
| How does the Pi differ from stock, and how do I undo each change? | `docs/pi-state.md` |
| Every command sent to the Pi, and its output | `logs/harness.jsonl`, `logs/runs/` |
| Full Pi configuration at a point in time | `logs/snapshots/<ts>_<label>/` |
| The numbers behind every figure | `notes.md` → `bench/results/*.json` |
| Kernel / speculation / vectorization / OS / baselines lab notebooks | `engine/bench/KERNELS.md`, `engine/SPEC.md`, `engine/CORE.md`, `os/kernel/KERNEL.md`, `bench/BASELINES.md` |

Figures are rendered from these records by `tools/make_figures.py` and `tools/make_figures2.py`.

## 8. Repository layout
```
engine/include/pinfer.h   public C API              engine/src/      loader, forward pass, MTP + speculative
engine/FORMAT.md          .pq format + exact math                     decoding, NEON kernels and per-token ops
engine/tools/             pinfer_run (CLI, bench), pinfer_ppl (perplexity)
engine/bench/             kbench, lbench, spec_exact.sh, KERNELS.md
python/                   pip package `pinfer` (ctypes, tokenizer, chat, streaming, CLI)
tools/                    convert.py, quant.py, make_figures*.py
ref/                      fp32 golden reference    tests/           golden, spec, Python suite (80 tests)
bench/                    llama.cpp / MNN / ORT / LiteRT / PyTorch baselines, perplexity, results/*.json
harness/pi.py             PC-side control of the Pi, auto-logged
os/stock/  os/pc/         flashing + headless first boot; direct-cable network
os/kernel/                custom split-core kernel build + tryboot scripts (built, tested, inactive)
docs/                     design, library guide, journal, pi-state, figures
```

## 9. Status and roadmap
- Done: engine, kernels, exact speculative decoding, Python package and CLI, baselines, OS experiment.
- In progress (Phase 6): long-context attention, fp16/int8 KV cache options, state save/restore for
  multi-turn chat without re-reading history, adaptive draft count.
- Open decisions: license; public release.
- Hardware setup from scratch: `os/stock/` scripts and `docs/journal/2026-10-21.md`.
