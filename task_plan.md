# Task plan: pinfer (Pi 4 native LLM runtime + split-core OS)

Design: [docs/design.md](docs/design.md). Findings: [notes.md](notes.md).
Running under `/loop` (self-paced, 2026-10-21): "get SOTA optimization whatever way it is but
do it" — own Pi-4-native library for **Qwen3.5-0.8B**, released as a reusable library, pushed
to max speed, every step measured vs llama.cpp + other standard runtimes, everything logged.

## Phase 0 — Setup ✅ (2026-10-21)
Toolchain, harness, flash, first boot, direct cable, EEPROM, logging system. See CHANGELOG.

## Phase 1 — Baselines (in progress)
- [x] memband: ~4.2 GB/s, saturated by 1 core
- [x] llama.cpp v0.7.0 static Cortex-A72 build; Qwen3.5-0.8B GGUF Q8_0/Q4_0/Q4_K_M
- [ ] llama-bench sweep (running) → table in notes.md
- [ ] Other standard runtimes (research agent running) → install + bench the viable ones
- [ ] Quality baseline: llama-perplexity per quantization on a fixed text

## Phase 2 — pinfer v0: correct (P0)
- [ ] M0 golden reference: transformers 5.20.2 fp32 on PC → token ids, per-layer hidden
      states, logits, greedy continuation (`ref/dump_reference.py`)
- [ ] M1 converter `tools/convert.py`: safetensors → `.pq` (bf16 text weights, aligned)
- [ ] M1 engine: C11 scalar fp32 compute, bf16 weights; ops: RMSNorm(1+w), gated RMSNorm,
      causal conv1d(k=4)+SiLU, gated delta rule (recurrent), gated full attention with
      partial RoPE (64/256 dims, θ=1e7), SwiGLU MLP, tied head
- [ ] M1 test: engine vs golden on PC (x86) and Pi — logits max-abs-diff, top-1 match

## Phase 3 — pinfer fast (P0)
- [ ] NEON bf16 matvec + threading (4 cores) → first Pi tok/s
- [ ] Q8 (group int8, SMLAL/SADALP — no SDOT on A72) → vs llama.cpp Q8_0
- [ ] Q4 kernels (weights pre-tiled for A72 loads) → vs llama.cpp Q4_0/Q4_K_M
- [ ] Mixed precision by sensitivity; output head (≈1/3 of bytes) at lower bits
- [ ] Each step: tok/s + perplexity, logged

## Phase 4 — beyond the bandwidth ceiling (P1)
- [ ] MTP self-speculative decoding (Qwen3.5 ships `mtp.*`, 1 layer)
- [ ] Two-stage output head (cheap approximate logits → exact rescoring of a shortlist)
- [ ] Prompt-lookup / n-gram drafting
- [ ] OS: performance governor, isolcpus, own kernel (NO_HZ_FULL, hugetlbfs), overclock study

## Phase 5 — release (P1)
- [ ] Stable C API (`pinfer.h`), Python package (ctypes + HF `tokenizers`), CLI, docs,
      tests, versioned releases (publishing anywhere public needs user approval)

## Decisions
- 2026-10-21: Level 2 (split-core Linux), bare metal later (user).
- 2026-10-21: C11 + NEON, Debian trixie, Python harness over SSH (call).
- 2026-10-21: Target Qwen3.5-0.8B hybrid (user); Qwen3-0.6B kept as reference.
- 2026-10-21: Compare with llama.cpp + other standard runtimes (user).
- 2026-10-21: Tokenization in Python (`tokenizers`); C API takes token ids (call).

## Blockers
- None. Pi Wi-Fi drops every few minutes (reason=7) — affects only Pi-side downloads.
