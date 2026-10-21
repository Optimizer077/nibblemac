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
- [x] llama-bench sweep → notes.md (Q4_0 5.59, Q4_K_M 5.41, Q8_0 4.12 tok/s)
- [x] MNN int4 6.67, ORT GenAI int4 3.10 (baselines agent); [ ] LiteRT-LM, PyTorch pending
- [x] Perplexity protocol (wikitext-2, 8×512): llama.cpp bf16 17.93 / Q8_0 17.97 / Q4_K_M 18.59 / Q4_0 21.25

## Phase 2 — pinfer v0: correct ✅ (2026-10-21, commit e570591)
- [x] M0 golden reference: transformers 5.20.2 fp32 on PC → token ids, per-layer hidden
      states, logits, greedy continuation (`ref/dump_reference.py`)
- [x] M1 converter `tools/convert.py`: safetensors → `.pq` (bf16 text weights, aligned)
- [x] M1 engine: C11 scalar fp32 compute, bf16 weights; ops: RMSNorm(1+w), gated RMSNorm,
      causal conv1d(k=4)+SiLU, gated delta rule (recurrent), gated full attention with
      partial RoPE (64/256 dims, θ=1e7), SwiGLU MLP, tied head
- [x] M1 test: engine vs golden (rel err 2.9e-6, 32/32 greedy) on PC (x86) and Pi — logits max-abs-diff, top-1 match

## Phase 3 — pinfer fast ✅ (2026-10-21, commits 23054c6, 87c4734, 9ac4b83)
- [x] NEON bf16 matvec + pinned pool → 2.27 tok/s (84% of ceiling)
- [x] Q8_G32 (SMLAL/SADALP) 4.22 tok/s vs llama.cpp Q8_0 4.12; kernel 101–103% of memband
- [x] Q4_G32 / Q4K_G32 6.88 tok/s vs llama.cpp Q4_0 5.59, MNN 6.67
- [x] Mixed (v_proj, mtp.fc Q8) via `--override`; head Q4K vs Q8 measured; [ ] pick default by PPL
- [x] tok/s logged; [ ] perplexity for 3 of 4 formats (running)

## Phase 4 — beyond the bandwidth ceiling (in progress)
- [x] Kernels follow-up: bit-exact batched kernels, NEON recurrence (23054c6→af06041)
- [x] Engine-core vectorization (fba8a1a): 7.88 plain / 16.49 spec / 37.4 prefill; verify ratio 1.28; DRAM floor 12% away
- [x] MTP self-speculative decoding, exact: Q4 6.88 → 12.56 tok/s (k=3, draft vocab 32768)
- [x] Draft-vocab restriction (32768 ids) for the draft head — same acceptance, draft step 140→30 ms; [ ] two-stage exact head: deferred (low gain left)
- [ ] Prompt-lookup / n-gram drafting
- [x] Own kernel 6.18.50-pinfer: tryboot K1–K3 — isolation, nohz_full, THP all < 1% (DRAM-bound); not made permanent. [ ] overclock (needs user ok; only prefill/compute would gain)

## Phase 5 — release (in progress)
- [x] C API, libpinfer.a/.so, Python package, CLI, docs, 80 tests (9c58a81); fast wheel built
- [ ] License (user); [ ] rebuild wheels after kernel follow-up; [ ] publish (user approval)

## Phase 6 — context window (added 2026-10-22 02:00, user request)
Facts: only the 6 full-attention layers have a KV cache (2 KV heads × 256 × K+V × fp32 =
24 KB/token → 96 MB at 4096, 768 MB at 32K); the 18 linear-attention layers have fixed state
(19 MB), so memory grows slowly but attention reads 24 KB × context per token (at 8K context
that is 192 MB/token — comparable to the 437 MB of weights).
- [x] First sweep (notes.md "Context length"): 4096 ctx → 5.56 plain / 5.69 spec / 18.1 prefill (from 7.88 / 16.49 / 37.4); context agent redoing it cleanly + 8192
- [ ] KV cache precision: fp16 (halves traffic; bit-exact?) and int8 (quarter); quality check
      (PPL at 512 ctx + logit diff at long positions); make it the default if lossless enough
- [ ] NEON attention for long contexts (dot products over the cache, softmax, GQA 4:1 sharing of
      K/V reads across the 4 query heads of a KV head), batched for verify passes
- [ ] Long-prompt prefill: chunked attention, confirm RoPE/positions correct at > 4096 vs reference
- [ ] State save/restore API (`pinfer_state_size/save/load`): recurrent + conv + KV snapshot, so
      multi-turn chat and a system prompt never re-prefill; Python `Model.save_state()`/`load_state()`
- [ ] Prefix cache in the Python package (keep the last state; extend when the new prompt is a
      superset — already partly there as `reuse_prefix`)
- [ ] Bounded-context mode for endless chat (attention sinks + sliding window over the KV, state
      kept) — optional, quality trade-off, measured
- [ ] Compare with llama.cpp at 2048/4096 context (same prompt) for the scoreboard

## Decisions
- 2026-10-21: Level 2 (split-core Linux), bare metal later (user).
- 2026-10-21: C11 + NEON, Debian trixie, Python harness over SSH (call).
- 2026-10-21: Target Qwen3.5-0.8B hybrid (user); Qwen3-0.6B kept as reference.
- 2026-10-21: Compare with llama.cpp + other standard runtimes (user).
- 2026-10-21: Tokenization in Python (`tokenizers`); C API takes token ids (call).

## Blockers
- None. Pi Wi-Fi drops every few minutes (reason=7) — affects only Pi-side downloads.
