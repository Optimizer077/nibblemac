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
- [x] Prompt-lookup / n-gram drafting (`ngram`, `--ngram`, `spec_ngram`; issue #6, engine/DRAFT.md):
      code-edit chat answer +15-19%, 2048 context +16%, chat unchanged; gated on match length /
      agreement with the MTP draft; [ ] default (recommended adaptive k<=3 + ngram 7, lead)
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
- [x] KV cache precision: fp16 / int8 opt-in (`kv_dtype`); PPL within noise at 8×512 and 8×2048,
      teacher-forced top-1 100% / 99.4%. **Decision 2026-10-22: fp32 stays the default** — fp16
      changes greedy output at the first near-tie and is only 5–8% faster (CONTEXT.md §5)
- [x] Attention rewrite (`forward_attn.c`, `forward_kv.c`): each K/V row read once per 8-token block,
      shared by the 4 query heads of its group; bit-identical; 4096 decode 5.56 → 6.38 (fp32) / 6.65 (fp16)
- [x] Long-prompt prefill: 4096 → 28.0 tok/s (from 18.1), 8192 → 22.0
- [ ] RoPE/positions vs the fp32 reference beyond 2048 tokens (PPL 8×2048 matches; > 4096 unchecked)
- [x] State save/restore API (`pinfer_state_size/save/load`, `Model.save_state()/load_state()`):
      bit-exact continuation; 2048-token state 70.5 MB, restore 0.1 s vs 63 s prefill
- [x] Prefix reuse: `pinfer chat` keeps the state between turns; speculative generation reuses a
      matching prefix (`pinfer_generate_greedy_cont`)
- [x] Adaptive speculation (`--adaptive`, `spec_adaptive`): decayed acceptance estimator; beats fixed
      k=3 at 2048/4096 context; **default not changed yet** — measure on chat and code prompts first
      → 2026-10-22 (issue #2, DRAFT.md): confidence-gated chain, 0.984-1.056x the best fixed k on
      chat, prompt64, ctx2048 (was 0.90-1.04x)
- [ ] Bounded-context mode for endless chat (attention sinks + sliding window) — designed only
      (CONTEXT.md §7)
- [x] Compare with llama.cpp at 64–4096 depth: pinfer 6.38 vs 3.43 tok/s at 4096 (1.86×)

## Decisions
- 2026-10-21: Level 2 (split-core Linux), bare metal later (user).
- 2026-10-21: C11 + NEON, Debian trixie, Python harness over SSH (call).
- 2026-10-21: Target Qwen3.5-0.8B hybrid (user); Qwen3-0.6B kept as reference.
- 2026-10-21: Compare with llama.cpp + other standard runtimes (user).
- 2026-10-21: Tokenization in Python (`tokenizers`); C API takes token ids (call).

## Blockers
- None. Pi Wi-Fi drops every few minutes (reason=7) — affects only Pi-side downloads.

## Phase 7 — finalize piquant / pinfer 0.2.0 (owner: "complete Qwen and finalize it, then move to Cerno"; 2026-10-22 15:05)
- [x] Kernels round (prefill +19%, k=4 one pass): Pi gates on `pinfer_run_merge6`; adaptive cap 3 vs 4 decided from `kcap-merge6`; merge + push
- [x] Install the final build as the Pi's `engine/pinfer_run_fast` (keep the previous as `*.pre-<tag>`)
- [ ] (moved after Cerno) Overclock test, 2.0 GHz / `over_voltage=6` (approved): snapshot, `config.txt` backup, stability under a sustained 4-core load (throttled = 0x0, temperature), plain / speculative / prefill / 4096-context A/B vs 1.5 GHz; owner decides whether it stays
- [x] Final numbers everywhere: README headline + highlights, figures (decode, prefill, progress), RESULTS.md, notes scoreboard, issue #10
- [x] Release 0.2.0 (github.com/Optimizer077/piquant/releases/tag/v0.2.0): release notes (`docs/release-0.2.0.md`), wheels (aarch64 NEON, x86-64), tag, GitHub release; model asset only if the owner wants it (issue #9)
- [ ] Clean-up: Pi scratch binaries and data, finished agent worktrees and local branches, stale scratch files
- [ ] Then: resume the Cerno agent (repo `cerno-pi`), and later Gemma 4 (llama.cpp check first)

## Phase 8 — after 0.2.0: Cerno done, 4B speed, audit, overclock, README redesign (2026-10-22 16:55)
Owner (16:55): "README needs major visual and structure updates, but first finalize the testing;
look carefully into the network and see what can be further optimized to make it work faster."
- [x] Cerno 4B on the Pi (repo `cerno-pi`, public, credit kept): decision median 27.6 s vs llama.cpp
      112.4 s (4.5×), 2.4 vs 3.6 GB, scores within 0.011 of Cerno's engine; pushed `ac2a89f`
- [x] Generic engine pieces merged into nibblepi: untied head, `linear_attn.A`, `eval_select`,
      lossless GGUF import + synthetic test (`2a7fae1`, `fdd264d`)
- [x] README data-flow animations (`0d49e00`); README links to cerno-pi (`2ebc642`)
- [ ] P0 4B prefill speed (agent, branch `worktree-agent-aff7b9e370c62872c`): last-layer tail,
      groups of 7s + 5s; gates + before/after at 1.5 GHz; 0.8B no regression → merge
- [ ] P0 Overclock test 2.0 GHz / `over_voltage=6` (scripts `os/overclock/`, `d0d4e0c`): after the
      4B measurements, when no agent is on the Pi (reboot drops the lock)
- [ ] P1 Network audit (agent): `docs/network-audit.md`, time budgets per workload, ranked
      candidates with exactness class; then implement the top items (new agents)
- [ ] P1 README redesign (visual + structure) with the final numbers, after the tests above
- [ ] P2 Owner decisions: Cerno sample set / Decision Index (7 GB, one gated dataset); PyPI name;
      Gemma 4 (llama.cpp check first); social-preview upload
