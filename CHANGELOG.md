# Changelog

All changes to this project **and to the Pi** are recorded here, newest first.
Details for each day: `docs/journal/YYYY-MM-DD.md`. Pi configuration changes: `docs/pi-state.md`.

## 0.2.0 (unreleased)
- ABI: `pinfer_params.kv_dtype` and `pinfer_spec_params.adaptive` (added after 0.1.0) make this
  0.2.0, SONAME `libpinfer.so.0.2`; binaries built against the 0.1 header must be rebuilt.
- State blobs are format 2 and carry a fingerprint of the model file; 0.1 blobs are refused.
- Stricter `.pq` validation (every tensor's size, header bounds); the caller's CPU affinity is
  no longer changed. Details: `engine/HARDENING.md`. `docs/release-0.1.0.md` stays the 0.1.0 note.

## 2026-10-22

### Hardening (issue #11, hardening agent, 07:00–; notebook `engine/HARDENING.md`)
- Fixed (loader): every directory entry's `nbytes` must equal the size its dtype and shape
  imply, quantized dtypes included (rows % 4, cols % 256); unknown dtypes refused; header
  dimensions bounded (`PQ_MAX_*`), element counts and products cannot wrap; `eos_id`, finite
  `rms_eps` / `rope_theta`, 8-byte-aligned directory; `lin_v_dim % 32` (`kern_lin.c` chunks).
  `mtp.c` uses `model.c`'s tensor views. Before, on the Pi: a short quantized tensor loaded and
  the matmul read past it; `lin_v_dim 80` crashed (SIGSEGV).
- Fixed (thread pool): `pf_pool_create` no longer pins the calling thread to CPU 0 (later
  threads of the application inherited that single core); workers are pinned to the 2nd..n-th
  CPU of the caller's mask, at most that many threads.
- Fixed (state): blobs carry `pf_model_fingerprint` (header, directory, sampled weights), state
  format 2; a state of another model file with the same dimensions is refused. Python
  `pinfer.state` names the cause (version, kv_dtype, other file).
- Fixed (`pinfer_ppl`): token ids validated before use (out-of-bounds logit read), `--chunks 0`
  refused (printed NaN), odd-sized token files refused.
- Fixed: `PF_FWD_STATS` / `PF_KERN_STATS` flags set atomically, tables under a mutex.
- Changed: version 0.2.0 (`pinfer.h`, SONAME `libpinfer.so.0.2`, Python package, wheel names).
- Added: `make -C engine host SAN=address,undefined`; `tests/test_malformed.py` (43 cases;
  baseline 18/43, now 43/43 under ASan+UBSan and on the Pi); `tests/test_long_context.py` +
  `ref/long_context_reference.py`; pytest `test_affinity.py`, `test_version.py`, 5 state cases.
- Measured: long context vs fp32 transformers to 8192 tokens: worst rel. error 2.0e-6, top-1
  18/18 (no RoPE or position bug). Speed unchanged: prompt64 plain / `--spec 3` 7.888 / 16.713 tok/s vs
  `pinfer_run_fast` 7.891 / 16.717 (3 interleaved rounds, `speed-ab-hard2_20261022_081251.json`);
  output bit-identical (spec_exact 20/20, dumps 5/5 identical).
- Pi: experimental binaries `engine/pinfer_run_hard2`, `pinfer_ppl_hard2` (scratch `tmp/hard/`
  removed at the end); `engine/pinfer_run_fast` untouched.

### Merged (lead, 06:50)
- Context-window work merged to `main` after the full gate (host golden / spec 81/81 / state /
  pytest 85/85; Pi build md5-identical to the gated `pinfer_run_ctx7`). The Pi's shared
  `engine/pinfer_run_fast` and `pinfer_ppl_fast` are now this build (previous: `*.pre-ctx`).
- Defaults decided: fp32 KV cache stays the default (exact); adaptive speculation stays opt-in.
- Fixed: `tests/test_spec.py` defaulted to a stale binary (`build/host/engine-spec/`).
- Docs: README long-context section and headline row, before/after `context_length` figure,
  RESULTS.md §6 final tables, CONTRIBUTING source-tree pytest command.

### Added (context window — details `engine/CONTEXT.md`)
- Attention over the KV cache rewritten (`engine/src/forward_attn.c`, `forward_kv.c`): K/V rows
  streamed once per block of 8 tokens and shared by the 4 query heads of a KV group; threads
  over (KV group × row half) for the scores and (KV group × head half) for softmax + P·V.
  Bit-identical per element (host md5 and Pi NEON twin checks), spec_exact 20/20.
- `pinfer_params.kv_dtype` (0 fp32, 1 fp16, 2 int8 + per-row scale), `pinfer_run --kv-dtype`,
  `pinfer_ppl --kv-dtype`, Python `Model.load(kv_dtype=)`, CLI `--kv-dtype`.
- State snapshot API: `pinfer_state_size / save / load`, continuation
  `pinfer_generate_greedy_cont`; `pinfer_run --save-state / --load-state`; Python
  `Model.save_state() / load_state()`, `context_tokens`, speculative `generate` continues from
  the context; `pinfer chat` keeps the state between turns (`--reprefill` for the old way).
- Adaptive speculation (`pinfer_spec_params.adaptive`, `--adaptive`, `spec_adaptive=`): the
  engine measures step / verify / draft costs and first-draft acceptance and picks k in 0..k
  per round, probing again in plain mode; output unchanged.
- Attention, second round: register-blocked P·V on an L1 block of 8 rows, two heads per pass
  (`pf_attn_pv_blk`), and single-token fp16 dots that widen K in registers (`pf_dot4_f16`);
  both bit-identical. Tried and dropped: software prefetch of the next rows (−1…−11%).
- Adaptive speculation estimator: decayed hits / trials over all drafts with a 2/3 prior
  (the first version collapsed to plain decoding after one early miss).
- Tests: `tests/test_state.py` (C round trip, 9 cases), `tests/python/test_state.py` (4),
  `tests/test_spec.py` configs `k3adapt` and `k3b2` (verify pass with a prefill batch < k+1).
- `bench/common/ctx{64,8192}.ids`, `engine/bench/ctx_check.sh`.

### Fixed
- Speculative verify pass with `-b` smaller than k+1 corrupted the rollback log (pre-existing);
  `draft_vocab` 1–3 produced an empty draft head; the speculative prefill could consume part of
  an oversized prompt before returning `PINFER_E_CTX_FULL`.

### Measured (Pi 4, Q4K, 4 threads, 32 generated; before → after)
- Plain decode tok/s: ctx 512 6.20 → 7.69, 2048 4.12 → 7.13, 4096 → 6.48, 8192 → 5.42
  (64: 7.81 → 7.92). Prefill tok/s: 512 21.6 → 36.8, 2048 11.6 → 32.3, 4096 28.0, 8192 22.0.
  Spec k=3 tok/s: 512 9.04 → 15.24, 2048 2.39 → 7.47, 4096 8.47, 8192 7.17.
- llama.cpp Q4_K_M decode at depth 2048 / 4096: 4.32 / 3.43 tok/s (pinfer 7.13 / 6.48).
- KV dtypes: perplexity within noise (8×2048: fp32 13.985, fp16 13.983, int8 13.989); fp16
  teacher-forced top-1 100% / KL 3.5e-8. fp16 decode at 8192 5.72 vs fp32 5.33 tok/s, RSS −97 MB.
- Adaptive k=3 (64 generated): 9.57 tok/s at 2048 (fixed 9.13, plain 7.09), 7.53 at 4096
  (fixed 7.38, plain 6.11), 16.42–16.82 on prompt64 (fixed 16.72–16.79).
- State: 2048-token state 70.5 MB, save 94 ms, load 90 ms (vs 63 s prefill); chat turn 2
  0.68 s instead of 2.10 s.

### Added
- Private GitHub repository `Optimizer077/piquant`; result figures in `docs/figures/` (`tools/make_figures.py`).
- Engine: NEON per-token ops (`forward_ops.c`, bit-exact scalar twins), threaded attention/MLP,
  `pf_pool_run`, `--bench` warm-up; opt-in `PINFER_HUGEPAGES`; Phase 6 context-window plan.
- `bench/BASELINES.md`: MNN / ORT GenAI / LiteRT-LM / PyTorch attempts and results.
- `tools/convert.py --quant {bf16,q8,q4,q4k} --head ... --override REGEX=DTYPE` (quantized
  `.pq` files via `tools/quant.py`); `engine/tools/pinfer_ppl` (llama-perplexity-compatible
  perplexity, cross-checked vs llama.cpp); `bench/quality/make_tokens.py`, `ppl_pc_c8.sh`.
- Pi wheel with the fast kernels (`manylinux_2_34_aarch64`), installed into `~/pinfer/venvs/pinfer`.

### Measured (Pi 4, Qwen3.5-0.8B, 64+64 tokens)
- **Vectorized engine (02:40):** Q4K default **7.88 plain / 16.49 speculative (k=3) / 37.4 prefill tok/s**;
  verify ratio 1.28; exactness 20/20; wheel pytest 80/80 on the Pi. Context 4096: 5.56 / 5.69 / 18.1.
- **Final kernels (01:10):** Q4K default 7.26 tok/s plain, **13.35 tok/s speculative (k=3)**, prefill
  26.1 tok/s, perplexity 18.48; speculative == greedy 20/20 on the Pi.
- pinfer Q4+Q4 head: 6.88 tok/s plain, **12.56 tok/s with exact MTP speculation** (k=3,
  draft vocab 32768) — vs MNN int4 6.67, llama.cpp Q4_0 5.59.
- Perplexity (wikitext-2, 8 chunks): pinfer mix-Q4K 18.50; llama.cpp bf16 17.93, Q8_0 17.97,
  Q4_K_M 18.59, Q4_0 21.25.

### Fixed
- Speculative decoding diverged from plain greedy at near-ties with the NEON kernels (batched
  verify pass rounded the Q4/Q4K offset term differently): kernels made bit-identical for every
  batch size, checked by kbench and `engine/bench/spec_exact.sh`.
- Harness: locked Pi jobs are killed after 20 min (a 38-min ORT run had stalled four agents).

## 2026-10-21

### Added (MTP self-speculative decoding — details `engine/SPEC.md`)
- `engine/src/mtp.{c,h}` (Qwen3.5 MTP draft head; semantics from vLLM `qwen3_5_mtp.py` +
  llama.cpp `qwen35.cpp`), `engine/src/spec.{c,h}` (draft k with the MTP chain, verify k+1 in
  ONE batched pass, exact rollback of linear-attention state by replaying logged rank-1
  updates — no state snapshots). `forward_lin.c` loops head-outer (state stays in L2),
  `forward.c` exports its attention/MLP blocks and `pf_forward_h` (final-normed hiddens).
- `pinfer.h` (additive): `pinfer_spec_params`, `pinfer_spec_stats`, `pinfer_default_spec_params`
  (k=2, draft_vocab=32768), `pinfer_has_mtp`, `pinfer_generate_greedy`,
  `pinfer_generate_greedy_ex` + `pinfer_gen_opts` (stop ids, per-token callback, tokens kept on
  context overflow).
- `pinfer_run --spec k [--draft-vocab N] [--dump-mtp f]`; RESULT json gains rounds / drafted /
  accepted / tokens_per_pass / per-pass timings.
- `ref/mtp_reference.py` + `ref/golden/qwen35-0.8b-mtp/`, `tests/test_spec.py` (PASS: MTP logits
  ≤ 4e-6 rel vs reference; spec output == plain greedy for 9 prompts × k=0..4 + draft vocab).

### Measured (speculative decoding, bf16, 64 tokens, Pi)
- Reference MTP acceptance 87.3% first draft (651 positions); controls 54.5% / 0.3%.
- NEON kernels, 4 threads, k=2 + draft vocab 32768: 4.13 / 3.27 / 4.50 tok/s vs 2.31 / 2.33 /
  2.33 plain (1.79× / 1.41× / 1.93×, Dickens / history / code), identical tokens.
  Scalar kernels: speculation 0.55–0.74× (compute-bound, expected).

### Added (release / packaging, pinfer 0.1.0 — not published)
- `engine/Makefile`: `lib` / `host-lib` / `pi-lib` (`libpinfer.a` + `libpinfer.so`, `-fPIC`,
  only `pinfer_*` exported via `engine/pkg/pinfer.map`; static archive localized with
  `ld -r` + `objcopy`), `install PREFIX= [TARGET=host|pi] [DESTDIR=]` (+ `pinfer.pc`),
  `examples`, `check-lib` (`engine/pkg/check_lib.sh`: exports, NEEDED, glibc ≤ 2.41 for the Pi),
  `wheel-host` / `wheel-pi` → `build/dist/`. Library objects use `-fno-math-errno`
  (otherwise `sqrtf@GLIBC_2.43` from WSL's glibc makes the .so unloadable on the Pi).
- `python/`: package `pinfer` (ctypes binding, tokenizer, sampling, Qwen chat template,
  streaming, speculative via `pinfer_generate_greedy`), CLI `pinfer`, `pyproject.toml` +
  `hatch_build.py` (bundles the .so; manylinux tag derived from its glibc symbol versions).
- `tests/python/` (80 tests), `examples/c/simple.c`, `examples/python/generate.py`,
  `docs/library.md`, `docs/release-0.1.0.md` (draft).

### Measured (release)
- pytest 80/80 on host (`logs/release/host_pytest_final.log`) and on the Pi
  (`logs/release/pi_pytest_final.log`); greedy ids via Python == golden 32/32 (plain and
  speculative).

### Pi (release)
- User-level venv `~/pinfer/venvs/pinfer` (offline install from `~/pinfer/wheels/pinfer/`),
  tokenizer files, golden meta.json, tests, C example — listed in `docs/pi-state.md`. No apt.

### Added
- `docs/design.md`: design for pinfer (Qwen engine + split-core Linux, level 2).
- `harness/pi.py`: `check`, `push`, `run`, `bench`, `snapshot`; `harness/runlog.py`: automatic
  audit log (`logs/harness.jsonl`), full `run` output in `logs/runs/`.
- `bench/memband/memband.c` + `Makefile`: DRAM bandwidth benchmark (static aarch64 build).
- `os/stock/render_firstboot.py`, `os/stock/flash_disk.ps1`, `os/stock/compare_disk.ps1`:
  headless first-boot config, verified raw flashing, block comparison.
- `docs/pi-state.md`, `docs/journal/`, this changelog, project `CLAUDE.md` logging rules.

### Pi
- Flashed Raspberry Pi OS Lite 64-bit 2026-10-06 (trixie) to the Crucial X6 SSD (old
  install erased with user approval). First boot 21:03:53 KST.
- Change #1: Wi-Fi power save disabled.
- Change #2: eth0 static 10.42.0.2/24 for a direct PC↔Pi cable; harness now uses it
  (Wi-Fi drops every few minutes with reason=7 — kept only for the Pi's internet).
- PC change P1: Windows Ethernet static 10.42.0.1/24 (`os/pc/eth_link.ps1`, has `-Undo`).

- Change #3: bootloader EEPROM 2026-05-17 → 2026-09-23.
- Placed llama.cpp v0.7.0 binaries and Qwen3.5-0.8B GGUFs under `~/pinfer`.

### Decided
- Target model: **Qwen3.5-0.8B** (hybrid linear/full attention), user choice. Qwen3-0.6B kept
  as reference. Compare against llama.cpp and other standard runtimes.

### Added (baseline tooling)
- `bench/llamacpp/build.sh` (pinned llama.cpp, static Cortex-A72 build + host quantizer),
  `make_ggufs.sh` (HF → GGUF bf16 → Q8_0/Q4_0/Q4_K_M), `llama_bench.sh` (thread sweep).
- `harness/pi.py push --dest <subdir>`.

### Measured
- memband: 4.20 / 4.08 / 4.07 GB/s with 1 / 2 / 4 threads (`bench/results/memband_20261021_210615.json`).

### Fixed
- Harness: pushed files lacked the exec bit; unknown throttle state counted as valid;
  stderr not saved; `run` with `;` could run outside the work dir.
- memband: threads not pinned to CPUs; slice alignment for odd thread counts.
