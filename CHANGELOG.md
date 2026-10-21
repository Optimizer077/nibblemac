# Changelog

All changes to this project **and to the Pi** are recorded here, newest first.
Details for each day: `docs/journal/YYYY-MM-DD.md`. Pi configuration changes: `docs/pi-state.md`.

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
