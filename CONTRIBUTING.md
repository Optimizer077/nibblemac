# Contributing to nibblepi / pinfer

This project measures one physical board. The rules below keep every number reproducible, every change to
the Pi reversible, and the engine's output exact. They are the project rules from [CLAUDE.md](CLAUDE.md),
with the incidents that produced them. Setup and commands: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## 1. Talk to the Pi only through the harness

- Use `python harness/pi.py check | push | run | bench | snapshot` for everything. Every call is appended
  to `logs/harness.jsonl` (command, host, exit code, duration, outputs, SHA-256 of pushed files); `run`
  output is saved in full to `logs/runs/<timestamp>_run.log`; `bench` writes
  `bench/results/<name>_<timestamp>.json`. Nothing depends on someone remembering to log
  (`harness/runlog.py`).
- Do not use raw `ssh` for anything that changes the Pi.
- Never delete logs or results. Report a failure as a failure, with its log path.

## 2. Benchmarks: the lock, the cap, validity

- Time things only with `bench`. Use `run --lock` for heavy untimed work (test suites, checksums of large
  files). Both hold the Pi-wide lock `/tmp/pinfer-bench.lock`, so parallel agents never measure on top of
  each other. Two kernel measurements were spoiled by other agents' unlocked jobs, a push and a checksum
  (`engine/bench/KERNELS.md` v1 and the prefetch sweep), which is why heavy work takes the lock too.
- Locked jobs are killed after 20 minutes (`timeout`, exit code 124). A 38-minute run once stalled four
  agents (`docs/journal/2026-10-21.md`, 23:05). Keep jobs short: 8-chunk perplexity, a few configurations
  per call.
- Push large files only when no benchmark is running. Cross-compile on the PC; do not build large things
  on the Pi.
- Quote only `valid=true` results and add them to the tables in `notes.md` with the result-file path.
  `valid` covers exit code, throttling and output parsing, not correctness: read `match`, `exact` or
  `identical` for that (`docs/REPRODUCE.md` §9).
- Give experimental binaries their own names on the Pi (`engine/pinfer_run_<topic>`).
  `engine/pinfer_run_fast` is the shared binary that everyone's comparisons use.

## 3. Changing the Pi (or the PC)

1. Before: `python harness/pi.py snapshot before-<change>`. It records packages, boot config, cmdline,
   services, network, EEPROM, disk layout and the kernel config hash in `logs/snapshots/<ts>_<label>/`.
2. Make the change through `harness/pi.py run`.
3. After: add a row to `docs/pi-state.md` with the date, the change, where it lives on the Pi, the
   **exact undo command**, and the evidence (snapshot, log). The Pi must stay reproducible from the stock
   image plus that list.
4. PC-side changes that need administrator rights are logged in `logs/pc/` and listed under
   "PC-side changes" in `docs/pi-state.md`. User-level files on the Pi (venvs, wheels, models) go in its
   "Files we placed" table with their undo.

## 4. Record every working session

- Append to `docs/journal/YYYY-MM-DD.md`: a timeline with evidence paths, findings, and each problem as
  problem → root cause → fix, plus open items.
- Add a `CHANGELOG.md` entry (sections used there: Added, Measured, Fixed, Pi, Decided).
- Update the lab notebook of the area you worked in (§8), `notes.md` for scoreboard results and
  `task_plan.md` for phase status.

## 5. Correctness rules for engine changes

**Anything on by default must be bit-exact** between the engine's own paths: a row computed in a batch
equals the same row computed alone, the NEON code equals its scalar twin, and the thread count does not
matter. Speculative decoding, prefix reuse and state restore all depend on it
([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §7). In practice:

- no fused multiply-add: keep `-std=c11 -ffp-contract=off` (`engine/Makefile`) and write separate
  multiply and add;
- reductions use a fixed lane order, combined as (s0+s1)+(s2+s3);
- threads split only independent rows, heads, channel ranges or KV groups, never a sum;
- private data layouts stay behind the `engine/src/ops.h` contract.

**Anything that changes outputs is opt-in** and off by default until its quality is measured, for
example `kv_dtype` fp16 / int8 (`engine/include/pinfer.h`).

**Tests before every engine commit:**

| Where | Command | Must show |
|---|---|---|
| Host (WSL, `MATVEC=ref`) | `make -C engine && /root/venv-ref/bin/python tests/test_golden.py` | PASS: per-layer error vs the fp32 reference, batched == token-by-token, greedy 32/32 |
| Host | `/root/venv-ref/bin/python tests/test_spec.py` | PASS: MTP head vs reference, `--spec k` == plain for every prompt |
| Host, state / spec / attention changes | `/root/venv-ref/bin/python tests/test_state.py` | PASS: restored state continues identically |
| Host, API or Python changes | `make -C engine host-lib && PYTHONPATH=python PINFER_LIB=$PWD/build/host/engine/lib/libpinfer.so PINFER_TEST_REQUIRE=1 /root/venv-pinfer-test/bin/python -m pytest tests/python -q` (source tree; without the two variables an installed wheel is tested instead) | all passed |
| Host, loader / format / state / tool changes | `make -C engine host SAN=address,undefined && python3 tests/test_malformed.py` | 43/43: broken `.pq` files, state blobs and token files refused cleanly, no sanitizer report |
| Host, attention / RoPE / position changes | `/root/venv-ref/bin/python tests/test_long_context.py` (fp32 reference up to 8192 tokens; ~45 min, 6.6 GB RAM) | worst rel. error ≤ 1e-3, top-1 18/18 |
| Pi, NEON build | `python harness/pi.py bench spec-exact-<topic> -- bash engine/spec_exact.sh ./engine/pinfer_run_<topic>` | 20/20 `"match":true` |
| Pi, kernel changes | `kbench` with `--check`, plus a run whose rows are all `"exact":true` | no failed check |
| Pi, forward-pass changes | `engine/bench/core2_check.sh`, `engine/bench/ctx_check.sh` | identical md5 / `"identical":true` |

The golden data in `ref/golden/` is gitignored; create it first with `ref/dump_reference.py` and
`ref/mtp_reference.py` (`docs/REPRODUCE.md` §7).

## 6. Commits

- Conventional Commits, as in the history: `type(scope): summary`, for example `feat(engine): …`,
  `feat(kernels): …`, `fix(harness): …`, `docs: …`, `chore: …`. Measurement and record commits use the
  repository's own types `bench:`, `plan:`, `notes:`, `journal:`.
- One coherent change per commit. A measurement commit includes its result files.
- Line endings are LF (scripts run on the Pi; `.gitattributes`), except `*.ps1` (CRLF).

## 7. No secrets, no large generated files

- Never commit `harness/pi.secret` (console password), `harness/pi.toml`, Wi-Fi passwords, PSKs or
  password hashes. `*.secret` and `harness/pi.toml` are gitignored; keep it that way.
- Generated or large data stays out of git: `build/`, `models/`, `*.pq`, `*.gguf`, `*.safetensors`,
  `images/`, `ref/golden/` (all gitignored).
- Scan the tracked files for secrets before pushing (done before the first push,
  `docs/journal/2026-10-22.md` 03:00).

## 8. Who owns what

| Area | Lab notebook | Main files |
|---|---|---|
| Matrix kernels, quantized formats | `engine/bench/KERNELS.md` | `engine/src/matvec_fast.c`, `kern_*.c`, `pq_quant.h`, `tools/quant.py`, `engine/bench/` |
| Speculative decoding | `engine/SPEC.md` | `engine/src/mtp.c`, `spec.c`, `ref/mtp_reference.py`, `tests/test_spec.py` |
| Engine core (forward pass) | `engine/CORE.md` | `engine/src/forward*.c`, `pinfer.c`, `model.c`, `matvec_ref.c`, `tests/test_golden.py` |
| Context window (Phase 6) | `engine/CONTEXT.md` | `engine/src/forward_attn.c`, `forward_kv.c`, `state.c`, `spec_adapt.c`, `tests/test_state.py` |
| Custom kernel and OS | `os/kernel/KERNEL.md` | `os/kernel/`, `os/stock/`, `os/pc/` |
| Baseline runtimes | `bench/BASELINES.md` | `bench/llamacpp/`, `bench/mnn/`, `bench/ort/`, `bench/litert/`, `bench/torch/` |
| Library and packaging | `docs/library.md` | `engine/Makefile` (lib, wheels), `engine/pkg/`, `python/`, `tests/python/` |
