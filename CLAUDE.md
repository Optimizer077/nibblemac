# pinfer project rules

Pi-4-native Qwen inference engine + split-core Linux. Design: `docs/design.md`.
Plan: `task_plan.md`. Findings: `notes.md`.

## Logging and documentation (user requirement: log everything carefully, document every update)
1. Talk to the Pi **only through `python harness/pi.py`** (check/push/run/bench/snapshot).
   Every call is auto-logged to `logs/harness.jsonl`; `run` output goes to `logs/runs/`.
   Do not use raw `ssh` for anything that changes the Pi.
2. Before changing Pi configuration: `python harness/pi.py snapshot before-<change>`.
   After: add a row to `docs/pi-state.md` with the exact undo command.
3. Every working session: append to `docs/journal/YYYY-MM-DD.md` (timeline with evidence
   paths, findings, problems → root cause → fix, open items) and add a `CHANGELOG.md` entry.
4. Benchmarks: only through `harness/pi.py bench`; quote only `valid=true` results; add
   them to the baseline table in `notes.md` with the result file path.
5. Report failures as failures, with the log path. Never delete logs or results.
5b. The Pi is shared by parallel agents: every `bench` holds the Pi-wide lock
   (`/tmp/pinfer-bench.lock`); use `run --lock` for anything heavy or timed. Never time
   anything on the Pi without the lock. Cross-compile on the PC; don't build big things on the Pi.
6. Never commit secrets: `harness/pi.secret`, `harness/pi.toml`, Wi-Fi/PSK material.

## Environment
- Cross-compile in WSL Ubuntu (`make`), static aarch64 binaries → `build/aarch64/`.
- From Git Bash, call WSL with `MSYS_NO_PATHCONV=1`.
- Pi control address: **10.42.0.2** over a direct PC↔Pi cable (PC is 10.42.0.1).
  Fallback over Wi-Fi: 192.168.1.166. `pi4.local` resolves wrongly on this network.
- The Pi's internet goes over Wi-Fi, which drops every few minutes: retry downloads, and run
  long jobs on the Pi so that a drop doesn't kill them.
- PC-side changes (need UAC) are logged in `logs/pc/` and listed in `docs/pi-state.md`.
