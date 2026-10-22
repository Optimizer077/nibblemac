# checkpoint (in-engine state checkpoint for Cerno decisions; branch worktree-agent-a4d7c6bb45401045d from 624e00a)

Note: this file lives in my worktree (`.claude/worktrees/agent-a4d7c6bb45401045d/plan/board/checkpoint.md`);
the lead syncs it to the main board. Not committed.

- 18:00 [plan] audit C3 (docs/network-audit.md §4): `pinfer_checkpoint(ctx)` copies only the conv +
  DeltaNet states (~50.3 MB for the 4B) into one slot inside the context and remembers n_past /
  MTP rows; `pinfer_rewind(ctx)` copies them back and sets n_past back (KV rows below n_past are
  never rewritten while n_past stays >= the checkpoint; rows above are overwritten before they are
  read). Reset / state_load / generate_greedy(_ex) invalidate the slot. Python `Model.checkpoint()` /
  `rewind()`; cerno-pi `decide.py` uses it when the library has it, else save_state / load_state.
  Touches `state.c`, `pinfer.c` (reset/free), `forward.h` (3 ctx fields), `pinfer.h` (additive
  symbols, no version bump: same precedent as `pinfer_eval_select`). No kernel / forward change.
- 18:00 [plan] Pi work only after the lead's "window closed". Cerno decision benches will be named
  `cerno-ckpt-*` (not `ckpt-*`): `bench/results/cerno-*` is gitignored because Cerno results stay out
  of the public repo (owner decision); engine gates on 0.8B use `ckpt-*`.
- 21:20 [integrating] committed `43783be` (branch `worktree-agent-a4d7c6bb45401045d`, merged main `dd7bc9e`):
  `pinfer_checkpoint / pinfer_rewind / pinfer_checkpoint_clear / pinfer_checkpoint_n_past` (additive, no
  ABI change), `Model.checkpoint() / rewind()`, `pinfer_run --rewind-at S`, `tests/test_state.py` run D,
  5 pytest cases; cerno-pi `feat/checkpoint` `8261d06` (`Decider(checkpoint=None|False|True)`,
  `decide_bench.py --no-checkpoint`). Touches `state.c`, `pinfer.c` (reset/free), `forward.h` (4 ctx
  fields), `pinfer.h`; no kernel / forward change, so no merge conflict with speed4b / lut expected.
- 21:20 [measured] host (WSL ref build, bf16 0.8B): golden PASS (32/32, 0 error); smoke `--rewind-at 20`
  identical tokens plain and spec 3, checkpoint 49-55 ms (first touch of the 20 MB slot), rewind 2.3 ms on
  the PC. Other gates running (WSL load 88: slow). Pi: ssh refused since 21:09 (still booting?) — Pi
  gates and the before -> after decision timing wait for it; scripts ready in `tools/ckpt/` (`ckpt_pi.sh`:
  4B row A / rewind / save / load; `decide_ab.sh`: decisions ckpt vs state, same lib, the lead's 5-request set).
- 21:27 [question:lead] my WSL host gates were killed twice: all my `pinfer_run` / `test_spec.py` / `gates.sh`
  processes got SIGTERM at ~21:23 (WSL stayed up, load ~90), then `wsl.exe` itself failed at 21:26
  ("Wsl/Service/E_UNEXPECTED", i.e. the WSL service restarted / `wsl --shutdown`). Is someone restarting WSL
  or killing `pinfer_run` processes? Every agent's host gates die with it. I relaunch when WSL answers again.
  Pi: ssh refused 21:09-21:25, then timed out at 21:26 (power cycle in progress?).
- 21:46 [using] no slot: host gates only (golden / spec / state / pytest / malformed / check-lib / Pi build), one
  `pinfer_run` at a time, 4 threads (`--jobs 1`), bf16 0.8B mmap'd (page cache). Relaunched 21:45 after the
  WSL restart; previous chains killed (SIGTERM at 21:23, WSL crash 21:26, then the thread-cap changes).
  I will claim **slot B** here before the PC decision A/B (cerno-pi `decide_bench.py` on the host build with
  the bf16 0.8B as the 4B stand-in, 4 threads, ~2 GB, ~20 min) once the gates are through.
- 21:50 [integrating] merged main `f1089ab` (speed4b's prefill tuning `dcc2219` included) into my branch as
  `004c732`; the only code conflict was `pinfer_run.c` (speed4b's `--dump-last` next to my `--rewind-at`, both
  kept). Host gate chain relaunched 21:49 on the 24 GB / 12-core WSL (one `pinfer_run` at a time, 4 threads; the
  chain's own Pi build comes last); the Pi artifacts (`engine/pinfer_run_ck1`, `tmp/ck/lib/libpinfer.so`,
  NEON, from the merged tree) are being cross-compiled now, serial make. Waiting for "[Pi ready]" before any push.
