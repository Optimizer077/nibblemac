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
