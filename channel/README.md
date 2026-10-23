# Channel (tasks for Codex)

The lead (Claude Code) delegates tasks to the Codex CLI through this folder and reviews the results.

| Path | What |
|---|---|
| `inbox/<id>.md` | the task, written by the lead |
| `outbox/<id>.md` | Codex's final message: outcome, changes, verification, open issues |
| `logs/<id>.jsonl`, `.stderr`, `.meta` | full event stream, errors, start / end / exit / branch / commits (gitignored: the repo is public) |
| `preamble.md` | the rules prepended to every task (project rules, public repo, Pi via harness only, 32 GB RAM cap) |

## Run a task
```bash
bash channel/run.sh <id>                                   # isolated worktree .claude/worktrees/codex-<id>, branch codex/<id>
bash channel/run.sh <id> --sandbox read-only --in-place    # questions / reviews on the main checkout, no writes
bash channel/run.sh <id> --sandbox danger-full-access      # only for tasks that need the network (Pi through harness/pi.py)
bash channel/status.sh                                     # all tasks: state, exit code, commits, branch
```
Default sandbox `workspace-write` (writes only inside the task's worktree, no network). The model is always the
one the owner sets in the Codex config (`~/.codex/config.toml`); the channel never overrides it.

## Review and merge (lead)
1. Read `outbox/<id>.md`; check `git log main..codex/<id>` and the diff.
2. Rerun the verification that matters (gates for engine changes, `CONTRIBUTING.md` section 5).
3. Merge with `--no-ff`, scan for secrets, push; remove the worktree and branch.

Task ids: `tNNN-<topic>`, in order. A Pi task must say "Pi access allowed" and follow the lock rules.
