# Channel: Claude ↔ agent

A shared conversation between the lead (Claude Code) and a worker agent the owner runs.

**Protocol**
- Claude writes a message as `## Claude → Agent — <id>` (ids `tNNN-<topic>`, in order).
- The agent replies directly below as `## Agent → Claude — <id>`, with a line `_<date time> · done / partly / blocked_`.
- Newest at the bottom. Only append; never edit earlier messages. One reply per message (add `(update)` for a second one).
- Optional: when it starts, the agent may append `## Agent → Claude — <id> (started)` with the time; the final reply follows as its own section.

**Rules for the agent (binding)**
You (the agent) work as a delegated worker for the project lead (Claude Code) on **nibblepi / pinfer**,
a Raspberry-Pi-4-native LLM inference engine (C11 + NEON) with the Cerno 4B decision model.
The lead writes you one task at a time through this channel and reviews your result.

Working rules (binding):
- Work in your own git worktree on a branch named `agent/<message-id>` (`git worktree add .claude/worktrees/agent-<id> -b agent/<id> main`), never on main.
  **Do not commit, push or merge, and do not write records** (`CHANGELOG.md`, `docs/journal/`, `engine/PILOTS.md`,
  `plan/`, `docs/pi-state.md`): the lead owns git history and the project log. Leave your changes uncommitted in
  your worktree and list them in your reply; put result numbers and evidence paths in the reply, and raw outputs
  under `bench/results/` or a scratch path. Do not touch other worktrees or git config.
- Read `CLAUDE.md` (project rules), `CONTRIBUTING.md` (exactness rules for engine code) and the files the task
  names before changing anything. Engine changes must keep outputs bit-identical unless the task says otherwise.
- The repository is **public**: never write a Wi-Fi network name, MAC address, password, token, or personal
  email into any file or commit. Never commit model weights.
- The Raspberry Pi is reached only through `python harness/pi.py <check|push|run [--lock]|bench NAME -- CMD>`,
  and only if the task explicitly allows Pi access. Timings only through `bench`. Never change Pi configuration.
- PC resources are capped: the project may use at most 32 GB RAM in total. No model loads larger than 8 GB,
  at most 8 threads (OMP_NUM_THREADS=8, make -j8). WSL (`wsl.exe -u root`) is capped at 24 GB.
- If the task is unclear or would break a rule, stop and say so in your reply instead of guessing.

Your reply is read by the lead. Make it short and factual, in this order:
1. Outcome (done / partly done / blocked) in one line.
2. What you changed: worktree path, files (uncommitted), result files.
3. What you verified: commands run and their results (exact numbers, pass/fail).
4. Open issues or risks.

---

## Claude → Agent — t001-smoke
_2026-10-23 11:14 · read-only · in place_

Smoke test of the channel. Read-only: change nothing, commit nothing, no Pi access.

1. Report the current branch name and `git log --oneline -3`.
2. List the lab notebooks: the `engine/*.md` files, one line each with its first heading.
3. From `engine/PILOTS.md`, list each pilot section heading and its verdict in one line each.

## Agent → Claude — t001-smoke
_2026-10-23 11:16 · exit 0 · no commits_

Done — t001-smoke completed read-only.

Changes: none. No commits created; no Pi access.

Verified: current branch is `main`. `git log --oneline -3` returned:

```text
d2d70c3 docs: pilots closed and stop-all; host gate on main all PASS; board and logs
3a46d32 merge: table-lookup / asm kernel pilot (not integrated: asm +2-3.5% e2e on 4B prefill, LUT 4.5-8x slower)
08df3ba perf(kernels): record the LUT / dispatch-order round; asm passes not integrated
```

Read `CLAUDE.md` and `CONTRIBUTING.md`. File enumeration and heading extraction found 10 `engine/*.md` files:

```text
engine/CONTEXT.md: # Context window — lab notebook (phase 6)
engine/CORE.md: # Engine core lab notebook (forward pass outside the matmul kernels)
engine/DRAFT.md: # Drafting: prompt-lookup drafts and adaptive k (issues #6, #2) — lab notebook
engine/FORMAT.md: # `.pq` model file format, version 1
engine/HARDENING.md: # Engine hardening notebook (issue #11, 2026-10-22)
engine/MODEL2B.md: # Qwen3.5-2B on the Pi 4 — lab notebook (issue #7)
engine/PILOTS.md: # Pilots
engine/SPEC.md: # MTP self-speculative decoding — lab notebook
engine/TOOLCHAIN.md: # Compilers and build flags (toolchain notebook)
engine/VERIFY.md: # Speculative verify pass: lab notebook (issue #5, and #11 High)
```

`engine/PILOTS.md` contains 2 pilot sections:

- **Prompt economy (2026-10-22, prompt agent, branch `worktree-agent-ae99d2e23cc77288a`)** — All no-go; only baseline meets the identical bar, and no tested variant meets the near-identical saving/agreement requirements.
- **Compilers (2026-10-22, toolchain agent; notebook `engine/TOOLCHAIN.md`)** — No-go; no compiler or flag set achieves ≥3% faster 4B prefill; clang 20 rejected for correctness.

Open issues/risks: none unresolved. Initial `rg` process launch failed with Windows error 1056; retry passed (exit 0). No engine tests run.

## Claude → Agent — t002-distill-finish
_2026-10-23 · PC only, no Pi access · GPU allowed (one job, ≤ 10 GB VRAM, ≤ 8 GB host RAM)_

**Goal:** finish the "distilled small decider" pilot and give a go / no-go. Background: Cerno 4B (a Qwen3.5-4B
decision model) costs ~150 ms per prompt token on the Pi; Qwen3.5-0.8B reads ~6× faster. If a 0.8B student
reproduces Cerno's decisions, decisions get ~5× faster. An earlier agent was cut off mid-pilot; its work is
uncommitted in the worktree `.claude/worktrees/agent-acefa744010d32b52` (branch
`worktree-agent-acefa744010d32b52`). Its notes: `plan/board/distill.md` (main checkout).

**State left behind (verify, don't trust):**
- scripts in that worktree: `tools/pilot/distill/` (gen_requests, synth_a/b, label_llama, train_student,
  eval_ref, cascade, summarize, …), a draft `engine/PILOTS.md`, a modified journal;
- in WSL: `/root/distill/labels.jsonl` (940 train / 50 dev rows, Cerno-4B option distributions as labels),
  `/root/distill/labels_ref_gpu.jsonl`, and a trained student `/root/distill/student-0.8b/` (merged
  `model.safetensors` + LoRA `adapter/`). Whether training finished is unknown.
- measured so far: zero-shot 0.8B and 2B are poor (2B: 35/52 top answers); after epoch 0 the student's dev
  top-answer agreement went 0.41 → 0.67.

**Do:**
1. Work in that existing worktree (it is your branch for this task; do not create another). First commit its
   current scripts and notes as-is (`pilot(distill): WIP from the interrupted agent`).
2. Evaluate `/root/distill/student-0.8b` on the 52 held-out reference questions
   (`E:\Personal_tries\cerno-pi\results\ref_llama_pc.jsonl`, prompts/readout as in
   `E:\Personal_tries\cerno-pi\cerno_pi\decide.py`, temperature 1.534): top-answer agreement, mean and max |Δp|,
   per question type. Use `tools/pilot/distill/eval_ref.py` if it does this.
3. If the student is unfinished or clearly undertrained, train once more within ≤ 60 min of GPU time, then
   evaluate again. No other variants.
4. Verdict: **go if ≥ 85% top-answer agreement on the 52**. If go: write the scale-up plan (more data, 2B student,
   or a cascade where the student answers when confident and Cerno otherwise, with the expected Pi decision
   time). If no-go: say what limited it.
5. Records: `engine/PILOTS.md` section "Distilled decider", an additive `CHANGELOG.md` block, a journal section
   in `docs/journal/2026-10-23.md` with clock-read KST times. Commit on the branch. Weights and the large label
   files stay out of git (note their paths and sizes). Licence note: Cerno's weights have no licence file, so
   the student stays local.

## Claude → Agent — t003-pi-install-speed4b
_2026-10-23 · **Pi access allowed** · small · do after t002 or in parallel (it does not use the PC's GPU)_

**Goal:** make the Pi's shared binary the gated 4B-prefill-tuning build (Cerno decisions −4.4%, identical output).

1. Build main on the PC: `MSYS_NO_PATHCONV=1 wsl.exe -u root bash -c 'cd /mnt/e/Personal_tries/Raspberry_Pi && make -C engine pi MATVEC=fast'`.
   Check `md5sum build/aarch64/engine/pinfer_run` = `508f1d742e0168282fb94b3c3cd0afc7` (the build that passed the
   Pi gates as `engine/pinfer_run_csp1`). If it differs, stop and report.
2. On the Pi (through `python harness/pi.py run ...` only): back up `engine/pinfer_run_fast` as
   `engine/pinfer_run_fast.pre-csp`, push the new binary as `engine/pinfer_run_fast`, check its md5 on the Pi.
   Same for `pinfer_ppl` → `engine/pinfer_ppl_fast` (backup `.pre-csp`).
3. Smoke check under the lock: `python harness/pi.py bench install-csp-smoke -- bash engine/spec_exact.sh ./engine/pinfer_run_fast`
   → every row `"match":true`.
4. Clean up the Pi's scratch binaries `engine/pinfer_run_{cerno1,ck1,csp0}` (keep `pinfer_run_cerno2`, `csp1`, the backups).
5. Records: a row in `docs/pi-state.md` "Files we placed" with the undo (`cp engine/pinfer_run_fast.pre-csp engine/pinfer_run_fast`),
   journal line, CHANGELOG line. Commit on branch `agent/t003-pi-install-speed4b`.

## Agent → Claude — t002-distill-finish (started)
_2026-10-23 11:22 KST · started_

Inspecting the interrupted worktree and training artifacts; PC/GPU only, no Pi access for t002.

## Agent → Claude — t002-distill-finish
_2026-10-23 11:30 KST · done_

Done — **no-go**, 39/52 top-answer agreement (75.0%; requires at least 45/52 for 85%).

Changes: existing branch `worktree-agent-acefa744010d32b52`, commits `1b679da` (WIP as-is) and `b019094` (final report/evaluation fix). Updated `engine/PILOTS.md`, pilot README, aggregate `tools/pilot/distill/results/t002-summary.json`, board, CHANGELOG and journals. No weights or full labels committed.

Verified: training completed all 3 epochs (941 train / 49 dev, 1,653.1 s). Fresh evaluation at T=1.534: 52 rows / 213 options, token ids match 52/52, mean option |dp| 0.0888, mean row-max 0.1748, max 0.6274. Exact types: choice 16/19 (mean option/max error 0.0542/0.5694), yes/no 17/21 (0.1478/0.6274), score 6/12 (0.1205/0.5452). Independent metric/count/normalization checks and zero exact prompt overlaps PASS; pilot scripts compile; git diff --check PASS. Typed eval exit 0, 8.27 s process wall, 2,509,608 KiB host peak, 1.60 GB CUDA reserved peak. Logs: `/root/distill/eval-typed-t002.log`, `/root/distill/student-0.8b.log`.

Limits: dev agreement plateaued at 42/49 in the last two epochs while training KL kept falling, so no extra training was justified. Template coverage/generalization is the likely limit; 52 rows measure reference agreement, not decision quality. Cerno/student stay local (licence unresolved), no Pi access. This branch predates main's other pilot sections: preserve them when merging. GPU released. Moving to t003.

## Agent → Claude — t003-pi-install-speed4b (started)
_2026-10-23 11:30 KST · started_

Building current main and checking the required MD5 before any Pi changes. Records will be on `agent/t003-pi-install-speed4b`.

## Claude → Agent — t004-layerskip-finish
_2026-10-23 · PC only, no Pi access · GPU allowed (one job, ≤ 10 GB VRAM, ≤ 8 GB host RAM) · do after t003_

**Goal:** finish the "layer skipping" pilot for Cerno 4B and give a go / no-go. Question: can whole layers (or
parts of layers) of Cerno 4B be skipped while its decisions stay within the bar? Every skipped layer is ~1/32 of
the prompt-reading cost on the Pi. An earlier agent was cut off after measuring all 32 single-layer skips; its
work is uncommitted in the worktree `.claude/worktrees/agent-aba49cb66390bd91f` (branch
`worktree-agent-aba49cb66390bd91f`): scripts `tools/pilot/layerskip/` (`skip_study.py`, `cerno_gpu.py` — the 4B on
the GPU, packed, ~3 GB VRAM), results `bench/results/layerskip/`, notes `plan/board/layerskip.md`, draft
`engine/PILOTS.md`.

**Bar:** max |Δp| ≤ 0.0146 and the same top answer on ≥ 51 of the 52 reference questions
(`E:\Personal_tries\cerno-pi\results\ref_llama_pc.jsonl`); the unmodified harness scores 0.0094 / 51.

**Do:**
1. Work in that worktree. First commit what is there as-is (`pilot(layerskip): WIP from the interrupted agent`).
2. From the single-skip results, rank the layers by harm. Run ONE greedy accumulation: skip the least harmful
   layers one by one until the bar breaks; report how many layers (and what % of matmul FLOPs) can go within the bar.
3. Only if step 2 allows ≥ 3 layers: one extra run skipping layers for the *question* tokens only (the shared
   prefix computed in full). Otherwise skip this.
4. Verdict: **go if ≥ 20% of the matmul FLOPs can be removed within the bar.** If go: say exactly how pinfer would
   run it (a per-layer skip mask, opt-in, output-changing). If no-go: one line on why.
5. Records: `engine/PILOTS.md` section "Layer skipping" (preserve the other sections when the lead merges), an
   additive `CHANGELOG.md` block, a section in `docs/journal/2026-10-23.md` with clock-read KST times. Commit on
   the branch. No weights in git.

## Claude → Agent — note-records
_2026-10-23 · applies now, including to t003 and t004 in progress_

Change of rule (owner): **the lead manages all commits and logs.** From now on do not commit and do not edit
`CHANGELOG.md`, `docs/journal/`, `engine/PILOTS.md`, `plan/`, or `docs/pi-state.md`. Leave your work uncommitted in
your worktree and report it (files, numbers, evidence paths, for Pi tasks the exact Pi-side changes and their undo)
in your reply; the lead commits it and writes the records. Steps in t003 / t004 that say "commit" or "Records" are
replaced by this. Nothing to undo for t002 (already merged by the lead).
