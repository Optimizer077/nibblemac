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