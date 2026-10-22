# Findings board (shared by the optimization agents, 2026-10-22)

Each agent keeps ONE file here, `plan/board/<name>.md`, in the MAIN checkout
(`E:\Personal_tries\Raspberry_Pi\plan\board\`, WSL `/mnt/e/Personal_tries/Raspberry_Pi/plan/board/`),
not in its worktree, so the others can read it while it works. Only edit your own file; read all
of them. The lead reads the board, relays findings and commits it.

Names: `speed4b` (4B prefill tuning), `audit` (network audit), `lut` (table-lookup kernels),
`compress` (low-rank / codebooks / mixed bits), `numerics` (Winograd, integer accumulation,
fast transcendental functions, fp16 state), `lead`.

## When to write (append, newest at the bottom, with KST time)
- a measured number others can use (with the result file), e.g. a kernel's GOP/s on a shape,
  a time budget, an acceptance rate, a perplexity;
- a dead end and why (so nobody repeats it);
- an idea that belongs to another agent's area: write it under "For <name>:";
- a change you are about to integrate that touches shared files (`kern_*.c`, `matvec_fast.c`,
  `forward*.c`, `pinfer.h`), so the others can plan merges.

## When to read
- before choosing what to implement, and before each integration;
- if another agent's finding changes your plan, say so in your own file ("after lut 17:50: ...").

Format: `- HH:MM [tag] text (evidence: path)`; tags: `measured`, `dead-end`, `idea`, `for:<name>`,
`integrating`, `question:<name>`.
