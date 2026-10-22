# Pilots

Short PC-side studies ("pilot quickly on the PC, then scale what works"). Each section: the
question, the method, the numbers with their result files, the verdict, and what a scale-up
would need. Pi numbers only come from `harness/pi.py bench`; a pilot's numbers are PC emulation.

## Prompt economy (2026-10-22, prompt agent, branch `worktree-agent-ae99d2e23cc77288a`)

**Question.** A Cerno decision costs ~150 ms per prompt token on the Pi (`docs/network-audit.md`:
38% of a decision is the shared state prefix, 61% the question rows; 44% of a question's tokens
are template). Can a prompt layout use fewer tokens per decision while the probabilities stay
identical (max |dp| <= 0.0146 and the same top answer on >= 51 of 52 questions vs the llama.cpp
reference `cerno-pi/results/ref_llama_pc.jsonl`) or near-identical (owner's call)?

**Method** (`tools/pilot/prompt/`, results `bench/results/pilot/prompt/`). The 4B in fp32 torch
on the PC from its `.pq` (the compress agent's `pqread.py` and emulation: on the one-row layout
it reproduces the reference to max |dp| 0.0094, same top 51/52), cerno-pi's readout (T = 1.534,
labels outside the top 260 count as 0), the 18 requests / 52 questions of
`cerno-pi/requests/correctness.json`. The run is on the Windows side (uv venv, torch 2.11 cu128,
transformers 5.20.2): layers 0..23 on the RTX 5060 Ti in fp32 with TF32 off (10 GB), the other 8
layers, the embedding and the head on the host CPU (the fp32 body, 14.3 GB, does not fit the
card; a first WSL CPU run was OOM-killed while three pilots loaded the 4B in the 31 GB VM). The
split reproduces the compress agent's pure-CPU probabilities to |dp| < 1e-6 (`smoke.log`,
`base_smoke_*.json`); model load 40 s, ~3 s per request. `variants.py` builds every layout and counts its tokens
(the `base` rows are checked equal to the reference's `server_ids`, 52/52); `run.py` scores them
(a request's prefix once, its cache copied per row; chains continue); `report.py` prints the table.
Tokens per decision = what the engine evaluates with cerno-pi's prefix snapshot: shared prefix
once + every row's suffix (base: 1435 + 2268 = 3703 on the set, as in the audit).

Layouts (decider's own builders where they exist, `third_party/decider/decider/prompt.py`):

| variant | layout | origin |
|---|---|---|
| `base` | `Context:\n<state>\n\nQuestion: <q>\nOptions:\n(A) <o>…\nAnswer: (`, one row per question, common prefix snapshotted | cerno-pi / `engine_llama.py` |
| `multi` | one state-first row with all questions: `\n\nQuestion k: <q>\nOptions:…\nAnswer k: (` per question, no answer inserted, one slot per question read from one pass | decider `build` (N questions) |
| `multi_ans` | `multi` as a chain: the 4B's own top letter + `)` inserted after each slot | pilot |
| `schema` | question blocks first (`Question k: …\nOptions:…`), then `\n\nContext:\n<state>`, then the slots `\n\nAnswer 1: (`, `\nAnswer 2: (` …; the question block depends only on the schema | decider `build_schema_first` |
| `compact_a` | `base` without the `\nOptions:` line | pilot |
| `compact_b` | `compact_a` with bare letters `\nA <o>`; slot `\nAnswer: (` | pilot |
| `compact_c` | `compact_b` with the slot `\nAnswer:` and the labels read as ` A`, ` B` … tokens | pilot |
| `compact_d` | `base` with the options on one line `\nOptions: (A) <o> (B) <o>` | pilot |
| `shared` | `base`; a request with >= 2 yes/no questions lists `(A) no (B) yes` once after the state and its yes/no rows drop the options block | pilot |

**Token counts** (`variants.py`, `tokens.json`; 18 requests, 52 questions):

| variant | tokens per set | saved | with the question block cached across decisions | saved |
|---|---|---|---|---|
| base | 3703 | 0 | 3703 | 0 |
| multi | 4011 | −8.3% | 4011 | −8.3% |
| multi_ans | 4079 | −10.2% | 4079 | −10.2% |
| schema | 4011 | −8.3% | **1766** | **52.3%** |
| compact_a | 3547 | 4.2% | 3547 | 4.2% |
| compact_b | 3334 | 10.0% | 3334 | 10.0% |
| compact_c | 3282 | 11.4% | 3282 | 11.4% |
| compact_d | 3703 | 0.0% | 3703 | 0.0% |
| shared | 3657 | 1.2% | 3657 | 1.2% |

Why the multi-question rows save nothing: cerno-pi already prefills the rows' common prefix
(`Context:\n<state>\n\nQuestion:`) once per decision and restores it per row, so a decision costs
prefix + sum(question suffixes) today; one row with all questions evaluates the same tokens plus
the per-question `Question k:` heads and numbers, and only removes the per-row restore (~1% of a
decision). `schema` saves tokens only when the same questions are asked about many states: the
question block is then prefilled once and every decision costs `\n\nContext:\n` + state + 4 tokens
per answer slot (1766 on this set, −52%; the time saved on the Pi would be ~150 ms x 1937 tokens
≈ 290 s over the 18 requests). `compact_d` saves nothing because ` (A` costs the same tokens as
`\n(A`. `shared` saves little: the legend costs ~12 prefix tokens per request against ~11 per yes/no
row, and only 6 requests have two or more yes/no questions; the score questions of the set all
have different legends.

**Agreement** (PC emulation vs the llama.cpp reference; "vs base" isolates the layout change from
the emulation's own error): see the table below (filled from `report.py`).

| variant | tokens (18 req.) | saved | with schema cache | max \|dp\| ref | same top ref | mean \|dp\| ref | max \|dp\| base | same top base | identical? |
|---|---|---|---|---|---|---|---|---|---|
| base | 3703 | 0.0% | 0.0% | 0.0094 | 51/52 | 0.0023 | 0.0000 | 52/52 | yes |
| multi | 4011 | −8.3% | −8.3% | 0.1494 | 51/52 | 0.0280 | 0.1399 | 50/52 | no |
| multi_ans | 4079 | −10.2% | −10.2% | 0.1534 | 50/52 | 0.0294 | 0.1439 | 49/52 | no |
| schema | 4011 | −8.3% | 52.3% | 0.5934 | 47/52 | 0.0944 | 0.5937 | 46/52 | no |
| schema_q | 4761 | −28.6% | 32.1% | – | – | – | – | – | not run |
| compact_a | 3547 | 4.2% | 4.2% | 0.1453 | 51/52 | 0.0159 | 0.1498 | 52/52 | no |
| compact_b | 3334 | 10.0% | 10.0% | 0.1735 | 50/52 | 0.0338 | 0.1762 | 51/52 | no |
| compact_c | 3282 | 11.4% | 11.4% | 0.2617 | 48/52 | 0.0606 | 0.2629 | 49/52 | no |
| compact_d | 3703 | 0.0% | 0.0% | – | – | – | – | – | not run |
| shared | 3657 | 1.2% | 1.2% | – | – | – | – | – | not run |

Result files: `base_20261022_213424.json`, `schema_20261022_213503.json`, `multi_20261022_213534.json`,
`multi_ans_20261022_213615.json`, `compact_a_20261022_213701.json`, `compact_b_20261022_213855.json`,
`compact_c_20261022_214452.json`, `tokens.json`, `report.md`, `run_win.log`. Largest |dp| vs the
reference: base contract_clause/favours 0.0094; multi contract_clause/favours 0.149, agent_guardrail/risk
0.135; schema long_incident/detection 0.593, country_wide/country 0.386, code_review/bug 0.263;
compact_a router_wide/category 0.145 (the 16-option question), then 0.066; compact_b router_wide/category
0.174, math_word/needs_arithmetic 0.145; compact_c code_review/bug 0.262, long_incident/process_gap 0.215.
`compact_d` (0% saved), `shared` (1.2%) and `schema_q` (a schema-first row whose slots repeat the
question text: 32% saved with the cache) were not scored: the run was stopped at 21:45 when the lead
asked for the PC's model slot, and the first two cannot change a verdict; `schema_q` is the one open
2-minute run (`run.py --variants schema_q --gpu-layers 24 --base <base json>`).

**Verdicts (all no-go).**
- *Identical bar* (max |dp| <= 0.0146, same top >= 51/52): only `base` passes. The smallest change,
  `compact_a` (the `\nOptions:` line dropped, 4.2% of the tokens), moves one question by 0.145 and
  the mean by 0.016, 7x the emulation's own 0.0023; the compact templates drift more as they save more
  (10.0%: 0.174; 11.4%: 0.262 with 48/52).
- *Near-identical bar* (>= 30% saved): nothing reaches 30% on the set itself. `schema` reaches 52%
  only with the question block cached across decisions and then drifts by 0.59 with 47/52 (46/52 vs
  the emulated base), i.e. different answers, not noise. The multi-question rows (decider's own
  layout) save no tokens (the prefix snapshot already gives prefix + sum of the questions) and drift
  by 0.15. Nothing here is worth an owner decision.
- Reading: the 4B's probabilities and its temperature (1.534, fitted on one-question state-first rows)
  are specific to that exact layout; every template token carries calibration. The prompt-side
  levers that stay exact are the application's own wording (question text 24% and option texts 32%
  of a question's tokens) and reusing the state across decisions (audit C11); the template is not one.

**Scale-up.** None (no go variant). If the owner wants the schema-first saving anyway (class c), the
precondition is a Cerno checkpoint trained or at least temperature-calibrated on decider's schema-first
layout (upstream decider trains with `layout="schema_first"`; the current weights were not). With such
a checkpoint the engine side is small: cerno-pi `Decider(layout="schema_first")` builds the question
block with decider's `schema_prefix_ids` (pieces encoded separately, so `tokenization="ids"`), prefills
it once with `eval_select(prefix, [])`, keeps its snapshot across decisions keyed by the block's token
ids (`save_state` blob in Python today; the in-engine checkpoint when merged), and per decision restores
it, calls `eval_select("\n\nContext:\n" + state + "\n\nAnswer 1: (", labels_1)` and then
`eval_select("\nAnswer k: (", labels_k)` for each later slot (3–4 tokens each, no restore between slots);
`Decision.prefix_tokens` reports 0 on a cache hit. pinfer needs no change: several slots are successive
`eval_select` calls at the same token cost; an optional `pinfer_eval_select_at(tokens, n, positions,
n_pos, ids, n_ids, out)` (selected logits at several positions of one chunked prefill) would only remove
the k − 1 small-call overheads, at most ~2% of a decision.

## Compilers (2026-10-22, toolchain agent; notebook `engine/TOOLCHAIN.md`)

**Question.** Does another compiler make the bit-identical engine faster on the Pi 4, first of all
the Q4_0 batched kernel of Cerno 4B prefill (n = 7 / 5 groups)? **Go** if a build is >= 3 % faster
on 4B prefill p256 and bit-identical to `engine/pinfer_run_fast`.

**Budget.** PC-side builds and qemu checks; ~1 h of Pi time under the lock (bench names `toolchain-*`).

**Builds** (same sources `69b31a3`, project flags `-O3 -mcpu=cortex-a72 -std=c11 -ffp-contract=off`,
static, cross-compiled in WSL): GCC 15.2 (shipping), GCC 14.3, GCC 16.0.1 (snapshot), clang 20.1.8,
clang 22.1.2, clang 22 + `-mllvm -enable-misched=false` on `kern_quant.c`.

**Exactness (PC).** kbench `--check`, 880 configurations per build under qemu: GCC 13 / 14 / 15 / 16
and clang 22 produce the same output bytes (yhash) as the shipping build on every row.
**clang 20 is rejected**: at `-O3` it miscompiles the max-by-magnitude loop of kbench's Q4 fill
(a different maximum in 1.7 % of the 32-element blocks; `-O2` or `-fno-unroll-loops` restore it),
i.e. it changes plain C floating-point results (TOOLCHAIN.md §3.1).

**A flag that looked good on paper and lost on the Pi.** `-fno-schedule-insns` on `kern_quant.c`
removes every stack spill of the Q4 loops (232 -> 217 instructions per block) and is 6-15 % *slower*
on every Q4 shape (speed4b's `cerno-speed-kb8-nosched_20261022_175206.json`): on the in-order A72
the interleaving of the multiplies with the reductions is worth more than the instruction count.

**Speed (Pi, 1.5 GHz, `bench/results/toolchain-kb-r1_20261022_220248.json`,
`toolchain-e2e4b-r1_20261022_220708.json`).** Q4_0 kernel, cold 4 threads, 4B shapes, medians of
3 rounds vs GCC 15: GCC 16 -2..+3 % and GCC 14 -3..+6 % (noise band), clang 22 +5..+15 % at n = 7,
clang 22 with the machine scheduler off +12..+22 %; every build's output hash identical on every row.
4B prefill p256 (one run each, owner's scope cut): GCC 15 6.915 tok/s, GCC 16 6.921 (+0.1 %),
GCC 14 6.789 (-1.8 %).

**Verdict: no-go.** No compiler or flag set is >= 3 % faster; the shortest code (no spills) loses
twice; clang 20 is excluded for correctness. `engine/Makefile` unchanged. The remaining lever for
this loop is instruction order inside the kernel (lut's asm passes), not the compiler.
