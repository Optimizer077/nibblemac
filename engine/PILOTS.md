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


## Distilled decider (2026-10-23, branch `worktree-agent-acefa744010d32b52`)

**Question.** Can a small Qwen3.5 model reproduce Cerno 4B's typed decisions closely enough to be a
fast first pass on the Pi 4 (0.8B prefill 43 tok/s vs 6.8 for the 4B: 5-6x per decision)?

**Method.** Notebook: `tools/pilot/distill/README.md`.
- Zero-shot: the untrained 0.8B and 2B with Cerno's exact prompt layout and readout (`cerno_prompt.py`,
  a port of `cerno-pi/cerno_pi/decide.py`; T = 1.534) on the 52 reference questions of
  `cerno-pi/results/ref_llama_pc.jsonl`. The HF tokenizer gives llama-server's ids on 52/52 rows.
- Teacher: Cerno 4B Q4_0 through llama.cpp on the GPU (the official b11818 CUDA build on the Windows
  side; the only local weights are the GGUF). Checked against the x86 reference on the 52 rows:
  max |Δp| 0.0147, mean over the 213 options 0.0019, top answer 51/52 (the 0.497/0.503 tie) — the same
  spread as ARM-vs-x86 llama.cpp (0.0146).
- Data: synthetic requests from templates (18 domains: support triage, agent guardrails, reviews,
  intents, routing, places, deliveries, JSON tickets, contracts, NLI, arithmetic, phishing, incident
  reports, medical triage, code review, translations, moderation, meeting notes), 1-5 typed questions
  each; labels = the teacher's full option distributions.
- Student: Qwen3.5-0.8B, LoRA r = 16 on every linear layer of the decoder, bf16, KL(teacher ‖ student)
  over the option labels at the answer slot with the student's logits divided by the same T, one row
  per step, 8 rows per update, 3 epochs, lr 2e-4 cosine. Evaluation on the 52 reference questions
  (held out: the synthetic set shares no state with them).

**Results** (52 held-out reference questions, 213 options; the engine-exactness bar is
max |Δp| ≤ 0.0146 and ≥ 51/52 top answers; the pilot's go bar is ≥ 85% top-answer agreement):

| model | top answer | mean \|Δp\|, options | mean row max \|Δp\| | max \|Δp\| | choice | yes/no | score |
|---|---|---|---|---|---|---|---|
| 0.8B zero-shot | 15/52 (28.8%) | 0.2089 | 0.4078 | 0.7417 | 7/19 | 6/21 | 2/12 |
| 2B zero-shot | 35/52 (67.3%) | 0.1598 | 0.3114 | 0.6506 | 12/19 | 17/21 | 6/12 |
| 0.8B distilled, completed 3 epochs | 39/52 (75.0%) | 0.0888 | 0.1748 | 0.6274 | 16/19 | 17/21 | 6/12 |

Per-type mean option / maximum probability error: choice 0.0542 / 0.5694; yes/no
0.1478 / 0.6274; score 0.1205 / 0.5452. Exact types come from
`cerno-pi/requests/correctness.json`, rather than guessing from the number of options.
Fresh evaluation: 52/52 tokenizations match, 213 options, T = 1.534, 2.7 s scoring on the PC GPU
(8.27 s process wall time), peak host RSS 2,509,608 KiB, CUDA allocator peak 1,595,932,672 bytes
reserved. These are PC evaluation costs, not Pi timings. Evidence and artifact hashes:
`tools/pilot/distill/results/t002-summary.json`; full local rows:
`/root/distill/results/student-0.8b-typed-t002.json`.

**Verdict.** **No-go:** 39/52 is below the required 85% (at least 45/52).
Distillation improves agreement substantially but does not reproduce Cerno closely enough.
The original run completed all 3 epochs (941 train / 49 dev rows, 1,653.1 s); dev agreement
plateaued at 42/49 (85.7%) in epochs 1 and 2, while late training KL fell to about 0.04 and
dev KL remained 0.152. This suggests a generalization / synthetic-data coverage gap rather
than an unfinished or clearly undertrained run, so no extra training or model variant was run.
Score questions remain at 6/12; confident probability differences remain large.

**Next step.** Close this pilot; keep Cerno as the decision model. A future separately authorized
pilot would need broader teacher-labelled states and a fresh evaluation set, rather than more
epochs on these templates. This 52-question correctness set measures agreement with Cerno,
not real-world decision quality. There are no exact prompt overlaps with the 990 synthetic
training/dev rows, but both sets cover similar domains. No Pi speedup is claimed for this student.

**Licences.** Qwen3.5 base models Apache-2.0. Cerno's weights: the model card says Apache-2.0 but the
weight repositories carry no LICENSE file (`cerno-pi/NOTICE`); the teacher is used locally only and
the distilled student stays local (`/root/distill/student-0.8b*`, not committed) until the owner
decides. The synthetic requests are original templates (no copied dataset, no real person).

## Layer skipping (2026-10-22, branch `worktree-agent-aba49cb66390bd91f`)

**Question.** Can Cerno 4B skip layers (residual passthrough), skip only the MLP or only the mixer of
a layer, skip layers only for the question tokens, or exit early, and still give the same decisions?
A decision costs ~150 ms per prompt token on the Pi, 93% of it in the matmuls (`docs/network-audit.md`
§1–2); the token count and the layer count are the only linear levers left. **Go if ≥ 20% of the
matmul MACs can be removed within the bar**: max |Δp| ≤ 0.0146 and the same top answer on ≥ 51 of 52
rows against Cerno's own llama.cpp reference (`cerno-pi/results/ref_llama_pc.jsonl`; pinfer on the Pi
scores 0.0108 / 51).

### Method
- **Harness** `tools/pilot/layerskip/` (`run.sh`, `cerno_gpu.py`, `skip_study.py`, `pqread.py`):
  transformers 5.20.2 `Qwen3_5ForCausalLM` in **exact fp32** on the PC GPU (RTX 5060 Ti, WSL CUDA,
  torch 2.11.0+cu128 in the uv venv `~/.venvs/layerskip`). The weights are the engine's file
  `models/cerno-4b/cerno-4b-q4_0.pq`: the Q4_0 body stays packed on the GPU (nibbles + fp16 block
  scales, 2.0 GB) and every matrix is dequantized per call (`d · (u − 8)` is exact in fp32); the Q8
  head is dequantized in row chunks, the Q8 embedding sits on the CPU. 2.6 GiB of VRAM in all (the GPU
  and the PC were shared with other agents' jobs: one 4B torch job at a time, host threads ≤ 4).
- **Rows and readout** as `tools/compress/cerno.py` (compress agent): the 52 question rows (18 requests,
  sets A + B) with the token ids llama-server received; each request's common prefix is prefilled
  once and its cache deep-copied per row; `p = softmax(label logits / 1.534)`, a label outside the
  top 260 of the vocabulary counts as 0 (`cerno_pi/decide.py`). Positions and the causal mask are
  passed explicitly (HF derives both from the first attention layer's cache, which a skipped layer
  leaves empty).
- **Checks** (`skip_study.py check`, 21:40 KST): explicit positions / mask vs the plain HF path
  max |Δh| 5.7e-6; prefix cache + suffix vs one full forward 6.2e-6; the full model vs the reference
  is in the table below (same numbers as the compress agent's CPU fp32 emulation, 0.0094 / 51).
- **Variants.** The decoder-layer forward is replaced by one with a per-layer mode: `skip` (the
  layer's input is its output), `nomlp` (mixer only), `nomix` (MLP only). Scope `all` = every token
  (prefix and question); scope `q` = question tokens only, the shared prefix state computed in full
  (its cache is reused across variants). **Early exit** after layer k: final norm + head on layer k's
  output, captured by a forward hook (= skipping layers k+1..31 for every token).
  **Accumulation:** units ranked by their single-unit max |Δp|; at each step the best of the
  `--shortlist 6` least harmful remaining units is added (greedy with re-evaluation); stops one step
  after the bar breaks. Units = whole layers, or MLP / mixer parts.
- **MACs saved** = the removed layers' share of the body's 3569 MMAC per token (`docs/network-audit.md`
  §1.1: MLP 70.78, DeltaNet mixer 42.11, full-attention mixer 36.70 MMAC per layer). For scope `q`
  the saving applies to the question rows only, 61% of a decision's time (§2.2), so the decision-level
  saving is 0.61 × the fraction; for scope `all` it is ~0.99 ×.
- Results: `bench/results/layerskip/{base,single,greedy_*}_<stamp>.json` (per-row probabilities
  in `base_*`; per-variant summaries elsewhere). `skip_study.py table FILE...` prints the tables.

### Results
**Full model (the harness itself)**, `base_20261022_214806.json`, 21:48 KST: max |Δp| **0.0094**, mean
0.0023, same top **51/52** (the miss is the reference's 0.006 near-tie sentiment_review / price_complaint;
the compress agent's CPU fp32 emulation gives the same 0.0094 / 0.0023 / 51). The worst row (favours) is
within the reference's own llama.cpp-vs-pinfer noise (0.0108). Every variant below is measured against
the reference (`ref`) and against this full model (`base`).

**Early exit** (final norm + head on layer k's output; the same file). Fails at every k: the top answer
often survives (52/52 at k = 25 and 30) but the probabilities move by 0.17–0.22 on average — the head's
calibration (T = 1.534 fitted to the last layer) does not transfer to earlier layers.

| exit after layer k | layers removed | MACs saved | max \|Δp\| ref | mean \|Δp\| ref | same top ref | max \|Δp\| base | same top base |
|---|---|---|---|---|---|---|---|
| 24 | 25–31 (7) | 21.8% | 0.689 | 0.217 | 48 | 0.689 | 47 |
| 25 | 26–31 (6) | 18.7% | 0.511 | 0.193 | 52 | 0.510 | 51 |
| 26 | 27–31 (5) | 15.5% | 0.511 | 0.195 | 51 | 0.510 | 52 |
| 27 | 28–31 (4) | 12.5% | 0.472 | 0.170 | 47 | 0.472 | 46 |
| 28 | 29–31 (3) | 9.3% | 0.469 | 0.179 | 50 | 0.468 | 49 |
| 29 | 30–31 (2) | 6.2% | 0.477 | 0.180 | 50 | 0.476 | 49 |
| 30 | 31 (1) | 3.0% | 0.509 | 0.209 | 52 | 0.509 | 51 |
| 31 | none (= full) | 0 | 0.0094 | 0.0023 | 51 | 0 | 52 |

**Single-layer skips, every token** (`single_fromlog_*.json`, parsed from the run log of
`skip_study.py all`, 21:48–22:08 KST; 27–72 s per 52-row evaluation on the shared GPU). Sorted by
max |Δp|. One layer = 3.0% (full attention, layers 3, 7, …, 31) or 3.2% (DeltaNet) of the body's MACs.
**No layer passes**: the best, layer 25, is 1.5× over the bar; layers 0–15 are 0.30–0.99.

| skipped layer | kind | MACs saved | max \|Δp\| ref | mean \|Δp\| ref | same top ref | max \|Δp\| base | same top base | pass |
|---|---|---|---|---|---|---|---|---|
| 25 | DeltaNet | 3.2% | 0.0214 | 0.0078 | 52 | 0.0206 | 51 | no |
| 26 | DeltaNet | 3.2% | 0.0324 | 0.0070 | 51 | 0.0338 | 52 | no |
| 29 | DeltaNet | 3.2% | 0.0358 | 0.0124 | 51 | 0.0399 | 50 | no |
| 21 | DeltaNet | 3.2% | 0.0453 | 0.0141 | 51 | 0.0466 | 52 | no |
| 20 | DeltaNet | 3.2% | 0.0595 | 0.0166 | 50 | 0.0689 | 49 | no |
| 28 | DeltaNet | 3.2% | 0.0761 | 0.0161 | 50 | 0.0711 | 51 | no |
| 22 | DeltaNet | 3.2% | 0.1013 | 0.0200 | 51 | 0.0988 | 50 | no |
| 30 | DeltaNet | 3.2% | 0.1084 | 0.0486 | 50 | 0.1060 | 51 | no |
| 17 | DeltaNet | 3.2% | 0.1127 | 0.0367 | 50 | 0.1221 | 51 | no |
| 23 | attention | 3.0% | 0.1178 | 0.0335 | 51 | 0.1228 | 50 | no |
| 18 | DeltaNet | 3.2% | 0.1283 | 0.0334 | 49 | 0.1251 | 50 | no |
| 24 | DeltaNet | 3.2% | 0.1589 | 0.0119 | 51 | 0.1602 | 52 | no |
| 19 | attention | 3.0% | 0.1745 | 0.0469 | 49 | 0.1839 | 50 | no |
| 27 | attention | 3.0% | 0.1858 | 0.0235 | 49 | 0.1808 | 50 | no |
| 16 | DeltaNet | 3.2% | 0.2079 | 0.0582 | 50 | 0.2090 | 51 | no |
| 3 | attention | 3.0% | 0.3044 | 0.0800 | 48 | 0.3014 | 47 | no |
| 5 | DeltaNet | 3.2% | 0.3195 | 0.0631 | 45 | 0.3289 | 46 | no |
| 14 | DeltaNet | 3.2% | 0.3274 | 0.0882 | 49 | 0.3209 | 50 | no |
| 15 | attention | 3.0% | 0.3292 | 0.0978 | 48 | 0.3387 | 49 | no |
| 9 | DeltaNet | 3.2% | 0.3362 | 0.0709 | 48 | 0.3335 | 47 | no |
| 13 | DeltaNet | 3.2% | 0.3368 | 0.0693 | 46 | 0.3338 | 47 | no |
| 10 | DeltaNet | 3.2% | 0.3402 | 0.0868 | 46 | 0.3496 | 45 | no |
| 4 | DeltaNet | 3.2% | 0.3457 | 0.0531 | 47 | 0.3466 | 46 | no |
| 1 | DeltaNet | 3.2% | 0.3609 | 0.0661 | 45 | 0.3579 | 44 | no |
| 8 | DeltaNet | 3.2% | 0.3678 | 0.0613 | 49 | 0.3714 | 48 | no |
| 7 | attention | 3.0% | 0.3708 | 0.0731 | 47 | 0.3720 | 46 | no |
| 6 | DeltaNet | 3.2% | 0.3726 | 0.0986 | 46 | 0.3710 | 45 | no |
| 12 | DeltaNet | 3.2% | 0.4579 | 0.0592 | 47 | 0.4615 | 48 | no |
| 31 | attention | 3.0% | 0.5092 | 0.2088 | 52 | 0.5089 | 51 | no |
| 2 | DeltaNet | 3.2% | 0.5712 | 0.0841 | 48 | 0.5647 | 47 | no |
| 11 | attention | 3.0% | 0.6115 | 0.1027 | 43 | 0.6126 | 44 | no |
| 0 | DeltaNet | 3.2% | 0.9902 | 0.5085 | 14 | 0.9902 | 15 | no |

Reading: the deviations are not noise — they are the same against the reference and against the
harness's own full model (columns `base`), so they are the effect of the skip, not of the emulation.
The late DeltaNet layers (20–29) are the most redundant; the full-attention layers, the first 16
layers and the last layer are not.

### Verdict
(pending)

### If scaled up: how a skip mask would go into pinfer
Class (b), opt-in, never default: the output changes.
1. **`pinfer_params.layer_skip`** (u32 bit mask, bit l = skip layer l for every token) and
   `layer_skip_q` (the same, applied only after `pinfer_eval_select` / a state restore marks the
   prefix as done — i.e. for the tokens that follow the snapshot), zero-initialised = off; plus
   `mlp_skip` / `mix_skip` masks if parts win over whole layers. Validated at `pinfer_load` against
   `n_layers`. Alternative: a header field in the `.pq` (`pq_header` has no reserved space in
   version 1; a new version 2 field would do) — the params route needs no file change and keeps one
   `.pq` per model, so prefer it.
2. **`forward_chunk` (`engine/src/forward.c`)**: inside `for (l …)`, `if (skip & (1u << l)) continue;`
   before `pf_norm_rows`; for part masks, wrap the mixer half (`pf_norm_rows` + `pf_full_attn` /
   `pf_linear_attn` + `pf_add_rows`) and the MLP half separately. A skipped layer leaves its KV
   rows / conv / recurrent state untouched (never read again for those tokens). ~15 lines.
3. **State blobs** (`state.c`): unchanged; a `q`-scope skip applies after `load_state`, so the
   fingerprint and the restore logic stay. Record the masks in the result JSON and in `cerno-pi`'s
   `decide.py` call (`eval_select`), gated by a `decider_config.json` field.
4. **Gates**: the 52-row check on the Pi (`cerno-pi/bench/compare.py` vs `ref_llama_pc.jsonl`)
   with the exact mask found here, bit-for-bit vs this pilot's probabilities; then the decision
   timing (`tmp/csp/decide_ab.sh`).

**Greedy accumulation (channel agent, t004, 2026-10-23 11:42 KST)**, `greedy_ranked_skip_all_20261023_114247.json`:
the first step (skip L25, the least harmful single layer, 3.2% of the body matmuls) already fails the bar
(max |Δp| 0.0214 > 0.0146, 52/52 top answers), so the ranked whole-layer accumulation allows **0 layers**. The
question-tokens-only follow-up was not run (it required ≥ 3 passing layers). Ranking: `ranking_t004.json`;
aggregate and file hashes: `t004-summary.json`.

**Verdict: no-go.** Cerno's probabilities are calibrated through every layer: no whole layer, and no early exit,
stays within the probability bar. Note: skipping L25 keeps the top answer on 52/52; only a looser bar (same top
answer instead of the same probabilities) would admit layer skipping, and that is the owner's decision.

