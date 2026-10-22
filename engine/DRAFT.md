# Drafting: prompt-lookup drafts and adaptive k (issues #6, #2) — lab notebook

Owner: drafting work (2026-10-22, branch `worktree-agent-ab157f97c4ad73f36` from main `792676c`).
Files: `engine/src/{spec.c,spec.h,spec_adapt.c,ngram.c,ngram.h}`, the speculative part of
`engine/include/pinfer.h` (additive), `engine/tools/pinfer_run.c` (`--ngram`),
`python/pinfer/{_lib.py,speculative.py}` and `model.py` (`spec_ngram`, a pass-through),
`engine/bench/{chat_spec.sh,spec_exact.sh,draft_trace.c,draft_sim.py,draft_sim_dropped.py}`,
`bench/common/make_copy_prompts.py` and its prompts, `tests/test_spec.py`, one test in
`tests/python/test_api.py`, the API lines of `docs/library.md`, `logs/draft/`.
Pi 4, Q4K-headq4k model, 4 threads; baseline binary `pinfer_run_fast` (md5 `23a76242`, = this
branch's base built in the worktree). All result files are in `bench/results/`, all
`valid=true` unless said otherwise.

## Result

**Issue #2, adaptive k ≤ 3** (same flags as before, `--spec 3 --draft-vocab 32768 --adaptive`),
Pi tok/s, mean of 2 runs (prompt64: 3). "Best fixed" is the best of k = 1..3 measured with the
same binary in the same job (k = 1 from the baseline: never the best).

| prompt | best fixed k | adaptive before | ratio before | **adaptive after** | **ratio after** |
|---|---|---|---|---|---|
| chat27 | k=2 11.01 | 10.49 | 0.97 | 11.02 | **1.00** |
| code, Python | k=2 13.13 | 11.89 | 0.90 | 12.95 | **0.986** |
| code, C | k=2 13.66 | 13.14 | 0.96 | 14.26 | **1.044** |
| list | k=2 12.55 | 11.95 | 0.94 | 12.40 | **0.988** |
| math | k=2 15.04 | 14.54 | 0.96 | 14.80 | **0.984** |
| story | k=2 11.19 | 11.29 | 1.00 | 11.17 | **0.998** |
| prompt64 | k=3 18.47 | 18.69 | 1.01 | 18.20 | **0.985** |
| ctx2048 (64 after 2048) | k=2 10.55 | 10.96 | 1.04 | 11.14 | **1.056** |

Target (≥ 0.98× the best fixed k on every prompt) met on all eight; the worst cases (math,
prompt64) are at 0.984–0.985. The new policy's choices do not depend on timing (only its
plain-mode guard does), so repeated runs make the same rounds.

**Issue #6, prompt-lookup drafts** (`--ngram 7`), tok/s (same jobs):

| prompt | MTP only: k=2 / k=3 / adaptive | k=2 + n-gram | k=3 + n-gram | adaptive + n-gram | n-gram only (k=0) |
|---|---|---|---|---|---|
| copy: edit code | 17.51 / 19.12 / 19.40 | **22.75** (+19% vs k=3) | 22.23 | 22.21 (+14.5%) | 22.05 |
| copy: fix typos | 16.94 / 19.27 / 19.40 | 17.08 | 18.87 | 18.94 (−2.4%) | 14.59 |
| prompt64 | 16.23 / 18.47 / 18.20 | | **20.47** (+11%) | 19.98 (median, +10%) | |
| ctx2048 | 10.55 / 9.95 / 11.14 | | 11.65 | **12.96** (+16%) | |
| six chat prompts | | −0.3…+3.1% vs k=2 | | −1.0…+0.1% vs adaptive | |

Copied code is where prompt lookup pays (+15–19%); fixing typos breaks the copy every few
words and costs ~2%; chat answers are unchanged within run noise. The gain is capped by the
cost of wide verify passes on these kernels (a pass of 8 tokens, cur + 7 drafts, costs 2.3× a
plain step, §1b).

## 1. Method: draft traces and an offline simulator

A round at position P with k MTP drafts accepts min(k, A(P)), where A(P) is how many of the
MTP head's chained drafts d1..d8 at P equal the greedy tokens after P. A(P) does not depend on
how earlier rounds were cut: the MTP KV cache holds main-hidden rows for every committed
position whatever the history (spec.h step 4), and batched passes equal single ones bit for
bit. So one trace per prompt (the drafts and their top-1 probabilities at every position of
plain greedy decoding) lets any draft policy be replayed offline; n-gram drafts depend only on
the tokens and are computed in the replay.

- `engine/bench/draft_trace.c`: plain greedy decoding plus, at every position, the 8-draft
  MTP chain the speculative loop would run, with each draft's top-1 probability over the
  draft vocabulary (32768).
- `engine/bench/draft_sim.py`: replays fixed k, the old policy (a port of `spec_adapt.c` at
  `792676c`), the candidates below, and n-gram rules, with the measured pass costs (§1b) and
  3% noise on the times an adaptive policy sees.
- **Premise check (host, bf16, chat27, 32 tokens):** the replay of fixed k = 1 / 2 / 3 gives
  19 / 15 / 14 rounds and 12 / 16 / 17 accepted drafts, exactly the engine's
  (`--spec k` on the same prompt; `logs/draft/premise_host_chat27_*.txt`).
- **Pi traces** (Q4K): `ng-trace-q4k_20261022_111051.json` (the six chat prompts, prompt64,
  the two copy prompts of §3, ctx2048); held-out prompts `ng-trace-val_20261022_113805.json` (§2e).
- **Accuracy:** with the costs below, the replay of fixed k gives prompt64 13.05 / 16.27 /
  18.49 tok/s for k = 1 / 2 / 3 against 13.09 / 16.29 / 18.53 measured, and chat within ~1%
  (chat27 k = 2: 10.96 vs 10.81 mean of 2 runs; code_py 13.13 vs 13.26). k >= 5 is replayed
  ~5% optimistic.

### 1b. Pass costs (prompt64, `ng-base-p64_20261022_111810.json`, medians of 3 runs)

| verify width n = drafts + 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| verify ms | 136.1 | 142.9 | 152.0 | 199.6 | 258 | 266 | 285 | 360 |
| fixed k = n−1, tok/s | 13.09 | 16.29 | **18.53** | 16.78 | 13.69 | 13.88 | 13.61 | 13.46 |

Plain step 126.1 ms (7.93 tok/s); one MTP pass (chain step or the pass after a round) 8.4 ms;
commit ~0.9 ms + 0.25 ms per verified row's argmax. Width 5 and up are expensive with these
kernels (`kern_quant.c` runs 5 vectors as 4 + 1 passes over each tile, VERIFY.md §7), so a
long draft pays only when nearly all of it is accepted.

## 2. Adaptive k (issue #2)

### 2a. Baseline (`ng-base-chat-r1_20261022_110504.json`, `ng-base-chat-r2_20261022_112540.json`)

| chat, 96 tokens, tok/s (2 runs) | plain | k=1 | k=2 | k=3 | adaptive k≤3 | adaptive / best fixed |
|---|---|---|---|---|---|---|
| chat27 | 7.93 | 10.59 | **10.81** | 10.28 | 10.49 | 0.97 |
| code, Python | 7.87 | 11.38 | **13.26** | 12.59 | 11.89 | **0.90** |
| code, C | 7.93 | 11.72 | **13.70** | 13.50 | 13.14 | 0.96 |
| list | 7.93 | 11.96 | **12.66** | 11.72 | 11.95 | 0.94 |
| math | 7.90 | 12.31 | **15.18** | 14.83 | 14.54 | 0.96 |
| story | 7.92 | 10.95 | 11.24 | 10.37 | **11.29** | 1.00 |

prompt64 (64 tokens, 3 runs): k=3 18.53, adaptive 18.69 (1.01×). chat27 k=2 varied 11.07 / 10.54
between the two runs (plain 7.92 / 7.94), so ±2.5% per run on that prompt is noise.

### 2b. Why the old policy loses

Replay of the old policy (`draft_sim.py`, `old3`): on code, C code and math it runs k = 3 in
89–92% of the rounds, plus 7–8% plain steps (one calibration step at the start and one every
32 rounds, and a fallback). Its cost model makes the verify pass linear in k from one
measured slope and assumes one acceptance rate for every draft position; with p ≈ 0.6–0.7
(its prior is 2/3) that model prefers k = 3, while the measured k = 2 is 2–5% faster on every
chat prompt. On story it settles at k = 2 (87%) and ties.

### 2c. How much is there to gain? (oracle and confidence)

- **Oracle** (dynamic programming over the trace with known A(P), any k ≤ 8 per round):
  chat27 12.82, code_py 15.47, story 13.09, prompt64 20.91 tok/s, i.e. 13–18% above the best
  fixed k. That is the ceiling of any per-round choice of k.
- **Confidence carries information:** the MTP head's top-1 probability is calibrated
  (first draft: below 0.2 accepted 27%, 0.6–0.8 79%, above 0.97 100%; third and later
  drafts: below 0.2 6%, above 0.97 96%; all Pi traces).
- **But little of the headroom is predictable:** the best threshold rule "chain d(s+1) iff the
  top-1 probability of d(s) ≥ τ", fitted on the same ten traces (a grid over two thresholds),
  gains only **+1% on average** over the best fixed k, min 0.987×. Most of the oracle's
  headroom is drafts that look confident and fail, or the reverse.

### 2d. Candidates (replay, tok/s, 20 noise seeds; best fixed k in bold)

| prompt | k=1 | k=2 | k=3 | old | cost model, per-width costs | calibrated confidence, ratio rule | same, average-reward rule | **threshold τ=0.6** |
|---|---|---|---|---|---|---|---|---|
| chat27 | 10.50 | **10.95** | 10.30 | 10.58 | 10.84 | 10.99 | 10.84 | **11.02** |
| code, Python | 11.41 | **13.11** | 12.44 | 12.09 | 12.33 | 12.60 | 12.48 | **13.00** |
| code, C | 11.61 | **13.70** | 13.34 | 13.00 | 13.25 | 13.75 | 13.83 | **14.30** |
| list | 12.03 | **12.54** | 11.58 | 12.04 | 12.03 | 12.65 | 12.37 | **12.46** |
| math | 12.28 | **15.11** | 14.80 | 14.63 | 14.71 | 15.06 | 14.76 | **14.92** |
| story | 10.84 | **11.10** | 10.23 | 11.10 | 10.77 | 10.83 | 10.82 | **11.18** |
| prompt64 | 13.05 | 16.23 | **18.45** | 18.72 | 18.72 | 18.24 | 18.24 | **18.30** |
| copy: fix typos | 13.28 | 17.27 | **19.60** | 19.79 | 19.76 | 19.87 | 19.95 | **19.88** |
| copy: edit code | 13.53 | 17.84 | **19.74** | 19.21 | 19.21 | 19.32 | 19.39 | **20.03** |
| ctx2048 | 9.62 | **10.72** | 10.24 | 11.06 | 9.56 | 11.19 | 11.21 | **11.22** |
| worst ratio to best fixed | | | | 0.92 | 0.89 | 0.96 | 0.95 | **0.987** |

- *cost model, per-width costs* (`new3`): the old structure with a verify cost per width and
  first-draft / later-draft rates; still picks wrong k and drops to plain on ctx2048.
- *calibrated confidence* (`dyn3`, `rate3`): per-draft acceptance from the top-1 probability
  (online bins), chain while the expected rate improves, plain mode from the cost model; the
  ratio rule maximizes each round's tokens / time, the average-reward rule E[tokens] − R·E[time]
  with R the realized rate. Both lose 3–5% on code / story: the per-round estimates are
  noisier than the gain they chase. Allowing k up to 8 made it worse (prompt64 13.9–17.7):
  exploring widths 5–9, whose costs are only known after trying them, does not pay back
  within a 64–96-token answer.
- **threshold τ = 0.6** (`s3`, the final policy, `spec_adapt.c`): d1 always; chain while the
  newest draft's top-1 probability is ≥ 0.6, up to k. No calibration step: the plain-step time
  for the plain-mode guard is extrapolated from the verify passes of widths 2 and 3
  (2·tv(2) − tv(3) = 129.3 ms on the Pi vs 126.1 measured) or measured when plain steps happen.
  Plain mode when the decayed rate of the speculative rounds is ≥ 5% below the plain rate,
  after 8 rounds (2 when a 2-token verify costs ≥ 1.5 plain steps: a compute-bound machine);
  probes of 2 rounds judged on their own after 8, 16, … 64 plain steps.
- Problems found on the way, all in the replay: (1) without the 8-round minimum, one rejected
  first round (1 token in 145 ms) sent chat27 to plain mode (replay 10.38); (2) the plain-step
  time extrapolated from two distant widths (n = 4 and 9 after n-gram rounds) came out at 27 ms
  and sent chat_edit to plain mode (replay 11.3): only widths 2 and 3 are used now; (3) a probe
  judged on the decayed rate of the earlier, bad rounds never left plain mode (Chinese replay
  8.4 tok/s with 60 of 96 tokens plain; k = 1: 10.8): probes now start from empty
  accumulators.

### 2e. Held-out prompts (not used for tuning)

`ng-trace-val_20261022_113805.json`: four new chat prompts (`chat_val_*`: summarize, list →
JSON, explain, SQL) and the seven varied prompts of the MTP reference (`val_varied*`). Replay,
ratio to the best fixed k = 1..3: summarize 1.00, JSON 0.982, explain 0.987, SQL 1.09, the
seven varied prompts 0.979–1.10 except **Chinese (`val_varied6`) 0.905 of k = 1** (= k = 2).
There the draft vocabulary (ids < 32768) leaves out most Chinese tokens: drafts with top-1
probability 0.2–0.6 are right 0–4% of the time, and a chained second draft after an accepted
first one was right 0 of 35 times (34–100% on every other prompt). Pi check (1 run,
`ng1-heldout_20261022_125937.json`): JSON adaptive 19.20 vs k = 3 19.00; SQL 13.70 vs 12.66;
Chinese 9.99 vs k = 1 10.43 (0.96×) and k = 2 9.90; with n-gram drafts 12.19.

Tried for Chinese and dropped: stop chaining when the decayed acceptance of chained drafts
falls below a cutoff. Every cutoff that lifts Chinese (to 0.92–0.95×) costs chat_list as much
(0.93–0.98×), whose chained drafts are right only 34% of the time. A per-script or
frequency-ranked draft vocabulary is the fix for that case, not the policy.

Host (compute-bound, replay with the host build's pass costs): the policy reaches 0.92× of plain
decoding on average over 96 tokens (the old one 0.96×, fixed k = 2 about 0.75×).

## 3. Prompt-lookup drafts (issue #6)

### 3a. Design (`ngram.c`, `spec.c`)

- **History:** the tokens fed so far plus the newest emitted one, kept per context in the spec
  buffers, appended as tokens are committed (`pf_ngram_push`). A continuation
  (`pinfer_generate_greedy_cont`) extends it when it still describes the context (length =
  n_past + 1: the emitted token, which the new prompt leads with, is popped and the prompt
  pushed); anything else (`pinfer_eval`, a loaded state) starts a new history. Drafts only
  ever depend on it, never the output.
- **Index:** for n = 2, 3, 4 a direct-mapped hash table (≥ 2 × max_ctx entries, 112 KB at 4096)
  from the n-gram to the position where it most recently ended; positions are indexed lazily
  by the next lookup; a hit is verified against the history, so a collision only loses a
  match. The lookup takes the longest matching tail n-gram and reports how far back the two
  occurrences agree (up to 32 tokens): the **extended match length**.
- **Round:** an n-gram draft replaces the MTP chain (no MTP passes); verify, accept, commit as
  before. The MTP rows of the committed positions are kept pending, like plain steps, and run
  in one batch before the next MTP round (`defer_rows`), so a run of n-gram rounds costs no MTP
  pass at all; the MTP cache is complete again at the end of every call.
- **k = 0, ngram > 0:** n-gram drafts only, plain steps otherwise; no MTP head needed.

### 3b. When is an n-gram draft right? (all Pi traces, drafts of up to 8)

| extended match | positions | first token right | mean accepted of 8 | if = MTP d1 | if ≠ MTP d1 |
|---|---|---|---|---|---|
| 2–3 | 102 | 0.63 | 2.39 | 0.95 (3.70) | **0.10** (0.28) |
| 4–7 | 96 | 0.69 | 3.38 | 0.92 (4.39) | 0.32 (1.76) |
| 8–15 | 60 | 0.92 | 6.47 | 0.98 (6.93) | 0/4 |
| 16+ | 153 | **0.98** | **7.27** | 1.00 (7.43) | 0.67 (4.78, 9 cases) |

The MTP head's first draft comes free with the previous round's MTP pass, and agreement with
it separates good short matches from bad ones. Together with the verify costs of §1b (a
3-draft pass 152 ms, an 8-draft pass 360 ms) this gives the rule in `spec.c ngram_pick`,
the same with and without adaptive:

- extended match ≥ 16: draft the whole continuation (up to `ngram`);
- 4–15: only if the first n-gram token equals d1 (pending MTP rows are run first to know d1;
  an MTP round would need them anyway), and then at most 3 tokens;
- below 4: ignored, MTP drafts;
- k = 0: no d1 to ask, so matches below 16 draft at most 3 tokens.

### 3c. Rules compared (replay, tok/s; `kNg7` = fixed k plus n-gram drafts of up to 7)

| prompt | k=2 | k=3 | unconditional n-gram (any match ≥ 2), k=3, ≤ 8 | k=2 + rule | k=3 + rule | adaptive + rule |
|---|---|---|---|---|---|---|
| chat27 | 10.95 | 10.30 | 9.54 | 10.96 | 10.32 | 11.03 |
| code, Python | 13.11 | 12.44 | 11.88 | 13.11 | 12.49 | 13.03 |
| code, C | 13.70 | 13.34 | 12.91 | 14.18 | 13.72 | 14.35 |
| list | 12.54 | 11.58 | 11.17 | 12.54 | 11.58 | 12.46 |
| math | 15.11 | 14.80 | 11.18 | 15.11 | 14.80 | 14.89 |
| story | 11.10 | 10.23 | 9.31 | 11.10 | 10.23 | 11.18 |
| prompt64 | 16.23 | 18.45 | 17.15 | 20.46 | 21.11 | 20.55 |
| copy: fix typos | 17.27 | 19.60 | 14.89 | 17.51 | 19.34 | 19.76 |
| copy: edit code | 17.84 | 19.74 | 23.84 | 24.30 | 24.04 | 24.04 |
| ctx2048 | 10.72 | 10.24 | 10.54 | 12.40 | 11.94 | 13.14 |

Unconditional prompt lookup (the textbook version: draft whatever followed the last match)
wins on copied code (+21% over k = 3) but against the best fixed k loses 26% on math, 16% on
stories and 7% even on prompt64: a wrong 8-token draft costs a 360 ms pass. The rule keeps
chat within 1% of the MTP-only policy and keeps most of the gains (edit code +22–36%, prompt64
+11–26%, ctx2048 +16–23%). Variants tried: trusting matches of ≥ 8 instead of 16 (code_py
−2%), full-length drafts for agreeing 8–15 matches (code_py −2%, prompt64 +5%), 1-token short
drafts (fix typos −16%), ignoring matches below 8 (fix typos −5%).

## 4. Measurements on the Pi

Q4K-headq4k, 4 threads, `engine/bench/chat_spec.sh` (draft vocab 32768; `--ngram 7` where
marked), all files `valid=true`, 45–61 °C, not throttled. Before = `pinfer_run_fast` `23a76242`,
after = `pinfer_run_ng1` `7d701ad6`. tok/s, mean of the runs (tokens per pass).

**Chat, 96 tokens, 2 runs.** Before: `ng-base-chat-r1_20261022_110504.json`,
`ng-base-chat-r2_20261022_112540.json`; after: `ng1-chat-r1_20261022_120702.json`,
`ng1-chat-r2_20261022_123441.json`.

| prompt | plain b / a | k=2 b / a | k=3 b / a | adaptive b → a | adaptive + n-gram | k=2 + n-gram |
|---|---|---|---|---|---|---|
| chat27 | 7.93 / 7.87 | 10.81 / 11.01 | 10.28 / 10.24 | 10.49 → **11.02** (1.67) | 11.03 | 11.01 |
| code, Python | 7.87 / 7.87 | 13.26 / 13.13 | 12.59 / 12.52 | 11.89 → **12.95** (2.07) | 12.95 | 13.14 |
| code, C | 7.93 / 7.88 | 13.70 / 13.66 | 13.50 / 13.45 | 13.14 → **14.26** (2.26) | 14.26 | 14.09 |
| list | 7.93 / 7.86 | 12.66 / 12.55 | 11.72 / 11.64 | 11.95 → **12.40** (2.02) | 12.41 | 12.53 |
| math | 7.90 / 7.86 | 15.18 / 15.04 | 14.83 / 14.84 | 14.54 → **14.80** (2.44) | 14.77 | 15.08 |
| story | 7.92 / 7.88 | 11.24 / 11.19 | 10.37 / 10.30 | 11.29 → **11.17** (1.70) | 11.06 | 11.16 |

Fixed k is the same code in both binaries; the before/after columns agree within run noise.
k = 1 (before only): 10.59 / 11.38 / 11.72 / 11.96 / 12.31 / 10.95.

**prompt64, 64 tokens, 3 runs.** Before: `ng-base-p64_20261022_111810.json`; after:
`ng1-p64_20261022_120929.json`.

| | plain | k=2 | k=3 | adaptive | adaptive + n-gram | k=3 + n-gram |
|---|---|---|---|---|---|---|
| before | 7.93 | 16.29 | 18.53 | 18.69 (3.32) | | |
| after | 7.87 | 16.23 | 18.47 | **18.20** (3.15) | 17.80 / 19.96 / 20.01 (3.94) | **20.47** (4.20) |

The first adaptive + n-gram run made the same 16 rounds as the other two but its verify
passes took 203 instead of 180 ms (a disturbed run); median 19.98.

**Copy prompts, 96 tokens, 2 runs.** Before: `ng-base-copy_20261022_121404.json`; after:
`ng1-copy_20261022_124147.json`.

| prompt | plain | k=2 b / a | k=3 b / a | adaptive b → a | k=2 + n-gram | k=3 + n-gram | adaptive + n-gram | n-gram only |
|---|---|---|---|---|---|---|---|---|
| edit code (`chat_edit`) | 7.80 | 17.52 / 17.51 | 19.11 / 19.12 | 18.90 → 19.40 | **22.75** (5.28) | 22.23 | 22.21 | 22.05 (4.75) |
| fix typos (`chat_fix`) | 7.76 | 16.87 / 16.94 | 19.40 / 19.27 | 19.43 → 19.40 | 17.08 | 18.87 | 18.94 | 14.59 |

On the code edit, 14–15 of 18 rounds draft n-grams and 70–72 of 74–75 drafted tokens are
accepted; with n-gram drafts only (no MTP head) the answer still runs at 2.8× plain.

**ctx2048** (`data/ctx2048.ids`, `-c 2176`, `--no-warmup`, 64 tokens, 2 runs). Before:
`ng-base-ctx2048-r1_20261022_124634.json`, `ng-base-ctx2048-r2_20261022_130421.json`; after:
`ng1-ctx2048-r1_20261022_125234.json`, `ng1-ctx2048-r2_20261022_131011.json`.

| | plain | k=2 | k=3 | adaptive | adaptive + n-gram | k=3 + n-gram |
|---|---|---|---|---|---|---|
| before | 7.05 | 10.56 | 10.13 | 10.96 (2.17) | | |
| after | | 10.55 | 9.95 | **11.14** (2.03) | **12.96** (2.52) | 11.65 (2.62) |

**Held-out** (1 run, after only, `ng1-heldout_20261022_125937.json`): see §2e.

## 5. Exactness gates (final code: `spec.c` md5 `d20902a3`, `spec_adapt.c` `2496dd67`, `ngram.c` `990d561e`)

| gate | result | evidence |
|---|---|---|
| host `tests/test_golden.py` | PASS | `logs/draft/host_gates_v2.txt` |
| host `tests/test_spec.py`: MTP head vs reference; 10 prompts (new: the 203-token code-edit prompt) × 13 speculative configs (new: `k3g8`, `g8`, `a3g8`, `k8g8`, `k2g5b2`) | PASS, 130/130 runs = plain greedy | same |
| host `tests/test_state.py` | PASS | same |
| host pytest, source tree, `PINFER_TEST_REQUIRE=1` (+ `test_speculative_ngram_matches_greedy`) | 127 passed | same |
| host state identity after speculation (`logs/draft/state_cmp.c`, VERIFY.md §4 extended): 12 configs (k = 1/2/3/4/6, full vocab, adaptive, k3 + n-gram 8, n-gram only, adaptive + n-gram, k = 8 + n-gram 8, k = 2 + n-gram 5) × 4 prompts incl. a continuation | 48/48 identical | `logs/draft/state_cmp_host_v2.txt` |
| Pi `spec_exact.sh ./engine/ng/pinfer_run_ng1` (md5 `7d701ad6`, checked on the Pi): the original 20 rows | 20/20, md5s = baseline (`703b5ff0`, `3f9a4f66`, `08c27da6`, `3f9a4f66`) | `spec-exact-ng1_20261022_115556.json` |
| same, 38 new rows (n-gram, adaptive, k = 8, `-b 2`; chat27, prompt64, chat_edit; both models) | 38/38 | same |
| Pi state identity (`state_cmp_ng1`, NEON, Q4K), same 12 configs × 4 prompts | 48/48 identical | `state-cmp-ng1_20261022_120036.json` |
| `-Wall -Wextra`, host (ref) and aarch64 (fast), engine, lib, tools | no warnings | build logs |

The v1 run of the same host gates (`logs/draft/host_gates_v1.txt`, `state_cmp_host.txt`) used
an earlier policy file with the same `spec.c`; also all PASS / 48 of 48.

## 6. Tried and dropped

| idea | evidence | decision |
|---|---|---|
| Cost-model k choice with per-width verify costs (`new3`) | §2d: 0.89× on ctx2048, 0.94× code | dropped |
| Calibrated-confidence chain with expected-rate stopping, ratio or average-reward rule, k ≤ 3 / ≤ 8 (`dyn*`, `rate*`) | §2d: 0.95–0.96× worst case; k ≤ 8 explores expensive widths | dropped for the threshold rule |
| Unconditional n-gram drafts of up to 8 | §3c: math −26%, story −16% vs the best fixed k | gated (§3b) |
| Chain stop on low acceptance of chained drafts | §2e: Chinese 0.91 → 0.92–0.95×, chat_list 0.99 → 0.93–0.98× | dropped |
| Grid-fitted two-level confidence thresholds | §2c: +1% mean in-sample, the one threshold kept instead | dropped |
| Mixed drafts (n-gram tokens, then MTP chained on them) | the MTP chain over n-gram tokens costs one MTP pass per token, the cost n-gram drafts avoid; per-round choice keeps both cheap | not built |

## 7. Recommendation on defaults (for the lead)

`generate(speculative=True)` and `pinfer serve` use `pinfer_default_spec_params()` = k = 2,
draft vocab 32768, fixed. Measured alternatives on this branch:

| default candidate | chat (6) vs k=2 | prompt64 | copy code | fix typos | ctx2048 | host (replay) |
|---|---|---|---|---|---|---|
| k = 2 (today) | 1.00 | 16.23 | 17.51 | 16.94 | 10.55 | ~0.75× plain |
| adaptive k ≤ 3 | 0.984–1.044 | 18.20 | 19.40 | 19.40 | 11.14 | 0.92× plain |
| **adaptive k ≤ 3 + ngram 7** | 0.982–1.044 | 19.98 | 22.21 | 18.94 | 12.96 | 0.92× plain |
| k = 2 + ngram 7 | 0.997–1.031 | | 22.75 | 17.08 | | |

Recommendation: **make adaptive k ≤ 3 with ngram 7 the default** for both, i.e.
`pinfer_default_spec_params()` = `{3, 32768, 1, 7}`. On the eight benchmark prompts it is
never more than 1.8% below the best fixed k (math 0.982); it beats today's k = 2 by 12–27% on
prose continuation, long repetitive context and copied code; on Chinese (held out) it reaches
12.2 tok/s against 9.9 for k = 2; and its guard keeps compute-bound machines (the x86 wheel)
near plain speed, where fixed k loses ~25%. The cost: ~2% against adaptive without n-gram on
answers that rewrite the prompt with small changes (typo fixing). In `pinfer serve` the n-gram
history follows the conversation through `pinfer_generate_greedy_cont`, so later turns can copy
from earlier ones. Not changed here: the default itself and the `server/engine.py` / `cli.py`
options are outside this work's files.

## 8. Open items

- Wide verify passes (5–9 tokens: 200–360 ms vs 152 ms at 4) cap both long MTP chains and long
  n-gram drafts; cheaper n = 5..8 passes in `kern_quant.c` (VERIFY.md §7) would raise the
  copy-code gain (all-accepted 8-token drafts: 285 ms / 8 tokens today).
- Chinese and other scripts outside ids < 32768: the confidence of drafts over the restricted
  vocabulary is overconfident; a per-script or frequency-ranked draft vocabulary would help
  (FR-Spec was dropped for English chat, notes.md).
- The adaptive state and the measured pass costs are reset per call; a server could keep the
  costs across requests (they are properties of the machine).
- `pinfer chat` / `pinfer serve` / `pinfer bench` have no `--ngram` / adaptive options for the
  server yet (Python API only: `generate(..., spec_ngram=)`).
- n-gram history after `pinfer_state_load` / `pinfer_eval`: starts again from the next prompt
  (only drafts are affected); a restored state does not carry its tokens.
