# prompt (prompt economy: fewer prompt tokens per Cerno decision)

Worktree `E:\Personal_tries\Raspberry_Pi\.claude\worktrees\agent-ae99d2e23cc77288a`
(branch `worktree-agent-ae99d2e23cc77288a`). Scripts `tools/pilot/prompt/`, results
`bench/results/pilot/prompt/`, notebook `engine/PILOTS.md` ("Prompt economy").

Method: PC fp32 torch emulation of the 4B from its `.pq` (compress agent's `pqread.py` /
`cerno.py`, `/root/venv-ref` in WSL, 8 threads), cerno-pi readout (T = 1.534, top-260 rule),
52 questions of `cerno-pi/results/ref_llama_pc.jsonl`; agreement per question vs the reference
(max |dp|, same top) and vs the emulated baseline layout (isolates the prompt change from the
emulation error).

- 21:08 [plan] variants: V1 multi-question rows (a: templates only, b: chain with the 4B's own
  answers), V2 compact template (drop `\nOptions:`, bare letters, inline options), V3 shared
  option sets listed once in the prefix (noul), V4 question-first. Token counts first (exact,
  tokenizer only), model runs for every variant after.
- 21:12 [note] V1 cannot save tokens against the current cerno-pi path: the snapshot already
  prefills the shared prefix once, so tokens are prefix + sum(question_i) today; V1 removes only
  the per-row restore (~1% of a decision, audit §2.2). Measured anyway for the agreement number.
- 21:17 [found] upstream decider (`third_party/cerno-inference/third_party/decider/decider/prompt.py`)
  defines both layouts natively: `build` with N questions = one state-first row, `Question k: …
  Answer k: (`, no answers inserted, N slots read from one pass; `build_schema_first` = question
  blocks first ("computed once per schema and reused for every state"), then `Context:`, then the
  slots. V1 and V4 are therefore in-format for decider-trained models; Cerno's engine only uses
  `plan_rows` (one row per question). The pilot mirrors both builders exactly.
- 21:21 [measured] token counts (`bench/results/pilot/prompt/tokens.json`): base 3703 = audit
  1435 + 2268; compact_a/b/c save 4.2 / 10.0 / 11.4%, shared 1.2%, compact_d 0; multi / multi_ans
  cost 8.3 / 10.2% more; schema costs 8.3% more per isolated decision but 52.3% less (1766) with
  the question block cached across decisions. Model runs started 21:20 (base, schema, multi,
  compact_c, compact_b, compact_a, multi_ans, shared; ~5 min each, fp32 CPU).
