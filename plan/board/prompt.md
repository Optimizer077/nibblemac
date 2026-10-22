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
- 21:24 [dead-end] the WSL run was OOM-killed at 14.6 GB RSS (`dmesg`: global OOM; three pilots
  were loading the 4B in the 31 GB VM). Moved off WSL: Windows venv (`.venv`, torch 2.11 cu128,
  transformers 5.20.2), fp32 kept (bf16 would blur the 0.0146 bar), layers 0..23 on the GPU with
  TF32 off, the other 8 layers + embedding + head on the host CPU (host RAM, not the WSL VM).
- 21:31 [using] Windows-side 4B torch job (NOT WSL): GPU ~11 GB + ~19 GB host RAM, 8 threads,
  until ~22:30. Per lead.md 21:28 this leaves the WSL big-model slot free.
- 21:45 [done] Windows-side 4B job ended (process gone, GPU back to 2.2 GB used by others, host
  RAM freed). 7 of 9 variants measured (base, schema, multi, multi_ans, compact_a/b/c); compact_d
  (0% tokens saved) and shared (1.2%) were cut when the lead asked for the slot — neither can
  change a verdict. One open 2-minute run if the slot is ever free: `schema_q`
  (`.venv\Scripts\python.exe tools\pilot\prompt\run.py --variants schema_q --gpu-layers 24
  --base bench\results\pilot\prompt\base_20261022_213424.json`). I will not reload the model
  without claiming the slot first.
- 21:45 [measured] agreement vs the reference (max |dp| / mean |dp|): base 0.0094 / 0.0023
  (anchor); schema 0.593 / 0.094; multi 0.149 / 0.028; multi_ans 0.153 / 0.029; compact_a 0.145 /
  0.016; compact_b 0.173 / 0.034; compact_c 0.262 / 0.061. No layout change stays inside the
  0.0146 bar; even dropping only the `\nOptions:` line moves one question by 0.145.
- 21:47 [done] verdict no-go for every variant; records in `engine/PILOTS.md`, `CHANGELOG.md`,
  `docs/journal/2026-10-22.md`; commit `8033f30` on `worktree-agent-ae99d2e23cc77288a` (not pushed,
  not merged). For lead: the prompt side has no exact lever left; token savings need the
  application's wording or a state cache (audit C11), or a recalibrated checkpoint (class c).
- 21:50 [ack] owner's RAM rule (32 GB total, Windows model jobs one at a time, <= 8 GB host): the 4B
  is not reloaded; `schema_q`, `compact_d`, `shared` stay "not run" in the table. Pilot closed.
