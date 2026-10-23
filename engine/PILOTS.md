# Pilots (PC-side experiments before scaling)

One section per pilot: question, method, numbers, verdict, scale-up plan. Scripts under
`tools/pilot/<name>/`; data and checkpoints outside git.

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
