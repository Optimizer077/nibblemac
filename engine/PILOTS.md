# Pilots (PC-side experiments before scaling)

One section per pilot: question, method, numbers, verdict, scale-up plan. Scripts under
`tools/pilot/<name>/`; data and checkpoints outside git.

## Distilled decider (2026-10-22, branch `worktree-agent-acefa744010d32b52`)

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

RESULTS_TABLE

**Verdict.** VERDICT

**Scale-up plan.** SCALEUP

**Licences.** Qwen3.5 base models Apache-2.0. Cerno's weights: the model card says Apache-2.0 but the
weight repositories carry no LICENSE file (`cerno-pi/NOTICE`); the teacher is used locally only and
the distilled student stays local (`/root/distill/student-0.8b*`, not committed) until the owner
decides. The synthetic requests are original templates (no copied dataset, no real person).
