# Distilled decider — pilot notebook (2026-10-22, KST)

Question: can a small Qwen3.5 model reproduce Cerno 4B's typed decisions (choice / yes-no / scored
scale) closely enough to serve as a fast first pass on the Pi 4? A 0.8B reads prompts at 43 tok/s
there, the 4B at 6.8: a decision would be 5-6x faster if the answers matched.

Method: pilot on the PC. (1) Zero-shot baselines: the untrained 0.8B and 2B with Cerno's exact
prompt layout and readout on the 52 reference questions. (2) Self-distillation: synthetic decision
requests written from templates, labelled by the 4B (full option distributions), LoRA on the 0.8B
with a KL loss on the option-label logits at the answer slot, evaluated on the 52 held-out
reference questions.

## Files (this directory, committed)

| File | Purpose |
|---|---|
| `cerno_prompt.py` | Standalone port of the prompt layout and readout of `cerno-pi/cerno_pi/decide.py` for HF tokenizers (no engine dependency) |
| `synth_a.py`, `synth_b.py`, `gen_requests.py` | Synthetic request generator: 18 domains, templates + word banks, deterministic for a seed |
| `sample_requests.json` | The first 10 generated requests (the full set stays outside git) |
| `label_llama.py` | Teacher labels through `llama-server`, the same protocol as the reference (`n_probs=260`, `p^(1/T)` renormalised, T = 1.534) |
| `train_student.py` | LoRA (or full) distillation, KL(teacher ‖ student) over the option labels at the answer slot |
| `eval_ref.py` | Scores an HF model on labelled rows; top-answer agreement, mean / max \|Δp\|, per question type; checks the token ids against `server_ids` |

## Data and checkpoints (outside git, WSL `/root/distill/`)

| Path | What | Size |
|---|---|---|
| `requests.json` | synthetic requests (seed 1, 25 per domain) | filled below |
| `labels.jsonl` | the 4B's label logprobs and distributions per row | filled below |
| `student-0.8b/` | merged distilled 0.8B (bf16 safetensors) + `adapter/` (LoRA) | filled below |
| `results/*.json` | per-row evaluations (contain the reference prompts: Cerno material, not for this public repo) | filled below |
| `models/` | local ext4 copies of the GGUF and the HF checkpoints (loads from `/mnt/e` were slow under load) | ~10 GB |

## Licences

- Qwen3.5-0.8B / 2B: Apache-2.0 (`models/Qwen3.5-*/LICENSE`).
- Cerno 4B weights (`apertixai/cerno-4b-GGUF`): the model card says Apache-2.0 but the weight
  repository has no LICENSE file (see `cerno-pi/NOTICE`). Used locally only, as the teacher. A
  student trained on its outputs stays local (`/root/distill/student-0.8b`) until the owner
  decides; nothing of it is committed.
- The synthetic requests are written here from templates (no copied dataset, no real person).

## Setup (WSL, root)

- `uv` installed to `~/.local/bin`; venv `/root/venv-distill` (Python 3.12): torch 2.11.0+cu128,
  transformers (qwen3_5), peft, accelerate. The RTX 5060 Ti (16 GB) is visible from WSL (driver 591.74).
- Teacher: `/root/src/llama.cpp-v0.7.0/build-host-server/bin/llama-server` (x86 CPU build, v0.7.0,
  the build used for `ref_llama_pc.jsonl`) with `cerno-4b-v1.1-q4_0.gguf`, `-c 4096 -t 12 --parallel 1`.
- Worktree symlink `models -> /mnt/e/Personal_tries/Raspberry_Pi/models` (untracked; `.gitignore`
  has `models/`, which does not match a symlink, so it is never `git add`ed).

## Protocol check

- The reference readout is reproduced from the stored label logprobs: `softmax(logprob / 1.534)`
  gives `probs` to 4 digits (row `support_triage/kind`).
- Reference rows: 52 questions, 213 options, 44-457 tokens per row (mean 141); 2 rows use the wide
  label path (16 and 24 options).

## Results (52 held-out reference questions, 213 options)

Metrics as in `cerno-pi/bench/compare_refs.py`: top-answer agreement; mean |Δp| over the 213
options; mean over rows of the per-row max |Δp|; the overall max |Δp|. Engine-exactness bar:
max |Δp| ≤ 0.0146 and ≥ 51/52. Pilot go bar: ≥ 85% top answers; "useful first pass": ≥ 90% and
mean |Δp| ≤ 0.05.

| model | top answer | mean \|Δp\| (options) | mean row max \|Δp\| | max \|Δp\| | choice/score | yes-no | GPU s |
|---|---|---|---|---|---|---|---|
| teacher check: 4B via llama.cpp CUDA (b11818) vs the x86 CPU reference | 51/52 (98.1%) | 0.0019 | 0.0036 | 0.0147 | — | — | ~10 |
| Qwen3.5-0.8B zero-shot (reference torch kernels) | 15/52 (28.8%) | 0.2089 | 0.4078 | 0.7417 | 9/31 | 6/21 | 11.8 |
| Qwen3.5-0.8B zero-shot (`fla` kernels) | 16/52 (30.8%) | 0.2087 | 0.4075 | 0.7563 | 10/31 | 6/21 | 46.0 (Triton compile) |
| Qwen3.5-2B zero-shot | 35/52 (67.3%) | 0.1598 | 0.3114 | 0.6506 | 18/31 | 17/21 | 8.9 |
STUDENT_ROWS

Notes.
- The teacher row bounds what any student can reach against this reference: the GPU teacher itself
  differs from the x86 CPU reference by up to 0.0147 (one 0.497 / 0.503 tie flips).
- `fla` (flash-linear-attention 0.5.2, Triton) vs the reference torch kernels: differences at the
  1e-3 level, one near-tie row flips. Training used `fla`; evaluation rows say which kernels ran.
- Zero-shot, both base models are badly calibrated (mean row max |Δp| 0.31–0.41) and the 0.8B's top
  answers are near chance on choices; the 2B gets 17/21 yes-no right but is still far from the teacher.

### Training data

| set | generator | requests | questions | bytes |
|---|---|---|---|---|
| 1 | seed 1, 25 per domain, 450 generated | 363 unique | 990 (341 choice, 481 yes-no, 168 score) | 303,508 (`requests.json`) |
| 2 | seed 2, 80 per domain, 1440 generated | 1009 unique | 2765 (940 / 1395 / 430) | 855,791 (`requests2.json`) |

The templates saturate (1440 → 1009 unique): more variety needs more templates, not more draws.
Labelling speed on the GPU teacher: ~2.8 requests/s (set 1 in 2 min; the CPU x86 build did 1
request in 3 min under the shared load, a dead end).

STUDENT_NOTES

## Timeline

- 21:08 start; 21:20 WSL crash (restarted by the lead 21:27), first zero-shot and server runs lost.
