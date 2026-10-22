# distill — pilot: a 0.8B student that reproduces Cerno 4B's decisions (2026-10-22, KST)

Branch `worktree-agent-acefa744010d32b52`. Scripts `tools/pilot/distill/`. Data and checkpoints
outside git (`/root/distill/` in WSL, listed in `tools/pilot/distill/README.md`).

## Plan
1. Zero-shot: Qwen3.5-0.8B and 2B (base) with the exact Cerno prompt + readout on the 52 reference
   questions (`cerno-pi/results/ref_llama_pc.jsonl`). Report top agreement, mean / max |dp|.
2. Synthetic requests (templates + variation, ~400 requests, 2-4 typed questions each) ->
   labelled by Cerno 4B through the x86 `llama-server` (same protocol as the reference) ->
   LoRA on the 0.8B, KL on the label logits at the answer slot, T = 1.534 -> eval on the 52.
3. Report in `engine/PILOTS.md` ("Distilled decider"), CHANGELOG, journal.

## Findings (newest at the bottom)
- 21:08 [measured] env: WSL Ubuntu 26.04 (root), RTX 5060 Ti 16 GB visible to `nvidia-smi` in WSL
  (driver 591.74), 20 cores / 31 GB, no uv, no nvcc; x86 `llama-server` (llama.cpp v0.7.0, CPU)
  at `/root/src/llama.cpp-v0.7.0/build-host-server/bin/llama-server`; 4B GGUF at
  `models/cerno-4b/gguf/cerno-4b-v1.1-q4_0.gguf`; HF 0.8B / 2B bf16 under `models/`.
- 21:20 [measured] reference readout checked: p = softmax(logprob / 1.534) over the label
  logprobs of `ref_llama_pc.jsonl` reproduces `probs` to 4 digits (row support_triage/kind).
