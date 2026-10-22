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
- 21:27 [using] x86 `llama-server` with Cerno 4B Q4_0 (CPU, 12 threads, ~4 GB RAM, port 8090, no
  torch) for the teacher labels, until about 22:15; the 0.8B / 2B evals and the LoRA training run on
  the GPU (0.8B bf16, under 6 GB VRAM, ~3 GB RAM). Jobs are detached with `setsid nohup`; logs in
  `/root/distill/*.log`. Lost once in the 21:20 WSL crash; relaunched 21:27.
- 21:45 [done] the CPU `llama-server` teacher (killed by the lead at ~21:43; it was at 9-13 tok/s under the
  load anyway, 1 request labelled in 3 min). No CPU model jobs from here on.
- 21:45 [using] **slot A claimed from the prompt pilot's [done]** for the student training (0.8B LoRA, bf16:
  about 5 GB VRAM, 3 GB WSL RAM, until ~22:40). The 4B teacher now runs **on the GPU from the Windows side**:
  the official llama.cpp b11818 CUDA 13.4 build (`llama-server.exe -ngl 99`, port 8091, ~3 GB VRAM, 4 threads,
  0 WSL RAM; weights mmap'd from E:), loaded 21:44. Labelling is llama.cpp's readout, the same protocol as
  `ref_llama_pc.jsonl`; a torch bf16 4B is not possible here (the FP8 HF weights were never downloaded,
  only the Q4_0 GGUF; CERNO.md §1). Totals in slot A stay under 10 GB VRAM / 12 GB RAM.
- 21:50 [using] **slot A (GPU): distill**, from 21:50 until ~22:45: Qwen3.5-0.8B LoRA training in bf16
  (about 5 GB VRAM, under 4 GB WSL RAM, 1 process), then the student evals (0.8B / 2B, under 6 GB VRAM).
  The Windows-side teacher `llama-server.exe` (CUDA, 3 GB VRAM, ~1 GB Windows RAM + file cache) stays
  loaded for the set-2 labels; GPU total stays under 9 GB. No CPU model jobs. [done] will follow here.
- 21:47 [measured] GPU teacher check: the Windows CUDA `llama-server` (b11818) on the 18 reference requests
  vs `ref_llama_pc.jsonl` (x86 CPU v0.7.0): max |dp| 0.0147, mean per-row max 0.0036, mean over the 213
  options 0.0019, top answer 51/52 (the 0.497/0.503 tie). Same noise as ARM-vs-x86 (0.0146): the labels
  are usable as the teacher signal. 52 rows in ~10 s (evidence: `/root/distill/labels_ref_gpu.jsonl`).
- 21:36 [measured] zero-shot Qwen3.5-0.8B (base, Cerno prompt + readout, T 1.534) on the 52 reference
  questions: top answer agrees on 15/52 (28.8%), mean |dp| 0.209 over the 213 options, mean per-row
  max |dp| 0.408, max 0.742; choice/score 9/31, yes-no 6/21. HF tokenizer == llama-server ids on 52/52
  rows. 52 rows in 11.8 s on the GPU (evidence: `/root/distill/results/zs-0.8b.json`).
- 21:20 [measured] reference readout checked: p = softmax(logprob / 1.534) over the label
  logprobs of `ref_llama_pc.jsonl` reproduces `probs` to 4 digits (row support_triage/kind).
