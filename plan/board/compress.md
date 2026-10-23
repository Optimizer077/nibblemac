# compress (low-rank / codebooks / mixed bits / head / drafters)

Note: this agent's harness refuses writes to the main checkout, so this file lives in its
worktree: `E:\Personal_tries\Raspberry_Pi\.claude\worktrees\agent-a0bab39a2c3dd469b\plan\board\compress.md`
(branch `worktree-agent-a0bab39a2c3dd469b`). Notebook `engine/COMPRESS.md`, offline results
`bench/results/compress/*.json`, scripts `tools/compress/`.

Method: PC fp32 transformers model with the matrices replaced by dequantized candidates (Q4K
emulation = the exact model), perplexity with pinfer_ppl's protocol, greedy agreement on the chat
prompts. Pi only for configurations that pass offline.

- 17:35 [plan] Q1 low rank (plain / whitened SVD, + quantized residual), Q2 head (projection
  bound only — audit's 17:27 dead end covers row-norm and bit-plane bounds; vocab subset;
  compressed MTP draft head), Q3 mixed bits (k-quant 3.5 / 2.6 bit, GPTQ rounding), Q4 compressed
  drafter (offline acceptance), Q5 Cerno 4B MLP low rank (MACs), Q6 codebooks (PQ / E8 at ~2 bpw;
  16-entry TBL codebooks).
- 17:39 [measured] PC emulation anchored to the engine (0.8B): bf16 8x512 PPL 17.929 (llama.cpp
  17.93); Q4K weights + int8 activation rounding 18.494 (engine 18.50), 8x2048 13.984 (engine
  13.985) (evidence: bench/results/compress/baseline_20261022_173846.json in this worktree).
- 17:50 [plan] after the owner's reorder: Cerno 4B first (low rank of gate / up / down / qkv,
  MLP channel pruning with a least-squares refit of down; max |dp| and same top answer vs
  ref_llama_pc.jsonl), 0.8B items only after that.
- 17:59 [measured] Cerno 4B PC emulation (torch fp32, the .pq's Q4_0 / Q8 weights dequantized,
  cerno-pi readout with T = 1.534 and the top-260 rule) vs ref_llama_pc.jsonl: max |dp| 0.0094,
  mean 0.0023, same top 51/52 (pinfer on the Pi: 0.0108, 51/52); the miss is sentiment_review /
  price_complaint, a 0.0056 near-tie in the reference. Rows with small reference top-2 gaps:
  price_complaint 0.006, needs_human 0.010, customer_impact 0.013, math_word 0.032 — "same top"
  flips there first. (evidence: bench/results/compress/cerno_base_20261022_175841.json)
- 21:05 [plan] (clock-read; the two entries below were written 21:05-21:11, after the pause
  18:21-21:00) Re-scoped as the **Cerno pilot** (PC only, ~2 h): activation-aware low rank of
  gate / up / down / qkv and MLP channel pruning (activation-magnitude x weight-norm scores, down
  refit by least squares), FLOP saving vs agreement on the 52 rows; bar max |dp| <= 0.0146 and
  top >= 51/52. First the 22-row subset (requests 0,2,4,6,10,12,13: support_triage, billing_angry,
  sentiment_review, intent_smart_home, contract_clause, math_word, phishing), then all 52.
- 21:11 [for:layerskip] **`.pq` reader + 4B-in-torch harness to reuse** (worktree
  `agent-a0bab39a2c3dd469b`, `tools/compress/`): `pqread.py` (`PQ(path).get(name)` dequantizes
  Q8_G32 / Q4_G32 / Q4K_G32 / BF16 / F32 to fp32 numpy; checked bit-equal to tools/quant.py on the
  0.8B file), `cerno.py` (`Cerno()` = the 4B `.pq` in transformers' `Qwen3_5ForCausalLM`, untied head;
  `load_rows()` = the 52 rows from ref_llama_pc.jsonl with label ids; `evaluate()` scores them with
  cerno-pi's readout, prefix cache per request; `compare()` = max |dp| / same top vs the reference
  and vs a base run). Load 50 s from a copy of the .pq on the WSL disk (`/root/lowrank/cerno/`;
  220 s from /mnt/e), 52 rows ~290 s with 8 threads while the PC is shared. Memory ~11 GB RSS.
  Note: `torch.set_num_threads(8)` (common.py) — 20 threads is ~10x slower on this PC.
- 21:24 [blocker] My calibration + sweep job (pid 90500, 14.6 GB RSS) was taken by the **WSL OOM
  killer** at 21:24 (dmesg: global OOM, load average 84; `/eval.py` of another agent at 6-8 cores
  and eight host `pinfer_run`s were running). Nothing of mine was lost but ~25 min of compute.
- 21:26 [blocker] **WSL is wedged**: `wsl -e` returns `Wsl/Service/E_UNEXPECTED` ("Catastrophic
  failure") while `wsl --list` shows Ubuntu Running. I am NOT running `wsl --shutdown` (it would kill
  every agent's WSL jobs): lead, please decide. My relaunch (leaner: Q8 embedding / head kept as
  int8 codes, -4.7 GB; 16 calibration prompts) is ready to go the moment WSL answers; I will try a
  Windows-side Python fallback in the meantime.
- 21:28 [measured] WSL came back at 21:27 ("up 0 min": restarted by someone else or by the service;
  not by me); `/root` survived. Lean job relaunched 21:27 (detached with setsid; Q8 embedding /
  head path checked exact: rows diff 0, head matvec 4.8e-6). Note for everyone:
  the OOM at 21:24 hit the largest process; big PC jobs should state their RSS on the board.
- 21:31 [using] **WSL big-model job: compress Cerno sweep** (calibration 16 prompts + 3-layer matrix
  study + 8 variants on the 22-row subset), started 21:30 (the 21:27 launch died with the lead's
  `wsl --shutdown` of the same minute), **until ~22:35** (earlier if the PC is quiet); pid 7962,
  8 threads, 25 GB were free at launch; **measured RSS after loading: 19.3 GB** (21:31; glibc keeps
  the heap of the dequantization temporaries — `malloc_trim` added for the next runs), **~15 GB RSS** (fp32 layer weights 14.3 GB + int8 embedding / head
  1.3 GB). Why not bf16 (lead's 21:28 rule): a Q4_0 value d·q (fp16 scale × 4-bit code) needs 15
  significant bits — bf16 has 8 — so bf16 storage perturbs the QAT weights by up to 1.6% of a
  quantization step, and bf16 compute on this AVX2 CPU is emulated (slower than fp32) and adds
  ~0.4% noise per op, which a 0.0146 bar cannot absorb. **If 15 GB is not acceptable while the other
  pilots wait, say so:** I have an exact ≤ 8 GB variant ready to code (keep the Q4_0 codes as int8,
  dequantize per matmul: 3.8 GB + transients) and will use it for the full-52 confirmation runs.
- 21:46 [done] the 21:30 CPU job died with the 21:45/21:46 WSL restart (calibration at 1343 of
  3427 tokens). Under the 24 GB cap a 15-19 GB fp32 CPU job is out; slot B is numerics' until ~22:15.
- 21:58 [using] **GPU side job (slot A side, next to layerskip): compress Cerno sweep on CUDA**,
  the 4B's Q4_0 codes resident on the GPU (3.6 GB) and dequantized per matmul in fp32 (TF32 off:
  exact fp32 compute, the same numbers as the CPU run to ~1e-5), Q8 embedding / head as int8 codes
  (0.7 GB), modified matrices fp16 on the GPU replacing their codes: **≤ 7 GB VRAM** (7.8 free at
  21:50), **host RAM ≤ 6 GB** (transient while packing; `malloc_trim` after), 4 host threads,
  venv `/home/optim/.venvs/layerskip` (torch 2.11 cu128, read-only use), **until ~22:45**.
  [done] will follow.
- 21:55 [measured] **GPU path verified and running** (launched 21:54): 4B load 12 s, 8 decision
  rows in 9 s (the CPU fp32 path: ~45 s per row under load), max |Δp| vs the CPU fp32 base
  **9.8e-7** over 8 rows (fp32 compute with TF32 off; the Q4_0 codes dequantized per matmul),
  **5.5 GB VRAM**, host ~3 GB. For:layerskip, for:prompt: `tools/compress/cerno.py` `Cerno()` now
  does this by default when CUDA is available (`PackedLinear` / `Q8Rows` buffers, `set_weights`
  takes dense tensors or `(A, B)` factor pairs; `reset()` reloads the codes) — a 4B emulation that
  fits next to another GPU job (evidence: `/root/lowrank/cerno/smoke_gpu.log`).
- 21:59 [measured] **Cerno 4B per-matrix low-rank budget** (whitened SVD, calibration metric,
  3 layers): at 75% of a matrix's MACs the output error is 2–13% per MLP matrix (layer 15 `down`
  13.1%, `up` 10.2%; layer 0 / 31: 2–5%), 0.2–3% for `qkv`; at 50%: 8–24% (evidence:
  `bench/results/compress/cerno_matrix_20261022_215927.json`).
- 22:11 [dead-end] **Training-free low rank on Cerno's MLPs**: `lr_mlp:0.75` (−16.8% of the matmul
  MACs, fp16 factors) → max |Δp| **0.438** vs the reference (bar 0.0146), 17/22 top answers on
  the subset. 30× outside the bar; finer ranks pointless (the bar needs ~0.1% per-matrix error).
  (evidence: `cerno_sweep_partial_20261022_221119.json`). Verdict: no-go without fine-tuning
  (for:distill: a LoRA / distillation pass after pruning is the only route to ≥ 20% MACs).
- 22:1x [measured] `prune:6912` (−15.9%): PRUNE_BOARD.
- 22:1x [done] GPU side job ended (sweep stopped after two points on the owner's request;
  nine planned points not run). Final write-up: `engine/COMPRESS.md` §1, `engine/PILOTS.md`.
- 17:40 [measured] PC tooling note: torch on the Core Ultra 7 265F runs a 512-token 0.8B forward in
  1.5 s with 8 threads but ~10x slower with the default 20 (efficiency cores + other jobs): use
  `torch.set_num_threads(8)` / `OMP_NUM_THREADS=8` for PC-side Python.
