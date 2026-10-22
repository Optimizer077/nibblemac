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
- 17:40 [measured] PC tooling note: torch on the Core Ultra 7 265F runs a 512-token 0.8B forward in
  1.5 s with 8 threads but ~10x slower with the default 20 (efficiency cores + other jobs): use
  `torch.set_num_threads(8)` / `OMP_NUM_THREADS=8` for PC-side Python.
