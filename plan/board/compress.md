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
- 17:40 [measured] PC tooling note: torch on the Core Ultra 7 265F runs a 512-token 0.8B forward in
  1.5 s with 8 threads but ~10x slower with the default 20 (efficiency cores + other jobs): use
  `torch.set_num_threads(8)` / `OMP_NUM_THREADS=8` for PC-side Python.
