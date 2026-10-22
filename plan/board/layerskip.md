# layerskip (pilot: can Cerno 4B skip layers / exit early and give the same decisions?)

Worktree `E:\Personal_tries\Raspberry_Pi\.claude\worktrees\agent-aba49cb66390bd91f` (branch
`worktree-agent-aba49cb66390bd91f`). Scripts `tools/pilot/layerskip/`, results
`bench/results/layerskip/`, record `engine/PILOTS.md` "Layer skipping". PC only; the Pi is offline.

Plan (budget ~2 h, started 21:07 KST):
1. Harness: the 4B in exact fp32 on the PC GPU (RTX 5060 Ti, WSL CUDA works) from the `.pq`
   (int8 codes + fp16 scales on the device, dequantized per matmul); 52-row protocol of
   tools/compress/cerno.py (T = 1.534, top-260 rule); check vs the reference (max |dp| ~0.0094 expected).
2. Single-unit skips: each layer x {skip, nomlp, nomix} x {all tokens, question tokens only}.
3. Accumulation: ranked / greedy over the least harmful units until the bar breaks.
4. Early exit after layer k = 24..31 (final norm + head on the layer output).
5. Report: table, verdict (go if >= 20% of the matmul MACs removed within the bar), pinfer plan.

- 21:07 [plan] started; `git reset --hard origin/main` (dd7bc9e).
- 21:16 [measured] PC tooling: CUDA works from WSL (driver 591.74, CUDA 13.1); uv venv
  `~/.venvs/layerskip` (user optim) with torch 2.11.0+cu128 + transformers 5.20.2 took 4.5 min
  (download over the PC link, ~3 GB). `/root/venv-ref` is root-only (CPU torch); not needed.
- 21:20 [plan] Reused tools/compress/pqread.py (compress branch, not merged) as a copy with a
  `codes_scales` reader; the HF decoder-layer forward is replaced by a per-layer-mode version;
  positions and the causal mask are passed explicitly so a skipped attention layer (empty KV cache)
  cannot shift them (HF derives both from the first attention layer's cache length).
