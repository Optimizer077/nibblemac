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
- 21:33 [plan] after lead 21:28 (WSL crash 21:20, restart 21:27; one 4B torch job at a time, announce it):
  my job died at load time; nothing measured yet. My 4B lives on the GPU, not in host RAM: the body as
  int8 codes + fp16 scales (3.8 GB VRAM), the head fp32 (2.5 GB VRAM), exact fp32 math; host RAM is only
  the memmap of the .pq during load (page cache) + ~1 GB.
- 21:33 [using] WSL big-model job: layerskip `skip_study.py check` (4B on the GPU, ~1 GB host RAM after
  load), until ~21:45; then `base` / `single` / `greedy` back to back, posted here as they start.
- 21:36 [measured] **The GPU is shared too**: a Windows-side `python3.13.exe` (pid 40396, 9.7 GB working
  set; another agent's job?) holds ~10.3 GB of the 16.3 GB VRAM (nvidia-smi 11.9 GB used, 0% util at
  21:36). Free: ~4.4 GB. My model now keeps the body packed (nibbles + fp16 scales, 2.0 GB) and the
  head as int8 codes (0.7 GB), ~3 GB VRAM in all; `LAYERSKIP_HEAD=cpu` moves the head to host RAM
  (2.5 GB fp32) if the GPU gets tighter. Whoever owns pid 40396: please post a [using] line.
- 21:45 [measured] harness check passed: explicit positions / mask vs the plain HF path max |dh| 5.7e-6,
  prefix cache + suffix vs one full forward 6.2e-6; needs_human p = 0.4985 / 0.5015 (ref 0.4950 /
  0.5050). Load 95 s, 2.59 GiB VRAM, first forward 45 s (CUDA warm-up). The 52-row timing pass was
  killed (exit 9) during the lead's 21:45 overload cleanup.
- 21:45 [using] GPU side job: layerskip, 2.6 GB VRAM, host threads 4, host RAM <= ~2 GB (embedding
  codes + CUDA context; the .pq memmap is dropped after the load), `skip_study.py all` (base + early
  exits, 192 single-unit variants, 4 accumulations), until ~23:00 at the latest; [done] will follow.
- 21:47 [plan] after lead 21:46 (WSL restarted, 24 GB / 12 cores): my `all` run died at 21:46 right after
  its load (8 s from the page cache; host maxrss 4.6 GiB at the load peak = the .pq memmap, released
  after). Re-run at 21:47 as the GPU side job (slot A side, 2.6 GB VRAM, 4 host threads).
- 21:49 [using] GPU side job: layerskip `skip_study.py all` (pid in WSL, python of ~/.venvs/layerskip),
  2.6 GB VRAM, 4 host threads, host RSS ~1.5 GB steady (4.6 GB peak for 8 s at load = the .pq memmap);
  alive after the 21:47 restart (loaded 21:47:14). One 52-row evaluation = 36 s; until ~22:50.
- 21:49 [measured] base (harness vs reference): max |dp| 0.0094, mean 0.0023, same top 51/52 (= compress's
  CPU emulation). **Early exit is dead**: exit after layer 24..30 gives max |dp| 0.47-0.69, mean 0.17-0.22
  (top answer often survives: 52/52 at k = 25, 30; the calibration does not). L0 skip: 14/52; L1 skip:
  max |dp| 0.36, 45/52. (evidence: bench/results/layerskip/base_20261022_214806.json, all.log)
- 22:09 [dead-end] **No single layer of Cerno 4B can be skipped within the bar** (32 whole-layer skips,
  every token, 52 rows): best L25 max |dp| 0.0214 (top 52/52), then L26 0.0324, L29 0.0358, L21 0.0453,
  L20 0.0595, L28 0.0761; all others > 0.10; layers 0-15 > 0.30 (L0: 14/52, L11: 0.61, L31: 0.51 with
  the top answer kept 52/52 - the last layer sets the calibration). Each skip saves only 3.0-3.2% of the
  MACs, so the >= 20% target would need 6-7 layers; even the single best breaks 0.0146 by 1.5x.
  (evidence: single_fromlog_*.json, all.log 21:48-22:08)
- 22:09 [using] GPU side job: layerskip ranked accumulation (L25, +L26, +L29: 3 evaluations, ~2.5 min,
  2.6 GB VRAM), then [done]. Per the lead 22:05: the MLP-only / mixer-only / question-only follow-ups
  are cancelled (the singles show no go); the `all` run was killed at 22:09.
