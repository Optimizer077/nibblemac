# Changelog

All changes to this project **and to the Pi** are recorded here, newest first.
Details for each day: `docs/journal/YYYY-MM-DD.md`. Pi configuration changes: `docs/pi-state.md`.

## 0.2.0 (unreleased)
- ABI: `pinfer_params.kv_dtype` and `pinfer_spec_params.adaptive` (added after 0.1.0) make this
  0.2.0, SONAME `libpinfer.so.0.2`; binaries built against the 0.1 header must be rebuilt.
- State blobs are format 2 and carry a fingerprint of the model file; 0.1 blobs are refused.
- Stricter `.pq` validation (every tensor's size, header bounds); the caller's CPU affinity is
  no longer changed. Details: `engine/HARDENING.md`. `docs/release-0.1.0.md` stays the 0.1.0 note.

## 2026-10-23

### Decided: distilled decider pilot no-go (t002)
- Recovered the interrupted agent's scripts and verified its completed three-epoch 0.8B student:
  39/52 held-out top answers (75.0%, below the 85% / 45-of-52 bar); mean option probability
  error 0.0888, maximum 0.6274. Choice 16/19, yes/no 17/21, score 6/12.
- Fixed pilot evaluation to use original request types and record CUDA memory peaks. Aggregate
  evidence and hashes: `tools/pilot/distill/results/t002-summary.json`; report: `engine/PILOTS.md`.
- No extra training: the completed run's dev agreement plateaued at 42/49. No Pi changes;
  weights and full labels remain local because Cerno's weight repository has no licence file.

## 2026-10-23

### Pi: pinfer 0.3 installed (channel agent t010, 16:55)
- `engine/pinfer_run_fast` `c6dcc195` and the Pi's Python package 0.3.0: checkpoint and opt-in GPU now available on the Pi;
  GPU off by default, output identical to the previous build (58/58, 5/5). Backups `*.pre-v03`; undo in `docs/pi-state.md`.

### Pi: shared binary = gated 4B-prefill build (channel agent t003, 11:41)
- `engine/pinfer_run_fast` `d48ffda1` → `508f1d74` (prefill tail + groups of 7s + 5s; Cerno decisions −4.4%, bit-identical);
  spec_exact 58/58 on the installed binary; old builds kept as `*.pre-csp`. `docs/pi-state.md` "Files we placed".

### Decided: no GPU mode for Qwen3.5-0.8B
- A Q4K GPU kernel (exact) and its engine split were built and gated, but 0.8B prefill got 1.6–2.8% slower and decode
  did not change: not merged (`engine/GPU.md`). The GPU mode stays for Cerno 4B.

### Added (GPU in the engine, 0.3)
- **Opt-in CPU + GPU prefill** (`gpu_split`, `pinfer_run --gpu`, `Model.load(gpu_split=)`; recommended 0.33): Cerno 4B
  decisions 20.67 → 17.26 s (**−16.5%**, 3 rounds); off by default (bit-identical, all gates pass); with it on,
  probabilities within 0.0128 of the reference and two near-50/50 flips (owner-accepted). Clean CPU fallback.
  ABI / package 0.3. `engine/GPU.md`, `docs/library.md`.

### Added (checkpoint)
- `pinfer_checkpoint` / `pinfer_rewind` (and `Model.checkpoint()` / `rewind()`): rewind to a saved prefix without copying
  the state blob; Cerno decisions −1.6%, bit-identical; 106 MiB less peak memory. Not yet in the Pi's `pinfer_run_fast`.

### Added (GPU)
- GPU co-processing pilot: **go** (CPU + GPU = 1.41–1.47× CPU alone on the Cerno 4B shapes).
- `engine/gpu/v3d.c`: the QPU Q4_0 kernel from C through our own DRM calls (no GPL code), byte-identical to the
  Python path, 28 GOP/s; GPU vs CPU differ by ~6e-7 relative, so the coming engine path is opt-in.

### Pilots
- Distilled decider: **no-go** (0.8B student 39/52 = 75% top-answer agreement with Cerno 4B, bar 85%).
- Layer skipping: **no-go** (no single layer or early exit stays within the probability bar; best L25 0.0214 vs 0.0146).

### Added
- `channel/channel.md`: conversation with a worker agent the owner runs; the lead writes tasks, reviews, commits.

## 2026-10-22

### Pilots closed (22:28): no-go for prompt economy, compilers, table-lookup / asm kernels, numeric approximations
- Cerno 4B is calibrated to its exact prompt and its Q4_0 numerics: every output-changing shortcut measured today exceeds the
  agreement bar; the exact kernel routes give <= 3.5% end to end. Verified full host gate on main (speed4b merge) all PASS.
- Interrupted by the API session limit (work saved on branches): layer skipping, low-rank, GPU co-processing, in-engine
  checkpoint, distilled 0.8B decider (epoch 0: 41% -> 67% dev agreement).

### Merged (21:25): 4B prefill tuning (speed4b agent) — note: landed as commit `dcc2219`, whose message says "docs(pi)"
- Prefill tail (the last layer computes only the last token's rows when only its logits are read) and prefill groups of
  7s + 5s with one fused job per call; bit-identical. Cerno decisions -4.4%, 4B prefill +4-5%, 0.8B prefill 43.8 -> 45.1,
  decode unchanged. `engine/bench/KERNELS.md` Part 4; `tests/test_prefill_tail.py`.

### In-context checkpoint (checkpoint agent, 18:00–; branch `worktree-agent-a4d7c6bb45401045d`)
- API (additive, version unchanged): `pinfer_checkpoint` / `pinfer_rewind` /
  `pinfer_checkpoint_clear` / `pinfer_checkpoint_n_past` — one rewind point held inside the
  context: the checkpoint copies only the conv + DeltaNet states of the linear layers (0.8B
  20.2 MB, 4B 52.7 MB, whatever the context length) and remembers the position; the rewind
  copies them back and returns to it, bit-identically, any number of times; the KV rows below it
  stay in place (never rewritten while the checkpoint is valid), rows above are overwritten
  later. Dropped by `pinfer_reset` / `pinfer_state_load` / the resetting generate calls.
  Python: `Model.checkpoint()` / `rewind()` / `clear_checkpoint()`, `supports_checkpoint`,
  `checkpoint_n_past`; `save_state` / `load_state` unchanged. Audit recommendation C3
  (`docs/network-audit.md` §4): removes the 4 Python copies of the ~51 MB snapshot blob and the
  K/V copy per restore in a Cerno decision; long states fit (3.5k → ~17k tokens).
  `engine/CONTEXT.md` §10, `docs/library.md` §3 / §4 / §9.
- Tests: `pinfer_run --rewind-at S`; `tests/test_state.py` run D (checkpoint → two rewound
  detours → the rest of the prompt must equal the full run, plain / spec 3 / adaptive); 5 pytest
  cases in `tests/python/test_state.py`.
- cerno-pi (`feat/checkpoint`): `Decider` restores the shared prefix with the checkpoint when
  the library has it, else `save_state` / `load_state` (`checkpoint=` argument,
  `decide_bench.py --no-checkpoint`); probabilities bit-identical between the paths.
- Measured: see `engine/CONTEXT.md` §10c (host gates) and §10d (Pi), filled in by this agent's
  journal section.


### Prompt economy pilot (prompt agent, 21:08–; branch `worktree-agent-ae99d2e23cc77288a`)
- Added `engine/PILOTS.md` ("Prompt economy") and `tools/pilot/prompt/` (`variants.py`: the prompt
  layouts and their token counts, the base rows checked equal to the reference's ids 52/52; `run.py`:
  fp32 torch emulation of the 4B from its `.pq`, GPU/CPU split; `report.py`: the table), results in
  `bench/results/pilot/prompt/` (PC emulation, no Pi numbers).
- Found: the upstream decider defines a multi-question state-first row and a schema-first layout
  (question blocks first, then the state, then the answer slots); Cerno's engine uses one row per question.
- Measured (tokens, 18 requests / 52 questions): compact templates save 4–11%, option sharing 1%,
  multi-question rows cost 8–10% more, schema-first 8% more per isolated decision and 52% less with the
  question block cached across decisions.
- Measured (PC fp32 emulation vs the llama.cpp reference, max |dp| / same top of 52): base 0.0094 / 51
  (anchor); compact templates 0.145–0.262 / 48–51; multi-question rows 0.149–0.153 / 50–51; schema-first
  0.593 / 47. **No-go for every variant**: no layout change stays inside the identical bar (0.0146), and
  the only large saving (schema-first, −52% with the question block cached) changes answers. The 4B's
  calibration is specific to the one-question state-first row.
- Process: a first WSL CPU run was OOM-killed (three pilots loading the 4B in the 31 GB VM); rerun on the
  Windows side with the GPU holding 24 fp32 layers; the run was stopped at 21:45 before two zero-value
  variants when the lead asked for the PC's model slot.


### Compilers pilot (toolchain agent, 17:37–; branch `worktree-agent-a75a53a7559d5b274`)
- Added `engine/TOOLCHAIN.md` (every compiler / flag set tried, with numbers and result files) and
  `engine/PILOTS.md` (index of pilot verdicts; first entry "Compilers").
- PC (WSL, apt): cross GCC 13.4 / 14.3 / 16.0.1 and clang 20.1.8 / 22.1.2 installed next to the shipping
  GCC 15.2; the engine builds static for the Pi with all of them at the project flags.
- Exactness under qemu (kbench `--check`, 880 configurations per build): GCC 13 / 14 / 15 / 16 and clang 22
  produce the shipping build's output bytes on every row; **clang 20 rejected**: at `-O3` it miscompiles a
  max-by-magnitude loop (kbench's Q4 fill, 1.7 % of blocks get a different maximum), i.e. it changes plain
  C floating-point results. No compiler contracts multiply-adds (`-ffp-contract=off` honoured by all).
- Dead end: `-fno-schedule-insns` on `kern_quant.c` removes every stack spill of the Q4 loops but is 6–15 %
  slower on the Pi (speed4b's `cerno-speed-kb8-nosched_*`): instruction count is the wrong proxy on the
  in-order A72.
- Pi (1.5 GHz, `toolchain-kb-r1_*`, `toolchain-e2e4b-r1_*`): GCC 14 and 16 equal GCC 15 within noise on the
  Q4_0 kernel (n = 5 / 7, 4B shapes) and on 4B prefill p256 (6.79 / 6.92 vs 6.92 tok/s); clang 22 is 5–15 %
  slower on the kernel, clang 22 without its machine scheduler 12–22 % slower. **Verdict: no-go**, the
  Makefile stays at GCC 15 `-O3 -mcpu=cortex-a72`; the `load` line of the 3.4 GB model is 6 ms (mmap), the
  page-in happens in the warm-up. Pi scratch files removed (`engine/pinfer_run_tc*`, `tmp/tc`).
- Scratchpad incident: three pre-existing files in the shared session scratchpad root overwritten (17:39–17:54)
  and removed (18:18); nothing in the repo or on the Pi affected; details in the journal.


### GPU co-processing pilot (GPU agent, 17:30–22:12; branch `worktree-agent-a459e8864e2e5ea3b`)
- Added: `engine/gpu/` — an integer Q4_0 x int8 matmul kernel for the Pi 4's V3D 4.2 QPUs
  (two rows packed per `smul24` operand, exact int16 block sums, activations as a uniform stream;
  generator `q4gpu.py`, dual-issue list scheduler `qsched.py`, numpy emulator check
  `qemu_check.py` bit-exact for n = 1–12, Pi benchmarks `bench_q4.py` / `corun.sh`, hardware
  diagnostics `diag.py`); notebook `engine/GPU.md`; pilot record `engine/PILOTS.md`
  ("GPU co-processing"); `engine/bench/kbench.c` gets the Qwen3.5-4B shapes `c4*` (same hunk
  as speed4b, selected by name only).
- Measured (Pi, stock 1.5 GHz, V3D 500 MHz): GPU alone 27–28 GOP/s on 9216x2560 / 2560x9216 /
  8192x2560 at batch 7 and 64, bit-exact vs the kernel's fp32 model; CPU + GPU together
  77.8 / 78.0 / 74.1 GOP/s vs CPU alone 55.2 / 55.4 / 50.3 = **1.41x / 1.41x / 1.47x**, CPU keeps
  94 %, throttled 0x0 (`bench/results/gpu-q4-alone_20261022_221029.json`,
  `gpu-corun_20261022_221209.json`). Verdict: go; scale-up design (DRM ioctls, class (b) opt-in)
  in `engine/GPU.md`.
- Found: py-videocore6 is GPL-2.0 (driver) / MIT (assembler), not BSD-3: used only as a venv
  dependency on the Pi, never vendored; numpy 2.5 breaks it (pinned 2.3.1). TMU loads without
  the `thrsw` signal hang the V3D 4.2 (two slots lost to it).
- Pi: `~/pinfer/venvs/gpu` (numpy 2.3.1, ioctl-opt, py-videocore6 @ `175741a` packages copied
  in, licence note) kept for the scale-up; `~/pinfer/tmp/gpu` removed. Rows in `docs/pi-state.md`.

### Pi: overclock test reverted (21:01)
- Boot partition edited on the PC (F:): `tryboot.txt` back to the custom-kernel test file, overclock files removed; `config.txt` / `cmdline.txt` unchanged; standard 1.5 GHz profile. `docs/pi-state.md` #5.

### Optimization agents, overclock test (lead, 16:55-18:20)
- Added: `docs/network-audit.md` (merged): a Cerno decision is ~150 ms per prompt token, 93% Q4_0 matmul;
  ranked candidates; exact head screening for 0.8B measured as a dead end.
- Added: `harness/pi.py` pause file (`harness/pi.pause`, exit 75) for reboot windows; `plan/board/`, a findings
  board shared by the optimization agents.
- Measured: 1.5 GHz baselines (`oc-1500-*`); A72 dual-issue rules (`lut-ubench2_*`): SMLAL pairs with D-form
  float / add ops, table lookups cannot beat SMLAL.
- Pi: overclock run 1 (2000 MHz, `over_voltage=6`, one-time tryboot) froze under load; not stable on this
  board; `docs/pi-state.md` #5, restore pending after the owner's power cycle.
- Decided: focus on Cerno (owner); the decode-streaming and chat-drafting agents stopped; work shut down at
  18:20 with the agents' work in progress committed to their branches (not merged).

### Network audit (audit agent, 16:55–17:50; branch `worktree-agent-a4e27a8ffe358907e`)
- Added `docs/network-audit.md`, Cerno 4B first (owner's focus): the network walk-through (32 layers,
  shapes, bytes, cost per prefill token, how a decision drives it); per-operation time budgets for the 4B
  prefill token and one Cerno decision, and for Qwen3.5-0.8B (plain step, k = 4 round, prefill, 4096
  context), with bytes, operations, achieved rate and % of roof; gaps (Amdahl); 25 Cerno and 13 0.8B
  candidates with exactness class, derivation, effort and prior attempts, and a ranked list for Cerno
  decision time that includes the other agents' board findings; Cerno memory budget; 2.0 GHz estimates.
- Findings: a Cerno decision is ~150 ms per token, 93% of it the Q4_0 kernel running compute-bound at
  93–98% of its kernel-alone rate (50–55% of the int8 instruction peak); 38% of the set's time is the
  shared prefix, 61% the question rows, ~2% snapshots, restores and Python. 44% of a question's tokens
  are template but none can move into the snapshot exactly. Exact head screening for 0.8B is dead
  (bounds keep 100% of the rows on real hidden states). SDRAM overclocking is not supported on the Pi 4
  (Raspberry Pi documentation).
- Recommendation: the approved 2.0 GHz test (est. decision 27.6 → 20.8–21.7 s), merge speed4b (−4.4%
  measured), a C-side state checkpoint (−1 to −1.5%, long states fit), then lut's dispatch-pairing
  question; class (b) only as measured opt-ins. Exact path: 19.6–20.5 s per decision.
- Added `tools/audit/` (budget, kbtable, cerno_rows, shapes, head_screen, cerno_ids, audit_08b.sh,
  audit_cerno.sh). Pi: scripts and id files pushed to `~/pinfer/tmp/audit/`; the two queued profile
  benches (`audit-08b-profile`, `audit-cerno-profile`) never got the lock before the overclock window
  (17:39) and were cancelled, so no result files. No binary, clock or configuration change; no engine
  code changed. Board: `plan/board/audit.md` and the answer in `plan/board/lead.md` (main checkout).

### README animations (figures agent, 15:08–16:25; branch `worktree-agent-a1fae4999a86b0632`)
- Added: `tools/make_animations.py` (helpers in `tools/animkit.py`) and two looping GIFs in four variants
  each (desktop 1676 px / phone 1080 px × light / dark, 12.5 frames/s, 0.45–0.92 MB each):
  `anim_decode`, one plain decode step (424.9 MB of weights stream from DRAM through the 1 MB L2 into
  the four cores, layer after layer, 126.7 ms per token), and `anim_speculative`, one k = 4 round (three
  chained drafts at ~8 ms, one 176 ms verify sweep over 5 positions = 1.39× a plain step, 3 drafts + 1
  token of the model's own kept: ≈ 211 ms against 508 ms for 4 plain steps; a prompt-lookup inset).
  Numbers from `mm2-e2e-decode_20261022_140420.json` and the lab notebooks, cited in the script; no
  project name in the images.
- README: `anim_decode` replaces `pi4_hardware` in §1 "The hardware" (the static diagram stays in
  `docs/ARCHITECTURE.md` §9) and `anim_speculative` replaces the k = 3-era `speculative_schematic` in
  §2.3 (its PNGs stay in `docs/figures/`); captions and alt texts describe each sequence. Title and
  header unchanged. `docs/RESULTS.md` §13 lists both.
- Existing figures unchanged: `make_figures*.py` re-render all 64 PNGs byte-identically.

### Merged (lead, 16:00): GGUF import pieces from the Cerno work
- `2a7fae1`: untied output head (`lm_head.weight`), `linear_attn.A`, `pinfer_eval_select` /
  `Model.eval_select`, `tools/gguf_to_pq.py` (lossless GGUF Q4_0 / Q8_0 → `.pq`). Host gates on the
  merged tree all pass (pytest 130, malformed 43/43); the Pi build is byte-identical to the gated
  `pinfer_run_cerno2` (`spec_exact` 58/58, `ctx_check` 5/5). Qwen3.5-0.8B output is unchanged.


### 4B prefill (Cerno) tuning (speed4b agent, 15:45–18:05; branch `worktree-agent-aff7b9e370c62872c`)
- Engine (`fd0ed20`): **prefill tail** — a call that needs only the last token's logits
  (`pinfer_eval`, `pinfer_eval_select`, `pinfer_run` without `--dump-*`) runs the last full-attention
  layer's query, attention, o_proj, MLP and final norm for the last token only; the other rows only
  store K / V. 1/32 less matmul work per prefill token on Qwen3.5-4B. `pinfer_run --dump-last f`
  writes those logits; `tests/test_prefill_tail.py` compares them with `--dump-logits` byte for byte.
- Kernels (`69b31a3`): quantized calls of more than 7 vectors are split into groups of ≤ 7 (was: only
  above 16; 8–16 vectors as one group cost 33–40 instead of 28–33 cycles per block-vector), groups
  are 7s with the remainder in 5s / 4s (6-vector passes are 8–10 % slower per vector), and all groups
  of a call run in one job after one parallel activation quantization.
- Bench tools: kbench Qwen3.5-4B shapes (`c4*`) and `--gfused / --qgroup / --gmode`; `lbench4`
  (recurrence at 4B head counts); `engine/bench/cerno_check.sh` (4B bit identity of two builds).
- Measured at 1.5 GHz, 3 rounds: 4B prefill 64 / 256 / 512 tokens 6.49 / 6.62 / 6.48 → **6.78 / 6.82 /
  6.80 tok/s** (+4.6 / +2.9 / +4.9 %); Cerno decisions (5 requests) mean 21.93 → **20.95 s** (−4.5 %),
  answers identical; 0.8B prompt64 plain 7.81 → 7.80, default speculative 20.19 → 20.26, prefill
  43.8 → **45.1 tok/s**. All outputs bit-identical (kbench yhash, spec_exact 58/58, ctx_check 5/5,
  core2, 4B logits and `eval_select`, host gates). Dropped after measuring: column-chunked activations,
  fused recurrence sweeps, `-fno-schedule-insns`. Details: `engine/bench/KERNELS.md` Part 4.
- Pi: `~/pinfer/engine/pinfer_run_csp1` added (`docs/pi-state.md`); scratch (`tmp/csp`,
  `engine/pinfer_run_csp0`) removed after the lead's overclock pause.


### Table-lookup kernels and the multiply loop's dispatch order (lut agent, 17:12–22:20; branch `worktree-agent-aa5998fa1cda1365d`) — measured, nothing integrated
- Engine unchanged (rebuild byte-identical to 69b31a3, md5 `508f1d74`). Added bench tools:
  `engine/bench/ubench2.c` (A72 pairing rules with the dispatch order controlled),
  `engine/bench/tbench.c` + `qvar_*.c` (A/B bench of Q4 / Q4K kernel variants, byte-compared with the
  C loops; `--core` times one L1-resident tile), `tools/gen_kern_q4s.py` (asm generator →
  `engine/bench/q4{s,e,m}_gen.h`). Details: `engine/bench/KERNELS.md` Part 5.
- Measured: SMLAL pairs perfectly with D-form ops (8.34 cycles per 8 + 8) but Q-form ops cost
  +0.5–0.7 cycle each next to it; TBL .8b issues 2 per cycle, TBL .16b 1.5 cycles
  (`bench/results/lut-ubench2_20261022_175209.json`, `lut-ubench2b_20261022_175214.json`).
- Dead end: exact T-MAC-style LUT kernel 4.5–8× slower than SMLAL (0.49 vs 0.26 instructions per
  MAC; c4gate n = 7 10.6 vs 53.7 GOP/s) (`lut-tbench-r1_20261022_215231.json`).
- Dispatch-ordered asm passes: −9 to −13 % cycles per block-vector on Q4 (one L1 tile, n = 7:
  26.0 → 22.6), exact (416 kbench configs, yhash = 69b31a3), but 4B prefill only +2.1 % (64 tokens)
  / +3.5 % (256) in one A/B round vs `pinfer_run_csp1` (`cerno-lut-e2e4b-r1_20261022_221018.json`,
  gitignored with the Cerno results): below the 5 % bar, not integrated.
- Pi: scratch `tmp/lut/`, `engine/pinfer_run_lut0/1` removed; `pinfer_run_fast` untouched (`d48ffda1`).

### Renamed (lead, 15:35): piquant → **nibblepi**
- Owner asked for a better name; a naming review checked meaning, collisions (GitHub, PyPI, npm, web)
  and trademark: `nibblepi` (a nibble = 4 bits, the weight size; pi = the board) was the only candidate
  with no collision. Repository `github.com/Optimizer077/nibblepi` (GitHub redirects the old URL); the
  engine and package keep the name `pinfer`. README title, logo alt text, banner, release notes and
  docs updated; README footer adds the Raspberry Pi trademark notice.

### Release 0.2.0 (lead, 15:40) — notes in `docs/release-0.2.0.md`
- Speculative default: adaptive k ≤ 4 with prompt-lookup drafts (ngram 7); on prompt64 20.22 tok/s vs
  19.78 with k ≤ 3 and 18.42 with fixed k = 4 (`kcap-merge6_20261022_145226.json`).
- README headline 20 tok/s, prefill 43 tok/s, chat answers 11.0–14.8 tok/s; figures re-rendered.
- The Pi's `engine/pinfer_run_fast` / `pinfer_ppl_fast` are the gated merge build `d48ffda1`
  (previous kept as `*.pre-0.2.0`).

### Hardware figures (figures agent, 13:35–14:10; branch `worktree-agent-a83b76690e63d2416`)
- Added: `tools/make_figures3.py` and three figures in four variants each (desktop / phone × light /
  dark): `pi4_hardware` (BCM2711 cores, caches and DRAM, and what pinfer keeps in L1 or streams from
  DRAM per token), `roofline` (DRAM roof 4.20 GB/s, Q4K kernel 47.5 GOP/s on qkv at a batch of 16,
  int8 instruction peak 96 GOP/s; plain decode 11.9, verify pass 39.3, prefill 37.2 GOP/s) and `arm_isa`
  (multiply-accumulates
  per 128-bit instruction: fp32 FMLA 4, int8 SMLAL 8 on the A72; fp16 FMLA 8, SDOT 16, SMMLA 32 not).
- README §1 opens with "The hardware" (the three figures, with L1/L2, DRAM bandwidth, arithmetic
  intensity and NEON defined); the three-facts list below it is shortened where the figures now show the
  numbers; §2.1 names the widening adds (`SADDLP`/`SADALP`) correctly. New `docs/ARCHITECTURE.md` §9
  "Hardware" with the derivations and the Arm / Raspberry Pi sources; `docs/RESULTS.md` §8 and §13.
- Not verified, so labelled as an estimate: the Pi 4's theoretical DRAM peak (≈ 9.6 GB/s from
  "LPDDR4-2400" × an assumed 32-bit bus; Raspberry Pi does not publish the width).
- Existing figures unchanged: `make_figures.py` and `make_figures2.py` re-render byte-identically.

### GGUF import, untied head, selected logits (Cerno agent, 13:45–16:45; branch `worktree-agent-a5c9f7577a224b708`, rebased on 0.2.0)
- Added: `tools/gguf_to_pq.py` — lossless llama.cpp GGUF (arch `qwen35`, F32 / Q4_0 / Q8_0) →
  `.pq`: Q4_0 = Q4_G32 and Q8_0 = Q8_G32 bit for bit, llama.cpp's tiled V-head order undone,
  `--verify` checks every scale, code and F32 value (`engine/FORMAT.md`, "Importing a llama.cpp
  GGUF"). First use: a Q4_0 Qwen3.5-4B fine-tune (3.37 GB), 427 tensors / 4.84 B weights verified.
- Engine: untied output head (`tied_embeddings` 0 + `lm_head.weight`), `linear_attn.A` as an
  alternative to `A_log`, Qwen3.5-4B dimensions (32 layers, 32 V heads over 16 K heads) — loads
  and runs unchanged otherwise. Existing models take the same code paths (bit-identical).
- API: `pinfer_eval_select` / `Model.eval_select` — the last token's logits for chosen vocabulary
  ids only, bit-identical to `pinfer_eval`, reading only the head tiles that hold them.
- Added: `tests/test_gguf.py` — synthetic GGUFs written the way llama.cpp writes them (12 V heads
  over 4 K heads): `--verify`, llama.cpp's own reorder of the converted weights gives the GGUF back,
  and an F32 model converts to exactly the tensors of a reference `.pq` built from the HF layout and
  generates the same tokens on the host engine; a reversed permutation fails it. It found that F32
  `out_proj` matrices (column permutation) were refused; they convert now.
- Checked on the rebased tree (host): golden / spec / state / dims PASS, pytest 130,
  malformed 43/43 under ASan/UBSan, test_gguf PASS. Pi (`pinfer_run_cerno2`, md5 `537fb9a3`):
  spec_exact 58/58 with all md5s equal to `spec-exact-merge6`, ctx_check 5/5 identical to
  `pinfer_run_fast` (d48ffda1). Model-specific work: `github.com/Optimizer077/cerno-pi`.

### Published (lead, 13:30)
- The repository is public at `github.com/Optimizer077/piquant` under **Apache-2.0** (`LICENSE`).
  The history was rewritten before publishing: the Wi-Fi network name and the router and Pi MAC
  addresses are placeholders (`<wifi-ssid>`, `<router-mac>`, `<pi-mac>`), and commits use the
  GitHub no-reply address. The previous private repository is `piquant-archive`.
- Issues #1–#11 recreated in the public repository; #8 (license) closed.
### Prompt-lookup drafts and adaptive k (issues #6, #2; drafting agent, 10:55–13:30; branch `worktree-agent-ab157f97c4ad73f36`; notebook `engine/DRAFT.md`)
- Added: prompt-lookup (n-gram) drafting, `pinfer_spec_params.ngram` (0..8, 0 = off),
  `pinfer_run --ngram N`, Python `generate(..., spec_ngram=N)`. `engine/src/ngram.c` indexes the
  token history (2–4-grams → most recent end position, verified hits); a draft is taken when
  the match extends over ≥ 16 tokens, or over 4–15 tokens and its first token equals the MTP
  head's free first draft (then ≤ 3 tokens). Its MTP rows are deferred like plain steps, so
  n-gram rounds cost no MTP pass. k = 0 with ngram > 0 works without an MTP head.
  `pinfer_spec_stats` gained `ngram_rounds` / `ngram_drafted` / `ngram_accepted` (ABI 0.2.0).
- Changed: the adaptive policy (`spec_adapt.c`, `pinfer_spec_params.adaptive`) chains MTP
  drafts while the newest draft's top-1 probability is ≥ 0.6 (up to k) and drops the
  calibration steps; a plain-mode guard compares measured rates (2-round probes). Output never
  changes (exact by construction; gates below).
- Added: `engine/bench/draft_trace.c` (per-position MTP drafts), `engine/bench/draft_sim.py`
  (offline replay of policies; exact against the engine), copy and held-out prompts
  (`bench/common/make_copy_prompts.py`), n-gram configs in `chat_spec.sh` / `spec_exact.sh`,
  `logs/draft/state_cmp.c`.
- Measured (Pi, Q4K, 4 threads, tok/s, before → after): adaptive k ≤ 3 on chat 10.49 → 11.02,
  11.89 → 12.95, 13.14 → 14.26, 11.95 → 12.40, 14.54 → 14.80, 11.29 → 11.17 (story); prompt64
  18.69 → 18.20; ctx2048 10.96 → 11.14. After / best fixed k: 0.984–1.056 on all eight prompts
  (before 0.90–1.04). With `--ngram 7`: code-edit chat answer 22.21 (adaptive, +14.5%) / 22.75
  (k=2, +19% vs k=3 19.12); prompt64 k=3 20.47 (+11%); ctx2048 adaptive 12.96 (+16%); chat
  within −1.0…+3.1%; typo fixing −2.4%. Files `bench/results/ng-*`, `ng1-*`.
- Measured: verify pass per width n = 2..9: 136 / 143 / 152 / 200 / 258 / 266 / 285 / 360 ms
  (`ng-base-p64_20261022_111810.json`); oracle per-round k is 13–18% above the best fixed k,
  of which confidence thresholds recover ~1% (DRAFT.md §2c).
- Gates: host golden / spec (130 speculative runs incl. n-gram and a copy prompt) / state
  PASS, pytest 127 passed, state identity 48/48 (`logs/draft/`); Pi spec_exact 58/58 incl.
  the original 20 rows with the baseline md5s (`spec-exact-ng1_20261022_115556.json`), state
  identity 48/48 (`state-cmp-ng1_20261022_120036.json`).
- Decided (recommendation for the lead, not applied): `pinfer_default_spec_params()` =
  adaptive k ≤ 3 with ngram 7 for `generate(speculative=True)` and `pinfer serve`
  (DRAFT.md §7).
- Pi: `~/pinfer/engine/ng/pinfer_run_ng1` (md5 `7d701ad6`) and `~/pinfer/data/chat_{fix,edit}.ids`
  kept; all other scratch removed; `pinfer_run_fast` untouched (`23a76242`).
### Qwen3.5-2B (issue #7, 2B agent, 10:50–13:20; branch `worktree-agent-a24ea2f8313af7f40`; notebook `engine/MODEL2B.md`)
- Added: Qwen3.5-2B support, proven against an fp32 reference. `Qwen/Qwen3.5-2B` @ `15852e8c`
  (Apache-2.0) differs from 0.8B only in hidden 2048 and ffn 6144 (same layers, attention,
  DeltaNet, vocab, tied head, 1-layer MTP head, tokenizer); the converter and the engine take
  every size from `config.json` / the `.pq` header, so no format change. Default Q4K file 1.09 GB.
- Added: `ref/mtp_reference.py --name`, `tests/test_golden.py --golden`, `tests/test_spec.py
  --golden` (defaults unchanged); golden sets `ref/golden/qwen35-2b{,-mtp}` (gitignored).
- Added: `tests/test_dims.py` (synthetic bf16 models with other dimension relations, plain and
  `--spec 1..3` under the sanitizers, speculative == plain); `bench/model2b/` (`speed.sh`,
  `spec_exact_2b.sh`, `llama_rss.sh` = llama-bench + peak RSS, `ppl_pc_c8.sh` for any model).
- Fixed (engine): the MTP head's fc input rows (2H floats) are built in `ctx->ta`, which was
  sized without 2H: a header with 2H above every projection width gave an ASan heap-buffer-overflow
  in `pf_mtp_forward` (`tests/test_dims.py` `mtp-wide`). `pinfer.c` now sizes the rows
  `max(…, 2H)`; no size change for 0.8B / 2B, outputs unchanged.
- Changed (converter): `tools/convert.py` quantizes in blocks of 16,384 rows: byte-identical files
  (md5 checked for 2B Q4K, 0.8B Q4K, 0.8B mix), peak RSS of the 2B Q4K conversion 17.2 → 8.4 GB.
- Measured (correctness): host golden 2B PASS (worst per-layer rel. error 1.18e-6, logits
  1.21e-6, top-1 10/10, greedy 32/32); MTP head ≤ 3.9e-6, drafts = reference; speculative ==
  plain for 9 prompts × 8 configs (host) and 10/10 on the Pi. 0.8B gates on this branch: golden /
  spec / state PASS, pytest 126/126, malformed 43/43, Pi spec_exact 20/20, ctx_check 5/5 identical
  to `pinfer_run_fast`.
- Measured (Pi 4, 2B Q4K, 4 threads): plain **3.41** tok/s (prompt64) / 3.42 (chat27),
  speculative k=2 **4.35 / 4.82**, k=3 4.16 / 4.79, prefill 14.6, peak RSS 1040 MiB (1080 spec),
  perplexity (8 × 512) **13.38** (0.8B 18.48). llama.cpp v0.7.0 Q4_K_M 2B: 2.49 tg64, 5.99 pp64,
  1314 MiB, perplexity 13.32 (PC). A Q8 MTP head does not raise acceptance (k=2 4.33 / 4.94).
  Files: `bench/results/{speed-2b-q4k-p64_20261022_114956, speed-2b-q4k-chat27_20261022_122132,
  ppl-c8-2b-q4k_20261022_122815, final-2b_20261022_131607, spec-exact-2b-q4k_20261022_114018,
  gates08-2b1_20261022_111351}.json`.
- Decided: 2B is an opt-in quality model (fits with ~2.4 GB to spare, 28% lower perplexity,
  ~4.8 tok/s chat); 0.8B stays the default. Next levers: more tokens per pass (verify of 3 rows
  costs 1.08× a step), fewer bytes (MLP 48% of a step), prefill groups for wide matrices.
- Pi: added `models/qwen35-2b-q4k-headq4k.pq`, `models/Qwen3.5-2B-Q4_K_M.gguf`,
  `engine/pinfer_{run,ppl}_2b1` (2.3 GB, rows in `docs/pi-state.md`); pushes of the GB files ran
  inside held lock windows; scratch (`tmp/m2b`, the Q8-MTP file) removed. Nothing shared replaced.
### Batched matmul and prefill (kernels agent, 10:55–14:30; branch `worktree-agent-a6d5914c435068003`; notebook `engine/bench/KERNELS.md` Part 3)
- Added (kernels, `kern_quant.c`): quantized register passes of up to 7 vectors, the batch split
  into ceil(n / 7) passes of equal size. n = 5 (k = 4 verification) is one pass instead of 4 + 1,
  which decoded every weight tile twice; n = 6 / 7 are one pass instead of 3 + 3 / 4 + 3.
- Added (`matvec_fast.c`): batches of more than 16 vectors (prefill) run in balanced groups of at
  most 7 vectors for the quantized dtypes (`kern_group_vecs`): one register pass per tile with
  its activations in L1, instead of groups of 16 (4 passes over 16–57 KB). Float dtypes keep
  groups of 16.
- Added: Q4K superblock scale decode in fewer NEON ops (6-bit unpack 14 → 5 ops; widening by
  zipping with the float exponent; FSUB + FMUL instead of ORR + MOV + FMA), exact.
- Added (kbench): `yhash` (FNV-1a of the batched output) in every RESULT, for bit-identity
  checks across kernel builds; `--gvecs`. `logs/verify/mm_t_scales.c`: scale decode vs scalar,
  12.8 M values, 0 mismatches.
- Measured (Q4K-headq4k, 4 threads, prompt64, 3 interleaved rounds vs `pinfer_run_fast`,
  medians; `bench/results/mm2-e2e-decode_20261022_140420.json`, `mm2-e2e-prefill_20261022_140719.json`):
  plain 7.875 → 7.877 tok/s; k=2 16.21 → 16.18; k=3 18.50 → 18.15 (n = 4 matmuls +1 %);
  **k=4 16.27 → 18.64** (verify 206.5 → 176.4 ms, now the best fixed k on prompt64);
  chat27 k=2 10.91 → 10.98; **prefill prompt64 36.2 → 43.1, ctx512 36.4 → 43.2 tok/s (+19 %)**.
  kbench (cold, 4 threads, `mm2-kbench-final_20261022_141346.json`): n = 5 −14…−28 %, n = 7
  −14…−26 %, n = 64 −14…−20 %, n = 1–4 ±2 %.
- Measured, not kept: duplicated activation layout with row-pair zipped weights (fewer
  reductions), magic-number int→float, decode-once "wide" passes, software-pipelined float
  update, 8-vector passes (spill), non-temporal LDNP weight loads (lead's suggestion: no change
  in the recurrence / attention phases, k=3 −1.8 %, `mm2-ldnp_20261022_135634.json`).
- Not met: the k=3 verify pass (n = 4) stays at ~1.2× a plain step (target 1.12×); n = 4 is
  compute-bound at ~29 cycles per block-vector and the instruction-cutting variants were slower.
- Gates: kbench `--check` 747 configs, memcmp + yhash identical to the 792676c kernels; Pi
  spec_exact 20/20 (baseline md5s), ctx_check 5/5 identical (`mm2-gates_20261022_134850.json`,
  `mm2-spec-exact_20261022_140927.json`); host golden / spec / state PASS, pytest 126 passed
  (`logs/verify/mm_host_gates.txt`).
- Pi: experiment binaries and scripts under `~/pinfer/kbench/` and `engine/pinfer_run_mm2nt`
  removed at the end; kept `engine/pinfer_run_mm2` (md5 `bb39468b`). Nothing shared replaced.

### OpenAI-compatible server (server agent, 08:55–10:05; branch `worktree-agent-a344482719d7c5ce0`)
- Added: `pinfer serve` (`python/pinfer/server/`, standard library only): `GET /health`,
  `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`; SSE streaming with
  `stream_options.include_usage`; `usage.prompt_tokens_details.cached_tokens` and a `pinfer`
  extension (cached tokens and source, prefill / decode seconds, tok/s, queue time,
  speculative stats); OpenAI error JSON (400 incl. `context_length_exceeded`, 401, 404, 500);
  `chat_template_kwargs.enable_thinking` → `reasoning_content`; optional `--api-key`.
  One generation at a time (FIFO); a client that disconnects stops its generation.
- Added: prefix reuse across requests — earlier replies are put back into the prompt as the
  token ids that were generated (`--history generated`, default), and state snapshots
  (`save_state`, LRU `--state-cache-mb 256`) keep replaced conversations and long system
  prompts. Greedy requests use speculative decoding (k=2), sampled ones plain decoding.
- Fixed (Python): closing a speculative stream that continued the context left
  `Model.context_tokens` behind the engine (4 vs 11 tokens in the regression test), so the next
  reuse would continue a wrong context; `TextStream.close()` / stop strings now end the C call
  at once.
- Added: `bench/serve/serve_client.py`, `bench/serve/serve_bench.sh`; `tests/python/test_server.py`
  (31 tests); `docs/library.md` §10 "Server". CLI `-c/--ctx` gains the alias `--max-ctx`.
- Measured (Pi 4, Q4K, greedy → speculative k=2, client on the Pi;
  `bench/results/serve_20261022_095259.json`, valid): single chat request TTFT 1.33 s, 9.0 tok/s.
  4-turn conversation: turns 2–4 reuse 86/111, 147/169, 232/256 prompt tokens, TTFT 0.96 / 0.89 /
  0.95 s; with the template's history rendering nothing is reused (TTFT 3.21 / 4.82 / 7.08 s,
  like `--no-reuse`). Interleaved conversations: second turns from snapshots in 1.03 / 1.01 s vs
  3.56 / 3.50 s without. Pi pytest with the NEON library: 124 passed, 1 skipped, 1 failed
  (`test_converter_is_available`: no `tools/convert.py` + `quant.py` in the scratch layout;
  passes once copied) — `logs/runs/20261022_092846_run.log`, `..._100244_run.log`. From the PC
  over the cable: openai SDK and curl (SSE) work, stream TTFT 0.83 s
  (`logs/runs/20261022_095338_run.log`).
- Decided: `pinfer serve` listens on 127.0.0.1 by default (`--host 0.0.0.0` for the network;
  no TLS, optional `--api-key`).
- Pi: scratch `~/pinfer/tmp/serve/` (package source, NEON `libpinfer.so` md5 `976b73eb…`,
  scripts), removed at the end; nothing shared replaced.

### Hardening (issue #11, hardening agent, 07:00–; notebook `engine/HARDENING.md`)
- Fixed (loader): every directory entry's `nbytes` must equal the size its dtype and shape
  imply, quantized dtypes included (rows % 4, cols % 256); unknown dtypes refused; header
  dimensions bounded (`PQ_MAX_*`), element counts and products cannot wrap; `eos_id`, finite
  `rms_eps` / `rope_theta`, 8-byte-aligned directory; `lin_v_dim % 32` (`kern_lin.c` chunks).
  `mtp.c` uses `model.c`'s tensor views. Before, on the Pi: a short quantized tensor loaded and
  the matmul read past it; `lin_v_dim 80` crashed (SIGSEGV).
- Fixed (thread pool): `pf_pool_create` no longer pins the calling thread to CPU 0 (later
  threads of the application inherited that single core); workers are pinned to the 2nd..n-th
  CPU of the caller's mask, at most that many threads.
- Fixed (state): blobs carry `pf_model_fingerprint` (header, directory, sampled weights), state
  format 2; a state of another model file with the same dimensions is refused. Python
  `pinfer.state` names the cause (version, kv_dtype, other file).
- Fixed (`pinfer_ppl`): token ids validated before use (out-of-bounds logit read), `--chunks 0`
  refused (printed NaN), odd-sized token files refused.
- Fixed: `PF_FWD_STATS` / `PF_KERN_STATS` flags set atomically, tables under a mutex.
- Changed: version 0.2.0 (`pinfer.h`, SONAME `libpinfer.so.0.2`, Python package, wheel names).
- Added: `make -C engine host SAN=address,undefined`; `tests/test_malformed.py` (43 cases;
  baseline 18/43, now 43/43 under ASan+UBSan and on the Pi); `tests/test_long_context.py` +
  `ref/long_context_reference.py`; pytest `test_affinity.py`, `test_version.py`, 5 state cases.
- Measured: long context vs fp32 transformers to 8192 tokens: worst rel. error 2.0e-6, top-1
  18/18 (no RoPE or position bug). Speed unchanged: prompt64 plain / `--spec 3` 7.888 / 16.713 tok/s vs
  `pinfer_run_fast` 7.891 / 16.717 (3 interleaved rounds, `speed-ab-hard2_20261022_081251.json`);
  output bit-identical (spec_exact 20/20, dumps 5/5 identical).
- Pi: experimental binaries `engine/pinfer_run_hard2`, `pinfer_ppl_hard2` (scratch `tmp/hard/`
  removed at the end); `engine/pinfer_run_fast` untouched.
### Changed (figures agent, 08:55) — README figures sized for desktop and phones
- Every figure is now rendered in 2 layouts × 2 themes (52 PNGs). Desktop `NAME.png` / `NAME-dark.png`
  are exactly 1676 px wide (GitHub's 838 CSS px README column at 2×); phone `NAME-mobile.png` /
  `NAME-mobile-dark.png` are 1080 px (a 360 CSS px column at 3×). Before, the 1120–1600 px PNGs were
  scaled by 0.52–0.75 on desktop (text 10–13 CSS px) and to ~0.22–0.32 on phones (text 4–6 px).
- Text at display size: desktop 13 CSS px (ticks, labels, legends, values), 14 subtitles, 18 titles;
  phone 13 / 13 / 16. Each layout renders at DPI = 72 × scale, so 1 pt = 1 CSS px.
- Phone layouts: small multiples stacked (`context_length`, `speculative_k`, `quantization`,
  `mtp_acceptance`); category names above the bars (`decode_speed`, `prefill_speed`, `os_experiment`,
  `step_breakdown`); legends under the title; titles and subtitles wrapped by measured width; the
  speculative schematic stacked top to bottom; shortened tick labels (`progress`: "exact kernels",
  "vectorized ops"; `context_length`: 2K / 4K / 8K with "K = 1024" in the axis label).
- `tools/figstyle.py`: `Layout` dataclass next to `Theme`; `header()` (title, subtitle, legend),
  `hbar_rows()`, `label_inside()`; `save()` writes a 256-colour palette PNG (no dither; dominant colours
  unchanged) and fails on a wrong width or if the final layout moves axes after px-sized marks.
  `docs/figures/`: 1,711,911 B in 26 files → 2,284,489 B in 52 files.
- `README.md` (13 blocks) and `docs/RESULTS.md` (12 blocks): `<picture>` gains the two
  `(max-width: 600px)` sources (light, dark); alt texts unchanged.
- Desktop readability fixes: integer ticks; the dashed reference rule in `muted` (it was nearly
  invisible in dark mode); value labels no longer cut the series lines (`progress`); legends that
  covered data moved under the title (`step_breakdown`, `mtp_acceptance`); the DRAM ceiling in a
  legend key (`kernel_bandwidth`).
- No data value changed: every numeric constant in both scripts compared equal to the previous commit.
### Changed (verify pass, issues #5 and #11; details in `engine/VERIFY.md`)
- Speculative rounds read and write the Gated DeltaNet states once instead of twice. Changes in
  `kern_lin.c`, `forward_lin.c`, `spec.c` and the `ops.h` `pf_lin_rule` contract; the scalar twin in
  `matvec_ref.c`:
  - The first verify token (`cur`, always committed) now updates the states in place.
  - The drafts run on a 16 KB per-thread stack copy of each state chunk.
  - `pf_lin_commit` defers the accepted rank-1 updates to the layer's next pass, which folds them
    into its first sweep. The log is double-buffered.
  - `pf_lin_flush` runs at the end of `pf_spec_decode`.

  Output is bit-identical.
- Fixed (#11 High): the shared `static` scratch state in `kern_lin.c`, and the same `static
  lin_scratch` in `matvec_ref.c` (the x86 wheel), are removed. Before the fix, two contexts
  decoding speculatively on two threads produced different tokens (Pi: 7 of 8 runs; host: 4 of 6).
  After it, 0 of 8 and 0 of 6 differ (`logs/runs/20261022_083027_run.log`,
  `logs/verify/thread_cmp_host.txt`).
- `engine/bench/lbench.c` checks the new contract (logged pass, replay, deferred commit in logged and
  in-place passes) and times the verify kinds.

### Measured (verify pass)
- prompt64, k=3, draft vocab 32768: **16.64 → 18.42 tok/s** (+10.7%). The verify pass went from
  159.5 to 152.9 ms and the commit from 14.3 to 1.9 ms. Verify / plain is 1.26 → 1.20, and
  (verify + commit) / plain is 1.37 → 1.22. Plain decode is unchanged at 7.87 tok/s
  (`ver-before-p64_20261022_070046.json`, `ver3-ab-p64_20261022_080513.json`).
- ctx2048 with `--adaptive`: 9.15 → 10.71 tok/s (+17%). Fixed k=3: 8.90 → 9.55
  (`ver-before-ctx2048_20261022_070717.json`, `ver3-ctx2048_20261022_082417.json`).
- Gates:
  - host golden / spec / state PASS, pytest 85/85;
  - state after speculative decoding byte-identical to plain decoding (21/21);
  - Pi lbench exact, spec_exact 20/20 (md5s = baseline), ctx_check 5/5
    (`ver3-exact_20261022_075930.json`).
- Tried and dropped, with numbers in VERIFY.md §6: fused token sweeps (slower at n = 4), software
  prefetch of the next chunk (slower), deferring the conv-state commit (no gain).
### Attention round 3 — issue #1 (attention agent, 06:50–; details `engine/CONTEXT.md` §9)
- **Changed** (`engine/src/forward_attn.c`, `forward_kv.c`, `forward_ops.c`; bit-identical for
  fp32 / fp16 / int8, batched == token by token): 4-head dots with named accumulators (the fp16
  dot stored its 16 accumulators to the stack every iteration) and FADDP final sums (15 instead
  of ~124 scalar ops per row); P·V split by (KV group × column range) so every V element is read
  once and feeds all 4 heads; softmax as its own pass with a 4-lane max and two exp chains;
  prefetch of the next P·V block (the column-blocked order defeats the hardware prefetcher);
  the dots read the 4 queries interleaved (single tokens and batches); single-token scores with
  the group's two threads in alternate 16-row blocks. (A bug of the batch interleave for a single
  int8 token with > 4 heads per group was caught by new hash shapes before use and fixed.)
- **Fixed:** int8 KV store, scalar tail: `(int)rintf(NaN)` was undefined behaviour (UBSan);
  now clamped in float, NaN → 0 as on NEON, finite values unchanged.
- **Added:** `engine/bench/abench.c` (attention-core microbench: cold layers, per-phase and
  per-thread times, stream ceilings, output hash for bit identity; `make -C engine/bench
  abench` / `abench-host`); `f16_check.sh` takes the KV dtype (int8 check); `bench/common/next2048.ids`.
- **Measured** (Pi 4, Q4K, 4 threads; decode-only from saved states, 3 rounds; `pinfer_run_fast`
  → `pinfer_run_att4`): decode at 4096 fp16 6.59 → **6.86 tok/s = 87.1% of 64-token decode**
  (issue #1 target met with the fp16 cache), fp32 6.29 → 6.43 (81.6%); 2048 fp16 7.21 → 7.36,
  fp32 6.89 → 7.10; 8192 fp16 5.67 → 6.07, fp32 5.30 → 5.34; 64 tokens unchanged (7.86 → 7.87).
  Prefill (one run each, saved states md5-identical): 4096 27.5 → 28.8–29.9, 8192 22.7 → 24.3
  (fp32) / 22.0 → 24.3 (fp16); a 64-token chunk after 4096 tokens 22.2 → 23.7 / 22.3 → 24.3.
  abench attention per layer: 4096 fp16 −29%, fp32 −13%; 8-token blocks −20%.
- **Not kept** (measured): P·V loop variants (4 rows per iteration, loads one row ahead, two
  register sets, 16-row blocks), prefetch two blocks ahead or into L2, 64-row score blocks,
  page pre-touching; a GCC `no-schedule-insns` variant (−7% kernel cycles, ~0.4% of decode).
- **Gates** (final `pinfer_run_att4`): host golden / spec / state PASS, 17 host dumps
  md5-identical to main; Pi `ctx_check` 5/5, fp16 check 3/3, int8 check 3/3, `spec_exact` 20/20
  (`att3-gates-att4_20261022_091651.json`, `att3-spec-exact-att4_20261022_092716.json`).
- **Pi:** `engine/pinfer_run_att4` added (scratch states / binaries in `tmp/att3` and
  `pinfer_run_att1`–`att3` removed); `data/next2048.ids` pushed (= `bench/common/next2048.ids`).

### Merged (lead, 06:50)
- Context-window work merged to `main` after the full gate (host golden / spec 81/81 / state /
  pytest 85/85; Pi build md5-identical to the gated `pinfer_run_ctx7`). The Pi's shared
  `engine/pinfer_run_fast` and `pinfer_ppl_fast` are now this build (previous: `*.pre-ctx`).
- Defaults decided: fp32 KV cache stays the default (exact); adaptive speculation stays opt-in.
- Fixed: `tests/test_spec.py` defaulted to a stale binary (`build/host/engine-spec/`).
- Docs: README long-context section and headline row, before/after `context_length` figure,
  RESULTS.md §6 final tables, CONTRIBUTING source-tree pytest command.

### Added (context window — details `engine/CONTEXT.md`)
- Attention over the KV cache rewritten (`engine/src/forward_attn.c`, `forward_kv.c`): K/V rows
  streamed once per block of 8 tokens and shared by the 4 query heads of a KV group; threads
  over (KV group × row half) for the scores and (KV group × head half) for softmax + P·V.
  Bit-identical per element (host md5 and Pi NEON twin checks), spec_exact 20/20.
- `pinfer_params.kv_dtype` (0 fp32, 1 fp16, 2 int8 + per-row scale), `pinfer_run --kv-dtype`,
  `pinfer_ppl --kv-dtype`, Python `Model.load(kv_dtype=)`, CLI `--kv-dtype`.
- State snapshot API: `pinfer_state_size / save / load`, continuation
  `pinfer_generate_greedy_cont`; `pinfer_run --save-state / --load-state`; Python
  `Model.save_state() / load_state()`, `context_tokens`, speculative `generate` continues from
  the context; `pinfer chat` keeps the state between turns (`--reprefill` for the old way).
- Adaptive speculation (`pinfer_spec_params.adaptive`, `--adaptive`, `spec_adaptive=`): the
  engine measures step / verify / draft costs and first-draft acceptance and picks k in 0..k
  per round, probing again in plain mode; output unchanged.
- Attention, second round: register-blocked P·V on an L1 block of 8 rows, two heads per pass
  (`pf_attn_pv_blk`), and single-token fp16 dots that widen K in registers (`pf_dot4_f16`);
  both bit-identical. Tried and dropped: software prefetch of the next rows (−1…−11%).
- Adaptive speculation estimator: decayed hits / trials over all drafts with a 2/3 prior
  (the first version collapsed to plain decoding after one early miss).
- Tests: `tests/test_state.py` (C round trip, 9 cases), `tests/python/test_state.py` (4),
  `tests/test_spec.py` configs `k3adapt` and `k3b2` (verify pass with a prefill batch < k+1).
- `bench/common/ctx{64,8192}.ids`, `engine/bench/ctx_check.sh`.

### Fixed
- Speculative verify pass with `-b` smaller than k+1 corrupted the rollback log (pre-existing);
  `draft_vocab` 1–3 produced an empty draft head; the speculative prefill could consume part of
  an oversized prompt before returning `PINFER_E_CTX_FULL`.

### Measured (Pi 4, Q4K, 4 threads, 32 generated; before → after)
- Plain decode tok/s: ctx 512 6.20 → 7.69, 2048 4.12 → 7.13, 4096 → 6.48, 8192 → 5.42
  (64: 7.81 → 7.92). Prefill tok/s: 512 21.6 → 36.8, 2048 11.6 → 32.3, 4096 28.0, 8192 22.0.
  Spec k=3 tok/s: 512 9.04 → 15.24, 2048 2.39 → 7.47, 4096 8.47, 8192 7.17.
- llama.cpp Q4_K_M decode at depth 2048 / 4096: 4.32 / 3.43 tok/s (pinfer 7.13 / 6.48).
- KV dtypes: perplexity within noise (8×2048: fp32 13.985, fp16 13.983, int8 13.989); fp16
  teacher-forced top-1 100% / KL 3.5e-8. fp16 decode at 8192 5.72 vs fp32 5.33 tok/s, RSS −97 MB.
- Adaptive k=3 (64 generated): 9.57 tok/s at 2048 (fixed 9.13, plain 7.09), 7.53 at 4096
  (fixed 7.38, plain 6.11), 16.42–16.82 on prompt64 (fixed 16.72–16.79).
- State: 2048-token state 70.5 MB, save 94 ms, load 90 ms (vs 63 s prefill); chat turn 2
  0.68 s instead of 2.10 s.

### Added
- Private GitHub repository `Optimizer077/piquant`; result figures in `docs/figures/` (`tools/make_figures.py`).
- Engine: NEON per-token ops (`forward_ops.c`, bit-exact scalar twins), threaded attention/MLP,
  `pf_pool_run`, `--bench` warm-up; opt-in `PINFER_HUGEPAGES`; Phase 6 context-window plan.
- `bench/BASELINES.md`: MNN / ORT GenAI / LiteRT-LM / PyTorch attempts and results.
- `tools/convert.py --quant {bf16,q8,q4,q4k} --head ... --override REGEX=DTYPE` (quantized
  `.pq` files via `tools/quant.py`); `engine/tools/pinfer_ppl` (llama-perplexity-compatible
  perplexity, cross-checked vs llama.cpp); `bench/quality/make_tokens.py`, `ppl_pc_c8.sh`.
- Pi wheel with the fast kernels (`manylinux_2_34_aarch64`), installed into `~/pinfer/venvs/pinfer`.

### Measured (Pi 4, Qwen3.5-0.8B, 64+64 tokens)
- **Vectorized engine (02:40):** Q4K default **7.88 plain / 16.49 speculative (k=3) / 37.4 prefill tok/s**;
  verify ratio 1.28; exactness 20/20; wheel pytest 80/80 on the Pi. Context 4096: 5.56 / 5.69 / 18.1.
- **Final kernels (01:10):** Q4K default 7.26 tok/s plain, **13.35 tok/s speculative (k=3)**, prefill
  26.1 tok/s, perplexity 18.48; speculative == greedy 20/20 on the Pi.
- pinfer Q4+Q4 head: 6.88 tok/s plain, **12.56 tok/s with exact MTP speculation** (k=3,
  draft vocab 32768) — vs MNN int4 6.67, llama.cpp Q4_0 5.59.
- Perplexity (wikitext-2, 8 chunks): pinfer mix-Q4K 18.50; llama.cpp bf16 17.93, Q8_0 17.97,
  Q4_K_M 18.59, Q4_0 21.25.

### Fixed
- Speculative decoding diverged from plain greedy at near-ties with the NEON kernels (batched
  verify pass rounded the Q4/Q4K offset term differently): kernels made bit-identical for every
  batch size, checked by kbench and `engine/bench/spec_exact.sh`.
- Harness: locked Pi jobs are killed after 20 min (a 38-min ORT run had stalled four agents).

## 2026-10-21

### Added (MTP self-speculative decoding — details `engine/SPEC.md`)
- `engine/src/mtp.{c,h}` (Qwen3.5 MTP draft head; semantics from vLLM `qwen3_5_mtp.py` +
  llama.cpp `qwen35.cpp`), `engine/src/spec.{c,h}` (draft k with the MTP chain, verify k+1 in
  ONE batched pass, exact rollback of linear-attention state by replaying logged rank-1
  updates — no state snapshots). `forward_lin.c` loops head-outer (state stays in L2),
  `forward.c` exports its attention/MLP blocks and `pf_forward_h` (final-normed hiddens).
- `pinfer.h` (additive): `pinfer_spec_params`, `pinfer_spec_stats`, `pinfer_default_spec_params`
  (k=2, draft_vocab=32768), `pinfer_has_mtp`, `pinfer_generate_greedy`,
  `pinfer_generate_greedy_ex` + `pinfer_gen_opts` (stop ids, per-token callback, tokens kept on
  context overflow).
- `pinfer_run --spec k [--draft-vocab N] [--dump-mtp f]`; RESULT json gains rounds / drafted /
  accepted / tokens_per_pass / per-pass timings.
- `ref/mtp_reference.py` + `ref/golden/qwen35-0.8b-mtp/`, `tests/test_spec.py` (PASS: MTP logits
  ≤ 4e-6 rel vs reference; spec output == plain greedy for 9 prompts × k=0..4 + draft vocab).

### Measured (speculative decoding, bf16, 64 tokens, Pi)
- Reference MTP acceptance 87.3% first draft (651 positions); controls 54.5% / 0.3%.
- NEON kernels, 4 threads, k=2 + draft vocab 32768: 4.13 / 3.27 / 4.50 tok/s vs 2.31 / 2.33 /
  2.33 plain (1.79× / 1.41× / 1.93×, Dickens / history / code), identical tokens.
  Scalar kernels: speculation 0.55–0.74× (compute-bound, expected).

### Added (release / packaging, pinfer 0.1.0 — not published)
- `engine/Makefile`: `lib` / `host-lib` / `pi-lib` (`libpinfer.a` + `libpinfer.so`, `-fPIC`,
  only `pinfer_*` exported via `engine/pkg/pinfer.map`; static archive localized with
  `ld -r` + `objcopy`), `install PREFIX= [TARGET=host|pi] [DESTDIR=]` (+ `pinfer.pc`),
  `examples`, `check-lib` (`engine/pkg/check_lib.sh`: exports, NEEDED, glibc ≤ 2.41 for the Pi),
  `wheel-host` / `wheel-pi` → `build/dist/`. Library objects use `-fno-math-errno`
  (otherwise `sqrtf@GLIBC_2.43` from WSL's glibc makes the .so unloadable on the Pi).
- `python/`: package `pinfer` (ctypes binding, tokenizer, sampling, Qwen chat template,
  streaming, speculative via `pinfer_generate_greedy`), CLI `pinfer`, `pyproject.toml` +
  `hatch_build.py` (bundles the .so; manylinux tag derived from its glibc symbol versions).
- `tests/python/` (80 tests), `examples/c/simple.c`, `examples/python/generate.py`,
  `docs/library.md`, `docs/release-0.1.0.md` (draft).

### Measured (release)
- pytest 80/80 on host (`logs/release/host_pytest_final.log`) and on the Pi
  (`logs/release/pi_pytest_final.log`); greedy ids via Python == golden 32/32 (plain and
  speculative).

### Pi (release)
- User-level venv `~/pinfer/venvs/pinfer` (offline install from `~/pinfer/wheels/pinfer/`),
  tokenizer files, golden meta.json, tests, C example — listed in `docs/pi-state.md`. No apt.

### Added
- `docs/design.md`: design for pinfer (Qwen engine + split-core Linux, level 2).
- `harness/pi.py`: `check`, `push`, `run`, `bench`, `snapshot`; `harness/runlog.py`: automatic
  audit log (`logs/harness.jsonl`), full `run` output in `logs/runs/`.
- `bench/memband/memband.c` + `Makefile`: DRAM bandwidth benchmark (static aarch64 build).
- `os/stock/render_firstboot.py`, `os/stock/flash_disk.ps1`, `os/stock/compare_disk.ps1`:
  headless first-boot config, verified raw flashing, block comparison.
- `docs/pi-state.md`, `docs/journal/`, this changelog, project `CLAUDE.md` logging rules.

### Pi
- Flashed Raspberry Pi OS Lite 64-bit 2026-10-06 (trixie) to the Crucial X6 SSD (old
  install erased with user approval). First boot 21:03:53 KST.
- Change #1: Wi-Fi power save disabled.
- Change #2: eth0 static 10.42.0.2/24 for a direct PC↔Pi cable; harness now uses it
  (Wi-Fi drops every few minutes with reason=7 — kept only for the Pi's internet).
- PC change P1: Windows Ethernet static 10.42.0.1/24 (`os/pc/eth_link.ps1`, has `-Undo`).

- Change #3: bootloader EEPROM 2026-05-17 → 2026-09-23.
- Placed llama.cpp v0.7.0 binaries and Qwen3.5-0.8B GGUFs under `~/pinfer`.

### Decided
- Target model: **Qwen3.5-0.8B** (hybrid linear/full attention), user choice. Qwen3-0.6B kept
  as reference. Compare against llama.cpp and other standard runtimes.

### Added (baseline tooling)
- `bench/llamacpp/build.sh` (pinned llama.cpp, static Cortex-A72 build + host quantizer),
  `make_ggufs.sh` (HF → GGUF bf16 → Q8_0/Q4_0/Q4_K_M), `llama_bench.sh` (thread sweep).
- `harness/pi.py push --dest <subdir>`.

### Measured
- memband: 4.20 / 4.08 / 4.07 GB/s with 1 / 2 / 4 threads (`bench/results/memband_20261021_210615.json`).

### Fixed
- Harness: pushed files lacked the exec bit; unknown throttle state counted as valid;
  stderr not saved; `run` with `;` could run outside the work dir.
- memband: threads not pinned to CPUs; slice alignment for odd thread counts.
