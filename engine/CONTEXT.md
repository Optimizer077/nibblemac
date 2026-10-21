# Context window — lab notebook (phase 6)

Owner: context-window work. Files: `engine/src/{forward_attn.c,forward_kv.c,state.c,spec_adapt.c}`
(new), `engine/src/{forward.c,forward.h,forward_ops.c,forward_stats.c,pinfer.c,spec.c,spec.h}`,
`engine/include/pinfer.h` (additive), `engine/tools/{pinfer_run.c,pinfer_ppl.c}`,
`python/pinfer/{_lib.py,model.py,speculative.py,cli.py}`, `tests/test_state.py`,
`tests/python/test_state.py`, `engine/bench/ctx_check.sh`, `bench/common/ctx{64,8192}.ids`.
Pi binaries (all in `~/pinfer/engine/`): `pinfer_run_ctx` (attention rewrite), `ctx2`/`ctx4`
(+ state, adaptive v1, kv dtypes; `ctx4` = review fixes), `ctx5` (adaptive v2), `ctx6` (blocked
P·V), `pinfer_run_m{0,1,2}p{0,16}` / `pv1` / `pv2` (A/B only), **`pinfer_run_ctx7` + `pinfer_ppl_ctx7`
= final**; `pinfer_ppl_ctx3`/`ctx4` measured the PPL (attention code identical to ctx4 for fp32). Started 2026-10-22 02:30 KST from HEAD `d53c835`.

## Plan / status (final: `pinfer_run_ctx7` = the source in the tree, 2026-10-22 06:25)
| # | step | status |
|---|---|---|
| 1 | Measure decode / prefill / RSS vs context (64 … 8192), plain + spec k=3; llama.cpp at the same depths | done (§1, §3, §3b) |
| 2 | Profile attention; optimise bit-exactly; target decode@4096 within 15% of decode@64 | done (§2, §3c, §3d). Plain fp32 4096/64 = **81%** — the target is below the DRAM floor for an fp32 cache (≈ 84.5%); fp16 cache **84.4%** (6.65 vs 7.88) |
| 3 | KV dtype option (fp32 / fp16 / int8): agreement, PPL, speed, RSS | done (§5): PPL within noise for both; fp32 stays default |
| 4 | State save/restore API + Python + `pinfer chat` keep-state + tests + docs | done (§6) |
| 4b | Adaptive speculation (lead's request): never slower than plain | done (§4); first estimator failed, fixed |
| 5 | Bounded-context mode (sinks + sliding window) | not done (§7: design only) |

### Summary (final binary unless noted; Q4K-headq4k, 4 threads, Pi 4)
| context | plain fp32 tok/s (HEAD → now) | plain fp16 | prefill tok/s (HEAD → now) | spec k=3 fixed | adaptive k≤3 | RSS fp32 / fp16 | llama.cpp Q4_K_M tg |
|---|---|---|---|---|---|---|---|
| 64 (prompt64, 64 gen) | 7.81 → 7.88 | 7.90 | 37.4 → 37.3 | 16.74 → 16.79 | 16.42–16.82 | 433 / 431 MB | 5.48 |
| 512 | 6.20¹ → 7.63–7.69 | 7.67 | 21.6¹ → 36.8–37.3 | 9.04¹ → 15.24 | — | 443 / 436 | 5.23 |
| 2048 | 6.57 → 7.09–7.13 | — | 25.7 → 31.4–32.3 | 2.39¹ → 7.47 (32 gen) / 9.13 (64) | **9.57** (64 gen) | 479 / — | 4.32 |
| 4096 | — → 6.36–6.48 | **6.51–6.65** | — → 27.8–28.0 | 8.47 (32) / 7.38 (64) | **7.53** (64) / 7.93 (32) | 528 / 479 | 3.43 |
| 8192 | — → 5.23–5.33² | **5.72**² | — → 22.0–22.4 | 7.17 | — | 625 / 528 | — |

¹ lead's runs with the pre-core2 binary (`pinfer-ctx{512,2048}-q4k_20261022_02*.json`); HEAD at 2048
from my profile run (`pinfer-ctx2048-profile-q4k_20261022_024112.json`). ² mean of 3 decode-only
runs from a saved state (§3d); single full runs: fp32 4.91–5.42, fp16 5.35–5.60.

## 1. Starting point (HEAD `d53c835`, Q4K-headq4k, 4 threads, 32 generated)
Prompts `bench/common/ctx{64,512,2048,4096,8192}.ids` = the first N tokens of
`bench/quality/wikitext2-qwen35.tokens.bin` (wikitext-2 test, 32768 tokens); `prompt64` = Dickens.

| ctx | plain tok/s | ms/step | prefill tok/s | spec k=3 tok/s | tok/pass | verify ms/pass | RSS MB | result file |
|---|---|---|---|---|---|---|---|---|
| 64 (Dickens) | 7.81 | 128 | 37.4 | 16.74 (64 gen) | 2.86 | 158.8 | 433 | `pinfer-core2-q4k_20261022_020959.json` |
| 512 | 6.20 | 161 | 21.6 | 9.04 | 3.10 | 283 | 443 | `pinfer-ctx512-q4k_20261022_021315.json` (pre-core2 binary) |
| 2048 | 4.12 | 243 | 11.6 | 2.39 | 1.63 | 586 | 479 | `pinfer-ctx2048-q4k_20261022_022009.json` (pre-core2 binary) |
| 2048 (core2 binary, profiled) | 6.57 | 152 | 25.7 | – | – | – | 478 | `pinfer-ctx2048-profile-q4k_20261022_024112.json` |

Profile of the core2 binary at 2048 (`PF_FWD_STATS=1 PF_KERN_STATS=1`, 16 generated): per plain
step attention 6 × 4.42 = **26.5 ms** of 152 (matmul 118); per 64-token prefill chunk and layer
attention **141 ms** = 27 s of the 80 s prefill. Cause: the per-(token, head) loop streamed the
whole K and V of the head's KV group from DRAM for every query head and every token: 8 passes
per token (192 MB at 2048; L2 sharing between the two threads of a group brought the effective
traffic to ~2×), and 64 × 8 passes per prefill chunk (≈ 400 GB over a 2048-token prefill).

## 2. New attention path (`forward_attn.c`, `forward_kv.c`), bit-identical
Per block of up to `PF_ATTN_TB = 8` tokens, two `pf_pool_run` passes over the cache:
1. **scores** — threads over (KV group × half of the rows): each K row is read once and dotted
   with the 4 query heads of its group (`pf_dot4`: one load of the row, pf_dot's 16-lane order per
   head) for every token of the block, in L1 blocks of 8 rows;
2. **softmax + P·V + gate** — threads over (KV group × half of its query heads): per-head
   softmax exactly as before (running max, `pf_softmax_exp`, × 1/sum), then V rows in blocks of 8
   accumulated into every (token, head) of the unit in row order (`pf_attn_pv_acc`).
Per element nothing changed, so the outputs are **bit-identical** to the old loop and still
independent of n and of the thread count. Checks: host dumps of prompt64 (batched and `-b 1`),
ctx512 logits and spec-k3 tokens — all md5 equal to the unmodified binary (bf16 model);
Pi NEON twin check `engine/bench/ctx_check.sh` (new vs `pinfer_run_fast`, Q4K): 5/5 identical
(`bench/results/ctx-check_20261022_025513.json`); `spec_exact.sh` 20/20
(`spec-exact-ctx_20261022_025355.json`); host `tests/test_golden.py` and `tests/test_spec.py` PASS.
Traffic per decode step: K+V of each group once (48 MB at 2048, 96 MB at 4096); per prefill
chunk: once per 8 tokens instead of once per (token, head) — 64× less.
Tried and not needed: software prefetch (the kernels work showed PRFM costs load slots on the
A72; the hardware prefetcher follows the 1 KB-row stream), per-chunk partial P·V sums with a
context split over 4 threads (would change the summation order; the (group × head-half) split
already keeps 4 cores busy and V is read ≤ 2× with L2 sharing).

## 3. Context sweep, new binary (`pinfer_run_ctx`, Q4K-headq4k, 4 threads, 32 generated)
`bench/results/pinfer-ctx{-p64,64,512,2048}-new-q4k_20261022_02*.json`,
`pinfer-ctx4096-new-q4k-{plain,spec3}_20261022_03*.json`, `pinfer-ctx8192-new-q4k-*` (§3b).

| ctx | plain tok/s (before → after) | ms/step | prefill tok/s (before → after) | spec k=3 tok/s (before → after) | tok/pass | verify ms/pass | RSS MB |
|---|---|---|---|---|---|---|---|
| 64 Dickens | 7.81 → **7.92** | 126 | 37.4 → 35.6 | 16.74 (64 gen) → 14.64 (32 gen; §4: 64 gen) | 2.82 | 157 | 432 / 451 |
| 64 wikitext | – → 7.94 | 126 | – → 37.8 | – → 12.26 | 2.38 | 158 | 432 / 451 |
| 512 | 6.20 → **7.69** | 130 | 21.6 → **36.8** | 9.04 → **15.24** | 3.10 | 165 | 443 / 463 |
| 2048 | 6.57 → **7.13** | 140 | 25.7 → **32.3** | 2.39 → **7.47** | 1.82 | 199 | 479 / 506 |
| 4096 | – → **6.48** | 154 | – → **28.0** | – → **8.47** | 2.38 | 231 | 528 / 562 |
| 8192 | §3b | | | | | | |

Decode at 4096 is 82% of decode at 64 with the fp32 cache (target 85%): the remaining cost is
the 96 MB of K+V per step at ~4 GB/s ≈ 24 ms, i.e. DRAM-bound now; fp16 halves it (§5).
Prefill at 4096 is 28 tok/s (vs 37 at 64): the O(n²) attention is now compute-bound (~3 s of
the 146 s).

### 3b. 8192 tokens and llama.cpp at the same depths
8192 (`pinfer-ctx8192-new-q4k-{plain,spec3}_20261022_03*.json`, no warm-up, 32 generated):
plain **5.42 tok/s** (184 ms/step; the K+V stream is 192 MB/step ≈ 46 ms), prefill **22.0
tok/s**, RSS 625 MB; spec k=3 **7.17 tok/s** (2.82 tok/pass, verify 318 ms), RSS 675 MB. The
old loop was not measured at 8192: at its 8 passes per token it would have needed > 20 min
of lock for the prefill alone.

llama.cpp v0.7.0 Q4_K_M, `llama-bench -t 4 -p 64 -n 32 -d <depth> -r 1`
(`llamacpp-ctx-d0-512-2048-q4km_20261022_032910.json`, `llamacpp-ctx-d4096-q4km_20261022_033523.json`;
`-d` = tokens already in the KV cache; its pp64 is 64 prompt tokens *after* that depth, not a
full prefill):

| depth | llama.cpp tg32 tok/s | llama.cpp pp64 tok/s | pinfer plain tok/s | pinfer spec k=3 tok/s | pinfer full prefill tok/s |
|---|---|---|---|---|---|
| 0 / 64 | 5.48 | 14.80 | 7.94 | 12.26–14.64 | 35.6–37.8 |
| 512 | 5.23 | 13.12 | 7.69 | 15.24 | 36.8 |
| 2048 | 4.32 | 10.27 | 7.13 | 7.47 | 32.3 |
| 4096 | 3.43 | 7.75 | 6.48 | 8.47 | 28.0 |
| 8192 | (not run: two 8192 prefills per llama-bench test > the 20-min lock) | | 5.42 | 7.17 | 22.0 |

pinfer keeps 82% of its 64-token decode speed at 4096 and 68% at 8192; llama.cpp keeps 63%
at 4096. Decode is 1.9× llama.cpp at 4096 (2.5× with speculation).

### 3c. Where the long-context time goes now (`pinfer_run_ctx6`, profile at 4096, 16 generated)
`PF_FWD_STATS=1 PF_KERN_STATS=1`, `bench/results/pinfer-profile-kv{0,1}-ctx4096_20261022_05*.json`;
"kv+scores" = K/V store, q norm/RoPE and the scores pass; "softmax+pv" = the second pass.

| per decode step (n = 1), per layer | fp32 KV | fp16 KV |
|---|---|---|
| kv+scores | 3.07 ms | 1.99 ms |
| softmax+P·V | 3.04 ms | 2.29 ms |
| attention per step (6 layers) | 36.7 ms | 25.7 ms |
| K+V bytes per layer / effective rate | 16 MB / 2.6 GB/s | 8 MB / 1.9 GB/s |
| decode tok/s (profiled) | 6.12 | 6.57 |

The 64 → 4096 slowdown (126 → 163 ms per step) is entirely this attention time. **Bandwidth
floor**: an fp32 cache is 96 MB per step at 4096 tokens = 23 ms at ~4.1 GB/s, so the brief's
target "decode@4096 within 15% of decode@64" (≤ 148 ms per step) is *below the DRAM floor* for
fp32 (best possible ≈ 84.5%); with fp16 the floor is 12 ms (≈ 91%). The phases run at ~63% of
DRAM bandwidth (fp32) and are compute-heavy for fp16 (widening + the 4-head dots: ~830 NEON ops
per K row). §3d: A/B of three P·V variants × software prefetch, plus fp16 dots that widen in
registers (no store/reload), decode-only from saved states.

Same-flag results of the final sweep (`pinfer-final-kv{0,1}-ctx{512,4096,8192}_20261022_05*.json`,
32 generated, `--no-warmup` except 512):

| ctx | fp32 decode | fp16 decode | fp32 / fp16 prefill | RSS fp32 / fp16 |
|---|---|---|---|---|
| 512 | 7.63 | 7.67 | 37.3 / 37.1 | 443 / 436 MB |
| 4096 | 6.36 | 6.51 | 27.8 / 27.7 | 528 / 479 MB |
| 8192 | 4.91 (`pinfer_run_ctx`: 5.42) | **5.60** (ctx4: 5.35) | 22.4 / 21.7 | 625 / 528 MB |

### 3d. Decode-only A/B of attention variants (saved states, `engine/bench/pv_ab.sh`)
One prefill per (context, kv dtype) saved with `--save-state`, then every binary restores it and
decodes 32 tokens, 3 rounds alternating (`bench/results/pinfer-ab-att-ctx{4096,8192}-kv{0,1}_20261022_05*/06*.json`;
temporary compile switches, removed afterwards). `m0` = blocked P·V on a memcpy'd / widened
L1 block (ctx6), `m1` = the former row-outer P·V, `m2` = blocked P·V reading the fp32 cache in
place; `p16` = software prefetch 16 rows ahead in both passes; all `m*` have the fp16 decode
dots that widen in registers. Every file: **one md5 of the generated tokens for all 7 binaries**
(the variants are bit-identical, including the fused fp16 dots vs ctx6's widen-then-dot).

| mean decode tok/s (min..max) | 4096 fp32 | 4096 fp16 | 8192 fp32 | 8192 fp16 |
|---|---|---|---|---|
| ctx6 | 6.34 | 6.54 | 5.23 (5.08..5.40) | 5.61 |
| m0p0 (**kept**) | 6.32 | **6.60** | 5.33 | **5.72** |
| m1p0 row-outer | 6.31 | 6.37 | 5.17 | 5.31 |
| m2p0 in place | 6.37 | 6.62 (= m0 code for fp16) | 5.17 | 5.73 |
| m0p16 / m1p16 / m2p16 prefetch | 6.05 / 6.15 / 6.28 | 6.45 / 6.13 / 6.43 | 4.90 / 4.99 / 5.22 | 5.47 / 5.01 / 5.37 |

fp32: all P·V variants within noise (ctx6 and m0p0 run the same fp32 code and differ by 2%,
the noise floor) — so the single-run 4.91 vs 5.42 at 8192 in §3c was run-to-run variance, not
a regression. fp16: blocked P·V +3.5–7% over row-outer, fused dots +1–2%. Prefetch loses
everywhere (1–11%), as it did for the matmuls (KERNELS.md). Final source = m0p0 without the
switches (`pinfer_run_ctx7`).

## 4. Adaptive speculation (`spec_adapt.c`, `pinfer_spec_params.adaptive`)
Model per round with k drafts: cost = t_step·(1 + cv·k) + k·t_draft + t_commit, yield =
1 + p + … + p^k, with p = EMA of P(first draft accepted), cv = (t_verify/t_step − 1)/k_meas from
the measured passes, t_step from one plain calibration step at the start and one every 32
rounds; the k in 0..k_max with the lowest cost/yield wins (3% hysteresis). In plain mode p is
unobservable, so speculation is probed again after 8, 16, … 64 plain steps. Plain steps keep
their final-normed hidden and get their MTP rows in one batched pass before the next
speculative round, so the MTP cache stays complete. Output never depends on it: host
`tests/test_spec.py` now includes `--spec 3 --adaptive` (PASS, 9 prompts); the host (scalar
kernels, compute-bound) correctly drops to plain: 299 rounds, 94 drafted.
Pi numbers: §4b (fixed k=3 vs adaptive at 64 / 2048 / 4096 context, 64 generated).

### 4a. First estimator failed (binary `pinfer_run_ctx4`)
v1 used an EMA of "first draft accepted" (0/1) whose first observation had full weight.
`pinfer-adaptive-p64_20261022_033753.json`: fixed 16.62 vs adaptive 16.91 tok/s (fine), but
`pinfer-adaptive-ctx2048_20261022_034023.json`: fixed k=3 **9.12** (2.17 tok/pass) vs adaptive
**6.96** (57 of 64 tokens as plain steps, 3/9 probe drafts accepted), and
`pinfer-adaptive-ctx4096_20261022_034555.json`: fixed k=3 **7.09** (2.10 tok/pass, verify
237 ms) vs adaptive **6.05** (58 plain steps, 1/10 probe drafts) — below plain decoding (6.48),
because the probe rounds and the batched MTP catch-up of the plain rows bought nothing.
One early miss set p = 0, every k then looked worse than plain, and the probes backed off
(8, 16, 32 steps).
Fix (`pinfer_run_ctx5`): p = decayed hits / trials over every verified draft (a round with a of
k accepted counts a hits and a + (a < k) trials, decay 0.9 per round) plus a prior of 2/3.
Offline mirror of the policy (scratchpad `adapt_sim.py`, measured 2048 costs: step 140 ms,
cv 0.114, 10.7 ms per MTP pass, commit 14 ms; 200 seeds × 64 tokens):

| true p | plain | fixed k=3 | adaptive (mean) |
|---|---|---|---|
| 0.10 | 7.14 | 4.79 | 6.82 |
| 0.25 | 7.14 | 5.73 | 6.92 |
| 0.40 | 7.14 | 6.97 | 7.43 |
| 0.58 (measured at 2048) | 7.14 | 9.07 (Pi: 9.12) | 9.05 |
| 0.75 | 7.14 | 11.71 | 11.50 |
| 0.90 | 7.14 | 14.52 | 14.25 |

So the revised policy stays within 2% of the better choice when speculation pays and within
~5% of plain when it does not (the probe rounds over a 64-token run), against −33% for fixed k.

### 4b. Pi, revised estimator (`pinfer_run_ctx6`, Q4K, 64 generated, `--no-warmup`)
`bench/results/pinfer-adaptive2-ctx{64,2048,4096}_20261022_05*.json`, `pinfer-adaptive2-p64_*.json`
(spec_exact 20/20 for this binary: `spec-exact-ctx6_20261022_045522.json`; all three modes of a
file generate the same tokens by construction and by the host tests):

| context (wikitext) | plain | fixed k=3 | adaptive v1 (ctx4) | **adaptive (ctx6)** | adaptive drafts / round |
|---|---|---|---|---|---|
| 64 Dickens (`prompt64`) | 7.88 | 16.72 (3.32 tok/pass) | 16.91 (fixed 16.62) | **16.82** (+0.6%) | — |
| 64 | 7.86 | 14.06 (2.74 tok/pass) | — | 13.41 (−4.6% vs fixed) | 2.65 |
| 2048 | 7.09 | 9.13 (2.17) | 6.96 | **9.57** (+4.8%) | 1.77 |
| 4096 | 6.11 | 7.38 (2.10) | 6.05 | **7.53** (+2.0%) | 1.84 |

Adaptive is now never slower than plain and beats fixed k=3 where acceptance is low (it picks
k ≈ 2 at p ≈ 0.55, as the cost model predicts); where every draft tends to be accepted it gives
up ~5% to fixed k=3 (calibration steps + a k it lowers after misses). Recommendation: keep
fixed k=2/3 as the library default for chat-length contexts; use `adaptive=1` (or k=3 +
adaptive) for long-context work — or make adaptive the default once it is measured on the
chat and code prompts of SPEC.md §4.3 (not done here).

## 5. KV cache precision (`pinfer_params.kv_dtype`: 0 fp32, 1 fp16, 2 int8 + fp32 row scale)
Rows are converted on store (`pf_kv_store`: FCVTN / vcvtnq, scalar mirrors bit-identical) and
widened to fp32 once per 8-row block per phase (`pf_kv_load`), so the attention kernels are
shared and the n = 1 / batched agreement holds for every format. The MTP head's cache uses the
same format (mtp.c allocates the fp32 size, the largest). Measurements: §5b.

### 5b. Measurements
**Greedy agreement, 256 tokens after the 4096-token wikitext prompt** (Pi, `pinfer_run_ctx4`,
plain decoding; `engine/bench/kv_agree.py` on `bench/results/pinfer-kv{0,1,2}-ctx4096-n256_20261022_03*.json`):

| kv_dtype | decode tok/s | prefill tok/s | peak RSS | tokens equal to fp32 | first difference |
|---|---|---|---|---|---|
| fp32 | 6.24 | 27.4 | 533 MB | 256/256 | — |
| fp16 | 6.42 (+3%) | 28.1 | 482 MB (−51) | 19/256 | token 11 |
| int8 | 6.35 (+2%) | 26.3 | 457 MB (−76) | 32/256 | token 30 |

Greedy continuations of an uncertain text diverge for good at the first near-tie flip, so the
exact-match count says little; int8 diverging *later* than fp16 (8× smaller error) shows that.
**Teacher-forced check** (host, bf16 model, logits of all 512 positions of `ctx512.ids`, scratchpad
`kvq.sh` / `kvq.py`):

| kv_dtype | top-1 agreement vs fp32 | rel. max logit error (median / max) | KL(fp32 ‖ x) mean / max |
|---|---|---|---|
| fp16 | **100.00%** (0 flips) | 1.0e-4 / 4.4e-4 | **3.5e-8** / 4.7e-7 |
| int8 | 99.41% (3 flips at top-2 margins 0.07, 0.02, 0.0002; median margin 1.09) | 8.2e-3 / 6.9e-2 | 2.3e-4 / 1.5e-2 |

So fp16 is numerically lossless for this model (its error is below the bf16 → fp32 reference
gap of the golden test, ~3e-6 relative per layer compounding to ~1e-4 in logits); int8 costs a
little. Perplexity (Q4K, Pi, 8 × 512 and 8 × 2048): §5c. The fp32-KV reference of this
binary, 18.5041 ± 1.25 (`pinfer-ppl-c8-kv0_20261022_040738.json`), equals HEAD's 18.5041
(`pinfer-ppl-c8-final2-q4k_20261022_024414.json`, lead): the attention rewrite changes no bit;
the older 18.4804 predates core2's NEON expf.

### 5c. Perplexity (Pi, Q4K, `pinfer_ppl_ctx4`, llama-perplexity protocol, wikitext-2 test)
| kv_dtype | 8 × 512 (2040 scored) | 8 × 2048 (8184 scored) | result files |
|---|---|---|---|
| fp32 | 18.5041 ± 1.248 | 13.9852 ± 0.459 | `pinfer-ppl-c8-kv0_20261022_040738.json`, `pinfer-ppl-c2048x8-kv0_20261022_042557.json` |
| fp16 | 18.5257 ± 1.250 (+0.12%) | 13.9828 ± 0.459 (−0.02%) | `pinfer-ppl-c8-kv1_20261022_041029.json`, `pinfer-ppl-c2048x8-kv1_20261022_043844.json` |
| int8 | 18.5007 ± 1.248 (−0.02%) | 13.9892 ± 0.460 (+0.03%) | `pinfer-ppl-c8-kv2_20261022_041316.json`, `pinfer-ppl-c2048x8-kv2_20261022_045059.json` |

Every difference is < 0.15% against a ±2.5–7% error bar: perplexity is within noise for both.
Decision: **fp32 stays the default**. Not because fp16 / int8 lose quality (fp16 is lossless to
1e-4 in logits, int8 within PPL noise; both keep batched == token-by-token and spec == greedy,
since every row goes through the same conversion) but because they buy no speed with the
current kernels (§5b, §3c) while changing greedy outputs relative to the fp32 reference at
the first near-tie. fp16 is the recommended option when memory matters (−50% KV: −97 MB at 8192
tokens, −0.8 GB at 32K), int8 when it matters a lot (−73%).

fp16 at longer context (`pinfer_run_ctx4`): spec k=3 at 4096 7.10 tok/s, verify 239 ms vs 231 ms
fp32 (`pinfer-kv1-ctx4096-spec3_20261022_035814.json`; tokens differ, so compare the verify
pass, not tok/s), RSS 506 vs 562 MB; plain at 8192 **5.35 vs 5.42** tok/s fp32, RSS **528 vs
625 MB** (`pinfer-kv1-ctx8192-plain_20261022_040436.json`). Half the bytes, no time saved:
at 8192 the attention is ~400 MFLOP per step (8 heads × 8192 rows × 256 × 2 phases × 2 —
unfused, for bit-exactness), ~17 ms at the 4-core NEON peak before loads, plus per-row
conversion and the row-outer P·V's o loads/stores (640 NEON ops per row for 2 heads), and
the two phases do not overlap with the DRAM stream. I.e. compute-bound once the old 8×
re-streaming was gone; §3c / §3d. Speed: the gain is small because
the per-row widening (fp16: 64 FCVTL + 64 stores + reloads per row; int8: 2× SXTL, SCVTF, FMUL)
costs about as much compute as the DRAM time it saves; §3c re-measures with the blocked P·V.

## 6. State save / restore (`state.c`; `pinfer_state_size/save/load`, `pinfer_generate_greedy_cont`)
Blob = 64-byte header (dims, n_past, kv_dtype, MTP rows) + conv and recurrent state of the 18
linear layers (20.2 MB) + K/V rows [0, n_past) of the 6 full layers (24 KB/token fp32) + the
MTP head's rows [0, mtp_rows) (4 KB/token). `spec.c` now tracks `ctx->mtp_rows`, commits only
the emitted tokens when a stop ends a round early, and runs the MTP pass after a stop too, so a
saved state (or the live context) can be continued speculatively. `pinfer_generate_greedy_cont`
= `_ex` without the reset. Host `tests/test_state.py`: 9/9 exact (3 prompts incl. a split inside
a prefill chunk × plain / spec 3 / adaptive); save 10–16 ms for 20–22 MB on the PC.
Python: `Model.save_state() -> bytes` (C blob + tokens + last logits), `load_state`,
`context_tokens`, `supports_state`, `kv_dtype=` in `load`, `spec_adaptive=` in `generate`;
speculative `generate` continues from the context when the prompt extends it
(`cached_tokens`). `pinfer chat` keeps the state between turns (appends
`\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>…` as token ids; the previous
reply stays as generated, think block included — the reference template would drop it from
the history); `--reprefill` restores the old behaviour. Host pytest 84/84 (`tests/python`,
4 new in `test_state.py`). Pi chat timings: §6b.

### 6b. `pinfer chat`, second turn kept vs re-prefilled (Pi, Q4K, Python package + `libpinfer.so` of ctx4)
Two turns ("Explain in two sentences what a Raspberry Pi 4 is." / "Which of its interfaces
would you use for a camera?"), 48 new tokens max; `logs/runs/20261022_045059_run.log` (spec
k=3), `logs/runs/20261022_045127_run.log` (plain). Prefill tok/s counts only uncached tokens.

| turn 2 | prompt tokens | cached | prefilled | prefill time (TTFT) | decode tok/s |
|---|---|---|---|---|---|
| plain, kept state | 83 | 58 | 25 | **0.68 s** (36.5 tok/s) | 7.86 |
| plain, `--reprefill` | 79 | 0 | 79 | 2.10 s (37.6 tok/s) | 7.88 |
| spec k=3, kept state | 83 | 58 | 25 | **0.77 s** (incl. MTP over the new rows) | 10.41 |
| spec k=3, `--reprefill` | 79 | 0 | 79 | 2.27 s | 9.48 |

Turn 2 starts 3× sooner; the saving grows with the conversation (re-prefill costs the whole
history every turn, ~28 ms per token; the kept state costs only the new turn). The prompts
differ by 4 tokens: the kept context still holds the first reply's empty
`<think>\n\n</think>\n\n` block, which the reference template drops from history, so the two
second answers diverge after ~25 tokens (both name a USB port; the right answer is the CSI
connector — model quality, not engine).

### 6c. State save / restore on the Pi (`engine/bench/state_check.sh`, `pinfer_run_ctx6`, Q4K)
`bench/results/state-check-ctx6_20261022_053906.json`. Run A = the 2112-token wikitext prompt,
B = its first 2048 tokens + `--save-state`, C = `--load-state` + the last 64 tokens; 16
generated tokens; C must equal A.

| mode | C == A | state size | save | load | run B (prefill 2048 + save + write) | run C (read + load + 64-token prefill + 16 tokens) |
|---|---|---|---|---|---|---|
| plain | **yes** | 70.5 MB | 94 ms | 90 ms | 63.4 s | 4.6 s (tail prefill 2.3 s) |
| spec k=3 | **yes** | 78.9 MB (+ MTP rows) | 103 ms | 102 ms | 68.1 s | 5.2 s (2.6 s) |
| spec k=3 adaptive | **yes** | 78.9 MB | 104 ms | 100 ms | 68.6 s | 5.0 s (2.6 s) |

Restoring a 2048-token context costs ~0.1 s against ~63 s to prefill it again (~600×); size
= 20.3 MB fixed + 24 KB per token (+ 4 KB per MTP row) with fp32 KV; save / load run at
~0.75 GB/s (they include first-touch page faults of the destination buffer).

## 7. Not done: bounded-context mode
Design: `pinfer_params.ctx_window` (W) + `ctx_sinks` (S); the KV cache becomes a ring over
positions ≥ S, score/P·V row ranges [0, S) ∪ [pos−W+1, pos]; because K rows are stored
post-RoPE, positions must be re-based (StreamingLLM assigns cache-relative positions), which
means storing K pre-RoPE and rotating per query block — a second kernel path. The recurrent
state is untouched, so the 18 linear layers keep the full history. Left for a later pass;
`pinfer_state_save/load` already covers the "long system prompt, many chats" use.
