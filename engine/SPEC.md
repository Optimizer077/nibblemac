# MTP self-speculative decoding — lab notebook

Owner: speculative-decoding work. Files: `engine/src/{mtp.c,mtp.h,spec.c,spec.h}`, changes in
`engine/src/{forward.c,forward.h,forward_lin.c,pinfer.c}`, `engine/include/pinfer.h` (additive),
`engine/tools/pinfer_run.c` (`--spec`, `--draft-vocab`, `--dump-mtp`), `ref/mtp_reference.py`,
`ref/golden/qwen35-0.8b-mtp/`, `tests/test_spec.py`.

## 1. MTP semantics (sources)

HF transformers 5.20.2 does not run the MTP head (`_keys_to_ignore_on_load_unexpected =
["^mtp.*"]`). Semantics were taken from two independent implementations, then validated
numerically (section 3).

| source | version | what it shows |
|---|---|---|
| vLLM `vllm/model_executor/models/qwen3_5_mtp.py` | main @ `db3b0f0fae4bc98492b3863685e541e62bc59576` (file last changed in `2c99ee9333821030daf84718b2a76ea28d94ba3c`), fetched 2026-10-21 | `Qwen3_5MultiTokenPredictor.forward`: `e = pre_fc_norm_embedding(embed(ids))`, `h = pre_fc_norm_hidden(hidden)`, `fc(cat([e, h]))`, one `Qwen3_5DecoderLayer(layer_type="full_attention")`, `norm`; tied `lm_head` when `tie_word_embeddings`. All norms are `GemmaRMSNorm` = `x·(1+w)` |
| vLLM `vllm/model_executor/models/qwen3_next.py` (same commit) | | the target model's returned hidden state is `self.norm(hidden, residual)`, i.e. **after** the final norm (line ~739) |
| vLLM `vllm/v1/spec_decode/llm_base_proposer.py` (same commit) | | `set_inputs_first_pass`: draft input ids are the target ids shifted by one (`input_ids[:-1] = target_ids[1:]`, last = sampled token) while **positions stay the target positions**; later draft steps feed the drafter's own returned hidden state (post `mtp.norm`) with position + 1 |
| llama.cpp `src/models/qwen35.cpp` `graph_mtp` | v0.7.0 (`/root/src/llama.cpp-v0.7.0`) | `concat(enorm(tok_embd), hnorm(h))` (embedding first), `eh_proj`, full-attention block with its own KV, `shared_head_norm` (= `mtp.norm`), tied output; the main graph exports `t_h_nextn` = state **after** `output_norm` |
| llama.cpp `conversion/qwen.py` | v0.7.0 | adds 1 to every `*norm.weight` except `linear_attn.norm` — including `mtp.pre_fc_norm_*`, `mtp.norm`, `mtp.layers.*` norms |

Resulting definition (implemented in `engine/src/mtp.c`, `ref/mtp_reference.py`):
```
MTP row at position i: inputs h_i (main model, AFTER the final norm) and token x_{i+1}
  u = fc([pre_fc_norm_embedding(embed(x_{i+1})) | pre_fc_norm_hidden(h_i)])   # [2H] -> [H]
  u = full-attention decoder block (q/k norm, partial RoPE at position i, output gate,
      SwiGLU), own KV cache
  m_i = mtp.norm(u);  logits = embed · m_i    -> argmax drafts x_{i+2}
chain step: row at position i+s with inputs (m of the previous step, previous draft)
```
Converter check (`tools/convert.py`): it stores `1 + w` for `mtp.norm`, `mtp.pre_fc_norm_*`
and (via the suffix rule) `mtp.layers.0.{input,post_attention}_layernorm`, `q_norm`, `k_norm`.
**Correct.** Extra evidence from the raw checkpoint: `mtp.pre_fc_norm_embedding` values lie in
[-0.66, +0.04] (as a plain multiplier that would flip the sign of nearly every channel; as
`1+w` ≈ 0.57), and `mtp.norm` (mean 3.44) has the profile of the main final norm (mean 3.31).
No format change needed.

## 2. Engine design

Algorithm per round (see `spec.h`): MTP chain drafts d1..dk → ONE batched main pass on
`[cur, d1..dk]` (`pf_forward_h`, n = k+1, `pf_matmul` reads every weight once) with logits and
final-normed hiddens of all rows → accept longest prefix + bonus → commit → one batched MTP pass
over the a+1 newly committed positions with the main model's hiddens (gives d1 of the next
round and overwrites the MTP KV rows the chain wrote with chained hiddens; vLLM does the same).

Rollback without snapshots:
- **Full-attention KV caches** (main + MTP): nothing to undo. Rows past `n_past` are stale and
  are overwritten before any query can read them (causal).
- **Linear attention** (18 layers, recurrent state 16×128×128 fp32 = 1 MB/layer, 18.9 MB total;
  conv state 3×6144 floats/layer): during verification `ctx->lin_log_on` makes
  `pf_linear_attn` run on scratch copies — one head's 64 KB state at a time (it stays in L2),
  one layer's 72 KB conv state — and log per token and head the rank-1 update
  (k [128], delta [128], decay [1]) plus the raw conv inputs (≈ 1 MB per token in total).
  The committed states are never written during the pass. `pf_lin_commit(keep = a+1)` then
  replays the first a+1 updates in place: `S = S·decay; S = S + k_i·delta_j` — the exact
  per-element fp32 operations of the recurrence (C11 mode ⇒ `-ffp-contract=off`, so no FMA
  fusion can change the rounding), hence bit-identical to having run only those tokens; the
  conv state becomes the last 3 rows of [old rows, raw inputs 0..a].
  Cost per round: one read of the state during verification + one read/write in the commit
  (≈ 57 MB of traffic, vs 38 MB for one plain decode step and ~1.5 GB of weights) — no 19 MB
  snapshot copies, no second state buffer.
- `forward_lin.c` loop order changed to head-outer / token-inner (state in L2 across the
  chunk); heads are independent, so outputs are bit-identical (golden test A vs B: 0 error).

Draft vocabulary (`--draft-vocab N` / `pinfer_spec_params.draft_vocab`): the tied head
(248320 × 1024 bf16 = 509 MB) is 93% of an MTP step's bytes (MTP block: 41 MB). Drafting from
token ids < N only (Qwen's BPE ids are roughly frequency-ordered) cuts a draft step from
550 MB to 41 + 2·1024·N bytes. Verification always uses the full head, so output is unchanged.

API (additive, `pinfer.h`): `pinfer_spec_params {k, draft_vocab}`, `pinfer_spec_stats`,
`pinfer_default_spec_params`, `pinfer_has_mtp`, `pinfer_generate_greedy`.
CLI: `pinfer_run ... --spec k [--draft-vocab N] [--bench]`, `--dump-mtp f`.

## 3. Validation

### 3.1 PyTorch reference (`ref/mtp_reference.py`, fp32, transformers 5.20.2 / torch 2.14.1)
Builds the head from HF blocks (`Qwen3_5DecoderLayer` full-attention, `Qwen3_5RMSNorm`), loads
`mtp.*` from the safetensors. Chain steps use an explicit block mask so every chained row sees
the main-hidden MTP rows 0..i plus its own chain (= what the engine's KV cache holds).
Log: `logs/spec/mtp_reference_20261021_215644.log`. Outputs in `ref/golden/qwen35-0.8b-mtp/`.

**Acceptance (semantics check)**: 7 varied prompts (prose, Dickens, Python, Q&A, history,
recipe list, Chinese), 96 greedy tokens each, 651 scored positions, per-position rates:

| variant | P(d1 ok) | P(d2 ok \| d1) | P(d3 ok \| d1,d2) | P(d4 ok \| …) | E[accepted] k=1..4 |
|---|---|---|---|---|---|
| **as specified, full vocab** | **0.873** | 0.868 | 0.897 | 0.937 | 0.87 / 1.63 / 2.31 / 2.95 |
| draft vocab 131072 | 0.873 | 0.868 | 0.897 | 0.937 | identical |
| draft vocab 65536 | 0.810 | 0.865 | 0.899 | 0.920 | 0.81 / 1.51 / 2.14 / 2.72 |
| draft vocab 32768 | 0.763 | 0.815 | 0.835 | 0.837 | 0.76 / 1.39 / 1.91 / 2.34 |
| control: hidden BEFORE final norm | 0.545 | 0.369 | 0.374 | 0.388 | 0.55 / 0.75 / 0.82 / 0.85 |
| control: concat order swapped | 0.003 | 0 | 0 | 0 | 0.003 |

Chance is ~0; 87% first-draft acceptance and the collapse of both controls confirm the
semantics (post-norm hidden, embedding half first). Per prompt the first-draft rate ranges
from 0.54 (Chinese) to 0.98 (Dickens, recipe list).

### 3.2 Engine MTP head vs reference (`tests/test_spec.py` part 1)
`pinfer_run --dump-mtp` on the golden prompt: 10 prompt rows + 3 chained rows, rel max error
≤ 3.9e-6 on every row, identical argmax on all 13 rows, chained drafts
[1542, 383, 866, 6165] = reference. (Logs: `logs/spec/test_spec_20261021_221629.log`.)

### 3.3 Exactness (`tests/test_spec.py` part 2, host build, `MATVEC=ref`)
9 prompts (golden, 6 other varied, 64-token bench prompt, 74-token two-chunk prompt) × {plain,
`--spec 0..4`, `--spec 3 --draft-vocab 32768`}, 32–40 generated tokens: **all 63 runs equal
the plain greedy tokens; golden prompt equals `greedy_ids.npy`. PASS.**
Logs: `logs/spec/test_spec_20261021_221121.log` (first run, 8 prompts),
`logs/spec/test_spec_20261021_221629.log` (after the prefill-chunk change, 9 prompts).
`tests/test_golden.py` still passes with the refactored forward pass (hidden ≤ 2.9e-6, A vs B
bit-identical): `logs/spec/golden_20261021_220403.log`.

### 3.4 Real acceptance in the engine (host, greedy, 64 tokens, 8 prompts)
Per-position rates over-estimate what a decoder sees: a round starts right after a
rejection, so hard positions are over-sampled (checked on the golden prompt: the engine's
drafts equal the reference drafts at every round start, the gap is pure selection). Engine
tokens per verification pass (= 1 + accepted / rounds), 7 varied prompts without the bench
prompt (logs `logs/spec/accept_sweep_20261021_222813.log`, `..._224314.log`):

| k \ draft vocab | full | 131072 | 65536 | 32768 | 16384 |
|---|---|---|---|---|---|
| 1 | 1.793 | 1.793 | 1.736 | 1.670 | |
| 2 | 2.423 | 2.423 | 2.216 | 2.120 | |
| 3 | 2.864 | 2.864 | 2.534 | 2.371 | |
| 4 | 3.219 | 3.219 | 2.827 | 2.625 | 2.564 |
| 5 | | | 3.021 | 2.827 | 2.689 |
| 6 | | | 3.084 | 2.882 | 2.791 |

131072 is free (identical drafts on every prompt) and halves a draft step's bytes.

## 4. Pi measurements (2026-10-21, `harness/pi.py bench`, all `valid=true`)

Prompt: `bench/common/prompt64.json` (64 ids, Dickens — easy, repetitive text), 64 generated
tokens, bf16 model `models/qwen35-0.8b-bf16.pq`. Decode tok/s counts tokens 2..64 (as the
plain loop). `c_v` = verify pass time / plain decode step time (measured cost ratio of ONE
batched pass of n = k+1 tokens). **Every run generated exactly the k=0 tokens** (checked from
the stored stdout).

### 4.1 Scalar kernels (`MATVEC=ref`, 1 thread; binary `engine/pinfer_run_spec`)
| config | decode tok/s | vs k=0 | tok/pass | accept rate | MTP ms/pass | verify ms/pass | c_v | commit ms | result file |
|---|---|---|---|---|---|---|---|---|---|
| k=0 | 1.546 | 1.00 | – | – | – | – | – | – | `pinfer-spec-ref-k0_20261021_222521.json` |
| k=1 | 1.151 | 0.74 | 1.800 | 0.800 | 230 | 1316 | 2.03 | 17.9 | `pinfer-spec-ref-k1_20261021_230945.json` |
| k=2 | 0.999 | 0.65 | 2.423 | 0.740 | 454 | 1950 | 3.02 | 20.7 | `pinfer-spec-ref-k2_20261021_231601.json` |
| k=3 | 0.841 | 0.54 | 2.864 | 0.641 | 675 | 2706 | 4.18 | 23.2 | `pinfer-spec-ref-k3_20261021_232744.json` |
| k=4 | 0.854 | 0.55 | 3.706 | 0.687 | 914 | 3400 | 5.26 | 27.2 | `pinfer-spec-ref-k4_20261021_233157.json` |

Scalar kernels are compute-bound (c_v ≈ k+1), so speculation can only lose here — expected.

### 4.2 NEON kernels (`MATVEC=fast`, 4 threads; binary `engine/pinfer_run_spec_fast`)
Kernel sources are the kernels agent's work in progress at build time; hashes in
`logs/spec/fast_build_hashes_20261021_*.txt`. k=0 runs at 440 ms/token, ~88% of the
1.506 GB / 3.9 GB/s bandwidth floor, i.e. mostly bandwidth-bound.

| config | decode tok/s | vs k=0 | tok/pass | MTP ms/pass | verify ms/pass | c_v | commit ms | result file |
|---|---|---|---|---|---|---|---|---|
| k=0 | 2.274 | 1.00 | – | – | – | – | – | `pinfer-spec-fast-k0_20261021_231307.json` |
| k=1 | 2.854 | 1.26 | 1.800 | 143 | 470 | 1.07 | 18.5 | `pinfer-spec-fast-k1_20261021_232558.json` |
| k=2 | 2.821 | 1.24 | 2.423 | 280 | 559 | 1.27 | 20.4 | `pinfer-spec-fast-k2_20261021_233014.json` |
| k=3 | 2.737 | 1.20 | 2.864 | 416 | 607 | 1.38 | 23.4 | `pinfer-spec-fast-k3_20261021_233227.json` |
| k=4 | 2.851 | 1.25 | 3.706 | 564 | 709 | 1.61 | 26.6 | `pinfer-spec-fast-k4_20261021_233316.json` |
| k=2, vocab 131072 | 3.497 | 1.54 | 2.423 | 160 | 511 | 1.16 | 21.2 | `pinfer-spec-fast-k2-v131072_20261021_233341.json` |
| k=3, vocab 131072 | 3.164 | 1.39 | 2.864 | 238 | 643 | 1.46 | 23.6 | `pinfer-spec-fast-k3-v131072_20261021_233408.json` |
| k=4, vocab 131072 | 3.366 | 1.48 | 3.706 | 326 | 748 | 1.70 | 27.6 | `pinfer-spec-fast-k4-v131072_20261021_233434.json` |
| **k=3, vocab 32768** | **3.993** | **1.76** | 2.864 | 89 | 604 | 1.37 | 23.6 | `pinfer-spec-fast-k3-v32768_20261021_233457.json` |
| k=4, vocab 32768 | 3.757 | 1.65 | 3.500 | 118 | 787 | 1.79 | 26.4 | `pinfer-spec-fast-k4-v32768_20261021_233521.json` |
| k=5, vocab 32768 | 3.759 | 1.65 | 3.706 | 145 | 814 | 1.85 | 27.2 | `pinfer-spec-fast-k5-v32768_20261021_233545.json` |

Run-to-run noise of `c_v` is ~±0.1 (k=2: 1.27 vs 1.16 in two runs with identical work).
A full-vocab MTP step costs ~140 ms (550 MB at ~3.9 GB/s) — a third of a decode step — which is
why the restricted draft vocabulary is the biggest lever; 32768 ids = 108 MB ≈ 30 ms.
### 4.3 Varied prompts, updated NEON kernels (binary `engine/pinfer_run_spec_fast2`)
Rebuilt after the kernels agent changed `matvec_fast.c` / `kern_float.c` (hashes:
`logs/spec/fast2_build_hashes_20261021_*.txt`); every row compares against k=0 of the same
binary and prompt; all spec runs generated exactly the k=0 tokens.

| prompt | k=0 tok/s | k=2/32768 | k=3/32768 | k=4/32768 | tok/pass k=2/3/4 |
|---|---|---|---|---|---|
| Dickens (bench64) | 2.308 | **4.130 (1.79×)** | 3.691 (1.60×) | 3.868 (1.68×) | 2.42 / 2.86 / 3.50 |
| "The French Revolution began in 1789 when" | 2.326 | **3.272 (1.41×)** | 3.124 (1.34×) | 2.780 (1.19×) | 1.91 / 2.17 / 2.33 |
| `def quicksort(arr):` + docstring | 2.331 | **4.503 (1.93×)** | 4.153 (1.78×) | 4.154 (1.78×) | 2.63 / 3.15 / 3.94 |

Result files: `bench/results/pinfer-spec-fast2-{dickens64,french,quicksort}-{k0,k2-v32768,
k3-v32768,k4-v32768}_20261021_23*.json`. Verify ms/pass at k=2/3/4 ≈ 505 / 590–660 /
705–800 (c_v ≈ 1.17 / 1.4 / 1.7 against ~433 ms/token).

Default `pinfer_default_spec_params()` is therefore **k = 2, draft_vocab = 32768** (best on
all three prompts with the current kernels; with a cheaper batched verify the optimum moves
to k = 3–4, §5).

## 5. Projection

Per verification pass: cost = c_v(k+1)·T_step + k·T_mtp(N) + T_commit, yield = tokens/pass.
Bytes (from the .pq directory): decode step 1505.8 MB (layers 997.2 + head 508.6); MTP step
40.9 MB + 2·1024·N B → ratio to a decode step 0.365 (full), 0.205 (131072), 0.116 (65536),
0.072 (32768), 0.049 (16384). Commit ≈ 20–27 ms (≈ 0.05 step).

**(a) Ideal bandwidth-bound verification (c_v = 1)**, tokens/pass from the host sweep on the
7 *varied* prompts (§3.4): k=2/N=32768: 2.120/(1+0.143+0.05) = **1.78×**;
k=3/32768: 2.371/1.265 = **1.87×**; k=4/32768: 2.625/1.337 = **1.96×**; k=4/16384:
2.564/1.246 = **2.06×**; k=5/16384: 2.689/1.295 = **2.08×**; k=3/full: 2.864/2.145 = 1.34×.
On the easy bench prompt (tok/pass 2.86–3.7) the same model gives 2.2–2.6×.

**(b) Current NEON kernels** (measured c_v ≈ 1.07, 1.2, 1.37, 1.7, 1.85 for n = 2..6;
T_mtp(32768) ≈ 30 ms, T_step 440 ms) on varied text: k=1 1.41×, **k=2 1.53×**, k=3 1.46×,
k=4 1.29×. Measured on the bench prompt: 1.76× (k=3). The gap between (a) and (b) is the
verify pass growing with n: the batched bf16 kernel does per-row work proportional to n, and
the per-token attention/delta-rule work is serial. Making `pf_matmul` with n = 2..5 cost ~ one
weight stream (kernels agent) is worth up to ~+30% on top of the current speculative gain.

Caveat for quantized models (Q8/Q4): the decode step shrinks with the weight bytes while the
per-token compute does not, so c_v(n) will rise; the draft vocabulary restriction stays
proportionally as valuable (the head is quantized like everything else).

## 6. Failures / notes
- First `--dump-mtp` attempt failed silently (exit 1): chain rows go 3 positions past the
  default `max_ctx = n_prompt + n_gen`. Fixed: default context +4 with `--dump-mtp`, error
  message printed.
- A Git Bash heredoc turned `\\n` inside a Python patch script into a real newline once;
  caught by the compiler, fixed with a direct edit.
- First look at engine acceptance on the golden prompt (1.21 accepted/pass vs 2.42
  per-position in the reference) looked like a bug; a per-round trace showed the engine's
  drafts equal the reference drafts at every round — it is selection bias (§3.4).
- `MATVEC=fast` did not compile for ~30 min (kernels agent mid-edit; not touched); the host
  build of `matvec_fast.c` never compiles (aarch64 `yield` asm), so exactness with the fast
  kernels is checked on the Pi only (identical tokens for every k, §4.2).
- The scalar Pi binary predates the "no MTP logits for non-final prefill chunks" change and
  the `pinfer_generate_greedy_ex` change; neither affects a ≤ 64-token prompt with EOS ignored.
- Pi lock queue stalled ~25 min behind another agent's ORT run (now capped by the harness).
- ~23:37 the WSL default user changed from `root` to `optim` (not by this work), so
  `/root/venv-ref` became unreadable: `logs/spec/golden_20261021_234204.log` is that failed
  attempt. Re-run with `wsl.exe -d Ubuntu -u root`: golden test PASS on the final build,
  `logs/spec/golden_20261021_234217.log`.

## 7. API additions requested by the packaging agent (2026-10-21)
`pinfer_generate_greedy_ex(ctx, prompt, n, out, n_max, sp, opts, stats, &status)` with
`pinfer_gen_opts {stop_ids, n_stop, on_token, user}`: extra stop tokens (EOS always stops),
per-token callback (nonzero = stop; tokens arrive in bursts of up to k+1 when speculating),
and no lost tokens on context overflow (returns the count, `status = PINFER_E_CTX_FULL`).
`pinfer_generate_greedy` keeps its contract (any error → negative code). Verified with a
throwaway C harness on the host for k = 0 and k = 3: stop id, callback stop after 7 tokens,
overflow at max_ctx = 25 returns 16 golden tokens + CTX_FULL — all ok.
