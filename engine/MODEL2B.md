# Qwen3.5-2B on the Pi 4 — lab notebook (issue #7)

Owner: 2B agent, 2026-10-22 (KST). Branch `worktree-agent-a24ea2f8313af7f40` from main `792676c`.
Question: does Qwen3.5-2B fit in the Pi 4's 4 GB, is the engine correct on it, and how fast is it
next to 0.8B and llama.cpp?

## 1. The checkpoint

| Item | Value |
|---|---|
| Repository | `Qwen/Qwen3.5-2B` (official; `base_model: Qwen/Qwen3.5-2B-Base`, post-trained) |
| Revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` (last modified 2026-03-02) |
| License | **Apache-2.0** (model card tag `license:apache-2.0`, `LICENSE` file in the repo) |
| Files | one shard `model.safetensors-00001-of-00001.safetensors`, 4,548,221,488 B, sha256 `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`; `config.json` sha256 `ed1c1723…` |
| Local copy | `models/Qwen3.5-2B` (main repo, gitignored), `hf download Qwen/Qwen3.5-2B --revision 15852e8c… --local-dir models/Qwen3.5-2B` (huggingface_hub 2.2.0, `/root/venv-ref/bin/hf`), 2 min |
| Tokenizer | `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` **byte-identical** to 0.8B (sha256 `5f9e4d49…`, `49e2b6e3…`, `273d8e0e…`), so every token-id file of the project (`prompt64`, `chat27`, wikitext) is valid for 2B |

`text_config` against 0.8B (`config.json`, everything else identical):

| Field | 0.8B | 2B |
|---|---|---|
| `hidden_size` | 1024 | **2048** |
| `intermediate_size` | 3584 | **6144** |
| `num_hidden_layers`, `layer_types` | 24 = 6 × (3 linear + 1 full) | same |
| full attention: heads, KV heads, `head_dim`, `partial_rotary_factor`, θ | 8, 2, 256, 0.25 (rot 64), 1e7 | same |
| linear attention: k / v heads, k / v head dim, conv kernel | 16 / 16, 128 / 128, 4 | same |
| `vocab_size`, `tie_word_embeddings`, `eos_token_id` | 248,320, true, 248044 | same |
| `mtp_num_hidden_layers` | 1 | 1 (MTP head present: `mtp.fc [2048, 4096]` + one full-attention block) |
| `rms_norm_eps` | 1e-6 | same |

Vision tower (not stored by the converter): depth 12 → 24, hidden 768 → 1024.

Text + MTP parameters (from the safetensors header; 335 tensors as for 0.8B): layers 1373.3 M,
tied embedding / head 508.6 M, MTP head 60.8 M, total **1.943 B** (0.8B: 0.85 B incl. MTP).
All matrices satisfy the quantized shape rule (rows % 4, cols % 256: cols are 2048, 4096, 6144).

## 2. Where the code assumed 0.8B dimensions

Audit of every source file (C engine, converter, tools, tests, references, Pi scripts) for
0.8B constants (1024, 3584, 248320, 6144, …) and implicit size relations:

| Place | Assumption | Status for 2B |
|---|---|---|
| `engine/src/model.c`, `pq_format.h` | none: every shape is checked against header fields; limits `PQ_MAX_DIM` 65536 etc. | loads 2B unchanged |
| `engine/src/pinfer.c` `set_dims` / `alloc_buffers` | buffers sized from header fields | fine |
| `engine/src/mtp.c` (`cat = ctx->ta`, rows of 2H) | **implicit**: `ta` rows are `max(conv_dim, 2·n_heads·hd, ffn)` floats and must hold 2H | 6144 ≥ 4096: fine for 2B (and for every Qwen3.5 size, where ffn ≥ 3H); see §2a |
| `engine/src/forward*.c`, `kern_*.c`, `matvec_*.c` | all loops over `ctx->H`, `ctx->ffn`, `W->cols`; stack buffers bound only head dims (`PF_KV_MAX_HD` 512, `PF_LIN_MAX_DK` 256, `lin_v_dim` ≤ 512) which 2B shares with 0.8B | fine |
| `kern_act_group` 16 (prefill groups) | tuned so that 16 × 1024 int8 activations stay in L1 | correct for any cols; at cols 2048 / 6144 a group is 32 / 96 KB (A72 L1D 32 KB), so prefill may be less tuned (kernels agent's knob) |
| `tools/convert.py`, `tools/quant.py` | none (header from `config.json`, shapes from safetensors) | converts 2B unchanged; peak memory 17.2 GB → row blocks (§3) |
| `ref/dump_reference.py` | defaults only (`--model`, `--revision`, `--name`) | used with 2B arguments |
| `ref/mtp_reference.py` | **golden / output directories hard-coded** to `qwen35-0.8b` | added `--name` (default unchanged) |
| `tests/test_golden.py`, `tests/test_spec.py` | **golden directory hard-coded**; "24 = final-normed" in a print | added `--golden` (default unchanged) |
| `engine/bench/*.sh`, `bench/llamacpp/*.sh`, `bench/quality/*.sh` | model path hard-coded to 0.8B files | new 2B scripts in `bench/model2b/` instead of edits |
| `engine/bench/kbench.c` | the 0.8B matrix shapes | not needed for this question |
| `python/pinfer` | none found (dimensions from the C API) | not exercised |

### 2a. Engine change: the MTP scratch rows (a latent heap overflow, not needed for 2B)

The 2B model runs on the engine unchanged. The one implicit relation found is real, though:
`pf_mtp_forward` builds its `n` fc-input rows of 2H floats in `ctx->ta`, which `pinfer.c`
allocated as 64 rows of `max(conv_dim, 2·n_heads·head_dim, ffn)` floats. A header that passes every
loader check can break that. `tests/test_dims.py` (new) writes small random-weight bf16 models;
its `mtp-wide` case (H 512, ffn 64, one 32-dim head, 40-token prompt, so the MTP prefill has 40
rows of 1024 floats against a 6144-float buffer) gives, on the unchanged sanitizer build,
**AddressSanitizer heap-buffer-overflow, WRITE in `pf_rmsnorm` ← `pf_mtp_forward` (`mtp.c:75`) ←
`pf_spec_prefill`**, for k = 1, 2, 3 (`/root/m2b-logs/dims_110239.log`, WSL).
Fix (`engine/src/pinfer.c` `alloc_buffers`): `ta` rows are `max(…, 2H)` floats. For 0.8B and 2B the
size is unchanged (6144 ≥ 2048 / 4096), so nothing else changes; after the fix all three cases pass
(`dims_110320.log`): `qwen-like` (3 linear + 1 full layer, GQA 4:1, ffn 3H), `mtp-wide`, `odd-heads`
(6 query heads per KV head, head_dim 48, `lin_v_heads` = 2 × `lin_k_heads`), plain and `--spec 1..3`
all clean and speculative == plain.

## 3. Conversion, bytes per token and the memory budget

Conversion (WSL, `/root/venv-ref/bin/python`, numpy only), `tools/convert.py` unchanged:

| File | Command | Size | Time | Peak RSS (PC) | md5 |
|---|---|---|---|---|---|
| `models/pq/qwen35-2b-bf16.pq` | `convert.py models/Qwen3.5-2B models/pq/qwen35-2b-bf16.pq` | 3,886,474,496 B | 1:08 | 8.3 GB | `f7f7363f…` |
| `models/pq/qwen35-2b-q4k-headq4k.pq` | `… --quant q4k` (the 0.8B default recipe: Q4K_G32 everywhere, Q4K head) | 1,094,714,624 B | 3:47 | **17.2 GB** | `6441a5bd368ec24418d2a26db5006f7b` |

The Q4K peak came from quantizing the 2 GB fp32 embedding (248,320 × 2048) in one piece
(`quant_q4k` keeps several full-size temporaries). **Changed:** `tools/convert.py` now quantizes in
blocks of 16,384 rows (`QUANT_ROW_BLOCK`); every quantized dtype's parameters are per row and the
packing is 4-row tiles in row order, so the bytes are the same. Checked by md5 after re-converting:
2B Q4K `6441a5bd…` (peak RSS **17.2 → 8.4 GB**, 4:55), 0.8B Q4K `a34be00e…` and 0.8B
`mix-headq4k` (Q8 overrides) `94498fae…`, all identical to the existing files (scratchpad
`m2b/sh/convert_check.sh`).

Bytes per token (from the file's directory): 

| Part | 0.8B Q4K | 2B Q4K | ratio |
|---|---|---|---|
| layer matrices | 279.9 MB | 772.2 MB | 2.76 |
| tied embedding / head (Q4K) | 143.0 MB | 286.1 MB | 2.00 |
| F32 vectors (norms, conv, gates) | 2.0 MB | 2.2 MB | |
| MTP head (read only when speculating) | 11.5 MB | 34.3 MB | 2.97 |
| **read per plain decode step** (all but MTP) | **424.9 MB** | **1060.4 MB** | **2.50** |
| + recurrent state read + write (18 × 16 × 128 × 128 × 4 B, twice) | 37.7 MB | 37.7 MB | |

Decode ceiling at the DRAM limit (4.2 GB/s, `memband`): 1098 MB → **3.8 tok/s**. At the stream
rate the 0.8B build reaches (462.6 MB × 7.89 tok/s = 3.65 GB/s): **3.3 tok/s**. 0.8B: 7.89.

Memory budget (Q4K, fp32 KV, `max_ctx` 4096; sizes from the allocation code, ARCHITECTURE.md §5):

| Item | 0.8B | 2B |
|---|---|---|
| Weights (`mmap`, page cache, shared) | 436.5 MB | 1094.7 MB |
| Recurrent + conv state | 18.9 + 1.3 MB | same (linear-attention dims are equal) |
| KV cache main / MTP | 24 / 4 KiB per token (96 / 16 MiB at 4096; pages touched as used) | same (attention dims are equal) |
| 64-row scratch + attention scores | ~4 MB + 1 MiB | ~6.6 MiB (x, xn, out 3 × 0.5 MiB; ta, tb 1.5 MiB each; to 0.5; tc 0.125; att 1) |
| Speculation, k = 3 | ~7 MB | ~8 MB (verify logits 4 MB, rollback log 3 MB, MTP rows 1 MiB) |
| Expected peak RSS, 64-token prompt | 433 MB measured | ~1.13 GB |

The Pi has 3794 MiB, of which 3576 MiB were available at idle (`free -m`, 10:55). 2B needs about
1.1 GB, leaving ~2.4 GB: even a 32k-token context (KV 768 MiB + MTP KV 128 MiB) fits. The bf16
2B file (3.89 GB) does not fit and is used only on the PC.

## 4. Correctness against the fp32 reference

Golden data: `ref/dump_reference.py --model models/Qwen3.5-2B --revision 15852e8c… --name
qwen35-2b` → `ref/golden/qwen35-2b/` (gitignored), same prompt as 0.8B ("The Raspberry Pi 4 is a
small computer that", 10 ids), fp32, transformers 5.20.2, torch 2.14.1+cpu; 40 s on the PC, peak
RSS 13.6 GB. Hidden states [25, 10, 2048], logits [10, 248320]. Greedy (32 tokens): " can be used
for a variety of purposes, including running a web server. To run a web server on the Raspberry
Pi 4, you can use the built" (0.8B: " can be used for a wide range of tasks. …").
MTP reference: `ref/mtp_reference.py --model models/Qwen3.5-2B --name qwen35-2b` →
`ref/golden/qwen35-2b-mtp/` (16 min, 13.6 GB).

Host tests (WSL, `MATVEC=ref` build of this branch, bf16 file; the 0.8B tolerances unchanged):

| Test | Result | 0.8B for comparison |
|---|---|---|
| `tests/test_golden.py --model models/pq/qwen35-2b-bf16.pq --golden ref/golden/qwen35-2b` | **PASS**: worst per-layer rel. max error **1.18e-6** (limit 1e-3; layer 15), logits 1.21e-6, top-1 10/10, batched == token-by-token bit-identical, greedy **32/32** | 3.19e-6, 2.30e-6, 10/10, 32/32 |
| `tests/test_spec.py --model … --golden ref/golden/qwen35-2b` | **PASS**: MTP head rel. error ≤ **3.9e-6** on 10 prompt rows + 3 chain rows, every argmax equal, chained drafts `[381, 1429, 364, 5090]` = reference; exactness 9 prompts × 8 configurations (k 1–4, dv 32768, adaptive, `-b 2`) all identical to plain | ≤ 4e-6 |
| Pi `bench/model2b/spec_exact_2b.sh ./engine/pinfer_run_2b1` (Q4K, NEON) | **10/10** `"match":true` (chat27, prompt64 × k = 0, 1, 2, 3, 4) | 20/20 |

Logs: session scratchpad `m2b-logs/golden_2b_105940.log`, `spec_2b_112050.log`;
`bench/results/spec-exact-2b-q4k_20261022_114018.json`.

MTP acceptance in the fp32 reference (7 prompts, 96 greedy tokens each, 651 rows; the semantics
controls must lose):

| | first draft, full vocab | first draft, dv 32768 | E[accepted], k = 2 / 3, dv 32768 | swapped concat | pre-norm hidden (steps 1–4) |
|---|---|---|---|---|---|
| 0.8B | 87.3% | 76.3% | 1.39 / 1.91 | 0.3% | 0.55, 0.37, 0.37, 0.39 |
| 2B | **83.4%** | **74.8%** | **1.31 / 1.72** | 1.7% | 0.74, 0.62, 0.60, 0.62 (vs 0.83, 0.78, 0.77, 0.78 correct) |

The larger model's MTP head agrees *less* often with its own greedy output, so speculation should
gain a little less on 2B than on 0.8B.

## 5. Pi measurements

Binary `engine/pinfer_run_2b1` / `pinfer_ppl_2b1` (this branch, `make -C engine pi MATVEC=fast`,
md5 `b18a5157…` / `7b616ead…`), model `models/qwen35-2b-q4k-headq4k.pq` (md5 `6441a5bd…`, checked
on the Pi), 4 threads, fp32 KV, 64 generated tokens, `--bench` (untimed warm-up prefill + 4 tokens
first). `bench/model2b/speed.sh`: plain, `--spec 2 --draft-vocab 32768`, `--spec 3 --draft-vocab
32768`, 3 rounds, medians. All runs valid, throttled 0x0. (`speed.sh` gained its optional MODES
argument after these two runs; the default runs the same three modes with the same output.)

| prompt | mode | decode tok/s (3 runs) | median | tokens / pass | MTP / verify ms per pass | prefill tok/s | peak RSS MiB |
|---|---|---|---|---|---|---|---|
| prompt64 | plain | 3.402, 3.414, 3.420 | **3.41** | | | **14.6** | 1040.5 |
| prompt64 | k = 2 | 4.379, 4.348, 4.320 | **4.35** | 1.58 | 37.3 / 316.5 | 13.8 | 1079.1 |
| prompt64 | k = 3 | 4.162, 4.084, 4.168 | 4.16 | 1.66 | 54.8 / 334.6 | 13.2 | 1080.9 |
| chat27 | plain | 3.422, 3.418, 3.409 | **3.42** | | | 13.7 | 1036.7 |
| chat27 | k = 2 | 4.816, 4.778, 4.824 | **4.82** | 1.75 | 37.3 / 316.8 | 13.0 | 1074.5 |
| chat27 | k = 3 | 4.782, 4.788, 4.873 | 4.79 | 1.97 | 54.5 / 346.2 | 13.1 | 1076.3 |

Files: `bench/results/speed-2b-q4k-p64_20261022_114956.json`, `speed-2b-q4k-chat27_20261022_122132.json`.

- **Plain decode 3.41 tok/s** = 293 ms per token for ~1098 MB (weights + recurrent state) → 3.75
  GB/s, 89% of the 4.2 GB/s DRAM limit and a better stream rate than 0.8B's 3.65 GB/s (bigger
  matrices, same fixed overheads). The prediction from 0.8B's rate was 3.3.
- **Verify is cheap, acceptance is the limit.** A k = 2 verify pass costs 1.08× a plain step
  (316.8 vs 292.6 ms; 0.8B: 1.19×), k = 3 1.14–1.18×, so the speed-up is set by tokens per pass:
  1.58–1.75 at k = 2 (0.8B on prompt64 at k = 3: 3.32). k = 2 ≥ k = 3 on both prompts.
- **prompt64 acceptance is lower on the Pi than on the host:** host bf16 (`MATVEC=ref`) on the
  same prompts: prompt64 2.25 (k = 2) / 2.42 (k = 3) tokens per pass, chat27 1.85 / 1.97
  (scratchpad `m2b/sh/host_p64_spec.sh`). On chat27 Q4K matches bf16; on prompt64 it does not
  (the Q4K greedy text departs from bf16's after a few tokens, so the drafted text differs).
  See the Q8 MTP variant below.
- Prefill 13–15 tok/s on 27–64-token prompts (0.8B: 37); the 512-token chunks of the perplexity
  run go at 10.2 tok/s.

**Perplexity** (wikitext-2 test, first 8 chunks × 512, 2040 scored tokens; same protocol as
0.8B): `pinfer_ppl_2b1 -m models/qwen35-2b-q4k-headq4k.pq -f data/wikitext2-qwen35.tokens.bin
--chunks 8 -t 4` → **13.375 ± 0.852** (402 s; `bench/results/ppl-c8-2b-q4k_20261022_122815.json`).
llama.cpp v0.7.0 on the same 8 chunks (PC, `bench/model2b/ppl_pc_c8.sh`, scratchpad
`m2b-logs/ppl_pc_112219.log`): bf16 12.809, Q8_0 12.806, Q4_K_M 13.322, Q4_0 13.726; control
0.8B Q4_K_M 18.591 (= the published 18.59). pinfer Q4K is +4.4% over bf16 (0.8B: 18.48 vs 17.93,
+3.1%), on par with Q4_K_M (+4.0%) and better than Q4_0 (+7.2%). **2B vs 0.8B: 13.38 vs 18.48,
−28%.**

**Q8 MTP head (experiment, `--quant q4k --override 'mtp\.=q8'`, 1125.1 MB, md5 `dc653367…`;
main model identical).** 2 rounds, `bench/results/final-2b_20261022_131607.json`:

| prompt | k | Q4K MTP (default) | Q8 MTP | tokens / pass | MTP ms / pass |
|---|---|---|---|---|---|
| prompt64 | 2 | 4.35 | 4.33 | 1.58 → 1.62 | 37.3 → 49.8 |
| prompt64 | 3 | 4.16 | 4.11 | 1.66 → 1.70 | 54.8 → 72.9 |
| chat27 | 2 | 4.82 | 4.94 | 1.75 → 1.85 | 37.3 → 50.2 |
| chat27 | 3 | 4.79 | 4.67 | 1.97 → 2.03 | 54.5 → 72.1 |

The Q8 head buys 0.04–0.10 tokens per pass and loses most of it to its own cost: not worth
+30 MB. So the low prompt64 acceptance is not the MTP head's quantization; the most likely cause
is the text (the Q4K model's own continuation of prompt64 differs from bf16's) — not verified.
The file was deleted from the Pi after the run.

**llama.cpp v0.7.0** (`620d2225`, the Pi build of `bench/BASELINES.md`), `Qwen3.5-2B-Q4_K_M.gguf`
(1,312,164,832 B, sha256 `727ffd79…`, made with `bench/llamacpp/make_ggufs.sh Qwen3.5-2B`,
5.36 bits/weight), `llama-bench -t 4 -p 64 -n 64 -r 3` through `bench/model2b/llama_rss.sh`
(VmHWM sampled every 0.2 s): pp64 **5.99 ± 0.03**, tg64 **2.49 ± 0.00** tok/s, peak RSS **1314
MiB** (`final-2b_20261022_131607.json`). Only 4 threads (0.8B: t = 3 and 4 within 0.2%).

**0.8B on the same binary, same session** (chat27, 2 rounds, same file): plain 7.83, k = 2
10.98 (1.75 tokens / pass), k = 3 10.31, prefill 36.5 tok/s, peak RSS 430 MiB.

### Speed / quality table (Pi 4, 4 threads, 64 generated tokens, medians, all valid)

| Engine, model | Weights | Plain decode (prompt64 / chat27) | Speculative (prompt64 / chat27) | Prefill (prompt64) | Peak RSS | PPL c8 |
|---|---|---|---|---|---|---|
| **pinfer, 2B Q4K** | 1.09 GB | **3.41 / 3.42** | k=2 **4.35 / 4.82**; k=3 4.16 / 4.79 | **14.6** | 1040 MiB (1080 spec) | **13.38** |
| llama.cpp, 2B Q4_K_M | 1.31 GB | 2.49 (tg64) | — | 5.99 (pp64) | 1314 MiB | 13.32 |
| pinfer, 0.8B Q4K | 0.44 GB | 7.89 / 7.83 | k=3 18.43 / 10.31; k=2 — / 10.98 | 37.1 | 433 / 430 MiB | 18.48 |
| llama.cpp, 0.8B Q4_K_M | 0.54 GB | 5.41 (tg64) | — | 14.8 (pp64) | — | 18.59 |

0.8B prompt64 rows from the merged build (`speed-ab-merge4_20261022_103609.json`, notes.md
scoreboard), chat27 rows from `final-2b_20261022_131607.json`; llama.cpp 0.8B from
`llamacpp-qwen35-0.8b-Q4_K_M_20261021_213929.json`; PPL from `ppl-c8-2b-q4k_…`,
`pinfer-ppl-c8-q4k-headq4k_20261022_001936.json` and the PC runs above.

- 2B vs llama.cpp 2B: plain decode **1.37×**, speculative **1.75–1.94×**, prefill **2.4×**, at
  equal quality (13.38 vs 13.32) and 21% less memory.
- 2B vs 0.8B: 2.50× the bytes per token, 0.43× the plain speed (the bigger matrices stream
  better), 0.39× the prefill, 0.44× on chat with k = 2 (4.82 vs 10.98), and 28% lower perplexity.

## 6. Recommendation

**Practical as an opt-in quality model, not as the default.** It fits easily (1040–1081 MiB peak
RSS of 3794 MiB; 32k tokens of context would still fit), it is exact (golden, MTP head and
speculative == plain on host and Pi), and it is clearly better text (perplexity 13.4 vs 18.5).
But a chat answer streams at **4.8 tok/s** (k = 2) against 11.0 for 0.8B, plain decoding is 3.4
tok/s, and prefill at 10–15 tok/s (10.2 in 512-token chunks) makes a 500-token prompt cost
~35–50 s before the first token (0.8B at 24.6–37 tok/s: ~14–20 s). Use 2B where answer quality matters more than latency (short prompts, offline or
batch generation); keep 0.8B as the interactive default. Against llama.cpp on the same 2B
weights the engine's lead holds (1.4–1.9× decode, 2.4× prefill).

What would make 2B faster, in order of expected gain:
1. **More tokens per pass.** The verify pass is nearly free on 2B (3 rows cost 1.08× a plain
   step, 4 rows 1.14–1.18×), but the MTP head drafts only 1.6–2.0 tokens per pass on the Pi
   (fp32 reference: E[accepted] at k = 2 of 1.31, i.e. ~2.3 tokens per pass). Cheap extra verify
   rows favour multi-candidate drafts (the top-2 MTP candidates for the first position as extra
   rows) and prompt-lookup / n-gram drafts on repetitive text. A larger k does not help (k = 3 ≤
   k = 2 on both prompts).
2. **Fewer bytes per step.** Of 1060 MB, the MLP matrices are 510 MB (48%) and the tied head
   286 MB (27%). A 3-bit k-quant for the MLP alone would save ~120 MB (≈ +12% decode); the
   quality cost has to be measured first (opt-in, like every output-changing format).
3. **Prefill** (compute-bound): `kern_act_group` = 16 was tuned for 1024-column activations
   (16 KB in L1); 2B's 2048 / 6144 columns make a group 32 / 96 KB against a 32 KB L1D, so smaller
   groups for wide matrices may pay off (kernels agent's knob).

## Files

- New: `engine/MODEL2B.md`, `tests/test_dims.py`, `bench/model2b/{speed.sh, spec_exact_2b.sh,
  llama_rss.sh, ppl_pc_c8.sh}`.
- Changed: `engine/src/pinfer.c` (MTP scratch rows, §2a), `tools/convert.py` (row blocks, §3),
  `ref/mtp_reference.py` (`--name`), `tests/test_golden.py`, `tests/test_spec.py` (`--golden`).
- Gitignored data (main repo): `models/Qwen3.5-2B/`, `models/pq/qwen35-2b-{bf16,q4k-headq4k,
  q4k-mtpq8}.pq`, `models/gguf/Qwen3.5-2B-{bf16,Q8_0,Q4_0,Q4_K_M}.gguf` + `.SHA256SUMS`,
  `ref/golden/qwen35-2b{,-mtp}/`.
- On the Pi (kept, `docs/pi-state.md`): `models/qwen35-2b-q4k-headq4k.pq`,
  `models/Qwen3.5-2B-Q4_K_M.gguf`, `engine/pinfer_{run,ppl}_2b1` — 2.3 GB. Removed: the Q8-MTP
  experiment file and `tmp/m2b/`.

## Timeline
| time (KST) | event | evidence |
|---|---|---|
| 10:50 | Start; read config of `Qwen/Qwen3.5-2B` | HF API |
| 10:54–10:56 | Download, sha256 | `models/Qwen3.5-2B` |
| 10:57–10:58 | fp32 golden reference | `ref/golden/qwen35-2b/meta.json` |
| 10:58–11:03 | bf16 and Q4K conversion | §3 |
| 10:59–11:01 | Host golden test 2B: PASS | §4 |
| 11:02–11:03 | `test_dims.py`: ASan overflow in the MTP scratch rows, fixed in `pinfer.c`, 3/3 PASS | §2a |
| 11:02–11:18 | MTP reference (acceptance) | §4 |
| 11:05–11:25 | 0.8B host gates on the fixed engine: golden / spec / state PASS, pytest 126, malformed 43/43, dims 3/3 | scratchpad `m2b-logs/gates08_110511/` |
| 11:05:32–11:06:10 | Model + binaries pushed in a held lock window, md5s checked on the Pi | `logs/harness.jsonl` |
| 11:07–11:14 | Pi 0.8B gates with `pinfer_run_2b1`: spec_exact 20/20, ctx_check 5/5 | `gates08-2b1_20261022_111351.json` |
| 11:15–11:20 | Mislabeled run (0.8B spec_exact again, own script overwritten on the Pi) | `spec-exact-2b1_20261022_112006.json` |
| 11:06–11:17 | llama.cpp GGUFs | §5 |
| 11:20–11:40 | Host `test_spec.py` on 2B: PASS | §4 |
| 11:22–11:28 | llama.cpp perplexity on the PC | §5 |
| 11:27:31–11:28:14 | GGUF pushed in a held lock window | `logs/harness.jsonl` |
| 11:36–11:40 | Pi spec_exact 2B 10/10 | `spec-exact-2b-q4k_20261022_114018.json` |
| 11:45–11:50 | prompt64 speed | `speed-2b-q4k-p64_20261022_114956.json` |
| 11:50–12:01 | Host bf16 acceptance on prompt64 / chat27 | §5 |
| 11:59–12:06 | Converter row blocks: byte-identical, 17.2 → 8.4 GB; Q8-MTP variant converted | §3 |
| 12:17–12:21 | chat27 speed | `speed-2b-q4k-chat27_20261022_122132.json` |
| 12:21–12:28 | Pi perplexity 13.375 | `ppl-c8-2b-q4k_20261022_122815.json` |
| 12:52:34–12:52:58 | Q8-MTP variant pushed in a held lock window | `logs/harness.jsonl` |
| 13:10–13:16 | Q8 MTP, llama.cpp 2B, 0.8B chat27 | `final-2b_20261022_131607.json` |
| 13:17 | Pi scratch removed | `logs/harness.jsonl` |
