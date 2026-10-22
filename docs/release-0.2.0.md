# pinfer 0.2.0

The first public release of pinfer, the Raspberry-Pi-4-native inference engine of
[piquant](https://github.com/Optimizer077/piquant), for the hybrid Qwen3.5 models. Apache-2.0.

## Highlights (Raspberry Pi 4, 4 GB, 1.5 GHz, Qwen3.5-0.8B Q4K, 4 threads)
| | 0.1.0 | 0.2.0 |
|---|---|---|
| Decode, plain greedy | 7.9 tok/s | 7.9 tok/s |
| Decode, speculative (exact), prose | 16.5 tok/s (k=3) | **20.2 tok/s** (default: adaptive k ≤ 4 + prompt-lookup drafts) |
| Decode, speculative, chat answers | — | **11.0–14.8 tok/s** |
| Prefill (prompt reading) | 37 tok/s | **43 tok/s** |
| Decode after a 4096-token context | 5.6 tok/s | **6.4** (fp32 KV) / **6.9** (fp16 KV) tok/s |
| Restore a 2048-token conversation | re-read: 63 s | **0.1 s** from a saved state |

Speculative output is token-for-token identical to plain greedy decoding in every setting. Every number
has a result file; sources in [RESULTS.md](RESULTS.md).

## What is new since 0.1.0
- **Speed:** a cheaper verify pass (the recurrent state is read and written once per round);
  prompt-lookup (n-gram) drafts next to the MTP head; a confidence-gated adaptive k; batched kernels
  with single register passes for up to 7 vectors and prefill groups that stay in L1.
- **Context window:** an attention rewrite (each K/V row read once per block and shared by the 4
  heads of a group), opt-in fp16 / int8 KV cache, state snapshots (`save_state` / `load_state`), a
  keep-state `pinfer chat`.
- **Server:** `pinfer serve`, an OpenAI-compatible HTTP server (chat and text completions, streaming);
  each turn of a conversation reads only the new message.
- **Models:** Qwen3.5-2B support (exact against the fp32 reference; 3.4 tok/s plain).
- **Robustness:** every `.pq` tensor and header field is validated (43 malformed-file cases refused
  cleanly under AddressSanitizer + UBSan); state blobs are tied to their model file; the thread pool
  no longer pins the caller's thread; a shared scratch buffer race is fixed.
- **API / ABI:** `pinfer_params.kv_dtype`, `pinfer_spec_params.{adaptive, ngram}`,
  `pinfer_state_size / save / load`, `pinfer_generate_greedy_cont`; SONAME `libpinfer.so.0.2`.
  Binaries built against 0.1.0 headers must be rebuilt. The speculative default changed from fixed
  k = 2 to adaptive k ≤ 4 with n-gram 7.

## Install
- **Raspberry Pi 4** (aarch64, glibc ≥ 2.34): `pip install pinfer-0.2.0-py3-none-manylinux_2_34_aarch64.whl`
  (NEON kernels).
- **x86-64 Linux** (glibc ≥ 2.33, development and tests; scalar reference kernels):
  `pip install pinfer-0.2.0-py3-none-manylinux_2_33_x86_64.whl`.
- Python ≥ 3.10; dependencies `numpy`, `tokenizers` (optional `jinja2`).
- The model is not bundled: download Qwen3.5-0.8B from Hugging Face (Apache-2.0) and convert it with
  `pinfer convert` or `tools/convert.py` (README "Quickstart"; full steps in [REPRODUCE.md](REPRODUCE.md)).

## Known limits
- One generation at a time per model; `pinfer serve` queues concurrent requests.
- fp16 / int8 KV caches can change greedy output at near-ties (quality is unchanged within noise), so
  fp32 stays the default.
- No PyPI release yet.
