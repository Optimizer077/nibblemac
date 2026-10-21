# pinfer library guide (v0.1.0)

pinfer runs **Qwen3.5-0.8B** on a Raspberry Pi 4. It ships as
- a C library, `libpinfer.so` / `libpinfer.a`, with one header, `pinfer.h`. Token ids go in, logits come out;
- a Python package, `pinfer` (ctypes binding, HF `tokenizers`, sampling, chat template, streaming);
- a command, `pinfer` (`generate`, `chat`, `bench`, `convert`, `info`).

> **Status 0.1.0 (2026-10-22):** the aarch64 wheel `pinfer-0.1.0-py3-none-manylinux_2_34_aarch64.whl`
> contains the NEON kernels (`MATVEC=fast`); the x86-64 wheel contains the scalar reference kernels.
> Recommended model file: `qwen35-0.8b-q4k-headq4k.pq` (`pinfer convert ... --quant q4k`, 437 MB):
> 7.9 tok/s plain greedy, 16.5 tok/s with `--speculative --spec-k 3`, 37 tok/s prefill, perplexity 18.48 (bf16: 17.93). See §7.
> Not published anywhere yet; no license chosen.

---

## 1. Install

### Raspberry Pi (aarch64, Raspberry Pi OS trixie / any glibc ≥ 2.33)
The Pi's Wi-Fi is unreliable, so download the wheels on a PC and copy them over:
```bash
# on the PC (any OS with pip): the pinfer wheel + its dependencies for CPython 3.13 aarch64
pip download --dest pi-wheels --python-version 3.13 --implementation cp --only-binary=:all: \
    --platform manylinux_2_28_aarch64 --platform manylinux_2_27_aarch64 \
    --platform manylinux_2_17_aarch64 --platform manylinux2014_aarch64 \
    numpy tokenizers jinja2            # + pytest for the test suite
cp build/dist/pinfer-0.1.0-py3-none-manylinux_2_34_aarch64.whl pi-wheels/
# copy pi-wheels/ to the Pi, then on the Pi:
python3 -m venv ~/venvs/pinfer
~/venvs/pinfer/bin/pip install --no-index --find-links pi-wheels "pinfer[chat]"
```
Give `--platform` once for each manylinux tag. A single `manylinux_2_28` tag does not match wheels such as `tokenizers` that are tagged `manylinux_2_17`.

### PC / x86-64 Linux (for development and tests)
```bash
pip install build/dist/pinfer-0.1.0-py3-none-manylinux_2_33_x86_64.whl "jinja2"
```
Windows cannot load the Linux `.so`, so use WSL. `import pinfer` and `pinfer convert` still work on Windows, because the library is loaded lazily.

### From source
```bash
make -C engine lib                       # build/{host,aarch64}/engine/lib/libpinfer.{a,so}
pip install -e python                    # editable: finds the library in build/ automatically
make -C engine wheel-host                # build/dist/pinfer-*-manylinux_*_x86_64.whl
make -C engine wheel-pi                  # build/dist/pinfer-*-manylinux_*_aarch64.whl
```
The wheel targets need a Python that has `hatchling` installed (`WHEEL_PY`, default `/root/venv-pinfer-build/bin/python`).

### C library
```bash
make -C engine install PREFIX=/opt/pinfer                    # host
make -C engine install TARGET=pi PREFIX=/usr/local DESTDIR=/tmp/pi-stage   # Pi tree to copy over
```
This installs `include/pinfer.h`, `lib/libpinfer.a`, `lib/libpinfer.so.0.1.0` (with the symlinks `libpinfer.so.0.1` and `libpinfer.so`) and `lib/pkgconfig/pinfer.pc`.

---

## 2. Convert a model
```bash
pinfer convert models/Qwen3.5-0.8B models/pq/qwen35-0.8b-bf16.pq      # [--no-mtp] [--no-tokenizer]
```
This command runs `tools/convert.py`, which is bundled in the wheel as `pinfer/_tools/convert.py`. It writes the `.pq` file (format: `engine/FORMAT.md`) and copies `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` into the sidecar directory `models/pq/qwen35-0.8b-bf16.tokenizer/`. `Model.load` then finds the tokenizer without a `tokenizer=` argument.

Keep the MTP head (do not pass `--no-mtp`) if you want speculative decoding.

**Tokenizer search order** for `model.pq`:
1. the `tokenizer=` argument (a directory or a `tokenizer.json` path);
2. `$PINFER_TOKENIZER`;
3. the sidecar `model.tokenizer`: either a directory, or a text file whose first line is a path, relative to the `.pq` directory;
4. `tokenizer.json` in the `.pq` directory.

If none of these exists, the model loads for token-id use only (`iter_tokens`, `eval`), and a warning is logged.

**Library search order:**
1. `Model.load(..., lib_path=...)`;
2. `$PINFER_LIB`;
3. `libpinfer.so` inside the package (the wheel);
4. `build/{host,aarch64}/engine/lib/` of a source checkout;
5. the system loader (`LD_LIBRARY_PATH`, ldconfig, `make install` prefixes).

---

## 3. Python
```python
import pinfer

with pinfer.Model.load("qwen35-0.8b-bf16.pq", tokenizer="models/Qwen3.5-0.8B",
                       n_threads=4, max_ctx=4096) as m:
    r = m.generate("The Raspberry Pi 4 is a small computer that", max_new_tokens=32)
    print(r.text, r.finish_reason, r.decode_tps)          # greedy by default

    for chunk in m.generate("Once upon a time", 64, temperature=0.8, top_p=0.95,
                            seed=1, stop=["\n\n"], stream=True):
        print(chunk, end="", flush=True)

    reply = m.chat([{"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "What is a Raspberry Pi?"}],
                   max_new_tokens=128)                    # Qwen chat template, no <think>
    print(reply.text)

    ids = m.tokenize("Hello")                             # token-level access
    logits = m.eval(ids)                                  # numpy float32 [n_vocab]
    for tok in m.iter_tokens(ids, max_new_tokens=8):      # no tokenizer needed
        ...

    if m.supports_speculative:                            # MTP speculative greedy decoding
        r = m.generate("The Raspberry Pi 4 is", 64, speculative=True, spec_k=3)
        print(r.spec_stats)                               # rounds / drafted / accepted
```
A complete script is in `examples/python/generate.py`.

### API reference
| Item | Description |
|---|---|
| `Model.load(path, tokenizer=None, n_threads=4, max_ctx=4096, lib_path=None, kv_dtype="fp32")` | Map a `.pq` file. `tokenizer`: a `Tokenizer`, a path, `None` (search), or `False` (none). `kv_dtype`: `"fp32"` (exact), `"fp16"` or `"int8"` KV cache (§9). |
| `model.generate(prompt, max_new_tokens=128, *, temperature=0, top_p=1, top_k=0, seed=None, stop=None, stop_token_ids=None, stream=False, reuse_prefix=True, speculative=False, spec_k=None, draft_vocab=None, spec_adaptive=None)` | `prompt` is text or token ids. Returns a `GenerationResult`, or a `TextStream` when `stream=True`. `spec_adaptive=True` lets the engine lower `spec_k` when that is faster (§9). |
| `model.save_state()` / `model.load_state(data)` | Snapshot / restore the whole context (recurrent + conv states, KV cache, MTP cache, tokens, last logits) as `bytes`; continuation is bit-exact. ~20 MB + 24 KB per token. §9. |
| `model.context_tokens` / `model.supports_state` | Token ids currently in the engine's context / library has the state API. |
| `model.chat(messages, max_new_tokens=512, *, enable_thinking=False, tools=None, **generate_kwargs)` | Applies the chat template, then calls `generate`. |
| `model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tools=None, use_jinja=None)` | Returns the prompt text. |
| `model.iter_tokens(prompt_ids, max_new_tokens=128, sampling=None, stop_token_ids=None, reuse_prefix=True)` | Token-level generator. |
| `model.eval(tokens)` / `model.reset()` | Appends tokens and returns the last token's logits (a copy) / clears the context. |
| `model.tokenize(text)` / `model.detokenize(ids, skip_special_tokens=False)` | Tokenizer shortcuts. |
| `model.info()` | Library path, version, build (`matvec`, compiler), model facts. |
| `model.n_vocab`, `eos_id`, `n_past`, `max_ctx`, `n_threads`, `kv_dtype`, `has_mtp`, `supports_speculative`, `version` | Properties. |
| `GenerationResult` | `text`, `token_ids`, `prompt_tokens`, `cached_tokens`, `finish_reason` (`stop` / `length` / `context`), `prefill_seconds`, `decode_seconds`, `prefill_tps`, `decode_tps`, `spec_stats`. |
| `TextStream` | Iterator of `str` chunks. `.result` is set once the stream is exhausted; `.close()` stops the generation. |
| `SamplingParams(temperature, top_k, top_p, seed)`, `Sampler` | numpy sampling. `temperature=0` or `top_k=1` means greedy (first maximum, like `torch.argmax`). |
| `Tokenizer.from_path(p)`, `.encode`, `.decode`, `.stop_token_ids()`, `.chat_template` | Wraps HF `tokenizers`. |
| `IncrementalDecoder` | Streams text without splitting UTF-8 characters. |
| `apply_chat_template`, `render_qwen_builtin` | Uses the Jinja template when `jinja2` is installed, otherwise a built-in Python port of the Qwen3.5 template (text, system, assistant, tool messages; tool definitions and tool calls need jinja2). |
| Exceptions | `PinferError` (base, `.code` = C error code), `LibraryNotFoundError`, `ModelLoadError`, `ModelNotFoundError` (also a `FileNotFoundError`), `ModelFormatError`, `ContextOverflowError`, `TokenizerNotFoundError`, `ModelClosedError`. |

**Behaviour notes**
- **Stops.** Generation stops at the model's EOS id and at Qwen's `<|im_end|>` and `<|endoftext|>`, plus any `stop_token_ids`. Stop strings cut the text before the first match; while streaming, text that could be the start of a stop string is held back until it is decided.
- **Prefix reuse (`reuse_prefix=True`).** If the new prompt extends the tokens already in the context, only the new tokens are evaluated — also for `speculative=True` (the library's `pinfer_generate_greedy_cont`). The engine's batched and token-by-token paths agree bit for bit, so the result equals a fresh run. `GenerationResult.cached_tokens` tells how many tokens were reused.
- **Context limit.** A prompt longer than `max_ctx` raises `ContextOverflowError`. When the context fills during generation, the result ends with `finish_reason="context"`.
- **Threads.** A `Model` is not thread-safe; use one per thread. The weights are memory-mapped, so several models share the page cache. ctypes releases the GIL during C calls.
- **Speculative decoding** (`speculative=True`) calls `pinfer_generate_greedy`. The output is the same tokens as plain greedy decoding. The call is blocking, so a stream delivers its text only after the call returns. The C function stops only at EOS: other stop tokens are applied afterwards, so any tokens generated past them are wasted work. See the proposals in §7.

### Command line
```bash
pinfer generate -m model.pq -p "The Raspberry Pi 4 is" -n 64 [--temperature 0.7 --top-p 0.9 --seed 1] [--stop "\n\n"] [--stats] [--speculative --spec-k 3]
echo "prompt from stdin" | pinfer generate -m model.pq
pinfer chat -m model.pq [--system "Be brief."] [--think]      # interactive; /reset, /exit; keeps the state between turns (--reprefill: re-render every turn)
pinfer chat -m model.pq -p "One question, one answer"
pinfer bench -m model.pq --prompt-tokens 64 -n 32 [--repeat 3] [--spec-k 3] [--json]
pinfer convert models/Qwen3.5-0.8B out.pq [--no-mtp] [--no-tokenizer]
pinfer info -m model.pq                                       # JSON: library, build, model
```
Common options: `--tokenizer DIR`, `-t/--threads`, `-c/--ctx`, `--kv-dtype fp32|fp16|int8`, `--lib PATH`, `-v` (log to stderr); `--adaptive` with `--speculative`. `python -m pinfer` works the same as `pinfer`.

---

## 4. C
```c
#include "pinfer.h"
pinfer_params p = pinfer_default_params();          /* n_threads 4, max_ctx 4096 */
int err;
pinfer_ctx *ctx = pinfer_load("model.pq", &p, &err);
float *logits = malloc(sizeof(float) * pinfer_n_vocab(ctx));
pinfer_eval(ctx, prompt_ids, n_prompt, logits);     /* prefill; logits of the last token */
int32_t tok = pinfer_argmax(logits, pinfer_n_vocab(ctx));
pinfer_eval(ctx, &tok, 1, logits);                  /* one decode step */
pinfer_free(ctx);
```
A complete program is in `examples/c/simple.c`. `make -C engine examples` builds it for the host (dynamically linked against `libpinfer.so`) and for the Pi (statically linked against `libpinfer.a`).
```bash
cc app.c $(pkg-config --cflags --libs pinfer)                 # shared
cc app.c -I/opt/pinfer/include /opt/pinfer/lib/libpinfer.a -lm -pthread   # static
```

| Function | Notes |
|---|---|
| `pinfer_default_params`, `pinfer_load`, `pinfer_free` | `pinfer_load` returns NULL on failure, with the reason in `*err`. |
| `pinfer_eval(ctx, tokens, n, logits)` | `logits` may be NULL. Returns `PINFER_E_CTX_FULL` or `PINFER_E_ARG` without changing state. |
| `pinfer_reset`, `pinfer_n_past`, `pinfer_n_vocab`, `pinfer_eos_id`, `pinfer_argmax`, `pinfer_version` | |
| `pinfer_has_mtp`, `pinfer_default_spec_params`, `pinfer_generate_greedy`, `pinfer_generate_greedy_ex` | Speculative greedy decoding with the MTP head (see `engine/src/spec.c`). `pinfer_spec_params.adaptive = 1`: k is an upper bound, the engine picks the fastest k per round (§9). |
| `pinfer_generate_greedy_cont(...)` | `pinfer_generate_greedy_ex` without the reset: the prompt is appended to the context (after `pinfer_eval`, `pinfer_state_load` or an earlier generation, whose last returned token is not yet in context and should lead the prompt). |
| `pinfer_state_size(ctx)`, `pinfer_state_save(ctx, buf, cap)`, `pinfer_state_load(ctx, buf, len)` | Snapshot / restore the context (recurrent + conv states, KV rows, MTP rows). Blob tied to the model file and `kv_dtype`; `E_FORMAT` otherwise, `E_CTX_FULL` if it holds more tokens than `max_ctx`. |
| `pinfer_params.kv_dtype` | `PINFER_KV_F32` (0, default), `PINFER_KV_F16`, `PINFER_KV_I8`: KV-cache storage (§9). |

Error codes: `PINFER_OK` 0, `E_IO` −1, `E_FORMAT` −2, `E_NOMEM` −3, `E_ARG` −4, `E_CTX_FULL` −5.

**ABI note (2026-10-22).** `pinfer_params` gained `kv_dtype` and `pinfer_spec_params` gained `adaptive` (both at the end, zero = the old behaviour). Callers that zero-initialise the structs through `pinfer_default_params` / `pinfer_default_spec_params` are source-compatible; binaries built against the older header must be rebuilt (0.x: any minor release may change the ABI).

**ABI.** The shared library exports only `pinfer_*` symbols (linker version script `engine/pkg/pinfer.map`). In the static archive, every other global symbol is made local (`ld -r` followed by `objcopy --keep-global-symbol`). While the version is 0.x, the SONAME is `libpinfer.so.MAJOR.MINOR`, because any minor release may change the ABI. `make -C engine check-lib` verifies the exports, the NEEDED libraries and the glibc symbol versions.

---

## 5. Wheels and glibc compatibility
- Wheel tag: `py3-none-manylinux_2_<N>_<arch>`. `python/hatch_build.py` reads both values from the bundled `.so`: the architecture from the ELF header, and `N` from the newest `GLIBC_2.N` symbol version the library references. pip therefore refuses the wheel on a system whose glibc is too old, instead of failing at `dlopen`.
- The WSL cross toolchain links against glibc 2.43, newer than the Pi's 2.41. Library objects are compiled with `-fno-math-errno`. Without that flag GCC emits a fallback `sqrtf` call, versioned `sqrtf@GLIBC_2.43`, which the Pi cannot load. The engine never reads `errno`, so results are bit-identical. `make check-lib` fails if the Pi `.so` needs anything newer than `PI_GLIBC_MAX` (default 2.41).
- Current needs: `MATVEC=ref` needs `fstat@GLIBC_2.33`, so the tag is `manylinux_2_33`. `MATVEC=fast` adds `pthread_*@GLIBC_2.34`, so the tag becomes `manylinux_2_34`.
- Dependencies: `libc.so.6`, `libm.so.6` and the dynamic loader only.

## 6. Tests
```bash
# host (WSL), against an installed wheel
PINFER_TEST_REQUIRE=1 python -m pytest tests/python -q
# Pi
PINFER_TEST_REQUIRE=1 PINFER_TEST_MODEL=~/pinfer/models/qwen35-0.8b-bf16.pq \
  PINFER_TEST_TOKENIZER=~/pinfer/models/qwen35-0.8b-tokenizer \
  PINFER_TEST_GOLDEN=~/pinfer/golden/qwen35-0.8b/meta.json \
  ~/pinfer/venvs/pinfer/bin/python -m pytest tests/python -q
```
The suite covers:
- golden greedy ids and text from `ref/golden/qwen35-0.8b/meta.json`, through `generate`, `iter_tokens`, prefix reuse and speculative decoding;
- tokenizer round trips and incremental decoding;
- the built-in chat template compared with the Jinja template;
- sampling statistics;
- streaming and stop strings;
- error handling: missing or corrupt model, invalid token, context overflow, closed model, missing tokenizer;
- the CLI.

`PINFER_TEST_REQUIRE=1` turns a missing model, tokenizer or golden file into a failure instead of a skip.

## 7. Performance notes
All numbers: Raspberry Pi 4 (4 GB, Cortex-A72 @ 1.5 GHz), 4 threads, 64-token prompt, 64 generated
tokens, stock Raspberry Pi OS kernel, taken through `harness/pi.py bench` (the full table with
result-file paths is the "Scoreboard" in `notes.md`).

| Model file (`pinfer convert` option) | Size | Decode tok/s plain | Decode tok/s `--speculative` | Prefill tok/s | Perplexity (wikitext-2) |
|---|---|---|---|---|---|
| `--quant q4k` (**recommended**) | 437 MB | 7.9 | 16.5 (k=3) / 14.5 (k=2) | 37 | 18.48 |
| `--quant q8` | 823 MB | 4.2 | 8.7 | 12 | 17.96 |
| bf16 (default) | 1.55 GB | 2.3 | 4.1 | 11 | 17.93 |
| llama.cpp v0.7.0 Q4_K_M, same Pi | 542 MB | 5.4 | — | 15 | 18.59 |
| MNN 3.6 int4, same Pi | ~450 MB | 6.7 | — | 31 | — |

- Decode is memory-bandwidth-bound (one Cortex-A72 core already saturates the ~4.2 GB/s DRAM);
  tok/s ≈ 4.2 GB/s ÷ bytes of weights per token, minus overheads. Speculative decoding with the
  model's MTP head gets past that ceiling by verifying several drafted tokens per weight pass.
- `--speculative` defaults: k = 2 drafts, draft vocabulary 32768 (drafts only consider the first
  32768 token ids; on varied text this lowers acceptance somewhat — 2.86 → 2.37 tokens per pass at k=3 —
  but makes each draft step ~5× cheaper, which was faster overall on all three test prompts).
  `--spec-k 3` is ~14% faster than k=2 on predictable text. At long contexts (≥ 2K tokens) the gain
  shrinks or reverses on unpredictable text; an adaptive policy is in progress (Phase 6).
- Exactness: speculative output is token-identical to plain greedy, with both the scalar and the
  NEON kernels (the batched verify pass is bit-identical to the single-token pass; checked by
  `engine/bench/spec_exact.sh` on the Pi, 20/20).
- Python overhead is negligible: `pinfer bench` (Python API) gives the same tok/s as `pinfer_run --bench` (C).
- `--quant q4` (symmetric 4-bit) is as fast as `q4k` but loses quality (perplexity 21.6, like llama.cpp Q4_0); prefer `q4k`.

## 8. Known limitations and proposed C API additions
- No sdist: a source distribution would need the C sources and a compiler on the target.
- `pinfer_generate_greedy_ex` (added 2026-10-21) covers stop tokens, partial output on context overflow and a per-token callback; the Python package uses it when present and falls back to `pinfer_generate_greedy` otherwise.
- There is no way to ask the library which kernels it was built with. Proposed: `const char *pinfer_build_info(void)`. Until then Python reads the `BUILD_INFO.json` that sits next to the `.so`.
- The library prints diagnostics with `fprintf(stderr)` / `perror`. Proposed: `void pinfer_set_log_callback(void (*cb)(int level, const char *msg, void *user), void *user)`.
- The tokenizer path is not stored in the `.pq` file. Proposed: an optional key-value metadata section in the format, and `const char *pinfer_meta_get(const pinfer_ctx *, const char *key)`. The sidecar directory covers this for now.
- ~~State save and restore~~ — added 2026-10-22 (`pinfer_state_size/save/load`, §9).

## 9. Context window (2026-10-22, details in `engine/CONTEXT.md`)
The 6 full-attention layers keep a KV cache of 24 KB per token (fp32); the 18 linear-attention
layers have a fixed 19 MB state, so memory grows slowly (625 MB RSS at 8192 tokens) and what
costs is the attention's pass over the cache. Pi 4, Q4K model, 4 threads, 32 generated tokens:

| context | plain decode tok/s (fp32 / fp16 KV) | prefill tok/s | speculative k=3 tok/s (fixed / adaptive) | llama.cpp Q4_K_M decode |
|---|---|---|---|---|
| 64 | 7.9 / 7.9 | 37 | 16.8 / 16.4 (easy text) | 5.5 |
| 512 | 7.6 / 7.7 | 37 | 15.2 | 5.2 |
| 2048 | 7.1 | 32 | 9.1 / **9.6** | 4.3 |
| 4096 | 6.4 / 6.7 | 28 | 7.4 / **7.5** | 3.4 |
| 8192 | 5.3 / 5.7 | 22 | 7.2 | — |

- **KV cache precision** (`kv_dtype`): `fp16` halves the cache (−97 MB at 8192 tokens), matches
  fp32 quality within noise for this model (perplexity 13.983 vs 13.985 on 8×2048 wikitext tokens,
  top-1 100% teacher-forced) and is ~5–8% faster at 4096–8192 tokens; `int8` cuts the cache by
  73% with perplexity still within noise. Both change greedy output at the first near-tie, so
  fp32 stays the default. Use `fp16` for long contexts or several models per Pi.
- **State snapshots** (`save_state` / `load_state`, C `pinfer_state_*`): prefill a system
  prompt or a document once, then restore it for every conversation; continuation is
  bit-exact. A 2048-token state is 70.5 MB and restores in 0.1 s (prefilling it takes 63 s).
  `pinfer chat` keeps the state between turns and only prefills the new turn (second turn of a
  short chat: 0.7 s instead of 2.1 s before the first token).
- **Adaptive speculation** (`spec_adaptive=True`, `--adaptive`): the engine measures its pass
  costs and the draft acceptance and picks k (down to plain steps) per round; same output.
  It beats fixed k=3 where acceptance is low (long wikitext context: 9.6 vs 9.1 tok/s at 2048)
  and stays within ~2–5% of it where drafts are almost always accepted.
