# pinfer library guide (v0.2.0)

pinfer runs **Qwen3.5-0.8B** on a Raspberry Pi 4. It ships as
- a C library, `libpinfer.so` / `libpinfer.a`, with one header, `pinfer.h`. Token ids go in, logits come out;
- a Python package, `pinfer` (ctypes binding, HF `tokenizers`, sampling, chat template, streaming);
- a command, `pinfer` (`generate`, `chat`, `bench`, `convert`, `info`, `serve`);
- an OpenAI-compatible HTTP server, `pinfer serve` (§10).

> **Status 0.2.0 (unreleased, 2026-10-22):** the aarch64 wheel `pinfer-0.2.0-py3-none-manylinux_2_34_aarch64.whl`
> contains the NEON kernels (`MATVEC=fast`); the x86-64 wheel contains the scalar reference kernels.
> Recommended model file: `qwen35-0.8b-q4k-headq4k.pq` (`pinfer convert ... --quant q4k`, 437 MB):
> 7.9 tok/s plain greedy, 16.5 tok/s with `--speculative --spec-k 3`, 37 tok/s prefill, perplexity 18.48 (bf16: 17.93). See §7.
> Source on GitHub under Apache-2.0 (`LICENSE`); not on PyPI yet.

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
cp build/dist/pinfer-0.2.0-py3-none-manylinux_2_34_aarch64.whl pi-wheels/
# copy pi-wheels/ to the Pi, then on the Pi:
python3 -m venv ~/venvs/pinfer
~/venvs/pinfer/bin/pip install --no-index --find-links pi-wheels "pinfer[chat]"
```
Give `--platform` once for each manylinux tag. A single `manylinux_2_28` tag does not match wheels such as `tokenizers` that are tagged `manylinux_2_17`.

### PC / x86-64 Linux (for development and tests)
```bash
pip install build/dist/pinfer-0.2.0-py3-none-manylinux_2_33_x86_64.whl "jinja2"
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
This installs `include/pinfer.h`, `lib/libpinfer.a`, `lib/libpinfer.so.0.2.0` (with the symlinks `libpinfer.so.0.2` and `libpinfer.so`) and `lib/pkgconfig/pinfer.pc`.

---

## 2. Convert a model
```bash
pinfer convert models/Qwen3.5-0.8B models/pq/qwen35-0.8b-bf16.pq      # [--no-mtp] [--no-tokenizer]
```
This command runs `tools/convert.py`, which is bundled in the wheel as `pinfer/_tools/convert.py`. It writes the `.pq` file (format: `engine/FORMAT.md`) and copies `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` into the sidecar directory `models/pq/qwen35-0.8b-bf16.tokenizer/`. `Model.load` then finds the tokenizer without a `tokenizer=` argument.

Keep the MTP head (do not pass `--no-mtp`) if you want speculative decoding.

**From a llama.cpp GGUF.** A Qwen3.5 GGUF (arch `qwen35`) with Q4_0 / Q8_0 / F32 tensors converts
without re-quantization: `python tools/gguf_to_pq.py model.gguf out.pq --tokenizer HF_DIR --verify`
(Q4_0 → Q4_G32, Q8_0 → Q8_G32 bit for bit, llama.cpp's V-head reordering undone, an untied
`output.weight` kept as `lm_head.weight`; details and the exactness check: `engine/FORMAT.md`,
"Importing a llama.cpp GGUF"). This is the path for models trained for the Q4_0 grid; fine-tuned
Qwen3.5-4B decision models distributed that way, such as Cerno 4B, run with it on the Pi.

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
| `model.generate(prompt, max_new_tokens=128, *, temperature=0, top_p=1, top_k=0, seed=None, stop=None, stop_token_ids=None, stream=False, reuse_prefix=True, speculative=False, spec_k=None, draft_vocab=None, spec_adaptive=None, spec_ngram=None)` | `prompt` is text or token ids. Returns a `GenerationResult`, or a `TextStream` when `stream=True`. `spec_adaptive=True` lets the engine lower `spec_k` when that is faster (§9). `spec_ngram=N` (0..8) adds prompt-lookup drafts of up to N tokens where the answer repeats the context (§9, `engine/DRAFT.md`). |
| `model.save_state()` / `model.load_state(data)` | Snapshot / restore the whole context (recurrent + conv states, KV cache, MTP cache, tokens, last logits) as `bytes`; continuation is bit-exact. ~20 MB + 24 KB per token. A state loads only into a model of the same `.pq` file and `kv_dtype`: otherwise `ModelFormatError`, naming what differs (`pinfer.state`). §9. |
| `model.context_tokens` / `model.supports_state` | Token ids currently in the engine's context / library has the state API. |
| `model.checkpoint()` / `model.rewind()` / `model.clear_checkpoint()` | In-context rewind point (C `pinfer_checkpoint` / `pinfer_rewind`): `checkpoint()` copies only the recurrent + conv states (0.8B 20 MB, 4B 53 MB, whatever the context length) and the tokens / last logits; `rewind()` returns to it bit-exactly, any number of times; the KV cache stays in place. For many continuations of one prefix (questions about one document). One per model; dropped by `reset()`, `load_state()` and a generation that resets the context; kept by `eval`, `eval_select` and generations that extend the context. `supports_checkpoint`, `checkpoint_n_past` (None without one). §9. |
| `model.chat(messages, max_new_tokens=512, *, enable_thinking=False, tools=None, **generate_kwargs)` | Applies the chat template, then calls `generate`. |
| `model.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tools=None, use_jinja=None)` | Returns the prompt text. |
| `model.iter_tokens(prompt_ids, max_new_tokens=128, sampling=None, stop_token_ids=None, reuse_prefix=True)` | Token-level generator. |
| `model.eval(tokens)` / `model.reset()` | Appends tokens and returns the last token's logits (a copy) / clears the context. |
| `model.eval_select(tokens, ids)` | Like `eval`, but returns only the logits of the vocabulary `ids` (float32 [len(ids)]), bit-identical to `eval(tokens)[ids]`; only the head rows holding them are computed (`pinfer_eval_select`). For readouts over a few label tokens. `ids` may be empty (feed only). |
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
- **CPU affinity.** The library never changes the calling thread's affinity. The NEON build pins its `n_threads - 1` worker threads to the 2nd..n-th CPU of the caller's mask (CPUs 1–3 normally) and uses at most as many threads as that mask has CPUs. Up to 0.1.0 it pinned the thread that called `pinfer_load` to CPU 0, and every thread the application created afterwards inherited that single core.
- **Profiling.** `PF_FWD_STATS=1` / `PF_KERN_STATS=1` print phase and matmul tables at exit. They are process-wide (mutex-guarded since 0.2.0): with several models on several threads the tables sum all of them, so profile one at a time.
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
pinfer serve -m model.pq [--host 0.0.0.0] [--port 8080]       # OpenAI-compatible HTTP server (§10)
```
Common options: `--tokenizer DIR`, `-t/--threads`, `-c/--ctx/--max-ctx`, `--kv-dtype fp32|fp16|int8`, `--lib PATH`, `-v` (log to stderr); `--adaptive` with `--speculative`. `python -m pinfer` works the same as `pinfer`.

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
| `pinfer_eval_select(ctx, tokens, n, ids, n_ids, out)` | `pinfer_eval` that writes the last token's logits of the vocabulary ids `ids[0..n_ids-1]` only (`out[i]` = `logits[ids[i]]` bit for bit); reads just the 4-row head tiles that hold them. |
| `pinfer_reset`, `pinfer_n_past`, `pinfer_n_vocab`, `pinfer_eos_id`, `pinfer_argmax`, `pinfer_version` | |
| `pinfer_has_mtp`, `pinfer_default_spec_params`, `pinfer_generate_greedy`, `pinfer_generate_greedy_ex` | Speculative greedy decoding with the MTP head (see `engine/src/spec.c`). `pinfer_spec_params.adaptive = 1`: k is an upper bound, the engine picks the fastest k per round (§9). `pinfer_spec_params.ngram = N`: prompt-lookup drafts of up to N tokens; `pinfer_spec_stats.ngram_*` count them. |
| `pinfer_generate_greedy_cont(...)` | `pinfer_generate_greedy_ex` without the reset: the prompt is appended to the context (after `pinfer_eval`, `pinfer_state_load` or an earlier generation, whose last returned token is not yet in context and should lead the prompt). |
| `pinfer_state_size(ctx)`, `pinfer_state_save(ctx, buf, cap)`, `pinfer_state_load(ctx, buf, len)` | Snapshot / restore the context (recurrent + conv states, KV rows, MTP rows). The blob (state format 2) carries a 64-bit fingerprint of the model file (its header, tensor directory and sampled weights) and is tied to it and to `kv_dtype`: `E_FORMAT` for another file, KV format or format version (0.1 blobs included), `E_CTX_FULL` if it holds more tokens than `max_ctx`. |
| `pinfer_checkpoint(ctx)`, `pinfer_rewind(ctx)`, `pinfer_checkpoint_clear(ctx)`, `pinfer_checkpoint_n_past(ctx)` | One rewind point inside the context (added after 0.2.0, additive): `checkpoint` copies the recurrent + conv states of the linear layers (20 MB for the 0.8B, 53 MB for the 4B, independent of the context length) and remembers the position; `rewind` copies them back and returns to that position (`E_ARG` if there is none; the context is then unchanged). The KV rows below the checkpoint are never rewritten while it is valid and rows above are overwritten before they are read, so feeding tokens after a rewind is bit-identical to feeding them at the checkpoint. One slot; `pinfer_reset`, `pinfer_state_load` and the generate calls that reset drop it; `pinfer_eval`, `pinfer_eval_select`, `pinfer_generate_greedy_cont` keep it; `clear` frees its memory (otherwise kept until `pinfer_free`). Not a persistence format (`pinfer_state_*` is). `engine/CONTEXT.md` §10. |
| `pinfer_params.kv_dtype` | `PINFER_KV_F32` (0, default), `PINFER_KV_F16`, `PINFER_KV_I8`: KV-cache storage (§9). |

Error codes: `PINFER_OK` 0, `E_IO` −1, `E_FORMAT` −2, `E_NOMEM` −3, `E_ARG` −4, `E_CTX_FULL` −5.

**ABI note (0.2.0).** After 0.1.0, `pinfer_params` gained `kv_dtype`, `pinfer_spec_params` gained `adaptive` and `ngram`, and `pinfer_spec_stats` gained `ngram_rounds` / `ngram_drafted` / `ngram_accepted` (all at the end, zero = the old behaviour), so the version is 0.2.0 and the SONAME `libpinfer.so.0.2`. Callers that zero-initialise the structs through `pinfer_default_params` / `pinfer_default_spec_params` are source-compatible; binaries built against the 0.1 header must be rebuilt (0.x: any minor release may change the ABI). State blobs saved by 0.1 are refused (format 2).

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
- state blobs refused for another model file (same layout, other weights), another `kv_dtype`, format version 1 or a flipped fingerprint (`test_state.py`);
- the caller's CPU affinity before and after `pinfer_load` / `pinfer_eval`, with a full and a restricted mask, and the workers' pinning on the NEON build (`test_affinity.py`);
- one version in `pinfer.h`, `pyproject.toml`, the package, the library and its SONAME (`test_version.py`);
- the CLI.

C-level checks outside pytest: `tests/test_malformed.py` (43 broken `.pq` files, state blobs and token files must fail with the documented error and no sanitizer report; run it on the sanitizer build, `make -C engine host SAN=address,undefined`) and `tests/test_long_context.py` (logits up to 8192 tokens vs the fp32 transformers reference). Details: `engine/HARDENING.md`.

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
- `--speculative` defaults (0.2): **adaptive k ≤ 4 with prompt-lookup drafts (n-gram 7)**, draft
  vocabulary 32768. The engine continues the MTP draft chain while the draft is confident, uses
  n-gram drafts when the text repeats the context, and falls back to plain steps when speculation
  would be slower. Within 0.98–1.06× of the best fixed k on every benchmark prompt and 7–27% faster on
  prose, long context and copied code (`engine/DRAFT.md`). The draft vocabulary (first 32768 ids)
  lowers acceptance a little but makes each draft ~5× cheaper. 0.1 used fixed k = 2.
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
- **In-context checkpoint** (`checkpoint()` / `rewind()`, C `pinfer_checkpoint` /
  `pinfer_rewind`; added after 0.2.0): for many continuations of one prefix inside one process,
  e.g. a decision model scoring several questions about one document. Only the recurrent state
  is copied (no blob, no KV rows: 0.8B 20 MB, 4B 53 MB whatever the context length), the rewind
  is one memcpy, and the result is bit-identical to a `save_state` / `load_state` round trip or
  a fresh prefill. `save_state` stays the way to keep a state beyond the process or move it
  between contexts. `engine/CONTEXT.md` §10 has the measurements.
- **Adaptive speculation** (`spec_adaptive=True`, `--adaptive`): `spec_k` becomes an upper
  bound; the MTP head keeps drafting while its newest draft is confident (top-1 probability
  ≥ 0.6), and the engine falls back to plain steps when its measured rate is below plain
  decoding's; same output. Within 0.984–1.056× of the best fixed k on the chat prompts,
  prompt64 and a 2048-token context (`engine/DRAFT.md`).
- **Prompt-lookup drafts** (`spec_ngram=N`, `pinfer_run --ngram N`): where the answer repeats
  the context (an edited function, a quoted passage, repetitive text), up to N tokens that
  followed the earlier occurrence are verified in one pass; same output. Code-edit chat answer
  22.2 vs 19.4 tok/s (adaptive, N = 7), 2048-token wikitext context 13.0 vs 11.1; ordinary chat
  unchanged.

## 10. Server (`pinfer serve`, OpenAI-compatible HTTP)
`pinfer serve` puts one model behind the OpenAI HTTP API, so the `openai` SDK, curl, Open WebUI or an
editor plugin can use the Pi as a local model. It uses only the standard library (`http.server`, one
thread per connection); the package's dependencies stay numpy and tokenizers (jinja2 optional).
```bash
pinfer serve -m models/qwen35-0.8b-q4k-headq4k.pq --tokenizer models/qwen35-0.8b-tokenizer \
    --host 0.0.0.0 --port 8080            # default --host 127.0.0.1: this machine only
```

| Option | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8080` | Listen address. `0.0.0.0` serves the network; there is no TLS, so use `--api-key` outside a trusted LAN. |
| `--api-key KEY` (or `$PINFER_API_KEY`) | none | Require `Authorization: Bearer KEY` on `/v1/*` (`/health` stays open). |
| `-c/--ctx/--max-ctx`, `--kv-dtype`, `-t`, `--tokenizer`, `--lib` | 4096, fp32, 4 | As for the other commands (§3, §9). |
| `--speculative` / `--no-speculative`, `--spec-k` | on, library default (2) | Greedy requests use MTP speculative decoding (same tokens as plain greedy). |
| `--no-reuse` | off | Prefill every prompt from scratch (for comparisons). |
| `--history generated` / `template` | `generated` | How earlier assistant turns are tokenized (see *Prefix reuse*). |
| `--state-cache-mb`, `--snapshot-min-tokens` | 256, 64 | Memory for state snapshots (0 = none); shortest prefix worth one. |
| `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--think` | 1024, 0, 1, 0, off | Defaults for requests that leave these out (`/v1/completions` keeps OpenAI's `max_tokens` 16). OpenAI's own default temperature is 1; this server defaults to greedy (fastest, deterministic). |
| `--model-name` | the `.pq` file stem | Model id in `/v1/models` and responses (the `model` field of requests is not checked). |

**Endpoints**

| Endpoint | Notes |
|---|---|
| `GET /health` | `status`, `busy`, `queued`, tokens in context, snapshot cache use. |
| `GET /v1/models`, `GET /v1/models/{id}` | One model; `max_model_len` = `--max-ctx`. |
| `POST /v1/chat/completions` | `messages` (system / developer / user / assistant / tool; text or text parts), `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k` (extension), `seed`, `stop` (a string or up to 16), `stream`, `stream_options.include_usage`, `chat_template_kwargs.enable_thinking`. |
| `POST /v1/completions` | `prompt` (text, `[text]` or token ids), the same options, `echo`. |

Penalties, `logit_bias` and `user` are accepted and ignored. Refused with HTTP 400: `n` > 1, `logprobs`,
`tools` / `functions`, a `response_format` other than text, `suffix`, `best_of` > 1, image or audio parts.

**Responses** follow OpenAI's shapes (`chat.completion`, `chat.completion.chunk`, `text_completion`).
`finish_reason` is `stop` (stop token or stop string) or `length` (`max_tokens` or a full context).
`usage` carries `prompt_tokens_details.cached_tokens`, and every response (in a stream: the chunk with
the `finish_reason` and the usage chunk) has a `pinfer` object:

| Field | Meaning |
|---|---|
| `cached_tokens`, `cache_source` | Prompt tokens not evaluated again, and where from: `context` (the live context), `snapshot` or `null`. |
| `spliced_tokens` | Leading prompt tokens taken from the server's own earlier request + reply (*Prefix reuse*). |
| `prefill_seconds` | Server time to the first token (snapshot restore + prefill of the uncached tokens). |
| `decode_seconds`, `tokens_per_second` | The rest of the generation; tokens after the first per second. |
| `queue_seconds` | Time spent waiting for another request to finish. |
| `speculative` | `k`, `rounds`, `drafted`, `accepted`, `tokens_per_pass`, or `null` for plain decoding. |

With `chat_template_kwargs: {"enable_thinking": true}` the model writes a think block first: its text
comes as `reasoning_content` (streamed as `delta.reasoning_content`) and the answer as `content`.

**Streaming** is Server-Sent Events over chunked transfer encoding: `data: {chunk}` events, the first
with `delta: {"role": "assistant", "content": ""}`, then text deltas, a chunk with the `finish_reason`,
with `include_usage` a last chunk with `choices: []` and `usage` (the other chunks then carry
`"usage": null`), and `data: [DONE]`. An error after the headers were sent arrives as a
`data: {"error": ...}` event.

**Errors** are OpenAI-style, `{"error": {"message", "type", "param", "code"}}`: 400 for a bad request
(`invalid_json`, a wrong field, or `context_length_exceeded` when the prompt does not fit in
`--max-ctx` with room for one reply token), 401 `invalid_api_key`, 404 `unknown_url` /
`model_not_found`, 411 / 413 for a missing or oversized body, 500 `internal_error` (the context is
then reset).

**One model, one context.** The engine context is not thread-safe, so generations run one at a time in
arrival order (FIFO); requests are parsed and tokenized in their own threads while they wait. A client
that disconnects while queued is dropped; one that disconnects during a stream stops the generation at
the next token (with speculative decoding the C call ends at its next token callback); a non-streaming
request whose client left is stopped at the first token after a half-second check. Greedy requests (`temperature` 0, the server
default, or `top_k` 1) use speculative decoding with the library default (adaptive k ≤ 4 with n-gram drafts,
1.25–1.8× plain on chat prompts, `engine/DRAFT.md`); sampled requests use plain decoding.

**Prefix reuse.** Clients resend the whole conversation every turn. Qwen3.5 is a hybrid model: its 18
linear-attention layers carry a recurrent state that cannot be rolled back, so the engine can continue
its context only when the new prompt *extends* every token already in it. Two mechanisms make that
happen:
- *History as generated* (`--history generated`). The chat template renders an earlier assistant turn
  without the think block the model saw while writing it (`<|im_start|>assistant\n<think>\n\n</think>\n\nA`
  becomes `<|im_start|>assistant\nA`), and re-tokenizing a reply need not give the generated ids back.
  The server remembers its recent replies (prompt tokens + generated tokens). When the previous
  assistant message of a conversation is one of them, the prompt is built from those tokens,
  `<|im_end|>` and the template's rendering of the new messages, so only the new turn is prefilled.
  With `--history template` (what a stateless server does) the prompt diverges inside the first
  earlier reply, and on this model nothing at all is reused from the second turn on. With thinking
  enabled the reasoning stays in the context (the template would drop it): more context is used, but
  nothing is prefilled again.
- *State snapshots.* Before the live context is replaced by another conversation it is saved
  (`save_state()`, ~20 MB + 24 KB per token with the fp32 KV cache) in an LRU bounded by
  `--state-cache-mb`; a system message of at least `--snapshot-min-tokens` tokens is checkpointed too.
  A request starts from the longest snapshot or live context that is a prefix of its prompt, so
  alternating conversations, or new ones with the same system prompt, do not evict each other.

Measured on the Pi 4 (Q4K, 4 threads, greedy so speculative k=2, ≤ 64 tokens per reply; client on
the Pi over localhost; `notes.md` "`pinfer serve` on the Pi 4", `bench/results/serve_20261022_095259.json`).
TTFT is the time from sending the request to the first content delta:

| | cached / prompt tokens | TTFT |
|---|---|---|
| single request (39-token prompt) | 0 / 39 | 1.33 s, then 9.0 tok/s (client = server rate) |
| 4-turn conversation, turns 2 / 3 / 4, default | 86 / 111, 147 / 169, 232 / 256 | 0.96 / 0.89 / 0.95 s |
| same, `--history template` | 0 / 107, 0 / 161, 0 / 244 | 3.21 / 4.82 / 7.08 s |
| same, `--no-reuse` | 0 | 3.09 / 4.58 / 6.90 s |
| two interleaved conversations, second turns | 97 / 120, 99 / 121 (snapshots) | 1.03 / 1.01 s (3.56 / 3.50 s without) |

**Limits.** One model and one generation at a time; concurrent requests are queued, not batched. No
tool / function calling, logprobs, `n` > 1, images, embeddings or TLS; penalties and `logit_bias` are
ignored. Regenerating the last answer (the same prompt again) prefills it from the longest snapshot
(often only the system prompt), and a reply cut by a stop string is re-tokenized for the next turn.
The server does not truncate history: a conversation longer than `--max-ctx` gets
`context_length_exceeded`.

Run the clients on the Pi, or replace `localhost` with the Pi's address (server started with
`--host 0.0.0.0`).

**curl**
```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen35-0.8b-q4k-headq4k",
  "messages": [{"role": "user", "content": "What is a Raspberry Pi?"}],
  "max_tokens": 64, "stream": true}'
```

**openai SDK**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")  # any key without --api-key
model = client.models.list().data[0].id
r = client.chat.completions.create(model=model, max_tokens=64,
                                   messages=[{"role": "user", "content": "What is a Raspberry Pi?"}])
print(r.choices[0].message.content, r.usage, r.model_extra["pinfer"])
stream = client.chat.completions.create(model=model, max_tokens=64, stream=True,
                                        messages=[{"role": "user", "content": "Count to ten."}])
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

**From Python**
```python
import pinfer
from pinfer.server import EngineConfig, serve

with pinfer.Model.load("qwen35-0.8b-q4k-headq4k.pq") as m:
    serve(m, EngineConfig(model_name="qwen"), host="127.0.0.1", port=8080)   # until Ctrl-C
```
`pinfer.server.make_server(Engine(m, cfg), host, port)` returns the `http.server` object instead
(`serve_forever()`, `shutdown()`). Code: `python/pinfer/server/` (`protocol.py` wire format, `engine.py`
context and generation, `history.py` prompt tokens, `cache.py` snapshots / replies / queue, `httpd.py`
HTTP); tests: `tests/python/test_server.py`. Measurement: `bench/serve/serve_client.py` (works with any
OpenAI-compatible server) and `bench/serve/serve_bench.sh` (starts the server once per configuration;
on the Pi in one locked `bench` call).
