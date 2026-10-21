# pinfer 0.1.0 — release notes (DRAFT, not published)

> Draft for review. Nothing has been uploaded anywhere. Publishing (PyPI, GitHub releases, any
> public index) needs the user's explicit approval. Open decisions before publishing are listed at
> the end.

## What it is
The first packaged release of pinfer, a Raspberry-Pi-4-native inference engine for **Qwen3.5-0.8B**.

## Highlights
- **C library.** `libpinfer.so` and `libpinfer.a` with a single public header, `pinfer.h` (API 0.1). The `.so` exports only `pinfer_*`. The static archive also hides every engine-internal symbol. `make install` writes headers, libraries and the pkg-config file `pinfer.pc`. The SONAME is `libpinfer.so.0.1`.
- **Python package `pinfer`:**
  - `Model.load(...)`, `generate(...)` (greedy or temperature / top-k / top-p sampling, stop strings, streaming), `chat(...)` (Qwen chat template), token-level `iter_tokens` / `eval`;
  - opt-in **MTP speculative greedy decoding** (`speculative=True`) through `pinfer_generate_greedy`, which produces the same tokens as plain greedy decoding;
  - a typed exception hierarchy.
- **CLI `pinfer`:** `generate`, `chat`, `bench`, `convert` (bundles `tools/convert.py` and writes a tokenizer sidecar), `info`.
- **Wheels:**
  - `pinfer-0.1.0-py3-none-manylinux_2_33_aarch64.whl` (Pi);
  - `pinfer-0.1.0-py3-none-manylinux_2_33_x86_64.whl` (development and tests).

  The wheel tag comes from the bundled library's newest glibc symbol version, so pip refuses a system whose glibc is too old.

## Correctness
- Greedy generation through the Python API reproduces the transformers 5.20.2 fp32 reference: all 32 golden token ids and the exact text. This holds for `generate`, `iter_tokens`, prefix reuse and speculative decoding.
- Test suite `tests/python`, 80 tests: 80/80 passed on x86-64 (WSL, Python 3.14.4, `logs/release/host_pytest_final.log`) and 80/80 on the Pi 4 (Python 3.13.5, Raspberry Pi OS trixie, `logs/release/pi_pytest_final.log`).
- The C example built against `libpinfer.a` (static, Pi) and `libpinfer.so` (host) prints the golden ids.

## Performance
The 0.1.0 wheels are built with the **reference** kernels (`MATVEC=ref`): about 1.5 tok/s decode and 2.0 tok/s prefill on the Pi 4, bf16, 1 thread. The NEON / quantized kernels (`MATVEC=fast`) are in development and will ship in a later release without API changes. Measured numbers: `notes.md`.

## Compatibility
- Linux only. aarch64 needs glibc ≥ 2.33 (the Pi has 2.41); x86-64 needs glibc ≥ 2.33. Runtime dependencies are libc and libm only.
- Python ≥ 3.10. Dependencies: `numpy`, `tokenizers`. Optional: `jinja2` to render the model's own chat template (a built-in Qwen3.5 template is used otherwise).
- `.pq` format version 1 (`engine/FORMAT.md`).

## Known limitations
- `pinfer_generate_greedy` (speculative decoding):
  - stops only at EOS (other stop tokens are applied afterwards);
  - cannot stream;
  - returns `E_CTX_FULL` without partial output (the Python wrapper clamps the length to the free context instead).
- A generation cannot roll back. Prefix reuse helps only when a prompt extends the previous context exactly.
- The library prints diagnostics to stderr; there is no log callback yet.
- No sdist.

## Before publishing (decisions for the user)
1. **License** for pinfer itself. None is chosen yet; the wheel metadata has no license field. The Qwen3.5 weights are Apache-2.0 and are not bundled.
2. Where to publish: PyPI, GitHub Releases or a private index. Also the package name: check that `pinfer` is free on PyPI.
3. Whether 0.1.0 ships with the `ref` kernels or waits for `MATVEC=fast`. The fast build's Pi `.so` needs `GLIBC_2.34` (pthread), so its wheel tag would be `manylinux_2_34_aarch64`.
4. Authors / maintainers / project URLs for the metadata.
