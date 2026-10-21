# Engine hardening notebook (issue #11, 2026-10-22)

Validation, safety and ABI work for 0.2.0, plus the first long-context correctness check. Each
finding: problem → root cause → fix → the test that proves it, with evidence. Output on valid
models is unchanged (bit-identical, §Gates). Branch `worktree-agent-ab48d7e263b2f8814` (hardening
agent); the `kern_lin.c` `scratch_state` race of issue #11 belongs to the verify-pass work.

Tests added: `tests/test_malformed.py` (43 broken inputs, C level, runs under ASan/UBSan on the
host and against a NEON build on the Pi), `tests/test_long_context.py` + `ref/long_context_reference.py`,
`tests/python/test_affinity.py`, `tests/python/test_version.py`, 5 cases in `tests/python/test_state.py`.
Sanitizer build: `make -C engine host SAN=address,undefined` → `build/host-san/engine/`.

## Summary

| # | Finding (issue #11) | Fix | Proof | Before (baseline `3c2c887`) → after |
|---|---|---|---|---|
| H1 | quantized tensors' `nbytes` never checked; `rows % 4`, `cols % 256` not enforced (`model.c` `get_mat` / `check_directory`, `mtp.c` `mat`) | `check_directory` validates every entry's size for its dtype and shape, quantized ones included; `mtp.c` uses `model.c`'s views | malformed `q-nbytes`, `q-nbytes-at-eof`, `q-rows`, `q-cols`, `q-ndim`, `q-mtp-nbytes`, `dtype-q8-for-bf16` | Pi `pinfer_run_fast`: `q-nbytes`, `q-nbytes-at-eof`, `q-mtp-nbytes`, `dtype-q8-for-bf16` accepted (exit 0; the matmul reads past the tensor); the bad shapes were refused only where a layer expects that exact shape → all `PINFER_E_FORMAT` at load, from the size check |
| H2 | `check_header` has no upper bounds; uint32 products in `build_layer` and the uint64 element count can wrap; `lin_v_dim % 32` unchecked (`pinfer.c:108`) | `PQ_MAX_*` limits (`pq_format.h`), element count built factor by factor against the file size, `eos_id < vocab`, finite `rms_eps` / `rope_theta`, 8-byte-aligned directory, `lin_v_dim % 32` in `pinfer_load` | malformed `hdr-*`, `shape-wrap`, `shape-huge`, `shape-zero`, `dir-misaligned`, `lin-v-dim-80`, `lin-v-dim-544` | Pi: `lin-v-dim-80` **SIGSEGV** in the DeltaNet kernel, `shape-wrap` / `hdr-eos` / `hdr-theta-inf` accepted; host UBSan: misaligned access at `model.c:42` (`dir-misaligned`) → all `PINFER_E_FORMAT` |
| H3 | `pinfer_params.kv_dtype`, `pinfer_spec_params.adaptive` added under 0.1.0 / `libpinfer.so.0.1` | version 0.2.0 everywhere: `pinfer.h`, SONAME `libpinfer.so.0.2` (Makefile derives it), Python package, wheel names, docs | `tests/python/test_version.py`; `make check-lib`; wheel names | `pinfer-0.2.0-py3-none-manylinux_2_33_x86_64.whl`, `..._2_34_aarch64.whl` |
| H4 | `pf_pool_create` pins the caller to CPU 0 and never restores it; later threads inherit it | the caller is never pinned; worker t is pinned to the t-th CPU of the caller's mask; at most that many threads | `tests/python/test_affinity.py` on the Pi | old library: caller `{0,1,2,3}` → `{0}`; a caller limited to `{2,3}` was moved to `{0}` → masks kept; speed unchanged (§H4) |
| H5 | `state.c` accepts any blob with matching dimensions | 64-bit fingerprint of the file (header, directory, 3 × 64-byte samples per tensor) in the former `reserved`; state format 2 | malformed `state-*`; 5 pytest cases; Pi `state-quick` | another file with the same layout (or quantization) accepted → `PINFER_E_FORMAT`; round trip still exact on the Pi |
| H6 | `pinfer_ppl.c:94`: the target indexes the logits before validation; `--chunks 0` prints NaN | every id the chunks use is checked against the vocabulary before the first pass; `--chunks >= 1`; odd-sized or empty token files refused | malformed `ppl-*` | host UBSan: `pinfer_ppl.c:45: load of address ... with insufficient space` (out-of-bounds read); `--chunks 0` exit 0 with a NaN ppl → exit 1 / 2 with a message |
| H7 | stats counters not synchronized | the `PF_FWD_STATS` / `PF_KERN_STATS` flags are set once by an atomic compare-exchange; the tables are updated under a mutex (only when profiling is on) | ASan/UBSan runs; `make check-lib` | lazy init raced on the public `pinfer_argmax`; tables raced with several contexts → thread-safe, documented as process-wide |
| B1 | positions past 4096 never compared with the fp32 reference | — (no bug found) | `tests/test_long_context.py` | 18 prefix lengths to 8192: worst rel. error 2.0e-6, top-1 18/18 |

## H1. Quantized tensor sizes

**Problem.** `check_directory` compared `nbytes` with the shape only for F32 / BF16
(`dtype_elem_bytes` returned 0 for quantized dtypes, "size not checked"), and `get_mat` / mtp.c's
`mat` checked only the shape. A quantized tensor whose `nbytes` is short loaded, and the
kernels then read `rows/4 · cols/256 · SB` bytes from its offset: past the tensor, into the next
one or past the end of the mapping. The tiling rule (`rows % 4`, `cols % 256`) was not checked
either: a header whose dimensions break it (say `ffn = 3583`) would have been accepted together
with matching tensors.

**Fix.** `entry_bytes()` (`engine/src/model.c`) gives every entry's exact size: F32 / BF16 elements,
or for a known quantized dtype a 2-D shape with `rows % 4 == 0`, `cols % 256 == 0` and
`rows/4 · cols/256 · SB` bytes; unknown dtypes are refused. `check_directory` requires
`nbytes == entry_bytes` for **every** entry, so the `mtp.*` head (mapped only by `--spec`) is
covered at load too. `mtp.c` now uses `model.c`'s `pf_model_mat` / `pf_model_vec` instead of
its own copies. FORMAT.md gained a "Validation" section.

**Test.** `tests/test_malformed.py` builds the cases from the real Q4K file's header and
directory (sparse copies, no weights copied): `q-nbytes` (one superblock short), `q-nbytes-at-eof`
(short, and moved to end exactly at the page-aligned end of the file), `q-rows` (3583),
`q-cols` (1000), `q-ndim`, `q-mtp-nbytes`, and `dtype-q8-for-bf16` (a BF16 matrix relabelled Q8).
On the host's scalar build the right message must appear before "dtype not supported by this
build", which proves the size check fired first.
Evidence: Pi, baseline `pinfer_run_fast` 18/43 (`logs/runs/20261022_074707_run.log`: the quantized
cases and `dtype-q8-for-bf16` exit 0; `q-nbytes-at-eof` did not crash in this run, the bytes after
the mapping were readable); new build 43/43 (`logs/runs/20261022_074644_run.log`).

## H2. Header bounds, wraps, lin_v_dim

**Problem.** `check_header` had lower bounds only. `n_heads * head_dim * 2`, `lin_k_heads *
lin_k_dim` and `2*kd + vd` are uint32 in `build_layer`; `n_heads = 2^23 + 8` makes the q_proj rows
wrap to the real 4096. The element count was a plain uint64 product of up to four uint32 dims:
`mtp.norm.weight` with shape `[1024, 14586017, 308761441, 1]` gives `4 · product = 2^64 + 4096`,
which wraps to its real `nbytes` and loaded. `eos_id` and the floats were unchecked
(`rope_theta = inf` loaded). `kern_lin.c` walks the value columns in chunks of 32 (`CW`) and
assumes `lin_v_dim % 32 == 0`; nothing checked it. A misaligned `dir_offset` made every
directory read a misaligned uint64 access.

**Fix.** `pq_format.h` `PQ_MAX_*` (hidden, ffn ≤ 65536; vocab ≤ 2²¹; n_layers ≤ 32; head
counts ≤ 256; head dims ≤ 1024; conv_k ≤ 8): every product of header fields stays far inside
uint32 and int. The element count is built factor by factor, each step checked against the file
size; zero dimensions are refused. `eos_id < vocab`; `rms_eps`, `rope_theta` finite and > 0;
`dir_offset % 8 == 0`. `pinfer_load`: `lin_v_dim % 32 == 0` (with the existing ≤ 512).

**Test.** `hdr-huge-hidden`, `hdr-layers-33`, `hdr-heads-wrap`, `hdr-lin-wrap`, `hdr-eos`,
`hdr-theta-inf`, `hdr-eps-nan`, `shape-wrap`, `shape-huge`, `shape-zero`, `dir-misaligned`,
`dir-n-tensors`, and `lin-v-dim-80` / `lin-v-dim-544`: self-consistent files (every dependent
tensor reshaped, so all format checks pass) that only the engine limit can refuse.
Evidence: Pi baseline `lin-v-dim-80` killed by **SIGSEGV** (`logs/runs/20261022_074707_run.log`);
host baseline under UBSan: `model.c:42:13: runtime error: member access within misaligned
address` for `dir-misaligned` (scratchpad `hard/malformed_base-asan-detail.log`).

## H3. ABI version 0.2.0

**Problem.** Two public structs grew (`kv_dtype`, `adaptive`) while the version and SONAME stayed
0.1.0 / `libpinfer.so.0.1`: a binary built against the 0.1.0 header could load the new library.

**Fix.** `PINFER_VERSION_MINOR 2` (`pinfer.h`; the Makefile derives `VERSION` and
`SONAME = libpinfer.so.0.2` from it), `python/pyproject.toml` and `pinfer.__version__` 0.2.0, the
"added in 0.1" comments (`pinfer.h`, `python/pinfer/_lib.py`) say 0.2, `docs/library.md` /
`docs/REPRODUCE.md` name the 0.2.0 wheels and `libpinfer.so.0.2.0`. `docs/release-0.1.0.md`
stays the historical 0.1.0 note.

**Test.** `tests/python/test_version.py` (new): `pinfer.h` == `pyproject.toml` ==
`pinfer.__version__` == the loaded library, and `BUILD_INFO.json` `soname == libpinfer.so.0.2`.
`make -C engine check-lib`: SONAME `libpinfer.so.0.2` for host and Pi; exports `pinfer_*` only;
host `.so` needs GLIBC 2.33, the fast Pi `.so` 2.34 (both as in 0.1.0). Wheels:
`pinfer-0.2.0-py3-none-manylinux_2_33_x86_64.whl`, `pinfer-0.2.0-py3-none-manylinux_2_34_aarch64.whl`.

## H4. The thread pool pinned the caller

**Problem.** `pf_pool_create` called `pin_to(0)` on the calling thread whenever `n_threads > 1`
and never undid it. In a Python app the main thread loads the model, so the interpreter's main
thread, and every thread created afterwards (they inherit the mask), ran on CPU 0 only. With a
restricted mask (e.g. `taskset -c 2,3`) it even moved the caller to a CPU outside its set.

**Fix.** The caller is not pinned at all (`engine/src/matvec_fast.c`): worker `t` is pinned to the
`t`-th CPU of the caller's affinity mask (CPU t when the mask holds every CPU, so the default
layout is unchanged: workers on CPUs 1–3), and `n_threads` is capped by the number of CPUs in
that mask (more threads than allowed cores would make spinning workers share one). Save and
restore was not an option: there is no point at which to restore, the inherited mask leaks into
any thread created between `pinfer_load` and `pinfer_free`.

**Test.** `tests/python/test_affinity.py` (new): the caller's mask before and after
`pinfer_load` and `pinfer_eval`, the mask of a thread created after the load, and, on the NEON
build, that the new threads are pinned to CPUs 1, 2, 3; a second case with the caller limited to
two CPUs (one worker, pinned to the second). Pi, old library (the installed 0.1.0 wheel's
`libpinfer.so`): both fail, `assert {0} == {0, 1, 2, 3}` and `assert ({0} == {2, 3}`
(`logs/runs/20261022_074518_run.log`, `..._074541_run.log`); new library: pass (Pi pytest 94/94,
`logs/runs/20261022_073746_run.log`).

**Speed.** With the workers spinning on CPUs 1–3 the scheduler keeps the caller on CPU 0.
Prompt64, Q4K, 4 threads, interleaved A/B in one locked bench call (`engine/bench/speed_ab.sh`, run from the
Pi's `tmp/hard/` during this work):

| binary | plain tok/s (runs) | `--spec 3 --draft-vocab 32768` tok/s (runs) | result file |
|---|---|---|---|
| `pinfer_run_fast` (reference) | 7.881, 7.893 | 16.743, 16.577 | `bench/results/speed-ab-hard1_20261022_072935.json` |
| `pinfer_run_hard1` | 7.828, 7.894 | 16.681, 16.742 | same file |
| `pinfer_run_fast` (reference) | 7.887, 7.893, 7.892 (mean 7.891) | 16.719, 16.679, 16.753 (mean 16.717) | `bench/results/speed-ab-hard2_20261022_081251.json` |
| `pinfer_run_hard2` (final) | 7.888, 7.888, 7.889 (mean 7.888) | 16.725, 16.712, 16.702 (mean 16.713) | same file |

Decode: −0.04% plain, −0.02% speculative, inside the run-to-run spread (targets 7.88 / 16.79).
Prefill over 64 tokens varies 34.4–38.4 tok/s from run to run for both binaries (overlapping
ranges, no direction). Where the caller runs (`engine/bench/caller_cpu.sh`, 20 samples of `ps -L`
during a 160-token decode): new build, caller allowed on 0–3 and found on CPU 0 in 20/20 samples,
workers on 1, 2, 3 (`bench/results/caller-cpu-hard2_20261022_082944.json`); old build, caller
allowed only on CPU 0 (`caller-cpu-fast_20261022_083008.json`).

## H5. State blobs tied to the model file

**Problem.** `pinfer_state_load` compared only the dimensions and `kv_dtype`. A state saved with
`qwen35-0.8b-q4k-headq4k.pq` loaded into `qwen35-0.8b-mix-headq4k.pq` (same dimensions, other
weights) and continued from a KV cache and recurrent state the model never produced.

**Fix.** `pf_model_fingerprint()` (`model.c`): 64-bit FNV-1a over the header, the tensor
directory and three 64-byte samples (start, middle, end) of every tensor, so a file with the same
layout but other weights (a fine-tune converted the same way) differs too. It is computed on
each save and load (~3 cache lines per tensor; no cost at `pinfer_load`), stored in the former
`reserved[2]` (`uint64_t model_fp`, header still 72 bytes) and compared on load;
`STATE_VERSION` 2, so 0.1 blobs are refused. `python/pinfer/state.py` reads the C header
(`c_header`, `split`) and turns `PINFER_E_FORMAT` into a `ModelFormatError` that names the cause:
format version, `kv_dtype`, or another model file.

**Test.** malformed `state-control` (round trip), `state-other-weights` (same header and
directory, zero weights), `state-v1`, `state-fingerprint`, `state-truncated`, `state-ctx-full`;
pytest `test_state_header_carries_version_and_fingerprint`,
`test_load_state_rejects_same_layout_other_weights`, `..._flipped_fingerprint`,
`..._version_1`, `..._other_kv_dtype`; host `tests/test_state.py` 9/9; Pi `state-quick`
(prompt64 split at 40, plain / k=3 / adaptive: exact; the Q4K state into the mix model: refused):
`bench/results/state-quick-hard1_20261022_073204.json`, final `state-quick-hard2_20261022_081558.json`;
the same script with `pinfer_run_fast` loads the Q4K state into the mix model without complaint
(exit 0, `state-quick-fast_20261022_083238.json`). Cost, 21.8 MB state of 64 tokens, 3 interleaved
rounds (`state-time-hard2_20261022_081814.json`): save 29.2–29.5 ms vs 28.6–29.1 ms (+0.7 ms), load
29.6–31.1 ms vs 27.0–27.7 ms (+2.9 ms; the load runs in a fresh process, where the sampled weight
pages are touched for the first time).

## H6. pinfer_ppl

**Problem.** The scored target `ch[pos+1]` indexed the logits before any forward pass had seen
it: the target of the last position of a 64-token sub-batch is the first token of the next one.
`--chunks 0` scored nothing and printed `"ppl":nan`, which is not JSON.

**Fix.** All `chunks · ctx` ids are checked against the vocabulary after loading;
`--chunks < 1` is a usage error (exit 2); the token reader refuses empty files, files whose size
is not a multiple of 4 and I/O errors; `ppl_err` no longer divides by zero for one scored token.

**Test.** `ppl-bad-target` (`-c 256`, token 192 = 2²⁰), `ppl-negative`, `ppl-chunks0`,
`ppl-odd-size`, `ppl-control` (valid JSON). Host baseline under UBSan: `pinfer_ppl.c:45:27:
runtime error: load of address ... with insufficient space` (scratchpad
`hard/malformed_base-asan-detail.log`).

## H7. Stats counters

**Problem.** `forward_stats.c` and `kern_stats.c` keep process-wide tables. The enable flag was
initialised lazily with a plain read and write, also from the public `pinfer_argmax` (callable from
any thread), and the tables were updated without a lock, so two contexts on two threads (the
documented "one Model per thread") raced as soon as profiling was on.

**Fix.** The flag is set once by `__atomic_compare_exchange_n` and read with acquire loads; the
table updates take a mutex, only when profiling is on (off: one atomic load per section, as
before one plain load). Not `pthread_once`: that symbol would have raised the library's glibc
requirement from 2.33 to 2.34 and the x86 wheel to `manylinux_2_34` (seen in `make check-lib`,
reverted). The tables are documented as process-wide sums (`docs/library.md` "Profiling").

## Long context (B1): positions past 4096 vs the fp32 reference

`tests/test_long_context.py`: reference = transformers 5.20.2 `Qwen3_5ForConditionalGeneration`,
fp32, sdpa, one forward over `bench/common/ctx8192.ids` (wikitext-2) with `logits_to_keep` at the
checked positions (210 s, 6.6 GB RSS on the PC). Engine = host `libpinfer.so` (scalar
`MATVEC=ref`, bit-identical to the Pi's NEON build) through the Python API, fed in pieces ending
at each prefix length n; the logits of token n−1 are compared (43 min). `pinfer_run --dump-logits`
was not used: it writes every position, 8 GB at 8192 tokens. Tolerances are test_golden's
(rel. max error ≤ 1e-3; top-1 must agree unless the reference's own top-2 margin is below twice
the max abs error).

| n | max\|ref\| | max abs err | rel max err | KL(ref‖ours) | top-1 ours / ref | ref top-2 margin |
|---|---|---|---|---|---|---|
| 512 | 22.23 | 2.06e-05 | 9.3e-07 | 1.7e-12 | 506 / 506 | 2.67 |
| 1024 | 24.52 | 2.19e-05 | 8.9e-07 | 5.3e-13 | 321 / 321 | 6.66 |
| 2048 | 19.33 | 1.88e-05 | 9.7e-07 | 1.1e-11 | 314 / 314 | 0.022 |
| 3000 | 15.78 | 2.03e-05 | 1.3e-06 | 6.2e-12 | 9290 / 9290 | 0.0047 |
| 3584 | 22.08 | 3.77e-05 | 1.7e-06 | 6.2e-12 | 10077 / 10077 | 1.13 |
| 4096 | 19.10 | 2.11e-05 | 1.1e-06 | 6.0e-12 | 18248 / 18248 | 1.25 |
| 5120 | 19.11 | 1.86e-05 | 9.7e-07 | 5.9e-12 | 42749 / 42749 | 1.41 |
| 6000 | 16.49 | 2.00e-05 | 1.2e-06 | 7.9e-12 | 381 / 381 | 0.054 |
| 6144 | 14.16 | 2.86e-05 | 2.0e-06 | 4.8e-12 | 1369 / 1369 | 1.19 |
| 7168 | 20.39 | 3.10e-05 | 1.5e-06 | 4.4e-12 | 440 / 440 | 1.25 |
| 8192 | 24.94 | 3.36e-05 | 1.3e-06 | 3.9e-15 | 16 / 16 | 9.41 |

All 18 checked lengths (every 512 plus 3000 and 6000) pass: worst rel. max error **2.0e-6**
(limit 1e-3), top-1 18/18 including near-ties (margins 0.005 and 0.02), top-5 5/5 everywhere.
The error grows from ~1e-6 at 512 to ~1.5e-6 past 6000: no position-dependent drift, so the
partial RoPE at θ = 1e7, the KV cache over 8192 rows and 8192 recurrence steps match the
reference. Full table: scratchpad `long/run_base.log` (baseline library; the loader changes do not
touch the forward pass, and the host gates show identical outputs).

**Sensitivity control** (scratchpad `hard/rope_control.py`, `hard/rope_control.json`): the same
reference with every position shifted by +1 differs from the true one by 5.0e-7–1.8e-6, top-1
18/18 (RoPE depends only on position differences, so this is pure angle rounding, the size of
the engine's error); with `rope_theta = 1e6` instead of 1e7 it differs by 1.5e-2–6.5e-2 (2.4e-2
at 512, 4.0e-2 at 4096, 6.1e-2 at 8192), top-1 17/18: 15–65× the limit and ~10⁴× the engine's
error, so the test would catch a wrong RoPE base by a wide margin.

## Gates (final binary `pinfer_run_hard2`)

`pinfer_run_hard2` / `pinfer_ppl_hard2` (md5 `307e66b1…` / `5b76656c…`) are byte-identical to a clean
rebuild of commit `cc6c3b0` (the last code commit), so these gates cover the committed code.
Every intermediate commit builds without warnings for host and aarch64.

| Gate | Result | Evidence |
|---|---|---|
| Host `tests/test_golden.py`, `test_spec.py`, `test_state.py` | PASS, PASS, PASS | scratchpad `hard/gates_h2/` |
| Host pytest (source tree, new `.so`) | 94 passed (85 + 9 new) | `hard/gates_h2/pytest.log` |
| Host ASan + UBSan (`halt_on_error=1`): golden, spec, state | PASS (76 s), PASS (1234 s), PASS (439 s) | `hard/gates_asan2/` |
| Host `tests/test_malformed.py`, sanitizer build | 43/43, no sanitizer report (baseline HEAD: 18/43) | `hard/malformed_asan2.log`, `malformed_base-asan.log` |
| Host `tests/test_long_context.py` | PASS, worst 2.0e-6 | `long/run_base.log` |
| mypy / ruff on the touched Python files | clean / clean | `hard/mypy.sh`, `hard/ruff.sh` |
| `make check-lib`, wheels | SONAME `libpinfer.so.0.2`; GLIBC 2.33 host, 2.34 Pi fast; `pinfer-0.2.0-…` | `hard/build_all.log` |
| Pi `engine/spec_exact.sh` | 20/20 `"match":true` | `bench/results/spec-exact-hard2_20261022_080210.json` |
| Pi dumps vs `pinfer_run_fast` (prompt64 b64 / b1 hidden + logits, ctx512 logits) | 5/5 identical | `ctx-check-hard2_20261022_080743.json` |
| Pi prompt64 speed A/B | 7.888 vs 7.891 plain, 16.713 vs 16.717 spec tok/s | `speed-ab-hard2_20261022_081251.json` |
| Pi state round trip (plain, k=3, adaptive) + other model | exact ×3, refused | `state-quick-hard2_20261022_081558.json` |
| Pi `tests/test_malformed.py` (NEON build) | 43/43 (baseline `pinfer_run_fast`: 18/43, one SIGSEGV) | `logs/runs/20261022_081814_run.log`; baseline `..._074707_run.log` |
| Pi pytest (scratch layout, NEON `.so` md5 `976b73eb…`, final test files) | 94 passed | `logs/runs/20261022_082427_run.log`, `..._082714_run.log` |

## Dropped / not done
- `kern_lin.c` `scratch_state` race: the verify-pass agent's (issue #11 says so).
- Save-and-restore of the caller's affinity: replaced by not pinning the caller (H4: there is no
  point at which a restore would be in time).
- Fingerprint at `pinfer_load`: computed per state save / load instead, so loading stays free.
