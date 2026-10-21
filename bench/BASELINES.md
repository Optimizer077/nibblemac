# Baselines lab notebook: standard runtimes for Qwen3.5-0.8B on the Raspberry Pi 4

Owner: baselines agent, 2026-10-21/22. Board: Pi 4B Rev 1.1 (Cortex-A72 @1.5 GHz, `fp asimd evtstrm crc32 cpuid`,
no dotprod/fp16/aes/lse), governor `ondemand`, throttled `0x0` on every valid run.
Model: `models/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`.

## Protocol (same as the llama.cpp baseline, as far as each runtime allows)
- Prompt: `bench/common/prompt64.{txt,json}` (built by `bench/common/make_prompt.py`, HF tokenizer):
  first 64 tokens of the opening of *A Tale of Two Cities*; raw text, no chat template, no BOS;
  the text re-encodes to the same 64 ids. `prompt64.json` also holds the HF fp32 greedy
  continuation (transformers 5.20.2 / torch 2.14.1, PC, `Qwen3_5ForCausalLM`) as a reference.
- 64 greedy tokens; prefill tok/s = 64 / prompt-forward time; decode tok/s = 64 single-token
  forwards / their time (token 1 comes from the prefill logits, as in llama-bench tg64).
- 1 warm-up run (excluded) + 3 reps per config; one process per thread count, so the peak RSS
  (VmHWM / ru_maxrss) belongs to that config. Python-based engines' RSS includes the interpreter.
- `min_ref_match` = number of leading generated tokens equal to the HF fp32 reference (min over reps).
- llama.cpp numbers for comparison: llama-bench pp64/tg64 (random tokens), 3 reps.
- All timed runs went through `python harness/pi.py bench` (Pi-wide lock).

## Results: speed (valid=true runs only)
| Runtime | Version | Quant | Threads | Prefill tok/s | Decode tok/s | Peak RSS MB | ref match | Result file |
|---|---|---|---|---|---|---|---|---|
| llama.cpp | v0.7.0 @620d2225 | Q4_0 | 4 | 15.68 | 5.59 | — | — | `bench/results/llamacpp-qwen35-0.8b-Q4_0_20261021_213343.json` |
| llama.cpp | v0.7.0 | Q4_K_M | 4 | 14.82 | 5.41 | — | — | `bench/results/llamacpp-qwen35-0.8b-Q4_K_M_20261021_213929.json` |
| llama.cpp | v0.7.0 | Q8_0 | 2 / 4 | 9.22 / 17.06 | 4.12 / 3.60 | — | — | `bench/results/llamacpp-qwen35-0.8b-Q8_0_20261021_214506.json` |
| **MNN** | 3.6.1 @d407447e | int4 HQQ asym b64 (lm_head int4, embed bf16) | 1 | 9.81 ± 0.10 | 4.11 ± 0.00 | 894 | 0 | `bench/results/mnn-qwen35-0.8b-int4_20261021_221038.json` |
| MNN | 3.6.1 | same | 2 | 18.54 ± 0.01 | **6.67 ± 0.19** | 893 | 0 | same |
| MNN | 3.6.1 | same | 3 | 23.67 ± 0.03 | 6.56 ± 0.03 | 894 | 0 | same |
| MNN | 3.6.1 | same | 4 | **30.94 ± 1.30** | 6.31 ± 0.07 | 894 | 0 | same |
| **ORT GenAI** | genai 0.18.0 / ort 1.31.0 | int4 RTN sym b32, accuracy_level 4 | 1 | 5.14 ± 0.02 | 0.84 ± 0.00 | 782 | 0 | `bench/results/ort-qwen35-0.8b-int4-acc4_20261021_222410.json` |
| ORT GenAI | same | same | 2 | 9.38 ± 0.03 | 1.64 ± 0.00 | 780 | 0 | same |
| ORT GenAI | same | same | 3 | 10.96 ± 0.39 | 2.36 ± 0.01 | 782 | 0 | same |
| ORT GenAI | same | same | 4 | 9.86 ± 0.08 | 3.10 ± 0.01 | 783 | 0 | same |
| ORT GenAI | same | int8 (embed fp32) | 1 | — | — | — | — | aborted, see below |
| LiteRT-LM | 0.15.0–0.18.1 wheels | int8 | — | — | — | — | — | **cannot run on A72** (SIGILL fail-fast) |
| LiteRT-LM | v0.18.1 own armv8-a build | int8 | — | not run on the Pi (stopped by lead) | | | | |
| PyTorch | 2.14.1+cpu / transformers 5.20.2 | fp32 | — | not run on the Pi (stopped by lead) | | | | |

llama.cpp t=1..4 values (tg64): Q4_0 2.32/4.20/5.58/5.59; MNN beats llama.cpp Q4_0 at every thread
count for decode (4.11/6.67/6.56/6.31) and ~2x for prefill at 4 threads (30.9 vs 15.7 tok/s).

`ref match` 0 for both int4 engines: their first greedy token is " the" where HF fp32 says " hope".
On the PC, ORT int4 logits for this position are " the" 14.89 vs " hope" 14.76 (a genuine 4-bit
quantization flip, close margin); ORT int8 picks " hope" and matches the first 5 tokens. MNN's
output stays coherent (" the world, it was the … of the world.\nIt was the best …").
llama.cpp results use random tokens (llama-bench), the others use the shared prompt.

## Results: quality (perplexity, llama.cpp v0.7.0 `llama-perplexity`)
Text: `bench/quality/wikitext-2-raw-test.txt` = wikitext-2-raw-v1 test (HF `Salesforce/wikitext`
parquet, rows concatenated unchanged by `bench/quality/make_wikitext.py`; 4358 rows, 1,285,622 chars,
sha256 `bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a`).
Settings: `-c 512 -b 512 --chunks 64`, KV f16; first 64×512 tokens; second half of each chunk scored
(64×255 = 16,320 tokens). Script: `bench/quality/ppl.sh`, driver `bench/quality/ppl_pc.sh`.
PC build: llama.cpp v0.7.0 @620d2225 `build-host` (x86, GGML_NATIVE=ON), 16 threads.

| GGUF | PPL (64 chunks) | ± | Δ vs bf16 | Platform |
|---|---|---|---|---|
| bf16 | 17.9445 | 0.427 | — | PC x86 |
| Q8_0 | 17.9715 | 0.428 | +0.15 % | PC x86 |
| Q4_K_M | 18.6413 | 0.443 | +3.9 % | PC x86 |
| Q4_0 | 22.0452 | 0.539 | +22.9 % | PC x86 |

Raw lines: `bench/quality/results_pc_c64.jsonl`; full logs: `bench/quality/logs/`.
**Not done:** the same runs on the Pi (`~/pinfer/llama/llama-perplexity`). At ~16 tok/s prefill one
64-chunk run would hold the lock ~35 min (over the 10/20-min limits), so it needs `--chunks 8–16`
per quant as a cross-platform consistency check. MNN/ORT perplexity is not measured.

## Runtime notes, attempts and failures (chronological)
### MNN 3.6.1 (works)
- Source `alibaba/MNN` tag 3.6.1 @ `d407447ed56c4121a11ccbd266dc184ca1ead0c2`, cloned to WSL `/root/src/MNN-3.6.1`.
- Cross-build: `bench/mnn/build.sh` (static, `-mcpu=cortex-a72`, MNN_BUILD_LLM/LOW_MEMORY/TRANSFORMER_FUSE ON,
  ARM82 ON (runtime-dispatched, A72 reports `i8sdot:0, fp16:0`), KLEIDIAI OFF, HTTP OFF) →
  `build/baselines/mnn/{llm_demo,llm_bench,mnn_bench}` + `BUILD_INFO.txt`.
- Model: `taobao-mnn/Qwen3.5-0.8B-MNN` @ `3208b55b2e3c948016c9d380a697be881a1f12f8` (export_args: quant_bit 4,
  block 64, HQQ, asym, lm_head 4-bit, embed bf16, tied). **Not text-only**: `is_visual: true`, ships visual.mnn.
  Only llm.mnn / llm.mnn.weight / llm.mnn.json / tokenizer.txt / configs pushed (no visual.mnn) → MNN prints
  "Can't open visual.mnn" and `load()` returns false for the vision part; text inference is unaffected
  (`llm_bench` ignores it too). Our wrapper records `load_ok:false`.
- Wrapper `bench/mnn/mnn_bench.cpp` (Llm::response on the 64 ids, MNN's prefill_us/decode_us counters,
  sampler greedy, reuse_kv false, precision low, memory low), sweep `bench/mnn/sweep.sh`.
- Failed attempt: text-only config (`bench/mnn/make_text_config.py` → `config_text.json` with `is_visual:false`)
  → plain `Llm` class → **Bus error (SIGBUS, exit 135)** on the Pi (likely 1-D position ids for this
  M-RoPE graph). Not used; files left in the model dir.
- MNN's own `llm_bench -p 8 -n 8 -kv true -t 4`: 26.0 prefill / 6.46 decode tok/s (smoke, not a result).

### ONNX Runtime GenAI 0.18.0 (int4 works, slow)
- WSL venv `/root/venv-ortgen` (uv-managed CPython 3.13.14): onnxruntime-genai 0.18.0, onnxruntime 1.31.0,
  torch 2.14.1+cpu, transformers 5.20.2, onnx 1.23.2, onnx-ir 1.1.0, onnxscript 0.7.2, pyarrow.
- Export: `bench/ort/export.sh <int4-acc4|int4-acc1|int8-acc4|int8-acc1>` =
  `python -m onnxruntime_genai.models.builder -m models/Qwen3.5-0.8B -o models/baselines/ort/<v> -p <int4|int8> -e cpu --extra_options accuracy_level=<4|1> hf_token=false`
  (builder defaults: RTN, symmetric, block 32; tied embedding shared for int4). int4: 541 MB data;
  int8: 1.93 GB data because the builder keeps `embed_tokens` **fp32** ("Gather only supports 4 bits quantization").
  The builder also exports an MTP head and sets `mtp.enabled: true`.
- Builder bug: `--help` crashes (`ValueError: unsupported format character ' '` in a help string).
- Load bug: `og.Model()` fails on the CPU EP with **"GetMemoryInfo for CPU should not be used."** because of
  the `shared_initializers` section the builder writes (x86 and aarch64; a `Config.overlay` with an empty
  list does not clear it). Workaround `bench/ort/fix_config.py` (keeps `genai_config.orig.json`, removes the section).
- `onnxruntime_genai` import fails ("Failed to load onnxruntime") unless `onnxruntime` is imported first.
- MTP speculation never engaged via the Python API (0 rounds even with `mtp.enabled`); all rows are plain AR.
- Wrapper `bench/ort/run_bench.py`, sweep `bench/ort/sweep.sh`, driver `bench/common/run_baselines.sh`.
- accuracy_level 4 (CompInt8) vs 1 could not be compared: the acc1 dirs symlinked `model.onnx.data` to the acc4
  dir and ORT 1.31 rejects it: **"External data path validation failed … External data path escapes model
  directory"** (`bench/results/ort-qwen35-0.8b-int4-acc1_20261021_231428.json`, `…int8-acc1_20261021_232559.json`,
  valid=false). Fix if rerun: hard links or copies, not symlinks. Which MLAS kernel acc4 uses on a
  non-dotprod A72 is therefore unverified.
- **int8 run aborted by lead (too slow, blocking):** `ort-qwen35-0.8b-int8-acc4`, threads=1, held the lock 38 min,
  killed at 23:09 (`bench/results/ort-qwen35-0.8b-int8-acc4_20261021_230818.json`, valid=false, "Terminated").
  If rerun: threads=4 only, 1 warm-up + 2 reps, ≤10 min.
- Prefill with 4 threads (9.86) is slower than 3 threads (10.96); decode scales ~linearly with threads → ORT int4
  decode is compute-bound on A72 (≈0.84 tok/s/thread), not bandwidth-bound.

### LiteRT-LM (prebuilt: impossible on Pi 4; source build: not tested on Pi)
- PyPI `litert-lm` 0.18.1 + `litert-lm-api` 0.18.1 (`manylinux_2_27_aarch64`) installed offline into
  `~/pinfer/venvs/litert`; `litert-lm --help` aborts:
  **"FATAL ERROR: This binary was compiled with aes enabled, but this feature is not available on this processor (go/sigill-fail-fast)."**
- Offline scan of `liblitert-lm.so` in api wheels 0.15.0, 0.16.1, 0.17.1, 0.18.1: all contain fail-fast checks for
  `aes crc dotprod fp16 lse rcpc rdm sha2` → every prebuilt aarch64 wheel needs ARMv8.2+ (Pi 5 / A76), none runs on A72.
  This is why a Pi 5 number exists and a Pi 4 number does not.
- Model: `litert-community/Qwen3.5-0.8B` @ `ff11c109e8e56797cd1ee5befa856975bb465afc`, `Qwen3.5-0.8B_int8.litertlm`
  sha256 `25447e2550082aed6145bbc755ec4da5d535f98f6a0c386865fdae2ccb88aed1` (2026-10-16 file).
- Source build (background sub-agent): `bench/litert/build.sh v0.18.1 4` (CMake super-build, `-march=armv8-a+crc
  -mtune=cortex-a72`, static; needed pinning TF `b4449a41` / LiteRT `ca21dc99`, several link fixes, Samsung NPU
  vendor disabled due to a GCC 15.2 ICE) → `build/baselines/litert/litert_lm_main` (sha256 `95369e77…`) and
  `litert_lm_advanced_main` (sha256 `d1d4adfa…`), `BUILD_INFO.txt`, `VERIFY.txt`. ISA scan: no fail-fast strings,
  no aes/sha/rcpc/rdm instructions; dotprod/fp16/i8mm only in runtime-dispatched XNNPACK/KleidiAI kernels.
  Under `qemu-aarch64 -cpu cortex-a72` it answered correctly and on `prompt64.txt` produced " hope, it was the
  spring of despair" (first 6 tokens = HF reference); qemu peak RSS 1975 MB. Took far beyond the time box;
  **stopped by the lead before any Pi run**. Not a stock runtime (patched build).
  Pi command if resumed: `litert_lm_advanced_main --backend=cpu --model_path=<.litertlm> --use_session --async=false
  --input_prompt_file=bench/common/prompt64.txt --max_output_tokens=64 --benchmark --num_cpu_threads=4
  --cache_dir=<dir under ~/pinfer> --report_peak_memory_footprint` (first run writes a 765 MB XNNPACK cache; warm it first).

### PyTorch 2.14.1 + transformers 5.20.2 (smoke only)
- PyPI aarch64 `torch==2.14.1` is the CUDA build (needs `cuda-toolkit==13.0.3`) → used
  `torch-2.14.1+cpu-cp313-cp313-manylinux_2_28_aarch64.whl` from download.pytorch.org/whl/cpu.
- **No SIGILL** on the A72: matmul/softmax/int64 add OK; `get_cpu_capability()` = DEFAULT, oneDNN 3.12, OpenMP.
- Wrapper `bench/torch/run_bench.py` / `sweep.sh` (manual KV-cache greedy loop). PC check: fp32 matches the HF
  reference 64/64; but peak RSS on x86 was 4.77 GB (bf16→fp32 + mmapped safetensors), above the Pi's ~3.5 GB.
  transformers falls back to its reference PyTorch implementations (no `causal_conv1d`, no `flash-linear-attention`).
- **Not run on the Pi** (model not pushed; stopped by lead). If resumed: threads=4, 1 warm-up + 2 reps, n_gen 32,
  expect swap/OOM for fp32; bf16 is the fallback.

## Pi changes made by this agent (user-level only; no apt, no system config)
| Path on the Pi | What | Undo |
|---|---|---|
| `~/pinfer/wheels/{ort,litert,torch}/` | offline wheels (ort 47 MB, litert 47 MB, torch incl. `torch-2.14.1+cpu` 159 MB; CUDA torch wheel removed) | `rm -rf ~/pinfer/wheels` |
| `~/pinfer/venvs/ort` | python3 -m venv; onnxruntime-genai 0.18.0, onnxruntime 1.31.0, numpy 2.5.4, psutil 7.2.2 (+flatbuffers, packaging, protobuf) | `rm -rf ~/pinfer/venvs/ort` |
| `~/pinfer/venvs/litert` | litert-lm 0.18.1, litert-lm-api 0.18.1, litert-lm-builder 0.18.1 (+deps) — **unusable on A72** | `rm -rf ~/pinfer/venvs/litert` |
| `~/pinfer/venvs/torch` | torch 2.14.1+cpu, transformers 5.20.2, tokenizers 0.23.3, safetensors 0.8.0, numpy 2.5.4 (+deps) | `rm -rf ~/pinfer/venvs/torch` |
| `~/pinfer/baselines/bench/{ort,mnn,common}/` | wrappers + prompt (copies of the repo files) | `rm -rf ~/pinfer/baselines/bench` |
| `~/pinfer/baselines/mnn/` | static MNN binaries (`llm_demo`, `llm_bench`, `mnn_bench`, BUILD_INFO.txt) | `rm -rf ~/pinfer/baselines/mnn` |
| `~/pinfer/baselines/models/Qwen3.5-0.8B-MNN/` | MNN model (no visual.mnn), SHA256SUMS verified on the Pi | `rm -rf …/Qwen3.5-0.8B-MNN` |
| `~/pinfer/baselines/models/ort/{int4-acc4,int8-acc4}` (+ `int4-acc1`, `int8-acc1` with symlinked data — broken, see above) | ORT GenAI exports (fixed config) | `rm -rf ~/pinfer/baselines/models/ort` |
| `~/pinfer/baselines/models/Qwen3.5-0.8B-litert/` | `.litertlm` (965 MB, sha256 verified) | `rm -rf …/Qwen3.5-0.8B-litert` |
| `~/pinfer/baselines/tmp` (if created by MNN `tmp_path`) | MNN scratch | `rm -rf ~/pinfer/baselines/tmp` |
Whole undo: `rm -rf ~/pinfer/baselines ~/pinfer/venvs ~/pinfer/wheels` (only if no other agent uses `venvs/`).

## PC/WSL changes (no Windows changes)
- WSL venv `/root/venv-ortgen` (see ORT); `pyarrow` installed there.
- WSL sources: `/root/src/MNN-3.6.1` (+`build-pi4`), `/root/src/LiteRT-LM-v0.18.1` (5 cmake files modified by build.sh),
  `/root/src/llama.cpp-v0.7.0/build-host/bin/llama-perplexity` (new host target).
- WSL apt (by the LiteRT sub-agent): `openjdk-17-jre-headless` 17.0.20.1+1-1~26.04, `rustup` 1.27.1-8 + deps
  (java-common, ca-certificates-java, libnss3, libnspr4, libasound2t64, alsa-ucm-conf, alsa-topology-conf,
  libpcsclite1, libjpeg-turbo8, libjpeg8); rustup stable (rustc/cargo 1.99.1) + `aarch64-unknown-linux-gnu` target.
  `/root/cache/*` (litert build tree 18 GB, uv binary, zlib, SafeInt). Undo: `apt purge openjdk-17-jre-headless rustup`, `rm -rf /root/.rustup /root/.cargo /root/cache/litert-*`.
- Note: `/etc/wsl.conf` default user became `optim` at 23:28 (not by this agent) → use `wsl.exe -d Ubuntu -u root`.
- Large artifacts (gitignored): `build/baselines/` (wheels, stage dirs, mnn, litert binaries), `models/baselines/`
  (MNN, LiteRT, ORT exports, wikitext parquet).

## Gotchas hit
- From Git Bash, `wsl.exe -- bash -c '…$VAR…'` expands `$VAR` in WSL's outer shell → use script files.
- Python `Path.write_text` on Windows writes CRLF → broke `ppl.sh` once; scripts converted to LF.
- I pushed and checksummed large files without the lock once (~22:03), which invalidated another agent's sweep;
  heavy commands now go through `run --lock`, and pushes happen only when no bench is running.

## Recommendations
1. **MNN 3.6.1 is the strongest standard baseline on the Pi 4**: 6.67 tok/s decode @2 threads (vs llama.cpp Q4_0 5.59
   @4) and 30.9 tok/s prefill @4 (2× llama.cpp). pinfer should be measured against MNN as well as llama.cpp.
   Caveat: different int4 recipe (HQQ asym b64, int4 lm_head); its quality was not measured.
2. ORT GenAI is not competitive on A72 (3.10 tok/s decode @4, prefill 9.9): drop it after one int8 4-thread point.
3. LiteRT-LM: report "prebuilt does not run on Pi 4 (ARMv8.2 required)". The patched source build is ready for one
   ≤10-min Pi run if the lead wants it, labelled as a non-stock build.
4. Quality: Q4_0 costs +23 % PPL vs bf16, Q4_K_M +3.9 %, Q8_0 +0.15 % → pinfer's 4-bit format should target
   Q4_K_M-level quality, not Q4_0. Pi-side PPL (8–16 chunks) is still open.
