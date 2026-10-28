# Public runtime comparison: M2 Pro / Qwen3.5-0.8B

Measured **2026-10-28**, on the same 10-core CPU / 16-core GPU M2 Pro with 16 GiB
unified memory. Three rounds, identical prompt **token IDs**, greedy generation,
128 generated IDs / 127 timed decode steps. No speculation. All runs retained.

**Result:** nibblemac CPU leads median decode on the two short prompts, but the
public GPU runtimes lead prefill and the 512-token request. nibblemac's current
Metal hybrid is slower than both public GPU baselines. This is **not** an
equal-quantization comparison or evidence of overall runtime superiority.

## Decode: tokens/sec

Each cell is **median [min–max]** across three rounds.

| Runtime | Prose, 64 prompt tokens | Chat, 36 prompt tokens | Repeated prose, 512 prompt tokens |
|---|---:|---:|---:|
| nibblemac CPU SDOT, 6 threads | **193.3** [161.3–220.1] | **204.3** [73.7–216.2] | 139.5 [128.9–189.5] |
| nibblemac Metal head-only, 6 CPU threads | 122.1 [112.1–126.9] | 120.4 [90.7–136.6] | 112.2 [100.9–125.3] |
| llama.cpp Metal, all layers offloaded | 148.2 [140.9–155.9] | 150.4 [142.0–156.2] | 150.0 [145.1–150.2] |
| MLX-LM Q4/group-32, synchronized adapter | 169.6 [145.0–171.7] | 157.5 [144.8–169.5] | **153.7** [145.8–168.5] |

**Timing variability matters:** one nibblemac CPU chat sample fell to 73.7 tok/s;
the 512-token CPU samples span 128.9–189.5. AC power and no recorded thermal
warning do not establish stable clocks or an idle computer. The cause is
unconfirmed. Do not substitute older, faster CPU numbers into this table or
interpret these medians as a stable universal ranking.

## Prefill: tokens/sec

Includes the final vocabulary projection; model loading and warmup excluded.

| Runtime | Prose 64 | Chat 36 | Repeated prose 512 |
|---|---:|---:|---:|
| nibblemac CPU SDOT, 6 threads | 807.6 [694.6–855.5] | 715.9 [668.5–780.9] | 731.3 [603.3–801.6] |
| nibblemac Metal head-only | 732.9 [618.1–774.2] | 713.5 [448.1–804.5] | 721.5 [423.0–775.4] |
| llama.cpp Metal | **1990.8** [1934.0–2010.6] | **1213.4** [1152.5–1217.5] | **2062.5** [2005.4–2078.8] |
| MLX-LM synchronized adapter | 1676.4 [1579.1–1708.7] | 1019.0 [922.8–1022.9] | 2030.7 [1984.3–2043.2] |

Prefill + generation **median compute seconds per request**, excluding loading,
warmup, tokenization, detokenization and process startup:

| Runtime | Prose 64 + 128 IDs | Chat 36 + 128 IDs | Prose 512 + 128 IDs |
|---|---:|---:|---:|
| nibblemac CPU SDOT | **0.736** | **0.672** | 1.611 |
| nibblemac Metal head-only | 1.127 | 1.105 | 1.968 |
| llama.cpp Metal | 0.889 | 0.874 | 1.101 |
| MLX-LM synchronized adapter | 0.787 | 0.842 | **1.078** |

Computed per run as `prompt_tokens / prefill_tps + 127 / decode_tps`, then
median-reduced. These are not application-level end-to-end or cold-start latencies.

## Quality and memory

WikiText-2 raw test prefix: **8 independent 512-token chunks**, reset state per
chunk, score positions 256..510 predicting 257..511: **2040 scored tokens**.
Same corpus, token IDs, boundaries and scoring formula across all three models.

| Runtime / model | Perplexity (lower is better) | Weight file size, decimal MB | Median process peak RSS across prompts, MiB |
|---|---:|---:|---:|
| nibblemac CPU, PQ Q4K/A8 | 18.5061 | 436.5 | 435–448 |
| nibblemac Metal, same weights | 18.5061* | 436.5 | 583–596 |
| llama.cpp, public Q4_K_M | 18.2005 | 532.5 | 699–700 |
| MLX-LM, affine Q4/group-32 | 20.5333 | 471.0 | 1004–1024 |

*Metal quality is established by the earlier exact CPU/Metal quality gates, not
a fourth quality run in this comparison. This sweep checks CPU/Metal generated
token equality; the earlier [full-model gates](MAC_BENCHMARKS.md) also check logits,
state and speculation. The short corpus is not a broad quality evaluation;
nibblemac's reported chunk-based PPL error is 1.24833. Do not infer statistical
significance from the small nibblemac/llama PPL difference.

RSS is Darwin process high-water RSS, including loading and warmup. It is **not**
total GPU/unified-memory allocation or a runtime-only working-set measurement.
The MLX Python process includes Transformers/tokenizer machinery; MLX separately
reports peak allocator use of about 609–727 MiB, which must not be added to RSS.

## Models and versions: important differences

- **nibblemac:** official `Qwen/Qwen3.5-0.8B` revision
  `2fc06364715b967f1860aea9cf38778875588b17`, locally converted with the inherited
  PQ Q4K/group-32 quantizer and int8 activation dots; fp32 KV/state. Model digest
  `3573deef290ff466f1295b023688ae3202c85d97600deadbe1e913c2a53ec350`.
- **llama.cpp:** unmodified public source commit
  `f4d8af608a6b39946406f9439edfb6e83ca6a923`, Release/Metal/Accelerate,
  full GPU offload confirmed as 25/25 layers; flash attention enabled, f16 K/V,
  six CPU/batch threads, context 2048, batch/microbatch 64. Uses public
  [Unsloth Q4_K_M GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF), revision
  `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`, digest
  `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`.
  Its conversion provenance is not guaranteed identical to the official local
  snapshot; Q4_K_M uses a different, mixed quantization recipe.
- **MLX-LM 0.32.2 / MLX and mlx-metal 0.32.4:** locally converted from the **same
  pinned official checkpoint as nibblemac**, affine four-bit/group-32,
  bfloat16 nonquantized tensors; converter reports **5.007 bits/weight** overall.
  Weight digest `bbd765b3fb4f8f34640da8b00ddf2af2c464bf4bcd6c2ba161b49f06ae0cfa16`.
  K/V and convolution state are bfloat16; recurrent matrices are fp32 (checked
  on the loaded model). Dynamic cache grows for the actual shared positions;
  unlike nibblemac/llama it does not preallocate a fixed 2048-token context.
  This is a reproducible local conversion using a public runtime, **not a
  downloaded prequantized MLX-community checkpoint**.

All model families are Qwen3.5-0.8B, but the files, quantization and arithmetic
are **not identical**. Four-bit labels do not imply equal storage or quality.
Do not interpret speed differences as isolated backend improvements. No nibblemac
precision or fast-math setting was changed; public libraries use their own normal
arithmetic/optimization policies.

## Timing and validation details

- Identical pretokenized prompts from `tools/bench_inference_macos.py`; tokenizers
  are not run in the timed region. One process/request at a time under the same
  advisory `build/macos-bench.lock`; configuration order reverses in round two.
- Same-prompt prefill + four decode-step warmup, then fresh/reset state. Shader
  compilation and loading excluded. macOS 27.2 / build 26B118; AC power before/after.
  No energy, temperature, clock-frequency or competing-load isolation measurement.
- First-token **logits** belong to prefill; rate is `127 / decode_seconds`. The
  inherited nibblemac CLI charges the first argmax to decode; public adapters
  charge it to prefill. This small boundary difference is retained and disclosed.
- MLX adapter uses public model/cache APIs with explicit evaluation and scalar
  token retrieval, not the asynchronous one-token-look-ahead `generate_step`
  throughput counter. It skips log-softmax and pre-fills all but the last token
  in chunks of up to 64, then evaluates the last token, as MLX-LM does. This
  prevents queued decode work crossing the timing boundary, but is **not a
  stock MLX-LM CLI performance claim**. A separate actual-device check matches
  its first 16 IDs exactly against native `generate_step` on all three prompts.
- llama adapter uses the current public batch/process API and host greedy argmax;
  GPU-side sampling and further runtime-specific tuning were not tested. This is
  not the stock `llama-bench` synthetic-token workload or a server benchmark.
- EOS is ignored for fixed-length throughput. Public models need not generate
  identical IDs across quantizations. Each runtime repeats its own full 128-ID
  continuation exactly across all three rounds; nibblemac CPU/Metal match each
  other. Finite public logits and finite/count-checked metrics are validated.
- `valid=true` means execution, parsing, metric/count checks, repeatability and
  nibblemac CPU/Metal token checks passed, **not** equal public-model quality,
  state/speculation parity, or timing isolation. No public-runtime state/rewind
  or speculative decoding parity is claimed.

## Evidence

- [36 throughput runs + three quality runs, full logs, IDs, hashes and metadata](../bench/results/metal/public-baselines-20261028T071703.539711Z.json).
- [MLX synchronized adapter vs native API, three exact 16-token cases](../bench/results/metal/public-mlx-api-gates-20261028T072130.867836Z.json).
- [Initial eight-ID smoke run](../bench/results/metal/public-smoke-20261028T071605.579337Z.json),
  preserved but **not** used in the tables. Its earlier MLX prefill eagerly
  materialized more logits; the final adapter follows MLX-LM's last-token split.
- `tests/test_public_baselines.py`: offline pinned-download, failure retention,
  argument, GPU-offload, token-count and invalid-metric checks.

The final record's protocol wording was clarified after measurement and one
parsed model path was redacted; measured values/logs were not changed. Source
hashes retain the measurement-time versions; later recorder validation/redaction
hardening does not change either runtime adapter.

## Reproduce

First complete nibblemac's [model setup](MAC_BENCHMARKS.md), then:

```sh
.venv/bin/python -m pip install -r requirements-public-macos.txt
python3 tools/setup_public_baselines.py
.venv/bin/python -m mlx_lm convert --hf-path models/Qwen3.5-0.8B \
  --mlx-path models/public-baselines/mlx-q4-g32 \
  --quantize --q-group-size 32 --q-bits 4 --dtype bfloat16
git clone https://github.com/ggml-org/llama.cpp.git build/third-party/llama.cpp
git -C build/third-party/llama.cpp checkout f4d8af608a6b39946406f9439edfb6e83ca6a923
.venv/bin/cmake -S build/third-party/llama.cpp -B build/public-llama -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.2.sdk \
  '-DCMAKE_CXX_FLAGS=-isystem /Library/Developer/CommandLineTools/SDKs/MacOSX26.2.sdk/usr/include/c++/v1'
.venv/bin/cmake --build build/public-llama --target llama-bench llama-perplexity --parallel 6
xcrun clang++ -std=c++17 -O3 \
  -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX26.2.sdk \
  -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX26.2.sdk/usr/include/c++/v1 \
  -Ibuild/third-party/llama.cpp/include -Ibuild/third-party/llama.cpp/ggml/include \
  tools/public_llama_bench.cpp -Lbuild/public-llama/bin -lllama -lggml -lggml-base \
  -Wl,-rpath,@executable_path/public-llama/bin -o build/public-llama-adapter
make mac-cpu mac-infer
.venv/bin/python tests/test_public_baselines.py
.venv/bin/python tests/test_public_mlx_adapter.py
.venv/bin/python tools/bench_public_macos.py --rounds 3 --generate 128 --quality-chunks 8
```

SDK paths above describe the measured machine; adjust to an installed SDK on
another Mac. Initial compilation failed with missing libc++ `<array>` headers;
the explicit SDK/header configuration fixed it without changing system settings
or upstream sources. Embedded Metal source avoids needing offline Metal tools.
Dependencies, weights and builds stay in ignored `.venv/`, `models/` and `build/`.
Do not run concurrent heavy jobs during measurement.
