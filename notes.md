# Notes and findings

## 2026-10-21 — PC environment
- Windows 11, PC on wired Ethernet (Realtek 2.5GbE), Wi-Fi off.
- Present: git, OpenSSH (ssh/scp), Python 3.13, uv, QEMU (`qemu-system-aarch64`).
- WSL2 Ubuntu 26.04, 20 threads, 31 GB RAM, runs as root.
- Installed in WSL: build-essential, cmake, ninja, gcc/g++-aarch64-linux-gnu 15.2,
  mmdebstrap 1.5.7, qemu-user + qemu-user-binfmt (aarch64 binfmt registered),
  device-tree-compiler, bison, flex, libssl-dev, dosfstools, parted, kpartx.
- Gotcha: `qemu-user-static` no longer exists on Ubuntu 26.04 → use `qemu-user-binfmt`.
- Gotcha: from Git Bash, call WSL with `MSYS_NO_PATHCONV=1` or `/mnt/...` paths get mangled.
- Existing SSH key: `~/.ssh/id_ed25519_desktop(.pub)`.

## 2026-10-21 — Flashing stock Raspberry Pi OS (trixie 2026-10-06)
- First boot is **cloud-init (NoCloud)** from boot partition `user-data`, `network-config`,
  `meta-data`; the old `firstrun.sh`/`custom.toml` paths are not used.
- Image ships a placeholder user `pi` (nologin). Pi's cloud-init distro class renames the
  first user in `users:` via userconf and masks `userconfig.service`.
- netplan → NetworkManager; a 64-hex PSK renders to `key-mgmt=wpa-psk`, `pmf=2` (optional).
  Router <wifi-ssid> is WPA2/WPA3 mixed; WPA3-only would need a plaintext passphrase.
- Validated offline: `cloud-init schema` + `netplan generate --root-dir` inside a qemu chroot.
- Windows raw write: take disk offline, write `\\.\PhysicalDriveN` (elevated). After the disk
  comes back online, Windows adds `System Volume Information` to the FAT partition, so a
  whole-disk hash will not match; compare per partition instead (os/stock/compare_disk.ps1).

## Pi 4 facts (to verify on device in P0/P1)
- `/proc/cpuinfo` Features should show `asimd` but not `asimddp` (no dot product).
- Page size via `getconf PAGESIZE` (expect 4096).
- Check `vcgencmd get_throttled` = `0x0` before trusting any benchmark.

## 2026-10-21 — First boot on the real Pi
- Board: Raspberry Pi 4 Model B **Rev 1.1** (early board, 1.5 GHz max), boots from USB SSD
  (`/dev/sda2`, resized to 917 GB). Kernel 6.18.50+rpt-rpi-v8. Python 3.13.5.
- Network: wlan0 on <wifi-ssid>, 192.168.1.166 (eth0 down). Internet OK (ping 1.1.1.1 ~5 ms).
- **`pi4.local` resolves to a public IP (218.38.137.27) on this PC** — a DNS server answers
  `.local`, so mDNS is never used. Harness uses the IP. Fix later: DHCP reservation on router.
- CPU features: `fp asimd evtstrm crc32 cpuid` → confirmed no dot product, no FP16 math.
- Stock kernel config: `NO_HZ_FULL` off, `HUGETLBFS` off, `TRANSPARENT_HUGEPAGE` off,
  `CPU_ISOLATION=y`, `PREEMPT=y`, 4 KB pages. → `isolcpus` works on stock, but tickless
  cores and huge pages **need our own kernel build** (design §4.2 risk confirmed).
- Bootloader EEPROM update available (not applied).

## Baseline numbers
| Date | Bench | Result | Notes |
|---|---|---|---|
| 2026-10-21 | memband read, 256 MB | 1 thr 4.20 / 2 thr 4.08 / 4 thr 4.07 GB/s | 38 °C, not throttled, governor ondemand |

### llama.cpp v0.7.0 (`620d2225`), Qwen3.5-0.8B, llama-bench pp64/tg64, 3 reps, governor ondemand
| Quant | File | tg64 tok/s t=1/2/3/4 | pp64 tok/s t=1/2/3/4 | Best tg vs ceiling (4.2 GB/s ÷ file) | Result |
|---|---|---|---|---|---|
| Q4_0 | 513 MB | 2.32 / 4.20 / 5.58 / **5.59** | 4.28 / 8.33 / 12.08 / 15.68 | 5.59 / 8.2 = 68% | `bench/results/llamacpp-qwen35-0.8b-Q4_0_20261021_213343.json` |
| Q4_K_M | 542 MB | 2.26 / 4.06 / 5.38 / **5.41** | 4.03 / 7.84 / 11.35 / 14.82 | 5.41 / 7.7 = 70% | `…Q4_K_M_20261021_213929.json` |
| Q8_0 | 834 MB | 2.83 / **4.12** / 4.04 / 3.60 | 4.73 / 9.22 / 13.22 / 17.06 | 4.12 / 5.0 = 82% | `…Q8_0_20261021_214506.json` |

Observations: decode stops scaling at 3 threads (4-bit) and **gets slower beyond 2 threads at
8-bit** (memory contention); prefill scales ~linearly with threads (compute-bound). All runs
valid (no throttling, ≤55 °C).

**Key finding:** one core already saturates DRAM (~4.2 GB/s). Extra cores add no bandwidth;
they are only useful for the compute (dequantization, math) that must keep pace with it.
Decode ceiling ≈ 4.2 GB/s ÷ weight bytes per token, e.g. ~0.4 GB of 4-bit weights → ~10 tok/s.

### pinfer MTP speculative decoding, bf16, 64 prompt + 64 gen tokens (2026-10-21, details `engine/SPEC.md` §4)
| Kernels | Prompt | plain tok/s | best spec config | spec tok/s | Result |
|---|---|---|---|---|---|
| ref, 1 thr | bench64 | 1.546 | k=1 | 1.151 (0.74×) | `bench/results/pinfer-spec-ref-k{0,1}_20261021_*.json` |
| fast, 4 thr | bench64 | 2.274 | k=3, draft vocab 32768 | 3.993 (1.76×) | `bench/results/pinfer-spec-fast-k3-v32768_20261021_233457.json` |
| fast2, 4 thr | bench64 | 2.308 | k=2, draft vocab 32768 | 4.130 (1.79×) | `bench/results/pinfer-spec-fast2-dickens64-k2-v32768_20261021_234027.json` |
| fast2, 4 thr | French history | 2.326 | k=2, draft vocab 32768 | 3.272 (1.41×) | `bench/results/pinfer-spec-fast2-french-k2-v32768_20261021_233722.json` |
| fast2, 4 thr | quicksort code | 2.331 | k=2, draft vocab 32768 | 4.503 (1.93×) | `bench/results/pinfer-spec-fast2-quicksort-k2-v32768_20261021_233856.json` |

## Scoreboard — Qwen3.5-0.8B on the Pi 4 (2026-10-21/22, 64-token prompt `bench/common/prompt64`, 64 generated, all runs valid)
Speed: `bench/results/pinfer-fast-*_20261021_23*.json`, `llamacpp-*`, `mnn-*`, `ort-*`.
Quality: wikitext-2 test, first 8 chunks × 512 (2040 scored tokens), llama-perplexity protocol
(`bench/quality/ppl_pc_c8.sh` for GGUF on the PC, `engine/tools/pinfer_ppl` on the Pi,
tool cross-checked on 2 chunks: 15.1743 vs llama.cpp 15.1695).

| Engine | Format | File | Decode tok/s plain | Decode tok/s MTP spec | Prefill tok/s | PPL (c8) |
|---|---|---|---|---|---|---|
| pinfer | Q4_G32 + Q4 head | 437 MB | **6.88** | **12.56** (k=3, dv 32768) | 21.3 | **21.61** (as bad as Q4_0) |
| pinfer | **Q4K_G32 everywhere + Q4K head** | 437 MB | **7.88** | **16.49** (k=3, dv 32768) | 37.4 | **18.50** (18.48 before the NEON expf; difference 0.02 ≪ ±1.25) ← default (vectorized engine `fba8a1a`, `bench/results/pinfer-final2-q4k_20261022_023950.json`; earlier: 7.26 / 13.35 / 26.1 at `af06041`, 6.83 / 11.57 / 19.3 before the kernel follow-up) |
| pinfer | Q4_G32 layers + Q4K head | 437 MB | — | — | — | 21.58 (loss is in the layers) |
| pinfer | Q4K (v_proj, mtp.fc Q8) + Q4K head | 439 MB | **6.88** | 10.98 (k=3, dv 32768) | 16.3 | **18.50** |
| pinfer | same + Q8 head | 567 MB | 5.72 | 9.77 (k=3) | 13.7 | 18.29 |
| pinfer | Q8_G32 + Q8 head | 823 MB | 4.22 | 8.73 (k=4, dv 16384) | 12.2 | 17.96 (= Q8_0 17.97, bf16 17.93) |
| pinfer | bf16 | 1.55 GB | 2.27 | 4.13 (k=2) | 11.0 | = reference |
| MNN 3.6 | int4 HQQ block64 | ~450 MB | 6.67 (t=2) | — | **30.9** | — |
| llama.cpp v0.7.0 | Q4_0 | 513 MB | 5.59 | — | 15.7 | 21.25 |
| llama.cpp v0.7.0 | Q4_K_M | 542 MB | 5.41 | — | 14.8 | 18.59 |
| llama.cpp v0.7.0 | Q8_0 | 834 MB | 4.12 (t=2) | — | 17.1 | 17.97 |
| llama.cpp v0.7.0 | bf16 | 1.56 GB | — | — | — | 17.93 |
| ONNX Runtime GenAI 0.18 | int4 rtn acc4 | — | 3.10 | — | 9.9 | — |
| LiteRT-LM 0.15–0.18.1 (prebuilt) | int8 `.litertlm` | 965 MB | **does not run on Pi 4**: binary needs aes/dotprod/fp16/lse (ARMv8.2) | — | — | — |
| PyTorch 2.14.1+cpu + transformers 5.20.2 | fp32 | 3 GB | not timed: 4.8 GB peak RSS > 3.5 GB free (runs on the A72, no SIGILL) | — | — | 64/64 greedy match on PC |

**Quality verdict:** pure Q4_G32 loses as much as llama.cpp Q4_0 (21.6 vs 21.3); the mixed
Q4K file (v_proj + mtp.fc at Q8, Q4K head) matches Q4_K_M quality (18.50 vs 18.59) at 2.0×
its speed with speculation (10.98 vs 5.41 tok/s). A Q8 head buys only −0.2 PPL for −11% speed.
Isolation (00:20–00:30): Q4K everywhere scores 18.48 — the Q8 overrides buy nothing, and
Q4 layers + Q4K head scores 21.58, so the symmetric Q4_G32 *layers* are what loses quality
(like llama.cpp's Q4_0); Q4K_G32's asymmetric 32-column groups fix it at the same 4.5 bits.
**Recommended default: `q4k-headq4k`** (PPL 18.48–18.50 ≈ Q4_K_M; speed with the current engine: 7.88 plain / 16.49 spec / 37.4 prefill — see the scoreboard row).

Baselines notebook: `bench/BASELINES.md` (MNN first token differs from the fp32 reference at a
near-tie, like ORT int4 — quantization flip, not a bug). PC-side 64-chunk perplexity (16,320
tokens): bf16 17.94, Q8_0 17.97, Q4_K_M 18.64, Q4_0 22.05 — same ordering as the 8-chunk slice.

Speculative decoding is exact: identical tokens to plain greedy (`tests/test_spec.py` on the host;
on the Pi `engine/bench/spec_exact.sh` 20/20 for both models × 2 prompts × k=0..4 with the final
NEON kernels, `bench/results/spec-exact-final_20261022_*.json`). The earlier divergence (batched
verify pass rounding the Q4/Q4K offset term differently from the single-token pass) is fixed.
Verify-pass cost vs one decode step (Q4K, vectorized engine): 1.19 (k=2) / 1.28 (k=3) / 1.61 (k=4);
per plain step: matmul 118.6 ms, recurrence 6.5, everything else 3.2 → the DRAM floor (~113 ms for
473 MB/token) is 12% away. Prefill 37.4 now leads MNN (30.9).

## Context window — Qwen3.5-0.8B Q4K on the Pi 4 (2026-10-22 02:30–, `engine/CONTEXT.md`)
Prompts: the first N tokens of wikitext-2 test (`bench/common/ctx{64,512,2048,4096,8192}.ids`), 32
generated, 4 threads, `pinfer_run_ctx` (attention rewrite, bit-identical: spec_exact 20/20,
NEON twin check 5/5). "before" = the vectorized core2 binary of 02:20 (512 / 2048 rows: the
lead's runs `pinfer-ctx{512,2048}-q4k_20261022_02*.json`).

| ctx | plain tok/s before → after | prefill tok/s before → after | spec k=3 tok/s before → after | RSS MB | llama.cpp Q4_K_M tg32 at that depth | result files |
|---|---|---|---|---|---|---|
| 64 | 7.81 → 7.92 | 37.4 → 35.6 | 16.74 (64 gen) → 14.64 (32 gen) | 432 / 451 | 5.48 (d=0) | `pinfer-ctx-p64-new-q4k_20261022_025529.json` |
| 512 | 6.20 → **7.69** | 21.6 → **36.8** | 9.04 → **15.24** | 443 / 463 | 5.23 | `pinfer-ctx512-new-q4k_20261022_025648.json` |
| 2048 | 4.12 → **7.13** | 11.6 → **32.3** | 2.39 → **7.47** | 479 / 506 | 4.32 | `pinfer-ctx2048-new-q4k_20261022_030125.json` |
| 4096 | **6.48** | **28.0** | **8.47** | 528 / 562 | 3.43 | `pinfer-ctx4096-new-q4k-{plain,spec3}_20261022_03*.json` |
| 8192 | **5.42** | **22.0** | **7.17** | 625 / 675 | — | `pinfer-ctx8192-new-q4k-{plain,spec3}_20261022_03*.json` |

llama.cpp: `llamacpp-ctx-d0-512-2048-q4km_20261022_032910.json`, `llamacpp-ctx-d4096-q4km_20261022_033523.json`
(`llama-bench -d`, its pp64 = 64 tokens after the depth: 14.8 / 13.1 / 10.3 / 7.8 tok/s).
Final binary `pinfer_run_ctx7` (blocked P·V + fused fp16 dots; NEON identity 5/5, spec_exact 20/20,
`bench/results/{ctx-check,spec-exact,f16-check}-ctx7_20261022_06*.json`): prompt64 plain 7.88 /
k=3 16.79 / adaptive 16.42; 4096 plain fp32 6.38, fp16 **6.65** (84.4% of 64-token decode),
adaptive fp32 7.93 (`pinfer-ctx7-*_20261022_06*.json`). 8192 decode-only A/B means: fp32
5.23–5.33, fp16 5.72 (`pinfer-ab-att-ctx8192-kv*_*.json`).

| KV dtype | PPL 8×512 | PPL 8×2048 | teacher-forced top-1 / KL vs fp32 (host, 512 pos.) | decode 4096 / 8192 | RSS 8192 |
|---|---|---|---|---|---|
| fp32 (default) | 18.5041 ± 1.25 | 13.9852 ± 0.46 | — | 6.38 / 5.3 | 625 MB |
| fp16 | 18.5257 | 13.9828 | 100% / 3.5e-8 | 6.65 / 5.72 | 528 MB |
| int8 | 18.5007 | 13.9892 | 99.4% / 2.3e-4 | 6.35 (ctx4) / — | 457 MB @4352 |

Adaptive speculation (`--adaptive`, 64 generated, wikitext): 2048 → 9.57 tok/s (fixed k=3 9.13,
plain 7.09), 4096 → 7.53 (7.38, 6.11) (`pinfer-adaptive2-*_20261022_05*.json`). State: 2048-token
snapshot 70.5 MB, save 94 ms, load 90 ms vs 63 s prefill, exact (`state-check-ctx6_*.json`);
`pinfer chat` turn 2: 0.68 s vs 2.10 s re-prefill. Details: `engine/CONTEXT.md`.

## Custom kernel 6.18.50-pinfer — effect on speed (2026-10-22 01:25–01:29, one-time tryboots)
Same binary (`pinfer_run_fast` @ `af06041`), same model (`q4k-headq4k`), 4 threads, 64+64 tokens.

| Kernel | cmdline extras | memband 1/2/4 thr GB/s | plain tok/s | spec k=3 tok/s | Result files |
|---|---|---|---|---|---|
| stock 6.18.50+rpt-rpi-v8 | — | 4.20 / 4.08 / 4.07 (21 Oct) | 7.26 | 13.35 | `pinfer-final-q4k_20261022_011230.json` |
| 6.18.50-pinfer (K1) | none | 4.20 / 4.18 / 4.29 | 7.28 | 13.51 | `memband-k1-*`, `pinfer-k1-pinfer-kernel-q4k_20261022_012707.json` |
| 6.18.50-pinfer (K2) | `isolcpus=nohz,domain,managed_irq,1-3 nohz_full=1-3 rcu_nocbs=1-3 irqaffinity=0` | 4.36 / 4.19 / 4.60 | 7.21 | 13.45 | `memband-k2-*`, `pinfer-k2-splitcore-q4k_20261022_012856.json` |

Verdict: the kernel build (NO_HZ_FULL/context tracking compiled in) costs nothing, and core
isolation + tickless cores change decode by < 1% — the engine is DRAM-bound and there was
little jitter to remove. 4-thread memband rises 4.07 → 4.60 GB/s under isolation (no
housekeeping on cores 1–3), which the engine does not translate into tok/s because its decode
already uses 1–3 threads for streaming. Huge pages (K3, `PINFER_HUGEPAGES=1`: anonymous copy of the model with
`madvise(MADV_HUGEPAGE)`, 428 of 437 MB landed in 2 MB pages per `AnonHugePages`): plain 7.27–7.34
vs 7.24–7.28 without, spec 13.42 vs 13.52 — **no gain either** (`pinfer-k3{a,b,c}-*_20261022_0134*`).
The A72's hardware prefetcher and page-walk caches already hide the 4 KB-page TLB cost of a
sequential weight stream. Methodology note found on the way: the first `--bench` after a reboot
reports prefill ~20 tok/s instead of ~27 because the mmap'd weights fault in from the SSD during
that first prefill (cold page cache) — decode is unaffected; warm-up to be added to `--bench`.
The Pi runs the stock kernel again (a plain reboot ends the tryboot); the custom kernel stays
installed side by side (`docs/pi-state.md` #4).

## Context length (2026-10-22 02:13–02:39, Q4K, 32 generated tokens; pre-optimization, see Phase 6)
| Context | Binary | Plain tok/s | Spec k=3 tok/s (tok/pass) | Prefill tok/s | Peak RSS | Result |
|---|---|---|---|---|---|---|
| 64 | vectorized | 7.88 | 16.49 (3.32) | 37.4 | 433 / 452 MB (plain / spec) | `pinfer-final2-q4k_20261022_023950.json` |
| 512 | pre-vectorization | 6.20 | 9.04 (3.10) | 21.6 | 463 MB | `pinfer-ctx512-q4k_20261022_021315.json` |
| 2048 | mixed (binary swapped mid-run) | 4.12 | 2.39 (1.63) | 11.6 | 505 MB | `pinfer-ctx2048-q4k_20261022_022009.json` |
| 4096 | vectorized | 5.56 | 5.69 (2.38) | 18.1 | 561 MB | `pinfer-ctx4096-q4k_20261022_023914.json` |

Memory is not the issue (KV cache 24 KB/token: +116 MB at 4096); the attention path is. Speculation
loses its gain on long wikitext (low acceptance + verify cost growing with context × batch) → adaptive k.

## Speculative decoding on chat prompts (2026-10-22 07:56–08:14, lead; `pinfer_run_fast` 9d30bbda)
Q4K, 4 threads, 96 generated tokens (pinfer_run ignores EOS), chat-templated prompts
`bench/common/chat_*.ids` (made by `bench/common/make_chat_prompts.py`), `engine/bench/chat_spec.sh`.
tok/s (tokens per pass); draft vocab 32768 unless "full":

| prompt | plain | k=2 | k=3 | adaptive k≤3 | k=3 full vocab | k=2 full vocab |
|---|---|---|---|---|---|---|
| chat27 (explain) | 7.97 | **9.85** (1.76) | 9.21 (1.83) | 9.31 (1.67) | 6.71 (1.98) | 7.81 (1.90) |
| code, Python | 7.88 | **11.71** (2.11) | 11.21 (2.21) | 10.91 (2.11) | 7.42 (2.16) | 8.44 (2.07) |
| code, C | 7.89 | **12.23** (2.21) | 12.02 (2.38) | 11.75 (2.26) | 8.08 (2.38) | 8.85 (2.16) |
| list | 7.69 | **11.15** (2.02) | 10.44 (2.07) | 10.91 (1.98) | 7.75 (2.26) | 9.24 (2.26) |
| math, step by step | 7.87 | **13.46** (2.44) | 13.20 (2.64) | 13.15 (2.57) | 9.29 (2.71) | 9.74 (2.50) |
| story | 7.67 | **9.96** (1.79) | 9.24 (1.83) | 8.80 (1.43) | 6.34 (1.86) | 7.48 (1.83) |

Files: `chat-spec-adaptive_20261022_075629.json` (plain/k3/adaptive, 2 reps), `chat-spec-vocab_20261022_080315.json`
(k2), `chat-spec-fullvocab_20261022_081036.json` (full vocab).
- On chat answers speculation gives **1.25–1.71×** (k=2), not the 2.1× of the prose-continuation
  headline (prompt64: 3.32 tokens/pass vs 1.8–2.6 here). **k=2 beats k=3 on every chat prompt**
  (+2…+8%); k=2 is already the library default (`pinfer_default_spec_params`).
- Draft vocab 32768 vs full: 5–25% of chat-answer tokens have ids ≥ 32768 (```, Fibonacci, :**,
  France…), but the full-vocabulary draft raises tokens/pass by only 0–9% and is 5× costlier →
  30% slower overall. **Frequency-ranked draft vocabulary (FR-Spec style) dropped**: at most those
  few % to gain. Low chat acceptance is the MTP head's, not the vocabulary cut.
- Adaptive k ≤ 3 is below fixed k=2 on chat (story: 1.43 tokens/pass — it falls back to plain
  too often); it stays opt-in. To fix in `spec_adapt.c`: never worse than the best fixed k.

## `pinfer serve` on the Pi 4 (2026-10-22 09:49–09:53, server agent; `docs/library.md` §10)
Q4K model, 4 threads, max_ctx 4096, fp32 KV, greedy chat requests → speculative k=2, at most 64
tokens per reply. Server and client (`bench/serve/serve_client.py`) both on the Pi, over localhost;
all four server configurations in one locked call (`bench/serve/serve_bench.sh`, NEON
`libpinfer.so` md5 `976b73eb…`). TTFT = request sent → first content delta, client side.
File: `bench/results/serve_20261022_095259.json` (valid, throttled 0x0, 49.6 → 50.6 °C).

Single request (system + user, 39-token prompt, 47-token answer): TTFT **1.33 s** streamed,
**9.0 tok/s** decode (1.68 tokens/pass; client-side SSE rate 9.06 = server 9.04, so HTTP costs
nothing measurable), 6.4 s total, blocking 6.35 s.

4-turn conversation, each turn resending the whole history (cached / prompt tokens, TTFT):

| turn | `--history generated` (default) | `--history template` | `--no-reuse` |
|---|---|---|---|
| 1 | 0 / 39, 1.34 s | 0 / 39, 1.24 s | 0 / 39, 1.25 s |
| 2 | 86 / 111, **0.96 s** | 0 / 107, 3.21 s | 0 / 107, 3.09 s |
| 3 | 147 / 169, **0.89 s** | 0 / 161, 4.82 s | 0 / 161, 4.58 s |
| 4 | 232 / 256, **0.95 s** | 0 / 244, 7.08 s | 0 / 244, 6.90 s |

- With the template's rendering (what a stateless server sends) **nothing** is reused: the prompt
  diverges inside the first reply (no `<think>\n\n</think>\n\n` there) and the linear-attention
  state cannot be rolled back to the common prefix. Building the history from the generated tokens
  reuses the whole previous exchange (77–91% of the prompt), and TTFT stays flat at ~0.9 s
  instead of growing with the conversation (7.4× faster at turn 4). Template mode is slightly
  slower than no reuse (it saves a snapshot of the replaced context it never uses, ~0.1 s).
- Two conversations interleaved (A1 B1 A2 B2): second turns start from their own snapshot,
  97 / 120 and 99 / 121 tokens cached, TTFT **1.03 / 1.01 s** vs 3.56 / 3.50 s with
  `--state-cache-mb 0`.
- Decode speed matches `pinfer_run` at the same tokens/pass (chat-prompt table above, k=2:
  1.76 → 9.85, 2.21 → 12.23 tok/s; server: 1.68 → 9.03, 2.25 → 12.34 tok/s).
