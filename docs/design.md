# pinfer: a Pi-4-native LLM runtime and OS

Status: draft v1 (2026-10-21). Decisions marked *(user)* were made by the user; *(call)* are
defaults chosen during design and open to change.

## 1. Intent

Run Qwen language models on a Raspberry Pi 4 (4 GB) as fast as the hardware allows, by
co-designing three things for this one board:

1. **pinfer**, our own inference library written for the Cortex-A72 (NEON, no dot-product).
2. A **split-core Linux** setup: Linux, SSH, Python and networking live on core 0; cores 1-3
   are isolated and run only the inference engine *(user: "level 2")*.
3. A **minimal custom Debian OS image** that boots into that setup.

The project is also a way to understand the Pi 4 architecture end to end, so every
optimization is measured on the real board and recorded.

## 2. Goals / non-goals / anti-goals

Goals
- Load Qwen weights (from Hugging Face safetensors) and generate text on the Pi 4.
- Beat llama.cpp decode speed (tokens/s) on the same Pi, same model, same bit-width
  *(call: default success bar; user has not set one)*.
- Keep Python 3 and internet on the Pi *(user)*.
- Control everything from the Windows PC: build, deploy, benchmark *(user)*.

Non-goals (for now)
- Bare-metal kernel (level 3) — later learning experiment *(user)*.
- GPU (VideoCore VI) compute — too weak to beat the CPU for this workload.
- Training. Models larger than ~3B. Other boards.

Anti-goals (would count as failure even if it "works")
- Speed numbers taken while the Pi was thermally throttled or under-volted.
- Faster but wrong: output that diverges from the reference model without us knowing.
- An OS so stripped that SSH/network break and the board becomes unrecoverable without a monitor.

## 3. Hardware facts that drive the design

| Part | Pi 4 (BCM2711) | Consequence |
|---|---|---|
| CPU | 4x Cortex-A72, ARMv8.0-A, 1.5-1.8 GHz | NEON (ASIMD) yes; **no SDOT/UDOT, no FP16 arithmetic** (both ARMv8.2) |
| Caches | 32 KB L1D per core, 1 MB shared L2, 64 B lines | Weight tiles must stream; nothing model-sized stays cached |
| RAM | 4 GB LPDDR4, 32-bit bus | Practical bandwidth ~4-5 GB/s (to be measured) → **decode is bandwidth-bound** |
| Pages | 4 KB by default | TLB misses while streaming GBs of weights → huge pages help |

Rule of thumb: decode tokens/s ≈ usable bandwidth ÷ bytes of weights read per token.

## 4. Architecture

```
Windows PC                                   Raspberry Pi 4
──────────                                   ──────────────
WSL2 Ubuntu: cross-compile (aarch64, static)  core 0: Linux, sshd, Python, network, IRQs
             build OS image (mmdebstrap)      cores 1-3: isolated, pinfer worker threads only
harness/pi.py ── SSH (Ethernet) ──────────▶   pinfer: mmap(model.pq) in huge pages
  check / push / run / bench → bench/results/
tools/convert.py: safetensors → model.pq
```

### 4.1 pinfer engine *(call: C11 + NEON intrinsics, one static library + CLI + Python ctypes binding)*
- Offline converter (on PC, Python) turns Qwen safetensors into a `.pq` file: weights
  quantized and **pre-tiled** in the exact order the NEON kernels read them.
- Kernels: quantized matrix-vector multiply first (≈95% of decode time), then RMSNorm,
  RoPE, SwiGLU, attention with GQA, KV cache.
- Quantization: int8 first (simple, accurate), then 4-bit with int8 activations. Kernels
  use `SMLAL`/`SADALP`-style widening multiply-accumulate since `SDOT` is unavailable.
- Threading: each layer's rows split across the isolated cores; spin barriers, no locks.
- Correctness: scalar fp32 reference path checked against Hugging Face logits on the PC;
  every fast path checked against the reference path.
- Later: speculative decoding (small Qwen drafts, larger Qwen verifies).

### 4.2 Split-core runtime
Kernel cmdline `isolcpus=1-3 nohz_full=1-3 rcu_nocbs=1-3 irqaffinity=0`, performance
governor, hugetlbfs 2 MB pages for weights, `mlock` to stop paging. Whether the stock
Raspberry Pi kernel supports `nohz_full` must be verified; if not, that is the first
reason to build our own kernel.

### 4.3 Custom OS *(call: Debian 13 "trixie" arm64 via mmdebstrap)*
Minimal rootfs + Raspberry Pi firmware + Raspberry Pi kernel (our config) + systemd,
sshd, networking, Python 3. No desktop. USB-C gadget networking as a backup way in.

### 4.4 Harness
`harness/pi.py` on the PC: `check` (board state), `push` (copy files), `run`, `bench`.
Every benchmark records temperature, clocks, throttle flags and kernel cmdline next to
the result in `bench/results/*.json`; a result taken while throttled is marked invalid.

## 5. Phases (each ends with a measured number)

| # | Phase | Exit criterion |
|---|---|---|
| 0 | Setup: toolchain, stock Pi OS Lite 64-bit flashed, SSH key, harness `check` | `check` prints board state |
| 1 | Baseline: memory bandwidth microbench; llama.cpp + Qwen tok/s | baseline table recorded |
| 2 | pinfer v0: converter + scalar fp32 engine | matches HF top-1 tokens |
| 3 | pinfer fast: int8 → 4-bit, NEON kernels, 4 threads | each step's tok/s recorded |
| 4 | Split-core on stock OS: isolation, huge pages, governor | delta vs phase 3 recorded |
| 5 | Custom OS image boots, keeps SSH/Python/net | harness `check` passes on it |
| 6 | Speculative decoding, Python binding | final table vs llama.cpp |

## 6. Constraints
- Static linking for Pi binaries (WSL glibc is newer than the Pi's).
- PC is on wired Ethernet; Pi joins the same network over Ethernet.
- Use the official 5.1 V / 3 A supply and a heatsink or fan during benchmarks.

## 7. Target model *(user, 2026-10-21)*
**Qwen3.5-0.8B** (HF `Qwen/Qwen3.5-0.8B` @ `2fc06364`), text path only. Hybrid: 24 layers =
18 `linear_attention` (Gated-DeltaNet-style, fixed-size recurrent state, no KV cache growth)
+ 6 `full_attention` (every 4th layer; 8 query heads, 2 KV heads, head_dim 256); hidden 1024,
FFN 3584, vocab 248,320, tied embeddings. The tied output head is ~254M of ~0.78B text
parameters, i.e. about a third of the bytes read per token — a prime optimization target.
Engine work therefore needs a linear-attention (delta-rule) kernel in addition to the usual
matvec/attention kernels. Qwen3-0.6B (@ `c1899de2`, plain transformer) is kept as a secondary
reference.

Comparison baselines: llama.cpp (primary) + other standard runtimes that support this
architecture on ARM64 CPU (being verified; see notes.md).

## 8. Left to the builder
File format details of `.pq`, tile sizes, thread scheduling.
