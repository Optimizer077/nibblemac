# Compilers and build flags (toolchain notebook)

Every compiler and flag set tried for the Pi 4 engine, with its numbers and result files. The engine
must stay bit-identical (CONTRIBUTING.md §5): no FMA contraction, fixed reduction order, so a compiler
is judged first on exactness (kbench `--check` yhash set, spec_exact / ctx_check md5s, Cerno logits
against `engine/pinfer_run_fast`) and only then on speed. Measurements at stock 1.5 GHz, 4 threads,
through `harness/pi.py bench` (`freq_khz` recorded in every result file).

Shipping build (engine/Makefile): `aarch64-linux-gnu-gcc` 15.2 (Ubuntu 26.04 cross package),
`-O3 -mcpu=cortex-a72 -std=c11 -ffp-contract=off`, static. All builds below use these flags unless a
column says otherwise, on the same sources (the 4B-prefill tree `69b31a3`, which the lut asm work also
builds from), with the shipping link order, so a GCC 15 build reproduces the shipped kbench byte for byte
(md5 6454499e).

## 1. Toolchains (installed in WSL Ubuntu 26.04 with apt, 2026-10-22 17:46 KST; PC-side only)

| name | package | version |
|---|---|---|
| g13 | gcc-13-aarch64-linux-gnu | 13.4.0 |
| g14 | gcc-14-aarch64-linux-gnu | 14.3.0 |
| g15 | gcc-15-aarch64-linux-gnu (default `aarch64-linux-gnu-gcc`) | 15.2.0 |
| g16 | gcc-16-aarch64-linux-gnu | 16.0.1 20260322 (experimental, trunk r16-8246) |
| c20 | clang-20 + lld-20 + llvm-20 (`--target=aarch64-linux-gnu -fuse-ld=lld`) | 20.1.8 |
| c22 | clang-22 + lld-22 + llvm-22 | 22.1.2 |

Build driver: one object per source with the Makefile's flags, link order core / matvec_fast / kern_*
(`scratchpad/toolchain-agent/tcbuild.sh`, PC only; not part of the repo). Static binaries, 0.80-0.82 MB
of text for every compiler.

## 2. Static codegen of the Q4 / Q4K block loops (kern_quant.c)

Instructions and stack spill ops (`[sp` loads / stores) per block of the nv-vector register pass,
from `llvm-objdump -d` of `kern_quant.o`. The nv = 1 row is the single-vector (decode) loop with its
8 blocks unrolled. The 4B prefill runs nv = 7 and 5 (groups of 7s + 5s).

| build | flags beyond the project's | Q4 nv=7 | nv=5 | nv=1 (8 blocks) | Q4K nv=7 |
|---|---|---|---|---|---|
| g15 (shipping) | — | 232 / 12 | 171 / 8 | 395 / 10 | 220 / 2 |
| g13 | — | 236 / 12 | 173 / 9 | 409 / 6 | 227 / 6 |
| g14, g16 | — | 232 / 12 | 171 / 8 | 399-400 / 10 | 220 / 2 |
| g15 + `-fno-schedule-insns` (kern_quant.c) | speed4b's find | 217 / 0 | 164 / 0 | 365 / 0 | 221 / 0 |
| g13 / g14 / g16 + `-fno-schedule-insns` | | 217-219 / 0 | 164-165 / 0 | 365-375 / 0 | 221-222 / 0 |
| g15 + `-fno-schedule-insns -funroll-loops` | | 217 / 0 | 161 / 0 | 360 / 0 | 216 / 0 |
| g15 + `-fsched-pressure` (algorithm 1 or 2) | | 232 / 12 | 171 / 8 | 395 / 10 | 220 / 2 |
| g15 + `-fno-schedule-insns -fira-algorithm=priority` | | 265 / 0 (+48 MOV) | 195 / 0 | 408 / 0 | 264 / 0 |
| g15 `-O2 -fno-schedule-insns` | | no multi-vector loop kept (passes not specialised) | | | |
| c22 | — | 239 / 32 | 161 / 7 | 355 / 0 | 217 / 12 |
| c20 | — | 247 / 33 | 164 / 9 | 355 / 0 | 240 / 30 |
| c20 / c22 + `-mllvm -enable-misched=false` (kern_quant.c) | | 210 / 0 | 156 / 0 | 358 / 0 | 208 / 0 |
| c22 + `-mtune=generic` | | 234 / 27 | 182 / 16 | 354 / 0 | 226 / 21 |
| c22 + `-mllvm -regalloc=pbqp` | | 305 / 96 | 175 / 20 | 370 / 0 | 236 / 33 |

No effect on the .text: `-mtune=cortex-a57` (= a72 for GCC 15 and clang 22), `-fno-tree-vectorize`
(the kernels are intrinsics), `-fira-region=one`; `-fno-schedule-insns2` on top of `-fno-schedule-insns`
changes the order but not the counts.

**The counts are the wrong proxy.** speed4b measured the spill-free `-fno-schedule-insns` object on the
Pi (`bench/results/cerno-speed-kb8-nosched_20261022_175206.json`, valid, 1.5 GHz): cold 4 threads Q4
n = 7 c4qkv 5509 / 5380 -> 5830 / 5951 us (+8 %), c4gate 5932 / 6050 -> 6979 / 6782 (+13 %), c4down
6372 / 6327 -> 7000 / 7001 (+10 %), c4out 2713 / 2720 -> 3157 / 3087 (+15 %); n = 5 +6-13 %; hot 1 thread
kv n = 7 522 / 512 -> 554 / 553 (+7 %). The A72 dispatches in order, one F0 and one F1 uop per cycle
(lut's ubench2, `lut-ubench2_20261022_175209.json`): GCC's first scheduling pass interleaves the SMLALs
with the reductions and float updates, and losing that interleaving costs more than the 15 instructions
and 12 spills it saves. Flag sets are therefore ranked on the Pi only (§4).

## 3. Exactness under qemu (PC, `qemu-aarch64 -cpu cortex-a72`)

`kbench --check` over f32 / bf16 / q8 / q4 / q4k x qkv, z, out, kv, gate, down, ab, dhead, c4qkv,
c4down, c4ab x 1 and 4 threads x n = 1-8 = 880 configurations per build (`~/tc/out/*/qk.txt`, WSL
scratch; not committed):

- 13 of 14 builds (g13, g14, g15, g16, c22, g15 / g13 / g14 / g16 + `-fno-schedule-insns`, g15 + LTO,
  g15 `-O2`, g15 `-funroll-loops`, c22 + misched off on kern_quant.c or everywhere, c22 LTO) pass with
  `exact` and `ok` on every row and with the SAME 880 yhash values as the shipping GCC 15 build.
- **clang 20 differs in all 176 q4 (Q4_G32) rows** (every shape, thread count and n); q4k, q8, f32 and
  bf16 are identical. `exact` (batched == matvec) and `ok` stay true, `rel_err_kernel` is identical to
  4 digits (3.802e-07 on qkv) and only `rel_err_fp` moves (8.082e-03 -> 7.574e-03), which is the
  signature of a different Q4 *matrix fill* (kbench_util.c's quantizer compiled by clang 20), not of a
  different kernel. Decided by the mixed link and the real-model dumps in §3.1.

### 3.1 clang 20 is rejected: it miscompiles a max-by-magnitude loop at -O3

Mixed links (`scratchpad/toolchain-agent/mix.sh`): clang 20's kernel objects linked with GCC's
`kbench_util.o` give GCC's yhashes; GCC's kernels with clang 20's `kbench_util.o` give clang 20's.
So clang 20's *kernels* are bit-identical and the difference is in kbench's Q4 fill
(`engine/bench/kbench_util.c` lines 72-76):

```c
float mx = 0;
for (int j = 0; j < 32; j++)
    if (fabsf(x[j]) > fabsf(mx)) mx = x[j];
sc[k * 4 + i] = (__fp16)(mx / -8.0f);
```

Dumping the packed Q4 matrix of `kb_fill_matrix` (512 x 1024, seed 42; `q4dump.c` / `q4diff.py`)
from a GCC 15 build and a clang 20 `-O3` build of that file: **274 of 16384 blocks (1.7 %) have a
different scale, none of them a sign flip** (fp16 0x531e vs 0x511d, 0x0d27 vs 0x349d, ...), i.e. clang
20 picks an element with a different |x| as the maximum, and 265 of those blocks' codes differ. Flag
bisection on that file alone (`bisect.sh`, `bisect2.sh`): `-O2`, `-O1`, `-O0`, `-fno-unroll-loops`,
`-fno-builtin-fabsf`, `-frounding-math`, `-ffp-exception-behavior=strict` and `-ffp-model=strict` each
restore GCC's result; `-fno-vectorize -fno-slp-vectorize`, `-fno-builtin-lrintf`, `-fno-builtin-fmaxf`,
`-mcpu=generic` do not. No FMA is involved (0 `fmadd` / `fmsub` in every object of every compiler:
`-ffp-contract=off` is honoured by all of them). GCC 13 / 14 / 15 / 16, clang 22.1.2 and clang 20 at
`-O2` agree with each other; clang 20.1.8 at `-O3` (unrolled loop + `fabsf` as a builtin) is alone.
Whatever the LLVM 20 transformation is, it changes the result of plain C floating-point code, which
CONTRIBUTING.md §5 forbids: **clang 20 is out**, whatever its kernel speed. (The engine's own objects
are not affected on this input, but the same idiom appears in the engine's quantizers.)

## 4. Pi pilot (compilers at the project flags)

Builds pushed as `engine/pinfer_run_tcN` / `tmp/tc/kbench_tcN` on the Pi:

| tag | build | pinfer_run md5 | kbench md5 |
|---|---|---|---|
| tc0 | g15 (shipping compiler) | 7db0de08 | 6454499e (= speed4b's kbench_csp1) |
| tc15 | g14 | e6e9f19d | b090836f |
| tc13 | g16 | 484ae808 | 9545da91 |
| tc14 | c20 | 04b45c64 | 28f20e6c |
| tc2 | c22 | 772a20d0 | 3e47ca1f |
| tc3 | c22 + `-mllvm -enable-misched=false` on kern_quant.c | b0e4c47d | f9a73bdd |

(results follow)
