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

**Real-model dumps under qemu** (`qe_seq.sh`; 4 threads; per build: prompt64 batched hidden states +
logits on `qwen35-0.8b-q4k-headq4k.pq` and on the Q4_0 `qwen35-0.8b-q4-headq4k.pq`, token-by-token
logits of its first 16 tokens, 12 generated tokens plain and `--spec 3 --draft-vocab 32768`, and the
Cerno 4B `cerno-4b-q4_0.pq` logits of the first 16 prompt tokens): **GCC 15 and clang 22 give the same
md5 on all 8 dumps** (h64k fe836caf, h64q cdd5ff61, l64k bb05dee8, l64q c4bd60be, l1k b0876251,
l16c df702b57, generated 080f184c plain = spec). GCC 14 / 16 and clang 22 + misched-off: run twice,
killed twice by the WSL restarts of 21:27 and 21:45 (see the journal); the Pi gates cover them.

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
different scale, none of them a sign flip**, i.e. clang 20 picks an element with a different |x| as the
maximum, and 265 of those blocks' codes differ. The first four differing blocks (d = mx / -8, fp16):

| block (tile, sb, k, row) | GCC 15 d | clang 20 d | max found by GCC / clang 20 |
|---|---|---|---|
| 0, 0, 3, 3 | 0x1e53 = 0.00617 | 0x1d51 = 0.00520 | 0.0494 / 0.0416 |
| 0, 2, 0, 2 | 0x1d01 = 0.00489 | 0x9c84 = -0.00441 | 0.0391 / 0.0353 |
| 0, 2, 6, 0 | 0x270d = 0.0275 | 0x9d34 = -0.00508 | 0.220 / 0.0406 |
| 1, 2, 2, 0 | 0x1d0a = 0.00491 | 0x1c57 = 0.00424 | 0.0393 / 0.0339 |

In **all 274 differing blocks clang 20's maximum is the smaller one** (274 smaller, 0 larger, 0 equal;
`q4diff.py`). A max-by-magnitude search returns the value of one element of the block, so the build
that reports the larger |x| has found an element the other build missed: GCC's answer (shared by GCC
13 / 14 / 16, clang 22 and clang 20 at `-O2`) is the true maximum and clang 20 `-O3`'s is wrong; in
block (0, 2, 6, 0) it misses a 6x outlier. Flag
bisection on that file alone (`bisect.sh`, `bisect2.sh`): `-O2`, `-O1`, `-O0`, `-fno-unroll-loops`,
`-fno-builtin-fabsf`, `-frounding-math`, `-ffp-exception-behavior=strict` and `-ffp-model=strict` each
restore GCC's result; `-fno-vectorize -fno-slp-vectorize`, `-fno-builtin-lrintf`, `-fno-builtin-fmaxf`,
`-mcpu=generic` do not. No FMA is involved (0 `fmadd` / `fmsub` in every object of every compiler:
`-ffp-contract=off` is honoured by all of them). GCC 13 / 14 / 15 / 16, clang 22.1.2 and clang 20 at
`-O2` agree with each other; clang 20.1.8 at `-O3` (unrolled loop + `fabsf` as a builtin) is alone.
Whatever the LLVM 20 transformation is, it changes the result of plain C floating-point code, which
CONTRIBUTING.md §5 forbids: **clang 20 is out**, whatever its kernel speed. (The engine's own objects
are not affected on this input, but the same idiom appears in the engine's quantizers.)

**Standalone reproducer: `engine/bench/clang20_maxabs.c`** (the generator and the loop copied from
kbench_util.c, every 32-column block of tile 0 checked against a volatile scalar pass). Under qemu:
GCC 15 `-O3`, clang 22 `-O3`, clang 20 `-O2` / `-O3 -fno-unroll-loops` / `-O3 -fno-builtin-fabsf`:
checksum -2.00933693, 0 wrong blocks; **clang 20 `-O3` for AArch64: 3 of 128 blocks wrong**, the first
"row 0 block 22: loop 0.0406423, true -0.220291" (= block (0, 2, 6, 0) above: the 6x outlier is
skipped); clang 20 `-O3` on x86-64: correct. So it is the AArch64 code generation of LLVM 20.1.8 for
an unrolled `if (fabsf(x) > fabsf(mx)) mx = x` chain, fixed by LLVM 22.

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

(tc14 = clang 20 was staged but not run: rejected in §3.1.)

### 4.1 Q4_0 batched kernel, kbench (`bench/results/toolchain-kb-r1_20261022_220248.json`, valid)

`tmp/tc/tc_kb.sh`: cold weights (every call reads a different copy from a 64 MB arena), 4 threads,
the 4B shapes, one register pass of n = 5 and n = 7 vectors (the sizes the prefill group plan uses);
plus hot 1-thread kv / c4kv. 3 rounds, build order rotated per round; medians; 1.5 GHz; 54 -> 50 C.
c/bv = cycles per block-vector (4 rows x 32 columns x 1 vector = 128 MACs) = us x 1.5e9 x threads /
(rows/4 x cols/32 x n). **The yhash of every build is identical on every row** (and `exact` / `ok`).

| shape, n | GCC 15 us (c/bv) | GCC 16 | GCC 14 | clang 22 | clang 22, misched off |
|---|---|---|---|---|---|
| c4qkv 8192x2560, n=7 | 5494.6 (28.7) | 5393.4 (28.2) -1.8 % | 5317.4 (27.8) -3.2 % | 5882.6 (30.8) +7.1 % | 6641.4 (34.7) +20.9 % |
| c4gate 9216x2560, n=7 | 6015.9 (28.0) | 5884.1 (27.4) -2.2 % | 6350.9 (29.5) +5.6 % | 6916.7 (32.2) +15.0 % | 7364.7 (34.2) +22.4 % |
| c4down 2560x9216, n=7 | 6485.1 (30.2) | 6449.4 (30.0) -0.6 % | 6547.0 (30.4) +1.0 % | 6834.0 (31.8) +5.4 % | 7740.9 (36.0) +19.4 % |
| c4out 2560x4096, n=7 | 2763.8 (28.9) | 2712.6 (28.4) -1.9 % | 2768.6 (29.0) +0.2 % | 3035.5 (31.8) +9.8 % | 3256.5 (34.1) +17.8 % |
| c4qkv, n=5 | 4035.2 (29.6) | 4028.8 (29.5) -0.2 % | 4032.1 (29.5) -0.1 % | 4194.9 (30.7) +4.0 % | 4663.7 (34.2) +15.6 % |
| c4gate, n=5 | 4543.5 (29.6) | 4509.9 (29.4) -0.7 % | 4467.0 (29.1) -1.7 % | 4600.8 (30.0) +1.3 % | 5151.0 (33.5) +13.4 % |
| c4down, n=5 | 5041.3 (32.8) | 4984.8 (32.5) -1.1 % | 5062.2 (33.0) +0.4 % | 4856.4 (31.6) -3.7 % | 5628.5 (36.6) +11.6 % |
| c4out, n=5 | 2128.7 (31.2) | 2200.2 (32.2) +3.4 % | 2133.7 (31.3) +0.2 % | 2194.8 (32.2) +3.1 % | 2448.8 (35.9) +15.0 % |
| hot 1 thr c4kv, n=7 | 2586.7 (27.1) | 2620.9 (27.4) +1.3 % | 2630.7 (27.5) +1.7 % | 2919.7 (30.5) +12.9 % | 3165.5 (33.1) +22.4 % |
| hot 1 thr c4kv, n=5 | 1933.2 (28.3) | 1935.0 (28.3) +0.1 % | 1979.6 (29.0) +2.4 % | 1978.5 (29.0) +2.3 % | 2293.2 (33.6) +18.6 % |
| hot 1 thr kv, n=7 | 507.2 (26.5) | 499.7 (26.1) -1.5 % | 508.7 (26.6) +0.3 % | 569.6 (29.8) +12.3 % | 605.8 (31.7) +19.4 % |
| hot 1 thr kv, n=5 | 373.7 (27.4) | 373.6 (27.4) 0.0 % | 373.1 (27.3) -0.2 % | 382.8 (28.0) +2.4 % | 446.9 (32.7) +19.6 % |

Reading: GCC 14 / 15 / 16 are the same compiler for this loop (identical instruction counts in §2,
within the +-3 % run-to-run noise here; GCC 16's -0.6..-2.2 % at n = 7 cold is the most consistent
difference and still inside the noise band). clang 22 is 5-15 % slower at n = 7, and **clang 22 with
the machine scheduler off, the build with the fewest instructions and no spills (210 per block), is
12-22 % slower on every row**: the second time today that the shortest code lost (speed4b's
`-fno-schedule-insns`, +6-15 %). The A72's in-order dispatch rewards the scheduler's interleaving of
the SMLALs with the reductions and the float update, not the instruction count.

### 4.2 Cerno 4B prefill p256 (`bench/results/toolchain-e2e4b-r1_20261022_220708.json`, valid)

`tmp/tc/tc_e2e4b.sh`: `pinfer_run -m models/cerno-4b-q4_0.pq -t 4 --ids-file tmp/csp/p256.ids -n 1
--bench` (in-process warm-up of p64 + 4 tokens, then the timed 256-token prefill), one run per build
(the owner cut the pilot to one round at 21:56), 1.5 GHz, 47 -> 61 C. The result file names the Cerno
model, so it stays out of the public repository like every `cerno-*` result (owner decision,
`.gitignore`); it is cited here by path.

| build | prefill p256 tok/s | vs GCC 15 | `load` line | warm-up |
|---|---|---|---|---|
| GCC 15 (shipping) | 6.915 | — | 0.006 s | 38.6 s |
| GCC 16 | 6.921 | +0.1 % | 0.005 s | 38.0 s |
| GCC 14 | 6.789 | -1.8 % | 0.006 s | 38.5 s |

(The same tree measured 6.80 / 6.87 tok/s on p256 in speed4b's 3-round A/B, `cerno-speed-e2e4b-r2`.)
The `load` line is the mmap and header checks only (6 ms for the 3.4 GB file); the model's first
touch is the warm-up's page-in, ~38 s with a warm page cache included in that number, and it does
not depend on the compiler. A cold-cache measurement (`tc_load.sh`, prepared, `dd iflag=nocache` to
evict the file) was not run under the shortened scope.

### 4.3 Verdict

**No-go.** No compiler at the project's exactness flags is >= 3 % faster than GCC 15 on the Q4_0
kernel or on 4B prefill: GCC 16 and GCC 14 are the same code within noise, clang 22 is slower
(5-15 % on the kernel at n = 7), clang 22 without the machine scheduler and GCC without the first
scheduling pass are much slower, and clang 20 is wrong. `engine/Makefile` stays as it is (GCC 15,
`-O3 -mcpu=cortex-a72 -std=c11 -ffp-contract=off`). The gate (spec_exact / ctx_check / Cerno logits
against `engine/pinfer_run_fast`) was therefore not run on the Pi for any build; the exactness
evidence per build is §3 (kbench yhash under qemu, 880 rows) and, for clang 22, the 8 real-model dumps.

Not measured (scope cut, PC-side builds exist and are exact under qemu): GCC 15 + LTO, `-O2`,
`-funroll-loops`, clang 22 + LTO; PGO (a `-fprofile-generate` build was proposed and declined). For the
compiler question, the decisive lever is in the kernel's instruction order (lut's asm passes, -3..-11 %
per block-vector in `lut-tbench-r1`), not in the compiler.
