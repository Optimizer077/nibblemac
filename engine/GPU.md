# GPU.md: the VideoCore VI QPUs as a second matmul engine (2026-10-22)

Question: can the Pi 4's GPU (V3D 4.2: 8 QPUs = 2 slices x 4, 16-way SIMD over 4 cycles,
500 MHz) add useful throughput to the 4 Cortex-A72 cores on the compute-bound Cerno 4B
matmuls (prefill, speculative verify, every Cerno decision)? The CPU's Q4 kernels run at
~54 GOP/s, 93-98 % of their best rate (plan/board/audit.md 17:27, speed4b.md 17:40).

Status and verdict: see "Results" and "Verdict" below.

## Route and setup

- **Library**: py-videocore6 (github.com/Idein/py-videocore6, commit `175741a`): a Python
  assembler for the QPU instruction set plus a DRM V3D driver wrapper (CSD compute jobs via
  `/dev/dri/card0`, no root; user `pi` is in `video` / `render`).
  **Licence: not BSD-3.** The repository LICENSE is GPL-2.0; `driver.py`, `drm_v3d.py`,
  `v3d.py` carry GPL-2.0-or-later headers; `assembler.py` and the examples are MIT. We use it
  only as a separately installed dependency on the Pi (never vendored into this Apache-2.0
  tree). Our scripts import it at run time. A C engine integration would use the MIT kernel
  uapi header `drm/v3d_drm.h` (present in the WSL aarch64 sysroot), not py-videocore6's driver.
- **numpy pin**: numpy 2.5 breaks py-videocore6 (`class Array(npt.NDArray[T])` raises
  `TypeError: typealias() takes exactly 2 positional arguments`); its `uv.lock` pins numpy
  2.3.1, whose aarch64 cp313 wheel (sha256 `18703df6…41a1`, equal to the lock file's) we use.
- **Pi files** (user level, no system change; rows in docs/pi-state.md):
  `~/pinfer/venvs/gpu` (python3 -m venv; numpy 2.3.1 + ioctl-opt 1.3.1, offline wheels;
  freeze in `~/pinfer/venvs/gpu.freeze.txt`), `~/pinfer/tmp/gpu/` (py-videocore6 source,
  wheels, our scripts). No apt package, no config.txt / clock / gpu_mem change.
- **Scripts** (`engine/gpu/`): `gpurun.sh` (venv + PYTHONPATH launcher), `hello.py` (smoke
  test), `qsched.py` (list scheduler), `q4gpu.py` (kernel generator, host layout, references),
  `qemu_check.py` (PC-side emulator check), `sweep_sched.py` (schedule knobs),
  `bench_q4.py` (GPU-alone benchmark), `corun.sh` (CPU + GPU co-run).

## Why an integer kernel (and not fp32 sgemm)

Per QPU instruction the hardware issues one add-ALU op and one mul-ALU op on 16 lanes.
- fp32: `fmul` (mul ALU) + `fadd` (add ALU) = 1 MAC per lane-instruction, 32 GFLOP/s peak,
  plus dequantizing the 4-bit weights (unpack, `itof`, scale) and 8x the weight bytes if the
  dequantized matrix were stored. py-videocore6's sgemm example is a naive kernel at
  3.9 GFLOP/s (its README, 1024^3).
- int: `smul24` (signed 24 x 24 -> low 32 bits, mul ALU) with **two rows packed in one
  operand**: `(u_a + 65536 u_b) * x = u_a x + 65536 u_b x`, i.e. 2 MACs per mul-ALU op, the
  accumulation on the add ALU (or on the mul ALU's integer `add` when it is free).

Exactness of the packing: with Q4_0 codes u in 0..15 (value (u - 8) d) and int8 activations
|x| <= 127 (kern_act.c), the accumulator starts at -8 * 65537 * sum(x) of the block, so after
the 32 columns it holds L + 65536 H with L = sum x (u_a - 8), H = sum x (u_b - 8),
|L|, |H| <= 32 * 8 * 127 = 32512: both 16-bit fields are exact (wrapping int32 arithmetic is
fine because the final value fits). L = asr(shl(S, 16), 16) (the shl is `smul24` by 65536),
H = asr(S - L, 16).

## Kernel design (`engine/gpu/q4gpu.py`)

- Lane i of a 32-row tile holds rows 32t + i (low field) and 32t + 16 + i (high field).
  GPU weight layout (made once on the host): per tile and 32-column block, 8 code words per
  lane (`(W >> 4m) & 0x000F000F` = the pair for column 4k + m), read as two 16-byte TMU vec4
  loads, and one scale word (fp16 d_a | fp16 d_b << 16, unpacked to fp32 for free by the
  `.l` / `.h` input modifiers of `fmul`). Same 4.5 bits per weight as Q4_0.
- Activations: a uniform stream, one 32-bit int8 value per `smul24` (`ldunif` signal,
  readable in r5 by the next instruction), plus per block and vector the start value
  -524296 sum(x) and the activation scale dx. All 8 QPUs read the same stream; the branch at
  each tile resets the stream address (`unif_addr`).
- Per block and vector: 32 `smul24` + 32 adds, then L / H extraction (4 integer ops),
  `itof` x 2, `fmul` (d_a dx, d_b dx), `fmul` x 2, `fadd` x 2. Output y[v][row] in fp32.
- Each QPU takes a contiguous range of tiles (T / 8); weights are requested two blocks ahead
  (TMU FIFO), so the loads have a full block of latency budget.
- **Scheduling**: `qsched.py` packs the op list into dual-issue instructions under the V3D
  4.2 rules (two register-file reads per instruction, one if a small immediate is used; a
  small immediate excludes `ldunif`; encodable signal sets; TMU and uniform order kept;
  RAW distance 1, WAR distance 0). Knobs swept with `sweep_sched.py`.
- **Verification without the GPU**: `qemu_check.py` runs the scheduled code in a numpy
  emulator (16 lanes, reads at instruction start / writes at the end, TMU FIFO, uniform
  stream) and must equal `reference_gpu` (the kernel's fp32 math in numpy) bit for bit;
  it also assembles the full program.

Instructions per 32-column block (body, + 4 for the loop branch): n = 7: 327, n = 8: 370,
n = 11: 476, i.e. 1.35 / 1.37 / 1.47 MAC per lane-instruction. No-stall model:
8 QPUs x 125 M instructions/s x 16 lanes x MAC/instr x 2 = 43 / 44 / 47 GOP/s.

## Pilot plan (owner's method: pilot, then scale what works; `engine/PILOTS.md`)

1. Smoke test under the lock: `hello.py` (lane add; the QPU ids of an 8-batch dispatch).
2. `gpu-q4-alone`: `bench_q4.py --shapes gate,down,qkv --n 7,64` (64 = 8 passes of 8,
   the weights re-read per pass as in the CPU's groups of 7). Per (shape, n): median time,
   GOP/s, bytes read per call (weights x passes + the uniform stream x 8 QPUs), bit-exactness
   against the kernel's fp32 model on 4 sampled tiles over all n vectors (a wrong int16
   block sum would change the output).
3. `gpu-corun`: `corun.sh` = kbench q4 4 threads on c4gate / c4down / c4qkv at n = 7
   (CPU alone), GPU alone, both at once (the GPU loop runs until kbench ends), CPU alone
   again; temperature, throttle flags and ARM / V3D clocks sampled every 2 s.
   Go if combined >= 1.2x CPU alone.

## Scale-up design (only if the pilot says go): the DRM ioctl path in the engine

- No py-videocore6 in the engine (GPL driver). The kernel uapi header `drm/v3d_drm.h` (MIT,
  in the aarch64 sysroot) gives everything needed: `DRM_IOCTL_V3D_CREATE_BO` (size ->
  handle + 32-bit GPU offset), `DRM_IOCTL_V3D_MMAP_BO` (-> mmap offset on the DRM fd),
  `DRM_IOCTL_V3D_SUBMIT_CSD` (cfg[7] = workgroup counts << 16, wg config, batches - 1,
  shader address | flags, uniforms address; bo_handles) and `DRM_IOCTL_V3D_WAIT_BO`.
  The machine code comes from the Python generator (`q4gpu.py` -> `assemble()` -> a
  `uint64[]` table compiled into the engine, one per pass size, with the uniform order).
- Buffers: one BO for the GPU-laid-out weights of the matrices given to the GPU (made at
  load from the Q4_0 tiles, same bits), a ring of BOs for the uniform streams (int8 codes,
  block sums and scales of the group's vectors, built by the CPU thread that also runs
  `kern_quant_act`), one BO for Y. `mmap` of a V3D BO is write-combined / uncached on the
  CPU side: write streams sequentially, read Y once into cacheable memory.
- Split per matmul: rows divided in 32-row tiles in proportion to the measured rates
  (the GPU gets ~1/3 of the tiles if it adds ~50 % to the CPU), the CPU threads take the rest
  with the existing kernel; one submit per group of vectors, the wait at the end of the
  group's CPU work; no per-layer synchronization needed since Y rows are disjoint.
- Exactness: the GPU rows are class (b) (fp32 mul / add instead of the CPU's FMLA / FMLS and
  a different summation order, ~1e-7 relative); opt-in only, with the Cerno agreement check
  (max |dp| <= 0.0146, top answer >= 51 / 52) and 0.8B perplexity before it is enabled.

## Results (2026-10-22, Pi at stock 1.5 GHz, V3D 500 MHz, all runs under the bench lock, throttled 0x0)

### Pi runs, in order
| time (KST) | run | result | file |
|---|---|---|---|
| 21:4x | smoke (`hello.py`, `run --lock`) | lane add OK; an 8-batch CSD dispatch lands on 8 distinct QPUs (ids 0-7) | `logs/harness.jsonl` |
| 21:48 | first kernel run, gate n = 7 | **GPU hang**: 1.02 s per call, NaN output (driver timeout + reset) | `bench/results/gpu-q4-first_20261022_214808.json` |
| 22:03 | `diag.py` + the 256x256 shape with 3 request plans | `unif` (uniform-stream loop + `unif_addr`) OK; `math` and `fifo` **hang** on the first TMU load; all 3 plans hang; dmesg `v3d_reset: Resetting GPU for hang, V3D_ERR_STAT 0x1000` x 6 | `bench/results/gpu-diag_20261022_220321.json` |
| 22:10 | `gpu-q4-alone` (thrsw fix) | all 6 configs bit-exact, 27-28 GOP/s | `bench/results/gpu-q4-alone_20261022_221029.json` |
| 22:12 | `gpu-corun` | CPU keeps 94 %, combined 1.41-1.47x | `bench/results/gpu-corun_20261022_221209.json` |

**The hang and its fix.** Every TMU *load* issued without the `thrsw` signal hangs the V3D 4.2
(even one regular 32-bit read; stores are fine; the uniform stream is fine). The smoke kernel's
load carried `sig=thrsw`, as every load in py-videocore6's examples and tests does, and worked.
Fix (`segment_ops`): `thrsw` on every load request, the two following instructions free of
signals, at most 4 words outstanding in the prologue. The TMU output FIFO depth could not be
probed (the probe hung for the same reason); the kernel keeps <= 7 words outstanding
(plan `split`), which runs.

### Question 1: GPU alone (`bench_q4.py --plan split`, median of 10 calls)
| shape | n | passes | ms / call | GOP/s | of no-stall model | bytes read / call | exact vs model |
|---|---|---|---|---|---|---|---|
| gate 9216x2560 | 7 | 1 | 12.16 | **27.2** | 63 % | 13.9 MB (1.14 GB/s) | yes (896 words) |
| gate 9216x2560 | 64 | 8 x 8 | 108.8 | **27.8** | 63 % | 111.7 MB (1.03 GB/s) | yes (8192 words) |
| down 2560x9216 | 7 | 1 | 12.03 | **27.5** | 64 % | 15.5 MB (1.29 GB/s) | yes |
| down 2560x9216 | 64 | 8 x 8 | 107.6 | **28.1** | 64 % | 126.2 MB (1.17 GB/s) | yes |
| qkv 8192x2560 | 7 | 1 | 10.88 | **27.0** | 62 % | 12.4 MB (1.14 GB/s) | yes |
| qkv 8192x2560 | 64 | 8 x 8 | 97.7 | **27.5** | 63 % | 99.9 MB (1.02 GB/s) | yes |

GOP/s = 2 R K n / t (same convention as the CPU numbers). "Exact vs model": the output equals
`reference_gpu` (exact integer block sums, then fp32 mul / add) bit for bit on 4 sampled
32-row tiles over all n vectors, so the int16 block sums are exact. The no-stall model is
8 QPUs x 125 M instructions/s with 328 / 369 instructions per block (n = 7 / 8): the QPUs
stall ~37 % (TMU / uniform latency with 7 words in flight). Bytes read = weights x passes +
the uniform stream x 8 QPUs; the feeding Python thread uses ~1 % of a core; the GPU rate did
not change when the ARM cores idled down to 600 MHz (ondemand) during this phase.
Reading Y back from the write-combined BO runs at 0.6-0.7 GB/s (2.4 MB for gate n = 64 =
~4 ms, 3.4 % of the call); writing the uniform stream at 1.9-2.7 GB/s.

### Question 2: CPU + GPU together vs CPU alone (`corun.sh gate 64 7 1.0 3 8`)
kbench `kbench_gpu1 --dtypes q4 --shapes c4qkv,c4gate,c4down --threads 4 --batch 7 --time 1.0
--reps 3` (cold weights), with the GPU looping on gate n = 64 during the "co-run" pass; the
sampler logged temperature, throttle flags and ARM / V3D clocks every 2 s (max 63.3 C,
throttled 0x0, ARM 1500 MHz in every loaded phase).

| shape (CPU, n = 7, 4 thr) | CPU alone #1 | CPU alone #2 | CPU during co-run | GPU during co-run (gate n = 64) | combined | vs CPU alone (mean) |
|---|---|---|---|---|---|---|
| c4qkv 8192x2560 | 5346.7 us = 54.9 GOP/s | 5287.6 = 55.5 | 5678.4 = **51.7** (-6 %) | 115.9 ms = **26.1** | **77.8** | **1.41x** |
| c4gate 9216x2560 | 6001.1 = 55.0 | 5933.2 = 55.7 | 6357.9 = **52.0** (-6 %) | 26.1 | **78.0** | **1.41x** |
| c4down 2560x9216 | 6510.0 = 50.7 | 6616.8 = 49.9 | 6882.8 = **48.0** (-5 %, kbench "noisy") | 26.1 | **74.1** | **1.47x** |

CPU weight stream 2.0-2.2 GB/s alone -> 1.9-2.1 GB/s in the co-run; the GPU adds ~1.0 GB/s;
the GPU itself lost 6 % (27.7 -> 26.1 GOP/s, 393 calls) and its feeding thread took 3 % of a
core. yhash of the CPU outputs identical in all three passes (the CPU path is untouched).

## Verdict

**Go** (combined >= 1.2x CPU alone: 1.41-1.47x). The QPUs add ~26 GOP/s to the ~52-55 GOP/s
of the four cores on the Cerno 4B shapes while the CPU keeps 94 % of its rate, with no
throttling. Caveats before scaling up: (1) exactness class (b) only (fp32 mul / add vs the
CPU's FMLA / FMLS, ~2e-7 relative on the GPU rows), opt-in with the Cerno agreement check;
(2) the kernel runs at 63 % of its instruction model: more TMU words in flight (now that
`thrsw` is on) is the first lever; (3) the Python feeder must become C (3 % of a core, plus
the uniform-stream build); (4) the row split must give the GPU ~1/3 of the tiles so that
both finish together (~26 : 52). Expected end-to-end gain on a 93 %-matmul decision:
up to ~1.3x if the split overlaps perfectly, less with the Y read-back and the stream build.

## Credits

py-videocore6 by Idein Inc. (assembler MIT, driver GPL-2.0-or-later), used as a run-time
dependency. ISA details: Mesa `src/broadcom/qpu`, the Linux `v3d` driver, py-videocore6's tests.


## Scale-up step 1: the kernel from C, own DRM path (channel agent t005, 2026-10-23 12:06 KST)

`engine/gpu/v3d.{c,h}` open `/dev/dri/renderD128`, create / mmap BOs, submit the compute job
(`DRM_IOCTL_V3D_SUBMIT_CSD`) and wait (3 s deadline, EINTR retries), using only the kernel's MIT uAPI header
(no py-videocore6 code). `emit_q4.py` writes the assembled kernels to `q4gpu_kernels.h` (n = 7 / 8 programs,
466 / 515 instructions), so nothing needs Python at run time. `v3d_q4_test.c` runs gate 9216×2560.

| n | C: ms / GOP/s | Python path: ms / GOP/s | C vs Python | GPU vs CPU NEON kernel |
|---|---|---|---|---|
| 7 | 11.993 / 27.54 | 12.112 / 27.27 | 64,512 / 64,512 bytes equal | max abs 1.24e-5, relative 5.5e-7 |
| 64 (8 passes of 8) | 107.996 / 27.96 | 108.787 / 27.76 | 589,824 / 589,824 equal | max abs 1.53e-5, relative 6.4e-7 |

Result files: `bench/results/gpu-c-python-reference_20261023_115904.json`, `gpu-c-gate-n7-final_20261023_120301.json`,
`gpu-c-gate-n64-final_20261023_120303.json` (all valid, throttled 0x0), raw runs in `bench/results/gpu-c/`. One failed
first attempt kept (`gpu-c-gate-n7_20261023_115935.json`, valid=false: a work-group reused physical QPUs; fixed by
eight work-groups in one supergroup). The GPU result is **not bit-identical to the CPU kernel** (different float
order, no FMA), so the engine integration is an **opt-in** flag gated on Cerno agreement and perplexity.
Pi files: `~/pinfer/tmp/gpu-c/` (undo `rm -rf ~/pinfer/tmp/gpu-c`).

Next (step 2): split each matmul's rows ~1/3 GPU / 2/3 CPU (GPU tiles in multiples of 8 × 32 rows), upload the GPU
rows once at model load within a bounded BO budget, submit the GPU part before the CPU part, join per matmul.
