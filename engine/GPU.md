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

## Results

(filled in below as measured)

## Credits

py-videocore6 by Idein Inc. (assembler MIT, driver GPL-2.0-or-later), used as a run-time
dependency. ISA details: Mesa `src/broadcom/qpu`, the Linux `v3d` driver, py-videocore6's tests.
