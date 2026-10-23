# gpu (VideoCore VI / V3D 4.2 QPUs as a second matmul engine; branch `worktree-agent-a459e8864e2e5ea3b`,
notebook `engine/GPU.md`, code `engine/gpu/`)

(Kept in my worktree: `.claude/worktrees/agent-a459e8864e2e5ea3b/plan/board/gpu.md`.)

- 17:58 [measured] **py-videocore6 licence is NOT BSD-3**: repo LICENSE is GPL-2.0; `driver.py`,
  `drm_v3d.py`, `v3d.py` carry GPL-2.0-or-later headers; `assembler.py` and examples are MIT. => usable as a
  separately installed dependency (venv on the Pi), never vendored into the Apache-2.0 tree; a C engine
  integration needs our own DRM ioctl code (from the MIT uapi header `drm/v3d_drm.h`). Its README sgemm is a
  naive kernel: 3.9 GFLOP/s at 1024^3 (the "24 GFLOPS" figure is not in the repo).
  Also: numpy 2.5 breaks it (`class Array(npt.NDArray[T])` -> TypeError); pinned numpy 2.3.1 (its uv.lock).
- 17:58 [idea] **Integer path beats fp32 on the QPU**: `smul24` with two rows packed in one operand
  (u_a + 65536 u_b) times an int8 activation = 2 MACs per mul-ALU op; with -8 * 65537 * sum(x) folded into
  the accumulator start, a 32-column Q4_0 block sum stays exact in 16-bit fields (|L|,|H| <= 32 * 8 * 127 =
  32512). fp32 is 1 MAC per instruction (fmul + fadd) plus dequant. Peak model: 8 QPU x 125 M instr/s x 16
  lanes x 2 = 64 GOP/s raw; with unpack / extraction overhead ~1.4 MAC per lane-instruction => ~45 GOP/s.
- 17:58 [measured] PC-side generator + list scheduler (`engine/gpu/q4gpu.py`, `qsched.py`), checked by a
  numpy emulator of the scheduled code (`qemu_check.py`: bit-identical to the fp32 model of the kernel for
  n = 1, 2, 7, 8, 11): instructions per 32-column block 334 / 376 / 488 at n = 7 / 8 / 11
  = 1.33 / 1.35 / 1.43 MAC per lane-instruction => model 42.8 / 43.4 / 45.8 GOP/s if no stalls.
  Not yet run on the GPU (Pi paused for the overclock window).
- 17:58 [idea] for:lead exactness: the CPU kernel uses fused FMLA / FMLS per block (`vfmaq_f32`,
  `vfmsq_n_f32`); the QPU has no FMA, so a bit-identical GPU path (class a) must leave the float tail to
  the CPU (GPU returns exact int16 block sums L/H) or emulate FMA (too slow). The fp32 GPU path differs from
  the CPU by ~1e-7 relative (class b).
- 21:14 [plan] Resumed as a pilot (owner's method). PC side ready: n = 64 as 8 passes of 8 (one dispatch each),
  emulator bit-exact up to n = 12 (516 instr / block, 1.48 MAC per lane-instruction). Pi plan under the lock, ~1.5 h:
  smoke -> `gpu-q4-alone` (gate / down / qkv at n = 7, 64) -> `gpu-corun` (kbench c4* 4 threads vs the same + GPU loop;
  go if combined >= 1.2x). The venv still has numpy 2.5.4 (the 2.3.1 downgrade was refused by the pause); first Pi
  command re-pushes the wheel. Pi down since the overclock test; polling `pi.py check`.
- 21:54 [measured] Pi came back 21:4x; smoke tests pass (lane kernel; an 8-batch CSD dispatch lands on 8 distinct
  QPUs, ids 0-7). First real kernel run **hangs the GPU**: gate n = 7, 1.02 s per call (= driver hang timeout +
  reset), output NaN (`bench/results/gpu-q4-first_20261022_214808.json`, valid=true but exact=false). The
  emulator is bit-exact, so it is structural: prime suspect the TMU output FIFO depth (I keep 13 words
  outstanding; if a single-threaded QPU gets 8, the request blocks and deadlocks). Fix in progress: selectable
  request plans (13 / 9 / 7 words outstanding), each emulator-checked; next Pi run = all three on a 256x256
  shape + dmesg, seconds each. **Pi slot request**: ~15 min after lut and toolchain, once "[Pi ready]" is posted.
- 21:57 [plan] for:lead, for:lut, for:toolchain: ready for my Pi slot after toolchain (<= 15 min under the lock, names
  gpu-*): (1) diag.py, seconds: uniform-stream loop with unif_addr, TMU FIFO depth probe (4 / 8 / 12 / 16 words
  outstanding), int / float op semantics vs numpy; (2) the kernel on a 256x256 shape with the three request plans;
  (3) if one plan runs, gate / down / qkv at n = 7 and 64 (~4 min) and the CPU + GPU co-run (~4 min). I will post
  "gpu: slot taken" / "gpu: slot released" here.
- 21:58 [plan] for:toolchain: I take a <= 3 min slot at your first release after 22:00 (diag + the 256x256 plan
  test, bench names gpu-diag / gpu-q4-plans); the longer runs (alone ~4 min, co-run ~4 min) at later boundaries,
  each announced here as "gpu: slot taken HH:MM" / "gpu: slot released HH:MM".
- 22:06 [measured] **Hang cause found** (`bench/results/gpu-diag_20261022_220321.json`): the uniform-stream loop with
  `unif_addr` works; every TMU *load* issued without the `thrsw` signal hangs the GPU (dmesg: `v3d_reset: Resetting GPU
  for hang, V3D_ERR_STAT 0x1000`), even one regular 32-bit read; the smoke kernel's load carried `sig=thrsw` (as all
  py-videocore6 examples do) and worked. Fix: thrsw on every load request, two signal-free slots after it, <= 4 words
  outstanding in the prologue; emulator still bit-exact (328 instr / block at n = 7).
- 22:06 [plan] **gpu: slot taken 22:06** (<= 15 min, one chained job: `gpu-q4-alone` then `gpu-corun`); then I release
  and stop (owner: decisive measurement only).
- 22:12 [plan] **gpu: slot released 22:12** (both jobs done, 22:06-22:12). Stopping per the owner (decisive measurement only).
- 22:12 [measured] **GPU alone, integer Q4 x int8 QPU kernel, bit-exact vs its fp32 model** on every config
  (`bench/results/gpu-q4-alone_20261022_221029.json`): gate 9216x2560 n = 7 / 64: 12.2 / 108.8 ms = **27.2 / 27.8 GOP/s**;
  down 2560x9216: 12.0 / 107.6 ms = 27.5 / 28.1; qkv 8192x2560: 10.9 / 97.7 ms = 27.0 / 27.5 (n = 64 = 8 passes of 8;
  63 % of the no-stall instruction model; reads 1.0-1.3 GB/s; feeding thread 1 % of a core).
- 22:12 [measured] **CPU + GPU together vs CPU alone = 1.41x / 1.41x / 1.47x (qkv / gate / down) -> GO** (>= 1.2x rule)
  (`bench/results/gpu-corun_20261022_221209.json`): kbench q4 4 thr n = 7 cold, CPU alone (two rounds, mean)
  qkv 55.2 / gate 55.4 / down 50.3 GOP/s; with the GPU looping on gate n = 64 at the same time: CPU 51.7 / 52.0 / 48.0
  (-6 %: DRAM contention, 2.1 -> 1.9-2.1 GB/s of CPU stream + ~1.0 GB/s GPU) and GPU 26.1 GOP/s (-6 %); combined
  77.8 / 78.0 / 74.1. throttled 0x0 throughout, temp <= 63.3 C, ARM 1500 MHz during every loaded phase.
- 22:12 [for:lead] Exactness class (b) only (fp32 mul / add on the GPU vs FMLA / FMLS on the CPU, ~2e-7 relative on the
  GPU rows; integer block sums exact). Scale-up design in `engine/GPU.md` (DRM ioctls, MIT uapi header, no py-videocore6 in
  the engine; rows split ~1/3 GPU). Next lever before integrating: more TMU words in flight now that thrsw is on
  (the kernel runs at 63 % of its instruction model), and a C feeder (the Python one costs 3 % of a core in the co-run).
