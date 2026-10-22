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
