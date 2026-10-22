# PILOTS.md: small pilots before scaling (2026-10-22)

Method (owner): pilot quickly under a fixed Pi-time budget, decide on one number, then scale
only what works. Each pilot states its question, the decisive number, the go / no-go rule, the
measurements with their result files, and the verdict.

## GPU co-processing (V3D 4.2 QPUs next to the 4 A72 cores)

Notebook: `engine/GPU.md`. Code: `engine/gpu/`. Branch `worktree-agent-a459e8864e2e5ea3b`.

- **Question**: do the 8 QPUs add throughput to the compute-bound Cerno 4B matmuls
  (9216x2560, 2560x9216, 8192x2560, Q4_0, batch 7 / 64) when they run next to the 4 CPU
  threads (which are at 93-98 % of their best rate, ~54 GOP/s)?
- **Decisive number**: CPU + GPU together vs CPU alone (kbench `c4gate,c4down,c4qkv`, q4,
  4 threads, cold, with the GPU kernel looping on the same shapes at the same time).
- **Go / no-go**: go (scale-up = engine integration through the DRM ioctls) if combined
  >= 1.2x CPU alone with `vcgencmd get_throttled` = 0x0; otherwise stop.
- **Budget**: ~1.5 h of Pi time under the bench lock, bench names `gpu-*`.
- **Plan**: (1) smoke test (`hello.py`: lane add, QPU ids of an 8-batch dispatch);
  (2) `gpu-q4-alone`: `bench_q4.py` on the three shapes at n = 7 and 64 (64 = 8 passes of 8),
  GOP/s, bytes read, bit-exactness of the output against the kernel's fp32 model (which
  implies exact int16 block sums) on sampled tiles; (3) `gpu-corun`: `corun.sh` = CPU alone,
  GPU alone, both, CPU alone again, with temperature / throttle / clock samples every 2 s.
- **Status**: done 22:12 KST (one Pi slot 22:06-22:12 after the diagnosis at 22:03).
- **Measurements** (`engine/GPU.md` "Results"): GPU alone, bit-exact vs its fp32 model,
  gate / down / qkv at n = 7: 27.2 / 27.5 / 27.0 GOP/s, at n = 64 (8 passes of 8):
  27.8 / 28.1 / 27.5 (`bench/results/gpu-q4-alone_20261022_221029.json`). CPU alone (kbench
  q4, 4 thr, n = 7, cold, mean of two rounds) qkv 55.2 / gate 55.4 / down 50.3 GOP/s; with
  the GPU looping on gate n = 64: CPU 51.7 / 52.0 / 48.0 (-6 %) + GPU 26.1 = **77.8 / 78.0 /
  74.1 = 1.41x / 1.41x / 1.47x** (`bench/results/gpu-corun_20261022_221209.json`); throttled
  0x0, <= 63.3 C, ARM 1500 MHz in every loaded phase.
- **Dead end on the way**: TMU loads without the `thrsw` signal hang the V3D 4.2 (two slots
  lost: `gpu-q4-first_20261022_214808.json`, `gpu-diag_20261022_220321.json`).
- **Verdict**: **go** (1.41-1.47x >= 1.2x). Scale-up = the DRM ioctl integration designed in
  `engine/GPU.md`, class (b) opt-in, with a C feeder and a ~1/3 row split; first lever on the
  kernel: more TMU words in flight (it runs at 63 % of its instruction model).
