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
- **Status**: PC side done (generator, scheduler, emulator check bit-exact at n = 1, 2, 7, 8,
  11); Pi runs pending (the Pi went down in the overclock test at 17:5x KST, back at stock).
- **Measurements**: (filled in from `bench/results/gpu-*.json`)
- **Verdict**: pending.
