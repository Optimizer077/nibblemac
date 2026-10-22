# toolchain (compilers, flags, LTO, PGO, linking, load time; notebook `engine/TOOLCHAIN.md` on branch
`worktree-agent-a75a53a7559d5b274`, base main ab8839a; measurements on the speed4b tree 69b31a3)

(Kept in my worktree: `.claude/worktrees/agent-a75a53a7559d5b274/plan/board/toolchain.md`.)

- 18:10 [measured] **Installed in WSL (apt):** aarch64 cross GCC 13.4 / 14.3 / 16.0.1 (20260322 snapshot)
  next to the existing 15.2, clang/lld/llvm 20.1.8 and 22.1.2. All build the engine static for the Pi.
- 18:10 [measured] **Static codegen of the Q4 kernel (kern_quant.c), one block of an nv-vector pass,
  instructions / stack spill ops** (llvm-objdump; scratch `~/tc/kq` in WSL):
  GCC 15 -O3: nv=7 232 / 12, nv=5 171 / 8, matvec nv=1 (8 blocks unrolled) 395 / 10.
  GCC 13/14/15/16 + `-fno-schedule-insns`: nv=7 217-219 / 0, nv=5 164-165 / 0, nv=1 365-375 / 0
  (GCC 14 = 15 = 16 instruction counts). `-fsched-pressure` (either algorithm) changes nothing;
  `-mtune=cortex-a57` = a72 (same .text); `-fira-algorithm=priority` +48 MOVs (worse).
  clang 22 -O3: nv=7 239 / 32 spills (worse than GCC), nv=5 161 / 7, nv=1 355 / 0.
  clang 20/22 + `-mllvm -enable-misched=false`: nv=7 **210 / 0**, nv=5 **156 / 0**, nv=1 358 / 0;
  Q4K nv=7 208 / 0 (GCC-ns 221). Counts are a proxy; Pi kbench A/B next.
- 18:10 [for:speed4b] your `-fno-schedule-insns` result holds for GCC 13, 14 and 16 too; it also removes
  the spills of the single-vector (decode) Q4 loop (395 -> 365 instructions per 8 blocks). I will
  measure it next to clang without the machine scheduler; I won't duplicate your Makefile change.
- 18:10 [for:lut] if you hand-order the Q4 inner loop: GCC's sched2 (post-RA) still reorders it under
  `-fno-schedule-insns`; `-fno-schedule-insns2` gives a different .text (same counts). Inline asm or
  `-fno-schedule-insns -fno-schedule-insns2` on that file keeps your order.
