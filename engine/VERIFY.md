# Speculative verify pass: lab notebook (issue #5, and #11 High)

Owner: verify-pass work (2026-10-22). Files: `engine/src/{kern_lin.c,forward_lin.c,spec.c,spec.h}`,
the linear-attention parts of `engine/src/{ops.h,kern.h,forward.h}`, `engine/bench/lbench.c`;
outside that list, minimally: `engine/src/matvec_ref.c` (the scalar twin of the same contract).
Q4K-headq4k model, 4 threads, Pi 4, all result files in `bench/results/`, all `valid=true`.

## Result

| prompt64, k=3, draft vocab 32768, 64 generated | before (`pinfer_run_fast`, md5 `9d30bbda`) | after (`pinfer_run_ver3`, md5 `82c7f4d2`) |
|---|---|---|
| speculative decode | 16.73 / 16.66 / 16.54 tok/s (mean **16.64**) | 18.36 / 18.49 / 18.39 tok/s (mean **18.42**, +10.7%) |
| verify pass (n = 4) | 159.5 ms | 152.9 ms |
| commit per round | 14.3 ms | 1.9 ms |
| MTP per round | 25.4 ms | 25.3 ms |
| plain decode step | 127.0 ms (7.87 tok/s) | 127.1 ms (7.87 tok/s) |
| verify / plain | 1.26 | **1.20** |
| (verify + commit) / plain | 1.37 | **1.22** |

Evidence: before `ver-before-p64_20261022_070046.json`; after `ver3-ab-p64_20261022_080513.json`.
That second job also ran the baseline interleaved: 16.67 / 16.68 tok/s, plus one disturbed run at
14.90 tok/s with verify 179.6 ms and MTP 27.4 ms, both inflated. Output is unchanged: spec_exact
20/20 with the baseline md5s (§4).

| ctx2048 (`data/ctx2048.ids`, `-c 2176`, 64 generated, `--no-warmup`) | before | after |
|---|---|---|
| `--spec 3 --draft-vocab 32768 --adaptive` | 9.31 / 8.82 / 9.33 tok/s (mean **9.15**) | 10.31 / 11.04 / 10.79 tok/s (mean **10.71**, +17%) |
| fixed k=3: tok/s, verify, commit | 8.90, 192.6 ms, 13.4 ms | 9.55, 187.3 ms, 0.8 ms |
| plain | 6.89 tok/s (145.1 ms) | 7.06 tok/s (141.7 ms) |
| fixed k=3 (verify + commit) / plain | 1.42 | 1.33 |

Evidence: `ver-before-ctx2048_20261022_070717.json`, `ver3-ctx2048_20261022_082417.json`. Adaptive
gains more than fixed k. The measured part: two of the three after-runs reach 2.17 tokens per
pass in 29 rounds, while all three before-runs stay at 1.80–1.97 in 32–35 rounds. The likely
cause (an inference) is that the cost model now sees cheaper rounds and keeps more drafts.
Adaptive runs vary run to run, because their decisions depend on measured times.

## 1. Re-profile of the shipping binary (06:51)

`PF_FWD_STATS=1 PF_KERN_STATS=1`, prompt64, `--no-warmup`
(`ver-before-profile_20261022_065138.json`):

| per plain step (63) | ms | per verify pass n = 4 (19) | ms |
|---|---|---|---|
| matmul (KSTATS n = 1) | ~115 | matmul (KSTATS n = 4, incl. ~13 MTP 4-row passes) | ~130 |
| recurrence `lin_rule` | 6.45 | recurrence incl. its scratch copy of the state | **22.4** |
| conv + l2 + gates | 1.04 | conv + l2 + gates | 2.45 |
| attention, MLP act, norms, gated norm, argmax | 2.2 | log copies (conv state + raw inputs) | 1.26 |
| | | attention 2.61, MLP act 0.89, norms 0.73, gated norm 0.67 | 4.9 |
| **step (bench)** | **127.0** | **pass (bench)** | **158.4**, then commit **13.5** |

The vectorized recurrence did not change the picture of the old profile (`engine/CORE.md`:
22.8 + 14.2 ms). The two items in this work's files are 22.4 + 13.5 = 36 ms per round, 18% of a
199 ms round. lbench (below) times the same n = 4 work in place at 16.1 ms, so the copy of the 1 MB
state into the shared scratch costs about 6 ms. The commit is a second DRAM read and write of
the whole 18.9 MB state, plus ~0.8 ms of acceptance argmaxes that `spec.c` times as part of it.

## 2. Design: token 0 in place, drafts on a stack copy, deferred commit

Problem: a round reads and writes the 18.9 MB recurrent state twice (copy + verify, then replay),
where a plain step does it once.

1. **Token 0 is always committed.** `e >= 1` in `pf_spec_decode`, so `cur` (row 0 of the
   verify pass) is always accepted. `pf_lin_rule` now runs token 0 in place on S, exactly like a
   plain token, and the conv pre-task shifts the live conv state for token 0.
2. **Drafts run on a per-thread copy.** For tokens 1..n-1, `kern_lin.c` writes the chunk being
   processed (16 KB = 128 rows × 32 columns, chunk-major layout) to a buffer on the thread's
   stack instead of S. That buffer is L1-resident and reused for every chunk. The conv state is
   copied per channel range, in parallel, inside the pre-task, after token 0. The log keeps rows
   t >= 1 only.
3. **Deferred commit.** `pf_lin_commit(keep)` updates the conv state (72 KB per layer) and only
   *records* the `keep - 1` rank-1 updates each layer still owes (`pf_lin_log.keep[l]`). The
   layer's next pass, verify or plain, applies them first, while each chunk is in L1 anyway.
   The log is double-buffered (`upd[l]` holds two halves, `wr` alternates), because the next
   verify pass writes its own log while it applies the previous one.
4. **Folded replay (v3).** The deferred updates are applied in registers during token 0's first
   sweep (`replay_first`): one load and one store per row instead of one per update plus one.
5. **No pending state outside the loop.** `pf_spec_decode` calls `pf_lin_flush` before it
   returns, so `pinfer_reset`, `pinfer_state_save/load`, `pinfer_eval` and the next prefill
   always see complete states. Without the flush the next forward pass would still self-heal,
   but `state_save` would not; the negative control in §4 shows it.

Exactness argument: every element goes through the same fp32 operations in the same order
(`S *= decay`, `S += k_i * delta_j`, rounded separately, no FMA under `-ffp-contract=off`), only
at a different time. The NEON kernel and the scalar reference (`matvec_ref.c`) implement the
same contract (`ops.h` `pf_lin_args`: `pend`, `keep`, `work`).

Traffic per round on the recurrent state: before, copy (read + write 18.9 MB), verify, then
commit (read + write 18.9 MB). After: one read and one write, the same as a plain step.
Memory: the double-buffered log adds 1.2 MB at k = 3 (a second half of 18 × 4 × 16 × 257
floats) and the
1 MB shared scratch is gone: peak RSS at k = 3 is 451.1 MB vs 452.0 MB
(`ver3-profile_20261022_081052.json`, `ver-before-profile_20261022_065138.json`).

## 3. Issue #11 (High): the shared `scratch_state`

`kern_lin.c` had `static float *scratch_state`, grown with `free` + `aligned_alloc` and used by every
logged pass of every context. `matvec_ref.c` had the same `static lin_scratch`, and that is the one
in the x86 wheel. Both are gone. The NEON kernel uses the per-thread stack chunk of §2.2
(`PF_LIN_MAX_DK × 32` floats; `pf_lin_log_alloc` refuses a model with `lk_dim > 256`, so such a
model can still decode plainly). The reference uses the per-context `pf_lin_log.S`, which was
allocated before but never used.

Evidence: a throwaway harness `logs/verify/thread_cmp.c` runs two contexts on two threads. Each
generates with k = 3 at the same time as the other, and the output is compared with the same
context generating alone.

| build | concurrent runs that differ from alone | evidence |
|---|---|---|
| host ref, HEAD `3c2c887` | 2/3 and 2/3 | `logs/verify/thread_cmp_host.txt` |
| host ref, this work | 0/3 and 0/3 | same file |
| Pi NEON, HEAD `3c2c887` (`thread_cmp_old`) | 3/4 and 4/4 | `logs/runs/20261022_083027_run.log` |
| Pi NEON, this work (`thread_cmp_new`) | 0/4 and 0/4 | same log |

## 4. Exactness gates (final code)

| gate | result | evidence |
|---|---|---|
| host `tests/test_golden.py` | PASS | `logs/verify/host_gates_v3.txt` |
| host `tests/test_spec.py` (incl. `k3adapt`, `k3b2`) | PASS | same |
| host `tests/test_state.py` | PASS | same |
| host pytest (source tree, `PINFER_TEST_REQUIRE=1`) | 85 passed | same |
| host state harness `logs/verify/state_cmp.c`: after k=1/2/3/4/6, full vocab, adaptive, and after a `generate_greedy_cont` continuation, the saved state (conv + recurrent + KV rows) is byte-identical to plain decoding; 3 prompts | 21/21 identical | `logs/verify/state_cmp.txt` |
| negative control: same harness, library without the final `pf_lin_flush` | 5/7 states differ (fails as it should) | same file |
| Pi `lbench_ver3 --check --n 1,2,3,4,5,8,64` (logged pass, replay, logged + deferred, in place + deferred, in place vs the scalar reference, memcmp) | all exact | `ver3-exact_20261022_075930.json` |
| negative control: lbench with the last deferred update skipped / the logged pass fully in place | FAIL for n >= 2, as it should (qemu) | this notebook |
| Pi `spec_exact.sh ./engine/pinfer_run_ver3` | 20/20 `match`, md5s = baseline (`703b5ff0`, `3f9a4f66`, `08c27da6`, `3f9a4f66` as in `spec-exact-ctx7_20261022_060828.json`) | `ver3-exact_20261022_075930.json` |
| Pi `ctx_check.sh ./engine/pinfer_run_ver3 ./engine/pinfer_run_fast` (non-speculative forward) | 5/5 identical | same |
| `-Wall -Wextra`, host (ref) and aarch64 (fast), engine and lbench | no warnings | build log |

## 5. Steps and what each was worth

| version | change | k=3 tok/s (same-job A/B) | evidence |
|---|---|---|---|
| baseline | | 16.64 | `ver-before-p64_20261022_070046.json` |
| v1 | §2.1–2.3, §2.5 | 18.31 / 17.88 / 18.21 (verify 155.9 ms, commit 1.87 ms) | `ver1-p64_20261022_071359.json` |
| v3 | + folded replay §2.4 | 18.58 / 18.03 / 18.60 vs v1 18.22 / 16.84 (disturbed) / 18.33: +1.5–2%, verify −3 ms | `ver1-vs-ver3-p64_20261022_081538.json` |

v1 profile (`ver1-profile_20261022_071535.json`): recurrence per verify pass 22.4 → 14.3 ms,
conv pre-task + log copies 3.71 → 3.37 ms, non-matmul forward per pass 31.0 → 22.6 ms. v3
profile (`ver3-profile_20261022_081052.json`; this run's matmuls were ~4% slower throughout,
plain n = 1 included): recurrence 13.7 ms, non-matmul 22.2 ms.

lbench, 4 threads, 18 layers, cold state, ms per pass (`lbench-ver1-vs-ver2k_20261022_071447.json`,
`lbench-ver1-vs-ver3f_20261022_074709.json`):

| n | in place (plain / prefill) | v1 verify with n−1 deferred | v1 logged + separate replay (old round without its copy) | v3 verify with n−1 deferred |
|---|---|---|---|---|
| 1 | 12.6 | | | |
| 2 | 13.8 | 14.1–14.9 | 22.8–23.2 | 13.7–14.1 |
| 3 | | 16.4–16.6 | | 14.9–15.1 |
| 4 | 16.1 | 18.3–18.8 | 25.7–26.0 | 16.2 (2 of 3 runs; the third 28.4, disturbed) |
| 64 | 141.9 | | | |

The v3 column and the v1 n = 3 values are from the A/B job `lbench-ver1-vs-ver3f_20261022_074709.json`;
the v1 ranges also include `lbench-ver1-vs-pf_20261022_072227.json` (its first v1 round was
disturbed: n = 1 19.2 ms, n = 2 verify 22.8 ms; those runs are excluded).

## 6. Tried and dropped

| idea | measurement | decision |
|---|---|---|
| **Fused sweeps** (v2k): pass 2 of token t and pass 1 of token t+1 in one sweep, so each row is loaded and stored once per token. The 32-column version spilled (393 q-register spills in the asm); the 16-column register blocks had none | n = 4: in place 16.88 vs 16.09 ms, verify 18.44 vs 18.29; n = 64: 143.0 vs 141.9; 1 thread n = 64: 544 vs 552 (−1.3%), n = 4: 44.3 vs 41.3 | slower where it matters → dropped (`lbench-ver1-vs-ver2k_20261022_071447.json`) |
| **Software prefetch** of the next 16 KB chunk during the first sweep, L1 and L2 hints | n = 4 verify 19.6–19.9 (L1) and 19.3–19.7 (L2) vs 18.6–18.8 ms; n = 1 unchanged | the hardware prefetcher already streams the contiguous chunks → dropped (`lbench-ver1-vs-pf_20261022_072227.json`) |
| **Deferred conv commit** (ver2): the conv-state shift of a commit moved into the next pass's parallel pre-task | commit 1.86 → 1.48 ms, verify +0.4 ms; k=3 18.11 vs 18.13 tok/s (same job) | no gain → reverted; the reverted build is md5-identical to v1 (`ver2-p64_20261022_073144.json`, `spec-exact-ver2_20261022_072815.json`) |

## 7. k sweep after the change (`ver1-ksweep_20261022_074459.json`, one run per cell)

| prompt | binary | plain | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| prompt64 (Dickens) | before | 7.93 | 14.45 | 16.09 | 15.63 |
| | v1 | 7.94 | 16.25 | **18.23** | 15.85 |
| chat27 | before | 7.96 | 9.88 | 9.45 | 7.28 |
| | v1 | 7.90 | **11.00** | 10.52 | 7.76 |

k = 4 stays poor because its n = 5 verify pass costs 205–213 ms against ~153 ms at n = 4: in
`kern_quant.c`, `VECTOR_PASSES` runs 5 vectors as a 4-pass plus a 1-pass over every weight
tile, so the tile is decoded twice. That is in the kernels' files and not changed here. The
default `pinfer_default_spec_params()` (k = 2) is left as it is, because k = 2 still wins on
chat27.

## 8. Open items

- The verify pass is now 1.20× a plain step. Most of the remaining gap is the batched matmuls
  (~130 ms at n = 4 vs ~115 ms at n = 1, in the kernels' files). The 1.1 target needs them.
- The argmaxes of the accepted verify rows run serially on the main thread (~0.24 ms per
  248k-row argmax, ~0.8 ms per round). Threading them would save ~0.6 ms per round (0.3%), but
  it has to reproduce `pf_argmax`'s NaN behaviour exactly; not done.
- `pf_lin_flush` costs ~12–14 ms once per `pf_spec_decode` call (one replay of the last round).
  A lazy flush across calls would need hooks in `pinfer_reset` and `pinfer_state_save` /
  `pinfer_state_load` (other owners' files).
- k = 4 is bound by the n = 5 matmul path (§7).
- `docs/ARCHITECTURE.md` §6 and the glossary describe the new rollback. The historical numbers
  in `engine/CORE.md`, `engine/bench/KERNELS.md` and `docs/RESULTS.md` are dated records and
  were left alone.
