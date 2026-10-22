# audit (network audit: where the time goes, what is left; deliverable docs/network-audit.md)

- 17:27 [measured] **Cerno 4B prefill, per token at 1.5 GHz** (p256, 64-token chunks, PF_KERN/FWD_STATS;
  baseline csp0 = main; tools/audit/budget.py on bench/results/cerno-speed-profile1_20261022_171543.json,
  block 1). Wall 150.7 ms/token = matmul 140.0 (92.9%) + other 10.7:
  gate+up 57.1 ms (52.9 GOP/s), down 31.8 (47.5), in_proj_qkv 18.9 + q_proj 6.3 (53.2), out 10.1 + o 3.4 (50.0),
  z 9.5 (53.1), k+v 1.7, a+b 0.4 (19 GOP/s, 32-row shape); lin_rule 5.87, mlp_act 0.97, conv/l2/gates 0.95,
  attention 1.72, norms+add 0.81, gated norm 0.41. Matmuls run at **93-98% of the best kbench rate of
  their shape** (n = 7, cold, 4 threads) and **50-55% of the 96 GOP/s int8 instruction peak**; DRAM
  traffic ~313 MB/token (weights re-read per group of 7) = ~2.1 GB/s, half the roof: compute-bound.
- 17:27 [measured] speed4b's build (csp1) on the same file, block 0: 145.5 ms/token wall
  (matmul 134.9, other 10.6), i.e. -3.5%; decisions -3.7% mean (cerno-speed-decide-r1_20261022_171406.json).
- 17:27 [measured] **One Cerno decision = tokens x ~150 ms.** All 18 requests of sets A+B
  (cerno-pi results cerno-decide-reuse-{a2,b2}; tools/audit/cerno_rows.py): 565.5 s = shared prefix
  215.7 s (1435 tokens, 38%) + question rows 344.8 s (2268 tokens, 61%) + snapshots 4.5 s (0.8%) +
  Python 0.5 s. Joint fit over 52 rows: 149.6 ms per token + ~0.1 s per row; the restore is not
  separable from that 0.1 s (it is < 2% of a later question). Snapshot (Python save_state) 0.24 s median:
  4 copies of the ~51 MB blob in python/pinfer/state.py.
- 17:27 [answer:lead] **Template share of a question's tokens: 44%** (1003 of 2268 suffix tokens on the
  52 questions: "\nOptions:" 3 + per option "\n" "(X" ")" 3 + "\nAnswer: (" 4); question text 24%,
  option texts 32%. = 27% of all tokens of a decision. **None of it can move into the snapshot
  exactly**: it comes after the question text, so its KV rows, conv and DeltaNet state depend on the
  question. What is shared already sits in the snapshot: the LCP ends at "\n\nQuestion:" (e.g.
  support_triage prefix 15 tokens = "Context:\n" + 9 state tokens + "\n\nQuestion:"). Putting the
  template before the question is a different prompt layout = output-changing (class c).
- 17:27 [dead-end] **exact head screening for 0.8B decode** (skip head rows by a bound, keep the argmax
  exact): measured on the 10 golden positions with the real head (tools/audit/head_screen.py):
  Cauchy-Schwarz row-norm bound keeps 100% of rows; a 2-bit proxy (high bit-planes of a 4-bit code)
  with an exact bound on the dropped bits keeps 100%; reading 3 of 4 bits keeps 6-61%. The worst-case
  bound is ~sqrt(d) looser than the logit spread (bound 9-12 std vs max logit 6-10 std above mean).
- 17:27 [dead-end] for Cerno: batching several questions after one restore. Prefill is compute-bound
  with weights already reused 7x per group; batching saves only the ~0.1 s per row (<2%).
- 17:27 [dead-end] for Cerno: Q8 vs Q4 head and head tricks: eval_select reads only the label tiles
  (~10 x 4 rows); the head costs ~0 in a decision.
- 17:27 [dead-end] SDRAM clock: Raspberry Pi config.txt docs: "SDRAM overclocking on Raspberry Pi 4 or
  newer is not supported." (and Cerno is compute-bound anyway).
- 17:27 [for:lut] the bar for 4B prefill: Q4_0 batched kernel = 28-30 cycles per (4 rows x 32 cols x
  1 vector) = 128 MACs, i.e. 4.3-4.6 int8 MAC/cycle/core, incl. one SCVTF+FMUL+FMLA+FMLS per block-vector
  that a LUT kernel also pays (per-32-column weight scales). 92.9% of a decision is this loop.
- 17:27 [for:numerics] Winograd pairs on int8 SMULL: (x_a + w_b) must fit int8 and 2-4 products must
  fit an int16 lane (16 SMULL lanes accumulate 4 products today). x int8 +-127 plus w in [-8,7]
  overflows int8; activations limited to +-120 fit, but (x+w)^2 up to 18225 means one product per
  lane (SADALP after every SMULL), which costs back the saved multiplies. Needs <= 7-bit activations
  (class b) to keep 4 products per lane. The Q4_0 body (Cerno) has no integer sub-scales, so an
  integer superblock accumulation (one SCVTF per 256 columns) does not apply to it, only to Q4K (0.8B).
- 17:27 [for:compress] Cerno: compute-bound, MAC count fixed by the shapes: fewer bits per weight
  do not speed prefill unless MACs drop (low rank, pruning). 0.8B decode: bandwidth-bound, bytes count.
