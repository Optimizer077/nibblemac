# lead

- 17:30 [measured] 4B prefill p256 (pre-tuning, csp0, with stats overhead): matmul 36.7 of 39.6 s (93%);
  per shape GOP/s at n = 64 groups: 9216x2560 and 2560x9216 dominate (15.0 + 8.2 s), the DeltaNet
  recurrence `lin_rule` 1.49 s (3.8%), attention 0.44 s, norms / activations 0.46 s
  (evidence: bench/results/cerno-speed-profile0_20261022_155058.json)
- 17:30 [measured] speed4b round 1: 4B prefill 6.49 / 6.62 / 6.46 -> 6.87 / 6.82 / 6.80 tok/s
  (64 / 256 / 512 tokens); 0.8B prefill 43.8 -> 45.1, plain 7.81 -> 7.80, spec 20.19 -> 20.26
  (evidence: cerno-speed-e2e4b-r1_20261022_171014.json, cerno-speed-e2e08_20261022_171753.json)
- 17:30 [dead-end] huge pages / THP for the weight stream: within 1% on the custom kernel
  (os/kernel/KERNEL.md K3 / K4, README section 4); the stock kernel has no THP at all.
- 17:30 [dead-end] DeltaNet chunked (WY) prefill: at most ~4% of 4B prefill (lin_rule share above).
- 17:30 [measured] 0.8B decode: 126.7 ms / token, ~118.6 ms of it weight streaming = 3.58 GB/s of the
  4.20 GB/s measured peak; head 143.0 of 424.9 MB (tools/make_figures2.py STEP_PLAIN).
- 17:30 [idea] for:numerics + for:lut: Winograd's inner product and the LUT kernel both attack the
  multiplier wall; if both work they may combine (pre-added pairs as LUT indices?). Compare notes.
- 17:30 [idea] for:compress + for:lut: a 2-bit codebook (16-entry) head would fit TBL tables; the
  LUT kernel's table build is the same machinery.
- 17:30 [question:audit] what fraction of a Cerno question's tokens is template shared by every question
  (movable into the snapshot)?
- 17:27 [answer:audit] 44% of a question's suffix tokens are template (1003 of 2268 on the 52 questions), 27% of a decision's tokens; none is movable into the snapshot exactly (it follows the question text); the LCP already ends at "\n\nQuestion:". Details: plan/board/audit.md 17:27.
- 17:52 [decision] **Owner: focus on Cerno.** The target is Cerno 4B decision time (= 4B prefill:
  ~150 ms per token, 93% Q4_0 matmul at n = 7 / 5 groups, K = 2560 / 4096 / 9216). 0.8B-only work
  (decode streaming, chat drafting) is stopped. Every agent: measure your idea on the 4B shapes and on
  Cerno decisions first (tmp/csp/decide_ab.sh on the Pi runs 5 requests / 14 questions; its 1.5 GHz
  baseline is cerno-speed-decide-r1_20261022_171406.json, mean 21.85 -> 21.04 s with speed4b);
  keep the 0.8B gates as regression checks only.
- 17:52 [plan] **Overclock window (lead):** soon, about 40 min, announced here with start time. During
  the window the Pi reboots once into 2.0 GHz and back; `harness/pi.py` will refuse other agents'
  run / bench / push with a clear "paused" message, so do PC-side work then and retry after.
