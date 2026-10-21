# Glossary

Terms that recur across the code, notebooks and result files. Details: [ARCHITECTURE.md](ARCHITECTURE.md),
[RESULTS.md](RESULTS.md).

**Acceptance (accept rate).** Share of drafted tokens the main model confirms: `accepted / drafted`, the
`accept_rate` field of a `pinfer_run --spec` result. See also *tokens per pass*.

**Bench lock.** A Pi-wide `flock` on `/tmp/pinfer-bench.lock`, held by every `harness/pi.py bench` and
`run --lock` call, so two timed jobs never overlap. A locked job is killed after 20 minutes (exit 124).

**Bit-exact.** Two code paths produce identical bits: batched vs single-token, NEON vs scalar twin, any
thread count. Required for speculative decoding to match plain decoding (ARCHITECTURE.md §7).

**Bonus token.** In a speculative round, the main model's own token after the last accepted draft. Every
round emits at least this one token.

**Commit (rollback by replay).** After a verify pass, `pf_lin_commit` hands the logged rank-1 updates
of the accepted tokens to the Gated DeltaNet states. The layer's next pass replays them, or
`pf_lin_flush` does at the end of generation. Rejected tokens are simply never replayed; no state
snapshot is taken (`engine/VERIFY.md` §2).

**Decode.** Generating one token per step. Each step reads every weight once, so on the Pi it is limited
by memory bandwidth: tok/s ≈ bandwidth ÷ bytes read per token (`docs/design.md` §3).

**Draft vocabulary** (`--draft-vocab N`, `pinfer_spec_params.draft_vocab`). Drafts take the argmax over
token ids below N only (default 32,768), which makes a draft step about 5× cheaper. It never changes the
output; it can lower acceptance (RESULTS.md §3).

**Gated DeltaNet.** The linear-attention layer type of Qwen3.5 (18 of 24 layers). Each head keeps a fixed
128 × 128 state updated per token by the gated delta rule (`S = S·decay; δ = (v − Sᵀk)·β; S += k ⊗ δ;
o = Sᵀq`), between a causal conv (kernel 4) and a gated RMSNorm (`engine/FORMAT.md`). Its memory does not
grow with context: 18.9 MB for all 18 layers.

**Golden reference.** Outputs of the official `transformers` implementation run in fp32 on the PC
(`ref/dump_reference.py`, `ref/mtp_reference.py` → `ref/golden/`, gitignored). `tests/test_golden.py` and
`tests/test_spec.py` compare the engine against it.

**GQA (grouped-query attention).** The 8 query heads share 2 key/value heads (4 per group), so the KV
cache stores 2 × 256 values for K and for V per layer and token: 24 KiB per token over 6 layers in fp32.

**isolcpus / nohz_full / rcu_nocbs.** Kernel command-line options that keep the scheduler off cores 1–3,
stop their periodic timer tick, and move RCU callbacks off them. `nohz_full` needs `NO_HZ_FULL`, which
the stock Raspberry Pi kernel lacks, hence the custom kernel (`os/kernel/KERNEL.md`). Measured effect on
pinfer: under 1%.

**k.** Drafts per speculative round (`--spec k`, `pinfer_spec_params.k`, at most 8). k = 0 is plain
decoding. API default 2; the benchmarks use 3.

**memband.** `bench/memband/memband.c`: a streaming read of a 256 MB buffer with 1, 2 and 4 threads. 4.20
GB/s on one thread, the ceiling for decode.

**MTP (multi-token prediction) head.** Qwen3.5's extra one-layer block (`mtp.*`) that predicts the token
after next from the main model's final-normed hidden state and the next token. pinfer uses it as its own
draft model (`engine/SPEC.md`).

**Partial RoPE.** Rotary position embedding on only the first 64 of each attention head's 256 dims
(rotate-half pairs i and i+32, θ = 1e7).

**Perplexity protocol.** The `llama-perplexity` method from llama.cpp: wikitext-2-raw test text, 512-token
chunks, state reset per chunk, the second half of each chunk scored (255 tokens). pinfer reproduces it in
`engine/tools/pinfer_ppl.c` and runs 8 chunks (2,040 scored tokens) on the Pi.

**`.pq` file.** pinfer's model format: a 4096-byte header, a tensor directory, then 64-byte-aligned
tensors, written by `tools/convert.py` (`engine/FORMAT.md`).

**Prefill.** Processing the prompt. pinfer feeds it in chunks of up to 64 tokens, so many tokens share
each pass over the weights and the cost moves from memory traffic to compute (`engine/bench/KERNELS.md`
Part 2).

**Q4K superblock (Q4K_G32).** The unit of the default 4-bit format: 4 rows × 256 columns in 576 bytes. A
64-byte scale line (per row an fp16 super-scale D and min M; per 32 columns a 6-bit sub-scale s and
sub-min m) is followed by 4-bit codes u; weight = D·s·u − M·m, 4.5 bits per weight. Q4_G32 and Q8_G32 use
the same tiling with plain fp16 scales (`engine/FORMAT.md`).

**Speculative decoding.** Draft k tokens cheaply, check them all in one batched pass of the main model,
keep the longest prefix that equals the model's own greedy choices plus the bonus token. In pinfer the
output is identical to plain greedy decoding (ARCHITECTURE.md §6).

**THP (transparent huge pages).** Linux maps memory in 2 MB pages instead of 4 KB ones. Off in the stock
Pi kernel, `madvise` mode in the custom kernel; `PINFER_HUGEPAGES=1` copies the weights into such a
mapping. No measurable speed gain (RESULTS.md §10).

**Tokens per pass.** `1 + accepted / rounds`: tokens emitted per verify pass (`pinfer.h`
`pinfer_spec_stats`).

**tryboot.** Raspberry Pi firmware one-shot boot: `sudo reboot "0 tryboot"` boots once with
`tryboot.txt` (here: the custom kernel from `/boot/firmware/pinfer/`). The next normal reboot, a panic
(`panic=10`) or a power cycle returns to the stock kernel (`os/kernel/KERNEL.md` §5).

**valid.** Flag in every `bench/results/*.json`: exit code 0, throttle flags `0x0` after the run, and every
`RESULT` line parsed. It says nothing about whether the output was correct.

**Verify pass, verify ratio.** The batched main-model pass over the current token and k drafts, and its
cost relative to one plain decode step: 1.28 at k = 3 on the default model (`engine/SPEC.md` calls it
`c_v`).
