"""PyTorch + transformers reference baseline: shared 64-token prompt, 64 greedy tokens.

Runs on the Pi inside ~/pinfer/venvs/torch (torch 2.14.1+cpu, transformers 5.20.2):
    venvs/torch/bin/python bench/torch/run_bench.py --model models/Qwen3.5-0.8B --threads 4
Prints one `RESULT {json}` line. One process per thread count (ru_maxrss = that config's peak).

Text-only load: AutoModelForCausalLM -> Qwen3_5ForCausalLM (vision tower and MTP not loaded).
Timing (time.perf_counter, torch.inference_mode, batch 1):
    prefill = one forward over the 64 prompt ids (use_cache=True)
    decode  = 64 single-token forwards with the returned cache, greedy argmax between them
              (token 1 comes from the prefill logits, as in llama-bench tg64)
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def one_run(model, prompt: list[int], n_gen: int) -> tuple[float, float, list[int]]:
    ids = torch.tensor([prompt])
    with torch.inference_mode():
        t0 = time.perf_counter()
        out = model(input_ids=ids, use_cache=True)
        t1 = time.perf_counter()
        nxt = int(out.logits[0, -1].argmax())
        past = out.past_key_values
        seq = [nxt]
        t2 = time.perf_counter()
        for _ in range(n_gen):
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            seq.append(nxt)
        t3 = time.perf_counter()
    return len(prompt) / (t1 - t0), n_gen / (t3 - t2), seq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.5-0.8B"))
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--warmup-gen", type=int, default=8, help="tokens in the (excluded) warm-up run")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    ap.add_argument("--prompt", default=str(ROOT / "bench/common/prompt64.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    spec = json.loads(Path(args.prompt).read_text(encoding="utf-8"))
    prompt, ref = spec["ids"], spec["reference_greedy"]["ids"]

    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype])
    model.eval()
    load_s = time.perf_counter() - t_load
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    _, _, warm = one_run(model, prompt, args.warmup_gen)
    pre, dec, matches = [], [], []
    for _ in range(args.reps):
        p, d, seq = one_run(model, prompt, args.n_gen)
        pre.append(p)
        dec.append(d)
        matches.append(next((k for k, (a, b) in enumerate(zip(seq, ref)) if a != b),
                            min(len(seq), len(ref))))

    sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0  # noqa: E731
    print("RESULT " + json.dumps({
        "engine": "pytorch+transformers",
        "engine_version": f"torch {torch.__version__}, transformers {transformers.__version__}",
        "model_file": args.model,
        "quant": args.dtype,
        "threads": args.threads,
        "n_prompt": len(prompt),
        "n_gen": args.n_gen,
        "reps": args.reps,
        "prefill_tps": statistics.mean(pre),
        "prefill_tps_sd": sd(pre),
        "decode_tps": statistics.mean(dec),
        "decode_tps_sd": sd(dec),
        "prefill_samples": [round(x, 4) for x in pre],
        "decode_samples": [round(x, 4) for x in dec],
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "rss_after_load_mb": rss_after_load,
        "load_s": round(load_s, 2),
        "min_ref_match": min(matches),
        "warmup_first8": warm[:8],
        "notes": f"{type(model).__name__}; text-only; manual KV-cache greedy loop; warm-up run "
                 f"({args.warmup_gen} tokens) excluded; cpu capability "
                 f"{torch.backends.cpu.get_cpu_capability()}; RSS includes Python",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
