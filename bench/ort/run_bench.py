"""ONNX Runtime GenAI baseline: shared 64-token prompt, 64 greedy tokens, one thread count.

Runs on the Pi inside ~/pinfer/venvs/ort (onnxruntime-genai 0.18.0, onnxruntime 1.31.0):
    venvs/ort/bin/python bench/ort/run_bench.py --model models/ort/int4-acc4 --threads 4
Prints one `RESULT {json}` line. Run one process per thread count, so ru_maxrss is that
config's peak RSS (it includes the Python interpreter, ~25 MB).

Timing (time.perf_counter around ORT GenAI calls):
    prefill = Generator.append_tokens(64 ids)        -> one forward over the prompt
    decode  = generate_next_token() calls 2..65       -> 64 single-token forwards (+ greedy argmax)
The first generate_next_token() only picks token 1 from the prefill logits, so it is excluded;
this matches llama-bench tg64 (64 single-token evaluations).
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import onnxruntime  # noqa: F401  (must load before onnxruntime_genai, or its dlopen fails)
import onnxruntime_genai as og

ROOT = Path(__file__).resolve().parents[2]


def build_model(model_dir: str, threads: int, mtp: bool) -> og.Model:
    config = og.Config(model_dir)
    config.clear_providers()  # CPU EP
    # The export dir must have been passed through bench/ort/fix_config.py (removes
    # shared_initializers, which break og.Model() on the CPU EP in genai 0.18.0).
    overlay = {"model": {"decoder": {"session_options": {
        "intra_op_num_threads": threads, "inter_op_num_threads": 1}},
        "mtp": {"enabled": mtp}}}
    config.overlay(json.dumps(overlay))
    return og.Model(config)


def one_run(model: og.Model, prompt: list[int], n_gen: int) -> tuple[float, float, list[int], dict]:
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(prompt) + n_gen + 8)
    gen = og.Generator(model, params)
    t0 = time.perf_counter()
    gen.append_tokens(prompt)
    t1 = time.perf_counter()
    gen.generate_next_token()  # token 1 from prefill logits (no forward)
    t2 = time.perf_counter()
    steps = 0
    while steps < n_gen and not gen.is_done():
        gen.generate_next_token()
        steps += 1
    t3 = time.perf_counter()
    seq = [int(t) for t in gen.get_sequence(0)][len(prompt):]
    try:
        stats = gen.get_speculative_stats()
        spec = {k: stats.get(k) for k in ("rounds", "draft_tokens_accepted", "target_forward_passes")}
    except Exception as e:  # noqa: BLE001  (stats API absent when not speculating)
        spec = {"error": str(e)}
    return (len(prompt) / (t1 - t0), steps / (t3 - t2), seq, {"decode_steps": steps, "spec": str(spec)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--prompt", default=str(ROOT / "bench/common/prompt64.json"))
    ap.add_argument("--mtp", action="store_true", help="leave MTP speculative decoding enabled")
    ap.add_argument("--quant", default="")
    args = ap.parse_args()

    spec = json.loads(Path(args.prompt).read_text(encoding="utf-8"))
    prompt, ref = spec["ids"], spec["reference_greedy"]["ids"]
    t_load = time.perf_counter()
    model = build_model(args.model, args.threads, args.mtp)
    load_s = time.perf_counter() - t_load

    pre, dec, matches, info = [], [], [], {}
    first = []
    for i in range(args.reps + 1):  # rep 0 = warm-up, excluded
        p, d, seq, info = one_run(model, prompt, args.n_gen)
        m = next((k for k, (a, b) in enumerate(zip(seq, ref)) if a != b), min(len(seq), len(ref)))
        if i == 0:
            first = seq[:16]
            continue
        pre.append(p)
        dec.append(d)
        matches.append(m)

    sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0  # noqa: E731
    result = {
        "engine": "onnxruntime-genai",
        "engine_version": f"genai {og.__version__}, ort {onnxruntime.__version__}",
        "model_file": args.model,
        "quant": args.quant,
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
        "load_s": round(load_s, 2),
        "min_ref_match": min(matches),
        "warmup_first16": first,
        "decode_steps": info.get("decode_steps"),
        "notes": f"mtp={'on' if args.mtp else 'off'}; CPU EP; intra_op={args.threads}; "
                 f"warm-up rep excluded; spec_stats={info.get('spec')}; RSS includes Python",
    }
    print("RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
