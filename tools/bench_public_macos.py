"""Matched local nibblemac / llama.cpp Metal / MLX-LM measurements.

Run setup_public_baselines.py, build the pinned llama adapter, convert MLX,
and build mac-cpu/mac-infer first. See docs/PUBLIC_BASELINES.md.
"""
from datetime import datetime, timezone
import argparse
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np
from tokenizers import Tokenizer
from bench_inference_macos import ROOT, prompts, sha256, source_hashes, text


def parse_metrics(stdout, stderr, cfg, generate, chunks):
    results = [json.loads(line[7:]) for line in stdout.splitlines() if line.startswith("RESULT ")]
    if len(results) != 1:
        raise ValueError("missing/duplicated RESULT")
    metrics = results[0]
    if cfg == "nibblemac-metal-head6" and "Metal calls=" not in stderr:
        raise ValueError("nibblemac Metal did no GPU work")
    if cfg == "llama-metal" and not re.search(r"offloaded [1-9]\d*/\d+ layers to GPU", stderr):
        raise ValueError("llama GPU offload not confirmed")
    if not chunks:
        if cfg.startswith("nibblemac"):
            match = re.search(r"^generated:\s*([\d,]+)", stdout, re.M)
            if not match:
                raise ValueError("missing generated IDs")
            metrics["generated_ids"] = list(map(int, match.group(1).split(",")))
        ids = metrics.get("generated_ids", [])
        if len(ids) != generate or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("generated count/type mismatch")
        for metric in ("prefill_tps", "decode_tps", "peak_rss_mb"):
            value = metrics.get(metric, 0)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid metric")
    elif not math.isfinite(metrics.get("ppl", float("nan"))) or metrics["ppl"] <= 0 or metrics.get("tokens_scored") != chunks * 255:
        raise ValueError("invalid quality metric")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--generate", type=int, default=128)
    ap.add_argument("--quality-chunks", type=int, default=8)
    ap.add_argument("--label", default="public-baselines")
    args = ap.parse_args()
    if args.rounds < 1 or args.generate < 2 or args.quality_chunks < 0 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        ap.error("invalid arguments")
    work = ROOT / "build/public-comparison"
    work.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / f"bench/results/metal/{args.label}-{stamp}.json"
    tokenizer_dir = ROOT / "models/Qwen3.5-0.8B"
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    cases = prompts(tokenizer, tokenizer_dir)
    pq = ROOT / "models/qwen35-0.8b-q4k.pq"
    gguf = ROOT / "models/public-baselines/Qwen3.5-0.8B-Q4_K_M.gguf"
    mlx = ROOT / "models/public-baselines/mlx-q4-g32"
    configs = ["nibblemac-cpu6", "nibblemac-metal-head6", "llama-metal", "mlx-q4-g32"]
    files = [pq, gguf, *sorted(mlx.glob("*.safetensors")), mlx / "config.json", tokenizer_dir / "tokenizer.json"]
    sources = source_hashes()
    for path in sorted((ROOT / "tools").glob("*public*")):
        if path.is_file():
            sources[str(path.relative_to(ROOT))] = sha256(path)
    record = {"kind": "public_runtime_comparison", "valid": False, "timestamp_utc": stamp,
        "revision": text("git", "rev-parse", "HEAD"), "source_sha256": sources,
        "models": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
        "provenance": {"official": json.loads((tokenizer_dir / "source.json").read_text()),
                       "gguf": json.loads((gguf.parent / "gguf-source.json").read_text())},
        "versions": {"llama_commit": text("git", "-C", "build/third-party/llama.cpp", "rev-parse", "HEAD"),
                     **{p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "mlx-metal", "transformers")}},
        "hardware": {"chip": text("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": text("sysctl", "-n", "hw.memsize"), "os": text("sw_vers", "-productVersion"),
            "build": text("sw_vers", "-buildVersion"), "power_before": text("pmset", "-g", "batt"),
            "thermal_before": text("pmset", "-g", "therm")},
        "protocol": {"rounds": args.rounds, "generated_ids": args.generate, "decode_steps": args.generate - 1,
            "prefill_batch": "64 maximum; MLX pre-fills all but last, then evaluates last token",
            "max_context": "nibblemac/llama 2048; MLX dynamic cache, same actual prompt/decode positions",
            "sampling": "greedy; ignore EOS",
            "first_token": "logits included in prefill; nibblemac charges first argmax to decode, public adapters to prefill; rate = (generated_ids-1)/decode_seconds",
            "warmup": "same prompt + 4 decode steps; reset cache before timing",
            "cache": {"nibblemac": "fp32", "llama": "f16 K/V, flash attention enabled", "mlx": "bfloat16 K/V and conv state, fp32 recurrent matrices"},
            "mlx_adapter": "direct synchronized model calls; no generate_step look-ahead or log-softmax",
            "quality": "reset each 512-token chunk; score positions 256..510 predicting 257..511",
            "limitations": ["Different quantizers and weight provenance; not a pure backend comparison.",
                "Cross-runtime output equality is not required; within-runtime repeatability is checked.",
                "RSS is process high-water RSS including loading/warmup, not total unified GPU memory.",
                "Sequential runs under advisory lock, working Mac, no thermal isolation."]},
        "prompts": cases, "runs": [], "summary": [], "quality": []}
    lock = (ROOT / "build/macos-bench.lock").open("a")
    references = {}

    def run(cfg, ids, chunks=0):
        path = work / "input.i32"
        np.asarray(ids, dtype="<i4").tofile(path)
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(("NM_METAL_", "PF_FWD_", "PF_KERN_", "GGML_")):
                env.pop(key)
        if cfg.startswith("nibblemac"):
            backend = "cpu" if cfg == configs[0] else "metal"
            binary = ROOT / f"build/macos-{backend}/engine/{'pinfer_ppl' if chunks else 'pinfer_run'}"
            argv = [str(binary), "-m", str(pq), "-t", "6"]
            if chunks:
                argv += ["-f", str(path), "-c", "512", "--chunks", str(chunks)]
            else:
                argv += ["--ids", ",".join(map(str, ids)), "-n", str(args.generate), "-b", "64", "-c", "2048", "--bench"]
            env.update(NM_METAL_MIN_BYTES="67108864", NM_METAL_KERNEL="packed", NM_METAL_MLP="0", NM_METAL_RECURRENCE="0")
        elif cfg == "llama-metal":
            binary = ROOT / "build/public-llama-adapter"
            argv = [str(binary), str(gguf), str(path), str(args.generate), str(chunks)]
        else:
            binary = ROOT / "tools/public_mlx_bench.py"
            argv = [sys.executable, str(binary), str(mlx), str(path), str(args.generate), str(chunks)]
        start = time.monotonic()
        p = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
        clean = lambda s: s.replace(str(ROOT), "<repo>").replace(str(Path.home()), "<home>")
        item = {"configuration": cfg, "returncode": p.returncode, "binary_sha256": sha256(binary),
                "wall_seconds": time.monotonic() - start, "stdout": clean(p.stdout), "stderr": clean(p.stderr)}
        (record["quality"] if chunks else record["runs"]).append(item)
        if p.returncode:
            raise ValueError(f"{cfg} failed ({p.returncode}): {clean(p.stderr[-1800:])}")
        metrics = parse_metrics(clean(p.stdout), clean(p.stderr), cfg, args.generate, chunks)
        item["metrics"] = metrics
        return item

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for round_index in range(args.rounds):
            order = configs if round_index % 2 == 0 else configs[::-1]
            for case, ids in cases.items():
                for cfg in order:
                    item = run(cfg, ids)
                    item.update(round=round_index + 1, case=case)
                    generated = item["metrics"]["generated_ids"]
                    key = (case, cfg)
                    references.setdefault(key, generated)
                    item["repeatable"] = references[key] == generated
                    if not item["repeatable"]:
                        raise ValueError(f"{cfg} {case} changed output across rounds")
                    item["generated_text"] = tokenizer.decode(generated, skip_special_tokens=False)
                    print(f"round={round_index + 1} {case} {cfg}: prefill={item['metrics']['prefill_tps']:.1f} decode={item['metrics']['decode_tps']:.1f}", flush=True)
                    out.write_text(json.dumps(record, indent=2) + "\n")
                if references[(case, configs[0])] != references[(case, configs[1])]:
                    raise ValueError("nibblemac CPU/Metal token mismatch")
        for case in cases:
            for cfg in configs:
                runs = [r["metrics"] for r in record["runs"] if r["case"] == case and r["configuration"] == cfg]
                record["summary"].append({"case": case, "configuration": cfg,
                    **{m: {"median": statistics.median(r[m] for r in runs), "min": min(r[m] for r in runs),
                           "max": max(r[m] for r in runs)} for m in ("prefill_tps", "decode_tps", "peak_rss_mb")}})
        if args.quality_chunks:
            corpus = ROOT / "bench/quality/wikitext-2-raw-test.txt"
            ids = tokenizer.encode(corpus.read_text(), add_special_tokens=False).ids[:args.quality_chunks * 512]
            record["quality_corpus"] = {"text_sha256": sha256(corpus), "tokens": len(ids), "chunks": args.quality_chunks}
            for cfg in (configs[0], configs[2], configs[3]):
                item = run(cfg, ids, args.quality_chunks)
                print(f"quality {cfg}: PPL={item['metrics']['ppl']:.4f}", flush=True)
                out.write_text(json.dumps(record, indent=2) + "\n")
        record["valid"] = True
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        record["error"] = str(exc).replace(str(ROOT), "<repo>")
        print(record["error"], file=sys.stderr)
        return 1
    finally:
        record["hardware"].update(power_after=text("pmset", "-g", "batt"), thermal_after=text("pmset", "-g", "therm"))
        out.write_text(json.dumps(record, indent=2) + "\n")
        lock.close()
        print(f"Result: {out} valid={record['valid']}")


if __name__ == "__main__":
    raise SystemExit(main())
