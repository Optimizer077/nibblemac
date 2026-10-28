"""Record real-model CPU/Metal inference, with exact tokens/logits correctness gates.

Example: .venv/bin/python tools/bench_inference_macos.py --config cpu:6
         --config metal:6:67108864 --rounds 3 --cases prose64,chat,long512
Configurations: cpu:THREADS, cpu-baseline:THREADS or
metal:THREADS:MIN_WEIGHT_BYTES[:simd|row|packed|packed4[:none|mlp|rec|graph]].
Append -specK to the backend for exact speculative decoding, e.g. cpu-spec3:4.
Run make mac-cpu mac-infer first. NumPy, tokenizers and Jinja2 are required.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parent.parent
PROSE = ("Apple Silicon combines a CPU and a GPU with unified memory. Efficient inference must "
         "minimize memory traffic, preserve numerical correctness, and avoid unnecessary synchronization. "
         "A useful benchmark reports prompt processing and token generation separately. ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text(*args: str) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else "unavailable"


def source_hashes() -> dict[str, str]:
    paths = [Path("Makefile"), Path("tools/convert.py"), Path("tools/quant.py"),
             Path("tools/bench_inference_macos.py"), Path("engine/include/pinfer.h")]
    for directory in ("engine/src", "engine/metal", "engine/macos"):
        paths += [p.relative_to(ROOT) for p in (ROOT / directory).iterdir()
                  if p.is_file() and (p.suffix in (".c", ".h", ".m", ".metal") or p.name == "Makefile")]
    return {str(p): sha256(ROOT / p) for p in sorted(paths)}


def config(value: str) -> dict:
    parts = value.split(":")
    prefix = re.fullmatch(r"(cpu|cpu-baseline|metal)(?:-spec([0-4]))?", parts[0])
    if len(parts) not in (2, 3, 4, 5) or prefix is None:
        raise argparse.ArgumentTypeError("Use cpu:THREADS or metal:THREADS:MIN_BYTES")
    backend, spec = prefix.groups()
    try:
        threads = int(parts[1])
        minimum = int(parts[2]) if len(parts) >= 3 else None
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Configuration needs integer values") from exc
    if threads < 1 or threads > 8 or (backend == "metal" and (minimum is None or minimum < 0)):
        raise argparse.ArgumentTypeError("Threads must be 1..8; Metal needs nonnegative MIN_BYTES")
    if backend.startswith("cpu") and minimum is not None:
        raise argparse.ArgumentTypeError("CPU configuration has no Metal threshold")
    kernel = parts[3] if len(parts) >= 4 else "packed"
    if kernel not in ("simd", "row", "packed", "packed4"):
        raise argparse.ArgumentTypeError("Metal kernel must be simd, row, packed or packed4")
    graph = parts[4] if len(parts) == 5 else "none"
    if graph not in ("none", "mlp", "rec", "graph"):
        raise argparse.ArgumentTypeError("Graph mode must be none, mlp, rec or graph")
    return {"name": value, "backend": backend, "threads": threads, "metal_min_bytes": minimum,
            "metal_kernel": kernel if backend == "metal" else None,
            "metal_mlp": graph in ("mlp", "graph"), "metal_recurrence": graph in ("rec", "graph"),
            "spec_k": int(spec) if spec is not None else None}


def prompts(tokenizer: Tokenizer, directory: Path) -> dict[str, list[int]]:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def raise_error(message):
        raise ValueError(message)
    env.globals["raise_exception"] = raise_error
    template = env.from_string((directory / "chat_template.jinja").read_text())
    chat = template.render(messages=[{"role": "user", "content":
        "Explain how unified memory helps machine learning on Apple Silicon. "
        "Write a detailed answer of about 200 words."}], add_generation_prompt=True,
        enable_thinking=False, tools=None)
    long_ids = tokenizer.encode(PROSE * 80, add_special_tokens=False).ids
    return {"prose64": long_ids[:64], "chat": tokenizer.encode(chat, add_special_tokens=False).ids,
            "long512": long_ids[:512]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "models/qwen35-0.8b-q4k.pq")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "models/Qwen3.5-0.8B")
    ap.add_argument("--config", type=config, action="append")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--generate", type=int, default=64)
    ap.add_argument("--cases", default="prose64,chat,long512")
    ap.add_argument("--label", default="inference")
    ap.add_argument("--profile", action="store_true", help="Instrument phases; not comparable to unprofiled timings")
    ap.add_argument("--adaptive", action="store_true", help="Adapt draft length for speculative configurations")
    args = ap.parse_args()
    configs = args.config or [config("cpu:4"), config("metal:4:67108864")]
    if not configs[0]["backend"].startswith("cpu") or configs[0]["spec_k"] is not None:
        ap.error("First configuration must be a CPU correctness reference")
    if args.rounds < 1 or args.generate < 2:
        ap.error("Need rounds >= 1 and generate >= 2")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        ap.error("Native Apple Silicon macOS is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        ap.error("Label must contain only letters, digits, - or _")
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    cases = prompts(tokenizer, args.tokenizer)
    selected = args.cases.split(",")
    if not selected or any(c not in cases for c in selected):
        ap.error("Cases must be prose64,chat,long512")
    out = ROOT / "bench/results/metal"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    result_path = out / f"{args.label}-{stamp}.json"
    model_source = args.tokenizer / "source.json"
    record = {
        "schema_version": 1, "kind": "real_model_inference", "timestamp_utc": stamp,
        "valid": False, "profiled": args.profile,
        "revision": text("git", "rev-parse", "HEAD"), "dirty": bool(text("git", "status", "--porcelain")),
        "source_sha256": source_hashes(),
        "model": {"file": args.model.name, "bytes": args.model.stat().st_size,
                  "sha256": sha256(args.model), "quantization": "upstream PQ_Q4K_G32 weights / int8 activations",
                  "source": json.loads(model_source.read_text()) if model_source.exists() else None},
        "hardware": {"chip": text("sysctl", "-n", "machdep.cpu.brand_string"),
                     "memory_bytes": text("sysctl", "-n", "hw.memsize"),
                     "os_version": text("sw_vers", "-productVersion"), "os_build": text("sw_vers", "-buildVersion"),
                     "compiler": text("xcrun", "clang", "--version").splitlines()[0],
                     "power_before": text("pmset", "-g", "batt"), "thermal_before": text("pmset", "-g", "therm")},
        "protocol": {"rounds": args.rounds, "generated_ids": args.generate,
                     "spec_adaptive": args.adaptive,
                     "timed_decode_steps": args.generate - 1, "kv_dtype": "fp32", "sampling": "greedy",
                     "prefill_batch": 64, "max_context": 2048,
                     "warmup": "same prompt + up to 4 decode steps, then reset; excluded from timing",
                     "eos": "ignored for fixed-length throughput; output after first EOS is not a chat answer",
                     "decode_formula": "(generated_ids - 1) / decode_seconds; first token belongs to prefill",
                     "prefill_formula": "prompt_tokens / prefill_seconds; includes final vocabulary projection",
                     "notes": ["One process/configuration at a time, with an advisory local benchmark lock.",
                               "Round order alternates to reduce fixed-order bias; no thermal/power isolation.",
                               "Tokenizer, model load, shader compilation and warmup excluded from tokens/sec.",
                               "Full-model execution is hybrid CPU/Metal, not fully GPU-resident.",
                               "Exactness is against the existing quantized CPU engine, not an fp32 teacher model."]},
        "prompts": {c: {"ids": cases[c], "tokens": len(cases[c]),
                        "text": tokenizer.decode(cases[c], skip_special_tokens=False)} for c in selected},
        "configurations": configs, "runs": [], "summary": [],
    }
    references = {}
    workdir = ROOT / "build/bench-inference"
    workdir.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "build/macos-bench.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for round_index in range(args.rounds):
            order = configs if round_index % 2 == 0 else list(reversed(configs))
            for case in selected:
                for cfg in order:
                    binary = ROOT / f"build/macos-{cfg['backend']}/engine/pinfer_run"
                    item = {"round": round_index + 1, "case": case, "configuration": cfg["name"],
                            "binary_sha256": sha256(binary), "correct": False}
                    record["runs"].append(item)
                    with tempfile.TemporaryDirectory(dir=workdir) as td:
                        logits_file = Path(td) / "last.f32"
                        argv = [str(binary), "-m", str(args.model), "-t", str(cfg["threads"]),
                                "-n", str(args.generate), "-b", "64", "-c", "2048", "--bench",
                                "--ids", ",".join(map(str, cases[case]))]
                        if cfg["spec_k"] is None:
                            argv += ["--dump-last", str(logits_file)]
                        else:
                            argv += ["--spec", str(cfg["spec_k"])]
                            if args.adaptive:
                                argv += ["--adaptive"]
                        env = os.environ.copy()
                        for key in ("PF_FWD_STATS", "PF_KERN_STATS", "NM_METAL_MIN_BYTES", "NM_METAL_SHADER", "NM_METAL_KERNEL",
                                    "NM_METAL_MLP", "NM_METAL_RECURRENCE"):
                            env.pop(key, None)
                        if cfg["backend"] == "metal":
                            env["NM_METAL_MIN_BYTES"] = str(cfg["metal_min_bytes"])
                            env["NM_METAL_KERNEL"] = cfg["metal_kernel"]
                            env["NM_METAL_MLP"] = str(int(cfg["metal_mlp"]))
                            env["NM_METAL_RECURRENCE"] = str(int(cfg["metal_recurrence"]))
                        if args.profile:
                            env.update(PF_FWD_STATS="1", PF_KERN_STATS="1")
                        start = time.monotonic()
                        p = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True,
                                           check=False, timeout=180)
                        item.update(wall_seconds=time.monotonic() - start, returncode=p.returncode,
                                    stdout=p.stdout.replace(str(ROOT), "<repo>"),
                                    stderr=p.stderr.replace(str(ROOT), "<repo>"))
                        if p.returncode:
                            raise RuntimeError(f"{case} {cfg['name']} failed with exit code {p.returncode}")
                        result_lines = [line[7:] for line in p.stdout.splitlines() if line.startswith("RESULT ")]
                        if len(result_lines) != 1:
                            raise ValueError("Missing or duplicated RESULT line")
                        metrics = json.loads(result_lines[0])
                        generated = re.search(r"^generated:\s*([\d,]+)", p.stdout, re.M)
                        if not generated:
                            raise ValueError("Missing generated IDs")
                        ids = list(map(int, generated.group(1).split(",")))
                        logits = np.fromfile(logits_file, dtype="<f4") if cfg["spec_k"] is None else None
                        if len(ids) != args.generate or (logits is not None and (not len(logits) or not np.isfinite(logits).all())):
                            raise ValueError("Invalid token count or non-finite logits")
                        if cfg["backend"] == "metal" and "Metal calls=" not in p.stderr:
                            raise ValueError("Metal configuration performed no GPU calls")
                        for feature, pattern in (("metal_mlp", r"mlp_calls=([1-9]\d*)"),
                                                 ("metal_recurrence", r"lin_calls=([1-9]\d*)")):
                            if cfg[feature] and not re.search(pattern, p.stderr):
                                raise ValueError(f"Requested {feature} performed no GPU work")
                        if case not in references:
                            if not cfg["backend"].startswith("cpu"):
                                raise ValueError("Missing CPU reference")
                            references[case] = (ids, logits.copy())
                        ref_ids, ref_logits = references[case]
                        item.update(metrics=metrics, generated_ids=ids,
                                    generated_text=tokenizer.decode(ids, skip_special_tokens=False),
                                    prefill_logits_sha256=sha256(logits_file) if logits is not None else None,
                                    tokens_exact=ids == ref_ids,
                                    logits_exact=np.array_equal(logits, ref_logits) if logits is not None else None,
                                    max_abs_logit_error=float(np.max(np.abs(logits - ref_logits))) if logits is not None else None)
                        item["correct"] = item["tokens_exact"] and item["logits_exact"] is not False
                        if not item["correct"]:
                            raise ValueError(f"CPU/Metal output mismatch for {case} {cfg['name']}")
                        print(f"round={round_index + 1} {case} {cfg['name']}: "
                              f"prefill={metrics['prefill_tps']:.2f} decode={metrics['decode_tps']:.2f} tok/s "
                              f"RSS={metrics['peak_rss_mb']:.1f} MiB exact", flush=True)
                    result_path.write_text(json.dumps(record, indent=2) + "\n")
        for case in selected:
            for cfg in configs:
                runs = [r for r in record["runs"] if r["case"] == case and r["configuration"] == cfg["name"]]
                summary = {"case": case, "configuration": cfg["name"], "rounds": len(runs)}
                for metric in ("prefill_tps", "decode_tps", "peak_rss_mb"):
                    values = [r["metrics"][metric] for r in runs]
                    summary[metric] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
                record["summary"].append(summary)
        record["valid"] = True
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        record["error"] = str(exc).replace(str(ROOT), "<repo>")
        print(record["error"], file=sys.stderr)
        return 1
    finally:
        record["hardware"].update(power_after=text("pmset", "-g", "batt"), thermal_after=text("pmset", "-g", "therm"))
        result_path.write_text(json.dumps(record, indent=2) + "\n")
        lock.close()
        print(f"Result: {result_path} valid={record['valid']}")


if __name__ == "__main__":
    raise SystemExit(main())
