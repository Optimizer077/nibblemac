"""Real Qwen model gates: CPU/Metal logits, tokens, state, replay, speculation and PPL.

Requires the ignored model from tools/setup_model_macos.py + tools/convert.py,
all three native builds, NumPy, tokenizers and Jinja2. Results preserve failures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_inference_macos import prompts, sha256, source_hashes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quality-chunks", type=int, default=2)
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--graphs", action="store_true", help="Enable single-submit MLP and persistent GPU DeltaNet for Metal")
    ap.add_argument("--kernel", choices=("simd", "row", "packed", "packed4"), default="packed")
    args = ap.parse_args()
    if args.quality_chunks < 1:
        ap.error("--quality-chunks must be positive")
    model = ROOT / "models/qwen35-0.8b-q4k.pq"
    tokenizer = Tokenizer.from_file(str(ROOT / "models/Qwen3.5-0.8B/tokenizer.json"))
    ids = prompts(tokenizer, ROOT / "models/Qwen3.5-0.8B")["prose64"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / f"bench/results/metal/real-gates-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"kind": "real_model_correctness_and_quality", "valid": False,
              "model_sha256": sha256(model), "source_sha256": source_hashes(),
              "test_sha256": sha256(Path(__file__)), "metal_graphs": args.graphs,
              "metal_kernel": args.kernel, "runs": []}
    work = ROOT / "build/real-macos-gates"
    work.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "build/macos-bench.lock").open("a")

    def run(backend, name, extra=(), tokens=None, generate=16, minimum="67108864", tool="pinfer_run"):
        binary = ROOT / f"build/macos-{backend}/engine/{tool}"
        argv = [str(binary), "-m", str(model), "-t", "4"]
        if tool == "pinfer_run":
            argv += ["-n", str(generate), "--ids", ",".join(map(str, ids if tokens is None else tokens))]
        argv += list(extra)
        env = os.environ.copy()
        for key in ("NM_METAL_MIN_BYTES", "NM_METAL_KERNEL", "PF_FWD_STATS", "PF_KERN_STATS", "NM_METAL_MLP", "NM_METAL_RECURRENCE"):
            env.pop(key, None)
        env.update(NM_METAL_MIN_BYTES=minimum, NM_METAL_KERNEL=args.kernel)
        if backend == "metal" and args.graphs:
            env.update(NM_METAL_MLP="1", NM_METAL_RECURRENCE="1")
        p = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
        item = {"name": name, "backend": backend, "binary_sha256": sha256(binary),
                "returncode": p.returncode, "stdout": p.stdout.replace(str(ROOT), "<repo>"),
                "stderr": p.stderr.replace(str(ROOT), "<repo>"), "exact": False}
        record["runs"].append(item)
        if p.returncode:
            raise ValueError(f"{name} failed with code {p.returncode}")
        if backend == "metal" and "Metal calls=" not in p.stderr:
            raise ValueError(f"{name} did not execute GPU work")
        if tool == "pinfer_run":
            match = re.search(r"^generated:\s*([\d,]+)", p.stdout, re.M)
            if not match:
                raise ValueError(f"{name} has no generated tokens")
            return item, list(map(int, match.group(1).split(",")))
        result = next(json.loads(line[7:].replace(str(ROOT), "<repo>"))
                      for line in p.stdout.splitlines() if line.startswith("RESULT "))
        item["metrics"] = result
        return item, result

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(dir=work) as directory:
            td = Path(directory)
            ref_file = td / "reference.f32"
            baseline, expected = run("cpu-baseline", "plain reference", ["--dump-last", str(ref_file)])
            baseline["exact"] = True
            reference_logits = ref_file.read_bytes()
            for backend, name, extra, minimum in [
                ("cpu", "SDOT plain", [], "67108864"),
                ("metal", "packed head offload", [], "67108864"),
                ("metal", "packed all Q4K matrices", [], "0"),
                ("cpu", "SDOT sequential prefill", ["-b", "1"], "67108864"),
                ("metal", "packed sequential prefill", ["-b", "1"], "67108864"),
            ]:
                last = td / "last.f32"
                item, got = run(backend, name, [*extra, "--dump-last", str(last)], minimum=minimum)
                item.update(tokens_exact=got == expected, logits_exact=last.read_bytes() == reference_logits)
                item["exact"] = item["tokens_exact"] and item["logits_exact"]
                if not item["exact"]:
                    raise ValueError(f"{name} differs from CPU reference")
                print(f"PASS {name}: exact tokens and logits", flush=True)
            state = td / "prefix.state"
            item, _ = run("cpu", "save CPU prefix state", ["--save-state", str(state)], tokens=ids[:32], generate=1)
            item["exact"] = True
            gpu_state = td / "gpu-prefix.state"
            item, _ = run("metal", "save GPU prefix state", ["--save-state", str(gpu_state)], tokens=ids[:32], generate=1)
            item.update(exact=state.read_bytes() == gpu_state.read_bytes(),
                        state_sha256=sha256(gpu_state), cpu_state_sha256=sha256(state))
            if not item["exact"]:
                raise ValueError("GPU-saved state bytes differ from CPU-saved state")
            print("PASS GPU/CPU prefix state bytes match exactly", flush=True)
            for backend in ("cpu", "metal"):
                for name, extra, prompt in [
                    ("restore CPU prefix state", ["--load-state", str(state)], ids[32:]),
                    ("restore GPU prefix state", ["--load-state", str(gpu_state)], ids[32:]),
                    ("checkpoint rewind", ["--rewind-at", "32"], ids),
                    ("speculative k=3", ["--spec", "3"], ids),
                    ("adaptive speculative k<=4", ["--spec", "4", "--adaptive"], ids),
                ]:
                    item, got = run(backend, name, extra, tokens=prompt)
                    item["exact"] = got == expected
                    if not item["exact"]:
                        raise ValueError(f"{backend} {name}: generated tokens differ")
                    print(f"PASS {backend} {name}: exact tokens", flush=True)
            if not args.skip_quality:
                text = ROOT / "bench/quality/wikitext-2-raw-test.txt"
                original_ids = ROOT / "bench/quality/wikitext2-qwen35.tokens.bin"
                quality_ids = np.asarray(tokenizer.encode(text.read_text(), add_special_tokens=False).ids,
                                         dtype="<i4")[:args.quality_chunks * 512]
                if len(quality_ids) != args.quality_chunks * 512:
                    raise ValueError("Insufficient quality corpus tokens")
                token_file = td / "wikitext.ids"
                quality_ids.tofile(token_file)
                existing = np.fromfile(original_ids, dtype="<i4")[:len(quality_ids)]
                record["quality"] = {"corpus": "inherited WikiText-2 raw test prefix",
                    "text_sha256": sha256(text), "tokens_sha256": sha256(token_file),
                    "tokenizer_matches_inherited_ids": np.array_equal(quality_ids, existing),
                    "chunks": args.quality_chunks, "context": 512,
                    "tokens_scored": args.quality_chunks * 255,
                    "protocol": "reset each chunk; score positions 256..510 predicting 257..511",
                    "note": "CPU/Metal agreement, not a fresh comparison against unquantized teacher weights."}
                reference_ppl = None
                for backend in ("cpu-baseline", "cpu", "metal"):
                    item, metrics = run(backend, "WikiText-2 perplexity", ["-f", str(token_file),
                        "-c", "512", "--chunks", str(args.quality_chunks)], tool="pinfer_ppl")
                    if reference_ppl is None:
                        reference_ppl = metrics["ppl"]
                    item["exact"] = metrics["ppl"] == reference_ppl
                    if not item["exact"]:
                        raise ValueError(f"{backend} perplexity differs")
                    print(f"PASS {backend}: WikiText-2 perplexity {metrics['ppl']:.4f} matches", flush=True)
        record["valid"] = True
        return 0
    except (OSError, ValueError, StopIteration, subprocess.TimeoutExpired) as exc:
        record["error"] = str(exc).replace(str(ROOT), "<repo>")
        print(record["error"], file=sys.stderr)
        return 1
    finally:
        out.write_text(json.dumps(record, indent=2) + "\n")
        lock.close()
        print(f"Result: {out} valid={record['valid']}")


if __name__ == "__main__":
    raise SystemExit(main())
