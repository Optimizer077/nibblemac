"""Synthetic full-model gates for the Mac CPU/Metal backends, including speculation.

Uses the existing dimension fixtures and upstream quantizer. No real model required.
Run after make mac-cpu-baseline mac-cpu mac-infer (NumPy required).
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import test_dims
from quant import Q4_G32, Q4K_G32, Q8_G32, quantize_pack

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=("q4k", "q4", "q8"), default="q4k")
    parser.add_argument("--i8mm", action="store_true", help="Also gate the optional I8MM CPU build")
    args = parser.parse_args()
    target = {"q4k": Q4K_G32, "q4": Q4_G32, "q8": Q8_G32}[args.quant]
    results = {"kind": "synthetic_full_model_correctness", "quantization": args.quant,
                "i8mm": args.i8mm,
               "metal_mlp": os.environ.get("NM_METAL_MLP", "0"),
               "metal_recurrence": os.environ.get("NM_METAL_RECURRENCE", "0"),
                "metal_kernel": os.environ.get("NM_METAL_KERNEL", "packed"),
                "metal_split_percent": os.environ.get("NM_METAL_SPLIT_PERCENT", "0"),
               "valid": False, "runs": []}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / f"bench/results/metal/model-gates-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    work = ROOT / "build/test-macos-inference"
    work.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=work) as directory:
            td = Path(directory)
            for ci, (name, dims) in enumerate(test_dims.CASES.items()):
                tensors = test_dims.tensors(dims, np.random.default_rng(ci))
                quantized = []
                for tensor_name, dtype, shape, data in tensors:
                    if dtype == test_dims.BF16 and len(shape) == 2 and shape[0] % 4 == 0 and shape[1] % 256 == 0:
                        w = (np.frombuffer(data, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
                        data = quantize_pack(target, w)[0]
                        dtype = target
                    quantized.append((tensor_name, dtype, shape, data))
                model = td / f"{name}.pq"
                with patch.object(test_dims, "tensors", return_value=quantized):
                    test_dims.write_pq(model, dims, ci)
                ids = np.random.default_rng(100 + ci).integers(0, dims["vocab"], 40).tolist()
                reference = None
                hidden_ref = logits_ref = None
                configurations = [("cpu-baseline", 1, None, None), ("cpu", 4, None, None),
                                  ("metal", 4, "0", None), ("metal", 4, "1048576", None)]
                configurations += [(backend, 4, "0" if backend == "metal" else None, k)
                                   for backend in ("cpu", "metal") for k in (1, 2, 3)]
                if args.i8mm:
                    configurations += [("cpu-i8mm", 4, None, k) for k in (None, 1, 2, 3)]
                for backend, threads, minimum, k in configurations:
                    binary = ROOT / f"build/macos-{backend}/engine/pinfer_run"
                    logits = td / "logits.bin"
                    hidden = td / "hidden.bin"
                    argv = [str(binary), "-m", str(model), "-t", str(threads), "-n", "12",
                            "--ids", ",".join(map(str, ids))]
                    if k is None:
                        argv += ["--dump-logits", str(logits), "--dump-hidden", str(hidden)]
                    else:
                        argv += ["--spec", str(k)]
                    env = os.environ.copy()
                    if minimum is not None:
                        env["NM_METAL_MIN_BYTES"] = minimum
                    p = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
                    item = {"fixture": name, "backend": backend, "threads": threads,
                            "minimum_bytes": minimum, "spec_k": k, "returncode": p.returncode,
                            "stdout": p.stdout.replace(str(ROOT), "<repo>"),
                             "stderr": p.stderr.replace(str(ROOT), "<repo>"), "exact": False}
                    split_calls = re.search(r"split_calls=(\d+)", p.stderr)
                    item["metal_split_calls"] = int(split_calls.group(1)) if split_calls else 0
                    results["runs"].append(item)
                    generated = re.search(r"^generated:\s*([\d,]+)", p.stdout, re.M)
                    if p.returncode or not generated:
                        raise ValueError(f"Execution failure: {name} {backend} spec={k}")
                    tokens = generated.group(1)
                    if reference is None:
                        reference = tokens
                        hidden_ref, logits_ref = hidden.read_bytes(), logits.read_bytes()
                    item["tokens_exact"] = tokens == reference
                    if k is None:
                        item["hidden_exact"] = hidden.read_bytes() == hidden_ref
                        item["logits_exact"] = logits.read_bytes() == logits_ref
                        item["hidden_sha256"] = hashlib.sha256(hidden.read_bytes()).hexdigest()
                        item["logits_sha256"] = hashlib.sha256(logits.read_bytes()).hexdigest()
                    item["exact"] = item["tokens_exact"] and item.get("hidden_exact", True) and item.get("logits_exact", True)
                    if not item["exact"]:
                        raise ValueError(f"Numerical mismatch: {name} {backend} spec={k}")
                    if target == Q4K_G32 and backend == "metal" and minimum == "0" and "Metal calls=" not in p.stderr:
                        raise ValueError("Metal fixture did not execute a GPU call")
                    if (target == Q4K_G32 and backend == "metal" and minimum == "0" and
                            int(results["metal_split_percent"]) > 0 and dims["H"] % 256 == 0 and
                            dims["vocab"] * int(results["metal_split_percent"]) // 100 >= 32 and
                            not item["metal_split_calls"]):
                        raise ValueError("Eligible Metal fixture did not execute a cooperative split")
                    print(f"PASS {name} {backend} min={minimum} spec={k}: exact tokens" +
                          (", hidden states, logits" if k is None else ""), flush=True)
        results["valid"] = True
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        results["error"] = str(exc).replace(str(ROOT), "<repo>")
        print(results["error"])
        return 1
    finally:
        output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Result: {output} valid={results['valid']}")


if __name__ == "__main__":
    raise SystemExit(main())
