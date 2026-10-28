"""Actual-device check: synchronized adapter tokens vs public MLX-LM generate_step.

Optional: needs requirements-public-macos.txt and the locally converted model.
"""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step, wired_limit
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_inference_macos import prompts, sha256


def main():
    model_dir = ROOT / "models/public-baselines/mlx-q4-g32"
    tokenizer_dir = ROOT / "models/Qwen3.5-0.8B"
    cases = prompts(Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json")), tokenizer_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / f"bench/results/metal/public-mlx-api-gates-{stamp}.json"
    adapter = ROOT / "tools/public_mlx_bench.py"
    record = {"kind": "mlx_adapter_vs_public_generate_step", "valid": False,
              "adapter_sha256": sha256(adapter), "test_sha256": sha256(Path(__file__)), "runs": []}
    lock = (ROOT / "build/macos-bench.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as td:
            for name, ids in cases.items():
                token_file = Path(td) / "ids.i32"
                np.asarray(ids, dtype="<i4").tofile(token_file)
                p = subprocess.run([sys.executable, str(adapter), str(model_dir), str(token_file), "16", "0"],
                                   capture_output=True, text=True, check=True, timeout=120)
                result = next(json.loads(s[7:]) for s in p.stdout.splitlines() if s.startswith("RESULT "))
                model, _ = load(str(model_dir))
                with wired_limit(model):
                    expected = [token for token, _ in generate_step(mx.array(ids), model, max_tokens=16,
                                                                   prefill_step_size=64)]
                    mx.synchronize()
                actual = result["generated_ids"]
                item = {"case": name, "native_ids": expected, "adapter_ids": actual, "exact": expected == actual}
                record["runs"].append(item)
                if not item["exact"]:
                    raise ValueError(f"{name}: adapter differs from MLX-LM generate_step")
                print(f"PASS {name}: 16 exact MLX-LM native API tokens", flush=True)
                del model
        record["valid"] = True
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record["error"] = str(exc).replace(str(ROOT), "<repo>")
        return 1
    finally:
        lock.close()
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Result: {out} valid={record['valid']}")


if __name__ == "__main__":
    raise SystemExit(main())
