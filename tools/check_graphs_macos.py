"""Record exact GPU graph tests and preserve failures (no model/Python packages needed)."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / f"bench/results/metal/graph-kernel-gates-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "build").mkdir(exist_ok=True)
    record = {"kind": "exact_gpu_graph_gates", "valid": False, "runs": [], "source_sha256": {}}
    for directory in ("engine/metal", "engine/src"):
        for p in (ROOT / directory).iterdir():
            if p.is_file():
                record["source_sha256"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    lock = (ROOT / "build/macos-bench.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for variant in ("simd", "row", "packed", "packed4"):
            p = subprocess.run(["make", "-s", "test-a8", "test-graphs"], cwd=ROOT,
                               env={**os.environ, "NM_METAL_KERNEL": variant},
                               capture_output=True, text=True, timeout=120)
            record["runs"].append({"variant": variant, "returncode": p.returncode,
                "stdout": p.stdout.replace(str(ROOT), "<repo>"),
                "stderr": p.stderr.replace(str(ROOT), "<repo>")})
            print(variant, "PASS" if p.returncode == 0 else "FAIL", flush=True)
            if p.returncode:
                return 1
        record["valid"] = True
        return 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["error"] = str(exc).replace(str(ROOT), "<repo>")
        return 1
    finally:
        lock.close()
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Result: {out} valid={record['valid']}")


if __name__ == "__main__":
    raise SystemExit(main())
