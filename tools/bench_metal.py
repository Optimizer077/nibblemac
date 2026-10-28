"""Build, correctness-gate, and record local Metal microbenchmarks (stdlib only).

No Pi access. Records use an allowlist of hardware metadata, never hostnames,
serial numbers, usernames, network configuration, or environment variables.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)


def text(*args: str) -> str:
    proc = command(*args)
    return proc.stdout.strip() if proc.returncode == 0 else "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "bench/results/metal")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Apple Silicon macOS is required (run a native, non-Rosetta Python)")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"baseline-{stamp}.json"
    sources = [Path("tools/bench_metal.py"), Path("Makefile"), *sorted(
        p.relative_to(ROOT) for p in (ROOT / "engine/metal").iterdir() if p.is_file()
    )]
    record = {
        "schema_version": 1,
        "kind": "synthetic_metal_microbenchmark_not_model_inference",
        "timestamp_utc": stamp,
        "valid": False,
        "revision": text("git", "rev-parse", "HEAD"),
        "dirty": bool(text("git", "status", "--porcelain")),
        "source_sha256": {
            str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources
        },
        "hardware": {
            "chip": text("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": text("sysctl", "-n", "hw.memsize"),
            "os_version": text("sw_vers", "-productVersion"),
            "os_build": text("sw_vers", "-buildVersion"),
            "architecture": platform.machine(),
            "compiler": text("xcrun", "clang", "--version").splitlines()[0],
            "power_state": text("pmset", "-g", "batt"),
            "thermal_state": text("pmset", "-g", "therm"),
        },
        "notes": [
            "Fast math disabled; fp32 activations; packed .pq Q4K weights.",
            "Resident synthetic buffers; no model, tokenization or tokens/sec measurement.",
            "Repeated dispatches in one submission are not single-request latency.",
            "Small weights may be cache-resident; no DRAM bandwidth claim is made.",
            "Thermal telemetry is best-effort; unrelated GPU workloads are not controlled.",
        ],
        "steps": [],
        "rounds": [],
    }
    try:
        executable = str(ROOT / "build/metal/nibblemac_metal")
        shader = str(ROOT / "build/metal/kernels.metal")
        steps = [("build", ("make", "metal")), ("correctness", (executable, "test", shader))]
        steps += [(f"benchmark-{i + 1}", (executable, "bench", shader)) for i in range(args.rounds)]
        for name, argv in steps:
            print(name, flush=True)
            proc = command(*argv)
            record["steps"].append({
                "name": name,
                "returncode": proc.returncode,
                "stdout": proc.stdout.replace(str(ROOT), "<repo>"),
                "stderr": proc.stderr.replace(str(ROOT), "<repo>"),
            })
            if proc.returncode:
                print(proc.stderr, file=sys.stderr)
                return 1
            if name.startswith("benchmark-"):
                measurements = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
                if len(measurements) != 5 or not all(m.get("correct") is True for m in measurements):
                    raise ValueError("Missing or incorrect benchmark measurements")
                record["rounds"].append(measurements)
        record["valid"] = True
        return 0
    except (OSError, ValueError) as exc:
        record["error"] = str(exc).replace(str(ROOT), "<repo>")
        print(record["error"], file=sys.stderr)
        return 1
    finally:
        output.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Result: {output} (valid={record['valid']})")


if __name__ == "__main__":
    raise SystemExit(main())
