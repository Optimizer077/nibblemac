#!/usr/bin/env python3
"""Tabulate kbench RESULT lines from harness result files as time, GOP/s and cycles.

Usage: python tools/audit/kbtable.py [--shapes c4qkv,c4gate] [--n 7,64] [--clock-ghz 1.5]
       bench/results/<file>.json ...

For every RESULT {"bench": "kbench", ...} line: us per call, achieved GOP/s
(2 x rows x cols x n / time) and cycles per block-vector (c/bv = time x clock x threads /
(rows/4 x cols/32 x n), the unit of engine/bench/KERNELS.md). Rows with `us` 0 (gate runs that
only check exactness) are skipped. Medians are taken over repeated (variant, dtype, shape, n).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def results(path: Path) -> list[dict]:
    """All kbench RESULT objects of one harness result file (stdout and parsed list)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for line in (d.get("stdout") or "").splitlines():
        if line.startswith("RESULT {"):
            try:
                r = json.loads(line[len("RESULT "):])
            except json.JSONDecodeError:
                continue
            if r.get("bench") == "kbench" and r.get("us", 0) > 0:
                out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--shapes", default="", help="comma list of shape names to keep")
    ap.add_argument("--n", default="", help="comma list of batch sizes to keep")
    ap.add_argument("--clock-ghz", type=float, default=1.5)
    a = ap.parse_args()
    keep_s = set(filter(None, a.shapes.split(",")))
    keep_n = {int(x) for x in a.n.split(",") if x}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for f in a.files:
        for r in results(f):
            if keep_s and r["shape"] not in keep_s:
                continue
            if keep_n and r["n"] not in keep_n:
                continue
            key = (f.stem[:40], r.get("variant", ""), r["dtype"], r["shape"], r["rows"], r["cols"],
                   r["threads"], r["n"])
            groups[key].append(r)
    print(f"{'file':40} {'variant':10} {'dt':4} {'shape':6} {'rows':>6}x{'cols':<5} thr {'n':>3} "
          f"{'us':>9} {'GOP/s':>6} {'c/bv':>5} reps")
    for key in sorted(groups):
        rs = groups[key]
        us = statistics.median(r["us"] for r in rs)
        _, _, _, _, rows, cols, thr, n = key
        gops = 2.0 * rows * cols * n / (us * 1e3)
        cbv = us * 1e-6 * a.clock_ghz * 1e9 * thr / ((rows / 4) * (cols / 32) * n)
        print(f"{key[0]:40} {key[1]:10} {key[2]:4} {key[3]:6} {rows:>6}x{cols:<5} {thr:>3} {n:>3} "
              f"{us:9.1f} {gops:6.1f} {cbv:5.1f} {len(rs)}")


if __name__ == "__main__":
    main()
