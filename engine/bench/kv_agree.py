"""Greedy-token agreement of the fp16 / int8 KV caches against fp32 (engine/CONTEXT.md §5).

    python engine/bench/kv_agree.py bench/results/pinfer-kv0-ctx4096-n256_*.json \
        bench/results/pinfer-kv1-ctx4096-n256_*.json bench/results/pinfer-kv2-ctx4096-n256_*.json

Reads the `generated:` line each harness result file kept from `pinfer_run --bench` and prints,
per file, the decode speed / RSS and how many of the generated tokens equal the first file's,
plus the position of the first difference (after which greedy continuations diverge for good,
so the exact-match count is what matters, not the per-position rate).
"""
from __future__ import annotations

import json
import re
import sys


def load(path: str) -> tuple[list[int], dict]:
    d = json.load(open(path, encoding="utf-8"))
    m = re.search(r"^generated:\s*([\d,]*)\s*$", d["stdout"], re.M)
    toks = [int(t) for t in m.group(1).split(",") if t] if m else []
    return toks, d["results"][0] if d["results"] else {}


def main(paths: list[str]) -> int:
    ref, _ = load(paths[0])
    for p in paths:
        toks, r = load(p)
        first = next((i for i, (a, b) in enumerate(zip(toks, ref)) if a != b), None)
        same = sum(1 for a, b in zip(toks, ref) if a == b)
        print(f"{p.split('/')[-1][:44]:44} kv_dtype {r.get('kv_dtype', '?')} "
              f"decode {r.get('decode_tps', 0):.2f} tok/s prefill {r.get('prefill_tps', 0):.1f} "
              f"RSS {r.get('peak_rss_mb', 0):.0f} MB | {len(toks)} tokens, {same} equal to fp32, "
              f"first difference at {first}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
