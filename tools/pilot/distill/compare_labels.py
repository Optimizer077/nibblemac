"""Compare two labelled row sets (``ref_llama_pc.jsonl`` or ``label_llama.py`` outputs) row by row:
max |dp| per row, mean over rows and over options, top-answer agreement.

    python compare_labels.py ref_llama_pc.jsonl labels_ref_gpu.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str], list[float]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("repeat", 0):
            continue
        for row in r["rows"]:
            out[(r["name"], row["question"])] = row["probs"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    a = ap.parse_args()
    ra, rb = load(a.a), load(a.b)
    keys = sorted(set(ra) & set(rb))
    d = [max(abs(x - y) for x, y in zip(ra[k], rb[k])) for k in keys]
    opts = [abs(x - y) for k in keys for x, y in zip(ra[k], rb[k])]
    am = sum(max(range(len(ra[k])), key=ra[k].__getitem__) == max(range(len(rb[k])), key=rb[k].__getitem__)
             for k in keys)
    worst = sorted(zip(d, keys), reverse=True)[:3]
    print(json.dumps({"rows": len(keys), "only_a": len(set(ra) - set(rb)), "only_b": len(set(rb) - set(ra)),
                      "max_dp": round(max(d), 4), "mean_max_dp_rows": round(st.mean(d), 4),
                      "mean_abs_dp_options": round(st.mean(opts), 4), "argmax_agree": am,
                      "worst": [(round(x, 4), list(k)) for x, k in worst]}, indent=1))


if __name__ == "__main__":
    main()
