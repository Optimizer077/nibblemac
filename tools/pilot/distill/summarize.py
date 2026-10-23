"""Print a Markdown table from ``eval_ref.py`` result files (one row per file).

    python summarize.py /root/distill/results/*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def type_map(requests: Path) -> dict[tuple[str, str], str]:
    """(request name, question key) -> "choice" | "score" | "noul" from a request file."""
    out = {}
    for r in json.loads(requests.read_text(encoding="utf-8")):
        for k, q in r["questions"].items():
            t = q.get("type", "choice")
            out[(r["name"], k)] = "noul" if t == "bool" else t
    return out


def main() -> None:
    args = sys.argv[1:]
    tmap: dict[tuple[str, str], str] = {}
    if "--requests" in args:
        i = args.index("--requests")
        tmap = type_map(Path(args[i + 1]))
        del args[i:i + 2]
    cols = ("choice", "score", "noul") if tmap else ("choice/score", "noul")
    print("| model | top answer | mean \\|Δp\\| (options) | mean row max \\|Δp\\| | max \\|Δp\\| | "
          + " | ".join(f"{c} (agree, mean row max \\|Δp\\|)" for c in cols) + " | s |")
    print("|---|---|---|---|---|" + "---|" * len(cols) + "---|")
    for p in args:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        s, rows = d["summary"], d["rows"]
        cells = []
        for c in cols:
            if tmap:
                sub = [r for r in rows if tmap.get((r["name"], r["question"])) == c]
            else:
                sub = [r for r in rows if (("noul" if len(r["labels"]) == 2 else "choice/score") == c)]
            if sub:
                cells.append(f"{sum(r['agree'] for r in sub)}/{len(sub)}, {sum(r['max_dp'] for r in sub) / len(sub):.3f}")
            else:
                cells.append("-")
        print(f"| {s['tag'] or Path(p).stem} | {s['agree']}/{s['rows']} ({s['agree_pct']}%) | {s['mean_abs_dp_options']:.4f} | "
              f"{s['mean_max_dp_rows']:.4f} | {s['max_dp']:.4f} | " + " | ".join(cells) + f" | {s['seconds']} |")


if __name__ == "__main__":
    main()
