"""Table of the prompt-economy pilot: variants x (tokens, % saved, max |dp|, same top, mean |dp|).

    python tools/pilot/prompt/report.py [bench/results/pilot/prompt]

Reads the newest <variant>_*.json per variant and tokens.json; prints Markdown.  The "identical"
bar: max |dp| <= 0.0146 and the same top answer on >= 51 of 52 (vs the llama.cpp reference);
go when >= 15% of the tokens are saved within that bar, or >= 30% within a near-identical bar
the owner may accept.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

IDENTICAL_DP, IDENTICAL_TOP = 0.0146, 51
VARIANTS = ["base", "multi", "multi_ans", "schema", "schema_q", "compact_a", "compact_b",
            "compact_c", "compact_d", "shared"]


def newest(d: Path, v: str) -> Path | None:
    """The newest full run of `v`: <v>_YYYYMMDD_HHMMSS.json (not <v>_ans_* or <v>_smoke_*)."""
    pat = re.compile(rf"^{re.escape(v)}_\d{{8}}_\d{{6}}\.json$")
    files = sorted(f for f in d.glob(f"{v}_*.json") if pat.match(f.name))
    return files[-1] if files else None


def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench/results/pilot/prompt")
    tokens = json.loads((d / "tokens.json").read_text())["totals"]
    base = tokens["base"]["tokens"]
    print("| variant | tokens (18 req.) | saved | with schema cache | max \\|dp\\| ref | "
          "same top ref | mean \\|dp\\| ref | max \\|dp\\| base | same top base | identical? |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    worst: dict[str, list] = {}
    for v in VARIANTS:
        t = tokens[v]
        saved = 100 * (base - t["tokens"]) / base
        cached = 100 * (base - t["tokens_cached"]) / base
        f = newest(d, v)
        if f is None:
            print(f"| {v} | {t['tokens']} | {saved:.1f}% | {cached:.1f}% | – | – | – | – | – | not run |")
            continue
        r = json.loads(f.read_text())
        vr, vb = r["vs_ref"], r.get("vs_base")
        ident = vr["max_dp"] <= IDENTICAL_DP and vr["same_top"] >= IDENTICAL_TOP
        rows = sorted(r["rows"], key=lambda x: -max(abs(a - b) for a, b in zip(x["p"], x["ref"])))
        worst[v] = [(x["request"], x["question"],
                     max(abs(a - b) for a, b in zip(x["p"], x["ref"]))) for x in rows[:3]]
        vb_s = (f"{vb['max_dp']:.4f} | {vb['same_top']}/{vb['rows']}" if vb else "– | –")
        print(f"| {v} | {t['tokens']} | {saved:.1f}% | {cached:.1f}% | {vr['max_dp']:.4f} | "
              f"{vr['same_top']}/{vr['rows']} | {vr['mean_dp']:.4f} | {vb_s} | "
              f"{'yes' if ident else 'no'} |")
    print()
    for v, w in worst.items():
        print(f"- {v}: largest |dp| vs ref: " + ", ".join(f"{a}/{b} {c:.4f}" for a, b, c in w))


if __name__ == "__main__":
    main()
