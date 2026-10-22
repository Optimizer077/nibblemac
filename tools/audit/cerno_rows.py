#!/usr/bin/env python3
"""Split Cerno decision timings (cerno-pi decide_bench result files) into per-token and fixed costs.

Usage: python tools/audit/cerno_rows.py <cerno-pi>/results/cerno-decide-reuse-a2_*.json ...

Per request: shared-prefix tokens and seconds, snapshot seconds, every row's suffix tokens and
seconds (row = restore + eval of the suffix + label readout; the first row has no restore) and
the rest of total_seconds (tokenization and Python). Over all files: least-squares fits
seconds = c + t * tokens for first rows and for later rows, so c_later - c_first estimates the
restore cost per question. Each suffix is also split into the question text, the template
("\\nOptions:", one "\\n", "(X", ")" per option, "\\nAnswer: (") and the option texts, using the
Qwen3.5 token ids of those pieces as they appear in the rows (198 "\\n", 3670 "Options", 25 ":",
15666 "Answer", 318 " (").
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

NL, OPTIONS, COLON, ANSWER, OPEN = 198, 3670, 25, 15666, 318


def requests(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for line in (d.get("stdout") or "").splitlines():
        if line.startswith("RESULT {"):
            r = json.loads(line[len("RESULT "):])
            if "timing" in r:
                out.append(r)
    return out


def split_suffix(suffix: list[int], n_opts: int) -> tuple[int, int, int]:
    """(question-text tokens, template tokens, option-text tokens) of one row suffix."""
    q = len(suffix)
    for i in range(len(suffix) - 2):
        if suffix[i:i + 3] == [NL, OPTIONS, COLON]:
            q = i
            break
    template = 3 + 4 + 3 * n_opts if q < len(suffix) else 0
    return q, template, max(0, len(suffix) - q - template)


def fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares y = c + t x -> (c, t)."""
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    t = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - t * mx, t


def main() -> None:
    first: list[tuple[int, float]] = []
    later: list[tuple[int, float]] = []
    pref: list[tuple[int, float]] = []
    snaps: list[float] = []
    rest: list[float] = []
    parts = [0, 0, 0]
    print(f"{'request':26} {'pfx tok':>7} {'pfx s':>6} {'snap s':>6} {'rows (tokens/seconds)':40} "
          f"{'other s':>7} {'total s':>7}")
    for f in map(Path, sys.argv[1:]):
        for r in requests(f):
            tm = r["timing"]
            rows = tm["rows"]
            other = tm["total_seconds"] - tm["prefix_seconds"] - tm["snapshot_seconds"] - sum(
                x["seconds"] for x in rows)
            pref.append((tm["prefix_tokens"], tm["prefix_seconds"]))
            snaps.append(tm["snapshot_seconds"])
            rest.append(other)
            for i, x in enumerate(rows):
                (first if i == 0 else later).append((x["tokens"], x["seconds"]))
                ids = r["rows"][x["question"]][tm["prefix_tokens"]:]
                for k, v in enumerate(split_suffix(ids, len(r["probabilities"][x["question"]]))):
                    parts[k] += v
            rs = " ".join(f"{x['tokens']}/{x['seconds']:.2f}" for x in rows)
            print(f"{r['name'][:26]:26} {tm['prefix_tokens']:7d} {tm['prefix_seconds']:6.2f} "
                  f"{tm['snapshot_seconds']:6.3f} {rs:40} {other:7.3f} {tm['total_seconds']:7.2f}")
    if len(later) > 2:
        c, t = fit([a for a, _ in later], [b for _, b in later])
        print(f"later rows (n={len(later)}): seconds = {c:.3f} + {t * 1e3:.1f} ms x tokens; "
              f"median tokens {st.median(a for a, _ in later)}")
    if len(first) > 2:
        c1, t1 = fit([a for a, _ in first], [b for _, b in first])
        print(f"first rows (n={len(first)}): seconds = {c1:.3f} + {t1 * 1e3:.1f} ms x tokens")
    if len(first) > 2 and len(later) > 2:
        # joint fit with one per-token slope: s = t x + c_first [first] + c_later [later]
        import numpy as np
        rows_all = first + later
        a_mat = np.array([[x, 1.0, 0.0] for x, _ in first] + [[x, 0.0, 1.0] for x, _ in later])
        y = np.array([s for _, s in rows_all])
        (tj, cf, cl), *_ = np.linalg.lstsq(a_mat, y, rcond=None)
        print(f"joint fit: {tj * 1e3:.1f} ms x tokens + {cf:.3f} s (first) / {cl:.3f} s (later); "
              f"restore+bookkeeping per later question ~ {cl - cf:.3f} s")
        n_pt = sum(a for a, _ in pref)
        s_pt = sum(b for _, b in pref)
        s_rows = sum(s for _, s in rows_all)
        n_rows = sum(x for x, _ in rows_all)
        total = s_pt + sum(snaps) + s_rows + sum(rest)
        print(f"all {len(pref)} requests: {total:.1f} s = prefix {s_pt:.1f} s ({n_pt} tok, "
              f"{s_pt / total:.0%}) + snapshots {sum(snaps):.2f} s ({sum(snaps) / total:.1%}) + rows "
              f"{s_rows:.1f} s ({n_rows} tok, {s_rows / total:.0%}; of it fixed later-row cost "
              f"{len(later) * (cl - cf):.1f} s = {len(later) * (cl - cf) / total:.1%}) + other "
              f"{sum(rest):.2f} s")
    pt = [a for a, b in pref if a > 0]
    if pt:
        rate = sum(a for a, b in pref if a > 0) / sum(b for a, b in pref if a > 0)
        print(f"prefix: {len(pt)} requests, median {st.median(pt)} tokens, {rate:.2f} tok/s overall")
    print(f"snapshot s: median {st.median(snaps):.3f}, max {max(snaps):.3f}; other s: median "
          f"{st.median(rest):.3f}")
    tot = sum(parts)
    print(f"suffix tokens: question text {parts[0]} ({parts[0] / tot:.0%}), template {parts[1]} "
          f"({parts[1] / tot:.0%}), option texts {parts[2]} ({parts[2] / tot:.0%}), total {tot}")


if __name__ == "__main__":
    main()
