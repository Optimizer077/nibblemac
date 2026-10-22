#!/usr/bin/env python3
"""Write the prefix and one question suffix of a real Cerno decision row as pinfer_run id files.

Usage: python tools/audit/cerno_ids.py <decide_bench result .json> <request> <question> <out dir>

Reads the RESULT line of <request> (its row token ids and shared-prefix length) and writes
<out dir>/c_prefix.ids (the prefix) and <out dir>/c_q.ids (the question's suffix), comma
separated, as tools/audit/audit_cerno.sh expects.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    src, name, question, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    for line in (d.get("stdout") or "").splitlines():
        if not line.startswith("RESULT {"):
            continue
        r = json.loads(line[len("RESULT "):])
        if r.get("name") != name:
            continue
        n = r["timing"]["prefix_tokens"]
        row = r["rows"][question]
        out.mkdir(parents=True, exist_ok=True)
        (out / "c_prefix.ids").write_text(",".join(map(str, row[:n])) + "\n", encoding="ascii")
        (out / "c_q.ids").write_text(",".join(map(str, row[n:])) + "\n", encoding="ascii")
        print(f"{name}/{question}: prefix {n} tokens, question {len(row) - n} tokens -> {out}")
        return
    raise SystemExit(f"request {name!r} not found in {src}")


if __name__ == "__main__":
    main()
