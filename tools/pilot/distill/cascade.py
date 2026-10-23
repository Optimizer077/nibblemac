"""Cascade analysis from an ``eval_ref.py`` result: the student answers when its top probability is
at least a threshold, the 4B answers the rest. Reports coverage, agreement among the covered rows,
the overall agreement (the 4B counts as right on its own rows) and the expected Pi time per question.

Pi prefill rates (tok/s): 0.8B 43, 2B 15, 4B 6.8 (`engine/PILOTS.md`); a question row costs its
tokens at the student's rate plus, when escalated, the same tokens at the 4B's rate.

    python cascade.py /root/distill/results/student-0.8b.json --student-rate 43
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", type=Path)
    ap.add_argument("--student-rate", type=float, default=43.0)
    ap.add_argument("--teacher-rate", type=float, default=6.8)
    a = ap.parse_args()
    rows = json.loads(a.result.read_text(encoding="utf-8"))["rows"]
    n = len(rows)
    tok = sum(r["tokens"] for r in rows) / n
    print(f"rows {n}, mean tokens per row {tok:.0f}; 4B alone: {tok / a.teacher_rate:.1f} s per question, "
          f"student alone: {tok / a.student_rate:.1f} s")
    print("| threshold | covered | agree among covered | overall agree | mean \\|Δp\\| covered | s per question |")
    print("|---|---|---|---|---|---|")
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        cov = [r for r in rows if max(r["p_student"]) >= thr]
        agree_cov = sum(r["agree"] for r in cov)
        overall = agree_cov + (n - len(cov))
        dp = sum(r["max_dp"] for r in cov) / len(cov) if cov else 0.0
        secs = tok / a.student_rate + (1 - len(cov) / n) * tok / a.teacher_rate
        print(f"| {thr:.2f} | {len(cov)}/{n} | {agree_cov}/{len(cov)} | {overall}/{n} ({100 * overall / n:.1f}%) | "
              f"{dp:.3f} | {secs:.1f} |")


if __name__ == "__main__":
    main()
