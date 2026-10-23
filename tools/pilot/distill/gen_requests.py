"""Generate the synthetic training requests for the distilled decider pilot.

    python gen_requests.py --out /root/distill/requests.json --per-domain 25 --seed 1

18 domains (synth_a.py, synth_b.py) x ``--per-domain`` requests, each with 1-5 typed questions,
in the shape of cerno-pi's ``requests/correctness.json``. Deterministic for a seed. The output
is not committed (a 10-request sample is: ``sample_requests.json``).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_a import GENERATORS_A
from synth_b import GENERATORS_B


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-domain", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sample", type=Path, help="also write the first 10 requests here")
    a = ap.parse_args()
    r = random.Random(a.seed)
    reqs = []
    for gen in GENERATORS_A + GENERATORS_B:
        for i in range(a.per_domain):
            reqs.append(gen(r, i))
    r.shuffle(reqs)
    seen: set[str] = set()
    uniq = []
    for q in reqs:                        # drop exact duplicate states (templates can collide)
        key = json.dumps(q["state"], sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(q)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(uniq, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.sample:
        a.sample.write_text(json.dumps(uniq[:10], ensure_ascii=False, indent=1), encoding="utf-8")
    nq = sum(len(q["questions"]) for q in uniq)
    types = Counter(v["type"] for q in uniq for v in q["questions"].values())
    print(f"{len(reqs)} generated, {len(uniq)} unique requests, {nq} questions, types {dict(types)} -> {a.out}")


if __name__ == "__main__":
    main()
