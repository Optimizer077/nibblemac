"""Prefill "tail" test: a prefill whose caller reads only the last token's logits skips the work
of the last layer that only the other rows' outputs would need (engine/src/forward.c,
forward_chunk). Its logits and the state it leaves must be bit-identical to the full pass.

    python3 tests/test_prefill_tail.py [--bin PINFER_RUN] [--model M.pq] [--ids F]
                                       [--batch 64,16,7,1]

For each prefill batch size b (several chunks per prompt when b < prompt length) it runs
pinfer_run twice on the same prompt and generates 8 tokens:
  full   --dump-logits (every row's logits: the tail is off) --dump-last
  tail   --dump-last only (the tail is on)
and checks that both --dump-last files and the last row of --dump-logits are equal byte for
byte and that both runs generate the same tokens (the decode steps read the K / V the tail
prefill stored). Standard library only, so it also runs on the Pi against a NEON build:
  python3 tests/test_prefill_tail.py --bin ./engine/pinfer_run_x --model models/cerno-4b-q4_0.pq
Default: the host build and the bf16 Qwen3.5-0.8B model.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(binary: str, model: str, ids: str, batch: int, extra: list[str]) -> tuple[str, str]:
    cmd = [binary, "-m", model, "-t", "4", "-n", "8", "-b", str(batch), "--ids-file", ids] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"FAIL: {' '.join(cmd)} exited {p.returncode}\n{p.stderr}")
    gen = next((ln for ln in p.stdout.splitlines() if ln.startswith("generated:")), "")
    load = next((ln for ln in p.stdout.splitlines() if ln.startswith("load")), "")
    return gen, load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(ROOT / "build" / "host" / "engine" / "pinfer_run"))
    ap.add_argument("--model", default=str(ROOT / "models" / "pq" / "qwen35-0.8b-bf16.pq"))
    ap.add_argument("--ids", default=str(ROOT / "bench" / "common" / "prompt64.ids"))
    ap.add_argument("--batch", default="64,16,7,1")
    a = ap.parse_args()
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for b in [int(x) for x in a.batch.split(",")]:
            fa, fl, tl = (os.path.join(tmp, f"{k}_b{b}.bin")
                          for k in ("all", "full_last", "tail_last"))
            g_full, l_full = run(a.bin, a.model, a.ids, b, ["--dump-logits", fa, "--dump-last", fl])
            g_tail, l_tail = run(a.bin, a.model, a.ids, b, ["--dump-last", tl])
            last = Path(fl).read_bytes()
            all_rows = Path(fa).read_bytes()
            tail = Path(tl).read_bytes()
            checks = {
                "full --dump-last == last row of --dump-logits": all_rows[-len(last):] == last,
                "tail --dump-last == full --dump-last": tail == last,
                "same generated tokens": g_full == g_tail and g_full != "",
            }
            for name, good in checks.items():
                print(f"  b={b:<3} {name:48} {'ok' if good else 'MISMATCH'}")
                ok &= good
            print(f"  b={b:<3} full: {l_full}\n  b={b:<3} tail: {l_tail}")
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
