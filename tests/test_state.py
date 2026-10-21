"""State save / restore test (pinfer_state_save / load, pinfer_generate_greedy_cont).

    /root/venv-ref/bin/python tests/test_state.py [--bin PATH] [--model PATH]

For every prompt and decoding mode (plain, --spec 3, --spec 3 --adaptive), three runs of the
host build of pinfer_run (through WSL when started from Windows):
  A. the full prompt, N greedy tokens;
  B. the first part of the prompt, --save-state after the prefill;
  C. --load-state of B's file, then the rest of the prompt appended, N greedy tokens.
C must generate exactly A's tokens: the saved state (conv + recurrent states, KV rows, MTP
rows) continued through the batched or the speculative path is bit-identical to never
having stopped. The split points include one inside a 64-token prefill chunk.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "ref" / "golden" / "qwen35-0.8b"
MODES = [("plain", []), ("spec3", ["--spec", "3", "--draft-vocab", "32768"]),
         ("adaptive", ["--spec", "3", "--draft-vocab", "32768", "--adaptive"])]


def wsl_path(p: Path) -> str:
    s = str(p.resolve())
    if os.name != "nt":
        return s
    return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")


def run(binary: Path, model: Path, ids: list[int], n_gen: int, extra: list[str]) -> tuple[list[int], str]:
    args = [wsl_path(binary), "-m", wsl_path(model), "-n", str(n_gen), "-c", "256",
            "--ids", ",".join(map(str, ids))] + extra
    if os.name == "nt":
        args = ["wsl.exe", "-d", "Ubuntu", "--"] + args
    proc = subprocess.run(args, capture_output=True, text=True,
                          env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    if proc.returncode != 0:
        raise RuntimeError(f"engine failed ({proc.returncode}) {extra}:\n{proc.stdout}\n{proc.stderr}")
    m = re.search(r"^generated:\s*([\d,]*)\s*$", proc.stdout, re.M)
    gen = [int(t) for t in m.group(1).split(",") if t] if m else []
    st = re.search(r"^state: (.*)$", proc.stdout, re.M)
    return gen, st.group(1) if st else ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin", type=Path, default=ROOT / "build" / "host" / "engine" / "pinfer_run")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "pq" / "qwen35-0.8b-bf16.pq")
    p.add_argument("--n-gen", type=int, default=24)
    p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()

    golden = np.load(GOLDEN / "input_ids.npy").astype(np.int32).tolist()
    bench = json.loads((ROOT / "bench" / "common" / "prompt64.json").read_text("utf-8"))["ids"]
    prompts = [("golden", golden, 4), ("bench64", bench, 40), ("long74", bench + golden, 64)]
    jobs = []
    for name, ids, split in prompts:
        for mode, extra in MODES:
            jobs.append((name, ids, split, mode, extra))
    print(f"{len(jobs)} prompt x mode cases, {a.jobs} in parallel")

    def case(j: tuple) -> tuple[bool, str]:
        name, ids, split, mode, extra = j
        with tempfile.TemporaryDirectory() as td:
            st = Path(td) / "state.bin"
            full, _ = run(a.bin, a.model, ids, a.n_gen, extra)
            _, info_b = run(a.bin, a.model, ids[:split], 1, extra + ["--save-state", wsl_path(st)])
            size = st.stat().st_size
            cont, info_c = run(a.bin, a.model, ids[split:], a.n_gen, extra + ["--load-state", wsl_path(st)])
        ok = len(full) == a.n_gen and cont == full
        return ok, (f"   {name:>8} split {split:>2} {mode:>8}: state {size} B, {info_b.strip()} | "
                    f"{'ok' if ok else 'DIFF'}" + ("" if ok else f"\n      A {full}\n      C {cont}"))

    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(case, jobs))
    for _, line in results:
        print(line)
    ok = all(r for r, _ in results)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
