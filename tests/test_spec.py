"""MTP speculative decoding test: exactness + MTP head vs the PyTorch reference.

    uv run --no-project --with numpy python tests/test_spec.py [--bin PATH] [--model PATH]
    (or in WSL: /root/venv-ref/bin/python tests/test_spec.py)

Runs the host build of pinfer_run (through WSL when started from Windows) and checks:
  1. MTP head: `--dump-mtp` on the golden prompt vs ref/golden/qwen35-0.8b-mtp
     (ref/mtp_reference.py): 10 prompt rows + 3 chained draft rows.
     rel max error <= 1e-3 per row (max|ours-gold| / max|gold|) and identical argmax on every
     row (so the chained drafts equal draft_ids.npy). Observed 2026-10-21 (matvec_ref):
     <= 4e-6; a wrong concat order, norm, position or hidden gives O(1) errors.
  2. Exactness: for every prompt (golden, the 6 other varied prompts of prompts.json, the
     64-token bench prompt, a 74-token prompt = two prefill chunks, the 203-token code-edit
     chat prompt whose answer copies the prompt), `--spec k` for k = 0..4, k = 3 with
     --draft-vocab 32768, k = 3 --adaptive (engine picks the chain length per round), and the
     prompt-lookup drafts of engine/DRAFT.md (--ngram: with k = 3, alone with k = 0, adaptive,
     the widest pass k = 8, and a prefill chunk smaller than the pass) must generate exactly
     the tokens of the plain greedy loop (no --spec); on the golden prompt both must equal
     greedy_ids.npy. Any rollback error (conv state, recurrent state, KV position) changes
     the main model's logits and shows up here.
Also prints the acceptance statistics per k (informational, not a pass criterion).
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
GOLDEN_MTP = ROOT / "ref" / "golden" / "qwen35-0.8b-mtp"
MTP_TOL = 1e-3
KS = [0, 1, 2, 3, 4]
DRAFT_VOCAB = 32768


def wsl_path(p: Path) -> str:
    """E:\\a\\b -> /mnt/e/a/b (only needed when this script runs on Windows)."""
    s = str(p.resolve())
    if os.name != "nt":
        return s
    return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")


def run(binary: Path, model: Path, ids: list[int], n_gen: int, extra: list[str]
        ) -> tuple[list[int], str]:
    args = [wsl_path(binary), "-m", wsl_path(model), "-n", str(n_gen),
            "--ids", ",".join(map(str, ids))] + extra
    if os.name == "nt":
        args = ["wsl.exe", "-d", "Ubuntu", "--"] + args
    proc = subprocess.run(args, capture_output=True, text=True,
                          env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    if proc.returncode != 0:
        raise RuntimeError(f"engine failed ({proc.returncode}) {extra}:\n{proc.stdout}\n"
                           f"{proc.stderr}")
    m = re.search(r"^generated:\s*([\d,]*)\s*$", proc.stdout, re.M)
    gen = [int(t) for t in m.group(1).split(",") if t] if m else []
    spec = [ln for ln in proc.stdout.splitlines() if ln.startswith("spec ")]
    return gen, spec[0] if spec else ""


def check_mtp(binary: Path, model: Path) -> bool:
    print("1. MTP head vs ref/golden/qwen35-0.8b-mtp (golden prompt)")
    ids = np.load(GOLDEN / "input_ids.npy").astype(np.int32).tolist()
    gold = np.concatenate([np.load(GOLDEN_MTP / "mtp_logits.npy"),
                           np.load(GOLDEN_MTP / "chain_logits.npy")])
    drafts = np.load(GOLDEN_MTP / "draft_ids.npy").tolist()
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "mtp.bin"
        run(binary, model, ids, 1, ["--dump-mtp", wsl_path(f)])
        ours = np.fromfile(f, dtype=np.float32).reshape(gold.shape)
    ok, T = True, len(ids)
    for r in range(gold.shape[0]):
        err = float(np.max(np.abs(ours[r].astype(np.float64) - gold[r])))
        rel = err / float(np.max(np.abs(gold[r])))
        a, b = int(ours[r].argmax()), int(gold[r].argmax())
        row_ok = rel <= MTP_TOL and a == b
        ok &= row_ok
        kind = f"row {r}" if r < T else f"chain {r - T + 2}"
        print(f"   {kind:>8}: rel max err {rel:.3e}  argmax ours {a:>6} gold {b:>6}"
              f"{'' if row_ok else '  <-- FAIL'}")
    chain = [int(ours[T - 1].argmax())] + [int(ours[r].argmax()) for r in range(T, len(ours))]
    ok &= chain == drafts
    print(f"   chained drafts ours {chain} gold {drafts} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin", type=Path,
                   default=ROOT / "build" / "host" / "engine" / "pinfer_run")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "pq" / "qwen35-0.8b-bf16.pq")
    p.add_argument("--n-gen", type=int, default=40, help="tokens per run (golden: 32)")
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()

    res1 = check_mtp(a.bin, a.model)

    golden_ids = np.load(GOLDEN / "input_ids.npy").astype(np.int32).tolist()
    golden_greedy = np.load(GOLDEN / "greedy_ids.npy").tolist()
    prompts = [("golden", golden_ids, len(golden_greedy))]
    for i, pr in enumerate(json.loads((GOLDEN_MTP / "prompts.json").read_text("utf-8"))):
        if pr["ids"] != golden_ids:
            prompts.append((f"varied{i}", pr["ids"], a.n_gen))
    bench = json.loads((ROOT / "bench" / "common" / "prompt64.json").read_text("utf-8"))
    prompts.append(("bench64", bench["ids"], a.n_gen))
    # > PF_MAX_BATCH (64) prompt ids: the MTP prefill runs over two chunks
    prompts.append(("long74", bench["ids"] + golden_ids, a.n_gen))
    # a chat answer that copies its prompt (n-gram drafts are taken most of the time)
    edit = [int(t) for t in re.findall(r"\d+", (ROOT / "bench" / "common" / "chat_edit.ids")
                                       .read_text("utf-8"))]
    prompts.append(("edit203", edit, a.n_gen))
    dv = ["--draft-vocab", str(DRAFT_VOCAB)]
    configs = [("plain", [])] + [(f"k{k}", ["--spec", str(k)]) for k in KS] + \
              [(f"k3v{DRAFT_VOCAB}", ["--spec", "3"] + dv),
               ("k3adapt", ["--spec", "3", "--adaptive"] + dv),
               ("k3b2", ["--spec", "3", "-b", "2"]),   # prefill chunk smaller than k+1
               ("k3g8", ["--spec", "3", "--ngram", "8"] + dv),
               ("g8", ["--spec", "0", "--ngram", "8"]),   # n-gram drafts only, no MTP head
               ("a3g8", ["--spec", "3", "--adaptive", "--ngram", "8"] + dv),
               ("k8g8", ["--spec", "8", "--ngram", "8"]),   # widest pass, PINFER_SPEC_MAX_K
               ("k2g5b2", ["--spec", "2", "--ngram", "5", "-b", "2"])]
    jobs = [(name, ids, n, cname, extra) for name, ids, n in prompts for cname, extra in configs]
    print(f"\n2. exactness: {len(prompts)} prompts x {len(configs)} configs "
          f"({len(jobs)} runs, {a.jobs} in parallel)")
    with ThreadPoolExecutor(a.jobs) as ex:
        outs = list(ex.map(lambda j: run(a.bin, a.model, j[1], j[2], j[4]), jobs))
    res = {(j[0], j[3]): o for j, o in zip(jobs, outs)}

    res2 = True
    stats: dict[str, list[int]] = {}
    for name, ids, n in prompts:
        ref = res[(name, "plain")][0]
        line = []
        ok_p = len(ref) == n and (name != "golden" or ref == golden_greedy)
        for cname, _ in configs[1:]:
            gen, spec = res[(name, cname)]
            same = gen == ref
            ok_p &= same
            line.append(f"{cname}:{'ok' if same else 'DIFF'}")
            m = re.search(r"rounds (\d+) drafted (\d+) accepted (\d+)", spec)
            g = re.search(r"ngram rounds (\d+)", spec)
            if m:
                s = stats.setdefault(cname, [0, 0, 0, 0])
                for i in range(3):
                    s[i] += int(m.group(i + 1))
                s[3] += int(g.group(1)) if g else 0
        res2 &= ok_p
        print(f"   {name:>8} ({len(ids):>2} prompt ids, {n} gen): {' '.join(line)}"
              f"{'' if ok_p else '  <-- FAIL'}")
    print(f"   -> {'PASS' if res2 else 'FAIL'}")

    print("\n   acceptance (informational): config rounds drafted accepted accept-rate "
          "tokens/pass ngram-rounds")
    for cname, (r, d, acc, ng) in stats.items():
        if r:
            print(f"   {cname:>10} {r:>6} {d:>7} {acc:>8} {acc / max(d, 1):>11.3f} "
                  f"{1 + acc / r:>11.3f} {ng:>12}")

    ok = res1 and res2
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
