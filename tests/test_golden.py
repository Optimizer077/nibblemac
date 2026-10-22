"""Golden test: pinfer host binary vs the fp32 transformers reference (ref/golden/).

    uv run --no-project --with numpy python tests/test_golden.py [--bin PATH] [--model PATH]

Runs build/host/engine/pinfer_run (through WSL when started from Windows) two ways:
  A. prompt in one batched prefill, dumping per-layer hidden states and all prompt logits,
     then 32 greedy tokens;
  B. same prompt token by token (-b 1), i.e. the decode path, dumping the same tensors;
and checks:
  1. per-layer hidden states of A vs golden  (relative max error, see THRESHOLDS)
  2. prompt logits of A vs golden            (relative max error + top-1 agreement per position)
  3. A vs B                                  (batched prefill == sequential decode path)
  4. greedy ids of A == greedy_ids.npy exactly.

Thresholds. The golden run uses fp32 weights that are the exact fp32 widening of the BF16
checkpoint values the .pq file stores, so the engine sees the same weights and the only error
source is fp32 rounding order (dot products, the chunked vs recurrent delta rule in the
reference). Observed (2026-10-21, matvec_ref): <= 3e-6 relative on every layer and the
logits; a wrong op (sign, scale, missing norm, wrong RoPE
pairing, off-by-one conv tap) gives O(1e-1..1). The limits leave 10-100x headroom over the
observed error (fast/NEON kernels may sum in other orders) yet stay far below any real bug:
  hidden rel. max error  <= 1e-3   (max|ours-gold| / max|gold|, per layer)
  logits rel. max error  <= 1e-3   (the brief allows <~1e-2 for bf16 weights + fp32 compute)
  top-1 agreement        == 100%   on all prompt positions
  A vs B rel. max error  <= 1e-5   (same kernels, only summation grouping may differ)
  greedy ids             exact match on all 32 tokens
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "ref" / "golden" / "qwen35-0.8b"
HIDDEN_TOL, LOGITS_TOL, AB_TOL = 1e-3, 1e-3, 1e-5


def wsl_path(p: Path) -> str:
    """E:\\a\\b -> /mnt/e/a/b (only needed when this script runs on Windows)."""
    s = str(p.resolve())
    if os.name != "nt":
        return s
    return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")


def run_engine(binary: Path, model: Path, ids: np.ndarray, n_gen: int, batch: int,
               tmp: Path) -> tuple[list[int], np.ndarray, np.ndarray, str]:
    hid_f, log_f = tmp / f"hidden_b{batch}.bin", tmp / f"logits_b{batch}.bin"
    args = [wsl_path(binary), "-m", wsl_path(model), "-n", str(n_gen), "-b", str(batch),
            "--ids", ",".join(map(str, ids.tolist())),
            "--dump-hidden", wsl_path(hid_f), "--dump-logits", wsl_path(log_f)]
    if os.name == "nt":
        args = ["wsl.exe", "-d", "Ubuntu", "--"] + args
    proc = subprocess.run(args, capture_output=True, text=True, check=False,
                          env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    if proc.returncode != 0:
        sys.exit(f"engine failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    m = re.search(r"^generated:\s*([\d,]*)\s*$", proc.stdout, re.M)
    gen = [int(t) for t in m.group(1).split(",") if t] if m else []
    stats = [ln for ln in proc.stdout.splitlines() if ln.startswith("load ")]
    return gen, np.fromfile(hid_f, dtype=np.float32), np.fromfile(log_f, dtype=np.float32), \
        stats[0] if stats else ""


def rel_max(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(max abs error, max abs error / max |b|)"""
    err = float(np.max(np.abs(a.astype(np.float64) - b)))
    return err, err / max(float(np.max(np.abs(b))), 1e-30)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin", type=Path, default=ROOT / "build" / "host" / "engine" / "pinfer_run")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "pq" / "qwen35-0.8b-bf16.pq")
    p.add_argument("--golden", type=Path, default=GOLDEN,
                   help="golden directory from ref/dump_reference.py (e.g. ref/golden/qwen35-2b)")
    a = p.parse_args()

    ids = np.load(a.golden / "input_ids.npy").astype(np.int32)
    g_hid = np.load(a.golden / "hidden_states.npy")
    g_log = np.load(a.golden / "logits.npy")
    g_greedy = np.load(a.golden / "greedy_ids.npy").tolist()
    n_ent, T, H = g_hid.shape
    V = g_log.shape[1]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        gen, hid, log, stats = run_engine(a.bin, a.model, ids, len(g_greedy), 64, tmp)
        print(f"[A] batched prefill: {stats}")
        _, hid_b, log_b, stats_b = run_engine(a.bin, a.model, ids, 1, 1, tmp)
        print(f"[B] token-by-token : {stats_b}")
    hid, hid_b = hid.reshape(n_ent, T, H), hid_b.reshape(n_ent, T, H)
    log, log_b = log.reshape(T, V), log_b.reshape(T, V)

    print(f"golden: {a.golden}\nmodel : {a.model}")
    print(f"\n1. hidden states vs golden (entry 0 = embeddings, {n_ent - 1} = final-normed)")
    print(f"   {'entry':>5} {'max|gold|':>10} {'max abs err':>12} {'rel max err':>12} "
          f"{'rel rms err':>12}")
    worst = 0.0
    for e in range(n_ent):
        err, rel = rel_max(hid[e], g_hid[e])
        rms = float(np.linalg.norm(hid[e] - g_hid[e]) / np.linalg.norm(g_hid[e]))
        worst = max(worst, rel)
        flag = "" if rel <= HIDDEN_TOL else "  <-- FAIL"
        print(f"   {e:>5} {np.max(np.abs(g_hid[e])):>10.3f} {err:>12.3e} {rel:>12.3e} "
              f"{rms:>12.3e}{flag}")
    res1 = worst <= HIDDEN_TOL
    print(f"   worst rel max err {worst:.3e} (limit {HIDDEN_TOL:g}) -> {'PASS' if res1 else 'FAIL'}")

    print("\n2. prompt logits vs golden")
    err, rel = rel_max(log, g_log)
    top_ours, top_gold = log.argmax(1), g_log.argmax(1)
    agree = int((top_ours == top_gold).sum())
    for t in range(T):
        e_t, r_t = rel_max(log[t], g_log[t])
        print(f"   pos {t}: top1 ours {top_ours[t]:>6} gold {top_gold[t]:>6} "
              f"{'ok' if top_ours[t] == top_gold[t] else 'MISMATCH'}  max abs err {e_t:.3e} "
              f"rel {r_t:.3e}")
    res2 = rel <= LOGITS_TOL and agree == T
    print(f"   max abs err {err:.3e}, rel max err {rel:.3e} (limit {LOGITS_TOL:g}), "
          f"top-1 {agree}/{T} -> {'PASS' if res2 else 'FAIL'}")

    print("\n3. batched prefill (A) vs token-by-token decode path (B)")
    _, rel_h = rel_max(hid, hid_b)
    _, rel_l = rel_max(log, log_b)
    res3 = max(rel_h, rel_l) <= AB_TOL
    print(f"   hidden rel max err {rel_h:.3e}, logits rel max err {rel_l:.3e} "
          f"(limit {AB_TOL:g}) -> {'PASS' if res3 else 'FAIL'}")

    print("\n4. greedy generation")
    match = sum(1 for x, y in zip(gen, g_greedy) if x == y)
    first_bad = next((i for i, (x, y) in enumerate(zip(gen, g_greedy)) if x != y), None)
    res4 = gen == g_greedy
    print(f"   ours: {gen}\n   gold: {g_greedy}")
    print(f"   {match}/{len(g_greedy)} match"
          + (f", first mismatch at {first_bad}" if first_bad is not None else "")
          + f" -> {'PASS' if res4 else 'FAIL'}")

    ok = res1 and res2 and res3 and res4
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
