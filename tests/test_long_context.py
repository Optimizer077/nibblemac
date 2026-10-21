"""Long-context test: host engine vs the fp32 transformers reference beyond 2048 tokens.

    /root/venv-ref/bin/python tests/test_long_context.py [--ids F] [--model PQ] [--lib SO]
        [--ref-dir DIR] [--save-engine F.npy] [--load-engine F.npy]

Positions past 4096 exercise the partial RoPE (64 of 256 dims, theta = 1e7) at angles the
short golden prompt never reaches, the KV cache over thousands of rows, and 8192 steps of the
Gated DeltaNet recurrence. The engine runs through the public API (the Python package over
the host libpinfer.so, MATVEC=ref, which is bit-identical to the Pi's NEON build): the prompt
is fed in pieces ending at each checked prefix length n, and the logits pinfer_eval returns
for the last token of the piece (token n-1) are compared with the reference's logits of the
same token (ref/long_context_reference.py; generated on first use, ~3 min, cached in
--ref-dir). pinfer_run --dump-logits is not used: it writes every prompt position, 8 GB at
8192 tokens. The engine part takes ~40 min on the host (scalar kernels, 3-5 tok/s); the
reference ~3.5 min and 6.6 GB of RAM.

Checks per prefix length (default: every 512 tokens to 8192, plus 3000 and 6000):
  1. logits rel. max error <= 1e-3  (max|ours - ref| / max|ref|)
  2. top-1 agrees, unless the reference's own top-2 margin is below 2x the max abs error at
     that position (a tie that fp32 rounding may legitimately flip; reported, not failed)
Tolerance. Same limit as tests/test_golden.py, whose short prompt shows <= 3e-6: the error
sources are the same (fp32 summation order; the reference's chunked delta rule vs the
engine's recurrent form), only more of them, so long-range drift can grow the error by one or
two orders of magnitude and still sit 10x below the limit, while a position bug (wrong RoPE
angle, wrong pairing, a KV row off by one) gives O(1e-1..1) like any wrong op.
Also printed: KL(ref || ours) and the top-5 overlap per position.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref"))
sys.path.insert(0, str(ROOT / "python"))

from long_context_reference import (
    DEFAULT_POSITIONS,
    ids_sha256,
    read_ids,
    reference_logits,
)

LOGITS_TOL = 1e-3
TIE_FACTOR = 2.0


def load_reference(ref_dir: Path, hf_model: Path, ids: list[int], positions: list[int],
                   chunk: int) -> np.ndarray:
    """Cached reference logits for (ids, positions), computed and saved when missing."""
    meta_f = ref_dir / "meta.json"
    if meta_f.is_file():
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        if meta.get("ids_sha256") == ids_sha256(ids) and meta.get("positions") == positions:
            print(f"reference: cached {ref_dir} ({meta['transformers']}, {meta['dtype']})")
            return np.load(ref_dir / "logits.npy")
    print(f"reference: computing with transformers (fp32, sdpa) -> {ref_dir}")
    logits, secs = reference_logits(hf_model, ids, positions, chunk)
    import torch
    import transformers

    ref_dir.mkdir(parents=True, exist_ok=True)
    np.save(ref_dir / "positions.npy", np.asarray(positions, dtype=np.int32))
    np.save(ref_dir / "logits.npy", logits)
    meta = {"model": str(hf_model), "n_ids": len(ids), "ids_sha256": ids_sha256(ids),
            "positions": positions, "transformers": transformers.__version__,
            "torch": torch.__version__, "dtype": "float32", "attn_implementation": "sdpa",
            "chunk": chunk, "seconds": round(secs, 1), "logits_shape": list(logits.shape)}
    meta_f.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"reference: {secs:.0f} s")
    return logits


def engine_logits(lib: Path, model: Path, ids: list[int], positions: list[int],
                  threads: int) -> np.ndarray:
    """Logits of token n-1 for every prefix length n, from one pass over the prompt."""
    import pinfer

    m = pinfer.Model.load(model, tokenizer=False, n_threads=threads,
                          max_ctx=max(positions), lib_path=lib)
    out = np.empty((len(positions), m.n_vocab), dtype=np.float32)
    done, last = 0, None
    t0 = time.time()
    try:
        for i in sorted(range(len(positions)), key=lambda j: positions[j]):
            p = positions[i]
            if p > done:
                last = m.eval(ids[done:p])
                done = p
                print(f"engine: {p} tokens, {time.time() - t0:.0f} s", flush=True)
            assert last is not None
            out[i] = last
    finally:
        m.close()
    return out


def compare(positions: list[int], ours: np.ndarray, ref: np.ndarray) -> bool:
    print(f"\n{'n':>6} {'max|ref|':>9} {'max abs err':>12} {'rel max err':>12} {'KL':>10} "
          f"{'top1 ours/ref':>16} {'ref margin':>10} {'top5':>5}")
    ok = True
    worst = 0.0
    for i, p in enumerate(positions):
        o, r = ours[i].astype(np.float64), ref[i].astype(np.float64)
        err = float(np.max(np.abs(o - r)))
        rel = err / float(np.max(np.abs(r)))
        worst = max(worst, rel)
        lr = r - r.max()
        lo = o - o.max()
        pr = np.exp(lr) / np.exp(lr).sum()
        kl = float(np.sum(pr * ((lr - np.log(np.exp(lr).sum())) - (lo - np.log(np.exp(lo).sum())))))
        t_o, t_r = int(o.argmax()), int(r.argmax())
        top2 = np.sort(r)[-2:]
        margin = float(top2[1] - top2[0])
        top5 = len(set(np.argsort(o)[-5:].tolist()) & set(np.argsort(r)[-5:].tolist()))
        flag = ""
        if rel > LOGITS_TOL:
            flag, ok = "  <-- FAIL (error)", False
        elif t_o != t_r:
            if margin < TIE_FACTOR * err:
                flag = "  (tie: top-1 flip within rounding)"
            else:
                flag, ok = "  <-- FAIL (top-1)", False
        print(f"{p:>6} {np.max(np.abs(r)):>9.3f} {err:>12.3e} {rel:>12.3e} {kl:>10.2e} "
              f"{t_o:>7}/{t_r:<8} {margin:>10.4f} {top5:>5}{flag}")
    print(f"\nworst rel max err {worst:.3e} (limit {LOGITS_TOL:g})")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", type=Path, default=ROOT / "bench" / "common" / "ctx8192.ids")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "pq" / "qwen35-0.8b-bf16.pq")
    p.add_argument("--hf-model", type=Path, default=ROOT / "models" / "Qwen3.5-0.8B")
    p.add_argument("--lib", type=Path,
                   default=ROOT / "build" / "host" / "engine" / "lib" / "libpinfer.so")
    p.add_argument("--ref-dir", type=Path, default=ROOT / "ref" / "golden" / "qwen35-0.8b-long")
    p.add_argument("--positions", default=",".join(map(str, DEFAULT_POSITIONS)))
    p.add_argument("--chunk", type=int, default=0, help="reference chunk size (0 = one pass)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--save-engine", type=Path, help="save the engine logits (.npy)")
    p.add_argument("--load-engine", type=Path, help="reuse saved engine logits (.npy)")
    a = p.parse_args()

    ids = read_ids(a.ids)
    positions = [int(x) for x in a.positions.split(",") if x]
    if not positions or min(positions) < 1 or max(positions) > len(ids):
        sys.exit(f"positions must lie in 1..{len(ids)}")
    print(f"{a.ids.name}: {len(ids)} ids, {len(positions)} prefix lengths up to {max(positions)}")
    ref = load_reference(a.ref_dir, a.hf_model, ids, positions, a.chunk)
    if a.load_engine:
        ours = np.load(a.load_engine)
    else:
        ours = engine_logits(a.lib, a.model, ids, positions, a.threads)
        if a.save_engine:
            np.save(a.save_engine, ours)
    ok = compare(positions, ours, ref)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
