#!/usr/bin/env python3
"""Can an exact bound skip rows of Qwen3.5-0.8B's 248,320-row output head? (offline, PC)

Usage: python tools/audit/head_screen.py models/pq/qwen35-0.8b-bf16.pq ref/golden/qwen35-0.8b

An argmax over the head stays exact if a row is skipped only when an upper bound on its logit is
below a lower bound on the best logit. This script measures, on the golden fp32 final-normed
hidden states (the head's real inputs), how many rows such bounds leave to read ("survivors"):

  norm    Cauchy-Schwarz with the row norm: l_i <= |W_i| |h|  (needs one float per row).
  hi2     read a 2-bit proxy (the high 2 bits of a 4-bit asymmetric per-32-column code, i.e.
          half of the code bytes), bound the dropped low 2 bits per 32-column block by
          1.5 d_b |h_b|_1 (|lo - 1.5| <= 1.5; the mean 1.5 d_b sum(h_b) is added exactly).
  hi3     the same with 3 of 4 bits read (bound 0.5 d_b |h_b|_1).
  hi2-l2  as hi2 but with the exact per-row residual L2 norm |E_i| |h| (extra float per row).

Survivors = rows whose upper bound >= max over rows of (proxy - bound), i.e. rows the screen
cannot exclude; the head is then read as proxy bytes + survivors' remaining bytes. The 4-bit code
here is a simulation (min/max asymmetric per 32 columns, like Q4K's affine form) on the bf16
weights, so the margins are representative of, not identical to, the shipped Q4K head.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HDR = "<8sIIQQI" + "I" * 17
ENTRY = "<80sII4IQQ8s"


def read_head(pq: Path) -> np.ndarray:
    """embed_tokens of a bf16 .pq file as fp32 [vocab, hidden]."""
    with pq.open("rb") as f:
        hdr = struct.unpack(HDR, f.read(struct.calcsize(HDR)))
        n_tensors, dir_off = hdr[2], hdr[3]
        f.seek(dir_off)
        for _ in range(n_tensors):
            e = struct.unpack(ENTRY, f.read(128))
            name = e[0].rstrip(b"\0").decode()
            if name != "embed_tokens.weight":
                continue
            dtype, shape, off, nbytes = e[1], e[3:7], e[7], e[8]
            assert dtype == 1, f"expected BF16 (1), got {dtype}"
            rows, cols = shape[0], shape[1]
            f.seek(off)                                  # absolute file offset (model.c)
            raw = np.frombuffer(f.read(nbytes), dtype=np.uint16).reshape(rows, cols)
            return (raw.astype(np.uint32) << 16).view(np.float32)
    raise SystemExit("embed_tokens.weight not found")


def quant4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric 4-bit per 32 columns: w ~ d * u + mn, u in 0..15. Returns u, d, mn [r, B]."""
    r, c = w.shape
    b = w.reshape(r, c // 32, 32)
    mn, mx = b.min(axis=2), b.max(axis=2)
    d = np.maximum((mx - mn) / 15.0, 1e-12)
    u = np.clip(np.rint((b - mn[..., None]) / d[..., None]), 0, 15).astype(np.int8)
    return u, d.astype(np.float32), mn.astype(np.float32)


def main() -> None:
    w = read_head(Path(sys.argv[1]))
    gold = Path(sys.argv[2])
    hs = np.load(gold / "hidden_states.npy")[-1]          # [T, H], final-normed
    ref = np.load(gold / "logits.npy")                     # [T, V]
    rows, cols = w.shape
    print(f"head {rows} x {cols}, positions {hs.shape[0]}")
    u, d, mn = quant4(w)
    hi, lo = u >> 2, u & 3
    wn = np.linalg.norm(w, axis=1)
    for t, h in enumerate(hs):
        logit = w @ h
        hb = h.reshape(cols // 32, 32)
        h1 = np.abs(hb).sum(axis=1)                        # |h_b|_1 per block
        hsum = hb.sum(axis=1)
        # exact quantized logits and proxies (all in float64 for the bound arithmetic)
        base = (mn * hsum).sum(axis=1)                      # sum_b mn_b sum(h_b)
        hi_dot = np.einsum("rbj,bj->rb", hi.astype(np.float32), hb)
        lq = (d * (4 * hi_dot + np.einsum("rbj,bj->rb", lo.astype(np.float32), hb))).sum(axis=1) + base
        p2 = (d * (4 * hi_dot + 1.5 * hsum)).sum(axis=1) + base      # lo replaced by its mid 1.5
        b2 = (1.5 * d * h1).sum(axis=1)
        hi3 = u >> 1
        p3 = (d * (2 * np.einsum("rbj,bj->rb", hi3.astype(np.float32), hb) + 0.5 * hsum)).sum(axis=1) + base
        b3 = (0.5 * d * h1).sum(axis=1)
        e2 = np.linalg.norm((d[..., None] * (lo - 1.5)).reshape(rows, cols), axis=1) * np.linalg.norm(h)
        m = lq.max()
        res = {
            "norm": (wn * np.linalg.norm(h) >= logit.max()).mean(),
            "hi2": (p2 + b2 >= (p2 - b2).max()).mean(),
            "hi3": (p3 + b3 >= (p3 - b3).max()).mean(),
            "hi2-l2": (p2 + e2 >= (p2 - e2).max()).mean(),
        }
        z = (m - lq.mean()) / lq.std()
        print(f"t={t}: argmax fp32 {int(logit.argmax())} ref {int(ref[t].argmax())} q4 {int(lq.argmax())}; "
              f"max logit {m:.2f} = mean + {z:.1f} std; bound/std hi2 {np.median(b2) / lq.std():.1f} "
              f"hi3 {np.median(b3) / lq.std():.1f} l2 {np.median(e2) / lq.std():.1f}; survivors "
              + " ".join(f"{k} {v:.1%}" for k, v in res.items()))


if __name__ == "__main__":
    main()
