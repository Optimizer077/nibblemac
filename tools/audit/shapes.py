#!/usr/bin/env python3
"""Weights, multiply-accumulates and state sizes per token for Cerno 4B and Qwen3.5-0.8B.

Usage: python tools/audit/shapes.py

Shapes from the .pq headers (docs/ARCHITECTURE.md section 1; cerno-pi CERNO.md section 2). Bytes:
Q4_0 / Q4K = 4.5 bits per weight (576 B per 4 rows x 256 columns), Q8_0 = 8.5 bits.
The DeltaNet recurrence counts 7 fp32 operations per state element and token (decay multiply;
S^T k multiply-add; rank-1 update multiply-add; S^T q multiply-add).
"""
from __future__ import annotations

Q4, Q8 = 4.5 / 8, 8.5 / 8


def net(name: str, h: int, f: int, n_lin: int, n_full: int, qkv: int, z: int, ab: int,
        q: int, kv: int, o_in: int, out_in: int, vocab: int, head_bpw: float,
        lin_heads: int, n_kv: int, hd: int, conv_dim: int) -> None:
    lin = {"in_proj_qkv": (qkv, h), "in_proj_z": (z, h), "in_proj_a + b": (2 * ab, h),
           "out_proj": (h, out_in), "gate + up": (2 * f, h), "down": (h, f)}
    full = {"q_proj (+ gate)": (q, h), "k_proj + v_proj": (2 * kv, h), "o_proj": (h, o_in),
            "gate + up": (2 * f, h), "down": (h, f)}
    print(f"== {name}: {n_lin} linear-attention + {n_full} full-attention layers")
    body = 0
    for kind, ops, n in (("linear", lin, n_lin), ("full", full, n_full)):
        t = 0
        for op, (r, c) in ops.items():
            print(f"  {kind:7} {op:16} {r:6d} x {c:5d}  {r * c * Q4 / 1e6:7.2f} MB/layer  "
                  f"{r * c / 1e6:7.2f} MMAC/layer  x{n} = {n * r * c * Q4 / 1e6:8.2f} MB")
            t += r * c
        print(f"  {kind:7} layer total {t * Q4 / 1e6:.2f} MB, {t / 1e6:.1f} MMAC")
        body += n * t
    print(f"  body {body * Q4 / 1e6:.1f} MB, {body / 1e9:.3f} GMAC = {2 * body / 1e9:.2f} GOP per token")
    print(f"  head {vocab} x {h}: {vocab * h * head_bpw / 1e6:.1f} MB, {vocab * h / 1e6:.1f} MMAC; "
          f"one 4-row tile {4 * h * head_bpw:.0f} B")
    st = n_lin * lin_heads * 128 * 128 * 4
    print(f"  recurrent state {st / 2**20:.1f} MiB ({st / 1e6:.1f} MB), conv state "
          f"{n_lin * 3 * conv_dim * 4 / 2**20:.2f} MiB, KV {n_full * 2 * n_kv * hd * 4 / 1024:.0f} KiB "
          f"per token (fp32); recurrence {n_lin * lin_heads * 7 * 128 * 128 / 1e6:.0f} MFLOP per token")


def main() -> None:
    net("Cerno 4B (Q4_0 body, Q8_0 head)", 2560, 9216, 24, 8, 8192, 4096, 32, 8192, 1024,
        4096, 4096, 248320, Q8, 32, 4, 256, 8192)
    net("Qwen3.5-0.8B (Q4K body, Q4K head)", 1024, 3584, 18, 6, 6144, 2048, 16, 4096, 512,
        2048, 2048, 248320, Q4, 16, 2, 256, 6144)


if __name__ == "__main__":
    main()
