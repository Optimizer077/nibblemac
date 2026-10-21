"""Reference quantizers, packers and error report for pinfer's quantized dtypes.

numpy only. The packed byte layouts are defined in engine/FORMAT.md ("Quantized dtypes")
and engine/src/pq_quant.h; this file is the reference the converter and the C kernels
must agree with (`golden` writes a test file that engine/bench/kbench checks).

    python tools/quant.py report  [--model models/Qwen3.5-0.8B] [--sample-embed 4]
    python tools/quant.py golden  out.bin   # packed test matrices + expected outputs
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# dtype ids (must match engine/src/pq_quant.h)
Q8_G32, Q4_G32, Q4K_G32 = 2, 3, 4
TILE = 4          # rows interleaved per tile
SB_COLS = 256     # columns per superblock (one scale line per superblock)

# ---------------------------------------------------------------- quantizers
# All quantizers take w [R, C] float32 (C % G == 0) and return integer codes plus
# per-group parameters already rounded to fp16, so dequant() is exactly what kernels see.


def _groups(w: np.ndarray, g: int) -> np.ndarray:
    r, c = w.shape
    return w.reshape(r, c // g, g)


def quant_sym(w: np.ndarray, g: int, bits: int, search: bool = True
              ) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric: w ~ d * q, q in [-(2^(b-1)-1), 2^(b-1)-1] for 8 bit, [-8, 7] for 4 bit.
    With search, a few scale candidates are tried per group and the min-SSE one is kept."""
    x = _groups(w.astype(np.float32), g)
    if bits == 8:
        qmin, qmax = -127, 127
        amax = np.abs(x).max(axis=-1, keepdims=True)
        base = amax / 127.0
    else:
        qmin, qmax = -8, 7
        # Q4_0 trick: map the signed extreme to -8 so the full 16-level range is used.
        idx = np.abs(x).argmax(axis=-1)[..., None]
        base = np.take_along_axis(x, idx, axis=-1) / -8.0
    factors = (1.0,) if not search else (1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 1.03)
    best_q, best_d, best_e = None, None, None
    for f in factors:
        d = (base * f).astype(np.float16).astype(np.float32)
        safe = np.where(d == 0, 1.0, d)
        q = np.clip(np.rint(x / safe), qmin, qmax)
        if bits == 4 and search:  # least-squares refit of d for these codes
            num = (x * q).sum(-1, keepdims=True)
            den = (q * q).sum(-1, keepdims=True)
            d = np.where(den > 0, num / np.where(den == 0, 1, den), d)
            d = d.astype(np.float16).astype(np.float32)
            safe = np.where(d == 0, 1.0, d)
            q = np.clip(np.rint(x / safe), qmin, qmax)
        e = ((x - d * q) ** 2).sum(-1, keepdims=True)
        if best_e is None:
            best_q, best_d, best_e = q, d, e
        else:
            m = e < best_e
            best_q = np.where(m, q, best_q)
            best_d = np.where(m, d, best_d)
            best_e = np.where(m, e, best_e)
    r, c = w.shape
    return best_q.reshape(r, c).astype(np.int8), best_d[..., 0].astype(np.float16)


def quant_asym4(w: np.ndarray, g: int, iters: int = 4
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric 4 bit: w ~ d * q + m, q in [0, 15]. Start from min/max, then alternate
    least-squares refits of (d, m) and re-rounding of q."""
    x = _groups(w.astype(np.float32), g)
    lo = x.min(-1, keepdims=True)
    hi = x.max(-1, keepdims=True)
    d = (hi - lo) / 15.0
    m = lo
    for it in range(iters + 1):
        d16 = d.astype(np.float16).astype(np.float32)
        m16 = m.astype(np.float16).astype(np.float32)
        safe = np.where(d16 == 0, 1.0, d16)
        q = np.clip(np.rint((x - m16) / safe), 0, 15)
        if it == iters:
            break
        # LS fit of x ~ d q + m per group
        n = float(g)
        sq, sx = q.sum(-1, keepdims=True), x.sum(-1, keepdims=True)
        sqq, sqx = (q * q).sum(-1, keepdims=True), (q * x).sum(-1, keepdims=True)
        den = n * sqq - sq * sq
        ok = den > 0
        dn = np.where(ok, (n * sqx - sq * sx) / np.where(ok, den, 1), d)
        mn = np.where(ok, (sx - dn * sq) / n, m)
        d, m = dn, mn
    r, c = w.shape
    return (q.reshape(r, c).astype(np.uint8), d16[..., 0].astype(np.float16),
            m16[..., 0].astype(np.float16))


def quant_q4k(w: np.ndarray, iters: int = 2
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """K-quant style 4 bit: per 32-column group w ~ D * s * q - M * m with q in [0, 15], and
    6-bit integers s, m in [0, 63] shared under fp16 super-scales D, M per 256 columns of a
    row. Returns (q uint8 [R, C], s uint8 [R, C/32], m uint8 [R, C/32], D f16, M f16 [R, C/256])."""
    r, c = w.shape
    x = w.astype(np.float32).reshape(r, c // 32, 32)
    _, dg, mg = quant_asym4(w, 32)                      # per-group LS fit (fp16-rounded)
    dg = dg.astype(np.float32)
    mn = np.maximum(-mg.astype(np.float32), 0.0)        # store -min >= 0 (mins are <= 0)
    xs = lambda a: a.reshape(r, c // 256, 8)
    D = (xs(dg).max(-1) / 63.0).astype(np.float16).astype(np.float32)
    M = (xs(mn).max(-1) / 63.0).astype(np.float16).astype(np.float32)
    Dg, Mg = np.repeat(D, 8, axis=1), np.repeat(M, 8, axis=1)
    sq = np.clip(np.rint(dg / np.where(Dg == 0, 1, Dg)), 0, 63)
    mq = np.clip(np.rint(mn / np.where(Mg == 0, 1, Mg)), 0, 63)
    for it in range(iters + 1):
        d_eff = (Dg * sq)[..., None]
        m_eff = (Mg * mq)[..., None]
        q = np.clip(np.rint((x + m_eff) / np.where(d_eff == 0, 1, d_eff)), 0, 15)
        if it == iters:
            break
        # refit the 6-bit sub-scale / sub-min of each group for these codes (LS, then round)
        n = 32.0
        sq_, sx = q.sum(-1), x.sum(-1)
        sqq, sqx = (q * q).sum(-1), (q * x).sum(-1)
        den = n * sqq - sq_ * sq_
        ok = den > 0
        dn = np.where(ok, (n * sqx - sq_ * sx) / np.where(ok, den, 1), Dg * sq)
        mm = np.where(ok, -(sx - dn * sq_) / n, Mg * mq)
        sq = np.clip(np.rint(dn / np.where(Dg == 0, 1, Dg)), 0, 63)
        mq = np.clip(np.rint(np.maximum(mm, 0) / np.where(Mg == 0, 1, Mg)), 0, 63)
    return (q.reshape(r, c).astype(np.uint8), sq.astype(np.uint8), mq.astype(np.uint8),
            D.astype(np.float16), M.astype(np.float16))


def dequant_q4k(q, s, m, D, M) -> np.ndarray:
    r, c = q.shape
    d = np.repeat(D.astype(np.float32), 8, axis=1) * s
    mn = np.repeat(M.astype(np.float32), 8, axis=1) * m
    return (q.reshape(r, c // 32, 32) * d[..., None] - mn[..., None]).reshape(r, c)


def dequant_sym(q: np.ndarray, d: np.ndarray, g: int) -> np.ndarray:
    r, c = q.shape
    return (q.reshape(r, c // g, g).astype(np.float32) * d.astype(np.float32)[..., None]
            ).reshape(r, c)


def dequant_asym(q: np.ndarray, d: np.ndarray, m: np.ndarray, g: int) -> np.ndarray:
    r, c = q.shape
    x = q.reshape(r, c // g, g).astype(np.float32) * d.astype(np.float32)[..., None]
    return (x + m.astype(np.float32)[..., None]).reshape(r, c)


# ---------------------------------------------------------------- packers
# Common structure: rows are taken 4 at a time (a tile); each tile is a run of superblocks
# of 256 columns; each superblock = [scale line(s)][codes], all 16-byte aligned.
# Codes within a superblock are ordered group by group, and within a group row by row.


def _check(r: int, c: int) -> None:
    if r % TILE or c % SB_COLS:
        raise ValueError(f"shape {r}x{c}: rows must be a multiple of {TILE}, cols of {SB_COLS}")


def _scale_line(s: np.ndarray, t: int, sb: int, gps: int) -> bytes:
    """fp16 scales of tile t, superblock sb: [group][row] order (4 rows contiguous)."""
    blk = s[t * TILE:(t + 1) * TILE, sb * gps:(sb + 1) * gps]   # [4, gps]
    return np.ascontiguousarray(blk.T).astype("<f2").tobytes()


def pack_q8(q: np.ndarray, d: np.ndarray) -> bytes:
    """Q8_G32: superblock = 64 B scales (8 groups x 4 rows fp16) + 8 x (4 rows x 32 B)."""
    r, c = q.shape
    _check(r, c)
    g, gps = 32, SB_COLS // 32
    out = bytearray()
    qq = q.reshape(r // TILE, TILE, c // g, g)              # [tile, row, group, 32]
    for t in range(r // TILE):
        for sb in range(c // SB_COLS):
            out += _scale_line(d, t, sb, gps)
            out += np.ascontiguousarray(qq[t, :, sb * gps:(sb + 1) * gps].transpose(1, 0, 2)
                                        ).astype(np.int8).tobytes()
    return bytes(out)


def _nibbles(u: np.ndarray) -> np.ndarray:
    """u [..., 32] in 0..15 -> [..., 16] bytes: low nibble = col j, high nibble = col j+16."""
    return (u[..., :16] | (u[..., 16:] << 4)).astype(np.uint8)


def pack_q4(q: np.ndarray, d: np.ndarray) -> bytes:
    """Q4_G32 (symmetric, codes stored as q+8): superblock = 64 B scales + 8 x (4 x 16 B)."""
    r, c = q.shape
    _check(r, c)
    g, gps = 32, SB_COLS // 32
    u = (q.astype(np.int16) + 8).astype(np.uint8).reshape(r // TILE, TILE, c // g, g)
    nb = _nibbles(u)                                        # [tile, row, group, 16]
    out = bytearray()
    for t in range(r // TILE):
        for sb in range(c // SB_COLS):
            out += _scale_line(d, t, sb, gps)
            out += np.ascontiguousarray(nb[t, :, sb * gps:(sb + 1) * gps].transpose(1, 0, 2)
                                        ).tobytes()
    return bytes(out)


def _pack6(v: np.ndarray) -> bytes:
    """32 values in [0, 63] -> 24 bytes: 16 B of low nibbles (byte i: value i | value i+16
    << 4) then 8 B of high bit pairs (byte i: bits 2k..2k+1 = value i + 8k >> 4)."""
    v = v.astype(np.uint8)
    lo = (v[:16] & 15) | ((v[16:] & 15) << 4)
    h = v >> 4
    hi = h[0:8] | (h[8:16] << 2) | (h[16:24] << 4) | (h[24:32] << 6)
    return lo.astype(np.uint8).tobytes() + hi.astype(np.uint8).tobytes()


def pack_q4k(q: np.ndarray, s6: np.ndarray, m6: np.ndarray, D: np.ndarray, M: np.ndarray
             ) -> bytes:
    """Q4K_G32: superblock = scale line [D: 4 rows fp16][M: 4 rows fp16][32 x 6-bit s]
    [32 x 6-bit m] (64 B; 6-bit arrays in [group][row] order, see _pack6), then the codes
    exactly as in Q4_G32 (8 blocks x 4 rows x 16 B, low nibble = col j, high = col j+16)."""
    r, c = q.shape
    _check(r, c)
    gps = SB_COLS // 32
    nb = _nibbles(q.astype(np.uint8).reshape(r // TILE, TILE, c // 32, 32))
    out = bytearray()
    for t in range(r // TILE):
        rows = slice(t * TILE, (t + 1) * TILE)
        for sb in range(c // SB_COLS):
            grp = slice(sb * gps, (sb + 1) * gps)
            out += D[rows, sb].astype("<f2").tobytes() + M[rows, sb].astype("<f2").tobytes()
            out += _pack6(np.ascontiguousarray(s6[rows, grp].T).reshape(-1))
            out += _pack6(np.ascontiguousarray(m6[rows, grp].T).reshape(-1))
            out += np.ascontiguousarray(nb[t, :, grp].transpose(1, 0, 2)).tobytes()
    return bytes(out)


def quantize_pack(dtype: int, w: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Quantize + pack one matrix. Returns (packed bytes, dequantized weights)."""
    if dtype == Q8_G32:
        q, d = quant_sym(w, 32, 8)
        return pack_q8(q, d), dequant_sym(q, d, 32)
    if dtype == Q4_G32:
        q, d = quant_sym(w, 32, 4)
        return pack_q4(q, d), dequant_sym(q, d, 32)
    if dtype == Q4K_G32:
        parts = quant_q4k(w)
        return pack_q4k(*parts), dequant_q4k(*parts)
    raise ValueError(f"unknown quantized dtype {dtype}")


def packed_nbytes(dtype: int, rows: int, cols: int) -> int:
    per_sb = {Q8_G32: 64 + 1024, Q4_G32: 64 + 512, Q4K_G32: 64 + 512}[dtype]
    return rows // TILE * cols // SB_COLS * per_sb


# ---------------------------------------------------------------- error report
# Variants studied (name, bits/weight, function returning the dequantized matrix).
VARIANTS = {
    "q8_g32": (8.5, lambda w: dequant_sym(*quant_sym(w, 32, 8), 32)),
    "q8_g64": (8.25, lambda w: dequant_sym(*quant_sym(w, 64, 8), 64)),
    "q8_g128": (8.125, lambda w: dequant_sym(*quant_sym(w, 128, 8), 128)),
    "q4s_g32_plain": (4.5, lambda w: dequant_sym(*quant_sym(w, 32, 4, search=False), 32)),
    "q4s_g32": (4.5, lambda w: dequant_sym(*quant_sym(w, 32, 4), 32)),
    "q4s_g64": (4.25, lambda w: dequant_sym(*quant_sym(w, 64, 4), 64)),
    "q4a_g32": (5.0, lambda w: dequant_asym(*quant_asym4(w, 32), 32)),
    "q4a_g64": (4.5, lambda w: dequant_asym(*quant_asym4(w, 64), 64)),
    "q4a_g128": (4.25, lambda w: dequant_asym(*quant_asym4(w, 128), 128)),
    "q4k_g32": (4.5, lambda w: dequant_q4k(*quant_q4k(w))),
}


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor_type(name: str) -> str:
    name = name.replace("model.language_model.", "")
    return re.sub(r"layers\.\d+\.", "", name).replace(".weight", "")


def report(model_dir: Path, sample_embed: int, rows_per_chunk: int,
           only: list[str] | None = None) -> dict:
    files = sorted(model_dir.glob("*.safetensors"))
    acc: dict[str, dict] = {}
    for path in files:
        header, base = read_header(path)
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        for name, info in header.items():
            if name == "__metadata__" or "visual" in name or len(info["shape"]) != 2:
                continue
            if info["dtype"] != "BF16":
                raise ValueError(f"{name}: expected BF16, got {info['dtype']}")
            r, c = info["shape"]
            s, e = info["data_offsets"]
            raw = mm[base + s:base + e].view(np.uint16).reshape(r, c)
            step = sample_embed if "embed_tokens" in name else 1
            ttype = tensor_type(name)
            a = acc.setdefault(ttype, {"n": 0, "shape": [r, c], "w2": 0.0,
                                       **{v: 0.0 for v in VARIANTS},
                                       **{"worst_" + v: 0.0 for v in VARIANTS}})
            a["n"] += 1
            t_w2, t_e2 = 0.0, {v: 0.0 for v in VARIANTS}
            for r0 in range(0, r, rows_per_chunk * step):
                blk = raw[r0:r0 + rows_per_chunk * step:step]
                w = (blk.astype(np.uint32) << 16).view(np.float32)
                t_w2 += float((w.astype(np.float64) ** 2).sum())
                for v, (_, fn) in VARIANTS.items():
                    t_e2[v] += float(((fn(w) - w).astype(np.float64) ** 2).sum())
            a["w2"] += t_w2
            for v in VARIANTS:
                a[v] += t_e2[v]
                a["worst_" + v] = max(a["worst_" + v], (t_e2[v] / t_w2) ** 0.5)
            logger.info("%-60s done", name)
    table = {}
    for ttype, a in acc.items():
        table[ttype] = {"count": a["n"], "shape": a["shape"],
                        **{v: (a[v] / a["w2"]) ** 0.5 for v in VARIANTS},
                        **{"worst_" + v: a["worst_" + v] for v in VARIANTS}}
    return table


def print_table(table: dict) -> None:
    cols = list(VARIANTS)
    print("relative RMS error ||W - Q(W)|| / ||W|| (aggregated over all tensors of a type), %")
    print(f"{'tensor type':44s}" + "".join(f"{v:>14s}" for v in cols))
    print(f"{'bits/weight':44s}" + "".join(f"{VARIANTS[v][0]:>14.3f}" for v in cols))
    for ttype in sorted(table):
        row = table[ttype]
        print(f"{ttype:44s}" + "".join(f"{100 * row[v]:>14.3f}" for v in cols))


# ---------------------------------------------------------------- golden test file
# Layout (all little-endian): u32 magic 'PQGT', u32 count; per case:
#   u32 dtype, rows, cols, n; u64 packed_nbytes; f32 X[n][cols]; f32 Ydeq[n][rows];
#   packed bytes. Ydeq = dequant(W) @ X in float64 (kernel error includes activation
#   quantization, so kbench checks with a tolerance).
GOLDEN_MAGIC = 0x54475150


def golden(out: Path, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    cases = []
    for dtype in (Q8_G32, Q4_G32, Q4K_G32):
        for rows, cols, n in ((8, 256, 1), (16, 512, 3), (64, 1024, 8)):
            w = rng.standard_normal((rows, cols)).astype(np.float32) * 0.05
            w[:, ::37] *= 6.0                       # some outlier columns
            w[1] += 0.02                            # an offset row (tests the Q4K min)
            x = rng.standard_normal((n, cols)).astype(np.float32)
            packed, wd = quantize_pack(dtype, w)
            assert len(packed) == packed_nbytes(dtype, rows, cols)
            y = (x.astype(np.float64) @ wd.astype(np.float64).T).astype(np.float32)
            cases.append((dtype, rows, cols, n, packed, x, y))
    with out.open("wb") as f:
        f.write(struct.pack("<II", GOLDEN_MAGIC, len(cases)))
        for dtype, rows, cols, n, packed, x, y in cases:
            f.write(struct.pack("<IIIIQ", dtype, rows, cols, n, len(packed)))
            f.write(x.tobytes())
            f.write(y.tobytes())
            f.write(packed)
    logger.info("wrote %d cases to %s", len(cases), out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="quantization error on the real checkpoint")
    r.add_argument("--model", type=Path, default=Path("models/Qwen3.5-0.8B"))
    r.add_argument("--sample-embed", type=int, default=4, help="use every k-th embed row")
    r.add_argument("--rows-per-chunk", type=int, default=2048)
    r.add_argument("--json", type=Path, help="also write the table as JSON")
    g = sub.add_parser("golden", help="write the kernel golden test file")
    g.add_argument("out", type=Path)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.cmd == "report":
        table = report(a.model, a.sample_embed, a.rows_per_chunk)
        print_table(table)
        if a.json:
            a.json.write_text(json.dumps({"variants": {v: VARIANTS[v][0] for v in VARIANTS},
                                          "sample_embed": a.sample_embed,
                                          "table": table}, indent=2))
    else:
        golden(a.out)


if __name__ == "__main__":
    main()
