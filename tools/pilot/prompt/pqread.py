"""Read a pinfer `.pq` file (engine/FORMAT.md) and dequantize its tensors to fp32 numpy.

Q8_G32 / Q4_G32 / Q4K_G32 tile layout: rows in tiles of 4, each tile = cols/256 superblocks,
each superblock = a 64-byte scale line + 8 blocks x 4 rows of codes.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

HEADER_FMT = "<8sIIQQ" + "I" * 18 + "ff"
ENTRY_FMT = "<80sII4IQQ8x"
F32, BF16, Q8_G32, Q4_G32, Q4K_G32 = 0, 1, 2, 3, 4
SB = {Q8_G32: 1088, Q4_G32: 576, Q4K_G32: 576}
CHUNK_ROWS = 8192          # dequantize large matrices in row chunks (multiple of 4)
HDR_NAMES = ["magic", "version", "n_tensors", "dir_offset", "data_offset", "arch", "hidden",
             "n_layers", "vocab", "ffn", "n_heads", "n_kv_heads", "head_dim", "rot_dim",
             "lin_k_heads", "lin_v_heads", "lin_k_dim", "lin_v_dim", "conv_k", "full_attn_mask",
             "eos_id", "tied_embeddings", "has_mtp", "rms_eps", "rope_theta"]


class PQ:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        vals = struct.unpack(HEADER_FMT, bytes(self.mm[:struct.calcsize(HEADER_FMT)]))
        self.header = dict(zip(HDR_NAMES, vals))
        self.entries = {}
        off = self.header["dir_offset"]
        for i in range(self.header["n_tensors"]):
            e = struct.unpack(ENTRY_FMT, bytes(self.mm[off + 128 * i:off + 128 * (i + 1)]))
            name = e[0].split(b"\0")[0].decode()
            ndim = e[2]
            self.entries[name] = {"dtype": e[1], "shape": list(e[3:3 + ndim]), "offset": e[7],
                                  "nbytes": e[8]}

    def raw(self, name: str) -> np.ndarray:
        e = self.entries[name]
        return self.mm[e["offset"]:e["offset"] + e["nbytes"]]

    def get(self, name: str, rows: slice | None = None) -> np.ndarray:
        e = self.entries[name]
        dt, shape = e["dtype"], e["shape"]
        buf = self.raw(name)
        if dt == F32:
            return np.frombuffer(buf, dtype="<f4").reshape(shape).copy()
        if dt == BF16:
            return (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32) \
                .reshape(shape).copy()
        rows_, cols = shape
        if rows_ <= CHUNK_ROWS:
            return dequant(dt, buf, rows_, cols)
        # big matrices (embedding, head) tile by tile: the transient is one chunk, not 3x the matrix
        out = np.empty((rows_, cols), dtype=np.float32)
        tile_bytes = (cols // 256) * SB[dt]           # one tile = 4 rows
        for r0 in range(0, rows_, CHUNK_ROWS):
            r1 = min(r0 + CHUNK_ROWS, rows_)
            out[r0:r1] = dequant(dt, buf[(r0 // 4) * tile_bytes:(r1 // 4) * tile_bytes], r1 - r0, cols)
        return out


def dequant(dt: int, buf: np.ndarray, rows: int, cols: int) -> np.ndarray:
    nsb = cols // 256
    sb = np.asarray(buf).reshape(rows // 4, nsb, SB[dt])
    scale = sb[:, :, :64]
    codes = sb[:, :, 64:]
    if dt == Q8_G32:
        d = scale.copy().view("<f2").astype(np.float32).reshape(rows // 4, nsb, 8, 4)  # [k][i]
        q = codes.copy().view(np.int8).astype(np.float32).reshape(rows // 4, nsb, 8, 4, 32)
        w = q * d[..., None]                                    # [t, sb, k, i, 32]
    else:
        nb = codes.reshape(rows // 4, nsb, 8, 4, 16)
        u = np.concatenate([nb & 15, nb >> 4], axis=-1).astype(np.float32)   # [t,sb,k,i,32]
        if dt == Q4_G32:
            d = scale.copy().view("<f2").astype(np.float32).reshape(rows // 4, nsb, 8, 4)
            w = (u - 8) * d[..., None]
        else:  # Q4K_G32
            D = scale[..., 0:8].copy().view("<f2").astype(np.float32)        # [t, sb, 4]
            M = scale[..., 8:16].copy().view("<f2").astype(np.float32)
            s6 = _unpack6(scale[..., 16:40]).reshape(rows // 4, nsb, 8, 4)
            m6 = _unpack6(scale[..., 40:64]).reshape(rows // 4, nsb, 8, 4)
            w = u * (D[:, :, None, :] * s6)[..., None] - (M[:, :, None, :] * m6)[..., None]
    # [t, sb, k, i, 32] -> [t, i, sb, k, 32] -> [rows, cols]
    return np.ascontiguousarray(w.transpose(0, 3, 1, 2, 4)).reshape(rows, cols)


def _unpack6(b: np.ndarray) -> np.ndarray:
    """24 bytes -> 32 values in [0, 63] (engine/FORMAT.md 6-bit arrays), vectorized."""
    lo = b[..., :16]
    hi = b[..., 16:24].astype(np.uint8)
    v = np.empty(b.shape[:-1] + (32,), dtype=np.float32)
    lo_l = (lo & 15).astype(np.uint8)
    lo_h = (lo >> 4).astype(np.uint8)
    h = [(hi >> (2 * k)) & 3 for k in range(4)]                  # bits for values j + 8k
    hv = np.concatenate(h, axis=-1)                              # 32 high-bit pairs, index j + 8k
    v[..., :16] = lo_l | (hv[..., :16] << 4)
    v[..., 16:] = lo_h | (hv[..., 16:] << 4)
    return v
