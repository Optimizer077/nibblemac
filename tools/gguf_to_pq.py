"""Convert a llama.cpp GGUF of a Qwen3.5 model (arch `qwen35`) into a pinfer `.pq` file, losslessly.

llama.cpp's Q4_0 and Q8_0 blocks are pinfer's Q4_G32 and Q8_G32 groups (engine/FORMAT.md): 32
weights of one row, one IEEE fp16 scale, value `d * (u - 8)` (Q4, u = nibble, low nibble = column
j, high nibble = column j + 16) or `d * q` (Q8, int8). Only the order of the blocks changes
(pinfer interleaves 4 rows per tile, 8 blocks per superblock), so every scale and code is copied
bit for bit. F32 tensors are copied unchanged. No weight is re-quantized: models trained for
the Q4_0 grid (quantization-aware training) keep exactly the weights they were trained with.

What the converter undoes (llama.cpp `conversion/qwen.py`, `_LinearAttentionVReorderBase`): when
a linear-attention layer has more V heads than K heads, llama.cpp stores the V heads in "tiled"
order (position i * n_k + g holds the HF head g * r + i, r = n_v / n_k) so that ggml_repeat
broadcasts the K heads. pinfer keeps the HF ("grouped") order, so the V rows of `attn_qkv`, the
rows of `attn_gate` (z), `ssm_alpha`, `ssm_beta`, the entries of `ssm_a`, `ssm_dt.bias`, the V
channels of `ssm_conv1d` and the 128-column V blocks of `ssm_out`'s input are permuted back.
These are row permutations, or column permutations by whole 32-column blocks, so they too are
exact. `ssm_a` holds A = -exp(A_log) (computed by the llama.cpp converter); it is written as
`layers.N.linear_attn.A` and the engine uses it directly, so no exp/log round trip is involved.
Norm weights are stored by llama.cpp as `1 + w` for the same tensors as in the .pq format, so
they are copied as they are. An `output.weight` that differs from `token_embd.weight` becomes
`lm_head.weight` (untied head, header `tied_embeddings = 0`).

    python tools/gguf_to_pq.py model.gguf out.pq [--tokenizer HF_DIR] [--verify]

`--verify` re-reads both files and checks, tensor by tensor, that every fp16 scale and every
code (and so every dequantized weight) of the .pq equals the GGUF's after the documented
permutation, and that tools/quant.py's reference packers rebuild the same bytes. Needs numpy.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import ALIGN, ENTRY_FMT, HEADER_BYTES, HEADER_FMT, MAGIC, VERSION
from quant import Q4_G32, Q8_G32, pack_q4, pack_q8

logger = logging.getLogger(__name__)

F32 = 0
GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0 = 0, 1, 2, 8
GGML_NAMES = {GGML_F32: "F32", GGML_F16: "F16", GGML_Q4_0: "Q4_0", GGML_Q8_0: "Q8_0"}
BLOCK_BYTES = {GGML_Q4_0: 18, GGML_Q8_0: 34}       # fp16 d + 16 nibble bytes / 32 int8
PQ_OF_GGML = {GGML_Q4_0: Q4_G32, GGML_Q8_0: Q8_G32}
CODE_BYTES = {Q4_G32: 16, Q8_G32: 32}              # bytes of one 32-weight block row
SIDECAR_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "decider_config.json")   # decider_config: readout temperature (pinfer.decide)

# ------------------------------------------------------------------------------ GGUF reader
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q",
           12: "d"}


@dataclass(frozen=True)
class GgufTensor:
    name: str
    dims: tuple[int, ...]     # ggml order: dims[0] = innermost (columns)
    ggml_type: int
    offset: int               # absolute file offset
    nbytes: int

    @property
    def shape(self) -> list[int]:
        """PyTorch order (rows first)."""
        return list(reversed(self.dims))


class Gguf:
    """Minimal GGUF v2/v3 reader: metadata and a memory map of the tensor data."""

    def __init__(self, path: Path):
        self.path = path
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self._o = 0
        if bytes(self.mm[:4]) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        self._o = 4
        self.version = self._u("I")
        if self.version not in (2, 3):
            raise ValueError(f"{path}: GGUF version {self.version} not supported")
        n_tensors, n_kv = self._u("Q"), self._u("Q")
        self.meta: dict = {}
        for _ in range(n_kv):
            key = self._str()
            self.meta[key] = self._value(self._u("I"))
        infos = []
        for _ in range(n_tensors):
            name = self._str()
            nd = self._u("I")
            dims = tuple(self._u("Q") for _ in range(nd))
            infos.append((name, dims, self._u("I"), self._u("Q")))
        align = int(self.meta.get("general.alignment", 32))
        data0 = -(-self._o // align) * align
        self.tensors: dict[str, GgufTensor] = {}
        for name, dims, t, off in infos:
            n = int(np.prod(dims, dtype=np.int64))
            if t in BLOCK_BYTES:
                if dims[0] % 32:
                    raise ValueError(f"{name}: {dims[0]} columns, not a multiple of 32")
                nbytes = n // 32 * BLOCK_BYTES[t]
            elif t == GGML_F32:
                nbytes = 4 * n
            elif t == GGML_F16:
                nbytes = 2 * n
            else:
                raise ValueError(f"{name}: ggml type {t} not supported (only F32, F16, Q4_0, Q8_0)")
            if data0 + off + nbytes > len(self.mm):
                raise ValueError(f"{name}: data outside the file")
            self.tensors[name] = GgufTensor(name, dims, t, data0 + off, nbytes)

    def _u(self, fmt: str):
        v = struct.unpack_from("<" + fmt, self.mm, self._o)[0]
        self._o += struct.calcsize("<" + fmt)
        return v

    def _str(self) -> str:
        n = self._u("Q")
        s = bytes(self.mm[self._o:self._o + n]).decode("utf-8", "replace")
        self._o += n
        return s

    def _value(self, t: int):
        if t in _SCALAR:
            return self._u(_SCALAR[t])
        if t == 8:
            return self._str()
        if t == 9:
            et, n = self._u("I"), self._u("Q")
            if et in _SCALAR and et != 7:          # numeric arrays: read in one go
                fmt = _SCALAR[et]
                size = struct.calcsize("<" + fmt)
                arr = np.frombuffer(self.mm, dtype=np.dtype("<" + fmt), count=n, offset=self._o)
                self._o += size * n
                return arr.tolist()
            return [self._value(et) for _ in range(n)]
        raise ValueError(f"GGUF metadata type {t} unknown")

    def raw(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        return self.mm[t.offset:t.offset + t.nbytes]

    def f32(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        if t.ggml_type != GGML_F32:
            found = GGML_NAMES.get(t.ggml_type, t.ggml_type)
            raise ValueError(f"{name}: expected F32, found {found}")
        return self.raw(name).view("<f4")

    def blocks(self, name: str) -> np.ndarray:
        """Quantized tensor as [rows, blocks per row, block bytes] (uint8)."""
        t = self.tensors[name]
        return self.raw(name).reshape(t.shape[0], t.dims[0] // 32, BLOCK_BYTES[t.ggml_type])


# ------------------------------------------------------------------------------ model facts
@dataclass(frozen=True)
class Dims:
    hidden: int
    n_layers: int
    vocab: int
    ffn: int
    n_heads: int
    n_kv: int
    head_dim: int
    rot: int
    lk_heads: int
    lv_heads: int
    lk_dim: int
    lv_dim: int
    conv_k: int
    full_mask: int
    eos: int
    eps: float
    theta: float

    @property
    def rep(self) -> int:
        return self.lv_heads // self.lk_heads


def read_dims(g: Gguf) -> Dims:
    m = g.meta
    arch = m.get("general.architecture")
    if arch != "qwen35":
        raise ValueError(f"architecture {arch!r}: only 'qwen35' (Qwen3.5 dense) is supported")
    p = "qwen35."
    n_layers = int(m[p + "block_count"])
    if int(m.get(p + "nextn_predict_layers", 0)):
        raise ValueError("GGUF with an MTP (nextn) block: not supported by this converter")
    recurrent = m[p + "attention.recurrent_layers"]
    if len(recurrent) != n_layers:
        raise ValueError("attention.recurrent_layers does not match block_count")
    mask = sum(1 << i for i, rec in enumerate(recurrent) if not rec)
    head_dim = int(m[p + "attention.key_length"])
    if int(m[p + "attention.value_length"]) != head_dim:
        raise ValueError("key_length != value_length")
    lv_heads = int(m[p + "ssm.time_step_rank"])
    inner = int(m[p + "ssm.inner_size"])
    if inner % lv_heads:
        raise ValueError("ssm.inner_size is not a multiple of ssm.time_step_rank")
    vocab = g.tensors["token_embd.weight"].shape[0]
    d = Dims(hidden=int(m[p + "embedding_length"]), n_layers=n_layers, vocab=vocab,
             ffn=int(m[p + "feed_forward_length"]), n_heads=int(m[p + "attention.head_count"]),
             n_kv=int(m[p + "attention.head_count_kv"]), head_dim=head_dim,
             rot=int(m[p + "rope.dimension_count"]), lk_heads=int(m[p + "ssm.group_count"]),
             lv_heads=lv_heads, lk_dim=int(m[p + "ssm.state_size"]), lv_dim=inner // lv_heads,
             conv_k=int(m[p + "ssm.conv_kernel"]), full_mask=mask,
             eos=int(m["tokenizer.ggml.eos_token_id"]),
             eps=float(m[p + "attention.layer_norm_rms_epsilon"]),
             theta=float(m[p + "rope.freq_base"]))
    if d.lv_heads % d.lk_heads:
        raise ValueError("V heads are not a multiple of K heads")
    return d


def grouped_from_tiled(d: Dims) -> np.ndarray:
    """perm[h] = tiled position of HF (grouped) V head h: h = g * r + i  <-  i * n_k + g."""
    h = np.arange(d.lv_heads)
    return (h % d.rep) * d.lk_heads + h // d.rep


def head_rows(perm: np.ndarray, width: int) -> np.ndarray:
    """Row (or channel) permutation for a per-head permutation, `width` rows per head."""
    return (perm[:, None] * width + np.arange(width)[None, :]).reshape(-1)


# ------------------------------------------------------------------------------ name map
def plan(g: Gguf, d: Dims) -> list[tuple[str, str, str]]:
    """[(pq name, gguf name, transform)] in the .pq file's order."""
    out = [("embed_tokens.weight", "token_embd.weight", "copy"),
           ("norm.weight", "output_norm.weight", "copy")]
    if "output.weight" in g.tensors and not np.array_equal(g.raw("output.weight"),
                                                           g.raw("token_embd.weight")):
        out.append(("lm_head.weight", "output.weight", "copy"))
    for i in range(d.n_layers):
        b, L = f"blk.{i}.", f"layers.{i}."
        out += [(L + "input_layernorm.weight", b + "attn_norm.weight", "copy"),
                (L + "post_attention_layernorm.weight", b + "post_attention_norm.weight", "copy"),
                (L + "mlp.gate_proj.weight", b + "ffn_gate.weight", "copy"),
                (L + "mlp.up_proj.weight", b + "ffn_up.weight", "copy"),
                (L + "mlp.down_proj.weight", b + "ffn_down.weight", "copy")]
        if (d.full_mask >> i) & 1:
            out += [(L + "self_attn.q_proj.weight", b + "attn_q.weight", "copy"),
                    (L + "self_attn.k_proj.weight", b + "attn_k.weight", "copy"),
                    (L + "self_attn.v_proj.weight", b + "attn_v.weight", "copy"),
                    (L + "self_attn.o_proj.weight", b + "attn_output.weight", "copy"),
                    (L + "self_attn.q_norm.weight", b + "attn_q_norm.weight", "copy"),
                    (L + "self_attn.k_norm.weight", b + "attn_k_norm.weight", "copy")]
        else:
            out += [(L + "linear_attn.in_proj_qkv.weight", b + "attn_qkv.weight", "rows_qkv"),
                    (L + "linear_attn.in_proj_z.weight", b + "attn_gate.weight", "rows_v"),
                    (L + "linear_attn.in_proj_b.weight", b + "ssm_beta.weight", "rows_head"),
                    (L + "linear_attn.in_proj_a.weight", b + "ssm_alpha.weight", "rows_head"),
                    (L + "linear_attn.out_proj.weight", b + "ssm_out.weight", "cols_v"),
                    (L + "linear_attn.A", b + "ssm_a", "rows_head"),
                    (L + "linear_attn.dt_bias", b + "ssm_dt.bias", "rows_head"),
                    (L + "linear_attn.conv1d.weight", b + "ssm_conv1d.weight", "rows_qkv"),
                    (L + "linear_attn.norm.weight", b + "ssm_norm.weight", "copy")]
    used = {gn for _, gn, _ in out} | {"output.weight"}
    left = sorted(set(g.tensors) - used)
    if left:
        raise ValueError(f"GGUF tensors not mapped: {left[:8]}{' ...' if len(left) > 8 else ''}")
    return out


def row_perm(d: Dims, transform: str, rows: int) -> np.ndarray | None:
    """Source row of every destination row (None: identity)."""
    perm = grouped_from_tiled(d)
    if transform == "rows_qkv":                 # q | k rows unchanged, V rows per head
        qk = 2 * d.lk_heads * d.lk_dim
        if rows != qk + d.lv_heads * d.lv_dim:
            raise ValueError(f"qkv rows {rows} != {qk} + {d.lv_heads * d.lv_dim}")
        return np.concatenate([np.arange(qk), qk + head_rows(perm, d.lv_dim)])
    if transform == "rows_v":
        return head_rows(perm, d.lv_dim)
    if transform == "rows_head":
        return perm
    return None


def col_block_perm(d: Dims, transform: str, n_blocks: int) -> np.ndarray | None:
    """Source 32-column block of every destination block (out_proj input = V heads)."""
    if transform != "cols_v":
        return None
    if d.lv_dim % 32 or n_blocks != d.lv_heads * d.lv_dim // 32:
        raise ValueError("out_proj columns are not whole 32-column blocks per V head")
    return head_rows(grouped_from_tiled(d), d.lv_dim // 32)


# ------------------------------------------------------------------------------ repacking
def to_tiles(blocks: np.ndarray, dtype: int) -> bytes:
    """[rows, nb, 2 + code] ggml blocks -> pinfer Q*_G32 superblocks (FORMAT.md tiling):
    tile t = rows 4t..4t+3; superblock sb = blocks 8sb..8sb+7; scale line = fp16 d[k][i] at
    4k + i; then block k, row i: the block's code bytes."""
    rows, nb, bb = blocks.shape
    if rows % 4 or nb % 8:
        raise ValueError(f"shape {rows} x {nb * 32}: rows must be a multiple of 4, cols of 256")
    t = blocks.reshape(rows // 4, 4, nb // 8, 8, bb).transpose(0, 2, 3, 1, 4)  # [T, sb, k, i, bb]
    scale = np.ascontiguousarray(t[..., :2]).reshape(rows // 4, nb // 8, 64)
    codes = np.ascontiguousarray(t[..., 2:]).reshape(rows // 4, nb // 8, 32 * (bb - 2))
    return np.concatenate([scale, codes], axis=2).tobytes()


def f32_converted(g: Gguf, d: Dims, gname: str, transform: str) -> np.ndarray:
    """An F32 tensor in .pq order, as [rows, cols] (vectors: [n, 1])."""
    shape = g.tensors[gname].shape
    x = g.f32(gname).reshape(shape if len(shape) > 1 else (shape[0], 1))
    p = row_perm(d, transform, x.shape[0])
    if p is not None:
        x = x[p]
    if transform == "cols_v":            # an F32 out_proj: whole V heads of columns
        x = x[:, head_rows(grouped_from_tiled(d), d.lv_dim)]
    return np.ascontiguousarray(x)


def convert_tensor(g: Gguf, d: Dims, gname: str, transform: str) -> tuple[int, list[int], bytes]:
    t = g.tensors[gname]
    shape = t.shape
    if t.ggml_type == GGML_F32:
        return F32, shape, f32_converted(g, d, gname, transform).tobytes()
    if t.ggml_type not in PQ_OF_GGML:
        raise ValueError(f"{gname}: ggml type {GGML_NAMES.get(t.ggml_type, t.ggml_type)} has no "
                         "lossless pinfer dtype")
    if len(shape) != 2:
        raise ValueError(f"{gname}: quantized tensor with {len(shape)} dims")
    blocks = g.blocks(gname)
    p = row_perm(d, transform, shape[0])
    if p is not None:
        blocks = blocks[p]
    c = col_block_perm(d, transform, blocks.shape[1])
    if c is not None:
        blocks = blocks[:, c]
    dtype = PQ_OF_GGML[t.ggml_type]
    return dtype, shape, to_tiles(np.ascontiguousarray(blocks), dtype)


def header_bytes(d: Dims, n_tensors: int, dir_off: int, data_off: int, tied: bool) -> bytes:
    fields = (MAGIC, VERSION, n_tensors, dir_off, data_off, 1, d.hidden, d.n_layers, d.vocab,
              d.ffn, d.n_heads, d.n_kv, d.head_dim, d.rot, d.lk_heads, d.lv_heads, d.lk_dim,
              d.lv_dim, d.conv_k, d.full_mask, d.eos, int(tied), 0, d.eps, d.theta)
    return struct.pack(HEADER_FMT, *fields).ljust(HEADER_BYTES, b"\0")


def pq_entry(t: GgufTensor) -> tuple[int, int]:
    """(pq dtype, nbytes) of a GGUF tensor after conversion."""
    if t.ggml_type == GGML_F32:
        return F32, t.nbytes
    if t.ggml_type not in PQ_OF_GGML:
        raise ValueError(f"{t.name}: ggml type {GGML_NAMES.get(t.ggml_type, t.ggml_type)} has no "
                         "lossless pinfer dtype")
    dtype = PQ_OF_GGML[t.ggml_type]
    rows, cols = t.shape
    return dtype, rows // 4 * (cols // 256) * (64 + 32 * CODE_BYTES[dtype])


def convert(gguf_path: Path, out: Path) -> dict:
    g = Gguf(gguf_path)
    d = read_dims(g)
    items = plan(g, d)
    tied = not any(n == "lm_head.weight" for n, _, _ in items)
    dir_off = HEADER_BYTES
    data_off = -(-(dir_off + 128 * len(items)) // 4096) * 4096
    entries, off = [], data_off
    for name, gname, _ in items:
        if len(name.encode()) >= 80:
            raise ValueError(f"name too long: {name}")
        t = g.tensors[gname]
        dtype, nbytes = pq_entry(t)
        entries.append((name, dtype, t.shape, off, nbytes))
        off = -(-(off + nbytes) // ALIGN) * ALIGN
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(header_bytes(d, len(items), dir_off, data_off, tied))
        for name, dtype, shape, o, n in entries:
            dims = list(shape) + [0] * (4 - len(shape))
            f.write(struct.pack(ENTRY_FMT, name.encode(), dtype, len(shape), *dims, o, n))
        for (name, gname, transform), (_, dtype, _, o, n) in zip(items, entries):
            got_dtype, _, data = convert_tensor(g, d, gname, transform)
            if got_dtype != dtype or len(data) != n:
                raise AssertionError(f"{name}: {got_dtype}/{len(data)} != {dtype}/{n}")
            f.write(b"\0" * (o - f.tell()))
            f.write(data)
            logger.debug("%s <- %s (%s)", name, gname, transform)
    size = out.stat().st_size
    logger.info("wrote %s: %d tensors, %.1f MB, tied=%s", out, len(items), size / 1e6, tied)
    return {"tensors": len(items), "tied": tied, "dims": d.__dict__, "bytes": size}


# ------------------------------------------------------------------------------ verification
def read_pq(path: Path) -> tuple[np.memmap, dict, dict[str, tuple[int, list[int], int, int]]]:
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    hdr = struct.unpack_from(HEADER_FMT, mm, 0)
    names = ["magic", "version", "n_tensors", "dir_offset", "data_offset", "arch", "hidden",
             "n_layers", "vocab", "ffn", "n_heads", "n_kv_heads", "head_dim", "rot_dim",
             "lin_k_heads", "lin_v_heads", "lin_k_dim", "lin_v_dim", "conv_k", "full_attn_mask",
             "eos_id", "tied_embeddings", "has_mtp", "rms_eps", "rope_theta"]
    h = dict(zip(names, hdr))
    ents = {}
    for i in range(h["n_tensors"]):
        e = struct.unpack_from(ENTRY_FMT, mm, h["dir_offset"] + 128 * i)
        name = e[0].split(b"\0")[0].decode()
        ents[name] = (e[1], list(e[3:3 + e[2]]), e[7], e[8])
    return mm, h, ents


def unpack_pq(buf: np.ndarray, dtype: int, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Packed Q4_G32 / Q8_G32 -> (fp16 scale bits [rows, cols/32] uint16, codes): Q4 codes are
    u in 0..15 [rows, cols], Q8 codes int8 [rows, cols]. Follows FORMAT.md index by index."""
    cb = CODE_BYTES[dtype]
    sb = buf.reshape(rows // 4, cols // 256, 64 + 32 * cb)
    d4 = sb[..., :64].copy().view("<u2").reshape(rows // 4, cols // 256, 8, 4)  # [T, sb, k, i]
    d = d4.transpose(0, 3, 1, 2).reshape(rows, cols // 32)
    c = sb[..., 64:].reshape(rows // 4, cols // 256, 8, 4, cb).transpose(0, 3, 1, 2, 4)
    c = c.reshape(rows, cols // 32, cb)
    if dtype == Q8_G32:
        return d, c.copy().view(np.int8).reshape(rows, cols)
    u = np.concatenate([c & 15, c >> 4], axis=2)           # byte j: col j (low), j + 16 (high)
    return d, u.reshape(rows, cols)


def unpack_ggml(blocks: np.ndarray, gtype: int) -> tuple[np.ndarray, np.ndarray]:
    rows, nb, _ = blocks.shape
    d = blocks[..., :2].copy().view("<u2").reshape(rows, nb)
    q = blocks[..., 2:]
    if gtype == GGML_Q8_0:
        return d, q.copy().view(np.int8).reshape(rows, nb * 32)
    u = np.concatenate([q & 15, q >> 4], axis=2)   # block_q4_0: qs[j] = x[j] | x[j+16] << 4
    return d, u.reshape(rows, nb * 32)


def dequant(d16: np.ndarray, codes: np.ndarray, dtype: int) -> np.ndarray:
    rows, cols = codes.shape
    v = codes.astype(np.float32) - (8.0 if dtype == Q4_G32 else 0.0)
    return (v.reshape(rows, cols // 32, 32) * d16.view(np.float16).astype(np.float32)[..., None]
            ).reshape(rows, cols)


def verify(gguf_path: Path, pq_path: Path, rows_per_chunk: int = 4096) -> dict:
    """Bit-exact check of every tensor; returns a summary (raises AssertionError on a mismatch)."""
    g = Gguf(gguf_path)
    d = read_dims(g)
    mm, h, ents = read_pq(pq_path)
    items = plan(g, d)
    assert set(ents) == {n for n, _, _ in items}, "tensor sets differ"
    assert h["tied_embeddings"] == int(not any(n == "lm_head.weight" for n, _, _ in items))
    stats = {"quantized_tensors": 0, "quantized_weights": 0, "f32_tensors": 0, "f32_values": 0,
             "permuted_tensors": 0, "repack_roundtrip_bytes": 0}
    for name, gname, transform in items:
        dtype, shape, off, nbytes = ents[name]
        t = g.tensors[gname]
        assert shape == t.shape, (name, shape, t.shape)
        perm = row_perm(d, transform, shape[0])
        if transform != "copy":
            stats["permuted_tensors"] += 1
        if t.ggml_type == GGML_F32:
            assert dtype == F32
            src = g.f32(gname).reshape(shape[0], -1)
            src = src if perm is None else src[perm]
            cperm = (col_block_perm(d, transform, src.shape[1] // 32) if transform == "cols_v"
                     else None)
            if cperm is not None:            # whole 32-column blocks, as for quantized tensors
                src = src.reshape(src.shape[0], -1, 32)[:, cperm].reshape(src.shape)
            got = mm[off:off + nbytes].view("<f4").reshape(src.shape)
            assert np.array_equal(src.view(np.uint32), got.view(np.uint32)), name
            stats["f32_tensors"] += 1
            stats["f32_values"] += src.size
            continue
        assert dtype == PQ_OF_GGML[t.ggml_type], name
        rows, cols = shape
        blocks = g.blocks(gname)
        cperm = col_block_perm(d, transform, cols // 32)
        dq_max = 0.0
        for r0 in range(0, rows, rows_per_chunk):
            r1 = min(rows, r0 + rows_per_chunk)
            src_rows = np.arange(r0, r1) if perm is None else perm[r0:r1]
            b = blocks[src_rows]
            if cperm is not None:
                b = b[:, cperm]
            gd, gq = unpack_ggml(b, t.ggml_type)
            sbb = 64 + 32 * CODE_BYTES[dtype]
            lo, hi = off + r0 // 4 * (cols // 256) * sbb, off + r1 // 4 * (cols // 256) * sbb
            pd, pq_ = unpack_pq(mm[lo:hi], dtype, r1 - r0, cols)
            assert np.array_equal(gd, pd), f"{name}: scales differ in rows {r0}..{r1}"
            assert np.array_equal(gq, pq_), f"{name}: codes differ in rows {r0}..{r1}"
            wg, wp = dequant(gd, gq, dtype), dequant(pd, pq_, dtype)
            assert np.array_equal(wg.view(np.uint32), wp.view(np.uint32)), f"{name}: weights differ"
            dq_max = max(dq_max, float(np.abs(wg).max()))
            # the project's reference packer (tools/quant.py) rebuilds the same bytes
            q_signed = (pq_.astype(np.int16) - 8) if dtype == Q4_G32 else pq_
            packed = (pack_q4 if dtype == Q4_G32 else pack_q8)(q_signed, pd.view(np.float16))
            assert packed == bytes(mm[lo:hi]), f"{name}: reference packer disagrees"
            stats["repack_roundtrip_bytes"] += len(packed)
        stats["quantized_tensors"] += 1
        stats["quantized_weights"] += rows * cols
        logger.debug("%s ok (max |w| %.4g)", name, dq_max)
    stats["tensors"] = len(items)
    return stats


def copy_tokenizer(src: Path, out: Path) -> Path:
    dest = out.with_suffix(".tokenizer")
    dest.mkdir(parents=True, exist_ok=True)
    for f in SIDECAR_FILES:
        if (src / f).is_file():
            shutil.copy2(src / f, dest / f)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("gguf", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--tokenizer", type=Path, help="HF model dir: copy the tokenizer files and "
                   "decider_config.json to <out>.tokenizer/")
    p.add_argument("--verify", action="store_true", help="check the written file bit for bit")
    p.add_argument("--verify-only", action="store_true", help="do not convert, only verify `out`")
    p.add_argument("--json", type=Path, help="write the conversion / verification summary here")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary: dict = {"gguf": str(a.gguf), "pq": str(a.out)}
    if not a.verify_only:
        summary["convert"] = convert(a.gguf, a.out)
        if a.tokenizer:
            summary["tokenizer"] = str(copy_tokenizer(a.tokenizer, a.out))
    if a.verify or a.verify_only:
        summary["verify"] = verify(a.gguf, a.out)
        logger.info("verify: OK %s", json.dumps(summary["verify"]))
    if a.json:
        a.json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
