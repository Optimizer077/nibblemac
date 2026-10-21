"""Context snapshots for :meth:`pinfer.Model.save_state` / :meth:`pinfer.Model.load_state`.

Blob layout: header ``<8sIIQ`` (magic ``PINFERPY``, tokens in context, 1 if the last logits
are included, size of the C state), then the token ids (int32), the last logits (float32
[n_vocab], optional) and the C library's ``pinfer_state_save`` blob (engine/src/state.c).

The C blob starts with its own 72-byte header (:func:`c_header`): magic ``PINFERS``, format
version (:data:`C_VERSION`, 2 since pinfer 0.2.0), tokens, kv_dtype, MTP rows, the model
dimensions and a 64-bit fingerprint of the ``.pq`` file it was saved with (its header, tensor
directory and sampled weights). The library refuses a blob of another model file, KV format
or format version with ``PINFER_E_FORMAT``; :func:`load` then names what differs.
"""
from __future__ import annotations

import ctypes
import struct
from typing import TYPE_CHECKING

import numpy as np

from . import errors
from ._lib import KV_DTYPES

if TYPE_CHECKING:
    from .model import Model

MAGIC = b"PINFERPY"
_HDR = "<8sIIQ"
C_MAGIC = b"PINFERS\0"
C_VERSION = 2
# state_hdr: magic, version, n_past, kv_dtype, mtp_rows, n_layers, hidden, n_kv, head_dim,
# lk_heads, lv_heads, lk_dim, lv_dim, conv_k, full_mask, model_fp
_C_HDR = "<8s14IQ"
_C_FIELDS = ("magic", "version", "n_past", "kv_dtype", "mtp_rows", "n_layers", "hidden",
             "n_kv", "head_dim", "lk_heads", "lv_heads", "lk_dim", "lv_dim", "conv_k",
             "full_mask", "model_fp")


def save(model: Model) -> bytes:
    """Snapshot ``model``'s context (see :meth:`pinfer.Model.save_state`)."""
    lib = model._lib
    if lib.pinfer_state_size is None or lib.pinfer_state_save is None:
        raise NotImplementedError("save_state needs a libpinfer with pinfer_state_save")
    h = model._handle()
    n = int(lib.pinfer_state_size(h))
    buf = ctypes.create_string_buffer(max(n, 1))
    rc = lib.pinfer_state_save(h, buf, n)
    if rc != errors.OK:
        raise errors.from_code(rc, "pinfer_state_save")
    toks = np.asarray(model._tokens, dtype=np.int32).tobytes()
    logits = model._logits.tobytes() if model._logits_valid else b""
    head = struct.pack(_HDR, MAGIC, len(model._tokens), int(model._logits_valid), n)
    return head + toks + logits + buf.raw[:n]


def split(data: bytes, n_vocab: int) -> tuple[list[int], np.ndarray | None, bytes]:
    """Parts of a :func:`save` blob: (token ids, last logits or None, C state blob).

    Raises:
        ValueError: Not a pinfer state blob, or truncated.
    """
    hs = struct.calcsize(_HDR)
    if len(data) < hs or data[:8] != MAGIC:
        raise ValueError("not a pinfer state blob")
    _, n_tok, has_logits, n_state = struct.unpack(_HDR, data[:hs])
    off = hs
    if len(data) < off + 4 * n_tok + (4 * n_vocab if has_logits else 0):
        raise ValueError("truncated pinfer state blob")
    toks = np.frombuffer(data, dtype=np.int32, count=n_tok, offset=off).tolist()
    off += 4 * n_tok
    logits = None
    if has_logits:
        logits = np.frombuffer(data, dtype=np.float32, count=n_vocab, offset=off).copy()
        off += 4 * n_vocab
    blob = data[off:off + n_state]
    if len(blob) != n_state:
        raise ValueError("truncated pinfer state blob")
    return toks, logits, blob


def c_header(blob: bytes) -> dict[str, int] | None:
    """Header fields of a C state blob (``model_fp`` = file fingerprint), or None if the blob
    is too short or not a pinfer C state."""
    n = struct.calcsize(_C_HDR)
    if len(blob) < n or blob[:8] != C_MAGIC:
        return None
    values = struct.unpack(_C_HDR, blob[:n])
    return {k: int(v) for k, v in zip(_C_FIELDS[1:], values[1:])}


def _refusal(model: Model, blob: bytes) -> str:
    """Why pinfer_state_load returned PINFER_E_FORMAT for ``blob`` (best effort)."""
    hdr = c_header(blob)
    if hdr is None:
        return "not a pinfer C state"
    if hdr["version"] != C_VERSION:
        return (f"state format version {hdr['version']}; pinfer {model._lib.version} reads "
                f"version {C_VERSION} (save the state again with this version)")
    names = {v: k for k, v in KV_DTYPES.items()}
    if names.get(hdr["kv_dtype"], hdr["kv_dtype"]) != model.kv_dtype:
        return (f"saved with kv_dtype {names.get(hdr['kv_dtype'], hdr['kv_dtype'])}; this "
                f"model uses kv_dtype {model.kv_dtype}")
    return (f"saved with another model file (fingerprint {hdr['model_fp']:#018x}), or the "
            "blob is truncated")


def load(model: Model, data: bytes) -> None:
    """Restore a snapshot into ``model`` (see :meth:`pinfer.Model.load_state`)."""
    lib = model._lib
    if lib.pinfer_state_load is None:
        raise NotImplementedError("load_state needs a libpinfer with pinfer_state_load")
    toks, logits, blob = split(data, model.n_vocab)
    rc = lib.pinfer_state_load(model._handle(), blob, len(blob))
    if rc == errors.E_FORMAT:
        raise errors.ModelFormatError(f"pinfer_state_load: {_refusal(model, blob)} "
                                      f"(code {rc})", rc)
    if rc != errors.OK:
        raise errors.from_code(rc, "pinfer_state_load")
    model._tokens = toks
    if logits is not None:
        model._logits[:] = logits
    model._logits_valid = logits is not None
