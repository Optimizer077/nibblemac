"""Context snapshots for :meth:`pinfer.Model.save_state` / :meth:`pinfer.Model.load_state`.

Blob layout: header ``<8sIIQ`` (magic ``PINFERPY``, tokens in context, 1 if the last logits
are included, size of the C state), then the token ids (int32), the last logits (float32
[n_vocab], optional) and the C library's ``pinfer_state_save`` blob (engine/src/state.c).
"""
from __future__ import annotations

import ctypes
import struct
from typing import TYPE_CHECKING

import numpy as np

from . import errors

if TYPE_CHECKING:
    from .model import Model

MAGIC = b"PINFERPY"
_HDR = "<8sIIQ"


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


def load(model: Model, data: bytes) -> None:
    """Restore a snapshot into ``model`` (see :meth:`pinfer.Model.load_state`)."""
    lib = model._lib
    if lib.pinfer_state_load is None:
        raise NotImplementedError("load_state needs a libpinfer with pinfer_state_load")
    hs = struct.calcsize(_HDR)
    if len(data) < hs or data[:8] != MAGIC:
        raise ValueError("not a pinfer state blob")
    _, n_tok, has_logits, n_state = struct.unpack(_HDR, data[:hs])
    off = hs
    if len(data) < off + 4 * n_tok + (4 * model.n_vocab if has_logits else 0):
        raise ValueError("truncated pinfer state blob")
    toks = np.frombuffer(data, dtype=np.int32, count=n_tok, offset=off).tolist()
    off += 4 * n_tok
    logits = None
    if has_logits:
        logits = np.frombuffer(data, dtype=np.float32, count=model.n_vocab, offset=off).copy()
        off += 4 * model.n_vocab
    blob = data[off:off + n_state]
    if len(blob) != n_state:
        raise ValueError("truncated pinfer state blob")
    rc = lib.pinfer_state_load(model._handle(), blob, n_state)
    if rc != errors.OK:
        raise errors.from_code(rc, "pinfer_state_load")
    model._tokens = toks
    if logits is not None:
        model._logits[:] = logits
    model._logits_valid = logits is not None
