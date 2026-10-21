"""Speculative greedy decoding through ``pinfer_generate_greedy`` (MTP drafts, optional API).

The C call is blocking and does the whole generation (reset, prefill, decode) at once, so a
"stream" of a speculative generation delivers its text only after the call returns. Output
is identical to plain greedy decoding (engine/SPEC.md); only the speed differs.

Gaps of the v0.1 C function that this wrapper works around:
  - it stops only at the model's EOS id, so other stop tokens (Qwen's ``<|im_end|>``) are
    applied afterwards and the tokens generated past them are wasted work;
  - it returns ``PINFER_E_CTX_FULL`` without the tokens generated so far, so ``n_max`` is
    clamped to the free context.
"""
from __future__ import annotations

import ctypes
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from . import errors
from ._lib import SpecParams, SpecStats

if TYPE_CHECKING:
    from .model import Model


def iter_speculative(model: Model, ids: list[int], max_new_tokens: int, stop_ids: set[int],
                     state: dict[str, Any], k: int | None = None,
                     draft_vocab: int | None = None) -> Iterator[int]:
    """Yield the tokens of one speculative greedy generation (see module docstring).

    ``state`` receives the same keys as :meth:`pinfer.Model.iter_tokens` plus ``spec_stats``.
    """
    lib = model._lib
    fn = lib.pinfer_generate_greedy
    if fn is None or not model.has_mtp:
        raise NotImplementedError("speculative decoding needs pinfer_generate_greedy and a "
                                  "model with an MTP head (.pq converted without --no-mtp)")
    if len(ids) > model.max_ctx:
        raise errors.ContextOverflowError(
            f"prompt has {len(ids)} tokens > max_ctx {model.max_ctx}", errors.E_CTX_FULL)
    sp = lib.pinfer_default_spec_params() if lib.pinfer_default_spec_params else SpecParams(3, 0)
    if k is not None:
        sp.k = int(k)
    if draft_vocab is not None:
        sp.draft_vocab = int(draft_vocab)
    n_max = min(max_new_tokens, model.max_ctx - len(ids))
    state.update(cached_tokens=0, prefill_seconds=0.0, decode_seconds=0.0,
                 finish_reason="length", spec_stats=None)
    if n_max <= 0:
        state["finish_reason"] = "context" if max_new_tokens > 0 else "length"
        return
    prompt = (ctypes.c_int32 * len(ids))(*ids)
    out = (ctypes.c_int32 * n_max)()
    stats = SpecStats()
    model._forget_state()  # the C call resets the context itself
    t0 = time.perf_counter()
    n = fn(model._handle(), prompt, len(ids), out, n_max, ctypes.byref(sp), ctypes.byref(stats))
    state["decode_seconds"] = time.perf_counter() - t0
    if n < 0:
        raise errors.from_code(n, "pinfer_generate_greedy")
    state["spec_stats"] = {"k": sp.k, "rounds": stats.rounds, "drafted": stats.drafted,
                           "accepted": stats.accepted}
    tokens = list(out[:n])
    if n == n_max and n_max < max_new_tokens:
        state["finish_reason"] = "context"
    for tok in tokens:
        if tok in stop_ids:
            state["finish_reason"] = "stop"
            return
        yield tok
