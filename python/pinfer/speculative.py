"""Speculative greedy decoding with the model's MTP head (optional C API).

Output is identical to plain greedy decoding (engine/SPEC.md); only the speed differs.

With ``pinfer_generate_greedy_ex`` (libpinfer >= 0.1 with the extended API) the C call runs
in a worker thread and reports every final token through a callback, so the generator
yields tokens as they are produced (in bursts of up to k+1), stop tokens end the C call at
once, and a full context returns the tokens made so far. With only the older
``pinfer_generate_greedy`` the call is blocking: tokens arrive after it returns, stop tokens
are applied afterwards (the tokens generated past them are wasted work) and ``n_max`` is
clamped to the free context.

Timing: the C call does reset + prefill + decode as one unit, so ``prefill_seconds`` is the
time to the first token (TTFT) and ``decode_seconds`` the rest.
"""
from __future__ import annotations

import ctypes
import queue
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from . import errors
from ._lib import ON_TOKEN, GenOpts, SpecParams, SpecStats

if TYPE_CHECKING:
    from .model import Model

_DONE = object()


def _spec_params(model: Model, k: int | None, draft_vocab: int | None,
                 adaptive: bool | None = None) -> SpecParams:
    lib = model._lib
    sp = lib.pinfer_default_spec_params() if lib.pinfer_default_spec_params else SpecParams(3, 0, 0)
    if k is not None:
        sp.k = int(k)
    if draft_vocab is not None:
        sp.draft_vocab = int(draft_vocab)
    if adaptive is not None:
        sp.adaptive = int(bool(adaptive))
    return sp


def iter_speculative(model: Model, ids: list[int], max_new_tokens: int, stop_ids: set[int],
                     state: dict[str, Any], k: int | None = None,
                     draft_vocab: int | None = None, adaptive: bool | None = None,
                     reuse_prefix: bool = True) -> Iterator[int]:
    """Yield the tokens of one speculative greedy generation (see module docstring).

    ``state`` receives the same keys as :meth:`pinfer.Model.iter_tokens` plus ``spec_stats``.
    With ``reuse_prefix`` and a library that has ``pinfer_generate_greedy_cont``, a prompt
    that extends the tokens already in context is continued from them (only the new tokens
    are prefilled, ``cached_tokens`` tells how many were reused).
    """
    lib = model._lib
    if (lib.pinfer_generate_greedy is None and lib.pinfer_generate_greedy_ex is None) \
            or not model.has_mtp:
        raise NotImplementedError("speculative decoding needs pinfer_generate_greedy[_ex] and "
                                  "a model with an MTP head (.pq converted without --no-mtp)")
    if len(ids) > model.max_ctx:
        raise errors.ContextOverflowError(
            f"prompt has {len(ids)} tokens > max_ctx {model.max_ctx}", errors.E_CTX_FULL)
    state.update(cached_tokens=0, prefill_seconds=0.0, decode_seconds=0.0,
                 finish_reason="length", spec_stats=None)
    if max_new_tokens <= 0:
        return
    sp = _spec_params(model, k, draft_vocab, adaptive)
    stats = SpecStats()
    # Continue from the context when the prompt extends it (and at least one token is new).
    have = model._tokens
    cont = (lib.pinfer_generate_greedy_cont is not None and reuse_prefix
            and 0 < len(have) < len(ids) and ids[:len(have)] == have)
    new_ids = ids[len(have):] if cont else ids
    prompt = (ctypes.c_int32 * len(new_ids))(*new_ids)
    state["cached_tokens"] = len(have) if cont else 0
    if not cont:
        model._forget_state()  # the C call resets the context itself
    model._logits_valid = False
    try:
        if lib.pinfer_generate_greedy_ex is not None:
            fn = lib.pinfer_generate_greedy_cont if cont else lib.pinfer_generate_greedy_ex
            yield from _iter_ex(model, fn, prompt, new_ids, len(ids), max_new_tokens,
                                stop_ids, state, sp, stats)
        else:
            yield from _iter_blocking(model, prompt, ids, max_new_tokens, stop_ids, state, sp,
                                      stats)
    finally:  # also when the consumer closes the generator early
        state["spec_stats"] = {"k": sp.k, "rounds": stats.rounds, "drafted": stats.drafted,
                               "accepted": stats.accepted}


def _iter_ex(model: Model, fn: Any, prompt: Any, ids: list[int], n_ctx: int,
             max_new_tokens: int, stop_ids: set[int], state: dict[str, Any], sp: SpecParams,
             stats: SpecStats) -> Iterator[int]:
    """Streaming path: C call ``fn`` (reset + prefill, or continuation) in a worker thread,
    tokens handed over through a queue. ``ids`` are the tokens fed by this call, ``n_ctx``
    the context length after its prefill."""
    stops = sorted(stop_ids - {model.eos_id})
    stop_arr = (ctypes.c_int32 * max(1, len(stops)))(*stops)
    n_max = min(max_new_tokens, max(0, model.max_ctx - n_ctx))
    out = (ctypes.c_int32 * max(1, n_max))()
    tokens: queue.Queue[Any] = queue.Queue()
    cancel = threading.Event()
    t0 = time.perf_counter()
    first = [0.0]

    def on_token(tok: int, _user: Any) -> int:
        if not first[0]:
            first[0] = time.perf_counter()
        tokens.put(tok)
        return 1 if cancel.is_set() else 0

    cb = ON_TOKEN(on_token)  # keep a reference: ctypes must not free it during the call
    opts = GenOpts(stop_arr, len(stops), cb, None)
    status = ctypes.c_int(0)
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            result["n"] = fn(
                model._handle(), prompt, len(ids), out, n_max, ctypes.byref(sp),
                ctypes.byref(opts), ctypes.byref(stats), ctypes.byref(status))
        except BaseException as e:  # noqa: BLE001 — every error is re-raised in the consumer thread
            result["error"] = e
        finally:
            tokens.put(_DONE)

    worker = threading.Thread(target=run, name="pinfer-spec", daemon=True)
    worker.start()
    try:
        while True:
            tok = tokens.get()
            if tok is _DONE:
                break
            if tok in stop_ids:
                state["finish_reason"] = "stop"
                continue  # the C side already stopped; drain to _DONE
            yield tok
    finally:
        cancel.set()  # consumer closed early: the next callback ends the C call
        worker.join()
        end = time.perf_counter()
        state["prefill_seconds"] = (first[0] or end) - t0
        state["decode_seconds"] = end - (first[0] or end)
        # Settle the context bookkeeping here, so that it is right after an early close
        # too: the engine fed the prompt and every returned token but the last (emitted,
        # not fed). After an error the context is unknown: the next call starts afresh.
        n_done = result.get("n")
        if "error" in result or not isinstance(n_done, int) or n_done < 0:
            model._forget_state()
        else:
            model._tokens = (model._tokens + list(ids)
                             + [int(t) for t in out[:max(0, n_done - 1)]])
    if "error" in result:
        raise result["error"]
    n = result["n"]
    if n < 0:
        raise errors.from_code(n, "pinfer_generate_greedy_ex")
    if status.value == errors.E_CTX_FULL or (n == n_max and n_max < max_new_tokens):
        state["finish_reason"] = "context"
    elif status.value != 0:
        raise errors.from_code(status.value, "pinfer_generate_greedy_ex")


def _iter_blocking(model: Model, prompt: Any, ids: list[int], max_new_tokens: int,
                   stop_ids: set[int], state: dict[str, Any], sp: SpecParams,
                   stats: SpecStats) -> Iterator[int]:
    """Fallback for libraries without the extended API (see module docstring)."""
    generate = model._lib.pinfer_generate_greedy
    if generate is None:  # iter_speculative takes this path only when the library has it
        raise NotImplementedError("libpinfer has no pinfer_generate_greedy")
    n_max = min(max_new_tokens, model.max_ctx - len(ids))
    if n_max <= 0:
        state["finish_reason"] = "context"
        return
    out = (ctypes.c_int32 * n_max)()
    t0 = time.perf_counter()
    n = generate(model._handle(), prompt, len(ids), out, n_max, ctypes.byref(sp),
                 ctypes.byref(stats))
    state["decode_seconds"] = time.perf_counter() - t0
    if n < 0:
        raise errors.from_code(n, "pinfer_generate_greedy")
    if n == n_max and n_max < max_new_tokens:
        state["finish_reason"] = "context"
    for tok in out[:n]:
        if tok in stop_ids:
            state["finish_reason"] = "stop"
            return
        yield tok
