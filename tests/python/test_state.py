"""Context state: save_state / load_state and continuation of speculative decoding."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _need_state(model: Any) -> None:
    if not model.supports_state:
        pytest.skip("libpinfer without pinfer_state_save")


def test_save_load_roundtrip_is_exact(fresh: Any, golden: dict[str, Any]) -> None:
    _need_state(fresh)
    ids = golden["input_ids"]
    fresh.eval(ids[:4])
    blob = fresh.save_state()
    assert len(blob) > 10_000_000 and fresh.context_tokens == tuple(ids[:4])
    logits_a = fresh.eval(ids[4:])
    gen_a = list(fresh.iter_tokens(ids, max_new_tokens=6))
    fresh.load_state(blob)
    assert fresh.n_past == 4 and fresh.context_tokens == tuple(ids[:4])
    logits_b = fresh.eval(ids[4:])
    assert np.array_equal(logits_a, logits_b)
    gen_b = list(fresh.iter_tokens(ids, max_new_tokens=6))
    assert gen_b == gen_a == golden["greedy_ids"][:6]


def test_save_state_keeps_last_logits(fresh: Any, golden: dict[str, Any]) -> None:
    _need_state(fresh)
    ids = golden["input_ids"]
    logits = fresh.eval(ids)
    blob = fresh.save_state()
    fresh.reset()
    fresh.load_state(blob)
    # same prompt again: nothing to prefill, the saved logits are used (cached_tokens = all)
    state: dict[str, Any] = {}
    gen = list(fresh.iter_tokens(ids, max_new_tokens=3, _state=state))
    assert state["cached_tokens"] == len(ids) and gen == golden["greedy_ids"][:3]
    assert int(np.argmax(logits)) == golden["greedy_ids"][0]


def test_load_state_rejects_garbage(fresh: Any) -> None:
    _need_state(fresh)
    with pytest.raises(ValueError):
        fresh.load_state(b"not a state")
    import pinfer

    with pytest.raises(pinfer.PinferError):
        fresh.load_state(b"PINFERPY" + b"\0" * 16 + b"\0" * 64)


def test_speculative_continues_from_context(fresh: Any, golden: dict[str, Any]) -> None:
    _need_state(fresh)
    if not fresh.supports_speculative or not fresh._lib.has("pinfer_generate_greedy_cont"):
        pytest.skip("no speculative continuation in this library / model")
    ids = golden["input_ids"]
    # reference: the whole thing in one fresh run
    ref = fresh.generate(ids, 10, speculative=True, spec_k=3, reuse_prefix=False)
    assert list(ref.token_ids) == golden["greedy_ids"][:10]
    # the same in two steps: 6 tokens, then continue with the rest of the prompt appended
    first = fresh.generate(ids, 6, speculative=True, spec_k=3)
    assert list(first.token_ids) == golden["greedy_ids"][:6]
    ctx = list(fresh.context_tokens)
    assert ctx == ids + golden["greedy_ids"][:5]   # the last token is emitted, not fed
    second = fresh.generate(ctx + [first.token_ids[-1]], 4, speculative=True, spec_k=3)
    assert second.cached_tokens == len(ctx)
    assert list(second.token_ids) == golden["greedy_ids"][6:10]
    assert fresh.context_tokens == tuple(ids + golden["greedy_ids"][:9])
