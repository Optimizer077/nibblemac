"""Streaming generation, stop strings, early stop."""
from __future__ import annotations

import threading
from typing import Any

import pytest
from pinfer.streaming import StopMatcher


def test_stream_equals_blocking(fresh: Any, golden: dict[str, Any]) -> None:
    blocking = fresh.generate(golden["prompt"], 6, reuse_prefix=False)
    stream = fresh.generate(golden["prompt"], 6, stream=True, reuse_prefix=False)
    assert stream.result is None
    chunks = list(stream)
    assert "".join(chunks) == blocking.text
    assert stream.result is not None and stream.result.token_ids == blocking.token_ids
    assert all(chunks)  # no empty chunks


def test_max_new_tokens(fresh: Any, golden: dict[str, Any]) -> None:
    r = fresh.generate(golden["prompt"], 3)
    assert len(r.token_ids) == 3 and r.finish_reason == "length"
    assert fresh.generate(golden["prompt"], 0).token_ids == ()


def test_stop_string(fresh: Any, golden: dict[str, Any]) -> None:
    # greedy text: " can be used for a wide range of tasks. It is ..."
    want = golden["greedy_text"].split("range")[0]
    r = fresh.generate(golden["prompt"], 32, stop=["range", "zzz"])
    assert r.text == want and r.finish_reason == "stop"
    chunks = list(fresh.generate(golden["prompt"], 32, stop="range", stream=True))
    assert "".join(chunks) == want


def test_early_close_keeps_model_usable(fresh: Any, golden: dict[str, Any]) -> None:
    stream = fresh.generate(golden["prompt"], 16, stream=True)
    first = next(stream)
    assert first
    stream.close()
    r = fresh.generate(golden["prompt"], 2, reuse_prefix=False)
    assert list(r.token_ids) == golden["greedy_ids"][:2]


def test_speculative_stream_close_settles_context(fresh: Any, golden: dict[str, Any]) -> None:
    """Closing a speculative stream (a continuation of the context) ends the C call at once
    and leaves context_tokens equal to what the engine holds, so the next prompt that
    extends it continues correctly."""
    if not fresh.supports_speculative or not fresh._lib.has("pinfer_generate_greedy_cont"):
        pytest.skip("no speculative continuation in this library / model")
    ids, want = golden["input_ids"], golden["greedy_ids"]
    fresh.generate(ids[:4], 1, speculative=True)          # context = ids[:4]
    stream = fresh.generate(ids, 12, speculative=True, stream=True)
    assert next(stream)
    stream.close()
    assert not [t for t in threading.enumerate() if t.name == "pinfer-spec"]
    ctx = list(fresh.context_tokens)
    assert len(ctx) == fresh.n_past and ctx[:len(ids)] == ids
    n = len(ctx) - len(ids)
    assert ctx[len(ids):] == want[:n]
    r = fresh.generate(ctx + [want[n]], 2, speculative=True)
    assert r.cached_tokens == len(ctx) and list(r.token_ids) == want[n + 1:n + 3]


def test_stop_matcher_holds_back_prefixes() -> None:
    m = StopMatcher(["</s>"])
    assert m.feed("hello <") == "hello "
    assert m.feed("/") == ""
    assert m.feed("x") == "</x"
    assert m.feed(" bye</s> tail") == " bye"
    assert m.stopped and m.text == "hello </x bye"


def test_stop_matcher_without_stops() -> None:
    m = StopMatcher([])
    assert m.feed("abc") == "abc"
    assert m.finish() == ""
