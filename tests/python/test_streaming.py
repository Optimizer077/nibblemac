"""Streaming generation, stop strings, early stop."""
from __future__ import annotations

from typing import Any

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
