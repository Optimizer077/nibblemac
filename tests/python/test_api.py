"""Model API: loading, properties, error handling, context limits."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pinfer
import pytest
from pinfer import errors


def test_version_matches_library(lib: Any) -> None:
    assert lib.version == pinfer.__version__


def test_library_exports_v01_api(lib: Any) -> None:
    for name in ("pinfer_load", "pinfer_eval", "pinfer_reset", "pinfer_free", "pinfer_argmax"):
        assert lib.has(name)
    assert not lib.has("pf_forward")  # internals stay hidden


def test_missing_library_path() -> None:
    with pytest.raises(pinfer.LibraryNotFoundError):
        pinfer.load_library("/nonexistent/libpinfer.so")


def test_missing_model(lib: Any, tmp_path: Path) -> None:
    with pytest.raises(pinfer.ModelNotFoundError) as ei:
        pinfer.Model.load(tmp_path / "nope.pq", tokenizer=False)
    assert isinstance(ei.value, FileNotFoundError)
    with pytest.raises(pinfer.ModelNotFoundError):
        pinfer.Model.load(tmp_path, tokenizer=False)  # a directory


@pytest.mark.parametrize("content", [b"tiny", b"\0" * 8192, b"PINFERQ\0" + b"\xff" * 8184])
def test_bad_model_file(lib: Any, tmp_path: Path, content: bytes) -> None:
    bad = tmp_path / "bad.pq"
    bad.write_bytes(content)
    with pytest.raises(pinfer.ModelFormatError) as ei:
        pinfer.Model.load(bad, tokenizer=False)
    assert ei.value.code == errors.E_FORMAT


@pytest.mark.parametrize("kwargs", [{"n_threads": 0}, {"max_ctx": 0}])
def test_bad_load_arguments(model_path: Path, kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        pinfer.Model.load(model_path, tokenizer=False, **kwargs)


def test_properties(fresh: Any) -> None:
    assert fresh.n_vocab == 248320
    assert fresh.eos_id == 248044
    assert fresh.n_past == 0
    info = fresh.info()
    assert info["version"] == pinfer.__version__ and info["n_vocab"] == 248320
    assert info["build"].get("matvec") in ("ref", "fast")


def test_eval_and_reset(fresh: Any, golden: dict[str, Any]) -> None:
    ids = golden["input_ids"]
    logits = fresh.eval(ids)
    assert logits.shape == (fresh.n_vocab,)
    assert int(logits.argmax()) == golden["greedy_ids"][0]
    assert fresh.n_past == len(ids)
    fresh.reset()
    assert fresh.n_past == 0


def test_invalid_token_leaves_state(fresh: Any) -> None:
    fresh.eval([760])
    with pytest.raises(pinfer.PinferError) as ei:
        fresh.eval([fresh.n_vocab])
    assert ei.value.code == errors.E_ARG
    assert fresh.n_past == 1
    with pytest.raises(ValueError):
        fresh.eval([])


def test_context_overflow(lib: Any, model_path: Path, tokenizer: Any,
                          golden: dict[str, Any]) -> None:
    prompt, gold = golden["input_ids"], golden["greedy_ids"]
    with pinfer.Model.load(model_path, tokenizer=tokenizer, max_ctx=8) as small:
        with pytest.raises(pinfer.ContextOverflowError):
            small.generate(prompt, 4)
        small.eval(prompt[:8])
        with pytest.raises(pinfer.ContextOverflowError) as ei:
            small.eval([prompt[8]])
        assert small.n_past == 8 and ei.value.code == errors.E_CTX_FULL
    with pinfer.Model.load(model_path, tokenizer=tokenizer, max_ctx=len(prompt) + 2) as m:
        r = m.generate(prompt, 32)
        assert r.finish_reason == "context"
        assert list(r.token_ids) == gold[:3]


def test_closed_model(lib: Any, model_path: Path) -> None:
    m = pinfer.Model.load(model_path, tokenizer=False)
    m.close()
    m.close()  # idempotent
    with pytest.raises(pinfer.ModelClosedError):
        m.eval([1])


def test_no_tokenizer(lib: Any, model_path: Path, golden: dict[str, Any]) -> None:
    with pinfer.Model.load(model_path, tokenizer=False) as m:
        with pytest.raises(pinfer.TokenizerNotFoundError):
            m.generate("hello", 1)
        ids = list(m.iter_tokens(golden["input_ids"], 1))
        assert ids == golden["greedy_ids"][:1]


def test_speculative_matches_greedy(fresh: Any, golden: dict[str, Any]) -> None:
    if not fresh.supports_speculative:
        with pytest.raises(NotImplementedError):
            fresh.generate("hi", 1, speculative=True)
        pytest.skip("library or model without speculative decoding")
    r = fresh.generate(golden["prompt"], 32, speculative=True)
    assert list(r.token_ids) == golden["greedy_ids"]
    assert r.text == golden["greedy_text"]
    assert r.spec_stats is not None and r.spec_stats["rounds"] > 0
    plain_k0 = fresh.generate(golden["prompt"], 8, speculative=True, spec_k=0)
    assert list(plain_k0.token_ids) == golden["greedy_ids"][:8]
    # The C call reset the engine; the plain path must notice and still be exact.
    after = fresh.generate(golden["prompt"], 2)
    assert list(after.token_ids) == golden["greedy_ids"][:2] and after.cached_tokens == 0


def test_speculative_context_clamp(lib: Any, model_path: Path, tokenizer: Any,
                                   golden: dict[str, Any]) -> None:
    prompt = golden["input_ids"]
    with pinfer.Model.load(model_path, tokenizer=tokenizer, max_ctx=len(prompt) + 3) as m:
        if not m.supports_speculative:
            pytest.skip("no speculative decoding")
        r = m.generate(prompt, 32, speculative=True)
        assert r.finish_reason == "context"
        assert list(r.token_ids) == golden["greedy_ids"][:3]


def test_speculative_rejects_sampling(fresh: Any) -> None:
    with pytest.raises(ValueError):
        fresh.generate("hi", 1, speculative=True, temperature=0.7)


@pytest.mark.parametrize("kwargs", [{"temperature": -1}, {"top_p": 0}, {"top_k": -1}])
def test_bad_sampling_arguments(fresh: Any, kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        fresh.generate("hi", 1, **kwargs)
