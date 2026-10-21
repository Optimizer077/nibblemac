"""Context state: save_state / load_state and continuation of speculative decoding."""
from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _need_state(model: Any) -> None:
    if not model.supports_state:
        pytest.skip("libpinfer without pinfer_state_save")


def _c_blob_offset(model: Any, data: bytes) -> int:
    """Byte offset of the C state blob inside a save_state() blob."""
    import pinfer.state

    _, _, blob = pinfer.state.split(data, model.n_vocab)
    return len(data) - len(blob)


@pytest.fixture()
def saved(fresh: Any, golden: dict[str, Any]) -> bytes:
    """save_state() after the first 4 golden prompt tokens."""
    _need_state(fresh)
    fresh.eval(golden["input_ids"][:4])
    return bytes(fresh.save_state())


@pytest.fixture(scope="module")
def same_layout(lib: Any, tmp_path_factory: pytest.TempPathFactory,
                model_path: Path) -> Iterator[Any]:
    """A model file with the real header and tensor directory but zero weights (a sparse
    copy of the first data_offset bytes): same dimensions and layout, different weights."""
    import pinfer

    with model_path.open("rb") as f:
        data_offset = struct.unpack_from("<Q", f.read(32), 24)[0]
        f.seek(0)
        head = f.read(data_offset)
    out = tmp_path_factory.mktemp("state") / "same-layout.pq"
    with out.open("wb") as f:
        f.write(head)
        f.truncate(model_path.stat().st_size)
    m = pinfer.Model.load(out, tokenizer=False, n_threads=1, max_ctx=256)
    yield m
    m.close()
    out.unlink()   # sparse, but on a tmpfs /tmp (the Pi) the pages it touched are real


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


def test_state_header_carries_version_and_fingerprint(fresh: Any, saved: bytes) -> None:
    import pinfer.state

    hdr = pinfer.state.c_header(saved[_c_blob_offset(fresh, saved):])
    assert hdr is not None
    assert hdr["version"] == pinfer.state.C_VERSION == 2
    assert hdr["n_past"] == 4 and hdr["kv_dtype"] == 0 and hdr["model_fp"] != 0
    again = bytes(fresh.save_state())
    hdr2 = pinfer.state.c_header(again[_c_blob_offset(fresh, again):])
    assert hdr2 is not None and hdr2["model_fp"] == hdr["model_fp"]   # deterministic


def test_load_state_rejects_same_layout_other_weights(saved: bytes, same_layout: Any) -> None:
    """Same dimensions, header and directory, other weights: the version-1 format (dimension
    check only) accepted this; the fingerprint samples the weights too."""
    import pinfer

    with pytest.raises(pinfer.ModelFormatError, match="another model file"):
        same_layout.load_state(saved)
    assert same_layout.n_past == 0   # unchanged on failure


def test_load_state_rejects_flipped_fingerprint(fresh: Any, saved: bytes) -> None:
    import pinfer

    b = bytearray(saved)
    b[_c_blob_offset(fresh, saved) + 64] ^= 1   # state_hdr.model_fp (engine/src/state.c)
    with pytest.raises(pinfer.ModelFormatError, match="another model file"):
        fresh.load_state(bytes(b))


def test_load_state_rejects_version_1(fresh: Any, saved: bytes) -> None:
    import pinfer

    b = bytearray(saved)
    struct.pack_into("<I", b, _c_blob_offset(fresh, saved) + 8, 1)
    with pytest.raises(pinfer.ModelFormatError, match="state format version 1"):
        fresh.load_state(bytes(b))


def test_load_state_rejects_other_kv_dtype(lib: Any, model_path: Path, saved: bytes) -> None:
    import pinfer

    with (pinfer.Model.load(model_path, tokenizer=False, n_threads=1, max_ctx=64,
                            kv_dtype="fp16") as m,
          pytest.raises(pinfer.ModelFormatError, match="kv_dtype fp32")):
        m.load_state(saved)
