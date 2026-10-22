"""Model.eval_select / pinfer_eval_select: logits of selected vocabulary ids at the last position."""
from __future__ import annotations

from typing import Any

import numpy as np
import pinfer
import pytest


def _need_select(model: Any) -> None:
    if model._lib.pinfer_eval_select is None:
        pytest.skip("libpinfer without pinfer_eval_select")


IDS = [0, 1, 2, 3, 5, 6, 7, 101, 102, 4095, 4096, 4097, 248319, 101, 32, 33, 34, 35, 36]


def test_select_equals_full_logits(fresh: Any, golden: dict[str, Any]) -> None:
    _need_select(fresh)
    prompt = golden["input_ids"]
    full = fresh.eval(prompt)
    fresh.reset()
    got = fresh.eval_select(prompt, IDS)
    assert got.dtype == np.float32 and got.shape == (len(IDS),)
    assert np.array_equal(got.view(np.uint32), full[IDS].view(np.uint32))   # bit for bit


def test_select_after_state_restore(fresh: Any, golden: dict[str, Any]) -> None:
    _need_select(fresh)
    prompt = golden["input_ids"]
    full = fresh.eval(prompt)
    fresh.reset()
    fresh.eval_select(prompt[:4], [])                 # feed only
    snap = fresh.save_state()
    first = fresh.eval_select(prompt[4:], IDS)
    fresh.load_state(snap)
    again = fresh.eval_select(prompt[4:], IDS)
    assert np.array_equal(first, full[IDS]) and np.array_equal(again, full[IDS])


def test_select_rejects_bad_ids_without_state_change(fresh: Any, golden: dict[str, Any]) -> None:
    _need_select(fresh)
    fresh.eval_select(golden["input_ids"][:3], [])
    n = fresh.n_past
    for bad in ([-1], [fresh.n_vocab]):
        with pytest.raises(pinfer.PinferError):
            fresh.eval_select(golden["input_ids"][3:5], bad)
        assert fresh.n_past == n
