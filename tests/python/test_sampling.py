"""Sampling (numpy only, no model needed)."""
from __future__ import annotations

import numpy as np
import pytest
from pinfer import Sampler, SamplingParams

LOGITS = np.log(np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32))


def test_greedy_is_first_argmax() -> None:
    s = Sampler(SamplingParams())
    assert s(np.array([1.0, 3.0, 3.0, 2.0], dtype=np.float32)) == 1


def test_top_k_one_is_greedy() -> None:
    s = Sampler(SamplingParams(temperature=5.0, top_k=1, seed=0))
    assert all(s(LOGITS) == 0 for _ in range(50))


def test_tiny_top_p_is_greedy() -> None:
    s = Sampler(SamplingParams(temperature=1.0, top_p=1e-6, seed=0))
    assert all(s(LOGITS) == 0 for _ in range(50))


def test_seed_reproducible() -> None:
    a = Sampler(SamplingParams(temperature=1.0, seed=123))
    b = Sampler(SamplingParams(temperature=1.0, seed=123))
    assert [a(LOGITS) for _ in range(100)] == [b(LOGITS) for _ in range(100)]


def test_distribution_matches_softmax() -> None:
    s = Sampler(SamplingParams(temperature=1.0, seed=7))
    counts = np.bincount([s(LOGITS) for _ in range(20000)], minlength=4) / 20000
    assert np.allclose(counts, [0.5, 0.3, 0.15, 0.05], atol=0.02)


def test_top_k_and_top_p_truncate() -> None:
    s = Sampler(SamplingParams(temperature=1.0, top_k=2, seed=1))
    assert set(s(LOGITS) for _ in range(500)) == {0, 1}
    s = Sampler(SamplingParams(temperature=1.0, top_p=0.7, seed=1))  # 0.5 + 0.3 >= 0.7
    assert set(s(LOGITS) for _ in range(500)) == {0, 1}


@pytest.mark.parametrize("kwargs", [{"temperature": -0.1}, {"temperature": float("nan")},
                                    {"top_p": 0.0}, {"top_p": 1.5}, {"top_k": -2}])
def test_invalid(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
