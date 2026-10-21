"""Next-token selection from logits (numpy): greedy, temperature, top-k, top-p."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingParams:
    """How to pick the next token.

    Attributes:
        temperature: 0 (default) means greedy argmax; > 0 samples from softmax(logits / T).
        top_k: Keep only the k most likely tokens (0 = no limit).
        top_p: Keep the smallest set of tokens whose probability mass reaches top_p (1 = all).
        seed: Seed of the random generator (None = nondeterministic).
    """

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    @property
    def greedy(self) -> bool:
        """True when sampling reduces to argmax."""
        return self.temperature == 0.0 or self.top_k == 1


class Sampler:
    """Stateful sampler (owns the random generator so a seed gives a reproducible sequence)."""

    def __init__(self, params: SamplingParams) -> None:
        self.params = params
        self._rng = np.random.default_rng(params.seed)

    def __call__(self, logits: np.ndarray) -> int:
        """Pick one token id from a 1-D float logits vector."""
        p = self.params
        if p.greedy:
            return int(np.argmax(logits))  # first maximum, same as pinfer_argmax / torch
        x = logits.astype(np.float64, copy=False)
        if p.top_k and p.top_k < x.size:
            idx = np.argpartition(x, -p.top_k)[-p.top_k:]
        else:
            idx = np.arange(x.size)
        z = x[idx] / p.temperature
        order = np.argsort(-z, kind="stable")
        idx, z = idx[order], z[order]
        probs = np.exp(z - z[0])
        probs /= probs.sum()
        if p.top_p < 1.0:
            # keep tokens until the mass *before* them reaches top_p (always keeps the first)
            keep = int(np.searchsorted(np.cumsum(probs), p.top_p) + 1)
            idx, probs = idx[:keep], probs[:keep] / probs[:keep].sum()
        return int(idx[self._rng.choice(idx.size, p=probs)])
