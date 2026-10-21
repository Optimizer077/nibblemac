"""Greedy generation through the Python API reproduces the fp32 transformers reference.

ref/golden/qwen35-0.8b/meta.json holds the prompt text, its token ids and the 32 greedy ids
of transformers 5.20.2 (fp32 widening of the BF16 checkpoint, as stored in the .pq file).
"""
from __future__ import annotations

from typing import Any


def test_prompt_tokenization(tokenizer: Any, golden: dict[str, Any]) -> None:
    assert tokenizer.encode(golden["prompt"]) == golden["input_ids"]


def test_greedy_text_prompt(fresh: Any, golden: dict[str, Any]) -> None:
    n = len(golden["greedy_ids"])
    r = fresh.generate(golden["prompt"], n, reuse_prefix=False)
    assert list(r.token_ids) == golden["greedy_ids"]
    assert r.text == golden["greedy_text"]
    assert r.finish_reason == "length"
    assert r.prompt_tokens == len(golden["input_ids"]) and r.cached_tokens == 0


def test_greedy_token_api(fresh: Any, golden: dict[str, Any]) -> None:
    ids = list(fresh.iter_tokens(golden["input_ids"], 8, reuse_prefix=False))
    assert ids == golden["greedy_ids"][:8]


def test_prefix_reuse_continues_exactly(fresh: Any, golden: dict[str, Any]) -> None:
    prompt, gold = golden["input_ids"], golden["greedy_ids"]
    first = fresh.generate(prompt, 4)
    assert list(first.token_ids) == gold[:4]
    # The engine consumed prompt + the first 3 generated tokens; extending reuses them.
    second = fresh.generate(prompt + gold[:4], 4)
    assert second.cached_tokens == len(prompt) + 3
    assert list(second.token_ids) == gold[4:8]
