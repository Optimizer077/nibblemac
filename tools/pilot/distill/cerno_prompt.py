"""Cerno's decision prompt and readout for Hugging Face tokenizers (pilot: distilled decider).

A standalone port of the prompt layout in ``cerno-pi/cerno_pi/decide.py`` (itself adapted from
decider / cerno-inference, Apache-2.0): one row per question, ``"Context:\\n" + state`` then
``"\\n\\nQuestion: ...\\nOptions:\\n(A) ...\\nAnswer: ("``; up to 10 options are lettered A..J as
text, more options get one label token each (A..Z, then two-letter tokens). The readout is
``softmax(label_logits / T)`` with Cerno's temperature (1.534). The row is decoded to text and
re-tokenized, as llama-server receives it (``tokenization="text"``), so the student sees the same
token ids as the teacher.

No engine dependency: only ``transformers`` tokenizers and numpy.
"""
from __future__ import annotations

import json
import math
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)
MAX_OPTIONS = 255
MAX_CHOICE, MAX_LEVELS = 255, 10
ANNOTATE_MIN = 8
NEUTRAL_NONE = "not listed here"
CERNO_TEMPERATURE = 1.534          # apertixai/cerno-4b decider_config.json


def _txt(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def annotate_indices(x: Any, min_len: int = ANNOTATE_MIN) -> Any:
    if isinstance(x, list):
        if len(x) >= min_len:
            return [({"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                     else {"_index": i, "value": annotate_indices(v, min_len)})
                    for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state), ensure_ascii=False)


@dataclass(frozen=True)
class Question:
    type: str
    text: str
    options: tuple[str, ...]
    names: tuple[Any, ...]


def render_question(spec: Mapping[str, Any]) -> Question:
    t = spec.get("type", "choice")
    ins = _txt(spec.get("instructions", spec.get("question", "")))
    crit = spec.get("criteria", spec.get("options"))
    if not ins:
        raise ValueError("question without instructions")
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_CHOICE:
            raise ValueError("choice criteria: a map of 2..255 options")
        names: tuple[Any, ...] = tuple(crit)
        opts = [n if crit[n] in (None, "") else f"{n}: {_txt(crit[n])}" for n in names]
    elif t == "score":
        if isinstance(crit, dict):
            crit = [crit[k] for k in sorted(crit, key=float)]
        if not isinstance(crit, (list, tuple)) or not 2 <= len(crit) <= MAX_LEVELS:
            raise ValueError("score criteria: an ordered list of 2..10 level descriptions")
        names = tuple(range(len(crit)))
        opts = [f"{i}: {_txt(c)}" for i, c in enumerate(crit)]
    elif t in ("noul", "bool"):
        names = (False, True)
        c = crit or {}
        f, tr = c.get("false", c.get(False)), c.get("true", c.get(True))
        opts = ["no" if f in (None, "") else f"no: {_txt(f)}",
                "yes" if tr in (None, "") else f"yes: {_txt(tr)}"]
        t = "noul"
    else:
        raise ValueError(f"unknown question type {t!r}")
    return Question(t, ins, tuple(opts), names)


def neutralize_options(options: Sequence[str]) -> list[str]:
    out = []
    for o in options:
        key = o.strip().lower()
        if key.startswith("none of the above") or key in ("none of the above", "none", "n/a",
                                                          "none of these"):
            out.append(NEUTRAL_NONE)
        else:
            out.append(o)
    return out


@dataclass(frozen=True)
class LabelTable:
    names: tuple[str, ...]
    ids: tuple[int, ...]
    open_ids: tuple[int, ...]


def label_table(tok: Any) -> LabelTable:
    """A..Z then the two-letter upper-case strings that are single tokens, 255 in all."""
    upper = string.ascii_uppercase
    out: list[tuple[str, int]] = []
    for n in list(upper) + [a + b for a in upper for b in upper]:
        t = tok.encode(n, add_special_tokens=False)
        if len(t) == 1:
            out.append((n, t[0]))
        if len(out) == MAX_OPTIONS:
            break
    if len(out) != MAX_OPTIONS or len({i for _, i in out}) != MAX_OPTIONS:
        raise ValueError("tokenizer has fewer than 255 single-token labels")
    return LabelTable(tuple(n for n, _ in out), tuple(i for _, i in out),
                      tuple(tok.encode("\n(", add_special_tokens=False)))


def build_row_text(tok: Any, labels: LabelTable, state_text: str, question: str,
                   options: Sequence[str]) -> str:
    """The row as text, exactly as llama-server receives it (decoded from decider's ids)."""
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError("2..255 options required")
    ctx_text = "Context:\n" + state_text
    head = f"\n\nQuestion: {question}\nOptions:"
    tail = "\nAnswer: ("
    if len(options) <= NARROW:
        lines = [head] + [f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)] + [tail]
        ids = tok.encode(ctx_text + "".join(lines), add_special_tokens=False)
        return tok.decode(ids)
    ids = list(tok.encode(ctx_text, add_special_tokens=False)) + tok.encode(head, add_special_tokens=False)
    for j, o in enumerate(options):
        ids += list(labels.open_ids) + [labels.ids[j]] + tok.encode(f") {o}", add_special_tokens=False)
    ids += tok.encode(tail, add_special_tokens=False)
    return tok.decode(ids)


@dataclass(frozen=True)
class Row:
    """One scored question: its text, the label strings of its options, the typed names."""
    key: str
    type: str
    text: str
    labels: tuple[str, ...]
    names: tuple[Any, ...]


def request_rows(tok: Any, labels: LabelTable, state: Any,
                 questions: Mapping[str, Mapping[str, Any]], neutralize: bool = True) -> list[Row]:
    state_text = render_state(state)
    out = []
    for key, spec in questions.items():
        q = render_question(spec)
        opts = neutralize_options(q.options) if neutralize else list(q.options)
        text = build_row_text(tok, labels, state_text, q.text, opts)
        out.append(Row(key, q.type, text, labels.names[:len(opts)], q.names))
    return out


def readout(label_logits: np.ndarray, temperature: float = CERNO_TEMPERATURE) -> list[float]:
    z = label_logits.astype(np.float64) / float(temperature)
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def readout_from_logprobs(logprobs: Sequence[float | None], temperature: float = CERNO_TEMPERATURE
                          ) -> list[float]:
    """llama-server's readout: p^(1/T) renormalised; a label missing from the top-n counts 0."""
    p = [math.exp(x) ** (1.0 / temperature) if x is not None else 0.0 for x in logprobs]
    s = sum(p)
    return [x / s for x in p] if s > 0 else [1.0 / len(p)] * len(p)
