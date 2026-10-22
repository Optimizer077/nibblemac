"""Chat prompt tokens that keep the conversation's earlier turns as they were generated.

Clients resend the whole conversation every turn. The engine can continue its context only
when the new prompt *extends* the tokens already in it, but the Qwen3.5 chat template renders
an earlier assistant turn differently from what the model saw while writing it:

    generated:  <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nReply<|im_end|>
    re-rendered: <|im_start|>assistant\\nReply<|im_end|>          (think block dropped)

and re-tokenizing the reply text need not give the generated token ids back. With
``history="generated"`` (the default) a conversation whose previous turn is a reply this
server generated is therefore built from that request's prompt tokens + the reply's token
ids + ``<|im_end|>``, followed by the template's rendering of the new messages. Only the new
turn is prefilled. ``history="template"`` renders everything with the template (what a
stateless server does); reuse then ends at the first earlier assistant turn.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .cache import Reply, ReplyRegistry
from .protocol import bad_request

if TYPE_CHECKING:
    from ..model import Model

IM_END = "<|im_end|>"


@dataclass(frozen=True)
class PromptPlan:
    """The tokens of one request's prompt.

    Attributes:
        ids: Prompt token ids.
        prefix_len: Tokens of the system message segment (a snapshot checkpoint), or 0.
        history: Canonical messages (chat), recorded with the reply for the next turn.
        spliced: Leading tokens taken from an earlier request + reply (0: none).
    """

    ids: list[int]
    prefix_len: int = 0
    history: tuple[str, ...] | None = None
    spliced: int = 0


def canonical(msg: Mapping[str, Any]) -> str:
    """What the template output depends on, as a comparable string (content stripped like the
    template does; an assistant's reasoning is ignored, it is not rendered for history)."""
    d = {"role": msg.get("role"), "content": str(msg.get("content") or "").strip()}
    if msg.get("tool_calls"):
        d["tool_calls"] = msg["tool_calls"]
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _plain_assistant(msg: Mapping[str, Any]) -> str | None:
    if msg.get("role") != "assistant" or msg.get("tool_calls"):
        return None
    return str(msg.get("content") or "").strip()


def _nth_end(text: str, n: int) -> int:
    """Index just after the n-th ``<|im_end|>`` in text, or -1."""
    pos = 0
    for _ in range(n):
        i = text.find(IM_END, pos)
        if i < 0:
            return -1
        pos = i + len(IM_END)
    return pos


def _template_error(e: Exception) -> bool:
    """True for the template's own refusals (raise_exception -> ValueError, jinja2 errors)."""
    return isinstance(e, (ValueError, TypeError)) or type(e).__module__.startswith("jinja2")


def tail_after(model: Model, messages: Sequence[Mapping[str, Any]], k: int, full: str,
               thinking: bool) -> str | None:
    """The template text that follows assistant message ``k`` in ``full`` (the rendering of
    all messages + generation prompt), starting with the newline after its ``<|im_end|>``.
    None if it cannot be located."""
    try:
        head = model.apply_chat_template(messages[:k + 1], False, thinking)
    except Exception as e:  # only template refusals; anything else is re-raised
        if not _template_error(e):
            raise
        return None
    pos = _nth_end(full, head.count(IM_END))
    if pos < 0 or not full.startswith("\n", pos):
        return None
    return full[pos:]


def system_prefix_len(model: Model, messages: Sequence[Mapping[str, Any]], full: str,
                      ids: Sequence[int]) -> int:
    """Token count of the system message segment if ``ids`` start with it, else 0."""
    if not messages or messages[0].get("role") != "system":
        return 0
    pos = _nth_end(full, 1)
    if pos < 0 or not full.startswith("\n", pos):
        return 0
    head = model.tokenize(full[:pos + 1])
    return len(head) if list(ids[:len(head)]) == head else 0


def chat_prompt(model: Model, registry: ReplyRegistry | None,
                messages: Sequence[Mapping[str, Any]], thinking: bool) -> PromptPlan:
    """Prompt tokens for a chat request (``registry=None``: plain template rendering)."""
    try:
        full = model.apply_chat_template(messages, True, thinking)
    except Exception as e:  # the template's own checks (system placement, ...)
        if not _template_error(e):
            raise
        raise bad_request(f"messages: {e}", "messages") from None
    history = tuple(canonical(m) for m in messages)
    ids: list[int] | None = None
    spliced = 0
    if registry is not None:
        hit = registry.find(history, [_plain_assistant(m) for m in messages])
        if hit is not None:
            ids = _splice(model, hit, messages, full, thinking)
            spliced = len(hit.prompt_ids) + len(hit.reply_ids) + 1 if ids else 0
    if ids is None:
        ids = model.tokenize(full)
    return PromptPlan(ids, system_prefix_len(model, messages, full, ids), history, spliced)


def _splice(model: Model, hit: Reply, messages: Sequence[Mapping[str, Any]], full: str,
            thinking: bool) -> list[int] | None:
    tail = tail_after(model, messages, len(hit.history), full, thinking)
    end = model.tokenizer.token_to_id(IM_END) if model.tokenizer else None
    if tail is None or end is None:
        return None
    return [*hit.prompt_ids, *hit.reply_ids, end, *model.tokenize(tail)]
