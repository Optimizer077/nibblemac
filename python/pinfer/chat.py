"""Chat prompt formatting for Qwen3.5.

:func:`apply_chat_template` renders the model's own Jinja template (``chat_template.jinja`` or
``tokenizer_config.json``) with ``jinja2`` the way Hugging Face ``transformers`` does. Without
``jinja2`` or without a template it falls back to :func:`render_qwen_builtin`, a Python port
of the Qwen3.5 template for text conversations (system / user / assistant / tool messages;
tool definitions, tool calls and images need the Jinja path).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

Message = Mapping[str, Any]


def _content_text(content: Any) -> str:
    """Message content -> text (string, list of ``{"text": ...}`` items, or None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = []
        for item in content:
            if not isinstance(item, Mapping) or "text" not in item:
                raise ValueError("the built-in chat template supports text content only")
            parts.append(str(item["text"]))
        return "".join(parts)
    raise ValueError(f"unexpected message content type {type(content).__name__}")


def _assistant_turn(msg: Message, content: str, after_last_query: bool) -> str:
    reasoning = msg.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = ""
        if "</think>" in content:
            reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            content = content.split("</think>")[-1].lstrip("\n")
    reasoning = reasoning.strip()
    if msg.get("tool_calls"):
        raise ValueError("tool calls need the Jinja chat template (pip install jinja2)")
    if after_last_query:
        return f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n"
    return f"<|im_start|>assistant\n{content}<|im_end|>\n"


def render_qwen_builtin(messages: Sequence[Message], add_generation_prompt: bool = True,
                        enable_thinking: bool = False) -> str:
    """Render a text conversation exactly like the Qwen3.5 chat template (no tools/images).

    Raises:
        ValueError: Empty conversation, misplaced system message, no user query, unknown role.
    """
    if not messages:
        raise ValueError("No messages provided.")
    texts = [_content_text(m.get("content")).strip() for m in messages]
    out: list[str] = []
    if messages[0].get("role") == "system":
        out.append(f"<|im_start|>system\n{texts[0]}<|im_end|>\n")
    last_query = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            t = texts[i]
            if not (t.startswith("<tool_response>") and t.endswith("</tool_response>")):
                last_query = i
                break
    if last_query is None:
        raise ValueError("No user query found in messages.")
    for i, msg in enumerate(messages):
        role, content = msg.get("role"), texts[i]
        if role == "system":
            if i != 0:
                raise ValueError("System message must be at the beginning.")
        elif role == "user":
            out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            out.append(_assistant_turn(msg, content, i > last_query))
        elif role == "tool":
            if i > 0 and messages[i - 1].get("role") != "tool":
                out.append("<|im_start|>user")
            out.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if i == len(messages) - 1 or messages[i + 1].get("role") != "tool":
                out.append("<|im_end|>\n")
        else:
            raise ValueError("Unexpected message role.")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
        out.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(out)


def _raise(message: str) -> None:
    raise ValueError(message)


def _tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None,
            separators: tuple[str, str] | None = None, sort_keys: bool = False) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                      sort_keys=sort_keys)


def render_jinja(template: str, messages: Sequence[Message], add_generation_prompt: bool = True,
                 enable_thinking: bool = False, tools: Sequence[Mapping[str, Any]] | None = None,
                 **extra: Any) -> str:
    """Render a Hugging Face chat template with jinja2 (same environment settings as HF).

    Raises:
        ImportError: jinja2 is not installed.
    """
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env.from_string(template).render(
        messages=[dict(m) for m in messages], add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking, tools=list(tools) if tools else None, **extra)


def jinja_available() -> bool:
    """True if jinja2 can be imported."""
    try:
        import jinja2  # noqa: F401
    except ImportError:
        return False
    return True


def apply_chat_template(messages: Sequence[Message], template: str | None = None,
                        add_generation_prompt: bool = True, enable_thinking: bool = False,
                        tools: Sequence[Mapping[str, Any]] | None = None,
                        use_jinja: bool | None = None) -> str:
    """Format a conversation into the prompt text the model expects.

    Args:
        messages: ``[{"role": "system"|"user"|"assistant"|"tool", "content": str}, ...]``.
        template: Jinja template text (from the tokenizer); None -> built-in Qwen template.
        add_generation_prompt: Append the assistant header so the model answers next.
        enable_thinking: Let the model write a ``<think>`` block first (Qwen3.5 option).
            Default False inserts an empty think block, i.e. a direct answer.
        tools: Tool (function) definitions; needs the Jinja path.
        use_jinja: True forces jinja2, False forces the built-in template, None picks jinja2
            when both a template and jinja2 are available.
    """
    if use_jinja is None:
        use_jinja = template is not None and jinja_available()
    if use_jinja:
        if template is None:
            raise ValueError("use_jinja=True but the tokenizer has no chat template")
        return render_jinja(template, messages, add_generation_prompt, enable_thinking, tools)
    if tools:
        raise ValueError("tools need the Jinja chat template (pip install jinja2)")
    logger.debug("using the built-in Qwen chat template")
    return render_qwen_builtin(messages, add_generation_prompt, enable_thinking)
