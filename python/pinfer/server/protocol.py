"""OpenAI wire format: request validation and response objects (no model access).

Parses the bodies of ``POST /v1/chat/completions`` and ``POST /v1/completions`` into
:class:`ChatRequest` / :class:`CompletionRequest` and builds the response, chunk, usage and
error objects in OpenAI's shapes. Every validation failure is an :class:`APIError` (HTTP 400
unless stated otherwise) carrying OpenAI's ``{"error": {"message", "type", "param", "code"}}``.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

ROLES = ("system", "user", "assistant", "tool")
# Accepted for compatibility but without effect on the output (logged at DEBUG).
IGNORED = ("presence_penalty", "frequency_penalty", "logit_bias", "user", "metadata", "store",
           "parallel_tool_calls", "service_tier", "tool_choice", "reasoning_effort")


class APIError(Exception):
    """An error returned to the client as OpenAI error JSON with an HTTP status."""

    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 code: str | None = None, param: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.code = code
        self.param = param

    def body(self) -> dict[str, Any]:
        """``{"error": {...}}`` as OpenAI sends it."""
        return {"error": {"message": self.message, "type": self.type, "param": self.param,
                          "code": self.code}}


def bad_request(message: str, param: str | None = None, code: str | None = None) -> APIError:
    """HTTP 400 ``invalid_request_error``."""
    return APIError(400, message, "invalid_request_error", code, param)


@dataclass(frozen=True)
class Defaults:
    """Server-side defaults for options a request leaves out."""

    max_tokens: int = 1024            # chat; completions use OpenAI's 16
    completion_max_tokens: int = 16
    temperature: float = 0.0          # greedy: fastest (speculative) and deterministic
    top_p: float = 1.0
    top_k: int = 0
    enable_thinking: bool = False


@dataclass(frozen=True)
class Options:
    """Generation options of one request."""

    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int | None
    stop: tuple[str, ...]

    @property
    def greedy(self) -> bool:
        """True when sampling reduces to argmax (speculative decoding applies)."""
        return self.temperature == 0.0 or self.top_k == 1


@dataclass(frozen=True)
class ChatRequest:
    """A validated chat completion request; ``messages`` are normalized for the template."""

    messages: tuple[dict[str, Any], ...]
    options: Options
    enable_thinking: bool
    stream: bool
    include_usage: bool


@dataclass(frozen=True)
class CompletionRequest:
    """A validated (legacy) text completion request."""

    prompt: str | tuple[int, ...]
    options: Options
    echo: bool
    stream: bool
    include_usage: bool


# ------------------------------------------------------------------ field helpers
def _number(body: Mapping[str, Any], name: str, default: float, lo: float,
            hi: float = math.inf, lo_open: bool = False) -> float:
    v = body.get(name)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise bad_request(f"'{name}' must be a number", name)
    if v < lo or (lo_open and v == lo) or v > hi:
        rng = f"({lo}, {hi}]" if lo_open else f"[{lo}, {hi}]"
        raise bad_request(f"'{name}' must be in {rng}, got {v}", name)
    return float(v)


def _integer(body: Mapping[str, Any], name: str, lo: int | None = None) -> int | None:
    v = body.get(name)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise bad_request(f"'{name}' must be an integer", name)
    if lo is not None and v < lo:
        raise bad_request(f"'{name}' must be >= {lo}, got {v}", name)
    return v


def _flag(body: Mapping[str, Any], name: str, default: bool = False) -> bool:
    v = body.get(name, default)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise bad_request(f"'{name}' must be a boolean", name)
    return v


def _stop(body: Mapping[str, Any]) -> tuple[str, ...]:
    v = body.get("stop")
    if v is None:
        return ()
    items = [v] if isinstance(v, str) else v
    if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
        raise bad_request("'stop' must be a string or a list of strings", "stop")
    if len(items) > 16:
        raise bad_request("'stop' takes at most 16 strings", "stop")
    return tuple(s for s in items if s)


def _include_usage(body: Mapping[str, Any]) -> bool:
    so = body.get("stream_options")
    if so is None:
        return False
    if not isinstance(so, Mapping):
        raise bad_request("'stream_options' must be an object", "stream_options")
    return _flag(so, "include_usage")


def _unsupported(body: Mapping[str, Any]) -> None:
    """Reject options this server cannot honour; log the ignored ones."""
    n = _integer(body, "n", 1)
    if n is not None and n != 1:
        raise bad_request("only n=1 is supported", "n")
    if body.get("logprobs") not in (None, False, 0) or body.get("top_logprobs"):
        raise bad_request("logprobs are not supported", "logprobs")
    if body.get("tools") or body.get("functions"):
        raise bad_request("tools / function calling are not supported by this server", "tools")
    fmt = body.get("response_format")
    if fmt is not None and not (isinstance(fmt, Mapping) and fmt.get("type") == "text"):
        raise bad_request("only response_format {'type': 'text'} is supported",
                          "response_format")
    ignored = [k for k in IGNORED if body.get(k) not in (None, 0, {}, [])]
    if ignored:
        logger.debug("ignoring request options: %s", ", ".join(ignored))


def _options(body: Mapping[str, Any], default_max: int, d: Defaults) -> Options:
    mt = _integer(body, "max_completion_tokens", 0)
    if mt is None:
        mt = _integer(body, "max_tokens", 0)
    seed = _integer(body, "seed")
    top_k = _integer(body, "top_k", 0)
    return Options(max_tokens=default_max if mt is None else mt,
                   temperature=_number(body, "temperature", d.temperature, 0.0, 2.0),
                   top_p=_number(body, "top_p", d.top_p, 0.0, 1.0, lo_open=True),
                   top_k=d.top_k if top_k is None else top_k,
                   seed=seed, stop=_stop(body))


# ------------------------------------------------------------------ messages
def _content(content: Any, where: str) -> str:
    """OpenAI message content (string, null or a list of parts) -> text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, Mapping) or part.get("type", "text") != "text" \
                    or not isinstance(part.get("text"), str):
                raise bad_request(f"{where}: only text content parts are supported", where)
            parts.append(part["text"])
        return "".join(parts)
    raise bad_request(f"{where}: content must be a string or a list of text parts", where)


def normalize_messages(raw: Any) -> tuple[dict[str, Any], ...]:
    """Validate ``messages`` and reduce them to what the chat template reads.

    ``developer`` becomes ``system``; content parts are joined; ``name`` and other extra
    fields are dropped; assistant ``reasoning_content`` / ``tool_calls`` are kept.
    """
    if not isinstance(raw, list) or not raw:
        raise bad_request("'messages' must be a non-empty list", "messages")
    out = []
    for i, m in enumerate(raw):
        where = f"messages[{i}]"
        if not isinstance(m, Mapping):
            raise bad_request(f"{where} must be an object", where)
        role = m.get("role")
        role = "system" if role == "developer" else role
        if role not in ROLES:
            raise bad_request(f"{where}: unknown role {role!r}", f"{where}.role")
        msg: dict[str, Any] = {"role": role, "content": _content(m.get("content"), where)}
        if role == "assistant":
            if isinstance(m.get("reasoning_content"), str):
                msg["reasoning_content"] = m["reasoning_content"]
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
        out.append(msg)
    return tuple(out)


# ------------------------------------------------------------------ requests
def _object(body: Any) -> Mapping[str, Any]:
    if not isinstance(body, Mapping):
        raise bad_request("the request body must be a JSON object")
    return body


def parse_chat(body: Any, d: Defaults) -> ChatRequest:
    """Validate a ``/v1/chat/completions`` body."""
    b = _object(body)
    _unsupported(b)
    kwargs = b.get("chat_template_kwargs")
    if kwargs is not None and not isinstance(kwargs, Mapping):
        raise bad_request("'chat_template_kwargs' must be an object", "chat_template_kwargs")
    thinking = _flag(kwargs or {}, "enable_thinking", d.enable_thinking)
    return ChatRequest(messages=normalize_messages(b.get("messages")),
                       options=_options(b, d.max_tokens, d), enable_thinking=thinking,
                       stream=_flag(b, "stream"), include_usage=_include_usage(b))


def parse_completion(body: Any, d: Defaults) -> CompletionRequest:
    """Validate a ``/v1/completions`` body (one prompt: text, [text] or token ids)."""
    b = _object(body)
    _unsupported(b)
    if b.get("suffix"):
        raise bad_request("'suffix' is not supported", "suffix")
    if (_integer(b, "best_of", 1) or 1) != 1:
        raise bad_request("only best_of=1 is supported", "best_of")
    p = b.get("prompt")
    if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (str, list)):
        p = p[0]
    prompt: str | tuple[int, ...]
    if isinstance(p, str):
        prompt = p
    elif isinstance(p, list) and p and all(isinstance(t, int) and not isinstance(t, bool)
                                           for t in p):
        prompt = tuple(p)
    else:
        raise bad_request("'prompt' must be a string or a list of token ids (one prompt)",
                          "prompt")
    if not prompt:
        raise bad_request("'prompt' must not be empty", "prompt")
    return CompletionRequest(prompt=prompt, options=_options(b, d.completion_max_tokens, d),
                             echo=_flag(b, "echo"), stream=_flag(b, "stream"),
                             include_usage=_include_usage(b))


# ------------------------------------------------------------------ responses
def finish_reason(pinfer_reason: str | None) -> str:
    """pinfer's ``stop`` / ``length`` / ``context`` -> OpenAI's ``stop`` / ``length``."""
    return "stop" if pinfer_reason == "stop" else "length"


def usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict[str, Any]:
    """OpenAI ``usage`` object (``prompt_tokens_details.cached_tokens`` included)."""
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens}}


def chat_message(content: str, reasoning: str | None) -> dict[str, Any]:
    """The assistant message of a non-streaming chat response."""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return msg


def envelope(kind: str, rid: str, created: int, model: str,
             choices: Sequence[Mapping[str, Any]], **extra: Any) -> dict[str, Any]:
    """Common response / chunk envelope (``kind`` is the OpenAI ``object`` field)."""
    out: dict[str, Any] = {"id": rid, "object": kind, "created": created, "model": model,
                           "choices": list(choices)}
    out.update(extra)
    return out
