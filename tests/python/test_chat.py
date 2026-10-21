"""Chat template: built-in Qwen3.5 port vs the model's Jinja template."""
from __future__ import annotations

from typing import Any

import pytest
from pinfer.chat import apply_chat_template, jinja_available, render_qwen_builtin

CONVERSATIONS = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "system", "content": "  You are terse.  "}, {"role": "user", "content": "2+2?"}],
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
     {"role": "user", "content": "How are you?"}],
    [{"role": "user", "content": "Q"},
     {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nA"},
     {"role": "user", "content": "Q2"},
     {"role": "assistant", "content": "A2", "reasoning_content": "r2"}],
    [{"role": "user", "content": "weather?"}, {"role": "assistant", "content": "checking"},
     {"role": "tool", "content": "sunny"}, {"role": "tool", "content": "20C"}],
    [{"role": "user", "content": [{"type": "text", "text": "multi"}, {"type": "text",
                                                                       "text": "part"}]}],
]


def test_builtin_simple() -> None:
    out = render_qwen_builtin([{"role": "user", "content": "Hello!"}])
    assert out == ("<|im_start|>user\nHello!<|im_end|>\n"
                   "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    think = render_qwen_builtin([{"role": "user", "content": "Hello!"}], enable_thinking=True)
    assert think.endswith("<|im_start|>assistant\n<think>\n")


@pytest.mark.parametrize("conv", CONVERSATIONS)
@pytest.mark.parametrize("thinking", [False, True])
def test_builtin_matches_jinja(tokenizer: Any, conv: list[dict[str, Any]],
                               thinking: bool) -> None:
    if not jinja_available():
        pytest.skip("jinja2 not installed")
    tmpl = tokenizer.chat_template
    jinja = apply_chat_template(conv, tmpl, enable_thinking=thinking, use_jinja=True)
    builtin = apply_chat_template(conv, tmpl, enable_thinking=thinking, use_jinja=False)
    assert builtin == jinja


@pytest.mark.parametrize("conv,msg", [
    ([], "No messages"),
    ([{"role": "assistant", "content": "x"}], "No user query"),
    ([{"role": "user", "content": "x"}, {"role": "system", "content": "s"}], "System message"),
    ([{"role": "robot", "content": "x"}, {"role": "user", "content": "y"}], "Unexpected"),
])
def test_builtin_errors(conv: list[dict[str, Any]], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        render_qwen_builtin(conv)


def test_tools_need_jinja() -> None:
    with pytest.raises(ValueError):
        apply_chat_template([{"role": "user", "content": "x"}], None,
                            tools=[{"name": "f"}], use_jinja=False)


def test_chat_generates(fresh: Any) -> None:
    r = fresh.chat([{"role": "user", "content": "What is 2+2? Answer with one number."}],
                   max_new_tokens=6)
    assert r.token_ids and r.finish_reason in ("stop", "length")
    assert "<|im_end|>" not in r.text
