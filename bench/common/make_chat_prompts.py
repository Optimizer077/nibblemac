"""Chat-formatted prompts for speculative-decoding benchmarks (bench/common/chat_*.ids).

Each prompt is one user turn rendered with the model's own chat template (thinking off, as
for chat27.ids), so the continuation is an assistant answer: code, a list, arithmetic or
free prose. Run from the project root:
    PYTHONPATH=python uv run --no-project --with numpy --with tokenizers --with jinja2 \
        python bench/common/make_chat_prompts.py models/Qwen3.5-0.8B
"""
from __future__ import annotations

import sys
from pathlib import Path

from pinfer.chat import apply_chat_template
from pinfer.tokenizer import Tokenizer

PROMPTS = {
    "code_py": "Write a Python function that returns the n-th Fibonacci number using memoization. "
               "Include type hints and a docstring.",
    "code_c": "Write a C function that reverses a singly linked list in place, with a short comment.",
    "list": "List ten European countries and their capitals, one per line, in the form Country: Capital.",
    "math": "A train travels 120 km in 1.5 hours and then 200 km in 2.5 hours. "
            "Solve step by step: what is its average speed for the whole trip?",
    "story": "Write a short story about a robot who learns to paint.",
}


def main(tokenizer_dir: str) -> None:
    tok = Tokenizer.from_path(tokenizer_dir)
    out = Path(__file__).resolve().parent
    for name, text in PROMPTS.items():
        ids = tok.encode(apply_chat_template([{"role": "user", "content": text}], tok.chat_template))
        (out / f"chat_{name}.ids").write_text(",".join(map(str, ids)) + "\n", encoding="utf-8")
        print(f"chat_{name}.ids: {len(ids)} tokens")


if __name__ == "__main__":
    main(sys.argv[1])
