"""Chat prompts whose answer copies the context (bench/common/chat_fix.ids, chat_edit.ids),
plus held-out validation prompts for the draft policies (chat_val_*.ids).

Prompt-lookup (n-gram) drafting should help where the answer repeats text from the prompt:
fixing typos in a paragraph, editing a function. Rendered like make_chat_prompts.py (one user
turn, the model's chat template, thinking off). Run from the project root:
    PYTHONPATH=python uv run --no-project --with numpy --with tokenizers --with jinja2 \
        python bench/common/make_copy_prompts.py models/Qwen3.5-0.8B
"""
from __future__ import annotations

import sys
from pathlib import Path

from pinfer.chat import apply_chat_template
from pinfer.tokenizer import Tokenizer

FIX = ("The Raspbery Pi is a small single-board computr that was first relesed in 2012. It was "
       "desinged to help students learn programing, but it quickly became populer with hobbyists "
       "and enginers. The fourth model has a quad-core procesor, up to eight gigabytes of memmory, "
       "and two micro HDMI ports. Many people use it as a home server, a media center, or the "
       "brain of a robbot. Becuase it uses very litle power, it can run all day without making "
       "the electricity bill much higer.")

EDIT = '''def word_frequencies(path, min_len=3):
    """Count how often each word of at least min_len letters occurs in a text file."""
    cnt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            for word in line.lower().split():
                word = word.strip(".,;:!?\\"'()[]")
                if len(word) < min_len:
                    continue
                cnt[word] = cnt.get(word, 0) + 1
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(cnt.values())
    return [(word, n, n / total) for word, n in items]'''

PROMPTS = {
    "fix": "Repeat the following paragraph exactly, but correct the spelling mistakes. Output "
           "only the corrected paragraph.\n\n" + FIX,
    "edit": "Rename the variable `cnt` to `count` in this Python function and return the whole "
            "function in a code block.\n\n```python\n" + EDIT + "\n```",
}

# Held-out prompts (chat_val_*.ids): drafted policies were tuned on the chat_*.ids traces
# only; these check that the tuning carries over (engine/DRAFT.md).
VALIDATION = {
    "val_summ": "Summarize the following text in two sentences.\n\nThe city council met on "
                "Tuesday to discuss the new bicycle lanes on Main Street. Several shop owners "
                "said the lanes had reduced the number of parking spaces and hurt their "
                "business. Cyclists and some residents replied that the street was much safer "
                "now and that more people were visiting the shops on foot. The council decided "
                "to collect traffic and sales data for six more months before making a final "
                "decision.",
    "val_json": "Convert this list into a JSON array of objects with the keys name, age and "
                "city:\nAlice, 34, Lisbon\nBob, 27, Oslo\nCarla, 45, Madrid\nDavid, 31, Vienna\n"
                "Emma, 29, Dublin\nFrank, 52, Prague",
    "val_explain": "Explain how a hash table works and why lookups are usually fast.",
    "val_sql": "Write an SQL query that returns the ten customers with the highest total order "
               "value, given tables customers(id, name) and orders(id, customer_id, amount).",
}


def main(tokenizer_dir: str) -> None:
    tok = Tokenizer.from_path(tokenizer_dir)
    out = Path(__file__).resolve().parent
    for name, text in {**PROMPTS, **VALIDATION}.items():
        ids = tok.encode(apply_chat_template([{"role": "user", "content": text}], tok.chat_template))
        (out / f"chat_{name}.ids").write_text(",".join(map(str, ids)) + "\n", encoding="utf-8")
        print(f"chat_{name}.ids: {len(ids)} tokens")


if __name__ == "__main__":
    main(sys.argv[1])
