"""Build the shared 64-token benchmark prompt for all baseline runtimes.

Run inside WSL with the reference venv (transformers 5.20.2, torch cpu):
    /root/venv-ref/bin/python bench/common/make_prompt.py models/Qwen3.5-0.8B

Writes:
    bench/common/prompt64.txt   the exact text (decoded from the 64 ids; re-encodes to the same ids)
    bench/common/prompt64.json  ids, text, tokenizer sha256, model revision, and the HF fp32
                                greedy continuation (64 new tokens) as a cross-engine sanity check

The prompt is raw text (no chat template, no BOS: Qwen has none), so every engine that accepts
token ids sees exactly these 64 ids. Engines that only take text get prompt64.txt.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

N_PROMPT = 64
N_GEN = 64
# Public-domain source text (Dickens, A Tale of Two Cities, 1859, opening paragraph).
SOURCE = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the "
    "age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the "
    "season of Light, it was the season of Darkness, it was the spring of hope, it was the winter "
    "of despair, we had everything before us, we had nothing before us, we were all going direct "
    "to Heaven, we were all going direct the other way"
)
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def main() -> int:
    model_dir = Path(sys.argv[1])
    out_dir = Path(__file__).resolve().parent
    tok_path = model_dir / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))

    ids = tok.encode(SOURCE, add_special_tokens=False).ids
    if len(ids) < N_PROMPT:
        sys.exit(f"source text has only {len(ids)} tokens")
    ids = ids[:N_PROMPT]
    text = tok.decode(ids, skip_special_tokens=False)
    again = tok.encode(text, add_special_tokens=False).ids
    if again != ids:
        sys.exit(f"round-trip mismatch: {len(again)} ids after re-encoding")

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=N_GEN, min_new_tokens=N_GEN,
                             do_sample=False, num_beams=1)
    gen = out[0, N_PROMPT:].tolist()

    (out_dir / "prompt64.txt").write_text(text, encoding="utf-8", newline="")
    record = {
        "n_prompt": N_PROMPT,
        "ids": ids,
        "text": text,
        "source": "Dickens, A Tale of Two Cities (1859), opening paragraph, first 64 tokens",
        "tokenizer": "models/Qwen3.5-0.8B/tokenizer.json",
        "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION,
        "chat_template": None,
        "reference_greedy": {
            "engine": f"transformers {__import__('transformers').__version__}, "
                      f"torch {torch.__version__}, fp32, PC x86",
            "class": type(model).__name__,
            "n_gen": N_GEN,
            "ids": gen,
            "text": tok.decode(gen, skip_special_tokens=False),
        },
    }
    (out_dir / "prompt64.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "ids"}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
