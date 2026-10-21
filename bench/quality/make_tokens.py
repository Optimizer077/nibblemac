"""Tokenize the fixed wikitext-2 slice with the model's HF tokenizer -> raw int32 ids.

Input to engine/tools/pinfer_ppl (same text as bench/quality/ppl.sh uses for llama.cpp).
    python bench/quality/make_tokens.py [--n 32768]
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=Path, default=ROOT / "bench/quality/wikitext-2-raw-test.txt")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "models/Qwen3.5-0.8B/tokenizer.json")
    p.add_argument("--n", type=int, default=64 * 512, help="tokens to keep (from the start)")
    p.add_argument("--out", type=Path, default=ROOT / "bench/quality/wikitext2-qwen35.tokens.bin")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    text = a.text.read_text(encoding="utf-8")
    ids = Tokenizer.from_file(str(a.tokenizer)).encode(text, add_special_tokens=False).ids
    out = np.asarray(ids[: a.n], dtype="<i4")
    out.tofile(a.out)
    logger.info("%s: %d of %d tokens, text sha256 %s, out sha256 %s", a.out.name, out.size,
                len(ids), hashlib.sha256(text.encode()).hexdigest()[:16],
                hashlib.sha256(out.tobytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
