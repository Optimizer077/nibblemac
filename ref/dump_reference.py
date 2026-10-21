"""Golden reference for pinfer: run the official transformers implementation in fp32.

Writes to ref/golden/<name>/:
    meta.json            model path/revision, prompt, versions, shapes
    input_ids.npy        int32 [T]
    hidden_states.npy    fp32 [L+1, T, H]  (embeddings, then each decoder layer's output;
                         transformers applies the final norm to the last entry)
    logits.npy           fp32 [T, V]
    greedy_ids.npy       int32 [N]  greedy continuation (no sampling)

Run in WSL:  /root/venv-ref/bin/python ref/dump_reference.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3.5-0.8B")
    p.add_argument("--revision", default="2fc06364715b967f1860aea9cf38778875588b17")
    p.add_argument("--name", default="qwen35-0.8b")
    p.add_argument("--prompt", default="The Raspberry Pi 4 is a small computer that")
    p.add_argument("--new-tokens", type=int, default=32)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    torch.manual_seed(0)
    model_dir = ROOT / a.model
    out = ROOT / "ref" / "golden" / a.name
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()

    ids = tok(a.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        res = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        gen = model.generate(ids, do_sample=False, max_new_tokens=a.new_tokens)
    hidden = torch.stack(res.hidden_states)[:, 0].float().numpy()
    logits = res.logits[0].float().numpy()
    greedy = gen[0, ids.shape[1]:].numpy().astype(np.int32)

    np.save(out / "input_ids.npy", ids[0].numpy().astype(np.int32))
    np.save(out / "hidden_states.npy", hidden)
    np.save(out / "logits.npy", logits)
    np.save(out / "greedy_ids.npy", greedy)
    meta = {
        "model": a.model, "revision": a.revision, "prompt": a.prompt,
        "transformers": transformers.__version__, "torch": torch.__version__, "dtype": "float32",
        "input_ids": ids[0].tolist(), "greedy_ids": greedy.tolist(),
        "greedy_text": tok.decode(greedy), "hidden_shape": list(hidden.shape),
        "logits_shape": list(logits.shape),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("wrote %s: T=%d, hidden %s, logits %s", out, ids.shape[1], hidden.shape, logits.shape)
    logger.info("greedy: %r", meta["greedy_text"])


if __name__ == "__main__":
    main()
