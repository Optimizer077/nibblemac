"""fp32 reference logits at long positions, for tests/test_long_context.py.

Runs the official transformers implementation of Qwen3.5-0.8B in fp32 with sdpa attention
over a long token-id prompt (default `bench/common/ctx8192.ids`, wikitext-2 text) and keeps
the logits of selected prefix lengths only: all 8192 positions would be 8 GB of logits.

Writes to OUT (default ref/golden/qwen35-0.8b-long/):
    meta.json       ids file and its sha256, positions, versions, chunking, timing
    positions.npy   int32 [P]     prefix lengths n; row i holds the logits of token n_i - 1
    logits.npy      fp32  [P, V]

Run in WSL:  /root/venv-ref/bin/python ref/long_context_reference.py [--ids F] [--out DIR]
             [--positions 3000,4096,...] [--chunk N]
--chunk N feeds the prompt in pieces of N tokens through the model's cache (recurrent and
conv states, KV cache) to bound the attention memory; 0 (default) is one forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
# Every 512 tokens, plus the prefix lengths named in the long-context task (3000, 6000).
DEFAULT_POSITIONS = sorted({*range(512, 8193, 512), 3000, 6000})


def read_ids(path: Path) -> list[int]:
    """Token ids from a comma/whitespace separated text file (bench/common/*.ids)."""
    return [int(t) for t in path.read_text(encoding="utf-8").replace(",", " ").split()]


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


def reference_logits(model_dir: Path, ids: list[int], positions: list[int],
                     chunk: int = 0) -> tuple[np.ndarray, float]:
    """fp32 logits [len(positions), V] of token positions[i] - 1, and the seconds taken."""
    import torch  # here, so that tests/test_long_context.py needs torch only to fill its cache
    from transformers import Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    t0 = time.time()
    rows: dict[int, torch.Tensor] = {}
    want = sorted(set(positions))
    with torch.no_grad():
        if chunk <= 0:
            keep = torch.tensor([p - 1 for p in want])
            out = model(input_ids=torch.tensor([ids[:want[-1]]]), use_cache=False,
                        logits_to_keep=keep)
            for i, p in enumerate(want):
                rows[p] = out.logits[0, i].float()
        else:
            cache = None
            for s in range(0, want[-1], chunk):
                e = min(s + chunk, want[-1])
                here = [p for p in want if s < p <= e]
                keep = torch.tensor([p - 1 - s for p in here]) if here else 1
                out = model(input_ids=torch.tensor([ids[s:e]]), past_key_values=cache,
                            use_cache=True, logits_to_keep=keep)
                cache = out.past_key_values
                for i, p in enumerate(here):
                    rows[p] = out.logits[0, i].float()
                logger.info("reference: %d / %d tokens, %.0f s", e, want[-1], time.time() - t0)
    logits = torch.stack([rows[p] for p in positions]).numpy().astype(np.float32)
    return logits, time.time() - t0


def main() -> None:
    import torch
    import transformers

    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=ROOT / "models" / "Qwen3.5-0.8B")
    p.add_argument("--ids", type=Path, default=ROOT / "bench" / "common" / "ctx8192.ids")
    p.add_argument("--out", type=Path, default=ROOT / "ref" / "golden" / "qwen35-0.8b-long")
    p.add_argument("--positions", default=",".join(map(str, DEFAULT_POSITIONS)))
    p.add_argument("--chunk", type=int, default=0)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ids = read_ids(a.ids)
    positions = [int(x) for x in a.positions.split(",") if x]
    if not positions or min(positions) < 1 or max(positions) > len(ids):
        raise SystemExit(f"positions must lie in 1..{len(ids)}")
    logits, secs = reference_logits(a.model, ids, positions, a.chunk)
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "positions.npy", np.asarray(positions, dtype=np.int32))
    np.save(a.out / "logits.npy", logits)
    meta = {
        "model": str(a.model), "revision": REVISION, "ids_file": str(a.ids),
        "n_ids": len(ids), "ids_sha256": ids_sha256(ids), "positions": positions,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "dtype": "float32", "attn_implementation": "sdpa", "chunk": a.chunk,
        "seconds": round(secs, 1), "logits_shape": list(logits.shape),
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("wrote %s: %d positions, %.0f s", a.out, len(positions), secs)


if __name__ == "__main__":
    main()
