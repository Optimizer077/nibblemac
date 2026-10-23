"""Evaluate the prompt-layout variants with the 4B in fp32 torch on the PC (engine/PILOTS.md,
"Prompt economy").

    OMP_NUM_THREADS=8 /root/venv-ref/bin/python tools/pilot/prompt/run.py --variants base schema

Model: models/cerno-4b/cerno-4b-q4_0.pq dequantized into transformers' Qwen3_5ForCausalLM
(the compress agent's emulation, tools/compress/cerno.py on branch worktree-agent-a0bab39a2c3dd469b:
max |dp| 0.0094 and the same top answer on 51/52 rows vs the llama.cpp reference).  Readout as
cerno_pi/decide.py: p = softmax(label logits / 1.534), labels outside the top 260 of the vocabulary
count as 0.  Every request's prefix is evaluated once and its cache deep-copied per row (chains
continue from the live cache).  Writes bench/results/pilot/prompt/<variant>_<stamp>.json with the
probabilities per question, the reference and the token counts.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pqread import PQ  # noqa: E402
from variants import (LETTERS, MODELS, ROOT, VARIANTS, Plan, Tok, build, check_base,  # noqa: E402
                      load_reference, load_requests)

logger = logging.getLogger(__name__)
CERNO_DIR = MODELS / "cerno-4b"
CERNO_PQ = Path(os.environ.get("PILOT_PQ", "/root/lowrank/cerno/cerno-4b-q4_0.pq"))
if not CERNO_PQ.exists() or CERNO_PQ.stat().st_size != (CERNO_DIR / "cerno-4b-q4_0.pq").stat().st_size:
    CERNO_PQ = CERNO_DIR / "cerno-4b-q4_0.pq"
RESULTS = ROOT / "bench" / "results" / "pilot" / "prompt"
TEMPERATURE = 1.534
TOP_N = 260
ZERO_CENTERED = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight",
                 "k_norm.weight")
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("PILOT_THREADS", "8")))


def hf_state_dict(pq: PQ) -> dict[str, torch.Tensor]:
    sd = {}
    for name in pq.entries:
        a = torch.from_numpy(pq.get(name))
        if name.endswith(ZERO_CENTERED) or name == "norm.weight":
            a = a - 1.0                                   # .pq stores 1 + w
        if name.endswith("linear_attn.conv1d.weight"):
            a = a[:, None, :]
        if name.endswith("linear_attn.A"):
            name = name[:-1] + "A_log"
            a = torch.log(-a)
        key = name if name == "lm_head.weight" else "model." + name
        sd[key] = a.contiguous()
    return sd


def _to(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple(_to(v, device) for v in x)
    if isinstance(x, list):
        return [_to(v, device) for v in x]
    if isinstance(x, dict):
        return {k: _to(v, device) for k, v in x.items()}
    return x


class Cerno:
    """fp32 everywhere.  `gpu_layers` > 0 keeps layers 0..gpu_layers-1 on the GPU (TF32 off) and
    the rest, the embedding and the head on the CPU: the 4B's fp32 body (14.3 GB) does not fit
    the 16 GB card."""

    def __init__(self, gpu_layers: int = 0) -> None:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
        t0 = time.time()
        cfg = json.loads((CERNO_DIR / "hf" / "config.json").read_text())
        cfg.pop("quantization_config", None)
        cfg["tie_word_embeddings"] = False
        config = Qwen3_5TextConfig(**cfg)
        with torch.device("meta"):
            model = Qwen3_5ForCausalLM(config)
        sd = hf_state_dict(PQ(CERNO_PQ))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if unexpected or [m for m in missing if "rotary" not in m]:
            raise RuntimeError(f"missing {missing} unexpected {unexpected}")
        model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        model.eval()
        self.model, self.lm = model, model.model
        self.gpu_layers = 0
        if gpu_layers > 0 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            n = len(self.lm.layers)
            self.gpu_layers = min(gpu_layers, n)
            for i in range(self.gpu_layers):
                self.lm.layers[i].to("cuda")
            self.lm.rotary_emb.to("cuda")
            # every CPU module pulls its inputs back (the model loop passes the same cuda
            # position embeddings / mask to every layer)
            for mod in list(self.lm.layers[self.gpu_layers:]) + [self.lm.norm]:
                mod.register_forward_pre_hook(
                    lambda m, args, kwargs: (_to(args, "cpu"), _to(kwargs, "cpu")),
                    with_kwargs=True)
            logger.info("GPU: layers 0..%d (%.1f GB), TF32 off", self.gpu_layers - 1,
                        torch.cuda.memory_allocated() / 2**30)
        logger.info("Cerno loaded from %s in %.0f s", CERNO_PQ, time.time() - t0)

    def forward(self, ids: list[int], cache=None) -> tuple[torch.Tensor, object]:
        x = torch.tensor([ids])
        if self.gpu_layers:
            emb = self.lm.embed_tokens(x).to("cuda")
            out = self.lm(inputs_embeds=emb, past_key_values=cache, use_cache=True)
        else:
            out = self.lm(input_ids=x, past_key_values=cache, use_cache=True)
        return out.last_hidden_state[0].cpu(), out.past_key_values

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head.weight @ h


def readout(logits: torch.Tensor, labels: list[int]) -> list[float]:
    lg = logits.double()
    top = torch.topk(lg, TOP_N).values[-1]
    z = lg[labels] / TEMPERATURE
    keep = lg[labels] >= top
    if not keep.any():
        return [1.0 / len(labels)] * len(labels)
    e = torch.where(keep, torch.exp(z - z[keep].max()), torch.zeros_like(z))
    return (e / e.sum()).tolist()


def run_plan(c: Cerno, p: Plan) -> dict[str, list[float]]:
    """Probabilities per question id."""
    out: dict[str, list[float]] = {}
    cache = None
    if p.prefix:
        _, cache = c.forward(p.prefix)
    prev_letter = ""
    for row in p.rows:
        ids = row.ids_fn(prev_letter) if row.ids_fn is not None else row.ids
        h, cache2 = c.forward(ids, cache if p.chain else copy.deepcopy(cache))
        for s in row.slots:
            pos = s.pos if s.pos >= 0 else len(ids) - 1
            out[s.key] = readout(c.logits(h[pos]), s.labels)
        if p.chain:
            cache = cache2
            prev_letter = LETTERS[int(np.argmax(out[row.slots[-1].key]))]
    return out


def stats(rows: list[dict], key: str) -> dict:
    d = [max(abs(a - b) for a, b in zip(r["p"], r[key])) for r in rows]
    top = [int(np.argmax(r["p"]) == np.argmax(r[key])) for r in rows]
    return {"max_dp": float(max(d)), "mean_dp": float(np.mean(d)), "same_top": int(sum(top)),
            "rows": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--requests", nargs="*", help="request names (default: all 18)")
    ap.add_argument("--base", type=Path, help="base result json for the vs-base columns")
    ap.add_argument("--gpu-layers", type=int, default=0, help="layers kept on the GPU (fp32)")
    ap.add_argument("--tag", default="", help="suffix for the result file names")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    tok = Tok()
    logger.info("base rows == reference ids: %d", check_base(tok))
    ref = load_reference()
    reqs = [r for r in load_requests() if not a.requests or r[0] in a.requests]
    base_p: dict[tuple[str, str], list[float]] = {}
    if a.base:
        for r in json.loads(a.base.read_text())["rows"]:
            base_p[(r["request"], r["question"])] = r["p"]
    c = Cerno(a.gpu_layers)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        t0 = time.time()
        rows, tokens, tokens_cached = [], 0, 0
        for name, state, qs in reqs:
            p = build(tok, v, name, state, qs)
            tokens += p.tokens
            tokens_cached += p.tokens_cached
            t1 = time.time()
            probs = run_plan(c, p)
            for q in qs:
                rows.append({"request": name, "question": q.key, "p": probs[q.key],
                             "ref": ref[(name, q.key)]["probs"],
                             "base": base_p.get((name, q.key))})
            logger.info("%s %s: %d questions, %d tokens, %.1f s", v, name, len(qs), p.tokens,
                        time.time() - t1)
        res = {"variant": v, "tokens": tokens, "tokens_cached": tokens_cached,
               "requests": len(reqs), "seconds": time.time() - t0, "gpu_layers": c.gpu_layers,
               "pq": str(CERNO_PQ), "vs_ref": stats(rows, "ref")}
        if all(r["base"] is not None for r in rows):
            res["vs_base"] = stats(rows, "base")
        elif v == "base":
            for r in rows:
                r["base"] = r["p"]
            base_p.update({(r["request"], r["question"]): r["p"] for r in rows})
            res["vs_base"] = stats(rows, "base")
        res["rows"] = rows
        path = RESULTS / f"{v}{a.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        logger.info("%s: tokens %d (cached %d) vs_ref %s vs_base %s -> %s", v, tokens,
                    tokens_cached, res["vs_ref"], res.get("vs_base"), path)


if __name__ == "__main__":
    main()
