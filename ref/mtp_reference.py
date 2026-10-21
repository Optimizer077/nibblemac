"""fp32 PyTorch reference of the Qwen3.5 MTP (multi-token prediction) head, on top of HF.

transformers drops the `mtp.*` tensors, so this file rebuilds the head from HF building
blocks (Qwen3_5DecoderLayer / Qwen3_5RMSNorm) with the semantics of vLLM's
`qwen3_5_mtp.py` + `spec_decode/llm_base_proposer.py` and llama.cpp's `qwen35.cpp`
(`graph_mtp`), see engine/SPEC.md for the source commits:

    MTP row at position i (inputs: main hidden h_i AFTER the final norm, token x_{i+1}):
        e = pre_fc_norm_embedding(embed(x_{i+1}));  g = pre_fc_norm_hidden(h_i)
        u = fc(concat(e, g))                       # [2H] -> [H], embedding half first
        u = full-attention decoder layer (own KV cache, RoPE position i)
        m_i = mtp.norm(u);  logits = embed^T m_i   # tied head; argmax drafts x_{i+2}
    Chain step s >= 2 (draft the token after the previous draft d): row at position i+s-1
    with inputs (m of the previous step, d). Its attention sees the MTP KV rows 0..i
    (computed from main hiddens) plus the earlier chain rows of the same draft.

Outputs (run in WSL: /root/venv-ref/bin/python ref/mtp_reference.py):
  ref/golden/qwen35-0.8b-mtp/
    mtp_logits.npy  fp32 [T, V]   MTP logits of rows i=0..T-1 of the golden prompt, with
                                  x_T = first greedy token (row T-1 drafts greedy[1])
    mtp_hidden.npy  fp32 [T, H]   m_i (after mtp.norm)
    chain_logits.npy fp32 [K-1, V] chain steps 2..K from base row T-1 (full-vocab argmax)
    draft_ids.npy   int32 [K]     the K chained drafts from base row T-1
    acceptance.json               measured draft acceptance on varied text (see below)
    meta.json
Acceptance: greedy-decode several prompts with HF, then for every position of the generated
text measure P(chain draft s is correct | drafts 1..s-1 correct) for s = 1..K, for several
draft vocab sizes (argmax over the first N token ids only) and two control variants
(pre-final-norm hidden, swapped concat order) that a correct semantics must beat.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "ref" / "golden" / "qwen35-0.8b"
OUT = ROOT / "ref" / "golden" / "qwen35-0.8b-mtp"
FULL_ATTN_LAYER = 3  # any full-attention layer index: only selects the block type
PROMPTS = [
    "The Raspberry Pi 4 is a small computer that",
    "It was the best of times, it was the worst of times, it was the age of wisdom,",
    "def quicksort(arr):\n    \"\"\"Sort a list of numbers.\"\"\"\n",
    "Q: Why is the sky blue?\nA:",
    "The French Revolution began in 1789 when",
    "Ingredients for a simple tomato soup:\n1.",
    "人工智能的发展历史可以追溯到",
]


class MTPHead(nn.Module):
    """Qwen3.5 MTP block; weights loaded from the checkpoint's `mtp.*` tensors."""

    def __init__(self, cfg, embed: nn.Embedding, rotary: nn.Module):
        super().__init__()
        H = cfg.hidden_size
        self.embed, self.rotary = embed, rotary
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layer = Qwen3_5DecoderLayer(cfg, FULL_ATTN_LAYER)
        self.norm = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)

    def load(self, ckpt: Path) -> None:
        sd = {}
        with safe_open(str(ckpt), framework="pt") as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    sd[k[4:].replace("layers.0.", "layer.")] = f.get_tensor(k).float()
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith(("embed.", "rotary."))]
        if missing or unexpected:
            raise RuntimeError(f"MTP weights: missing {missing}, unexpected {unexpected}")

    def forward(self, h: torch.Tensor, tok: torch.Tensor, pos: torch.Tensor,
                mask: torch.Tensor, swap: bool = False) -> torch.Tensor:
        """h [T,H], tok [T], pos [T], mask [T,T] additive -> m [T,H] (after mtp.norm)."""
        e = self.pre_fc_norm_embedding(self.embed(tok))
        g = self.pre_fc_norm_hidden(h)
        u = self.fc(torch.cat([g, e] if swap else [e, g], dim=-1))[None]
        cos_sin = self.rotary(u, pos[None])
        u = self.layer(u, position_embeddings=cos_sin, attention_mask=mask[None, None])
        return self.norm(u[0])


def causal(n: int) -> torch.Tensor:
    return torch.full((n, n), float("-inf")).triu(1)


def chain_rows(T: int, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask + positions for main rows 0..T-1 followed by `steps` blocks of chain rows.

    Chain row (s, i) (s = 1..steps) sits at position i + s and sees main rows 0..i and the
    chain rows (1..s, i) of the same base i."""
    n = T * (steps + 1)
    allow = torch.zeros(n, n, dtype=torch.bool)
    allow[:T, :T] = torch.ones(T, T, dtype=torch.bool).tril()
    idx = torch.arange(T)
    pos = [idx]
    for s in range(1, steps + 1):
        r = T * s + idx
        allow[r] = allow[idx]                       # main rows 0..i
        for s2 in range(1, s + 1):
            allow[r, T * s2 + idx] = True           # its own chain so far
        pos.append(idx + s)
    mask = torch.zeros(n, n).masked_fill(~allow, float("-inf"))
    return mask, torch.cat(pos)


@torch.no_grad()
def chain_drafts(mtp: MTPHead, head_w: torch.Tensor, h: torch.Tensor, x: torch.Tensor,
                 K: int, nvoc: int, swap: bool = False) -> torch.Tensor:
    """Drafts [K, T-1]: draft s for base row i predicts x_{i+s+1}. Rows use x_{i+1}."""
    T = h.shape[0] - 1                    # rows 0..T-1 need x_{i+1}
    drafts, prev_m, prev_d = [], None, None
    for s in range(K):
        mask, pos = chain_rows(T, s)
        hs, ts = [h[:T]], [x[1:T + 1]]
        for j in range(s):                # earlier chain blocks, recomputed identically
            hs.append(prev_m[j])
            ts.append(prev_d[j])
        m = mtp(torch.cat(hs), torch.cat(ts), pos, mask, swap)
        mb = m[T * s:]
        d = (mb @ head_w[:nvoc].T).argmax(-1)
        drafts.append(d)
        prev_m = (prev_m or []) + [mb]
        prev_d = (prev_d or []) + [d]
    return torch.stack(drafts)


def acceptance(drafts: torch.Tensor, x: torch.Tensor, start: int) -> dict:
    """Conditional per-step acceptance and expected accepted length over base rows i whose
    targets x_{i+2..i+K+1} all lie in the generated region (i + 2 >= start)."""
    K, T = drafts.shape
    rows = [i for i in range(T) if i + 2 >= start and i + K + 1 < len(x)]
    ok = torch.stack([drafts[s, rows] == x[torch.tensor(rows) + s + 2] for s in range(K)])
    prefix = ok.cumprod(0).float()
    cond = [float(ok[0].float().mean())]
    for s in range(1, K):
        den = float(prefix[s - 1].sum())
        cond.append(float(prefix[s].sum()) / den if den else 0.0)
    return {"n_rows": len(rows), "p_cond": cond,
            "mean_accept_by_k": [float(prefix[:k].sum(0).mean()) for k in range(1, K + 1)]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3.5-0.8B")
    p.add_argument("--new-tokens", type=int, default=96)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--vocabs", default="248320,131072,65536,32768,16384,8192")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)

    mdir = ROOT / a.model
    tokz = AutoTokenizer.from_pretrained(mdir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        mdir, dtype=torch.float32, attn_implementation="eager").eval()
    lm = model.model.language_model
    cfg = lm.config
    mtp = MTPHead(cfg, lm.embed_tokens, lm.rotary_emb)
    mtp.load(next(mdir.glob("*.safetensors")))
    mtp.eval()
    head_w = lm.embed_tokens.weight
    pre_norm = {}
    lm.norm.register_forward_hook(lambda mod, inp, out: pre_norm.__setitem__("h", inp[0][0]))

    def main_hidden(ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return lm(input_ids=ids[None], use_cache=False).last_hidden_state[0]

    # ---- golden MTP logits on the golden prompt --------------------------------------
    ids = torch.from_numpy(np.load(GOLDEN / "input_ids.npy")).long()
    greedy = torch.from_numpy(np.load(GOLDEN / "greedy_ids.npy")).long()
    T = len(ids)
    with torch.no_grad():
        h = main_hidden(ids)
        x = torch.cat([ids, greedy[:1]])
        m = mtp(h, x[1:], torch.arange(T), causal(T))
        logits = m @ head_w.T
        drafts, ms, chain_logits = [int(logits[-1].argmax())], [m[-1]], []
        for s in range(1, a.k):           # chain from base row T-1, one new row per step
            hs = torch.cat([h, torch.stack(ms)])
            ts = torch.cat([x[1:], torch.tensor(drafts)])
            pos = torch.cat([torch.arange(T), T - 1 + torch.arange(1, s + 1)])
            mm = mtp(hs, ts, pos, causal(T + s))   # all chain rows share base T-1
            cl = mm[-1] @ head_w.T
            chain_logits.append(cl)
            drafts.append(int(cl.argmax()))
            ms.append(mm[-1])
    np.save(OUT / "mtp_logits.npy", logits.numpy().astype(np.float32))
    np.save(OUT / "mtp_hidden.npy", m.numpy().astype(np.float32))
    np.save(OUT / "chain_logits.npy", torch.stack(chain_logits).numpy().astype(np.float32))
    np.save(OUT / "draft_ids.npy", np.array(drafts, dtype=np.int32))
    gold_next = greedy[1:a.k + 1].tolist()
    logger.info("golden: chained drafts %s, true greedy %s", drafts, gold_next)

    # ---- acceptance on varied text ---------------------------------------------------
    vocabs = [int(v) for v in a.vocabs.split(",")]
    per_prompt, agg = [], {}
    for text in PROMPTS:
        pids = tokz(text, return_tensors="pt").input_ids[0]
        with torch.no_grad():
            gen = model.generate(pids[None], do_sample=False, max_new_tokens=a.new_tokens)[0]
            h = main_hidden(gen)
            h_pre = pre_norm["h"].clone()
        rec = {"prompt": text, "ids": pids.tolist(), "n_prompt": len(pids),
               "n_gen": len(gen) - len(pids)}
        variants = [(f"vocab{v}", h, v, False) for v in vocabs]
        variants += [("ctrl_prenorm_hidden", h_pre, vocabs[0], False),
                     ("ctrl_swapped_concat", h, vocabs[0], True)]
        for name, hh, nv, swap in variants:
            d = chain_drafts(mtp, head_w, hh, gen, a.k, nv, swap)
            r = acceptance(d, gen, len(pids))
            rec[name] = r
            g = agg.setdefault(name, {"n": 0, "ok": np.zeros(a.k), "den": np.zeros(a.k),
                                      "acc": np.zeros(a.k)})
            n = r["n_rows"]
            g["n"] += n
            g["acc"] += np.array(r["mean_accept_by_k"]) * n
            for s in range(a.k):  # recover counts from conditional rates
                den = n if s == 0 else n * np.prod(r["p_cond"][:s])
                g["den"][s] += den
                g["ok"][s] += den * r["p_cond"][s]
        per_prompt.append(rec)
        logger.info("%-40r n_gen=%d step1 acc (full vocab) %.3f", text[:40], rec["n_gen"],
                    rec[f"vocab{vocabs[0]}"]["p_cond"][0])
    summary = {name: {"n_rows": g["n"], "p_cond": (g["ok"] / np.maximum(g["den"], 1)).tolist(),
                      "mean_accept_by_k": (g["acc"] / max(g["n"], 1)).tolist()}
               for name, g in agg.items()}
    for name, s in summary.items():
        logger.info("%-22s rows %4d  p_cond %s  E[accepted] k=1..%d %s", name, s["n_rows"],
                    [round(v, 3) for v in s["p_cond"]], a.k,
                    [round(v, 3) for v in s["mean_accept_by_k"]])
    (OUT / "prompts.json").write_text(json.dumps(   # token ids for tests/test_spec.py
        [{"text": r["prompt"], "ids": r["ids"]} for r in per_prompt], indent=1,
        ensure_ascii=False), encoding="utf-8")
    (OUT / "acceptance.json").write_text(json.dumps(
        {"k": a.k, "new_tokens": a.new_tokens, "summary": summary, "per_prompt": per_prompt},
        indent=1, ensure_ascii=False))
    meta = {"model": a.model, "golden_prompt_ids": ids.tolist(), "first_greedy": int(greedy[0]),
            "draft_ids": drafts, "greedy_next": gold_next, "k": a.k,
            "semantics": "see module docstring / engine/SPEC.md",
            "torch": torch.__version__}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
