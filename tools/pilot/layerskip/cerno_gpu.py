"""Cerno 4B (Qwen3.5-4B decision model, Q4_0) in exact fp32 torch on the PC GPU, with per-layer
skip modes and the cerno-pi scoring protocol (engine/PILOTS.md "Layer skipping").

* Weights: `models/cerno-4b/cerno-4b-q4_0.pq` (or `$LAYERSKIP_PQ`). The body's Q4_G32 matrices stay
  packed (nibbles + fp16 block scales, 2.0 GB) on the device and are dequantized per call: exact
  fp32 (`d * (u - 8)` has at most 11 + 3 significant bits). The Q8 embedding sits on the CPU as
  codes (rows are gathered and dequantized per token); the Q8 head is dequantized in row chunks on
  `$LAYERSKIP_HEAD` (cuda, default: 0.7 GB of codes; or cpu: fp32 once, 2.5 GB RAM). Same numbers
  as tools/compress/cerno.py's fp32 CPU emulation (max |dp| 0.0094 vs the reference there).
  The GPU is shared with other agents' jobs: the whole model takes ~3 GB of VRAM.
* Model: transformers 5.20.2 `Qwen3_5ForCausalLM` (text config of models/cerno-4b/hf/config.json,
  untied head). The decoder layer forward is replaced by one that honours a per-layer mode:
  `full`, `skip` (residual passthrough), `nomlp` (mixer only), `nomix` (MLP only).
* Rows: the 52 question rows of cerno-pi/results/ref_llama_pc.jsonl (18 requests); each request's
  common prefix is prefilled once and its cache deep-copied per row (checked by `skip_study.py check`).
  Positions and the causal mask are passed explicitly, so a skipped attention layer (empty KV
  cache) does not shift them.
* Readout (cerno_pi/decide.py): p = softmax(label logits / T), T = 1.534; a label outside the top 260
  of the full vocabulary counts as 0.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pqread import F32, PQ, Q4_G32, Q8_G32

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
CERNO_DIR = ROOT / "models" / "cerno-4b"
CERNO_PQ = Path(os.environ.get("LAYERSKIP_PQ", "")) if os.environ.get("LAYERSKIP_PQ") \
    else CERNO_DIR / "cerno-4b-q4_0.pq"
CERNO_REF = Path(os.environ.get("CERNO_REF", "/mnt/e/Personal_tries/cerno-pi/results/ref_llama_pc.jsonl"))
RESULTS = ROOT / "bench" / "results" / "layerskip"
TEMPERATURE = 1.534
TOP_N = 260
N_LAYERS = 32
FULL_ATTN = {3, 7, 11, 15, 19, 23, 27, 31}
# MMAC per token (docs/network-audit.md 1.1): MLP, DeltaNet mixer, full-attention mixer, body.
MMAC_MLP, MMAC_DN, MMAC_ATT = 70.78, 42.11, 36.70
MMAC_BODY = 24 * (MMAC_MLP + MMAC_DN) + 8 * (MMAC_MLP + MMAC_ATT)
ZERO_CENTERED = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight",
                 "k_norm.weight")
HEAD_CHUNK = 15520                       # rows per head chunk (16 chunks of 248320)
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

MODES: dict[int, str] = {}      # layer -> 'skip' | 'nomlp' | 'nomix' (absent = full)
ACTIVE = [True]                 # False: run every layer in full (prefix passes of question-only variants)
CAPTURE: dict[int, torch.Tensor] | None = None   # layer -> last-position output, when not None


def host_rss_gib() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def layer_mmac(layer: int, mode: str) -> float:
    mixer = MMAC_ATT if layer in FULL_ATTN else MMAC_DN
    return {"skip": MMAC_MLP + mixer, "nomlp": MMAC_MLP, "nomix": mixer, "full": 0.0}[mode]


def saved_fraction(modes: dict[int, str]) -> float:
    """Fraction of the body's matmul MACs removed per token the modes apply to."""
    return sum(layer_mmac(l, m) for l, m in modes.items()) / MMAC_BODY


def dequant_q8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """int8 codes [r, c], fp16 scales [r, c/32] -> fp32 [r, c] (exact)."""
    r, c = codes.shape
    return (codes.float().view(r, c // 32, 32) * scales.float()[..., None]).view(r, c)


def dequant_q4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """packed nibbles [r, c/2] (16 bytes per 32-column block), fp16 scales -> fp32 [r, c] (exact)."""
    r, c2 = packed.shape
    p = packed.view(r, c2 // 16, 16)
    u = torch.cat([p & 15, p >> 4], dim=-1).float() - 8.0           # [r, c/32, 32]
    return (u * scales.float()[..., None]).view(r, 2 * c2)


class Q4Linear(nn.Module):
    """y = x W^T, W dequantized from the packed Q4_0 codes at every call."""

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, dequant_q4(self.packed, self.scales))


class Q8Embedding(nn.Module):
    def __init__(self, codes: torch.Tensor, scales: torch.Tensor, out_device: torch.device):
        super().__init__()
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.out_device = out_device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        i = ids.to(self.codes.device)
        c, s = self.codes[i].float(), self.scales[i].float()
        return (c.view(*ids.shape, -1, 32) * s[..., None]).view(*ids.shape, -1).to(self.out_device)


class Q8Head:
    """logits = W h over the full vocabulary, W dequantized chunk by chunk (or once on the CPU)."""

    def __init__(self, codes: torch.Tensor, scales: torch.Tensor, device: torch.device):
        self.device = device
        if device.type == "cpu":
            self.w32 = dequant_q8(codes, scales)
        else:
            self.w32 = None
            self.codes, self.scales = codes.to(device), scales.to(device)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device)
        if self.w32 is not None:
            return self.w32 @ h
        out = []
        for r0 in range(0, self.codes.shape[0], HEAD_CHUNK):
            w = dequant_q8(self.codes[r0:r0 + HEAD_CHUNK], self.scales[r0:r0 + HEAD_CHUNK])
            out.append(w @ h)
        return torch.cat(out)


def _patched_layer_forward(self, hidden_states, position_embeddings, attention_mask=None,
                           position_ids=None, past_key_values=None, **kwargs):
    mode = MODES.get(self.layer_idx, "full") if ACTIVE[0] else "full"
    if mode == "skip":
        return hidden_states
    if mode != "nomix":
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            h = self.linear_attn(hidden_states=h, cache_params=past_key_values,
                                 attention_mask=attention_mask, **kwargs)
        else:
            h, _ = self.self_attn(hidden_states=h, attention_mask=attention_mask,
                                  position_ids=position_ids, past_key_values=past_key_values,
                                  position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + h
    if mode != "nomlp":
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
    return hidden_states


def _capture_hook(layer_idx: int):
    def hook(module, args, out):
        if CAPTURE is not None:
            h = out[0] if isinstance(out, tuple) else out
            CAPTURE[layer_idx] = h[0, -1].clone()
    return hook


class Cerno:
    def __init__(self, device: str = "cuda"):
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        t0 = time.time()
        self.device = torch.device(device)
        head_device = torch.device(os.environ.get("LAYERSKIP_HEAD", device))
        cfg = json.loads((CERNO_DIR / "hf" / "config.json").read_text())
        cfg.pop("quantization_config", None)
        cfg["tie_word_embeddings"] = False
        config = Qwen3_5TextConfig(**cfg)
        with torch.device("meta"):
            model = Qwen3_5ForCausalLM(config)
        pq = PQ(CERNO_PQ)
        f32: dict[str, torch.Tensor] = {}
        for name, e in pq.entries.items():
            if e["dtype"] == F32:
                a = torch.from_numpy(pq.get(name))
                if name.endswith(ZERO_CENTERED) or name == "norm.weight":
                    a = a - 1.0                               # .pq stores 1 + w
                if name.endswith("linear_attn.conv1d.weight"):
                    a = a[:, None, :]
                if name.endswith("linear_attn.A"):
                    name = name[:-1] + "A_log"
                    a = torch.log(-a)
                f32["model." + name] = a.contiguous().to(self.device)
                continue
            c, s = (torch.from_numpy(x) for x in pq.codes_scales(name))
            if name == "lm_head.weight":
                assert e["dtype"] == Q8_G32
                self.head = Q8Head(c, s, head_device)
                continue
            parent, attr = self._module(model, name)
            if name == "embed_tokens.weight":
                assert e["dtype"] == Q8_G32
                setattr(parent, attr, Q8Embedding(c, s, self.device))      # stays on the CPU
            else:
                assert e["dtype"] == Q4_G32, name
                setattr(parent, attr, Q4Linear(c.to(self.device), s.to(self.device)))
        missing, unexpected = model.load_state_dict(f32, strict=False, assign=True)
        bad = [m for m in missing if "rotary" not in m and m != "lm_head.weight"
               and not m.endswith(("codes", "scales", "packed"))]
        if unexpected or bad:
            raise RuntimeError(f"missing {bad} unexpected {unexpected}")
        model.model.rotary_emb = mq.Qwen3_5TextRotaryEmbedding(config).to(self.device)
        model.lm_head = None
        model.eval()
        for n, t in list(model.named_parameters()) + list(model.named_buffers()):
            if t.device.type == "meta":
                raise RuntimeError(f"{n} still on meta")
        mq.Qwen3_5DecoderLayer.forward = _patched_layer_forward
        for i, layer in enumerate(model.model.layers):
            layer.layer_idx = i
            layer.register_forward_hook(_capture_hook(i))
        self.model, self.lm, self.config = model, model.model, config
        del pq                                                 # release the .pq memmap (host RAM)
        mem = f", VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB" if self.device.type == "cuda" else ""
        logger.info("Cerno loaded on %s (head on %s) in %.0f s, attn impl %s%s, host maxrss %.1f GiB",
                    device, head_device, time.time() - t0, config._attn_implementation, mem,
                    host_rss_gib())

    @staticmethod
    def _module(model, name: str):
        parts = ("model." + name).split(".")[:-1]          # drop '.weight'
        m = model
        for p in parts[:-1]:
            m = getattr(m, p)
        return m, parts[-1]

    def forward(self, ids: list[int], cache=None, past: int = 0):
        """Hidden state (after the final norm) at the last position, and the cache."""
        n = len(ids)
        x = torch.tensor([ids], device=self.device)
        pos = torch.arange(past, past + n, device=self.device)[None]
        q = pos[0][:, None]
        k = torch.arange(past + n, device=self.device)[None, :]
        mask = (k <= q)[None, None]                          # [1, 1, n, past + n] bool, True = attend
        out = self.lm(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=True,
                      attention_mask={"full_attention": mask, "linear_attention": None})
        return out.last_hidden_state[0, -1], out.past_key_values

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

    def logits_from_layer(self, h_layer: torch.Tensor) -> torch.Tensor:
        """Early exit: final norm + head on a layer's output."""
        return self.head(self.lm.norm(h_layer[None])[0])


def load_rows() -> list[dict]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(CERNO_DIR / "cerno-4b-q4_0.tokenizer" / "tokenizer.json"))
    rows = []
    for li, line in enumerate(CERNO_REF.read_text(encoding="utf-8").splitlines()):
        req = json.loads(line)
        for row in req["rows"]:
            lab = []
            for s in row["labels"]:
                t = tok.encode(s, add_special_tokens=False).ids
                assert len(t) == 1, s
                lab.append(t[0])
            rows.append({"request": req["name"], "req_index": li, "question": row["question"],
                         "ids": row["server_ids"], "labels": lab, "ref": row["probs"]})
    return rows


def readout(logits: torch.Tensor, labels: list[int]) -> list[float]:
    lg = logits.double()
    top = torch.topk(lg, TOP_N).values[-1]
    z = lg[labels] / TEMPERATURE
    keep = lg[labels] >= top
    if not keep.any():
        return [1.0 / len(labels)] * len(labels)
    e = torch.where(keep, torch.exp(z - z[keep].max()), torch.zeros_like(z))
    return (e / e.sum()).tolist()


def common_prefix(seqs: list[list[int]]) -> int:
    short = min(len(s) for s in seqs) - 1
    n = 0
    while n < short and all(s[n] == seqs[0][n] for s in seqs):
        n += 1
    return n


class Evaluator:
    def __init__(self, c: Cerno, rows: list[dict]):
        self.c = c
        self.by_req: dict[int, list[dict]] = {}
        for r in rows:
            self.by_req.setdefault(r["req_index"], []).append(r)
        self.prefix_len = {ri: (common_prefix([r["ids"] for r in rs]) if len(rs) > 1
                                else len(rs[0]["ids"]) - 1) for ri, rs in self.by_req.items()}
        self._full_prefix: dict[int, object] = {}

    def prefix_cache(self, ri: int, full: bool):
        """The request's prefix cache; the full model's caches are kept for question-only variants."""
        if full and ri in self._full_prefix:
            return self._full_prefix[ri]
        n = self.prefix_len[ri]
        ACTIVE[0] = not full
        _, cache = self.c.forward(self.by_req[ri][0]["ids"][:n])
        ACTIVE[0] = True
        if full:
            self._full_prefix[ri] = cache
        return cache

    def evaluate(self, modes: dict[int, str], question_only: bool = False,
                 exits: list[int] | None = None) -> list[dict]:
        """Row results {request, question, p, ref[, exit: {k: p}]} for the given layer modes."""
        global CAPTURE
        MODES.clear()
        MODES.update(modes)
        out = []
        for ri, rs in self.by_req.items():
            n = self.prefix_len[ri]
            base = self.prefix_cache(ri, full=question_only or not modes)
            for r in rs:
                CAPTURE = {} if exits else None
                h, _ = self.c.forward(r["ids"][n:], copy.deepcopy(base), past=n)
                res = {"request": r["request"], "question": r["question"],
                       "p": readout(self.c.logits(h), r["labels"]), "ref": r["ref"]}
                if exits:
                    res["exit"] = {k: readout(self.c.logits_from_layer(CAPTURE[k]), r["labels"])
                                   for k in exits}
                CAPTURE = None
                out.append(res)
        MODES.clear()
        return out


def compare(res: list[dict], base: list[dict] | None = None, key: str = "p") -> dict:
    ps = [r[key] for r in res]
    d_ref = [max(abs(a - b) for a, b in zip(p, r["ref"])) for p, r in zip(ps, res)]
    top_ref = [int(np.argmax(p) == np.argmax(r["ref"])) for p, r in zip(ps, res)]
    out = {"rows": len(res), "max_dp_ref": float(max(d_ref)), "mean_dp_ref": float(np.mean(d_ref)),
           "same_top_ref": int(sum(top_ref)),
           "worst_ref": res[int(np.argmax(d_ref))]["question"]}
    if base is not None:
        d_b = [max(abs(a - b) for a, b in zip(p, q["p"])) for p, q in zip(ps, base)]
        top_b = [int(np.argmax(p) == np.argmax(q["p"])) for p, q in zip(ps, base)]
        out.update({"max_dp_base": float(max(d_b)), "mean_dp_base": float(np.mean(d_b)),
                    "same_top_base": int(sum(top_b))})
    return out


def passes(cmp: dict, max_dp: float = 0.0146, min_top: int = 51) -> bool:
    return cmp["max_dp_ref"] <= max_dp and cmp["same_top_ref"] >= min_top


def save_json(name: str, obj: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    logger.info("wrote %s", path)
    return path
