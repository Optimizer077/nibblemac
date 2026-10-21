"""Convert a Hugging Face Qwen3.5 checkpoint (safetensors) into a pinfer `.pq` file (v1).

Text model (+ MTP head) only; vision tensors are skipped. 2-D weights stay BF16 (bit-exact
copy) or are quantized with tools/quant.py; norms, conv and gate parameters become F32 with
the conventions in engine/FORMAT.md. Needs only numpy.

    python tools/convert.py models/Qwen3.5-0.8B models/pq/qwen35-0.8b-bf16.pq
    python tools/convert.py models/Qwen3.5-0.8B models/pq/qwen35-0.8b-q4.pq --quant q4 --head q8
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import struct
from pathlib import Path

import numpy as np

from quant import Q4_G32, Q4K_G32, Q8_G32, quantize_pack

logger = logging.getLogger(__name__)

MAGIC = b"PINFERQ\0"
VERSION = 1
HEADER_BYTES = 4096
ALIGN = 64
F32, BF16 = 0, 1
QUANT_IDS = {"bf16": BF16, "q8": Q8_G32, "q4": Q4_G32, "q4k": Q4K_G32}
HEADER_FMT = "<8sIIQQ" + "I" * 18 + "ff"   # must match pq_header in engine/src/pq_format.h
ENTRY_FMT = "<80sII4IQQ8x"                 # must match pq_tensor_entry (128 bytes)
TEXT_PREFIX = "model.language_model."
# Qwen3_5RMSNorm multiplies by (1 + w): store the effective multiplier.
ZERO_CENTERED_SUFFIXES = ("input_layernorm.weight", "post_attention_layernorm.weight",
                          "q_norm.weight", "k_norm.weight")
ZERO_CENTERED_EXACT = ("norm.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
                       "mtp.pre_fc_norm_hidden.weight")


def read_safetensors(paths: list[Path]) -> dict[str, tuple[str, list[int], memoryview]]:
    """Map tensor name -> (dtype string, shape, raw bytes) without copying file data."""
    out = {}
    for path in paths:
        raw = memoryview(path.read_bytes())
        n = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(bytes(raw[8:8 + n]))
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            out[name] = (info["dtype"], info["shape"], raw[base + start:base + end])
    return out


def bf16_to_f32(buf: memoryview) -> np.ndarray:
    return (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def to_f32(dtype: str, buf: memoryview) -> np.ndarray:
    if dtype == "BF16":
        return bf16_to_f32(buf)
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32).copy()
    raise ValueError(f"unsupported source dtype {dtype}")


def pq_name(hf_name: str) -> str | None:
    if hf_name.startswith(TEXT_PREFIX):
        return hf_name[len(TEXT_PREFIX):]
    if hf_name.startswith("mtp."):
        return hf_name
    return None  # vision tower etc.


def convert_tensor(name: str, dtype: str, shape: list[int], buf: memoryview, target: int = 1
                   ) -> tuple[int, list[int], bytes]:
    """Return (pq dtype, shape, bytes) for one tensor following FORMAT.md conventions.
    `target` is the pq dtype for 2-D matrices: BF16 (1) or a quantized dtype."""
    if name.endswith(ZERO_CENTERED_SUFFIXES) or name in ZERO_CENTERED_EXACT:
        return F32, shape, (1.0 + to_f32(dtype, buf)).astype(np.float32).tobytes()
    if name.endswith("linear_attn.conv1d.weight"):          # [C, 1, K] -> [C, K]
        return F32, [shape[0], shape[2]], to_f32(dtype, buf).astype(np.float32).tobytes()
    if len(shape) == 1:                                       # gated norm, A_log, dt_bias
        return F32, shape, to_f32(dtype, buf).astype(np.float32).tobytes()
    if len(shape) == 2 and target != BF16:
        packed, _ = quantize_pack(target, to_f32(dtype, buf).reshape(shape))
        return target, shape, packed
    if len(shape) == 2 and dtype == "BF16":
        return BF16, shape, bytes(buf)
    if len(shape) == 2 and dtype == "F32":
        f = np.frombuffer(buf, dtype=np.float32)
        return BF16, shape, (f.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    raise ValueError(f"{name}: unexpected shape {shape} / dtype {dtype}")


def build_header_fields(cfg: dict, n_tensors: int, dir_off: int, data_off: int, has_mtp: bool
                        ) -> bytes:
    t = cfg.get("text_config", cfg)
    head_dim = t["head_dim"]
    rot_dim = int(head_dim * t["rope_parameters"].get("partial_rotary_factor", 1.0))
    mask = 0
    for i, kind in enumerate(t["layer_types"]):
        if kind == "full_attention":
            mask |= 1 << i
    eos = t.get("eos_token_id", cfg.get("eos_token_id"))
    eos = eos[0] if isinstance(eos, list) else eos
    fields = (MAGIC, VERSION, n_tensors, dir_off, data_off,
              1, t["hidden_size"], t["num_hidden_layers"], t["vocab_size"], t["intermediate_size"],
              t["num_attention_heads"], t["num_key_value_heads"], head_dim, rot_dim,
              t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"],
              t["linear_value_head_dim"], t["linear_conv_kernel_dim"],
              mask, eos, int(t.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True))),
              int(has_mtp), t["rms_norm_eps"], t["rope_parameters"]["rope_theta"])
    return struct.pack(HEADER_FMT, *fields)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("model_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--no-mtp", action="store_true", help="drop the MTP head")
    p.add_argument("--quant", choices=sorted(QUANT_IDS), default="bf16",
                   help="dtype for layer and MTP matrices")
    p.add_argument("--head", choices=sorted(QUANT_IDS), default=None,
                   help="dtype for embed_tokens (= tied output head); default: --quant")
    p.add_argument("--override", action="append", default=[], metavar="REGEX=DTYPE",
                   help="per-tensor dtype, first match wins, e.g. 'v_proj=q8' 'mtp\\.fc=q8'")
    a = p.parse_args()
    overrides = []
    for spec in a.override:
        pattern, _, dt = spec.rpartition("=")
        if dt not in QUANT_IDS or not pattern:
            p.error(f"bad --override {spec!r}")
        overrides.append((re.compile(pattern), dt))
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = json.loads((a.model_dir / "config.json").read_text())
    src = read_safetensors(sorted(a.model_dir.glob("*.safetensors")))
    tensors = []
    for hf_name in sorted(src):
        name = pq_name(hf_name)
        if name is None or (a.no_mtp and name.startswith("mtp.")):
            continue
        if len(name.encode()) >= 80:
            raise ValueError(f"name too long: {name}")
        choice = a.head if (name == "embed_tokens.weight" and a.head) else a.quant
        choice = next((dt for rx, dt in overrides if rx.search(name)), choice)
        tensors.append((name, *convert_tensor(name, *src[hf_name], target=QUANT_IDS[choice])))
        logger.debug("%s -> %s", name, choice)
    has_mtp = any(t[0].startswith("mtp.") for t in tensors)

    dir_off = HEADER_BYTES
    data_off = -(-(dir_off + 128 * len(tensors)) // 4096) * 4096
    entries, offset = [], data_off
    for name, dtype, shape, data in tensors:
        entries.append((name, dtype, shape, offset, len(data)))
        offset = -(-(offset + len(data)) // ALIGN) * ALIGN

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("wb") as f:
        f.write(build_header_fields(cfg, len(tensors), dir_off, data_off, has_mtp)
                .ljust(HEADER_BYTES, b"\0"))
        for name, dtype, shape, off, nbytes in entries:
            dims = list(shape) + [0] * (4 - len(shape))
            f.write(struct.pack(ENTRY_FMT, name.encode(), dtype, len(shape), *dims, off, nbytes))
        for (_, _, _, data), (_, _, _, off, _) in zip(tensors, entries):
            f.write(b"\0" * (off - f.tell()))
            f.write(data)
    total = sum(len(t[3]) for t in tensors)
    logger.info("wrote %s: %d tensors, %.1f MB data, mtp=%s", a.out, len(tensors), total / 1e6, has_mtp)


if __name__ == "__main__":
    main()
