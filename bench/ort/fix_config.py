"""Work around an onnxruntime-genai 0.18.0 CPU bug in builder output.

The builder writes `shared_initializers` (tied lm_head/embedding shared between the decoder and
the MTP session). With it, og.Model() fails on the CPU EP: "GetMemoryInfo for CPU should not be
used." (reproduced on x86 and aarch64; a Config.overlay with an empty list does not clear it).
Without it, each session loads those tensors itself from model.onnx.data; weights are unchanged.

Usage: python bench/ort/fix_config.py <export dir> [...]
Keeps the builder's file as genai_config.orig.json (idempotent).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fix(d: Path) -> None:
    orig = d / "genai_config.orig.json"
    cfg_path = d / "genai_config.json"
    if not orig.exists():
        orig.write_bytes(cfg_path.read_bytes())
    cfg = json.loads(orig.read_text(encoding="utf-8"))
    cfg["model"]["decoder"].pop("shared_initializers", None)
    if "mtp" in cfg["model"]:
        cfg["model"]["mtp"].pop("shared_initializers", None)
    cfg_path.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")
    print(f"fixed {cfg_path}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        fix(Path(arg))
