"""Make a text-only config for taobao-mnn/Qwen3.5-0.8B-MNN (originals untouched).

The official export sets "is_visual": true in llm_config.json, so Llm::createLLM() builds an Omni
model that also loads visual.mnn (63 MB weights) — not needed for text benchmarks and it would
inflate peak RSS. This writes:
    config_text.json      = config.json + "llm_config": "llm_config_text.json"
    llm_config_text.json  = llm_config.json with "is_visual": false
Usage: python bench/mnn/make_text_config.py <model dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    d = Path(sys.argv[1])
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    llm_cfg = json.loads((d / "llm_config.json").read_text(encoding="utf-8"))
    cfg["llm_config"] = "llm_config_text.json"
    llm_cfg["is_visual"] = False
    (d / "config_text.json").write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")
    (d / "llm_config_text.json").write_text(json.dumps(llm_cfg, indent=4, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(f"wrote {d / 'config_text.json'} and {d / 'llm_config_text.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
