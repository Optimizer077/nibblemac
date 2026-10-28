"""Download a pinned public Qwen checkpoint and tokenizer into ignored models/.

Requires curl and Python stdlib. Conversion additionally requires NumPy.
No credentials are required or read; model files are never added to git.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "model.safetensors.index.json", "model.safetensors-00001-of-00001.safetensors"]
# LFS SHA-256 IDs published by Hugging Face for the pinned revision.
EXPECTED = {
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "model.safetensors-00001-of-00001.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
}


def main() -> None:
    out = ROOT / "models/Qwen3.5-0.8B"
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        path = out / name
        if not path.exists():
            partial = path.with_name(path.name + ".partial")
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            print(f"Downloading {name}", flush=True)
            subprocess.run(["curl", "--fail", "--location", "--retry", "4", "--retry-delay", "2",
                            "--continue-at", "-", "--output", str(partial), url], check=True)
            partial.rename(path)
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        digest = h.hexdigest()
        if name in EXPECTED and digest != EXPECTED[name]:
            raise ValueError(f"SHA-256 mismatch for {name}; file retained for diagnosis")
        hashes[name] = {"bytes": path.stat().st_size, "sha256": digest}
    (out / "source.json").write_text(json.dumps({
        "model": MODEL, "revision": REVISION, "license": "Apache-2.0", "files": hashes
    }, indent=2) + "\n")
    print("Checkpoint downloaded. Convert with:")
    print(".venv/bin/python tools/convert.py models/Qwen3.5-0.8B models/qwen35-0.8b-q4k.pq --quant q4k")


if __name__ == "__main__":
    main()
