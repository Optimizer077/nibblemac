"""Download a pinned public GGUF baseline and verify its published LFS digest.

Weights remain in ignored models/. Uses no Hugging Face credentials.
Build llama.cpp under ignored build/ and convert MLX separately (see the report).
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
REPO = "unsloth/Qwen3.5-0.8B-GGUF"
REVISION = "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0"
FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"
EXPECTED = "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"


def main() -> None:
    out = ROOT / "models/public-baselines"
    out.mkdir(parents=True, exist_ok=True)
    path = out / FILE
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}"
    if not path.exists():
        partial = path.with_name(path.name + ".partial")
        subprocess.run(["curl", "--fail", "--location", "--retry", "4", "--continue-at", "-",
                        "--output", str(partial), url], check=True)
        partial.rename(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED:
        raise ValueError("Public GGUF SHA-256 mismatch; file retained for diagnosis")
    (out / "gguf-source.json").write_text(json.dumps({
        "repository": REPO, "revision": REVISION, "file": FILE, "url": url,
        "sha256": EXPECTED, "bytes": path.stat().st_size, "license": "Apache-2.0",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Public Q4_K_M conversion; not bit-identical to nibblemac PQ_Q4K_G32 weights."
    }, indent=2) + "\n")
    print(f"Verified {FILE}: {path.stat().st_size} bytes, sha256={EXPECTED}")


if __name__ == "__main__":
    main()
