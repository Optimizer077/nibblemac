"""Automatic audit logging for the harness.

Every harness command appends one JSON line to logs/harness.jsonl (what ran, on which host,
exit code, duration, and where its output went). `run` output is streamed to the terminal
and saved in full to logs/runs/. Nothing here depends on anyone remembering to log.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
EVENTS = LOG_DIR / "harness.jsonl"
# The repository is public: hardware addresses (the Pi's /proc/cmdline carries its Ethernet MAC)
# and the strings listed in the gitignored harness/redact.secret (one per line, e.g. the Wi-Fi
# network name) never reach logs or results.
_MAC = re.compile(r"(?<![0-9A-Fa-f:])[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}(?![0-9A-Fa-f:])")
_REDACT_FILE = Path(__file__).resolve().parent / "redact.secret"
_REDACT = ([s.strip() for s in _REDACT_FILE.read_text(encoding="utf-8").splitlines() if s.strip()]
           if _REDACT_FILE.is_file() else [])


def redact(text: str) -> str:
    """Replace MAC addresses and the private strings of harness/redact.secret with placeholders."""
    text = _MAC.sub("<mac>", text)
    for s in _REDACT:
        text = text.replace(s, "<redacted>")
    return text


def stamp() -> str:
    """Filesystem-safe local timestamp, e.g. 20261021_210615."""
    return f"{datetime.now():%Y%m%d_%H%M%S}"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def log_event(event: dict) -> None:
    """Append one event (with timestamp) to logs/harness.jsonl."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(redact(json.dumps(record, ensure_ascii=False)) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe_paths(paths: list[str]) -> list[dict]:
    """Name, size and SHA-256 of pushed files (directories: file count and total size)."""
    out = []
    for p in map(Path, paths):
        if p.is_file():
            out.append({"path": p.as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        elif p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            out.append({"path": p.as_posix(), "files": len(files),
                        "bytes": sum(f.stat().st_size for f in files)})
    return out


def tee_command(argv: list[str], out_path: Path, header: str) -> int:
    """Run argv, stream stdout+stderr live to the terminal and save it all to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as log:
        log.write(redact(header) + "\n\n")
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = redact(raw.decode(errors="replace"))
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        code = proc.wait()
        log.write(f"\n[exit {code}]\n")
    return code
