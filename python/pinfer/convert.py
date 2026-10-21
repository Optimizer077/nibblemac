"""``pinfer convert``: Hugging Face checkpoint -> ``.pq`` (wraps tools/convert.py).

The converter script ships inside the wheel as ``pinfer/_tools/convert.py`` (with its sibling
modules); in a source checkout it is ``tools/convert.py``. After converting, the tokenizer
files are copied to the sidecar directory ``<out>.tokenizer/`` so that
``Model.load(out)`` finds them without a ``tokenizer=`` argument.
"""
from __future__ import annotations

import logging
import os
import runpy
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).resolve().parent
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def find_converter() -> Path:
    """Path of convert.py (bundled copy first, then the source tree)."""
    bundled = _PKG_DIR / "_tools" / "convert.py"
    for cand in (bundled, _PKG_DIR.parent.parent / "tools" / "convert.py"):
        if cand.is_file():
            return cand
    raise FileNotFoundError("convert.py not found (neither bundled nor in a source checkout)")


def run_converter(argv: Sequence[str]) -> int:
    """Run convert.py with ``argv`` in this interpreter; return its exit status."""
    script = find_converter()
    old_argv, old_path = sys.argv[:], sys.path[:]
    sys.argv = [str(script), *argv]
    sys.path.insert(0, str(script.parent))  # sibling modules such as quant.py
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    finally:
        sys.argv, sys.path = old_argv, old_path
    return 0


def copy_tokenizer(model_dir: str | os.PathLike[str],
                   out_pq: str | os.PathLike[str]) -> Path | None:
    """Copy tokenizer files from the HF model dir to ``<out>.tokenizer/``; None if absent."""
    src = Path(model_dir)
    files = [src / f for f in TOKENIZER_FILES if (src / f).is_file()]
    if not any(f.name == "tokenizer.json" for f in files):
        logger.warning("%s has no tokenizer.json; no tokenizer sidecar written", src)
        return None
    dest = Path(out_pq).with_suffix(".tokenizer")
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest / f.name)
    logger.info("tokenizer files copied to %s", dest)
    return dest
