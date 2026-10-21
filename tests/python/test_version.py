"""One version everywhere: pinfer.h, python/pyproject.toml, pinfer.__version__, the loaded
library and its SONAME (0.x: libpinfer.so.MAJOR.MINOR, engine/Makefile)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pinfer
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _header_version() -> str:
    h = ROOT / "engine" / "include" / "pinfer.h"
    if not h.is_file():
        pytest.skip("needs a source checkout (engine/include/pinfer.h)")
    text = h.read_text(encoding="utf-8")
    parts = [re.search(rf"#define PINFER_VERSION_{k}\s+(\d+)", text) for k in
             ("MAJOR", "MINOR", "PATCH")]
    assert all(parts), "PINFER_VERSION_* macros not found"
    return ".".join(m.group(1) for m in parts if m)


def test_header_pyproject_and_package_agree() -> None:
    ver = _header_version()
    pp = (ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pp, re.MULTILINE)
    assert m and m.group(1) == ver == pinfer.__version__


def test_library_version_and_soname(lib: Any) -> None:
    assert lib.version == pinfer.__version__
    info = Path(lib.path).with_name("BUILD_INFO.json")
    if not info.is_file():
        pytest.skip("no BUILD_INFO.json next to the library")
    major, minor, _ = pinfer.__version__.split(".")
    build = json.loads(info.read_text(encoding="utf-8"))
    assert build["version"] == pinfer.__version__
    assert build["soname"] == f"libpinfer.so.{major}.{minor}"
