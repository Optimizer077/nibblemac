"""Fixtures for the pinfer Python test suite.

Runs against an installed pinfer (wheel) or a source checkout (PYTHONPATH=python).
Inputs, overridable by environment variables:
  PINFER_TEST_MODEL      .pq file          (default models/pq/qwen35-0.8b-bf16.pq)
  PINFER_TEST_TOKENIZER  tokenizer dir     (default models/Qwen3.5-0.8B)
  PINFER_TEST_GOLDEN     golden meta.json  (default ref/golden/qwen35-0.8b/meta.json)
  PINFER_TEST_REQUIRE=1  missing inputs FAIL instead of skipping (use for release checks)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
REQUIRE = os.environ.get("PINFER_TEST_REQUIRE") == "1"


def _path(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if os.environ.get(env) else default


MODEL = _path("PINFER_TEST_MODEL", ROOT / "models" / "pq" / "qwen35-0.8b-bf16.pq")
TOKENIZER = _path("PINFER_TEST_TOKENIZER", ROOT / "models" / "Qwen3.5-0.8B")
GOLDEN = _path("PINFER_TEST_GOLDEN", ROOT / "ref" / "golden" / "qwen35-0.8b" / "meta.json")


def _need(path: Path, what: str) -> Path:
    if not path.exists():
        msg = f"{what} not found at {path}"
        if REQUIRE:
            pytest.fail(msg)
        pytest.skip(msg)
    return path


@pytest.fixture(scope="session")
def lib() -> Any:
    """The loaded libpinfer (Linux only)."""
    if not sys.platform.startswith("linux"):
        pytest.skip("libpinfer.so needs Linux")
    import pinfer

    return pinfer.load_library()


@pytest.fixture(scope="session")
def model_path() -> Path:
    return _need(MODEL, "test model (.pq)")


@pytest.fixture(scope="session")
def tokenizer_path() -> Path:
    return _need(TOKENIZER, "tokenizer directory")


@pytest.fixture(scope="session")
def golden() -> dict[str, Any]:
    return json.loads(_need(GOLDEN, "golden meta.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def tokenizer(tokenizer_path: Path) -> Any:
    import pinfer

    return pinfer.Tokenizer.from_path(tokenizer_path)


@pytest.fixture(scope="session")
def model(lib: Any, model_path: Path, tokenizer: Any) -> Any:
    """One shared model (loading maps 1.5 GB; generation state is reset per test)."""
    import pinfer

    m = pinfer.Model.load(model_path, tokenizer=tokenizer, n_threads=4, max_ctx=256)
    yield m
    m.close()


@pytest.fixture()
def fresh(model: Any) -> Any:
    """The shared model with an empty context."""
    model.reset()
    return model
