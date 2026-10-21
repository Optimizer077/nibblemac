"""The pinfer console script."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pinfer
import pytest
from pinfer.cli import main
from pinfer.convert import find_converter, run_converter


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0
    assert pinfer.__version__ in capsys.readouterr().out


def test_generate(lib: Any, model_path: Path, tokenizer_path: Path, tokenizer: Any,
                  golden: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["generate", "-m", str(model_path), "--tokenizer", str(tokenizer_path),
               "-p", golden["prompt"], "-n", "4", "--stats"])
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == tokenizer.decode(golden["greedy_ids"][:4]) + "\n"
    assert "decode" in out.err


def test_bench_json(lib: Any, model_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["bench", "-m", str(model_path), "--prompt-tokens", "4", "-n", "2", "--json"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["n_prompt"] == 4 and rec["decode_tps"] > 0 and rec["version"] == pinfer.__version__


def test_bench_speculative(lib: Any, model_path: Path, tokenizer_path: Path,
                           capsys: pytest.CaptureFixture[str]) -> None:
    with pinfer.Model.load(model_path, tokenizer=False) as m:
        if not m.supports_speculative:
            pytest.skip("no speculative decoding")
    rc = main(["bench", "-m", str(model_path), "--prompt-tokens", "4", "-n", "4", "--spec-k",
               "2", "--json"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["spec_k"] == 2 and rec["total_tps"] > 0 and rec["runs"][0]["rounds"] > 0


def test_chat_keeps_state_between_turns(lib: Any, model_path: Path, tokenizer_path: Path,
                                        monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    """Second turn: only the new turn is prefilled (cached tokens = the whole first turn)."""
    lines = iter(["Say hi.", "Say it again."])

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    rc = main(["chat", "-m", str(model_path), "--tokenizer", str(tokenizer_path), "-n", "3",
               "--stats", "-c", "256"])
    assert rc == 0
    stats = re.findall(r"\[(\d+) prompt tok \((\d+) cached\)", capsys.readouterr().err)
    assert len(stats) == 2
    (p1, c1), (p2, c2) = [(int(p), int(c)) for p, c in stats]
    assert c1 == 0 and p2 > p1
    assert c2 >= p1 + 3 - 1   # the whole first turn (prompt + reply but its last token) reused
    assert p2 > c2            # only the new turn was prefilled


def test_info(lib: Any, model_path: Path, tokenizer_path: Path,
              capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "-m", str(model_path), "--tokenizer", str(tokenizer_path)]) == 0
    assert json.loads(capsys.readouterr().out)["n_vocab"] == 248320


def test_error_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "-m", str(tmp_path / "missing.pq")]) == 1
    assert "not found" in capsys.readouterr().err


def test_converter_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    assert find_converter().is_file()
    assert run_converter(["--help"]) == 0
    assert "safetensors" in capsys.readouterr().out
