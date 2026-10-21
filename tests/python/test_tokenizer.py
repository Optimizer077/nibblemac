"""Tokenizer: round trips, special tokens, incremental decoding, tokenizer discovery."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from pinfer import IncrementalDecoder, Tokenizer, resolve_tokenizer_path

TEXTS = [
    "The Raspberry Pi 4 is a small computer that",
    "라즈베리 파이 4는 작은 컴퓨터입니다.",
    "Emoji: 🍓🥧 and accents: café, naïve, Ελληνικά, 日本語",
    "def f(x):\n    return x ** 2  # comment\n\n\tTabs\r\nCRLF",
    "  leading and trailing spaces  ",
    "<|im_start|>user\nhi<|im_end|>\n",
]


@pytest.mark.parametrize("text", TEXTS)
def test_round_trip(tokenizer: Any, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens(tokenizer: Any) -> None:
    assert tokenizer.token_to_id("<|im_end|>") == 248046
    assert tokenizer.token_to_id("<|endoftext|>") == 248044
    assert {248044, 248046} <= tokenizer.stop_token_ids()
    assert tokenizer.encode("<|im_start|>")[0] == 248045  # specials are not split
    assert tokenizer.vocab_size <= 248320
    assert tokenizer.chat_template and "im_start" in tokenizer.chat_template


@pytest.mark.parametrize("text", TEXTS[1:3])
def test_incremental_decoder(tokenizer: Any, text: str) -> None:
    ids = tokenizer.encode(text)
    dec = IncrementalDecoder(tokenizer, skip_special_tokens=False)
    chunks = [dec.push(t) for t in ids]
    chunks.append(dec.flush())
    assert "".join(chunks) == text
    assert not any("�" in c for c in chunks)


def test_resolution_order(tokenizer_path: Path, tmp_path: Path,
                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINFER_TOKENIZER", raising=False)
    pq = tmp_path / "m.pq"
    assert resolve_tokenizer_path(pq) is None
    # tokenizer.json beside the .pq
    shutil.copy2(tokenizer_path / "tokenizer.json", tmp_path / "tokenizer.json")
    assert resolve_tokenizer_path(pq) == tmp_path / "tokenizer.json"
    # sidecar text file holding a relative path wins over the file beside it
    (tmp_path / "tok").mkdir()
    (tmp_path / "m.tokenizer").write_text("tok\n", encoding="utf-8")
    assert resolve_tokenizer_path(pq) == tmp_path / "tok"
    # environment variable wins over the sidecar, explicit argument over everything
    monkeypatch.setenv("PINFER_TOKENIZER", str(tokenizer_path))
    assert resolve_tokenizer_path(pq) == tokenizer_path
    assert resolve_tokenizer_path(pq, "/x") == Path("/x")


def test_sidecar_directory(tokenizer_path: Path, tmp_path: Path,
                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINFER_TOKENIZER", raising=False)
    side = tmp_path / "m.tokenizer"
    side.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(tokenizer_path / name, side / name)
    found = resolve_tokenizer_path(tmp_path / "m.pq")
    assert found == side
    tok = Tokenizer.from_path(found)
    assert tok.chat_template  # from tokenizer_config.json when chat_template.jinja is absent


def test_missing_tokenizer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Tokenizer.from_path(tmp_path)
