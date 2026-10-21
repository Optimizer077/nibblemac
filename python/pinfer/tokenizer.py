"""Tokenizer for pinfer models: HF ``tokenizers`` + the files that come with the checkpoint.

A tokenizer is a directory (or a ``tokenizer.json`` file) holding
  tokenizer.json           required (HF fast-tokenizer file)
  tokenizer_config.json    optional: eos token, chat template
  chat_template.jinja      optional: chat template (preferred over tokenizer_config.json)

For a model ``path/to/name.pq`` the tokenizer is found by :func:`resolve_tokenizer_path`:
  1. the explicit ``tokenizer=`` argument,
  2. ``$PINFER_TOKENIZER``,
  3. the sidecar ``path/to/name.tokenizer`` — a directory (written by ``pinfer convert``), or a
     text file whose first line is a path to the tokenizer (relative to the .pq directory),
  4. ``path/to/tokenizer.json`` (a .pq stored inside the HF model directory).
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .errors import TokenizerNotFoundError

logger = logging.getLogger(__name__)

# Qwen end-of-turn / end-of-text markers, used as stop tokens in addition to the eos token
# named in tokenizer_config.json and the model's eos id.
DEFAULT_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def _tokenizer_dir(path: Path) -> Path:
    """Directory that holds tokenizer.json (``path`` itself or its parent for a file)."""
    return path if path.is_dir() else path.parent


def _sidecar_target(sidecar: Path) -> Path | None:
    """Resolve a ``name.tokenizer`` sidecar (directory, or text file holding a path)."""
    if sidecar.is_dir():
        return sidecar
    if sidecar.is_file():
        lines = sidecar.read_text(encoding="utf-8").strip().splitlines()
        target = Path(lines[0].strip()) if lines else None
        if target is None:
            return None
        return target if target.is_absolute() else (sidecar.parent / target)
    return None


def resolve_tokenizer_path(model_path: str | os.PathLike[str] | None,
                           tokenizer: str | os.PathLike[str] | None = None) -> Path | None:
    """Find the tokenizer for a model; return None if there is none (see module docstring)."""
    if tokenizer is not None:
        return Path(tokenizer)
    env = os.environ.get("PINFER_TOKENIZER")
    if env:
        return Path(env)
    if model_path is None:
        return None
    mp = Path(model_path)
    sidecar = _sidecar_target(mp.with_suffix(".tokenizer"))
    if sidecar is not None:
        return sidecar
    beside = mp.parent / "tokenizer.json"
    if beside.is_file():
        return beside
    return None


def _load_chat_template(directory: Path, config: dict) -> str | None:
    jinja = directory / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    tmpl = config.get("chat_template")
    if isinstance(tmpl, list):  # [{"name": "default", "template": ...}, ...]
        named = {t.get("name"): t.get("template") for t in tmpl if isinstance(t, dict)}
        tmpl = named.get("default") or next(iter(named.values()), None)
    return tmpl if isinstance(tmpl, str) else None


class Tokenizer:
    """Text <-> token ids for one model.

    Attributes:
        source: Directory the tokenizer files were read from.
        config: Parsed ``tokenizer_config.json`` (empty if absent).
        chat_template: Jinja chat template text, or None (the built-in Qwen template is used).
    """

    def __init__(self, hf_tokenizer: Any, source: Path | None = None,
                 config: dict | None = None, chat_template: str | None = None) -> None:
        self._tok = hf_tokenizer
        self.source = source
        self.config = config or {}
        self.chat_template = chat_template

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> Tokenizer:
        """Load from a directory containing ``tokenizer.json`` or from the file itself.

        Raises:
            TokenizerNotFoundError: ``tokenizer.json`` does not exist.
            ImportError: The ``tokenizers`` package is not installed.
        """
        p = Path(path)
        tj = p / "tokenizer.json" if p.is_dir() else p
        if not tj.is_file():
            raise TokenizerNotFoundError(f"tokenizer.json not found at {tj}")
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as e:  # pragma: no cover - dependency of the package
            raise ImportError("pinfer needs the 'tokenizers' package: "
                              "pip install tokenizers") from e
        directory = _tokenizer_dir(tj)
        cfg_path = directory / "tokenizer_config.json"
        config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        tok = cls(HFTokenizer.from_file(str(tj)), directory, config,
                  _load_chat_template(directory, config))
        logger.debug("tokenizer loaded from %s (vocab %d)", tj, tok.vocab_size)
        return tok

    @property
    def vocab_size(self) -> int:
        """Number of entries in the tokenizer vocabulary (including added tokens)."""
        return int(self._tok.get_vocab_size(with_added_tokens=True))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Text -> token ids. Qwen adds no BOS, so ``add_special_tokens`` rarely matters."""
        return list(self._tok.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Token ids -> text."""
        return str(self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens))

    def token_to_id(self, token: str) -> int | None:
        """Id of a token string such as ``"<|im_end|>"``, or None."""
        tid = self._tok.token_to_id(token)
        return None if tid is None else int(tid)

    def stop_token_ids(self) -> set[int]:
        """Ids that end a generation: the configured eos token plus Qwen's end markers."""
        names = list(DEFAULT_STOP_TOKENS)
        eos = self.config.get("eos_token")
        if isinstance(eos, dict):
            eos = eos.get("content")
        if isinstance(eos, str):
            names.append(eos)
        return {i for i in (self.token_to_id(n) for n in names) if i is not None}


class IncrementalDecoder:
    """Turn a growing list of token ids into text deltas without splitting UTF-8 characters.

    Byte-level BPE tokens can end in the middle of a multi-byte character; the decoder holds
    such text back (it would decode to U+FFFD) until the character is complete.
    """

    def __init__(self, tokenizer: Tokenizer, skip_special_tokens: bool = True) -> None:
        self._tok = tokenizer
        self._skip = skip_special_tokens
        self.ids: list[int] = []
        self._prefix = 0  # start of the window that is decoded together
        self._read = 0    # tokens whose text has been emitted

    def _decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=self._skip)

    def push(self, token_id: int) -> str:
        """Add one token; return the newly completed text (may be empty)."""
        self.ids.append(token_id)
        prefix_text = self._decode(self.ids[self._prefix:self._read])
        new_text = self._decode(self.ids[self._prefix:])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self._prefix, self._read = self._read, len(self.ids)
            return new_text[len(prefix_text):]
        return ""

    def flush(self) -> str:
        """Return any held-back text (incomplete characters become U+FFFD)."""
        prefix_text = self._decode(self.ids[self._prefix:self._read])
        new_text = self._decode(self.ids[self._prefix:])
        self._prefix = self._read = len(self.ids)
        return new_text[len(prefix_text):]
