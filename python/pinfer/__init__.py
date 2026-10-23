"""pinfer: Raspberry-Pi-4-native inference for Qwen3.5 (Python binding of libpinfer).

    import pinfer
    m = pinfer.Model.load("qwen35-0.8b-bf16.pq", tokenizer="models/Qwen3.5-0.8B")
    print(m.generate("The Raspberry Pi 4 is", max_new_tokens=32).text)
    for chunk in m.chat([{"role": "user", "content": "Hi!"}], stream=True):
        print(chunk, end="", flush=True)

See docs/library.md for the full guide.
"""
from __future__ import annotations

import logging

from ._lib import load_library
from .chat import apply_chat_template, render_qwen_builtin
from .errors import (
    ContextOverflowError,
    LibraryNotFoundError,
    ModelClosedError,
    ModelFormatError,
    ModelLoadError,
    ModelNotFoundError,
    PinferError,
    TokenizerNotFoundError,
)
from .model import Model
from .sampling import Sampler, SamplingParams
from .streaming import GenerationResult, TextStream
from .tokenizer import IncrementalDecoder, Tokenizer, resolve_tokenizer_path

__version__ = "0.3.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ContextOverflowError",
    "GenerationResult",
    "IncrementalDecoder",
    "LibraryNotFoundError",
    "Model",
    "ModelClosedError",
    "ModelFormatError",
    "ModelLoadError",
    "ModelNotFoundError",
    "PinferError",
    "Sampler",
    "SamplingParams",
    "TextStream",
    "Tokenizer",
    "TokenizerNotFoundError",
    "__version__",
    "apply_chat_template",
    "load_library",
    "render_qwen_builtin",
    "resolve_tokenizer_path",
]
