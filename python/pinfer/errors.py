"""Exceptions raised by the pinfer Python package, and the C error-code mapping."""
from __future__ import annotations

# Error codes from engine/include/pinfer.h (enum PINFER_E_*).
OK = 0
E_IO = -1
E_FORMAT = -2
E_NOMEM = -3
E_ARG = -4
E_CTX_FULL = -5

_MESSAGES = {
    E_IO: "cannot open or map the model file",
    E_FORMAT: "not a valid .pq file or unsupported format version",
    E_NOMEM: "out of memory",
    E_ARG: "invalid argument",
    E_CTX_FULL: "context is full (more tokens than max_ctx)",
}


class PinferError(Exception):
    """Base class for every error raised by pinfer.

    Attributes:
        code: The C error code (``PINFER_E_*``), or ``None`` for Python-side errors.
    """

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class LibraryNotFoundError(PinferError, OSError):
    """``libpinfer.so`` could not be found or loaded."""


class ModelLoadError(PinferError):
    """``pinfer_load`` failed (I/O, memory)."""


class ModelNotFoundError(ModelLoadError, FileNotFoundError):
    """The model file does not exist."""


class ModelFormatError(ModelLoadError):
    """The file is not a valid ``.pq`` model for this engine version."""


class ContextOverflowError(PinferError):
    """More tokens than the context (``max_ctx``) can hold."""


class TokenizerNotFoundError(PinferError, FileNotFoundError):
    """No ``tokenizer.json`` was found for the model."""


class ModelClosedError(PinferError):
    """The model was used after :meth:`pinfer.Model.close`."""


def describe(code: int) -> str:
    """Return a human-readable message for a C error code."""
    return _MESSAGES.get(code, f"unknown error {code}")


def from_code(code: int, context: str) -> PinferError:
    """Build the exception that matches a C error code.

    Args:
        code: Negative ``PINFER_E_*`` value returned by the C API.
        context: What was being done, e.g. ``"pinfer_eval"`` or a file path.
    """
    msg = f"{context}: {describe(code)} (code {code})"
    if code == E_FORMAT:
        return ModelFormatError(msg, code)
    if code in (E_IO, E_NOMEM):
        return ModelLoadError(msg, code)
    if code == E_CTX_FULL:
        return ContextOverflowError(msg, code)
    return PinferError(msg, code)
