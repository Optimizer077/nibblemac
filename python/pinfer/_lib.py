"""Locate, load and type the C library ``libpinfer.so`` (ctypes).

Search order (first hit wins):
  1. the ``path`` argument of :func:`load_library`,
  2. the ``PINFER_LIB`` environment variable (full path to the .so),
  3. ``libpinfer.so`` inside this package (what the platform wheels ship),
  4. a source-tree build: ``build/{host,aarch64}/engine/lib/libpinfer.so`` (development),
  5. the system loader (``LD_LIBRARY_PATH``, ldconfig cache, ``make install`` prefixes).

The library is loaded lazily, so ``import pinfer`` works on any platform (for example to run
``pinfer convert`` on Windows); only creating a :class:`pinfer.Model` needs the .so.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform
import threading
from pathlib import Path
from typing import Any

from .errors import LibraryNotFoundError

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).resolve().parent
_LIB_NAME = "libpinfer.so"
_lock = threading.Lock()
_cache: dict[str, Lib] = {}


class Params(ctypes.Structure):
    """Mirror of ``pinfer_params`` in pinfer.h."""

    _fields_ = [("n_threads", ctypes.c_int), ("max_ctx", ctypes.c_int)]


class SpecParams(ctypes.Structure):
    """Mirror of ``pinfer_spec_params`` (speculative decoding, optional API)."""

    _fields_ = [("k", ctypes.c_int), ("draft_vocab", ctypes.c_int)]


class SpecStats(ctypes.Structure):
    """Mirror of ``pinfer_spec_stats``."""

    _fields_ = [("rounds", ctypes.c_int64), ("drafted", ctypes.c_int64),
                ("accepted", ctypes.c_int64)]


_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_int32_p = ctypes.POINTER(ctypes.c_int32)
_c_int_p = ctypes.POINTER(ctypes.c_int)

# int (*on_token)(int32_t tok, void *user) of pinfer_gen_opts.
ON_TOKEN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int32, ctypes.c_void_p)


class GenOpts(ctypes.Structure):
    """Mirror of ``pinfer_gen_opts`` (stop tokens + per-token callback, optional API)."""

    _fields_ = [("stop_ids", _c_int32_p), ("n_stop", ctypes.c_int),
                ("on_token", ON_TOKEN), ("user", ctypes.c_void_p)]

# name -> (restype, argtypes) for every function of the v0.1 C API.
_REQUIRED = {
    "pinfer_default_params": (Params, []),
    "pinfer_load": (ctypes.c_void_p, [ctypes.c_char_p, ctypes.POINTER(Params), _c_int_p]),
    "pinfer_free": (None, [ctypes.c_void_p]),
    "pinfer_reset": (None, [ctypes.c_void_p]),
    "pinfer_eval": (ctypes.c_int, [ctypes.c_void_p, _c_int32_p, ctypes.c_int, _c_float_p]),
    "pinfer_n_vocab": (ctypes.c_int, [ctypes.c_void_p]),
    "pinfer_n_past": (ctypes.c_int, [ctypes.c_void_p]),
    "pinfer_eos_id": (ctypes.c_int, [ctypes.c_void_p]),
    "pinfer_argmax": (ctypes.c_int, [_c_float_p, ctypes.c_int]),
    "pinfer_version": (ctypes.c_char_p, []),
}


class Lib:
    """A loaded ``libpinfer.so`` with typed entry points.

    Attributes:
        path: File the library was loaded from (or the soname given to the loader).
        cdll: The raw :class:`ctypes.CDLL`.
        version: ``pinfer_version()`` of the loaded library.
    """

    # Typed in __init__ from _REQUIRED (declared here for type checkers).
    pinfer_default_params: Any
    pinfer_load: Any
    pinfer_free: Any
    pinfer_reset: Any
    pinfer_eval: Any
    pinfer_n_vocab: Any
    pinfer_n_past: Any
    pinfer_eos_id: Any
    pinfer_argmax: Any
    pinfer_version: Any

    def __init__(self, path: str, cdll: ctypes.CDLL) -> None:
        self.path = path
        self.cdll = cdll
        for name, (restype, argtypes) in _REQUIRED.items():
            try:
                fn = getattr(cdll, name)
            except AttributeError as e:
                raise LibraryNotFoundError(f"{path}: missing symbol {name}; "
                                           "library too old or not pinfer") from e
            fn.restype = restype
            fn.argtypes = argtypes
            setattr(self, name, fn)
        self.version: str = self.pinfer_version().decode()
        # Optional API (feature-detected; None when the library predates it).
        self.pinfer_has_mtp = self.optional("pinfer_has_mtp", ctypes.c_int, [ctypes.c_void_p])
        self.pinfer_default_spec_params = self.optional("pinfer_default_spec_params",
                                                        SpecParams, [])
        self.pinfer_generate_greedy = self.optional(
            "pinfer_generate_greedy", ctypes.c_int,
            [ctypes.c_void_p, _c_int32_p, ctypes.c_int, _c_int32_p, ctypes.c_int,
             ctypes.POINTER(SpecParams), ctypes.POINTER(SpecStats)])
        self.pinfer_generate_greedy_ex = self.optional(
            "pinfer_generate_greedy_ex", ctypes.c_int,
            [ctypes.c_void_p, _c_int32_p, ctypes.c_int, _c_int32_p, ctypes.c_int,
             ctypes.POINTER(SpecParams), ctypes.POINTER(GenOpts), ctypes.POINTER(SpecStats),
             _c_int_p])

    def has(self, name: str) -> bool:
        """Return True if the library exports ``name`` (feature detection for newer APIs)."""
        return hasattr(self.cdll, name)

    def optional(self, name: str, restype: Any, argtypes: list[Any]) -> Any | None:
        """Return a typed function pointer for ``name``, or None if the library lacks it."""
        if not self.has(name):
            return None
        fn = getattr(self.cdll, name)
        fn.restype = restype
        fn.argtypes = argtypes
        return fn


def _source_tree_candidates() -> list[Path]:
    """Libraries built by ``make -C engine lib`` when running from a repository checkout."""
    root = _PKG_DIR.parent.parent  # python/pinfer -> repo root
    sub = "aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "host"
    return [root / "build" / sub / "engine" / "lib" / _LIB_NAME]


def candidate_paths(path: str | os.PathLike[str] | None = None) -> list[str]:
    """Return the library locations tried by :func:`load_library`, in order."""
    out: list[str] = []
    if path:
        out.append(os.fspath(path))
    env = os.environ.get("PINFER_LIB")
    if env:
        out.append(env)
    out.append(str(_PKG_DIR / _LIB_NAME))
    out.extend(str(p) for p in _source_tree_candidates())
    found = ctypes.util.find_library("pinfer")
    if found:
        out.append(found)
    out.append(_LIB_NAME)  # plain dlopen search (LD_LIBRARY_PATH, /etc/ld.so.cache)
    return out


def load_library(path: str | os.PathLike[str] | None = None) -> Lib:
    """Load ``libpinfer.so`` (cached per resolved path).

    Args:
        path: Explicit library path. If given, no other location is tried.

    Raises:
        LibraryNotFoundError: No candidate could be loaded.
    """
    candidates = [os.fspath(path)] if path else candidate_paths()
    errors: list[str] = []
    with _lock:
        for cand in candidates:
            is_file = os.sep in cand or "/" in cand
            if is_file and not os.path.exists(cand):
                errors.append(f"{cand}: not found")
                continue
            key = os.path.realpath(cand) if is_file else cand
            if key in _cache:
                return _cache[key]
            try:
                cdll = ctypes.CDLL(cand)
            except OSError as e:
                errors.append(f"{cand}: {e}")
                continue
            lib = Lib(cand, cdll)
            logger.debug("loaded %s (pinfer %s)", cand, lib.version)
            _cache[key] = lib
            return lib
    raise LibraryNotFoundError(
        "cannot load libpinfer.so; set PINFER_LIB=/path/to/libpinfer.so or install a "
        "platform wheel. Tried:\n  " + "\n  ".join(errors))
