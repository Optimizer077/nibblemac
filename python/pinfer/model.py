"""High-level model API: load a .pq file, generate text, chat."""
from __future__ import annotations

import contextlib
import ctypes
import json
import logging
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import errors
from . import state as _state
from ._lib import Lib, Params, load_library, parse_kv_dtype
from .chat import apply_chat_template
from .errors import ContextOverflowError, ModelClosedError, ModelNotFoundError, PinferError
from .sampling import Sampler, SamplingParams
from .speculative import iter_speculative
from .streaming import GenerationResult, StopMatcher, TextStream
from .tokenizer import IncrementalDecoder, Tokenizer, resolve_tokenizer_path

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class Model:
    """A loaded pinfer model plus its tokenizer.

    Not thread-safe: use one Model per thread (they can share the same .pq file; the
    weights are memory-mapped and shared by the OS page cache).

    Example::

        with pinfer.Model.load("qwen35-0.8b-bf16.pq", tokenizer="Qwen3.5-0.8B/") as m:
            print(m.generate("The Raspberry Pi 4 is", max_new_tokens=32).text)
    """

    def __init__(self, lib: Lib, ctx: int, path: Path, tokenizer: Tokenizer | None,
                 n_threads: int, max_ctx: int, kv_dtype: str = "fp32") -> None:
        self._lib = lib
        self._ctx: int | None = ctx
        self.path = path
        self.tokenizer = tokenizer
        self.n_threads = n_threads
        self.max_ctx = max_ctx
        self.kv_dtype = kv_dtype
        self.n_vocab = int(lib.pinfer_n_vocab(ctx))
        self.eos_id = int(lib.pinfer_eos_id(ctx))
        self._logits = np.zeros(self.n_vocab, dtype=np.float32)
        self._logits_p = self._logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._tokens: list[int] = []      # tokens the engine has consumed since reset
        self._logits_valid = False        # _logits belongs to the last of _tokens

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def load(cls, path: str | os.PathLike[str], tokenizer: str | os.PathLike[str] | Tokenizer
             | None | bool = None, n_threads: int = 4, max_ctx: int = 4096,
             lib_path: str | os.PathLike[str] | None = None, kv_dtype: str | int = "fp32"
             ) -> Model:
        """Load a ``.pq`` model.

        Args:
            path: The ``.pq`` file (from ``pinfer convert``).
            tokenizer: A :class:`Tokenizer`, a tokenizer directory / ``tokenizer.json`` path,
                None to search next to the model (see :mod:`pinfer.tokenizer`), or False for
                none (token-id API only).
            n_threads: Worker threads for matrix ops (the Pi 4 has 4 cores).
            max_ctx: Maximum tokens kept in context (KV cache size).
            lib_path: Explicit ``libpinfer.so`` (default search: see :mod:`pinfer._lib`).
            kv_dtype: KV-cache storage: ``"fp32"`` (exact, default), ``"fp16"`` or ``"int8"``
                (smaller and faster at long context, slightly different numerics; see
                ``engine/CONTEXT.md``). Needs a libpinfer with ``pinfer_params.kv_dtype``.

        Raises:
            ModelNotFoundError, ModelFormatError, ModelLoadError, LibraryNotFoundError,
            TokenizerNotFoundError, ValueError.
        """
        if int(n_threads) < 1 or int(max_ctx) < 1:
            raise ValueError(f"n_threads and max_ctx must be >= 1 (got {n_threads}, {max_ctx})")
        kv_dtype, kv_code = parse_kv_dtype(kv_dtype)
        p = Path(path)
        if not p.is_file():
            raise ModelNotFoundError(f"model file not found: {p}", errors.E_IO)
        tok = cls._load_tokenizer(p, tokenizer)
        lib = load_library(lib_path)
        if kv_code != 0 and not lib.has("pinfer_state_size"):
            raise NotImplementedError("kv_dtype needs a libpinfer with the context-window API")
        params = Params(n_threads=int(n_threads), max_ctx=int(max_ctx), kv_dtype=kv_code)
        err = ctypes.c_int(0)
        ctx = lib.pinfer_load(os.fsencode(p), ctypes.byref(params), ctypes.byref(err))
        if not ctx:
            raise errors.from_code(err.value or errors.E_IO, str(p))
        model = cls(lib, ctx, p, tok, int(n_threads), int(max_ctx), kv_dtype)
        if tok is not None and tok.vocab_size > model.n_vocab:
            model.close()
            raise PinferError(f"tokenizer vocabulary ({tok.vocab_size}) is larger than the "
                              f"model's ({model.n_vocab}): wrong tokenizer for {p}?")
        logger.info("loaded %s: vocab %d, max_ctx %d, %d threads, libpinfer %s (%s)", p,
                    model.n_vocab, max_ctx, n_threads, lib.version, lib.path)
        return model

    @staticmethod
    def _load_tokenizer(path: Path, tokenizer: Any) -> Tokenizer | None:
        if tokenizer is False:
            return None
        if isinstance(tokenizer, Tokenizer):
            return tokenizer
        found = resolve_tokenizer_path(path, tokenizer)
        if found is None:
            logger.warning("no tokenizer found for %s; only the token-id API works "
                           "(pass tokenizer=... or set PINFER_TOKENIZER)", path)
            return None
        return Tokenizer.from_path(found)

    def close(self) -> None:
        """Free the engine context (idempotent)."""
        if self._ctx is not None:
            self._lib.pinfer_free(self._ctx)
            self._ctx = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):  # never raise from a finalizer
            self.close()

    # ------------------------------------------------------------------ info
    @property
    def n_past(self) -> int:
        """Tokens currently in the context."""
        return int(self._lib.pinfer_n_past(self._handle()))

    @property
    def version(self) -> str:
        """Version string of the loaded C library."""
        return self._lib.version

    @property
    def has_mtp(self) -> bool:
        """True if the model file has an MTP head (False with a library that cannot tell)."""
        fn = self._lib.pinfer_has_mtp
        return bool(fn is not None and fn(self._handle()))

    @property
    def supports_speculative(self) -> bool:
        """True if ``generate(speculative=True)`` works: library entry point + MTP head."""
        return self._lib.pinfer_generate_greedy is not None and self.has_mtp

    @property
    def supports_state(self) -> bool:
        """True if the library has ``pinfer_state_save`` / ``load`` (save_state, load_state)."""
        return self._lib.pinfer_state_save is not None

    @property
    def context_tokens(self) -> tuple[int, ...]:
        """The token ids the engine's context currently holds (in order)."""
        return tuple(self._tokens)

    def info(self) -> dict[str, Any]:
        """Library, build and model facts (for logs and bug reports)."""
        build: dict[str, Any] = {}
        bi = Path(self._lib.path).with_name("BUILD_INFO.json")
        if bi.is_file():
            build = json.loads(bi.read_text(encoding="utf-8"))
        return {"model": str(self.path), "library": self._lib.path, "version": self.version,
                "build": build, "n_vocab": self.n_vocab, "eos_id": self.eos_id,
                "max_ctx": self.max_ctx, "n_threads": self.n_threads, "kv_dtype": self.kv_dtype,
                "tokenizer": str(self.tokenizer.source) if self.tokenizer else None,
                "has_mtp": self.has_mtp, "speculative": self.supports_speculative}

    # ------------------------------------------------------------------ low level
    def _handle(self) -> int:
        if self._ctx is None:
            raise ModelClosedError("model is closed")
        return self._ctx

    def reset(self) -> None:
        """Clear the context (KV cache and recurrent state)."""
        self._lib.pinfer_reset(self._handle())
        self._forget_state()

    def _forget_state(self) -> None:
        """Python-side bookkeeping no longer describes the engine state (forces a reset)."""
        self._tokens = []
        self._logits_valid = False

    def _eval_into(self, tokens: Sequence[int]) -> None:
        n = len(tokens)
        if n == 0:
            return
        if self.n_past + n > self.max_ctx:
            raise ContextOverflowError(
                f"{self.n_past} tokens in context + {n} new > max_ctx {self.max_ctx}",
                errors.E_CTX_FULL)
        arr = (ctypes.c_int32 * n)(*tokens)
        rc = self._lib.pinfer_eval(self._handle(), arr, n, self._logits_p)
        if rc != errors.OK:
            raise errors.from_code(rc, "pinfer_eval")
        self._tokens.extend(int(t) for t in tokens)
        self._logits_valid = True

    def eval(self, tokens: Sequence[int]) -> np.ndarray:
        """Append tokens to the context and return the logits of the last one (a copy).

        Raises:
            ContextOverflowError: The context would exceed ``max_ctx`` (state unchanged).
            PinferError: Invalid token id (state unchanged).
        """
        if len(tokens) == 0:
            raise ValueError("eval needs at least one token")
        self._eval_into(list(tokens))
        out: np.ndarray = self._logits.copy()
        return out

    # ------------------------------------------------------------------ state
    def save_state(self) -> bytes:
        """Snapshot the context (recurrent + conv states, KV cache, MTP cache, the tokens in
        context and the last logits) as bytes; :meth:`load_state` restores it bit-exactly.
        Prefill a system prompt or a document once and reuse it for many conversations.
        Size: ~20 MB + 24 KB per token in context (fp32 KV cache). Layout: :mod:`pinfer.state`.

        Raises:
            NotImplementedError: The library predates ``pinfer_state_save``.
        """
        return _state.save(self)

    def load_state(self, data: bytes) -> None:
        """Restore a snapshot made by :meth:`save_state` on a model loaded from the same
        ``.pq`` file with the same ``kv_dtype`` (the context is unchanged on failure).

        Raises:
            ValueError: Not a pinfer state (or truncated).
            ModelFormatError: The state belongs to another model or KV format.
            ContextOverflowError: The state holds more tokens than ``max_ctx``.
        """
        _state.load(self, data)

    def _prefill(self, ids: list[int], reuse_prefix: bool) -> int:
        """Bring the context to exactly ``ids``; return how many tokens were reused."""
        if len(ids) > self.max_ctx:
            raise ContextOverflowError(f"prompt has {len(ids)} tokens > max_ctx {self.max_ctx}",
                                       errors.E_CTX_FULL)
        n = len(self._tokens)
        if reuse_prefix and 0 < n <= len(ids) and ids[:n] == self._tokens:
            if n == len(ids) and self._logits_valid:
                return n
            if n < len(ids):
                self._eval_into(ids[n:])
                return n
        self.reset()
        self._eval_into(ids)
        return 0

    def iter_tokens(self, prompt_ids: Sequence[int], max_new_tokens: int = 128,
                    sampling: SamplingParams | None = None,
                    stop_token_ids: Iterable[int] | None = None, reuse_prefix: bool = True,
                    _state: dict[str, Any] | None = None) -> Iterator[int]:
        """Token-level generation: yield new token ids (no tokenizer needed).

        Stops before a stop token (the model's eos id plus ``stop_token_ids``), after
        ``max_new_tokens``, or when the context is full. ``_state`` (internal) receives
        ``finish_reason``, ``cached_tokens``, ``prefill_seconds``, ``decode_seconds``.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        ids = [int(t) for t in prompt_ids]
        if not ids:
            raise ValueError("the prompt must contain at least one token")
        state = _state if _state is not None else {}
        stops = {self.eos_id} | {int(t) for t in (stop_token_ids or ())}
        sample = Sampler(sampling or SamplingParams())
        t0 = time.perf_counter()
        state["cached_tokens"] = self._prefill(ids, reuse_prefix)
        state["prefill_seconds"] = time.perf_counter() - t0
        state["decode_seconds"] = 0.0
        state["finish_reason"] = "length"
        for i in range(max_new_tokens):
            tok = sample(self._logits)
            if tok in stops:
                state["finish_reason"] = "stop"
                return
            yield tok
            if i + 1 == max_new_tokens:
                return
            if self.n_past >= self.max_ctx:
                state["finish_reason"] = "context"
                return
            t1 = time.perf_counter()
            self._eval_into([tok])
            state["decode_seconds"] += time.perf_counter() - t1

    # ------------------------------------------------------------------ text API
    def _require_tokenizer(self) -> Tokenizer:
        if self.tokenizer is None:
            raise errors.TokenizerNotFoundError(
                "this model was loaded without a tokenizer; pass tokenizer=... to Model.load "
                "or use iter_tokens() with token ids")
        return self.tokenizer

    def tokenize(self, text: str) -> list[int]:
        """Text -> token ids with the model's tokenizer."""
        return self._require_tokenizer().encode(text)

    def detokenize(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Token ids -> text with the model's tokenizer."""
        return self._require_tokenizer().decode(ids, skip_special_tokens=skip_special_tokens)

    def generate(self, prompt: str | Sequence[int], max_new_tokens: int = 128, *,
                 temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                 seed: int | None = None, stop: str | Sequence[str] | None = None,
                 stop_token_ids: Iterable[int] | None = None, stream: bool = False,
                 reuse_prefix: bool = True, speculative: bool = False,
                 spec_k: int | None = None, draft_vocab: int | None = None,
                 spec_adaptive: bool | None = None) -> GenerationResult | TextStream:
        """Generate a continuation of ``prompt`` (text or token ids).

        Args:
            prompt: Prompt text, or prompt token ids.
            max_new_tokens: Upper bound on generated tokens.
            temperature, top_p, top_k, seed: Sampling (default greedy; see SamplingParams).
            stop: Stop string(s); the text is cut before the first match.
            stop_token_ids: Extra stop tokens (the eos token and Qwen's ``<|im_end|>`` /
                ``<|endoftext|>`` always stop).
            stream: Return a :class:`TextStream` of text chunks instead of a result.
            reuse_prefix: Keep the engine state when the new prompt extends the tokens already
                in context (saves the shared prefill; bit-exact with the current engine, whose
                batched and token-by-token paths agree). Also used by speculative decoding
                when the library has ``pinfer_generate_greedy_cont``.
            speculative: Greedy speculative decoding with the model's MTP head (same tokens
                as plain greedy, faster). Needs greedy sampling and
                :attr:`supports_speculative`. Streams token by token with a libpinfer that
                has ``pinfer_generate_greedy_ex``; older libraries deliver the text when the
                blocking call returns. See :mod:`pinfer.speculative`.
            spec_k: Draft tokens per verification pass (default: the library's).
            draft_vocab: Draft only over token ids below this (cheaper drafts; output
                unchanged). Default: the library's.
            spec_adaptive: Let the engine lower ``spec_k`` (down to plain steps) whenever
                acceptance makes that faster; output unchanged. Default: the library's.

        Returns:
            A :class:`GenerationResult`, or a :class:`TextStream` when ``stream=True``.
        """
        tok = self._require_tokenizer()
        sampling = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if speculative and not sampling.greedy:
            raise ValueError("speculative decoding is greedy only (temperature=0)")
        if speculative and not self.supports_speculative:
            raise NotImplementedError("speculative decoding needs a libpinfer with "
                                      "pinfer_generate_greedy and a .pq with an MTP head")
        ids = tok.encode(prompt) if isinstance(prompt, str) else [int(t) for t in prompt]
        stops = [stop] if isinstance(stop, str) else list(stop or [])
        extra = {int(t) for t in stop_token_ids or ()} | tok.stop_token_ids()
        state: dict[str, Any] = {}
        result: list[GenerationResult] = []
        if speculative:
            tokens = iter_speculative(self, ids, max_new_tokens, extra | {self.eos_id}, state,
                                      spec_k, draft_vocab, spec_adaptive, reuse_prefix)
        else:
            tokens = self.iter_tokens(ids, max_new_tokens, sampling, extra, reuse_prefix, state)
        chunks = self._text_chunks(tokens, len(ids), stops, state, result)
        stream_obj = TextStream(chunks, lambda: result[0] if result else None)
        if stream:
            return stream_obj
        for _ in stream_obj:
            pass
        return result[0]

    def _text_chunks(self, tokens: Iterator[int], n_prompt: int, stops: list[str],
                     state: dict[str, Any], result: list[GenerationResult]) -> Iterator[str]:
        """Turn generated token ids into text chunks; append the final result at the end.

        The token generator is closed as soon as the text is complete (stop string) or the
        consumer closes this generator, not when it is garbage-collected: for speculative
        decoding that ends the C call (and settles the context bookkeeping) right away.
        """
        decoder = IncrementalDecoder(self._require_tokenizer())
        matcher = StopMatcher(stops)
        out_ids: list[int] = []
        try:
            for t in tokens:
                out_ids.append(t)
                chunk = matcher.feed(decoder.push(t))
                if chunk:
                    yield chunk
                if matcher.stopped:
                    state["finish_reason"] = "stop"
                    break
        finally:
            close = getattr(tokens, "close", None)
            if close is not None:
                close()
        if not matcher.stopped:  # text held back for an incomplete character / stop prefix
            tail = matcher.feed(decoder.flush())
            if matcher.stopped:
                state["finish_reason"] = "stop"
            else:
                tail += matcher.finish()
            if tail:
                yield tail
        result.append(GenerationResult(
            text=matcher.text, token_ids=tuple(out_ids), prompt_tokens=n_prompt,
            cached_tokens=state.get("cached_tokens", 0),
            finish_reason=state.get("finish_reason", "length"),
            prefill_seconds=state.get("prefill_seconds", 0.0),
            decode_seconds=state.get("decode_seconds", 0.0),
            spec_stats=state.get("spec_stats")))

    def apply_chat_template(self, messages: Sequence[Mapping[str, Any]],
                            add_generation_prompt: bool = True, enable_thinking: bool = False,
                            tools: Sequence[Mapping[str, Any]] | None = None,
                            use_jinja: bool | None = None) -> str:
        """Render a conversation with the tokenizer's chat template (see :mod:`pinfer.chat`)."""
        template = self.tokenizer.chat_template if self.tokenizer else None
        return apply_chat_template(messages, template, add_generation_prompt, enable_thinking,
                                   tools, use_jinja)

    def chat(self, messages: Sequence[Mapping[str, Any]], max_new_tokens: int = 512, *,
             enable_thinking: bool = False, tools: Sequence[Mapping[str, Any]] | None = None,
             **generate_kwargs: Any) -> GenerationResult | TextStream:
        """Answer the last user message of a conversation.

        ``messages`` is ``[{"role": "user", "content": "..."}, ...]``. Other keyword arguments
        go to :meth:`generate` (``stream``, ``temperature``, ``stop``...).
        """
        prompt = self.apply_chat_template(messages, True, enable_thinking, tools)
        return self.generate(prompt, max_new_tokens, **generate_kwargs)
