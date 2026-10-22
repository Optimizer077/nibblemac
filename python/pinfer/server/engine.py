"""The server's single model context: request planning, prefix reuse, generation.

One :class:`Engine` owns one :class:`pinfer.Model` (not thread-safe) and a
:class:`~pinfer.server.cache.RequestQueue`; the HTTP handlers call :meth:`Engine.run` only
while they hold the queue. Before each generation :meth:`Engine._prepare` picks the longest
reusable prefix of the prompt:

1. the live context, if the prompt extends it (the usual next turn of a conversation);
2. otherwise the longest cached state snapshot that is a prefix of the prompt; the live
   context is saved first, so a conversation that is replaced can come back later;
3. otherwise a fresh context.

A system message of at least ``snapshot_min_tokens`` tokens is checkpointed on the way, so
new conversations with the same system prompt start from it.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..speculative import iter_speculative
from ..streaming import GenerationResult, TextStream
from .cache import Reply, ReplyRegistry, RequestQueue, StateCache, is_prefix
from .history import PromptPlan, chat_prompt
from .protocol import APIError, ChatRequest, CompletionRequest, Defaults, Options, bad_request

if TYPE_CHECKING:
    from ..model import Model

logger = logging.getLogger("pinfer.server")

THINK_END = "</think>"


@dataclass(frozen=True)
class EngineConfig:
    """Server behaviour (``pinfer serve`` options).

    Attributes:
        model_name: Model id reported by ``/v1/models`` and in responses.
        speculative: Greedy requests use MTP speculative decoding (same output, faster).
        spec_k: Draft tokens per pass (None: the library default, 2).
        reuse: Reuse token prefixes across requests (live context + snapshots).
        history: ``"generated"`` (earlier replies as generated) or ``"template"``.
        state_cache_mb: Memory budget of the snapshot LRU (0 disables snapshots).
        snapshot_min_tokens: Shortest prefix worth a snapshot (a snapshot is ~20 MB + 24 KB
            per token with the fp32 KV cache).
        defaults: Defaults for options a request leaves out.
    """

    model_name: str
    speculative: bool = True
    spec_k: int | None = None
    reuse: bool = True
    history: str = "generated"
    state_cache_mb: float = 256.0
    snapshot_min_tokens: int = 64
    defaults: Defaults = field(default_factory=Defaults)


@dataclass
class RunStats:
    """Accounting of one request (the ``pinfer`` extension object and the log line)."""

    endpoint: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cache_source: str | None = None   # "context" | "snapshot" | None
    spliced_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "length"
    queue_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    speculative: dict[str, Any] | None = None
    cancelled: bool = False
    error: str | None = None

    @property
    def tokens_per_second(self) -> float:
        """Decode speed: tokens after the first / decode time."""
        n = self.completion_tokens - 1
        return n / self.decode_seconds if n > 0 and self.decode_seconds > 0 else 0.0

    def extension(self) -> dict[str, Any]:
        """The ``pinfer`` object added to responses."""
        return {"cached_tokens": self.cached_tokens, "cache_source": self.cache_source,
                "spliced_tokens": self.spliced_tokens,
                "prefill_seconds": round(self.prefill_seconds, 4),
                "decode_seconds": round(self.decode_seconds, 4),
                "tokens_per_second": round(self.tokens_per_second, 3),
                "queue_seconds": round(self.queue_seconds, 4), "speculative": self.speculative}

    def log_line(self) -> str:
        """One line per request for the server log."""
        spec = ""
        if self.speculative:
            spec = f" spec k={self.speculative['k']} {self.speculative['tokens_per_pass']} tok/pass"
        end = "cancelled" if self.cancelled else self.error or self.finish_reason
        return (f"{self.endpoint} prompt={self.prompt_tokens} cached={self.cached_tokens}"
                f" ({self.cache_source or 'none'}) completion={self.completion_tokens}"
                f" end={end} queue={self.queue_seconds:.2f}s prefill={self.prefill_seconds:.2f}s"
                f" decode={self.decode_seconds:.2f}s {self.tokens_per_second:.2f} tok/s{spec}")


class ThinkSplitter:
    """Routes generated text to ``reasoning`` until ``</think>``, then to ``content``
    (thinking mode); text that could start ``</think>`` is held back until decided."""

    def __init__(self, thinking: bool) -> None:
        self.thinking = thinking
        self._in_reasoning = thinking
        self._buf = ""
        self._lead = thinking       # strip the newlines between </think> and the answer
        self.reasoning = ""
        self.content = ""

    def feed(self, delta: str) -> Iterator[tuple[str, str]]:
        """Yield ``("reasoning" | "content", text)`` events for a text delta."""
        if self._in_reasoning:
            self._buf += delta
            i = self._buf.find(THINK_END)
            if i < 0:
                hold = next((k for k in range(min(len(THINK_END) - 1, len(self._buf)), 0, -1)
                             if self._buf.endswith(THINK_END[:k])), 0)
                cut = len(self._buf) - hold
                out, self._buf = self._buf[:cut], self._buf[cut:]
                yield from self._emit("reasoning", out)
                return
            out, delta = self._buf[:i], self._buf[i + len(THINK_END):]
            self._buf, self._in_reasoning = "", False
            yield from self._emit("reasoning", out)
        if self._lead:
            delta = delta.lstrip()
            self._lead = not delta
        yield from self._emit("content", delta)

    def finish(self) -> Iterator[tuple[str, str]]:
        """Release held-back text at the end of generation."""
        out, self._buf = self._buf, ""
        yield from self._emit("reasoning" if self._in_reasoning else "content", out)

    def _emit(self, kind: str, text: str) -> Iterator[tuple[str, str]]:
        if not text:
            return
        if kind == "reasoning":
            self.reasoning += text
        else:
            self.content += text
        yield kind, text


class Engine:
    """One model context shared by all requests (see module docstring)."""

    def __init__(self, model: Model, cfg: EngineConfig) -> None:
        if model.tokenizer is None:
            raise ValueError("the server needs a model loaded with a tokenizer")
        if cfg.history not in ("generated", "template"):
            raise ValueError(f"history must be 'generated' or 'template', got {cfg.history!r}")
        self.model = model
        self.cfg = cfg
        self.queue = RequestQueue()
        self.cache = StateCache(int(cfg.state_cache_mb * 2**20))
        self.replies = ReplyRegistry()
        self.recent: deque[RunStats] = deque(maxlen=64)
        self.created = int(time.time())
        lib = model._lib
        self.speculative = cfg.speculative and model.supports_speculative
        self._spec_prefill = (self.speculative and lib.pinfer_generate_greedy_cont is not None
                              and lib.pinfer_generate_greedy_ex is not None)
        self.snapshots = cfg.reuse and model.supports_state and self.cache.max_bytes > 0
        self._live_saved = False    # the live context is already in the snapshot cache

    # ------------------------------------------------------------------ planning
    def _check_fits(self, n: int, param: str) -> None:
        if n >= self.model.max_ctx:
            raise APIError(400, f"This model's maximum context length is "
                           f"{self.model.max_ctx} tokens, but the prompt has {n} tokens and "
                           "the reply needs at least one more. Shorten the prompt, or start "
                           "the server with a larger --max-ctx.", "invalid_request_error",
                           "context_length_exceeded", param)

    def plan_chat(self, req: ChatRequest) -> PromptPlan:
        """Prompt tokens of a chat request (thread-safe; call before queueing)."""
        use_history = self.cfg.reuse and self.cfg.history == "generated"
        plan = chat_prompt(self.model, self.replies if use_history else None,
                           req.messages, req.enable_thinking)
        self._check_fits(len(plan.ids), "messages")
        return plan

    def plan_completion(self, req: CompletionRequest) -> PromptPlan:
        """Prompt tokens of a completion request (thread-safe; call before queueing)."""
        if isinstance(req.prompt, str):
            ids = self.model.tokenize(req.prompt)
        else:
            ids = list(req.prompt)
            if any(t < 0 or t >= self.model.n_vocab for t in ids):
                raise bad_request(f"token ids must be in [0, {self.model.n_vocab})", "prompt")
        if not ids:
            raise bad_request("the prompt has no tokens", "prompt")
        self._check_fits(len(ids), "prompt")
        return PromptPlan(ids)

    # ------------------------------------------------------------------ context reuse
    def _save_live(self) -> None:
        """Snapshot the live context before it is replaced (if worth keeping)."""
        m = self.model
        ctx = m.context_tokens
        if not self.snapshots or self._live_saved or len(ctx) < self.cfg.snapshot_min_tokens:
            return
        if self.cache.put(ctx, m.save_state(), "context"):
            logger.debug("saved the live context (%d tokens) as a snapshot", len(ctx))
        self._live_saved = True

    def _prefill_to(self, prefix: list[int]) -> None:
        """Extend the context (a prefix of ``prefix``) to exactly ``prefix``."""
        m = self.model
        if self._spec_prefill:   # through the MTP path, so the MTP rows are filled too
            state: dict[str, Any] = {}
            for _ in iter_speculative(m, prefix, 1, set(), state, k=self.cfg.spec_k):
                pass             # the one generated token is emitted, not fed
        else:
            m.eval(prefix[len(m.context_tokens):])

    def _prepare(self, plan: PromptPlan) -> tuple[int, str | None]:
        """Bring the context to the longest reusable prefix of the prompt (see module
        docstring); returns (prompt tokens reused, where from)."""
        m = self.model
        if not self.cfg.reuse:
            return 0, None
        ids = plan.ids
        ctx = m.context_tokens
        live = len(ctx) if 0 < len(ctx) < len(ids) and is_prefix(ctx, ids) else 0
        snap = self.cache.lookup(ids, len(ids) - 1) if self.snapshots else None
        if snap is not None and len(snap.key) > live:
            self._save_live()
            m.load_state(snap.blob)
            self._live_saved = True
            reused, source = len(snap.key), "snapshot"
        elif live:
            reused, source = live, "context"
        else:
            self._save_live()
            m.reset()
            reused, source = 0, None
        n_sys = plan.prefix_len
        if (self.snapshots and reused < n_sys < len(ids) and n_sys >= self.cfg.snapshot_min_tokens
                and not self.cache.contains(ids[:n_sys])):
            self._prefill_to(ids[:n_sys])
            self._live_saved = self.cache.put(ids[:n_sys], m.save_state(), "prefix")
            logger.debug("checkpointed the system prompt (%d tokens)", n_sys)
        return reused, source

    def recover(self) -> None:
        """After an unexpected error: start from an empty context next time."""
        try:
            self.model.reset()
        finally:
            self._live_saved = False

    # ------------------------------------------------------------------ generation
    def run(self, plan: PromptPlan, opt: Options, thinking: bool,
            stats: RunStats) -> Generator[tuple[str, str], None, None]:
        """Generate (the caller holds :attr:`queue`); yields ``(kind, text)`` events with
        kind ``"content"`` or ``"reasoning"``. Closing the generator stops the generation
        promptly (speculative decoding included); ``stats`` is complete afterwards."""
        m = self.model
        t0 = time.perf_counter()
        spec = self.speculative and opt.greedy
        stats.prompt_tokens, stats.spliced_tokens = len(plan.ids), plan.spliced
        stats.cached_tokens, stats.cache_source = self._prepare(plan)
        prep = time.perf_counter() - t0
        stream = m.generate(plan.ids, opt.max_tokens, temperature=opt.temperature,
                            top_p=opt.top_p, top_k=opt.top_k, seed=opt.seed,
                            stop=list(opt.stop) or None, stream=True,
                            reuse_prefix=self.cfg.reuse, speculative=spec,
                            spec_k=self.cfg.spec_k if spec else None)
        assert isinstance(stream, TextStream)
        split = ThinkSplitter(thinking)
        done = False
        t_first = 0.0
        try:
            for chunk in stream:
                t_first = t_first or time.perf_counter()
                yield from split.feed(chunk)
            yield from split.finish()
            done = True
        finally:
            stream.close()
            self._live_saved = False
            t_end = time.perf_counter()
            self._account(plan, stream.result, stats, prep, t0, t_first or t_end, t_end)
            stats.cancelled = not done
            if done and stream.result is not None and plan.history is not None:
                self._remember(plan, stream.result, split)

    def _account(self, plan: PromptPlan, r: GenerationResult | None, stats: RunStats,
                 prep: float, t0: float, t_first: float, t_end: float) -> None:
        if r is None:   # closed early: count what reached the context, wall-clock timing
            ctx = self.model.context_tokens
            fed = len(ctx) - len(plan.ids) + 1 if is_prefix(plan.ids, ctx) else 0
            stats.completion_tokens = max(0, fed)
            stats.prefill_seconds, stats.decode_seconds = t_first - t0, t_end - t_first
            return
        stats.completion_tokens = len(r.token_ids)
        stats.finish_reason = r.finish_reason
        stats.prefill_seconds = prep + r.prefill_seconds
        stats.decode_seconds = r.decode_seconds
        if r.spec_stats:
            st: dict[str, Any] = dict(r.spec_stats)
            st["tokens_per_pass"] = round(1 + st["accepted"] / st["rounds"], 3) \
                if st["rounds"] else 0.0
            stats.speculative = st

    def _remember(self, plan: PromptPlan, r: GenerationResult, split: ThinkSplitter) -> None:
        """Record a finished chat reply for the next turn's prompt (history.py)."""
        if not (self.cfg.reuse and self.cfg.history == "generated" and plan.history):
            return
        tok = self.model.tokenizer
        assert tok is not None
        exact = r.text == tok.decode(r.token_ids, skip_special_tokens=True)
        reply_ids = r.token_ids if exact else tuple(tok.encode(r.text))
        self.replies.add(Reply(plan.history, tuple(plan.ids), tuple(reply_ids),
                               split.content.strip()))

    # ------------------------------------------------------------------ info
    def health(self) -> dict[str, Any]:
        """Body of ``GET /health``."""
        return {"status": "ok", "model": self.cfg.model_name, "busy": self.queue.busy,
                "queued": self.queue.waiting, "context_tokens": len(self.model.context_tokens),
                "max_ctx": self.model.max_ctx, "speculative": self.speculative,
                "state_cache": self.cache.summary()}

    def model_card(self) -> dict[str, Any]:
        """One entry of ``GET /v1/models``."""
        m = self.model
        return {"id": self.cfg.model_name, "object": "model", "created": self.created,
                "owned_by": "pinfer", "max_model_len": m.max_ctx,
                "pinfer": {"version": m.version, "kv_dtype": m.kv_dtype,
                           "speculative": self.speculative, "spec_k": self.cfg.spec_k,
                           "reuse": self.cfg.reuse, "history": self.cfg.history}}
