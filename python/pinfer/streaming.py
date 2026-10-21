"""Generation results, stop-string matching and the streaming iterator."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of one :meth:`pinfer.Model.generate` call.

    Attributes:
        text: Generated text (stop string and stop token excluded).
        token_ids: Generated token ids (stop token excluded; tokens past a stop string kept).
        prompt_tokens: Number of prompt tokens.
        cached_tokens: Prompt tokens reused from the previous call (not evaluated again).
        finish_reason: ``"stop"`` (stop token or stop string), ``"length"`` (max_new_tokens)
            or ``"context"`` (max_ctx reached).
        prefill_seconds: Time to evaluate the (uncached) prompt.
        decode_seconds: Time spent in single-token evaluations after the first token
            (speculative mode: the whole C generation call, prefill included).
        spec_stats: Speculative decoding counters (``rounds``, ``drafted``, ``accepted``,
            ``k``), or None for plain decoding.
    """

    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    cached_tokens: int
    finish_reason: str
    prefill_seconds: float
    decode_seconds: float
    spec_stats: Mapping[str, int] | None = None

    @property
    def prefill_tps(self) -> float:
        """Prompt tokens evaluated per second (uncached part only)."""
        n = self.prompt_tokens - self.cached_tokens
        return n / self.prefill_seconds if self.prefill_seconds > 0 else 0.0

    @property
    def decode_tps(self) -> float:
        """Generated tokens per second, excluding the first (it comes from the prefill)."""
        n = len(self.token_ids) - 1
        return n / self.decode_seconds if n > 0 and self.decode_seconds > 0 else 0.0


class StopMatcher:
    """Finds stop strings in a growing text and says how much text is safe to emit.

    Text that could be the start of a stop string is held back until it is decided.
    """

    def __init__(self, stops: Sequence[str]) -> None:
        self.stops = [s for s in stops if s]
        self.text = ""
        self.emitted = 0
        self.stopped = False

    def feed(self, delta: str) -> str:
        """Add generated text; return the part that can be shown now."""
        self.text += delta
        if not self.stops:
            return self._emit(len(self.text))
        hits = [i for i in (self.text.find(s) for s in self.stops) if i >= 0]
        if hits:
            self.stopped = True
            self.text = self.text[:min(hits)]
            return self._emit(len(self.text))
        hold = 0
        for s in self.stops:  # longest suffix of text that is a proper prefix of a stop
            for k in range(min(len(s) - 1, len(self.text)), hold, -1):
                if self.text.endswith(s[:k]):
                    hold = k
                    break
        return self._emit(len(self.text) - hold)

    def finish(self) -> str:
        """Release held-back text at the end of generation."""
        return self._emit(len(self.text))

    def _emit(self, end: int) -> str:
        out = self.text[self.emitted:end] if end > self.emitted else ""
        self.emitted = max(self.emitted, end)
        return out


class TextStream(Iterator[str]):
    """Iterator over text chunks of a streaming generation.

    After the iterator is exhausted, :attr:`result` holds the :class:`GenerationResult`.
    Closing it early (``close()`` or ``break`` + garbage collection) stops generation.
    """

    def __init__(self, chunks: Iterator[str], get_result: Callable[[], GenerationResult | None]
                 ) -> None:
        self._chunks = chunks
        self._get_result = get_result

    def __iter__(self) -> TextStream:
        return self

    def __next__(self) -> str:
        return next(self._chunks)

    def close(self) -> None:
        """Stop the generation."""
        close = getattr(self._chunks, "close", None)
        if close is not None:
            close()

    @property
    def result(self) -> GenerationResult | None:
        """Final result, available once the stream is exhausted."""
        return self._get_result()
