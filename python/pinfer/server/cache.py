"""What the server remembers between requests: state snapshots and its own replies.

* :class:`StateCache` — an LRU of ``Model.save_state()`` blobs keyed by the token prefix they
  hold, bounded in bytes. The engine context can only grow (the 18 linear-attention layers
  have a recurrent state that cannot be rolled back), so a snapshot is the only way back to an
  earlier prefix: a system prompt shared by several conversations, or a conversation that
  another one replaced in the live context.
* :class:`ReplyRegistry` — the token ids of the replies the server generated, so the next
  turn of a conversation can contain the reply exactly as it was generated (see
  :mod:`pinfer.server.history`).
* :class:`RequestQueue` — one generation at a time, in arrival order.
"""
from __future__ import annotations

import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Tokens = tuple[int, ...]


def is_prefix(key: Sequence[int], ids: Sequence[int]) -> bool:
    """True if ``key`` is a prefix of ``ids``."""
    n = len(key)
    return n <= len(ids) and tuple(ids[:n]) == tuple(key)


@dataclass
class Snapshot:
    """One saved context: ``key`` = the tokens it holds, ``kind`` = ``"prefix"`` (a
    checkpoint such as the system prompt) or ``"context"`` (a replaced live context)."""

    key: Tokens
    blob: bytes
    kind: str
    hits: int = 0


class StateCache:
    """LRU of state snapshots bounded by their total size in bytes (thread-safe)."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[Tokens, Snapshot] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        """Bytes held."""
        return self._bytes

    def contains(self, key: Sequence[int]) -> bool:
        """True if a snapshot of exactly ``key`` is cached."""
        with self._lock:
            return tuple(key) in self._items

    def lookup(self, ids: Sequence[int], max_len: int) -> Snapshot | None:
        """The snapshot with the longest key that is a prefix of ``ids`` and at most
        ``max_len`` tokens long; it becomes the most recently used."""
        ids = tuple(ids)
        with self._lock:
            best = None
            for snap in self._items.values():
                n = len(snap.key)
                if n <= max_len and (best is None or n > len(best.key)) and ids[:n] == snap.key:
                    best = snap
            if best is not None:
                best.hits += 1
                self._items.move_to_end(best.key)
            return best

    def put(self, key: Sequence[int], blob: bytes, kind: str) -> bool:
        """Store a snapshot (least recently used ones are evicted to make room). A
        ``"context"`` snapshot replaces the older context snapshots it extends (the same
        conversation, saved earlier). Returns False if the blob alone exceeds the budget."""
        k = tuple(key)
        if len(blob) > self.max_bytes:
            return False
        with self._lock:
            self._drop(k)
            if kind == "context":
                for old in [s.key for s in self._items.values()
                            if s.kind == "context" and is_prefix(s.key, k)]:
                    self._drop(old)
            while self._items and self._bytes + len(blob) > self.max_bytes:
                self._drop(next(iter(self._items)))
            self._items[k] = Snapshot(k, blob, kind)
            self._bytes += len(blob)
            return True

    def _drop(self, key: Tokens) -> None:
        snap = self._items.pop(key, None)
        if snap is not None:
            self._bytes -= len(snap.blob)

    def summary(self) -> dict[str, object]:
        """Counts for ``/health`` and logs."""
        with self._lock:
            return {"entries": len(self._items), "mb": round(self._bytes / 2**20, 1),
                    "max_mb": round(self.max_bytes / 2**20, 1),
                    "tokens": [len(k) for k in self._items]}


@dataclass(frozen=True)
class Reply:
    """A reply the server generated.

    Attributes:
        history: Canonical form of the request's messages (:func:`history.canonical`).
        prompt_ids: The prompt tokens it was generated from.
        reply_ids: The tokens that stand for the reply in the next turn: the generated ones
            (stop token excluded), or the re-tokenized text when a stop string cut it.
        content: The (stripped) ``content`` returned to the client.
    """

    history: tuple[str, ...]
    prompt_ids: Tokens
    reply_ids: Tokens
    content: str


class ReplyRegistry:
    """The most recent replies (thread-safe, bounded)."""

    def __init__(self, maxlen: int = 64) -> None:
        self._items: deque[Reply] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, reply: Reply) -> None:
        """Remember a reply (the oldest is forgotten when full)."""
        with self._lock:
            self._items.append(reply)

    def find(self, history: Sequence[str], contents: Sequence[str | None]) -> Reply | None:
        """The reply whose request is the longest prefix of ``history`` and is followed by
        an assistant message with the same content (``contents[i]``: the stripped content
        of message i if it is a plain assistant message, else None)."""
        with self._lock:
            best = None
            for r in self._items:
                k = len(r.history)
                if k < len(history) and contents[k] == r.content \
                        and (best is None or k > len(best.history)) \
                        and tuple(history[:k]) == r.history:
                    best = r
            return best


class RequestQueue:
    """FIFO admission for the single engine context: one generation at a time."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._waiting: deque[object] = deque()
        self._busy = False

    @property
    def busy(self) -> bool:
        """True while a generation runs."""
        return self._busy

    @property
    def waiting(self) -> int:
        """Requests queued behind the running one."""
        return len(self._waiting)

    def acquire(self, abandoned: Callable[[], bool], poll: float = 0.25) -> bool:
        """Wait for this request's turn. Returns False (without the turn) if ``abandoned()``
        becomes true while waiting, for example because the client disconnected."""
        ticket = object()
        with self._cond:
            self._waiting.append(ticket)
            try:
                while self._busy or self._waiting[0] is not ticket:
                    self._cond.wait(poll)
                    if abandoned():
                        return False
                self._busy = True
                return True
            finally:
                self._waiting.remove(ticket)
                self._cond.notify_all()

    def release(self) -> None:
        """End the running generation; the next queued request starts."""
        with self._cond:
            self._busy = False
            self._cond.notify_all()
