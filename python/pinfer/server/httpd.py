"""HTTP front end: ``http.server`` with one thread per connection (standard library only).

Routes: ``GET /health``, ``GET /v1/models[/{id}]``, ``POST /v1/chat/completions``,
``POST /v1/completions`` (``OPTIONS`` answers CORS preflights). Requests are parsed and
tokenized in their own thread, then wait in the engine's FIFO queue; a client that
disconnects while queued is dropped, one that disconnects while streaming stops the
generation at the next token. Streams are Server-Sent Events over chunked transfer encoding
(keep-alive works), ending with ``data: [DONE]``.
"""
from __future__ import annotations

import json
import logging
import select
import socket
import sys
import time
import uuid
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .. import __version__
from ..errors import ContextOverflowError
from . import protocol as P
from .engine import Engine, RunStats
from .history import PromptPlan

logger = logging.getLogger("pinfer.server")

MAX_BODY = 16 * 2**20
Events = Generator[tuple[str, str], None, None]


class ClientGone(Exception):
    """The client closed the connection."""


class Server(ThreadingHTTPServer):
    """The HTTP server; ``engine`` serves every request."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], engine: Engine,
                 api_key: str | None = None) -> None:
        super().__init__(address, Handler)
        self.engine = engine
        self.api_key = api_key

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A client that resets an idle keep-alive connection is not an error worth a
        traceback on stderr (socketserver's default); anything else is logged."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            logger.debug("connection from %s ended: %s", client_address, exc)
        else:
            logger.error("error while handling a request from %s", client_address,
                         exc_info=exc)


def _expected(e: BaseException) -> bool:
    """Errors caused by the request (400-type), as opposed to internal failures."""
    return isinstance(e, (P.APIError, ContextOverflowError))


def _error_for(e: BaseException) -> P.APIError:
    """Exception -> OpenAI error."""
    if isinstance(e, P.APIError):
        return e
    if isinstance(e, ContextOverflowError):
        return P.APIError(400, str(e), "invalid_request_error", "context_length_exceeded")
    return P.APIError(500, f"internal error: {type(e).__name__}: {e}", "server_error",
                      "internal_error")


def _internal(e: BaseException, engine: Engine) -> None:
    """Log an unexpected error during a generation and reset the context (the caller still
    holds the engine queue, so no other request is using the model)."""
    if not _expected(e):
        logger.error("internal error during generation", exc_info=e)
        engine.recover()


class Handler(BaseHTTPRequestHandler):
    """One connection (HTTP/1.1 keep-alive)."""

    protocol_version = "HTTP/1.1"
    server_version = f"pinfer/{__version__}"
    timeout = 300     # idle keep-alive connections and slow uploads
    server: Server

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s %s", self.address_string(), format % args)

    # ------------------------------------------------------------------ plumbing
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, status: int, obj: Any) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, err: P.APIError) -> None:
        self._send_json(err.status, err.body())

    def _client_gone(self) -> bool:
        """True if the peer closed the connection (non-blocking peek)."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return bool(self.connection.recv(1, socket.MSG_PEEK) == b"")
        except (OSError, ValueError):
            return True

    def _read_json(self) -> Any:
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            self.close_connection = True
            raise P.APIError(411, "a Content-Length header is required", code="length_required")
        n = int(length)
        if n > MAX_BODY:
            self.close_connection = True
            raise P.APIError(413, f"request body larger than {MAX_BODY} bytes",
                             code="request_too_large")
        raw = self.rfile.read(n)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise P.bad_request(f"invalid JSON body: {e}", code="invalid_json") from None

    def _authorized(self) -> bool:
        key = self.server.api_key
        return not key or self.headers.get("Authorization", "") == f"Bearer {key}"

    # ------------------------------------------------------------------ routes
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/")
        eng = self.server.engine
        if path == "/health":
            return self._send_json(200, eng.health())
        if not self._authorized():
            return self._send_error(P.APIError(401, "invalid API key", code="invalid_api_key"))
        card = eng.model_card()
        if path == "/v1/models":
            return self._send_json(200, {"object": "list", "data": [card]})
        if path == f"/v1/models/{card['id']}":
            return self._send_json(200, card)
        if path.startswith("/v1/models/"):
            return self._send_error(P.APIError(404, f"model {path[11:]!r} not found",
                                               code="model_not_found"))
        self._send_error(P.APIError(404, f"unknown path {path!r}", code="unknown_url"))

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/")
        eng = self.server.engine
        try:
            if path not in ("/v1/chat/completions", "/v1/completions"):
                self.close_connection = True
                raise P.APIError(404, f"unknown path {path!r}", code="unknown_url")
            body = self._read_json()
            if not self._authorized():
                raise P.APIError(401, "invalid API key", code="invalid_api_key")
            d = eng.cfg.defaults
            if path == "/v1/chat/completions":
                chat = P.parse_chat(body, d)
                plan = eng.plan_chat(chat)
                self._generate("chat", chat, plan, chat.enable_thinking)
            else:
                comp = P.parse_completion(body, d)
                plan = eng.plan_completion(comp)
                self._generate("completion", comp, plan, False)
        except Exception as e:  # every error becomes an OpenAI error reply
            if not _expected(e) and not isinstance(e, OSError):
                logger.error("internal error", exc_info=e)
            try:
                self._send_error(_error_for(e))
            except OSError:
                self.close_connection = True

    # ------------------------------------------------------------------ generation
    def _generate(self, kind: str, req: P.ChatRequest | P.CompletionRequest,
                  plan: PromptPlan, thinking: bool) -> None:
        eng = self.server.engine
        stats = RunStats(endpoint=kind, prompt_tokens=len(plan.ids))
        t0 = time.perf_counter()
        if not eng.queue.acquire(self._client_gone):
            stats.cancelled = True
            self.close_connection = True
            logger.info("%s dropped: the client left while queued (%.1fs)", kind,
                        time.perf_counter() - t0)
            return
        stats.queue_seconds = time.perf_counter() - t0
        err: P.APIError | None = None
        try:
            events = eng.run(plan, req.options, thinking, stats)
            if req.stream:
                self._stream(kind, req, events, stats)
            else:
                self._blocking(kind, req, events, stats)
        except ConnectionError:   # the client left while the reply was written
            self.close_connection = True
        except Exception as e:  # noqa: BLE001 - replied as an OpenAI error below
            stats.error = type(e).__name__
            _internal(e, eng)
            err = _error_for(e)
        finally:
            eng.recent.append(stats)
            eng.queue.release()
            logger.info("%s", stats.log_line())
        if err is not None:
            self._send_error(err)

    def _ids(self, kind: str) -> tuple[str, int]:
        return f"{'chatcmpl' if kind == 'chat' else 'cmpl'}-{uuid.uuid4().hex[:24]}", \
            int(time.time())

    def _echo(self, req: P.ChatRequest | P.CompletionRequest) -> str:
        if not isinstance(req, P.CompletionRequest) or not req.echo:
            return ""
        p = req.prompt
        return p if isinstance(p, str) else self.server.engine.model.detokenize(p)

    def _blocking(self, kind: str, req: P.ChatRequest | P.CompletionRequest,
                  events: Events, stats: RunStats) -> None:
        eng = self.server.engine
        parts: dict[str, list[str]] = {"content": [], "reasoning": []}
        checked = time.perf_counter()
        try:
            for k, text in events:
                parts[k].append(text)
                if time.perf_counter() - checked > 0.5:   # stop if nobody waits for it
                    checked = time.perf_counter()
                    if self._client_gone():
                        self.close_connection = True
                        return
        finally:
            events.close()
        rid, created = self._ids(kind)
        content = "".join(parts["content"])
        fr = P.finish_reason(stats.finish_reason)
        extra = {"usage": P.usage(stats.prompt_tokens, stats.completion_tokens,
                                  stats.cached_tokens),
                 "system_fingerprint": f"pinfer-{__version__}", "pinfer": stats.extension()}
        if isinstance(req, P.ChatRequest):
            reasoning = "".join(parts["reasoning"]).strip() if req.enable_thinking else None
            choice: dict[str, Any] = {"index": 0, "message": P.chat_message(content, reasoning),
                                      "logprobs": None, "finish_reason": fr}
            body = P.envelope("chat.completion", rid, created, eng.cfg.model_name, [choice],
                              **extra)
        else:
            choice = {"index": 0, "text": self._echo(req) + content, "logprobs": None,
                      "finish_reason": fr}
            body = P.envelope("text_completion", rid, created, eng.cfg.model_name, [choice],
                              **extra)
        self._send_json(200, body)

    def _write_chunk(self, data: bytes) -> None:
        if self._client_gone():
            raise ClientGone
        try:
            self.wfile.write(b"%X\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        except OSError as e:   # BrokenPipe, ConnectionReset, ConnectionAborted
            raise ClientGone from e

    def _sse(self, obj: Any) -> None:
        payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        self._write_chunk(f"data: {payload}\n\n".encode())

    def _stream(self, kind: str, req: P.ChatRequest | P.CompletionRequest,
                events: Events, stats: RunStats) -> None:
        eng = self.server.engine
        rid, created = self._ids(kind)
        name = eng.cfg.model_name
        obj = "chat.completion.chunk" if kind == "chat" else "text_completion"
        extra: dict[str, Any] = {"usage": None} if req.include_usage else {}

        def chunk(delta: dict[str, Any] | str, fr: str | None = None, **more: Any) -> Any:
            c: dict[str, Any] = {"index": 0, "logprobs": None, "finish_reason": fr}
            if isinstance(delta, str):
                c["text"] = delta
            else:
                c["delta"] = delta
            return P.envelope(obj, rid, created, name, [c], **extra, **more)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        try:
            try:
                if kind == "chat":
                    self._sse(chunk({"role": "assistant", "content": ""}))
                elif self._echo(req):
                    self._sse(chunk(self._echo(req)))
                for k, text in events:
                    if kind != "chat":
                        self._sse(chunk(text))
                    else:
                        self._sse(chunk({"content" if k == "content" else "reasoning_content":
                                         text}))
                fr = P.finish_reason(stats.finish_reason)
                self._sse(chunk({} if kind == "chat" else "", fr, pinfer=stats.extension()))
                if req.include_usage:
                    self._sse(P.envelope(obj, rid, created, name, [], usage=P.usage(
                        stats.prompt_tokens, stats.completion_tokens, stats.cached_tokens),
                        pinfer=stats.extension()))
            except ClientGone:
                raise
            except Exception as e:  # noqa: BLE001 - reported in the stream (headers are out)
                stats.error = type(e).__name__
                _internal(e, eng)
                self._sse(_error_for(e).body())
            finally:
                events.close()
            self._sse("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (ClientGone, OSError):
            self.close_connection = True
            logger.debug("client disconnected during the stream")


def make_server(engine: Engine, host: str = "127.0.0.1", port: int = 8080,
                api_key: str | None = None) -> Server:
    """Create (bind) the HTTP server; call ``serve_forever()`` on it."""
    return Server((host, port), engine, api_key)
