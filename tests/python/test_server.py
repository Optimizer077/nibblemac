"""pinfer serve: the OpenAI-compatible HTTP server (pinfer.server).

The first tests need no model (request parsing, caches, the queue). The others start the
server on a free port in a thread, against the library under test; generations are kept to
a few tokens because the host build uses the slow scalar reference kernels.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pinfer.server import cache, protocol
from pinfer.server.engine import ThinkSplitter

SYSTEM = {"role": "system", "content": "You are a helpful assistant. Answer in one word."}


# ------------------------------------------------------------------ no model needed
def test_parse_chat_options() -> None:
    d = protocol.Defaults()
    r = protocol.parse_chat({"messages": [{"role": "developer", "content": "be brief"},
                                          {"role": "user", "content": [
                                              {"type": "text", "text": "Hi "},
                                              {"type": "text", "text": "there"}]}],
                             "max_completion_tokens": 5, "max_tokens": 9, "temperature": 0.5,
                             "top_k": 20, "seed": 3, "stop": "\n", "stream": True,
                             "stream_options": {"include_usage": True},
                             "chat_template_kwargs": {"enable_thinking": True},
                             "presence_penalty": 0.5}, d)
    assert r.messages == ({"role": "system", "content": "be brief"},
                          {"role": "user", "content": "Hi there"})
    o = r.options
    assert (o.max_tokens, o.temperature, o.top_k, o.seed, o.stop) == (5, 0.5, 20, 3, ("\n",))
    assert r.stream and r.include_usage and r.enable_thinking and not o.greedy
    plain = protocol.parse_chat({"messages": [{"role": "user", "content": "x"}]}, d)
    assert plain.options.max_tokens == d.max_tokens and plain.options.greedy
    assert not plain.stream and not plain.enable_thinking


@pytest.mark.parametrize("body,param", [
    ({}, "messages"),
    ({"messages": "hi"}, "messages"),
    ({"messages": [{"role": "robot", "content": "x"}]}, "messages[0].role"),
    ({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, "messages[0]"),
    ({"messages": [{"role": "user", "content": "x"}], "n": 2}, "n"),
    ({"messages": [{"role": "user", "content": "x"}], "temperature": -1}, "temperature"),
    ({"messages": [{"role": "user", "content": "x"}], "top_p": 0}, "top_p"),
    ({"messages": [{"role": "user", "content": "x"}], "max_tokens": "8"}, "max_tokens"),
    ({"messages": [{"role": "user", "content": "x"}], "stop": [1]}, "stop"),
    ({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]},
     "tools"),
    ({"messages": [{"role": "user", "content": "x"}], "logprobs": True}, "logprobs"),
])
def test_parse_chat_rejects(body: dict[str, Any], param: str) -> None:
    with pytest.raises(protocol.APIError) as ei:
        protocol.parse_chat(body, protocol.Defaults())
    assert ei.value.status == 400 and ei.value.param == param
    assert ei.value.body()["error"]["type"] == "invalid_request_error"


def test_parse_completion_prompt_forms() -> None:
    d = protocol.Defaults()
    assert protocol.parse_completion({"prompt": "a"}, d).prompt == "a"
    assert protocol.parse_completion({"prompt": ["a"]}, d).prompt == "a"
    r = protocol.parse_completion({"prompt": [1, 2, 3], "echo": True}, d)
    assert r.prompt == (1, 2, 3) and r.echo and r.options.max_tokens == 16
    for bad in ({"prompt": ["a", "b"]}, {"prompt": ""}, {"prompt": 5}, {"prompt": "a",
                                                                        "suffix": "z"}):
        with pytest.raises(protocol.APIError):
            protocol.parse_completion(bad, d)


def test_think_splitter() -> None:
    s = ThinkSplitter(thinking=True)
    events = [e for d in ("Let me", " think.\n</th", "ink>\n\nThe", " answer.") for e in s.feed(d)]
    events += list(s.finish())
    assert events == [("reasoning", "Let me"), ("reasoning", " think.\n"),
                      ("content", "The"), ("content", " answer.")]
    assert s.reasoning == "Let me think.\n" and s.content == "The answer."
    s = ThinkSplitter(thinking=False)
    assert list(s.feed("a </think> b")) == [("content", "a </think> b")]
    s = ThinkSplitter(thinking=True)
    assert list(s.feed("cut off </thi")) == [("reasoning", "cut off ")]
    assert list(s.finish()) == [("reasoning", "</thi")]


def test_state_cache_lru_and_prefixes() -> None:
    c = cache.StateCache(max_bytes=100)
    assert c.put((1, 2), b"x" * 40, "prefix")
    assert c.put((1, 2, 3, 4), b"y" * 40, "context")
    def key(ids: list[int], max_len: int) -> tuple[int, ...] | None:
        snap = c.lookup(ids, max_len)
        return None if snap is None else snap.key

    assert key([1, 2, 3, 4, 5], 9) == (1, 2, 3, 4)                # longest prefix wins
    assert key([1, 2, 3, 4], 3) == (1, 2)                         # bounded by max_len
    assert key([2, 1], 9) is None
    assert c.put((1, 2, 3, 4, 5, 6), b"z" * 40, "context")        # same conversation, later:
    assert not c.contains((1, 2, 3, 4)) and c.contains((1, 2))    # replaces its old snapshot
    assert c.put((7,), b"w" * 40, "context")                      # over budget: LRU evicted
    assert len(c) == 2 and c.nbytes == 80 and not c.contains((1, 2))
    assert not c.put((8,), b"v" * 101, "prefix")                  # larger than the budget


def test_reply_registry_finds_longest_prefix() -> None:
    reg = cache.ReplyRegistry(maxlen=4)
    reg.add(cache.Reply(("u1",), (1,), (2,), "a1"))
    reg.add(cache.Reply(("u1", "a1", "u2"), (1, 2, 3), (4,), "a2"))
    hist = ("u1", "a1", "u2", "a2", "u3")
    contents = [None, "a1", None, "a2", None]
    hit = reg.find(hist, contents)
    assert hit is not None and hit.content == "a2"
    hit = reg.find(hist, [None, "a1", None, "other", None])
    assert hit is not None and hit.content == "a1"
    assert reg.find(("u9", "a1"), [None, "a1"]) is None


def test_request_queue_is_fifo() -> None:
    q = cache.RequestQueue()
    assert q.acquire(lambda: False)
    order: list[int] = []

    def worker(i: int) -> None:
        assert q.acquire(lambda: False)
        order.append(i)
        q.release()

    threads = []
    for i in range(4):
        threads.append(threading.Thread(target=worker, args=(i,)))
        threads[-1].start()
        while q.waiting < i + 1:
            time.sleep(0.005)
    gone = threading.Event()
    result: list[bool] = []
    leaver = threading.Thread(target=lambda: result.append(q.acquire(gone.is_set, 0.01)))
    leaver.start()
    gone.set()
    leaver.join(5)
    assert result == [False]          # left while queued: not admitted
    q.release()
    for t in threads:
        t.join(5)
    assert order == [0, 1, 2, 3] and not q.busy and q.waiting == 0


# ------------------------------------------------------------------ live server
class Client:
    """Minimal HTTP client (http.client handles the chunked transfer encoding)."""

    def __init__(self, port: int) -> None:
        self.port = port

    def request(self, method: str, path: str, body: Any = None, raw: bytes | None = None,
                headers: dict[str, str] | None = None, timeout: float = 300
                ) -> tuple[int, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        data = raw if raw is not None else None if body is None else json.dumps(body).encode()
        conn.request(method, path, body=data,
                     headers={"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        text = resp.read().decode()
        conn.close()
        try:
            return resp.status, json.loads(text)
        except json.JSONDecodeError:
            return resp.status, text

    def post(self, path: str, body: Any, **kw: Any) -> tuple[int, Any]:
        return self.request("POST", path, body, **kw)

    def ok(self, path: str, body: Any) -> dict[str, Any]:
        status, data = self.post(path, body)
        assert status == 200, data
        assert isinstance(data, dict)
        return data

    def stream(self, path: str, body: dict[str, Any]) -> list[Any]:
        status, text = self.post(path, {**body, "stream": True})
        assert status == 200 and isinstance(text, str), text
        events = [e[len("data: "):] for e in text.split("\n\n") if e]
        assert events[-1] == "[DONE]"
        return [json.loads(e) for e in events[:-1]]


@pytest.fixture(scope="module")
def server(lib: Any, model_path: Path, tokenizer: Any) -> Iterator[Any]:
    """A running server (snapshots from 8 tokens on, so the short test prompts use them)."""
    import pinfer
    from pinfer.server import Engine, EngineConfig, make_server

    m = pinfer.Model.load(model_path, tokenizer=tokenizer, n_threads=4, max_ctx=256)
    eng = Engine(m, EngineConfig(model_name="test-model", snapshot_min_tokens=8,
                                 state_cache_mb=512))
    srv = make_server(eng, "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                         daemon=True)
    t.start()
    srv.client = Client(srv.server_address[1])   # type: ignore[attr-defined]
    yield srv
    srv.shutdown()
    srv.server_close()
    t.join(10)
    m.close()


@pytest.fixture()
def client(server: Any) -> Client:
    c: Client = server.client
    return c


def test_models_and_health(client: Client) -> None:
    status, data = client.request("GET", "/v1/models")
    assert status == 200 and data["object"] == "list"
    assert [d["id"] for d in data["data"]] == ["test-model"]
    assert data["data"][0]["max_model_len"] == 256
    assert client.request("GET", "/v1/models/test-model")[1]["id"] == "test-model"
    assert client.request("GET", "/v1/models/other")[0] == 404
    status, health = client.request("GET", "/health")
    assert status == 200 and health["status"] == "ok" and health["busy"] is False


def test_completion_matches_golden(client: Client, tokenizer: Any,
                                   golden: dict[str, Any]) -> None:
    want = tokenizer.decode(golden["greedy_ids"][:4])
    r = client.ok("/v1/completions", {"model": "x", "prompt": golden["prompt"],
                                      "max_tokens": 4, "temperature": 0})
    assert r["object"] == "text_completion" and r["choices"][0]["text"] == want
    assert r["choices"][0]["finish_reason"] == "length"
    u = r["usage"]
    assert (u["prompt_tokens"], u["completion_tokens"]) == (len(golden["input_ids"]), 4)
    assert r["pinfer"]["tokens_per_second"] > 0 and r["pinfer"]["prefill_seconds"] > 0
    chunks = client.stream("/v1/completions", {"prompt": golden["input_ids"], "max_tokens": 4,
                                               "echo": True})
    assert all(c["object"] == "text_completion" for c in chunks)
    assert "".join(c["choices"][0]["text"] for c in chunks) == golden["prompt"] + want
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_stop_string(client: Client, golden: dict[str, Any]) -> None:
    r = client.ok("/v1/completions", {"prompt": golden["prompt"], "max_tokens": 10,
                                      "stop": ["range", "zzz"]})
    assert r["choices"][0]["text"] == golden["greedy_text"].split("range")[0]
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"]["completion_tokens"] <= 8


def test_chat_blocking_and_streaming_agree(client: Client) -> None:
    msgs = [{"role": "user", "content": "Name a primary color."}]
    r = client.ok("/v1/chat/completions", {"model": "test-model", "messages": msgs,
                                           "max_tokens": 3})
    assert r["object"] == "chat.completion" and r["id"].startswith("chatcmpl-")
    msg = r["choices"][0]["message"]
    assert msg["role"] == "assistant" and msg["content"]
    assert "<|im_end|>" not in msg["content"] and "<think>" not in msg["content"]
    assert r["usage"]["total_tokens"] == r["usage"]["prompt_tokens"] + \
        r["usage"]["completion_tokens"]
    assert r["pinfer"]["speculative"] is None or r["pinfer"]["speculative"]["k"] == 3  # library default
    chunks = client.stream("/v1/chat/completions", {"messages": msgs, "max_tokens": 3,
                                                    "stream_options": {"include_usage": True}})
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == msg["content"]
    finish = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert len(finish) == 1 and finish[0]["choices"][0]["delta"] == {}
    usage = chunks[-1]["usage"]
    assert chunks[-1]["choices"] == [] and usage["total_tokens"] == r["usage"]["total_tokens"]
    assert all(c["usage"] is None for c in chunks[:-1])


@pytest.mark.parametrize("raw,status,code", [
    (b"{not json", 400, "invalid_json"),
    (b"[1, 2]", 400, None),
    (b'{"messages": [{"role": "user", "content": "x"}], "n": 3}', 400, None),
])
def test_bad_requests(client: Client, raw: bytes, status: int, code: str | None) -> None:
    got, body = client.request("POST", "/v1/chat/completions", raw=raw)
    assert got == status and set(body["error"]) == {"message", "type", "param", "code"}
    assert body["error"]["code"] == code


def test_unknown_path_and_context_overflow(client: Client) -> None:
    status, body = client.post("/v1/embeddings", {"input": "x"})
    assert status == 404 and body["error"]["code"] == "unknown_url"
    status, body = client.post("/v1/completions", {"prompt": [100] * 300, "max_tokens": 1})
    assert status == 400 and body["error"]["code"] == "context_length_exceeded"
    assert "maximum context length is 256" in body["error"]["message"]
    long = [{"role": "user", "content": "word " * 300}]
    status, body = client.post("/v1/chat/completions", {"messages": long, "max_tokens": 1})
    assert status == 400 and body["error"]["code"] == "context_length_exceeded"
    status, body = client.post("/v1/completions", {"prompt": [10**7], "max_tokens": 1})
    assert status == 400 and "token ids" in body["error"]["message"]


def test_second_turn_reuses_prefix(client: Client, server: Any) -> None:
    """Turn 2 resends turn 1: the whole first exchange comes from the live context, although
    the template renders the earlier reply without its (empty) think block."""
    eng = server.engine
    turn1 = [SYSTEM, {"role": "user", "content": "Is the sky blue?"}]
    r1 = client.ok("/v1/chat/completions", {"messages": turn1, "max_tokens": 2})
    p1, c1 = r1["usage"]["prompt_tokens"], r1["usage"]["completion_tokens"]
    ctx = eng.model.context_tokens
    turn2 = turn1 + [r1["choices"][0]["message"], {"role": "user", "content": "Is grass?"}]
    r2 = client.ok("/v1/chat/completions", {"messages": turn2, "max_tokens": 2})
    ext = r2["pinfer"]
    assert ext["cache_source"] == "context" and ext["cached_tokens"] == len(ctx)
    assert ext["cached_tokens"] >= p1 + c1 - 1 > 0
    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] == ext["cached_tokens"]
    # the plain template rendering would have shared only the text before the first reply
    plain = eng.model.tokenize(eng.model.apply_chat_template(turn2))
    shared = next((i for i, (a, b) in enumerate(zip(plain, ctx)) if a != b), len(ctx))
    assert shared < p1 <= ext["cached_tokens"]


def test_alternating_conversations_use_snapshots(client: Client) -> None:
    """Two conversations take turns: each comes back from its own snapshot, and the second
    one's first turn starts from the shared system prompt's checkpoint."""
    a1 = [SYSTEM, {"role": "user", "content": "Say yes."}]
    b1 = [SYSTEM, {"role": "user", "content": "Say no."}]
    ra = client.ok("/v1/chat/completions", {"messages": a1, "max_tokens": 2})
    rb = client.ok("/v1/chat/completions", {"messages": b1, "max_tokens": 2})
    assert rb["pinfer"]["cache_source"] == "snapshot"     # the system prompt checkpoint
    assert 8 <= rb["pinfer"]["cached_tokens"] < rb["usage"]["prompt_tokens"]
    a2 = a1 + [ra["choices"][0]["message"], {"role": "user", "content": "Again."}]
    r = client.ok("/v1/chat/completions", {"messages": a2, "max_tokens": 2})
    assert r["pinfer"]["cache_source"] == "snapshot"
    assert r["pinfer"]["cached_tokens"] >= ra["usage"]["prompt_tokens"] + \
        ra["usage"]["completion_tokens"] - 1
    b2 = b1 + [rb["choices"][0]["message"], {"role": "user", "content": "Again."}]
    r = client.ok("/v1/chat/completions", {"messages": b2, "max_tokens": 2})
    assert r["pinfer"]["cache_source"] == "snapshot"
    assert r["pinfer"]["cached_tokens"] >= rb["usage"]["prompt_tokens"] + \
        rb["usage"]["completion_tokens"] - 1


def test_concurrent_requests_are_serialized(client: Client, server: Any,
                                            golden: dict[str, Any],
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from pinfer.streaming import TextStream

    model = server.engine.model
    real = model.generate
    active, peak = [0], [0]
    lock = threading.Lock()

    def counting(*args: Any, **kwargs: Any) -> TextStream:
        """generate(), counting the streams being consumed at the same time."""
        stream = real(*args, **kwargs)

        def gen() -> Iterator[str]:
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            try:
                yield from stream
            finally:
                with lock:
                    active[0] -= 1
        return TextStream(gen(), lambda: stream.result)

    monkeypatch.setattr(model, "generate", counting)
    results: list[tuple[int, Any]] = []

    def call(i: int) -> None:
        results.append(client.post("/v1/completions", {"prompt": golden["input_ids"][:4 + i],
                                                       "max_tokens": 2}))

    threads = [threading.Thread(target=call, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    assert [s for s, _ in results] == [200, 200, 200]
    assert peak[0] == 1
    assert max(r["pinfer"]["queue_seconds"] for _, r in results) > 0


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_stream_disconnect_stops_generation(client: Client, server: Any,
                                            golden: dict[str, Any], temperature: float) -> None:
    """The client reads one event and hangs up: the generation (speculative at T=0) stops at
    once and the next request is served."""
    body = json.dumps({"prompt": golden["prompt"], "max_tokens": 200, "stream": True,
                       "temperature": temperature, "seed": 1}).encode()
    sock = socket.create_connection(("127.0.0.1", client.port), timeout=120)
    sock.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json"
                 b"\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
    seen = b""
    while b'"text": "' not in seen.split(b"\r\n\r\n", 1)[-1]:
        data = sock.recv(4096)
        assert data, "server closed the stream early"
        seen += data
    sock.close()
    t0 = time.perf_counter()
    r = client.ok("/v1/completions", {"prompt": golden["prompt"], "max_tokens": 1})
    assert time.perf_counter() - t0 < 120
    assert r["choices"][0]["text"]
    gone = [s for s in server.engine.recent if s.cancelled]
    assert gone and gone[-1].completion_tokens < 200
    assert server.engine.model.n_past == len(server.engine.model.context_tokens)


def test_api_key(server: Any) -> None:
    from pinfer.server import make_server

    srv = make_server(server.engine, "127.0.0.1", 0, api_key="sesame")
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    try:
        c = Client(srv.server_address[1])
        status, body = c.request("GET", "/v1/models")
        assert status == 401 and body["error"]["code"] == "invalid_api_key"
        assert c.request("GET", "/v1/models", headers={"Authorization": "Bearer sesame"})[0] \
            == 200
        assert c.request("GET", "/health")[0] == 200       # health stays open
        status, _ = c.post("/v1/completions", {"prompt": "x"})
        assert status == 401
    finally:
        srv.shutdown()
        srv.server_close()
