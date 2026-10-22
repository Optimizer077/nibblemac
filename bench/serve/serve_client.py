"""Measure a running ``pinfer serve`` (or any OpenAI-compatible server); standard library only.

Prints one ``RESULT {json}`` line per request (the format ``harness/pi.py bench`` collects):

    python bench/serve/serve_client.py --url http://127.0.0.1:8091 --label generated \
        --tests single,conversation,alternating --max-tokens 64

Tests:
  single        one chat request without and one with streaming (TTFT, tok/s)
  conversation  a 4-turn chat, streamed; each turn resends the whole conversation
  alternating   two conversations, two turns each, interleaved A1 B1 A2 B2

TTFT = request sent -> first non-empty content delta, measured by the client. ``cached`` and
``prefill_s`` come from the server's ``pinfer`` extension (absent on other servers).
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from typing import Any
from urllib.parse import urlsplit

SYSTEM = {"role": "system", "content": "You are a helpful assistant running on a Raspberry Pi."}
TURNS = ["What is a Raspberry Pi? Answer in two sentences.",
         "Which models have 4 GB of RAM or more?",
         "Give three ideas for projects at home.",
         "Summarize our conversation in one sentence."]
ALT = {"A": ["Name three uses of Python.", "Which of them is the most popular?"],
       "B": ["Explain what a CPU cache is.", "Why does it make programs faster?"]}


class Client:
    """Tiny OpenAI chat client over http.client (keep-alive, SSE parsing)."""

    def __init__(self, url: str, timeout: float) -> None:
        u = urlsplit(url)
        self.host, self.port = u.hostname or "127.0.0.1", u.port or 80
        self.timeout = timeout

    def _post(self, path: str, body: dict[str, Any]) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read().decode(errors='replace')}")
        return resp

    def chat(self, messages: list[dict[str, Any]], max_tokens: int,
             stream: bool) -> dict[str, Any]:
        """One request; returns the reply text, usage, the pinfer extension and timings."""
        body = {"model": "pinfer", "messages": messages, "max_tokens": max_tokens,
                "temperature": 0, "stream": stream}
        t0 = time.perf_counter()
        if not stream:
            data = json.loads(self._post("/v1/chat/completions", body).read())
            t1 = time.perf_counter()
            return {"text": data["choices"][0]["message"]["content"], "usage": data["usage"],
                    "pinfer": data.get("pinfer") or {}, "total_s": t1 - t0, "ttft_s": None,
                    "finish": data["choices"][0]["finish_reason"]}
        body["stream_options"] = {"include_usage": True}
        resp = self._post("/v1/chat/completions", body)
        text: list[str] = []
        usage: dict[str, Any] = {}
        ext: dict[str, Any] = {}
        first, finish, chunks = 0.0, None, 0
        for raw in resp:   # SSE: "data: {...}" lines separated by blank lines
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                resp.read()   # the end of the chunked body, so the socket closes cleanly
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage, ext = ev["usage"], ev.get("pinfer") or ext
            for ch in ev.get("choices", []):
                piece = ch.get("delta", {}).get("content") or ""
                if piece:
                    first = first or time.perf_counter()
                    text.append(piece)
                    chunks += 1
                finish = ch.get("finish_reason") or finish
        t1 = time.perf_counter()
        return {"text": "".join(text), "usage": usage, "pinfer": ext, "total_s": t1 - t0,
                "ttft_s": (first - t0) if first else None, "finish": finish, "chunks": chunks}


def record(label: str, test: str, step: str, r: dict[str, Any]) -> dict[str, Any]:
    """The RESULT row of one request."""
    u, ext = r["usage"], r["pinfer"]
    n = int(u.get("completion_tokens", 0))
    ttft = r["ttft_s"]
    stream_tps = (n - 1) / (r["total_s"] - ttft) if ttft is not None and n > 1 and \
        r["total_s"] > ttft else None
    spec = ext.get("speculative") or {}
    return {"bench": "serve", "label": label, "test": test, "step": step,
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": n,
            "cached_tokens": ext.get("cached_tokens"), "cache_source": ext.get("cache_source"),
            "ttft_s": None if ttft is None else round(ttft, 3), "total_s": round(r["total_s"], 3),
            "client_tps": None if stream_tps is None else round(stream_tps, 2),
            "server_prefill_s": ext.get("prefill_seconds"),
            "server_decode_tps": ext.get("tokens_per_second"),
            "tokens_per_pass": spec.get("tokens_per_pass"), "finish": r["finish"],
            "chunks": r.get("chunks")}


def emit(row: dict[str, Any]) -> None:
    print("RESULT " + json.dumps(row), flush=True)


def run(c: Client, label: str, tests: list[str], max_tokens: int) -> None:
    """Run the selected tests against the server."""
    if "single" in tests:
        msgs = [SYSTEM, {"role": "user", "content": TURNS[0]}]
        emit(record(label, "single", "blocking", c.chat(msgs, max_tokens, stream=False)))
        emit(record(label, "single", "stream", c.chat(msgs, max_tokens, stream=True)))
    if "conversation" in tests:
        msgs = [SYSTEM]
        for i, q in enumerate(TURNS, 1):
            msgs.append({"role": "user", "content": q})
            r = c.chat(msgs, max_tokens, stream=True)
            emit(record(label, "conversation", f"turn{i}", r))
            msgs.append({"role": "assistant", "content": r["text"]})
    if "alternating" in tests:
        convs: dict[str, list[dict[str, Any]]] = {k: [SYSTEM] for k in ALT}
        for turn in range(2):
            for name, qs in ALT.items():
                convs[name].append({"role": "user", "content": qs[turn]})
                r = c.chat(convs[name], max_tokens, stream=True)
                emit(record(label, "alternating", f"{name}{turn + 1}", r))
                convs[name].append({"role": "assistant", "content": r["text"]})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--label", default="run", help="name of the server configuration")
    p.add_argument("--tests", default="single,conversation,alternating")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--warmup", action="store_true", help="one short request first")
    a = p.parse_args(argv)
    c = Client(a.url, a.timeout)
    if a.warmup:
        c.chat([{"role": "user", "content": "Hi"}], 4, stream=False)
    run(c, a.label, a.tests.split(","), a.max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
