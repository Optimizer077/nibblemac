"""The ``pinfer`` command line: generate, chat, bench, convert, info, serve.

    pinfer generate -m model.pq -p "The Raspberry Pi 4 is" -n 64
    pinfer chat -m model.pq [--system "..."] [--think] [--reprefill]
    pinfer bench -m model.pq --prompt-tokens 64 -n 32 [--json]
    pinfer convert models/Qwen3.5-0.8B models/pq/qwen35-0.8b-bf16.pq [--no-mtp]
    pinfer info -m model.pq
    pinfer serve -m model.pq [--host 0.0.0.0] [--port 8080]      # OpenAI-compatible HTTP
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .convert import copy_tokenizer, run_converter
from .errors import PinferError
from .model import Model
from .speculative import iter_speculative
from .streaming import GenerationResult, TextStream

logger = logging.getLogger("pinfer.cli")


def _add_model_args(p: argparse.ArgumentParser, tokenizer: bool = True) -> None:
    p.add_argument("-m", "--model", required=True, help=".pq model file")
    if tokenizer:
        p.add_argument("--tokenizer", help="tokenizer dir or tokenizer.json "
                       "(default: <model>.tokenizer/ or $PINFER_TOKENIZER)")
    p.add_argument("-t", "--threads", type=int, default=4, help="worker threads (default 4)")
    p.add_argument("-c", "--ctx", "--max-ctx", type=int, default=4096,
                   help="max context tokens (default 4096)")
    p.add_argument("--kv-dtype", choices=("fp32", "fp16", "int8"), default="fp32",
                   help="KV-cache storage (fp16 / int8: smaller, faster at long context)")
    p.add_argument("--lib", help="path to libpinfer.so (default: search, see docs)")


def _add_sampling_args(p: argparse.ArgumentParser, default_n: int) -> None:
    p.add_argument("-n", "--max-new-tokens", type=int, default=default_n)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int)
    p.add_argument("--stop", action="append", default=[], help="stop string (repeatable)")
    p.add_argument("--stats", action="store_true", help="print speed statistics to stderr")
    p.add_argument("--speculative", action="store_true",
                   help="MTP speculative greedy decoding (same output, blocking, faster)")
    p.add_argument("--spec-k", type=int, help="draft tokens per pass (default: library's)")
    p.add_argument("--draft-vocab", type=int, help="draft over token ids < N only")
    p.add_argument("--adaptive", action="store_true",
                   help="speculative: let the engine lower k when plain steps are faster")


def _load(a: argparse.Namespace, tokenizer: bool = True) -> Model:
    tok = (a.tokenizer if a.tokenizer else None) if tokenizer else False
    return Model.load(a.model, tokenizer=tok, n_threads=a.threads, max_ctx=a.ctx,
                      lib_path=a.lib, kv_dtype=a.kv_dtype)


def _sampling_kwargs(a: argparse.Namespace) -> dict[str, Any]:
    return {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "seed": a.seed,
            "stop": a.stop or None, "speculative": a.speculative, "spec_k": a.spec_k,
            "draft_vocab": a.draft_vocab, "spec_adaptive": a.adaptive or None}


def _print_stats(r: GenerationResult | None) -> None:
    if r is None:
        return
    print(f"[{r.prompt_tokens} prompt tok ({r.cached_tokens} cached), {len(r.token_ids)} new, "
          f"finish={r.finish_reason}, prefill {r.prefill_tps:.2f} tok/s, "
          f"decode {r.decode_tps:.2f} tok/s]", file=sys.stderr)
    if r.spec_stats:
        st = r.spec_stats
        per = 1 + st["accepted"] / st["rounds"] if st["rounds"] else 0.0
        secs = r.decode_seconds
        print(f"[speculative k={st['k']}: {st['rounds']} passes, {st['accepted']}/"
              f"{st['drafted']} drafts accepted, {per:.2f} tok/pass, "
              f"{len(r.token_ids) / secs if secs else 0:.2f} tok/s incl. prefill]",
              file=sys.stderr)


def cmd_generate(a: argparse.Namespace) -> int:
    prompt = a.prompt if a.prompt is not None else sys.stdin.read()
    with _load(a) as m:
        stream = m.generate(prompt, a.max_new_tokens, stream=True, **_sampling_kwargs(a))
        assert isinstance(stream, TextStream)
        for chunk in stream:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()  # keep the stats line (stderr) after the text
        if a.stats:
            _print_stats(stream.result)
    return 0


def _turn_suffix(user: str, think: bool) -> str:
    """What the Qwen chat template appends for one more user turn + the assistant header."""
    header = "<think>\n" if think else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{user.strip()}<|im_end|>\n<|im_start|>assistant\n{header}"


def cmd_chat(a: argparse.Namespace) -> int:
    """Multi-turn chat. By default the engine state is kept between turns: the next turn is
    appended to the tokens already in context (the previous reply stays as generated, think
    block included, closed by <|im_end|>), so only the new tokens are prefilled. --reprefill
    re-renders the whole conversation each turn instead (the reference template drops old
    think blocks, so the two prompts differ slightly)."""
    messages: list[dict[str, str]] = []
    if a.system:
        messages.append({"role": "system", "content": a.system})
    interactive = a.prompt is None
    if interactive:
        print("pinfer chat: empty line or /exit quits, /reset clears the history",
              file=sys.stderr)
    with _load(a) as m:
        ctx_ids: list[int] = []     # tokens in context + the last reply (keep-state mode)
        im_end = m.tokenize("<|im_end|>")
        while True:
            if interactive:
                try:
                    line = input("> ").strip()
                except EOFError:
                    break
            else:
                line = a.prompt
            if line in ("", "/exit", "/quit"):
                break
            if line == "/reset":
                messages = messages[:1] if a.system else []
                ctx_ids = []
                continue
            messages.append({"role": "user", "content": line})
            if a.reprefill or not ctx_ids:
                text = m.apply_chat_template(messages, True, a.think)
                prompt: str | list[int] = text
                if not a.reprefill:
                    ctx_ids = m.tokenize(text)
                    prompt = ctx_ids
            else:
                ctx_ids = ctx_ids + m.tokenize("\n" + _turn_suffix(line, a.think))
                prompt = ctx_ids
            stream = m.generate(prompt, a.max_new_tokens, stream=True, **_sampling_kwargs(a))
            assert isinstance(stream, TextStream)
            for chunk in stream:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
            r = stream.result
            reply = r.text if r else ""
            messages.append({"role": "assistant", "content": reply})
            if r and not a.reprefill:
                gen = list(r.token_ids)
                ctx_ids = ctx_ids + gen + (im_end if gen[-1:] != im_end else [])
            if a.stats:
                _print_stats(r)
            if not interactive:
                break
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    """Prefill + greedy decode speed through the Python API (no tokenizer needed)."""
    with _load(a, tokenizer=False) as m:
        rng = np.random.default_rng(0)
        ids = rng.integers(100, 100_000, size=a.prompt_tokens).tolist()
        runs = []
        for _ in range(a.repeat):
            if a.spec_k is not None:
                runs.append(_bench_spec(m, ids, a.max_new_tokens, a.spec_k))
                continue
            m.reset()
            t0 = time.perf_counter()
            logits = m.eval(ids)
            t_prefill = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(a.max_new_tokens - 1):
                logits = m.eval([int(np.argmax(logits))])
            t_decode = time.perf_counter() - t0
            runs.append({"prefill_tps": a.prompt_tokens / t_prefill,
                         "decode_tps": (a.max_new_tokens - 1) / t_decode
                         if a.max_new_tokens > 1 else 0.0})
        keys = ("total_tps",) if a.spec_k is not None else ("prefill_tps", "decode_tps")
        best = {k: max(r[k] for r in runs) for k in keys}
        info = m.info()
        out = {"engine": f"pinfer-{info['build'].get('matvec', '?')}", "version": m.version,
               "model": str(m.path), "threads": a.threads, "n_prompt": a.prompt_tokens,
               "n_gen": a.max_new_tokens, "spec_k": a.spec_k, "runs": runs, **best}
    if a.json:
        print(json.dumps(out))
    else:
        speeds = ", ".join(f"{k} {v:.2f} tok/s" for k, v in best.items())
        print(f"{out['engine']} {out['version']}  threads={a.threads}  prompt={a.prompt_tokens} "
              f"gen={a.max_new_tokens}  spec_k={a.spec_k}  best of {a.repeat}: {speeds}")
    return 0


def _bench_spec(m: Model, ids: list[int], n_gen: int, k: int) -> dict[str, Any]:
    """One speculative run: generated tokens / wall time of the whole C call."""
    state: dict[str, Any] = {}
    toks = list(iter_speculative(m, ids, n_gen, set(), state, k=k))
    secs = state["decode_seconds"]
    return {"total_tps": len(toks) / secs if secs else 0.0, "n_gen": len(toks),
            "seconds": secs, **(state["spec_stats"] or {})}


def cmd_serve(a: argparse.Namespace) -> int:
    """OpenAI-compatible HTTP server (pinfer.server)."""
    from .server import Defaults, EngineConfig, serve

    def stop(_sig: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)   # `kill` shuts down like Ctrl-C
    defaults = Defaults(max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p,
                        top_k=a.top_k, enable_thinking=a.think)
    with _load(a) as m:
        cfg = EngineConfig(model_name=a.model_name or Path(a.model).stem,
                           speculative=a.speculative, spec_k=a.spec_k, reuse=not a.no_reuse,
                           history=a.history, state_cache_mb=a.state_cache_mb,
                           snapshot_min_tokens=a.snapshot_min_tokens, defaults=defaults)
        serve(m, cfg, a.host, a.port, a.api_key or os.environ.get("PINFER_API_KEY"))
    return 0


def cmd_convert(a: argparse.Namespace, rest: Sequence[str]) -> int:
    rc = run_converter([a.model_dir, a.out, *rest])
    if rc == 0 and not a.no_tokenizer:
        copy_tokenizer(a.model_dir, a.out)
    return rc


def cmd_info(a: argparse.Namespace) -> int:
    with _load(a) as m:
        print(json.dumps(m.info(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the ``pinfer`` command."""
    p = argparse.ArgumentParser(prog="pinfer", description="Qwen3.5 inference on the Pi 4")
    p.add_argument("--version", action="version", version=f"pinfer {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="log to stderr (INFO)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="continue a text prompt")
    _add_model_args(g)
    g.add_argument("-p", "--prompt", help="prompt text (default: read stdin)")
    _add_sampling_args(g, 128)

    c = sub.add_parser("chat", help="chat with the model (Qwen chat template)")
    _add_model_args(c)
    c.add_argument("-p", "--prompt", help="ask one question and exit")
    c.add_argument("--system", help="system prompt")
    c.add_argument("--think", action="store_true", help="let the model write a <think> block")
    c.add_argument("--reprefill", action="store_true",
                   help="re-render and re-prefill the whole conversation every turn")
    _add_sampling_args(c, 512)

    b = sub.add_parser("bench", help="measure prefill and decode speed")
    _add_model_args(b, tokenizer=False)
    b.add_argument("--prompt-tokens", type=int, default=64)
    b.add_argument("-n", "--max-new-tokens", type=int, default=32)
    b.add_argument("--repeat", type=int, default=1)
    b.add_argument("--json", action="store_true", help="one JSON line")
    b.add_argument("--spec-k", type=int, help="measure speculative decoding with k drafts "
                   "(whole-call tok/s, prefill included; EOS still stops)")

    s = sub.add_parser("serve", help="OpenAI-compatible HTTP server (/v1/chat/completions)")
    _add_model_args(s)
    s.add_argument("--host", default="127.0.0.1",
                   help="address to listen on (default 127.0.0.1; 0.0.0.0 = all interfaces)")
    s.add_argument("--port", type=int, default=8080, help="TCP port (default 8080)")
    s.add_argument("--model-name", help="model id in /v1/models (default: the .pq file name)")
    s.add_argument("--api-key", help="require 'Authorization: Bearer KEY' (or $PINFER_API_KEY)")
    s.add_argument("--speculative", action=argparse.BooleanOptionalAction, default=True,
                   help="greedy requests use MTP speculative decoding (default on)")
    s.add_argument("--spec-k", type=int, help="draft tokens per pass (default: library's, 2)")
    s.add_argument("--no-reuse", action="store_true",
                   help="re-prefill every prompt (no prefix reuse, no snapshots; for comparison)")
    s.add_argument("--history", choices=("generated", "template"), default="generated",
                   help="earlier assistant turns as generated (default) or as the template "
                   "renders them")
    s.add_argument("--state-cache-mb", type=float, default=256.0,
                   help="memory for state snapshots (default 256; 0 = none)")
    s.add_argument("--snapshot-min-tokens", type=int, default=64,
                   help="shortest prefix worth a snapshot (default 64)")
    s.add_argument("--max-tokens", type=int, default=1024,
                   help="chat max_tokens when a request gives none (default 1024)")
    s.add_argument("--temperature", type=float, default=0.0,
                   help="when a request gives none (default 0 = greedy)")
    s.add_argument("--top-p", type=float, default=1.0, help="when a request gives none")
    s.add_argument("--top-k", type=int, default=0, help="when a request gives none")
    s.add_argument("--think", action="store_true",
                   help="enable_thinking when a request does not set chat_template_kwargs")

    v = sub.add_parser("convert", help="HF safetensors checkpoint -> .pq "
                       "(extra options go to tools/convert.py)")
    v.add_argument("model_dir")
    v.add_argument("out")
    v.add_argument("--no-tokenizer", action="store_true",
                   help="do not copy tokenizer files to <out>.tokenizer/")

    i = sub.add_parser("info", help="print library, build and model facts")
    _add_model_args(i)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``pinfer`` console script."""
    parser = build_parser()
    a, rest = parser.parse_known_args(argv)
    if rest and a.cmd != "convert":
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    logging.basicConfig(level=logging.INFO if a.verbose or a.cmd in ("convert", "serve")
                        else logging.WARNING,
                        format=("%(asctime)s " if a.cmd == "serve" else "")
                        + "%(levelname)s %(name)s: %(message)s")
    try:
        if a.cmd == "convert":
            return cmd_convert(a, rest)
        return {"generate": cmd_generate, "chat": cmd_chat, "bench": cmd_bench,
                "info": cmd_info, "serve": cmd_serve}[a.cmd](a)
    except (PinferError, OSError, ValueError) as e:
        print(f"pinfer: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
