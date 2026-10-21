"""The ``pinfer`` command line: generate, chat, bench, convert, info.

    pinfer generate -m model.pq -p "The Raspberry Pi 4 is" -n 64
    pinfer chat -m model.pq [--system "..."] [--think]
    pinfer bench -m model.pq --prompt-tokens 64 -n 32 [--json]
    pinfer convert models/Qwen3.5-0.8B models/pq/qwen35-0.8b-bf16.pq [--no-mtp]
    pinfer info -m model.pq
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence

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
    p.add_argument("-c", "--ctx", type=int, default=4096, help="max context tokens")
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


def _load(a: argparse.Namespace, tokenizer: bool = True) -> Model:
    tok = (a.tokenizer if a.tokenizer else None) if tokenizer else False
    return Model.load(a.model, tokenizer=tok, n_threads=a.threads, max_ctx=a.ctx,
                      lib_path=a.lib)


def _sampling_kwargs(a: argparse.Namespace) -> dict:
    return {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "seed": a.seed,
            "stop": a.stop or None, "speculative": a.speculative, "spec_k": a.spec_k,
            "draft_vocab": a.draft_vocab}


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


def cmd_chat(a: argparse.Namespace) -> int:
    messages: list[dict[str, str]] = []
    if a.system:
        messages.append({"role": "system", "content": a.system})
    interactive = a.prompt is None
    if interactive:
        print("pinfer chat: empty line or /exit quits, /reset clears the history",
              file=sys.stderr)
    with _load(a) as m:
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
                continue
            messages.append({"role": "user", "content": line})
            stream = m.chat(messages, a.max_new_tokens, enable_thinking=a.think, stream=True,
                            **_sampling_kwargs(a))
            assert isinstance(stream, TextStream)
            for chunk in stream:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
            reply =stream.result.text if stream.result else ""
            messages.append({"role": "assistant", "content": reply})
            if a.stats:
                _print_stats(stream.result)
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


def _bench_spec(m: Model, ids: list[int], n_gen: int, k: int) -> dict:
    """One speculative run: generated tokens / wall time of the whole C call."""
    state: dict = {}
    toks = list(iter_speculative(m, ids, n_gen, set(), state, k=k))
    secs = state["decode_seconds"]
    return {"total_tps": len(toks) / secs if secs else 0.0, "n_gen": len(toks),
            "seconds": secs, **(state["spec_stats"] or {})}


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
    _add_sampling_args(c, 512)

    b = sub.add_parser("bench", help="measure prefill and decode speed")
    _add_model_args(b, tokenizer=False)
    b.add_argument("--prompt-tokens", type=int, default=64)
    b.add_argument("-n", "--max-new-tokens", type=int, default=32)
    b.add_argument("--repeat", type=int, default=1)
    b.add_argument("--json", action="store_true", help="one JSON line")
    b.add_argument("--spec-k", type=int, help="measure speculative decoding with k drafts "
                   "(whole-call tok/s, prefill included; EOS still stops)")

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
    logging.basicConfig(level=logging.INFO if a.verbose or a.cmd == "convert" else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        if a.cmd == "convert":
            return cmd_convert(a, rest)
        return {"generate": cmd_generate, "chat": cmd_chat, "bench": cmd_bench,
                "info": cmd_info}[a.cmd](a)
    except (PinferError, OSError, ValueError) as e:
        print(f"pinfer: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
