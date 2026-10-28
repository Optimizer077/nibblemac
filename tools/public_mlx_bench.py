"""Synchronized MLX-LM model adapter; one request, fresh unquantized cache.

Uses model calls rather than generate_step's one-token asynchronous look-ahead,
so no decode work is hidden in the prefill boundary. EOS is deliberately ignored.
"""
import argparse
import json
import resource
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import wired_limit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("ids")
    ap.add_argument("generate", type=int)
    ap.add_argument("quality_chunks", type=int)
    args = ap.parse_args()
    ids = np.fromfile(args.ids, dtype="<i4").tolist()
    if not ids or args.generate < 2 or args.quality_chunks < 0:
        ap.error("invalid arguments")
    model, _ = load(args.model)

    def run(count):
        cache = make_prompt_cache(model)
        start = time.perf_counter()
        # Like MLX-LM generate_step, prefill all but the final token without the head.
        prefix = ids[:-1]
        for pos in range(0, len(prefix), 64):
            model(mx.array([prefix[pos:pos + 64]]), cache=cache)
            mx.eval([c.state for c in cache])
        logits = model(mx.array([ids[-1:]]), cache=cache)
        last = logits[:, -1, :].astype(mx.float32)
        token = mx.argmax(last, axis=-1)
        mx.eval(token)
        generated = [token.item()]
        prefill_s = time.perf_counter() - start
        if not mx.all(mx.isfinite(last)).item():
            raise ValueError("Non-finite prefill logits")
        start = time.perf_counter()
        for _ in range(count - 1):
            logits = model(token[None], cache=cache)[:, -1, :].astype(mx.float32)
            token = mx.argmax(logits, axis=-1)
            mx.eval(token)
            generated.append(token.item())
        mx.synchronize()
        decode_s = time.perf_counter() - start
        if not mx.all(mx.isfinite(logits)).item():
            raise ValueError("Non-finite final logits")
        return {"prefill_tps": len(ids) / prefill_s, "decode_tps": (count - 1) / decode_s,
                "prefill_seconds": prefill_s, "decode_seconds": decode_s,
                "generated_ids": generated}

    with wired_limit(model):
        if args.quality_chunks:
            if len(ids) != args.quality_chunks * 512:
                raise ValueError("quality input length mismatch")
            nll = 0.0
            for chunk in range(args.quality_chunks):
                cache = make_prompt_cache(model)
                tokens = ids[chunk * 512:(chunk + 1) * 512]
                for pos in range(0, 512, 64):
                    logits = model(mx.array([tokens[pos:pos + 64]]), cache=cache)
                    mx.eval([c.state for c in cache])
                    if pos < 256:
                        continue
                    values = np.array(logits[0].astype(mx.float32)).astype(np.float64)
                    if not np.isfinite(values).all():
                        raise ValueError("Non-finite quality logits")
                    for j in range(min(64, 511 - pos)):
                        row = values[j]
                        maximum = row.max()
                        nll += maximum + np.log(np.exp(row - maximum).sum()) - row[tokens[pos + j + 1]]
            result = {"ppl": float(np.exp(nll / (args.quality_chunks * 255))),
                      "nll": nll, "tokens_scored": args.quality_chunks * 255}
        else:
            run(5)
            result = run(args.generate)
            result.update(peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576,
                          mlx_peak_allocated_mb=mx.get_peak_memory() / 1048576)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
