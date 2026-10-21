"""pinfer Python example: plain generation, streaming chat, token-level access.

    python examples/python/generate.py models/pq/qwen35-0.8b-bf16.pq [models/Qwen3.5-0.8B]

The tokenizer argument is optional when `pinfer convert` wrote <model>.tokenizer/ next to
the .pq file.
"""
from __future__ import annotations

import sys

import pinfer


def main() -> None:
    model_path = sys.argv[1]
    tokenizer = sys.argv[2] if len(sys.argv) > 2 else None
    with pinfer.Model.load(model_path, tokenizer=tokenizer, n_threads=4) as model:
        # 1. Greedy continuation (deterministic).
        r = model.generate("The Raspberry Pi 4 is a small computer that", max_new_tokens=24)
        print(f"greedy: {r.text!r}\n  ({len(r.token_ids)} tokens, {r.decode_tps:.2f} tok/s)")

        # 2. Streaming chat with sampling.
        messages = [{"role": "user", "content": "Give me one fun fact about raspberries."}]
        print("chat: ", end="")
        for chunk in model.chat(messages, max_new_tokens=64, temperature=0.7, top_p=0.9,
                                seed=0, stream=True):
            print(chunk, end="", flush=True)
        print()

        # 3. Token ids and raw logits.
        ids = model.tokenize("Hello")
        model.reset()
        logits = model.eval(ids)
        print(f"eval: {len(ids)} tokens -> logits {logits.shape}, argmax "
              f"{model.detokenize([int(logits.argmax())])!r}")


if __name__ == "__main__":
    main()
