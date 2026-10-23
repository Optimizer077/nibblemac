"""Label decision requests with Cerno 4B through a running llama-server (teacher labels).

Same protocol as cerno-inference's ``engine_llama.py`` (the reference of ``ref_llama_pc.jsonl``):
one row per question, ``n_predict=1``, ``n_probs=260``, temperature 0, prompt cache on; the
label probabilities are read pre-sampling from the top-n list and renormalised as ``p^(1/T)``.
Output: one JSON line per request with, per question, the row text, the label logprobs and the
teacher's option distribution.

    python label_llama.py --requests /root/distill/requests.json --out /root/distill/labels.jsonl \
        --tokenizer models/Qwen3.5-0.8B --base-url http://127.0.0.1:8090
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cerno_prompt import CERNO_TEMPERATURE, label_table, readout_from_logprobs, request_rows


def post(base_url: str, payload: dict) -> dict:
    req = urllib.request.Request(base_url + "/completion", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def row_logprobs(base_url: str, prompt: str, labels: tuple[str, ...], n_probs: int = 260
                 ) -> tuple[dict[str, float | None], dict]:
    out = post(base_url, {"prompt": prompt, "n_predict": 1, "n_probs": max(n_probs, len(labels) + 2),
                          "temperature": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
                          "repeat_penalty": 1.0, "cache_prompt": True})
    cp = out.get("completion_probabilities") or []
    if not cp:
        raise RuntimeError(f"no completion_probabilities: {json.dumps(out)[:300]}")
    cands = cp[0].get("top_logprobs") or cp[0].get("top_probs") or cp[0].get("probs") or []
    raw: dict[str, float] = {}
    for c in cands:
        t = c.get("token", c.get("tok_str"))
        raw[t] = float(c["logprob"]) if "logprob" in c else float(np_log(c["prob"]))
    return {lab: raw.get(lab) for lab in labels}, {"tokens_evaluated": out.get("tokens_evaluated"),
                                                   "tokens_cached": out.get("tokens_cached"),
                                                   "timings": out.get("timings")}


def np_log(x: float) -> float:
    import math
    return math.log(x) if x > 0 else float("-inf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True, help="HF dir with Qwen3.5 tokenizer")
    ap.add_argument("--base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--temperature", type=float, default=CERNO_TEMPERATURE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip request names already in --out")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    labels = label_table(tok)
    reqs = json.loads(a.requests.read_text(encoding="utf-8"))
    if a.limit:
        reqs = reqs[:a.limit]
    done: set[str] = set()
    if a.resume and a.out.exists():
        done = {json.loads(l)["name"] for l in a.out.read_text(encoding="utf-8").splitlines() if l.strip()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    n_rows = 0
    with a.out.open("a" if a.resume else "w", encoding="utf-8") as f:
        for i, r in enumerate(reqs):
            if r["name"] in done:
                continue
            rows = request_rows(tok, labels, r["state"], r["questions"])
            rec = {"name": r["name"], "source": r.get("source", ""), "rows": []}
            t0 = time.perf_counter()
            for row in rows:
                lp, info = row_logprobs(a.base_url, row.text, row.labels)
                p = readout_from_logprobs([lp[lab] for lab in row.labels], a.temperature)
                rec["rows"].append({"question": row.key, "type": row.type, "prompt": row.text,
                                    "labels": list(row.labels), "names": [str(n) for n in row.names],
                                    "label_top": lp, "probs": p, **info})
                n_rows += 1
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"[{i + 1}/{len(reqs)}] rows={n_rows} {time.perf_counter() - t_all:.0f}s",
                      flush=True)
    print(f"done: {len(reqs)} requests, {n_rows} rows, {time.perf_counter() - t_all:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
