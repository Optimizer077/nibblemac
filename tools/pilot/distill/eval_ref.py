"""Score a Hugging Face Qwen3.5 model (base or distilled student) on labelled decision rows and
compare with the teacher: top-answer agreement, mean / max |dp|, per question type.

Inputs: cerno-pi's ``results/ref_llama_pc.jsonl`` (52 held-out questions, Cerno 4B through
llama.cpp) or a ``label_llama.py`` output. Every row's prompt text is tokenized exactly as the
teacher saw it (checked against ``server_ids`` when present) and the label logits at the last
position are read as ``softmax(z / T)``.

    python eval_ref.py --model models/Qwen3.5-0.8B --ref ref_llama_pc.jsonl --out zs-0.8b.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cerno_prompt import CERNO_TEMPERATURE, label_table, readout


def load_model(path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    kw = {"dtype": torch.bfloat16, "device_map": device}   # straight to the device: low host RAM
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(path, **kw)
    model.eval()
    return tok, model


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("repeat", 0):
            continue
        for row in r["rows"]:
            rows.append({"name": r["name"], "question": row["question"], "prompt": row["prompt"],
                         "labels": row["labels"], "p_teacher": row["probs"],
                         "server_ids": row.get("server_ids"), "type": row.get("type")})
    return rows


def q_type(row: dict) -> str:
    if row["type"]:
        return row["type"]
    if len(row["labels"]) == 2:
        return "noul"
    return "choice/score"


@torch.no_grad()
def score_row(model, ids: list[int], label_ids: list[int], device: str, temperature: float) -> list[float]:
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x, logits_to_keep=1)
    z = out.logits[0, -1].float()
    return readout(z[label_ids].cpu().numpy(), temperature)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=CERNO_TEMPERATURE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tok, model = load_model(a.model, a.device)
    labels = label_table(tok)
    rows = load_rows(a.ref)
    t0 = time.perf_counter()
    mismatched = 0
    for row in rows:
        ids = tok.encode(row["prompt"], add_special_tokens=False)
        if row["server_ids"] is not None and ids != row["server_ids"]:
            mismatched += 1
            ids = row["server_ids"]
        lab = [labels.ids[labels.names.index(s)] for s in row["labels"]]
        row["p_student"] = score_row(model, ids, lab, a.device, a.temperature)
        row["tokens"] = len(ids)
    secs = time.perf_counter() - t0
    for row in rows:
        pt, ps = np.array(row["p_teacher"]), np.array(row["p_student"])
        row["max_dp"] = float(np.abs(pt - ps).max())
        row["agree"] = bool(pt.argmax() == ps.argmax())
        row["teacher_top_p"] = float(pt.max())
    all_dp = [abs(x - y) for row in rows for x, y in zip(row["p_teacher"], row["p_student"])]
    by_type: dict[str, dict] = {}
    for t in sorted({q_type(r) for r in rows}):
        sub = [r for r in rows if q_type(r) == t]
        by_type[t] = {"rows": len(sub), "agree": sum(r["agree"] for r in sub),
                      "mean_max_dp": round(st.mean(r["max_dp"] for r in sub), 4),
                      "max_dp": round(max(r["max_dp"] for r in sub), 4)}
    summary = {"model": a.model, "tag": a.tag, "ref": str(a.ref), "rows": len(rows), "options": len(all_dp),
               "agree": sum(r["agree"] for r in rows), "agree_pct": round(100 * sum(r["agree"] for r in rows) / len(rows), 1),
               "mean_abs_dp_options": round(st.mean(all_dp), 4),
               "mean_max_dp_rows": round(st.mean(r["max_dp"] for r in rows), 4),
               "max_dp": round(max(r["max_dp"] for r in rows), 4),
               "median_max_dp": round(st.median(r["max_dp"] for r in rows), 4),
               "tokenization_mismatches": mismatched, "seconds": round(secs, 1), "by_type": by_type,
               "temperature": a.temperature, "device": a.device}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
