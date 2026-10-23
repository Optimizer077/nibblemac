"""Distil Cerno 4B's option distributions into a Qwen3.5 student (LoRA or full fine-tune).

Loss: KL(teacher || student) over the option labels at the answer slot, with the student's label
logits read as ``softmax(z / T)`` at Cerno's temperature, so the student is used with the same
readout as the teacher. One row per step (no padding: the hybrid linear-attention layers then
see exactly the inference-time sequence), gradients accumulated over ``--accum`` rows.

    python train_student.py --model models/Qwen3.5-0.8B --labels /root/distill/labels.jsonl \
        --out /root/distill/student-0.8b --epochs 3 --lr 2e-4 --rank 16
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cerno_prompt import CERNO_TEMPERATURE, label_table
from eval_ref import load_model, load_rows


def lora_targets(model) -> list[str]:
    """Every nn.Linear inside the decoder layers (attention, linear attention, MLP)."""
    names = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and ".layers." in n and "lm_head" not in n and "visual" not in n:
            names.append(n)
    return names


def kl_row(model, ids: list[int], label_ids: list[int], p_teacher: list[float], device: str,
           temperature: float) -> torch.Tensor:
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x, logits_to_keep=1)
    z = out.logits[0, -1].float()[label_ids] / temperature
    logq = F.log_softmax(z, dim=-1)
    p = torch.tensor(p_teacher, device=device, dtype=torch.float32)
    return (p * (torch.log(p.clamp_min(1e-9)) - logq)).sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", type=Path, required=True, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=CERNO_TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log", type=Path)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok, model = load_model(a.model, a.device)
    labels = label_table(tok)
    rows, seen_prompts = [], set()
    for fi, p in enumerate(a.labels):
        for r in load_rows(p):
            if r["prompt"] in seen_prompts:        # the same row generated twice (across seeds)
                continue
            seen_prompts.add(r["prompt"])
            r["name"] = f"{fi}:{r['name']}"        # request names repeat across label files
            rows.append(r)
    for r in rows:
        r["ids"] = tok.encode(r["prompt"], add_special_tokens=False)
        r["label_ids"] = [labels.ids[labels.names.index(s)] for s in r["labels"]]
    rows = [r for r in rows if len(r["ids"]) <= a.max_tokens]
    names = sorted({r["name"] for r in rows})
    random.shuffle(names)
    n_dev = max(1, int(len(names) * a.dev_frac))
    dev_names = set(names[:n_dev])
    dev = [r for r in rows if r["name"] in dev_names]
    train = [r for r in rows if r["name"] not in dev_names]
    print(f"rows: train {len(train)} dev {len(dev)} (requests {len(names)}, dev {n_dev})", flush=True)

    if a.full:
        for p in model.parameters():
            p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none",
                         target_modules=lora_targets(model))
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        params = [p for p in model.parameters() if p.requires_grad]
    model.gradient_checkpointing_enable() if a.full else None
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    steps_total = math.ceil(len(train) / a.accum) * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * steps_total))) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))
    log = a.log.open("a", encoding="utf-8") if a.log else None

    def evaluate(subset: list[dict]) -> dict:
        model.eval()
        kls, agree = [], 0
        with torch.no_grad():
            for r in subset:
                kl = kl_row(model, r["ids"], r["label_ids"], r["p_teacher"], a.device, a.temperature)
                kls.append(kl.item())
                x = torch.tensor([r["ids"]], device=a.device)
                z = model(input_ids=x, logits_to_keep=1).logits[0, -1].float()[r["label_ids"]]
                agree += int(z.argmax().item() == max(range(len(r["p_teacher"])), key=r["p_teacher"].__getitem__))
        model.train()
        return {"kl": sum(kls) / max(1, len(kls)), "agree": agree / max(1, len(subset)), "n": len(subset)}

    t_start = time.perf_counter()
    print("dev before:", evaluate(dev), flush=True)
    model.train()
    step = 0
    for ep in range(a.epochs):
        random.shuffle(train)
        run, n_run = 0.0, 0
        for i, r in enumerate(train):
            loss = kl_row(model, r["ids"], r["label_ids"], r["p_teacher"], a.device, a.temperature) / a.accum
            loss.backward()
            run += loss.item() * a.accum
            n_run += 1
            if (i + 1) % a.accum == 0 or i + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    msg = f"ep {ep} step {step}/{steps_total} kl {run / n_run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.perf_counter() - t_start:.0f}s"
                    print(msg, flush=True)
                    if log:
                        log.write(msg + "\n")
                        log.flush()
                    run, n_run = 0.0, 0
        d = evaluate(dev)
        msg = f"epoch {ep} done: dev kl {d['kl']:.4f} agree {d['agree']:.3f} ({d['n']}) {time.perf_counter() - t_start:.0f}s"
        print(msg, flush=True)
        if log:
            log.write(msg + "\n")
            log.flush()
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.full:
        model.save_pretrained(a.out / "adapter")
        model = model.merge_and_unload()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    (a.out / "distill_config.json").write_text(json.dumps({
        "teacher": "cerno-4b-v1.1-q4_0.gguf via llama-server", "student_base": a.model, "temperature": a.temperature,
        "lora": None if a.full else {"rank": a.rank, "alpha": a.alpha}, "epochs": a.epochs, "lr": a.lr,
        "train_rows": len(train), "dev_rows": len(dev), "seconds": round(time.perf_counter() - t_start, 1),
        "labels": [str(p) for p in a.labels]}, indent=1), encoding="utf-8")
    print(f"saved {a.out} ({time.perf_counter() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
