"""Layer-skipping / early-exit pilot on Cerno 4B (engine/PILOTS.md "Layer skipping").

    run.sh skip_study.py check                  # cache / mask / position plumbing vs the plain HF path
    run.sh skip_study.py base                   # full model + early exits k = 24..31
    run.sh skip_study.py single [--modes skip,nomlp,nomix] [--scope all,q]
    run.sh skip_study.py greedy --single FILE [--strategy ranked|greedy] [--units skip|parts] [--scope all|q]
    run.sh skip_study.py table FILE...          # markdown table from result files

Bar (docs/network-audit.md): max |dp| <= 0.0146 and the same top answer on >= 51 of 52 rows, against
cerno-pi/results/ref_llama_pc.jsonl. Results: bench/results/layerskip/<cmd>_<stamp>.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch

import cerno_gpu as cg
from cerno_gpu import Cerno, Evaluator, compare, load_rows, passes, saved_fraction, save_json

logger = logging.getLogger("skip_study")
BAR = {"max_dp": 0.0146, "min_top": 51}


def setup(args) -> tuple[Cerno, Evaluator, list[dict]]:
    c = Cerno(args.device)
    ev = Evaluator(c, load_rows())
    t0 = time.time()
    base = ev.evaluate({}, exits=list(range(24, 32)))
    logger.info("base %s (%.1f s)", compare(base), time.time() - t0)
    return c, ev, base


def variant_record(name: str, modes: dict[int, str], scope: str, res: list[dict],
                   base: list[dict], seconds: float) -> dict:
    cmp = compare(res, base)
    frac = saved_fraction(modes)
    return {"variant": name, "modes": {str(k): v for k, v in sorted(modes.items())}, "scope": scope,
            "saved_frac": frac, "saved_frac_decision": frac * (0.61 if scope == "q" else 1.0),
            "pass": passes(cmp, **BAR), "seconds": seconds, **cmp, "rows": res}


def cmd_check(args) -> None:
    c = Cerno(args.device)
    rows = load_rows()
    ev = Evaluator(c, rows)
    ri = args.request
    rs = ev.by_req[ri]
    n = ev.prefix_len[ri]
    r = rs[-1]
    t0 = time.time()
    h_full, _ = c.forward(r["ids"])
    t1 = time.time()
    out = c.lm(input_ids=torch.tensor([r["ids"]], device=c.device), use_cache=True)
    h_hf = out.last_hidden_state[0, -1]
    _, cache = c.forward(r["ids"][:n])
    h_pre, _ = c.forward(r["ids"][n:], cache, past=n)
    lg = [c.logits(h) for h in (h_full, h_hf, h_pre)]
    ps = [cg.readout(l, r["labels"]) for l in lg]
    print(f"request {r['request']} row {r['question']}: {len(r['ids'])} tokens, prefix {n}; "
          f"forward {t1 - t0:.2f} s; attn impl {c.config._attn_implementation}")
    print(f"  |h explicit - h HF default| max {float((h_full - h_hf).abs().max()):.3e}")
    print(f"  |h prefix-cache - h full|  max {float((h_full - h_pre).abs().max()):.3e}")
    print(f"  label logits full {[round(float(x), 4) for x in lg[0][r['labels']]]}")
    print(f"  p full {ps[0]}\n  p hf   {ps[1]}\n  p pre  {ps[2]}\n  ref    {r['ref']}")
    t0 = time.time()
    base = ev.evaluate({}, exits=[31])
    dt = time.time() - t0
    cmp = compare(base)
    cmp_exit = compare([{"p": x["exit"][31], "ref": x["ref"], "question": x["question"]} for x in base])
    print(f"52 rows in {dt:.1f} s: {cmp}\n  exit-31 (= full, via hook): {cmp_exit}")
    if c.device.type == "cuda":
        print(f"  GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def run_base(ev: Evaluator, base: list[dict]) -> Path:
    out = {"bar": BAR, "base": compare(base), "exits": {}, "rows": base}
    for k in range(24, 32):
        res = [{"p": x["exit"][k], "ref": x["ref"], "question": x["question"]} for x in base]
        modes = {l: "skip" for l in range(k + 1, 32)}
        cmp = compare(res, base)
        out["exits"][k] = {"saved_frac": saved_fraction(modes), "pass": passes(cmp, **BAR), **cmp}
        logger.info("exit after layer %d: saved %.3f %s", k, saved_fraction(modes), cmp)
    return save_json("base", out)


def cmd_base(args) -> None:
    c, ev, base = setup(args)
    run_base(ev, base)


def run_single(ev: Evaluator, base: list[dict], modes: list[str], scopes: list[str],
               layers: list[int] | None = None) -> dict:
    out = {"bar": BAR, "base": compare(base), "variants": []}
    path = save_json("single", out)                  # rewritten after every variant
    for scope in scopes:
        for mode in modes:
            for layer in layers or range(32):
                t0 = time.time()
                res = ev.evaluate({layer: mode}, question_only=(scope == "q"))
                rec = variant_record(f"L{layer}:{mode}", {layer: mode}, scope, res, base,
                                     time.time() - t0)
                del rec["rows"]
                out["variants"].append(rec)
                logger.info("%s %s saved %.3f max_dp %.4f mean %.4f top %d (vs base %.4f / %d) %.1f s",
                            scope, rec["variant"], rec["saved_frac"], rec["max_dp_ref"],
                            rec["mean_dp_ref"], rec["same_top_ref"], rec["max_dp_base"],
                            rec["same_top_base"], rec["seconds"])
                path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def load_single(path: str) -> dict:
    """A single-unit result file, or the run log of one (lines `<scope> L<l>:<mode> saved ... s`)."""
    if path.endswith(".json"):
        return json.loads(Path(path).read_text())
    pat = re.compile(r"(all|q) L(\d+):(skip|nomlp|nomix) saved ([\d.]+) max_dp ([\d.]+) mean ([\d.]+) "
                     r"top (\d+) \(vs base ([\d.]+) / (\d+)\) ([\d.]+) s")
    variants = []
    for m in pat.finditer(Path(path).read_text(encoding="utf-8", errors="replace")):
        scope, layer, mode = m.group(1), int(m.group(2)), m.group(3)
        rec = {"variant": f"L{layer}:{mode}", "modes": {str(layer): mode}, "scope": scope,
               "saved_frac": float(m.group(4)), "max_dp_ref": float(m.group(5)),
               "mean_dp_ref": float(m.group(6)), "same_top_ref": int(m.group(7)),
               "max_dp_base": float(m.group(8)), "same_top_base": int(m.group(9)),
               "seconds": float(m.group(10))}
        rec["pass"] = rec["max_dp_ref"] <= BAR["max_dp"] and rec["same_top_ref"] >= BAR["min_top"]
        variants.append(rec)
    out = {"bar": BAR, "source_log": path, "variants": variants}
    save_json("single_fromlog", out)
    return out


def cmd_single(args) -> None:
    c, ev, base = setup(args)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    run_single(ev, base, args.modes.split(","), args.scope.split(","), layers)


def run_greedy(ev: Evaluator, base: list[dict], single: dict, scope: str, units: str,
               strategy: str, shortlist: int, extra: int) -> dict:
    """Accumulate units (whole layers, or MLP / mixer parts) from the least harmful single result."""
    cand = [v for v in single["variants"] if v["scope"] == scope
            and (v["modes"][next(iter(v["modes"]))] == "skip") == (units == "skip")]
    cand.sort(key=lambda v: (v["max_dp_ref"], v["mean_dp_ref"]))
    pool = [(int(k), m) for v in cand for k, m in v["modes"].items()]
    chosen: dict[int, str] = {}
    out = {"bar": BAR, "base": compare(base), "strategy": strategy, "units": units,
           "scope": scope, "steps": []}
    fails = 0
    while pool and fails <= extra:
        cands = [u for u in pool if u[0] not in chosen]
        cands = cands[:1] if strategy == "ranked" else cands[:shortlist]
        if not cands:
            break
        best = None
        for layer, mode in cands:
            trial = dict(chosen)
            trial[layer] = mode
            t0 = time.time()
            res = ev.evaluate(trial, question_only=(scope == "q"))
            rec = variant_record(f"+L{layer}:{mode}", trial, scope, res, base, time.time() - t0)
            logger.info("  try %s -> max_dp %.4f mean %.4f top %d", rec["variant"],
                        rec["max_dp_ref"], rec["mean_dp_ref"], rec["same_top_ref"])
            if best is None or (rec["max_dp_ref"], rec["mean_dp_ref"]) < (best["max_dp_ref"], best["mean_dp_ref"]):
                best = rec
        chosen = {int(k): v for k, v in best["modes"].items()}
        pool = [u for u in pool if u[0] not in chosen]
        rows = best.pop("rows")
        best["worst_rows"] = sorted(((max(abs(a - b) for a, b in zip(r["p"], r["ref"])), r["question"])
                                     for r in rows), reverse=True)[:3]
        out["steps"].append(best)
        fails += 0 if best["pass"] else 1
        logger.info("step %d: %s saved %.3f max_dp %.4f mean %.4f top %d pass %s", len(out["steps"]),
                    sorted(chosen.items()), best["saved_frac"], best["max_dp_ref"], best["mean_dp_ref"],
                    best["same_top_ref"], best["pass"])
    save_json(f"greedy_{strategy}_{units}_{scope}", out)
    return out


def cmd_greedy(args) -> None:
    single = load_single(args.single)
    c, ev, base = setup(args)
    run_greedy(ev, base, single, args.scope, args.units, args.strategy, args.shortlist, args.extra)


def cmd_all(args) -> None:
    """base + early exits, every single unit, then the accumulations, with one model load."""
    c, ev, base = setup(args)
    run_base(ev, base)
    single = run_single(ev, base, ["skip", "nomlp", "nomix"], ["all", "q"])
    for scope in ("all", "q"):
        for units in ("skip", "parts"):
            run_greedy(ev, base, single, scope, units, args.strategy, args.shortlist, args.extra)


def cmd_table(args) -> None:
    print("| variant | scope | layers / parts removed | matmul MACs saved | max dp | mean dp | same top | pass |")
    print("|---|---|---|---|---|---|---|---|")
    for f in args.files:
        d = json.loads(Path(f).read_text())
        recs = d.get("variants") or d.get("steps") or []
        if "exits" in d:
            recs = [{"variant": f"exit after layer {k}", "scope": "all", "modes": {f"{k + 1}..31": "skip"}, **v}
                    for k, v in d["exits"].items()]
        for v in recs:
            mods = ", ".join(f"{k}:{m}" for k, m in v["modes"].items())
            print(f"| {v['variant']} | {v['scope']} | {mods} | {100 * v['saved_frac']:.1f}% | "
                  f"{v['max_dp_ref']:.4f} | {v['mean_dp_ref']:.4f} | {v['same_top_ref']}/52 | "
                  f"{'yes' if v['pass'] else 'no'} |")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--request", type=int, default=0)
    sub.add_parser("base")
    p = sub.add_parser("single")
    p.add_argument("--modes", default="skip,nomlp,nomix")
    p.add_argument("--scope", default="all,q")
    p.add_argument("--layers", default="", help="comma-separated layer indices (default: all 32)")
    p = sub.add_parser("greedy")
    p.add_argument("--single", required=True)
    p.add_argument("--strategy", default="greedy", choices=["ranked", "greedy"])
    p.add_argument("--units", default="skip", choices=["skip", "parts"])
    p.add_argument("--scope", default="all", choices=["all", "q"])
    p.add_argument("--shortlist", type=int, default=6)
    p.add_argument("--extra", type=int, default=1, help="steps to continue after the bar breaks")
    p = sub.add_parser("all")
    p.add_argument("--strategy", default="greedy", choices=["ranked", "greedy"])
    p.add_argument("--shortlist", type=int, default=6)
    p.add_argument("--extra", type=int, default=1)
    p = sub.add_parser("table")
    p.add_argument("files", nargs="+")
    args = ap.parse_args()
    {"check": cmd_check, "base": cmd_base, "single": cmd_single, "greedy": cmd_greedy,
     "all": cmd_all, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
