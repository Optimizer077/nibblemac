#!/usr/bin/env python3
"""Per-operation time budgets from PF_KERN_STATS / PF_FWD_STATS profiles (harness result files).

Usage: python tools/audit/budget.py {4b|08b} <result.json> [--block N] [--tokens T] [--n N]

Splits the stdout of a profiled run into blocks (one per `pinfer_run` process: a block ends at its
`RESULT {...}` line), picks block N, and prints one row per operation class:
weights (MB) read per token, multiply-accumulates per token, measured ms per token, achieved
GOP/s (2 x MACs / time), the roof and the share of it.

Roofs (sources in docs/network-audit.md): DRAM 4.20 GB/s (memband, 1 thread); int8 instruction
peak 96 GOP/s (8 SMULL/SMLAL MACs per cycle x 4 cores x 1.5 GHz); per-shape kernel roof = the
best cold 4-thread kbench time of that shape at a 7-vector group (KROOF below, us -> GOP/s).
A KSTATS shape shared by two layer kinds (4B: q_proj and in_proj_qkv are both 8192 x 2560) is
split by call counts. `--n` keeps KSTATS rows of one batch width (1 = decode, 64 = prefill chunk).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DRAM_GBS, INT8_PEAK = 4.20, 96.0
BPW = {2: 8.5 / 8, 3: 4.5 / 8, 4: 4.5 / 8, 0: 4.0, 1: 2.0}     # bytes per weight by dtype id

# (rows, cols) -> [(op name, calls per token-pass)] for each network
SHAPES = {
    "4b": {
        (8192, 2560): [("in_proj_qkv (24 lin)", 24), ("q_proj + gate (8 full)", 8)],
        (4096, 2560): [("in_proj_z (24 lin)", 24)],
        (32, 2560): [("in_proj_a + b (24 lin)", 48)],
        (2560, 4096): [("out_proj (24 lin)", 24), ("o_proj (8 full)", 8)],
        (1024, 2560): [("k_proj + v_proj (8 full)", 16)],
        (9216, 2560): [("gate + up (32)", 64)],
        (2560, 9216): [("down (32)", 32)],
        (248320, 2560): [("head (Q8, full vocab)", 1)],
    },
    "08b": {
        (6144, 1024): [("in_proj_qkv (18 lin)", 18)],
        (2048, 1024): [("in_proj_z (18 lin)", 18)],
        (16, 1024): [("in_proj_a + b (18 lin)", 36)],
        (1024, 2048): [("out_proj (18 lin) + o_proj (6 full)", 24)],
        (4096, 1024): [("q_proj + gate (6 full)", 6)],
        (512, 1024): [("k_proj + v_proj (6 full)", 12)],
        (3584, 1024): [("gate + up (24)", 48)],
        (1024, 3584): [("down (24)", 24)],
        (248320, 1024): [("head (Q4K, full vocab)", 1)],
        (32768, 1024): [("MTP draft head (32768 rows)", 1)],
    },
}
# best measured cold 4-thread us per call at n = 7 (kbench) -> kernel roof GOP/s for that shape
KROOF = {
    "4b": {(8192, 2560): 5208.6, (4096, 2560): 2633.2, (2560, 4096): 2725.1,
           (9216, 2560): 6029.3, (2560, 9216): 6428.6, (1024, 2560): 759.4, (32, 2560): 58.1},
}


def blocks(stdout: str) -> list[list[str]]:
    out, cur = [], []
    for line in stdout.splitlines():
        cur.append(line)
        if line.startswith("RESULT {"):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("net", choices=["4b", "08b"])
    ap.add_argument("file", type=Path)
    ap.add_argument("--block", type=int, default=-1)
    ap.add_argument("--tokens", type=float, help="tokens the block processed (default: n_prompt)")
    ap.add_argument("--n", type=int, help="keep KSTATS rows of this batch width only")
    ap.add_argument("--stderr", action="store_true",
                    help="stats are in stderr (older runs): blocks start at each KSTATS header; the "
                         "RESULT of the same index is taken from stdout")
    a = ap.parse_args()
    d = json.loads(a.file.read_text(encoding="utf-8"))
    if a.stderr:
        sblk: list[list[str]] = []
        for line in (d.get("stderr") or "").splitlines():
            if re.match(r"KSTATS\s+rows", line):
                sblk.append([])
            if sblk:
                sblk[-1].append(line)
        rlines = [x for x in d["stdout"].splitlines() if x.startswith("RESULT {")]
        blk = sblk[a.block] + [rlines[a.block]]
    else:
        blk = [b for b in blocks(d["stdout"]) if any(x.startswith("KSTATS") for x in b)][a.block]
    res = json.loads(next(x for x in blk if x.startswith("RESULT {"))[7:])
    toks = a.tokens or res["n_prompt"]
    print(f"{a.file.name} block {a.block}: {res.get('model')} n_prompt {res['n_prompt']} "
          f"prefill {res.get('prefill_tps')} tok/s decode {res.get('decode_tps')} tok/s; per {toks} tokens")
    shapes = SHAPES[a.net]
    rows_out = []
    tot_ms = 0.0
    for line in blk:
        m = re.match(r"KSTATS\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)", line)
        if not m:
            continue
        r, c, dt, n, calls, ms = (int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), float(m[6]))
        if a.n is not None and n != a.n:
            continue
        ops = shapes.get((r, c)) or [(f"{r}x{c}", 1)]
        wsum = sum(k for _, k in ops)
        groups = -(-n // 7) if n > 7 else 1          # each group of <= 7 vectors re-streams W
        for name, k in ops:
            share = k / wsum
            ms_t = ms * share / toks
            macs = r * c * n * calls * share / toks    # per token
            mb = r * c * BPW.get(dt, 0.5625) * calls * share * groups / toks / 1e6
            gops = 2 * macs / (ms_t * 1e6) if ms_t else 0.0
            kr = KROOF.get(a.net, {}).get((r, c))
            roof = 2 * r * c * 7 / (kr * 1e3) if kr else None
            rows_out.append((name, n, mb, macs / 1e6, ms_t, gops, roof))
            tot_ms += ms_t
    print("MB = weight bytes streamed from DRAM per token (groups of <= 7 vectors each re-read W)")
    print(f"{'op':38} {'n':>3} {'MB':>7} {'MMAC':>7} {'ms/tok':>7} {'GOP/s':>6} {'GB/s':>5} "
          f"{'%DRAM':>5} {'kroof':>6} {'%kroof':>6} {'%int8':>6}")
    for name, n, mb, mmac, ms_t, gops, roof in rows_out:
        kr_s = f"{roof:6.1f}" if roof else "     -"
        pk_s = f"{gops / roof:6.0%}" if roof else "     -"
        gbs = mb / ms_t if ms_t else 0.0
        print(f"{name:38} {n:3d} {mb:7.2f} {mmac:7.1f} {ms_t:7.2f} {gops:6.1f} {gbs:5.2f} "
              f"{gbs / DRAM_GBS:5.0%} {kr_s} {pk_s} {gops / INT8_PEAK:6.0%}")
    want = None if a.n is None else ("n=1" if a.n == 1 else "batched")
    fs: defaultdict[tuple[str, str], float] = defaultdict(float)
    for line in blk:
        m = re.match(r"FSTATS\s+(.+?)\s{2,}(n=1|batched|logged)\s+([\d.]+)", line)
        if m and m[1] != "total" and (want is None or m[2] == want):
            fs[(m[1], m[2])] += float(m[3])
    other = 0.0
    for (ph, kind), ms in sorted(fs.items(), key=lambda kv: -kv[1]):
        print(f"{ph + ' [' + kind + ']':38} {'':3} {'':7} {'':7} {ms / toks:7.2f}")
        other += ms / toks
    wall = 1e3 / res["prefill_tps"] if res.get("prefill_tps") and a.n != 1 else None
    print(f"matmul {tot_ms:.2f} ms/token, other phases {other:.2f} ms/token"
          + (f", prefill wall {wall:.2f} ms/token" if wall else ""))


if __name__ == "__main__":
    main()
