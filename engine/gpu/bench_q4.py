"""GPU-alone benchmark of the Q4_0 x int8 QPU kernel on the Cerno 4B shapes (Pi side).

usage: bash tmp/gpu/gpurun.sh bench_q4.py [--shapes gate,down,qkv] [--n 7,64] [--iters 10]
                                          [--seconds S] [--pass 8]
--seconds S: instead of --iters, repeat each configuration for S seconds (for the CPU + GPU
co-run test). n > --pass vectors run as ceil(n / pass) passes of equal size, one dispatch
each, the weights re-read per pass (as the CPU's groups of 7). Prints one `RESULT {json}`
line per (shape, n): median seconds per call, GOP/s (2 R K n / t), bytes read per call,
bit-exact check against the fp32 model of the kernel on 4 sampled tiles (all n vectors).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import q4gpu
from videocore6.driver import Driver

SHAPES = {"gate": (9216, 2560), "down": (2560, 9216), "qkv": (8192, 2560), "c08gate": (3584, 1024),
          "tiny": (256, 256), "small": (1024, 1024)}


def pass_sizes(n: int, max_pass: int) -> list[int]:
    """n vectors in ceil(n / max_pass) passes of equal size (+-1), larger first (as the CPU)."""
    np_ = (n + max_pass - 1) // max_pass
    return [n // np_ + (p < n % np_) for p in range(np_)]


def run_shape(name: str, n: int, iters: int, seconds: float, until_file: str = "",
              tag: str = "gpu-alone", max_pass: int = 8, seed: int = 1) -> dict:
    R, K = SHAPES[name]
    T, B = R // 32, K // 32
    assert T % 8 == 0
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 16, (R, K), dtype=np.uint8)
    d = (rng.standard_normal((R, B)) * 0.02).astype(np.float16)
    x = rng.standard_normal((n, K)).astype(np.float32)
    q, dx = q4gpu.quant_act(x)
    sizes = pass_sizes(n, max_pass)
    kerns = {s: q4gpu.Kernel(s) for s in set(sizes)}
    codes, scales = q4gpu.layout_weights(u, d)
    unifs, v0s, v0 = [], [], 0
    for s in sizes:
        unifs.append(q4gpu.build_uniforms(kerns[s], q[v0:v0 + s], dx[v0:v0 + s]))
        v0s.append(v0)
        v0 += s
    size = codes.nbytes + scales.nbytes + sum(uu.nbytes for uu in unifs) + n * R * 4 + (1 << 20)
    with Driver(data_area_size=size + (4 << 20)) as drv:
        progs = {s: drv.program(k.program()) for s, k in kerns.items()}
        cbuf = drv.alloc(codes.size + 128 * 3, dtype=np.uint32)   # + padding blocks (prefetch)
        sbuf = drv.alloc(scales.size + 16 * 3, dtype=np.uint32)
        ubufs = [drv.alloc(uu.size, dtype=np.uint32) for uu in unifs]
        ybuf = drv.alloc((n, R), dtype=np.float32)
        pbufs = [drv.alloc(len(q4gpu.PARAMS), dtype=np.uint32) for _ in sizes]
        cbuf[:codes.size] = codes.reshape(-1)
        cbuf[codes.size:] = 0
        sbuf[:scales.size] = scales.reshape(-1)
        sbuf[scales.size:] = 0
        for ub, uu in zip(ubufs, unifs):
            ub[:] = uu
        ybuf[:] = np.nan
        nt = T // 8
        for pb, ub, v0 in zip(pbufs, ubufs, v0s):
            params = {"codes": cbuf.addresses()[0], "scales": sbuf.addresses()[0],
                      "y": ybuf.addresses()[v0, 0], "act": ub.addresses()[0], "nt": nt, "nb": B,
                      "tsc": nt * B * 512, "tss": nt * B * 64, "tsy": nt * 128, "r4": 4 * R,
                      "mask": q4gpu.MASK_PAIR, "k65536": 65536, "k512": 512, "k64": 64}
            pb[:] = [params[k] for k in q4gpu.PARAMS]
        jobs = [(progs[s], pb.addresses()[0]) for s, pb in zip(sizes, pbufs)]

        def call():
            for prog, pa in jobs:
                drv.execute(prog, local_invocation=(16, 1, 1), uniforms=pa, thread=8)

        unif = unifs[0]
        ubuf = ubufs[0]
        kern = kerns[sizes[0]]
        call()
        # CPU access speed of the (write-combined) BO mapping: read Y, write the uniform stream
        t0 = time.perf_counter()
        for _ in range(5):
            y = np.array(ybuf)
        y_read_gbs = 5 * y.nbytes / (time.perf_counter() - t0) / 1e9
        t0 = time.perf_counter()
        for _ in range(5):
            ubuf[:] = unif
        u_write_gbs = 5 * unif.nbytes / (time.perf_counter() - t0) / 1e9
        host = np.empty_like(y)
        t0 = time.perf_counter()
        for _ in range(5):
            host[:] = y
        host_copy_gbs = 5 * y.nbytes / (time.perf_counter() - t0) / 1e9
        # bit-exact check on 4 tiles (first, last, two random)
        tiles = sorted({0, T - 1, *rng.integers(0, T, 2).tolist()})
        rows = np.concatenate([np.arange(32 * t, 32 * t + 32) for t in tiles])
        ref = q4gpu.reference_gpu(u[rows], d[rows], q, dx)
        exact = bool(np.array_equal(y[:, rows].view(np.uint32), ref.view(np.uint32)))
        nbad = int((y[:, rows].view(np.uint32) != ref.view(np.uint32)).sum())
        finite = bool(np.isfinite(y).all())
        times = []
        t_end = time.perf_counter() + seconds
        if until_file:
            print("LOOP_START", flush=True)
        c0, w0 = time.process_time(), time.perf_counter()
        while (len(times) < iters) if seconds <= 0 else (time.perf_counter() < t_end):
            if until_file and os.path.exists(until_file):
                break
            t0 = time.perf_counter()
            call()
            times.append(time.perf_counter() - t0)
        host_cpu = (time.process_time() - c0) / max(1e-9, time.perf_counter() - w0)
        t = float(np.median(times))
        model_t = sum((T // 8) * (B * (kerns[s].stats()["instr"] + 4)) * 4 / 500e6 for s in sizes)
        wbytes = (codes.nbytes + scales.nbytes) * len(sizes)   # weights re-read per pass
        ubytes = sum(uu.nbytes for uu in unifs) * 8             # every QPU reads the stream
        res = {"shape": name, "rows": R, "cols": K, "n": n, "passes": sizes, "plan": q4gpu.PLAN,
               "calls": len(times),
               "sec": t, "gops": 2 * R * K * n / t / 1e9, "exact_vs_model": exact,
               "bad_words": nbad, "checked_words": int(ref.size), "all_finite": finite,
               "weight_bytes_per_call": wbytes, "uniform_bytes_per_call": ubytes,
               "read_gbs": (wbytes + ubytes) / t / 1e9,
               "instr_per_block": {s: kerns[s].stats()["instr"] for s in kerns},
               "model_sec_no_stall": model_t, "efficiency_vs_model": model_t / t,
               "t_min": float(np.min(times)), "t_max": float(np.max(times)),
               "t_p10": float(np.percentile(times, 10)), "t_p90": float(np.percentile(times, 90)),
               "host_cpu_frac": host_cpu, "tag": tag, "bo_read_gbs": y_read_gbs,
               "bo_write_gbs": u_write_gbs, "host_copy_gbs": host_copy_gbs}
        return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="gate,down,qkv")
    ap.add_argument("--n", default="7,8,11")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--until-file", default="", help="stop the timed loop when this file exists")
    ap.add_argument("--tag", default="gpu-alone")
    ap.add_argument("--pass", dest="max_pass", type=int, default=8,
                    help="max vectors per register pass (n > this runs several passes)")
    ap.add_argument("--plan", default=q4gpu.PLAN, choices=sorted(q4gpu.PLANS),
                    help="TMU request plan (max outstanding words: mid4 13, end 9, split 7)")
    a = ap.parse_args()
    q4gpu.PLAN = a.plan
    for name in a.shapes.split(","):
        for n in [int(v) for v in a.n.split(",")]:
            res = run_shape(name, n, a.iters, a.seconds, a.until_file, a.tag, a.max_pass)
            print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
