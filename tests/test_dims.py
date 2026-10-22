"""Engine dimensions test: synthetic models with other dimensions than Qwen3.5-0.8B.

    /root/venv-ref/bin/python tests/test_dims.py [--bin build/host-san/engine/pinfer_run]

The engine takes every size from the .pq header (engine/MODEL2B.md §2). This test writes small
bf16 models with random weights and dimension relations the real checkpoints do not exercise,
and runs each with plain greedy decoding and with MTP speculation (k = 1..3) on a 40-token
prompt (the MTP prefill then runs 40 rows at once). It checks:
  - exit code 0 and no AddressSanitizer / UBSan report (use the sanitizer build, the default);
  - every speculative run generates exactly the plain tokens (exactness, as tests/test_spec.py).
Cases:
  qwen-like  the 0.8B / 2B pattern at small scale (3 linear + 1 full layer, GQA 4:1, ffn = 3H);
  mtp-wide   2H larger than every projection width (conv_dim, 2*n_heads*head_dim, ffn): the
             MTP fc input rows [embedding | hidden] (2H floats each) must still fit the scratch
             buffer they are built in (engine/src/mtp.c uses ctx->ta);
  odd-heads  6 query heads per KV head (not a multiple of 4: attention pads the head quads),
             head_dim 48, lin_v_heads = 2 * lin_k_heads.
Only numpy is needed besides the engine binary; the files go to a temporary directory.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from convert import ENTRY_FMT, HEADER_FMT, MAGIC  # noqa: E402  (format strings, one source)

F32, BF16 = 0, 1
SAN_MARKERS = ("ERROR: AddressSanitizer", "runtime error:", "ERROR: LeakSanitizer",
               "SUMMARY: UndefinedBehaviorSanitizer")
CASES = {
    "qwen-like": dict(H=256, ffn=768, layers=4, vocab=1024, nh=8, nkv=2, hd=32, rot=8,
                      lkh=4, lvh=4, lkd=32, lvd=32),
    "mtp-wide": dict(H=512, ffn=64, layers=4, vocab=512, nh=1, nkv=1, hd=32, rot=8,
                     lkh=1, lvh=1, lkd=32, lvd=32),
    "odd-heads": dict(H=192, ffn=384, layers=4, vocab=768, nh=6, nkv=1, hd=48, rot=16,
                      lkh=2, lvh=4, lkd=32, lvd=64),
}


def tensors(d: dict, rng: np.random.Generator) -> list[tuple[str, int, list[int], bytes]]:
    """(name, dtype, shape, data) for every tensor model.c and mtp.c map."""
    H, ffn, hd, nh, nkv = d["H"], d["ffn"], d["hd"], d["nh"], d["nkv"]
    kd, vd = d["lkh"] * d["lkd"], d["lvh"] * d["lvd"]
    conv = 2 * kd + vd
    out = []

    def mat(name: str, rows: int, cols: int) -> None:
        w = (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)
        out.append((name, BF16, [rows, cols], (w.view(np.uint32) >> 16).astype("<u2").tobytes()))

    def vec(name: str, shape: list[int], lo: float, hi: float) -> None:
        out.append((name, F32, shape, rng.uniform(lo, hi, shape).astype("<f4").tobytes()))

    def full(p: str) -> None:
        mat(p + "self_attn.q_proj.weight", nh * hd * 2, H)
        mat(p + "self_attn.k_proj.weight", nkv * hd, H)
        mat(p + "self_attn.v_proj.weight", nkv * hd, H)
        mat(p + "self_attn.o_proj.weight", H, nh * hd)
        vec(p + "self_attn.q_norm.weight", [hd], 0.8, 1.2)
        vec(p + "self_attn.k_norm.weight", [hd], 0.8, 1.2)

    def block(p: str, is_full: bool) -> None:
        vec(p + "input_layernorm.weight", [H], 0.8, 1.2)
        vec(p + "post_attention_layernorm.weight", [H], 0.8, 1.2)
        mat(p + "mlp.gate_proj.weight", ffn, H)
        mat(p + "mlp.up_proj.weight", ffn, H)
        mat(p + "mlp.down_proj.weight", H, ffn)
        if is_full:
            full(p)
            return
        mat(p + "linear_attn.in_proj_qkv.weight", conv, H)
        mat(p + "linear_attn.in_proj_z.weight", vd, H)
        mat(p + "linear_attn.in_proj_b.weight", d["lvh"], H)
        mat(p + "linear_attn.in_proj_a.weight", d["lvh"], H)
        mat(p + "linear_attn.out_proj.weight", H, vd)
        vec(p + "linear_attn.A_log", [d["lvh"]], -1.0, 1.0)
        vec(p + "linear_attn.dt_bias", [d["lvh"]], -0.5, 0.5)
        vec(p + "linear_attn.norm.weight", [d["lvd"]], 0.8, 1.2)
        vec(p + "linear_attn.conv1d.weight", [conv, 4], -0.5, 0.5)

    mat("embed_tokens.weight", d["vocab"], H)
    vec("norm.weight", [H], 0.8, 1.2)
    for i in range(d["layers"]):
        block(f"layers.{i}.", i % 4 == 3)
    mat("mtp.fc.weight", H, 2 * H)
    for n in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        vec(f"mtp.{n}.weight", [H], 0.8, 1.2)
    block("mtp.layers.0.", True)
    return out


def write_pq(path: Path, d: dict, seed: int) -> None:
    ts = tensors(d, np.random.default_rng(seed))
    mask = sum(1 << i for i in range(d["layers"]) if i % 4 == 3)
    dir_off = 4096
    data_off = -(-(dir_off + 128 * len(ts)) // 4096) * 4096
    hdr = struct.pack(HEADER_FMT, MAGIC, 1, len(ts), dir_off, data_off, 1, d["H"], d["layers"],
                      d["vocab"], d["ffn"], d["nh"], d["nkv"], d["hd"], d["rot"], d["lkh"],
                      d["lvh"], d["lkd"], d["lvd"], 4, mask, 0, 1, 1, 1e-6, 1e7)
    with path.open("wb") as f:
        f.write(hdr.ljust(4096, b"\0"))
        off = data_off
        offs = []
        for name, dt, shape, data in ts:
            offs.append(off)
            dims = shape + [0] * (4 - len(shape))
            f.write(struct.pack(ENTRY_FMT, name.encode(), dt, len(shape), *dims, off, len(data)))
            off = -(-(off + len(data)) // 64) * 64
        for (_, _, _, data), o in zip(ts, offs):
            f.write(b"\0" * (o - f.tell()))
            f.write(data)


def run(binary: Path, model: Path, ids: list[int], extra: list[str]) -> tuple[int, str, str]:
    p = subprocess.run([str(binary), "-m", str(model), "-t", "2", "-n", "12",
                        "--ids", ",".join(map(str, ids))] + extra,
                       capture_output=True, text=True,
                       env={**os.environ, "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"})
    gen = next((ln for ln in p.stdout.splitlines() if ln.startswith("generated:")), "")
    return p.returncode, gen, p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=ROOT / "build" / "host-san" / "engine" / "pinfer_run")
    a = ap.parse_args()
    ok_all = True
    with tempfile.TemporaryDirectory() as td:
        for ci, (name, d) in enumerate(CASES.items()):
            model = Path(td) / f"{name}.pq"
            write_pq(model, d, seed=ci)
            ids = [int(x) for x in np.random.default_rng(100 + ci).integers(0, d["vocab"], 40)]
            rc, ref, err = run(a.bin, model, ids, [])
            ok = rc == 0 and bool(ref) and not any(m in err for m in SAN_MARKERS)
            line = [f"plain:{'ok' if ok else f'FAIL rc={rc}'}"]
            for k in (1, 2, 3):
                rk, gk, ek = run(a.bin, model, ids, ["--spec", str(k)])
                san = any(m in ek for m in SAN_MARKERS)
                ok_k = rk == 0 and not san and gk == ref
                ok &= ok_k
                line.append(f"k{k}:{'ok' if ok_k else ('SAN' if san else f'DIFF rc={rk}')}")
                if san or rk != 0:
                    print("\n".join(ek.splitlines()[:12]))
            ok_all &= ok
            print(f"{name:>10} H={d['H']} ffn={d['ffn']} heads={d['nh']}/{d['nkv']}x{d['hd']} "
                  f"lin={d['lkh']}/{d['lvh']}x{d['lkd']}/{d['lvd']}: {' '.join(line)}"
                  f"{'' if ok else '  <-- FAIL'}")
    print(f"\nOVERALL: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
