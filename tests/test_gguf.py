"""GGUF import test: synthetic llama.cpp-style GGUFs through tools/gguf_to_pq.py.

    /root/venv-ref/bin/python tests/test_gguf.py [--bin build/host-san/engine/pinfer_run]

Two small Qwen3.5-shaped models (2 layers: one linear-attention layer with 12 V heads over 4 K
heads, one full-attention layer; every matrix 256 columns wide or more) are written as GGUF
files the way llama.cpp's converter writes them (conversion/qwen.py: V heads reordered from the
HF "grouped" to the "tiled" order, ssm_a = -exp(A_log), norms stored as 1 + w, conv squeezed):

  quantized  random Q4_0 / Q8_0 blocks. `--verify` must pass, and an independent check must hold:
             llama.cpp's own reorder (`_reorder_v_heads`, reimplemented here) applied to the
             converted .pq's dequantized weights gives back the GGUF's dequantized weights bit for
             bit. A second file whose `output.weight` equals `token_embd.weight` must convert
             to a tied model (no lm_head.weight).
  e2e        random F32 weights in HF layout, written once directly as a .pq (A_log, grouped V
             heads, untied lm_head) and once as a GGUF. Every tensor of the converted file must
             equal the reference .pq's bit for bit, except linear_attn.A = -exp(A_log); the host
             engine must then generate the same tokens from both files, with logits within
             1e-5 (relative) of each other.

numpy only, besides the engine binary; files go to a temporary directory.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gguf_to_pq as g2p  # noqa: E402
from convert import ENTRY_FMT, HEADER_FMT, MAGIC  # noqa: E402

# 12 V heads over 4 K heads: the grouped <-> tiled permutation is not its own inverse (with
# 2 heads per group over 2 groups it would be, and a reversed permutation would pass)
D = dict(H=256, ffn=512, vocab=512, nh=2, nkv=1, hd=128, rot=32, nk=4, nv=12, dk=32, dv=64,
         conv=4, eos=1)
GGML_F32, GGML_Q4_0, GGML_Q8_0 = 0, 2, 8
SAN_MARKERS = ("ERROR: AddressSanitizer", "runtime error:", "SUMMARY: UndefinedBehaviorSanitizer")


# ------------------------------------------------------------------ llama.cpp's transforms
def reorder_v_heads(x: np.ndarray, dim: int, nk: int, r: int, hd: int) -> np.ndarray:
    """conversion/qwen.py _reorder_v_heads: grouped [nk, r, hd] -> tiled [r, nk, hd] on `dim`."""
    shape = list(x.shape)
    new = shape[:dim] + [nk, r, hd] + shape[dim + 1:]
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return x.reshape(new).transpose(perm).reshape(shape)


def to_gguf_layout(name: str, w: np.ndarray) -> np.ndarray:
    """Apply llama.cpp's V-head reorder to one HF-layout tensor (pq names)."""
    nk, r, qk = D["nk"], D["nv"] // D["nk"], 2 * D["nk"] * D["dk"]
    if name.endswith(("in_proj_qkv.weight", "conv1d.weight")):
        return np.concatenate([w[:qk], reorder_v_heads(w[qk:], 0, nk, r, D["dv"])])
    if name.endswith("in_proj_z.weight"):
        return reorder_v_heads(w, 0, nk, r, D["dv"])
    if name.endswith(("in_proj_a.weight", "in_proj_b.weight", "linear_attn.A_log",
                      "linear_attn.dt_bias")):
        return reorder_v_heads(w.reshape(w.shape[0], -1), 0, nk, r, 1).reshape(w.shape)
    if name.endswith("out_proj.weight"):
        return reorder_v_heads(w, 1, nk, r, D["dv"])
    return w


GGUF_NAME = {
    "embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight",
    "norm.weight": "output_norm.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight", "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias", "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.norm.weight": "ssm_norm.weight",
}


def gguf_name(pq: str) -> str:
    if pq.startswith("layers."):
        _, i, rest = pq.split(".", 2)
        return f"blk.{i}.{GGUF_NAME[rest]}"
    return GGUF_NAME[pq]


# ------------------------------------------------------------------ model in HF layout
def hf_tensors(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """pq-named tensors (HF order, FORMAT.md conventions: norms already 1 + w), float32."""
    H, f, v = D["H"], D["ffn"], D["vocab"]
    kd, vd = D["nk"] * D["dk"], D["nv"] * D["dv"]
    m = lambda r, c, s=0.05: (rng.standard_normal((r, c)) * s).astype(np.float32)  # noqa: E731
    n = lambda k: (1 + 0.1 * rng.standard_normal(k)).astype(np.float32)  # noqa: E731
    t = {"embed_tokens.weight": m(v, H, 0.5), "lm_head.weight": m(v, H, 0.5), "norm.weight": n(H)}
    for i, full in ((0, False), (1, True)):
        L = f"layers.{i}."
        t |= {L + "input_layernorm.weight": n(H), L + "post_attention_layernorm.weight": n(H),
              L + "mlp.gate_proj.weight": m(f, H), L + "mlp.up_proj.weight": m(f, H),
              L + "mlp.down_proj.weight": m(H, f)}
        if full:
            hd = D["hd"]
            t |= {L + "self_attn.q_proj.weight": m(D["nh"] * hd * 2, H),
                  L + "self_attn.k_proj.weight": m(D["nkv"] * hd, H),
                  L + "self_attn.v_proj.weight": m(D["nkv"] * hd, H),
                  L + "self_attn.o_proj.weight": m(H, D["nh"] * hd),
                  L + "self_attn.q_norm.weight": n(hd), L + "self_attn.k_norm.weight": n(hd)}
        else:
            t |= {L + "linear_attn.in_proj_qkv.weight": m(2 * kd + vd, H),
                  L + "linear_attn.in_proj_z.weight": m(vd, H),
                  L + "linear_attn.in_proj_a.weight": m(D["nv"], H),
                  L + "linear_attn.in_proj_b.weight": m(D["nv"], H),
                  L + "linear_attn.out_proj.weight": m(H, vd),
                  L + "linear_attn.A_log": (rng.standard_normal(D["nv"]) * 0.5).astype(np.float32),
                  L + "linear_attn.dt_bias":
                      (rng.standard_normal(D["nv"]) * 0.5).astype(np.float32),
                  L + "linear_attn.conv1d.weight": m(2 * kd + vd, D["conv"], 0.3),
                  L + "linear_attn.norm.weight": n(D["dv"])}
    return t


def header_fields(n_tensors: int, dir_off: int, data_off: int) -> tuple:
    return (MAGIC, 1, n_tensors, dir_off, data_off, 1, D["H"], 2, D["vocab"], D["ffn"], D["nh"],
            D["nkv"], D["hd"], D["rot"], D["nk"], D["nv"], D["dk"], D["dv"], D["conv"], 0b10,
            D["eos"], 0, 0, 1e-6, 1e7)


def write_pq(path: Path, t: dict[str, np.ndarray]) -> None:
    """Reference .pq straight from the HF-layout tensors (all F32, untied head)."""
    names = sorted(t)
    data_off = -(-(4096 + 128 * len(names)) // 4096) * 4096
    entries, off = [], data_off
    for nm in names:
        entries.append((nm, list(t[nm].shape), off, t[nm].nbytes))
        off = -(-(off + t[nm].nbytes) // 64) * 64
    with path.open("wb") as f:
        head = struct.pack(HEADER_FMT, *header_fields(len(names), 4096, data_off))
        f.write(head.ljust(4096, b"\0"))
        for nm, shape, o, nb in entries:
            dims = shape + [0] * (4 - len(shape))
            f.write(struct.pack(ENTRY_FMT, nm.encode(), 0, len(shape), *dims, o, nb))
        for nm, _, o, _ in entries:
            f.write(b"\0" * (o - f.tell()))
            f.write(t[nm].tobytes())


# ------------------------------------------------------------------ GGUF writer
def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, tensors: list[tuple[str, int, list[int], bytes]]) -> None:
    """tensors: (gguf name, ggml type, shape in PyTorch order, raw bytes)."""
    kv = [("general.architecture", 8, "qwen35"), ("qwen35.block_count", 4, 2),
          ("qwen35.embedding_length", 4, D["H"]), ("qwen35.feed_forward_length", 4, D["ffn"]),
          ("qwen35.attention.head_count", 4, D["nh"]),
          ("qwen35.attention.head_count_kv", 4, D["nkv"]),
          ("qwen35.attention.key_length", 4, D["hd"]),
          ("qwen35.attention.value_length", 4, D["hd"]),
          ("qwen35.rope.dimension_count", 4, D["rot"]), ("qwen35.rope.freq_base", 6, 1e7),
          ("qwen35.attention.layer_norm_rms_epsilon", 6, 1e-6),
          ("qwen35.ssm.conv_kernel", 4, D["conv"]), ("qwen35.ssm.state_size", 4, D["dk"]),
          ("qwen35.ssm.group_count", 4, D["nk"]), ("qwen35.ssm.time_step_rank", 4, D["nv"]),
          ("qwen35.ssm.inner_size", 4, D["nv"] * D["dv"]),
          ("tokenizer.ggml.eos_token_id", 4, D["eos"])]
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv) + 1))
    for k, typ, val in kv:
        out += _s(k) + struct.pack("<I", typ)
        out += _s(val) if typ == 8 else struct.pack("<I" if typ == 4 else "<f", val)
    out += _s("qwen35.attention.recurrent_layers") + struct.pack("<IIQ", 9, 7, 2) + bytes([1, 0])
    offs, off = [], 0
    for _, _, _, raw in tensors:
        offs.append(off)
        off = -(-(off + len(raw)) // 32) * 32
    for (name, typ, shape, _), o in zip(tensors, offs):
        dims = list(reversed(shape))
        out += _s(name) + struct.pack("<I", len(dims))
        out += b"".join(struct.pack("<Q", x) for x in dims)
        out += struct.pack("<IQ", typ, o)
    out += b"\0" * (-len(out) % 32)
    base = len(out)
    for (_, _, _, raw), o in zip(tensors, offs):
        out += b"\0" * (base + o - len(out)) + raw
    path.write_bytes(bytes(out))


def gguf_from_hf(t: dict[str, np.ndarray], quant: dict[str, bytes] | None = None
                 ) -> list[tuple[str, int, list[int], bytes]]:
    """llama.cpp's transforms of the HF-layout tensors; `quant` overrides a tensor's raw data
    (already in GGUF layout) with a quantized type chosen by name."""
    out = []
    for nm in sorted(t):
        w = t[nm]
        if nm.endswith("linear_attn.A_log"):
            w = -np.exp(w)
        w = to_gguf_layout(nm, w)
        gn = gguf_name(nm)
        if quant and gn in quant:
            typ, raw = quant[gn]
            out.append((gn, typ, list(w.shape), raw))
        else:
            out.append((gn, GGML_F32, list(w.shape), np.ascontiguousarray(w, np.float32).tobytes()))
    return out


def random_blocks(rng: np.random.Generator, rows: int, cols: int, typ: int) -> bytes:
    nb = cols // 32
    d = (rng.uniform(0.002, 0.02, (rows, nb)) * rng.choice([-1, 1], (rows, nb))).astype(np.float16)
    if typ == GGML_Q4_0:
        qs = rng.integers(0, 256, (rows, nb, 16), dtype=np.uint8)
    else:
        qs = rng.integers(-127, 128, (rows, nb, 32), dtype=np.int16).astype(np.int8).view(np.uint8)
    return np.concatenate([d.view(np.uint8).reshape(rows, nb, 2), qs], axis=2).tobytes()


# ------------------------------------------------------------------ cases
def case_quantized(tmp: Path, rng: np.random.Generator) -> list[str]:
    errs = []
    t = hf_tensors(rng)
    quant = {}
    for nm, w in t.items():
        if w.ndim == 2 and not nm.endswith("conv1d.weight"):
            typ = GGML_Q8_0 if nm in ("embed_tokens.weight", "lm_head.weight") else GGML_Q4_0
            quant[gguf_name(nm)] = (typ, random_blocks(rng, w.shape[0], w.shape[1], typ))
    for tied in (False, True):
        if tied:
            quant["output.weight"] = quant["token_embd.weight"]
        gg, pq = tmp / f"q{int(tied)}.gguf", tmp / f"q{int(tied)}.pq"
        write_gguf(gg, gguf_from_hf(t, quant))
        info = g2p.convert(gg, pq)
        stats = g2p.verify(gg, pq)
        if info["tied"] != tied:
            errs.append(f"quantized: tied={info['tied']}, expected {tied}")
        mm, h, ents = g2p.read_pq(pq)
        if ("lm_head.weight" in ents) == tied or h["tied_embeddings"] != int(tied):
            errs.append("quantized: lm_head.weight / tied_embeddings wrong")
        # independent check: llama.cpp's reorder of the converted weights gives the GGUF back
        gf = g2p.Gguf(gg)
        for nm in t:
            gn = gguf_name(nm)
            if gn not in quant or nm not in ents:   # lm_head.weight is absent when tied
                continue
            dtype, shape, off, nbytes = ents[nm]
            pd, pc = g2p.unpack_pq(mm[off:off + nbytes], dtype, *shape)
            gd, gc = g2p.unpack_ggml(gf.blocks(gn), gf.tensors[gn].ggml_type)
            w_pq, w_gg = g2p.dequant(pd, pc, dtype), g2p.dequant(gd, gc, dtype)
            if not np.array_equal(to_gguf_layout(nm, w_pq).view(np.uint32), w_gg.view(np.uint32)):
                errs.append(f"quantized: {nm} differs from the GGUF after llama.cpp's reorder")
        print(f"  quantized tied={tied}: {stats['tensors']} tensors, {stats['quantized_tensors']} "
              f"quantized, {stats['permuted_tensors']} permuted -> {'ok' if not errs else 'FAIL'}")
    return errs


def run(binary: Path, model: Path, ids: str, dump: Path) -> tuple[str, np.ndarray, str]:
    p = subprocess.run([str(binary), "-m", str(model), "-t", "1", "-n", "8", "--ids", ids,
                        "--dump-logits", str(dump)], capture_output=True, text=True, timeout=300,
                       check=False)
    gen = next((ln for ln in p.stdout.splitlines() if ln.startswith("generated:")), "")
    logits = np.fromfile(dump, dtype=np.float32) if dump.exists() else np.zeros(0, np.float32)
    return gen, logits, p.stderr + (f"\nexit {p.returncode}" if p.returncode else "")


def case_e2e(tmp: Path, rng: np.random.Generator, binary: Path) -> list[str]:
    errs = []
    t = hf_tensors(rng)
    ref, gg, conv = tmp / "ref.pq", tmp / "e2e.gguf", tmp / "e2e.pq"
    write_pq(ref, t)
    write_gguf(gg, gguf_from_hf(t))
    g2p.convert(gg, conv)
    g2p.verify(gg, conv)
    mr, _, er = g2p.read_pq(ref)
    mc, _, ec = g2p.read_pq(conv)
    for nm, (dt, shape, off, nb) in er.items():
        if nm.endswith("linear_attn.A_log"):
            a = ec[nm[:-len("A_log")] + "A"]
            got = mc[a[2]:a[2] + a[3]].view("<f4")
            want = -np.exp(mr[off:off + nb].view("<f4"))
            if not np.array_equal(got, want):
                errs.append(f"e2e: {nm}: A != -exp(A_log)")
            continue
        c = ec.get(nm)
        same = c is not None and c[:2] == (dt, shape) and \
            np.array_equal(mc[c[2]:c[2] + c[3]], mr[off:off + nb])
        if not same:
            errs.append(f"e2e: {nm} differs from the reference .pq")
    ids = ",".join(str(x) for x in rng.integers(0, D["vocab"], 12))
    g_ref, l_ref, e_ref = run(binary, ref, ids, tmp / "l_ref.bin")
    g_cv, l_cv, e_cv = run(binary, conv, ids, tmp / "l_cv.bin")
    for e in (e_ref, e_cv):
        if any(s in e for s in SAN_MARKERS) or "exit" in e:
            errs.append(f"e2e: engine error: {e.strip()[-300:]}")
    rel = (np.abs(l_ref - l_cv).max() / max(np.abs(l_ref).max(), 1e-30)) if l_ref.size and \
        l_ref.size == l_cv.size else float("inf")
    if not g_ref or g_ref != g_cv:
        errs.append(f"e2e: generated tokens differ ({g_ref!r} vs {g_cv!r})")
    if rel > 1e-5:
        errs.append(f"e2e: logits rel. max diff {rel:.3g} > 1e-5")
    print(f"  e2e: tensors equal except A, {g_ref} (both), logits rel. max diff {rel:.3g} "
          f"-> {'ok' if not errs else 'FAIL'}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path,
                    default=ROOT / "build" / "host-san" / "engine" / "pinfer_run")
    a = ap.parse_args()
    if not a.bin.exists():
        a.bin = ROOT / "build" / "host" / "engine" / "pinfer_run"
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        errs = case_quantized(tmp, rng) + case_e2e(tmp, rng, a.bin)
    for e in errs:
        print("  " + e)
    print("OVERALL:", "PASS" if not errs else "FAIL")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
