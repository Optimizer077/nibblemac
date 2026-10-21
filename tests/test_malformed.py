"""Malformed-input test: broken .pq files, state blobs and token files must fail cleanly.

    python3 tests/test_malformed.py [--bin PINFER_RUN] [--ppl PINFER_PPL] [--model BF16.pq]
                                    [--qmodel QUANT.pq] [--only NAME,...] [--list]

Default binaries: the sanitizer build (`make -C engine host SAN=address,undefined` ->
build/host-san/engine/pinfer_run, pinfer_ppl). Linux only and standard library only, so it
also runs on the Pi against a NEON build (--bin ./engine/pinfer_run_x --ppl ...).

Broken models are made from valid ones without copying weights: a .pq file is its header and
tensor directory (the first data_offset bytes, engine/FORMAT.md) followed by tensor data, and
the loader checks both against the file size before it reads any weight. Each case writes
the real header + directory, patched, into a sparse file of the real size (or a cut one).
Unpatched sparse copies are the positive controls: they load and run (on zero weights).

Every case runs the binary and checks
  - the exit code (1 for a refused input, 0 for the controls) and that no signal killed it,
  - the error code and the reason: the loader's message names the check that fired, so each
    case proves its own check and not a later one,
  - no AddressSanitizer / UBSan report.
The quantized cases use --qmodel (default models/pq/qwen35-0.8b-q4k-headq4k.pq; skipped if
absent). The host's scalar build refuses quantized dtypes, so these cases also show that the
size and tiling checks fire before that.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# pq_header (engine/src/pq_format.h): field -> (byte offset, struct format)
HDR = {"magic": (0, "8s"), "version": (8, "I"), "n_tensors": (12, "I"),
       "dir_offset": (16, "Q"), "data_offset": (24, "Q"), "arch": (32, "I"),
       "hidden": (36, "I"), "n_layers": (40, "I"), "vocab": (44, "I"), "ffn": (48, "I"),
       "n_heads": (52, "I"), "n_kv_heads": (56, "I"), "head_dim": (60, "I"),
       "rot_dim": (64, "I"), "lin_k_heads": (68, "I"), "lin_v_heads": (72, "I"),
       "lin_k_dim": (76, "I"), "lin_v_dim": (80, "I"), "conv_k": (84, "I"),
       "full_attn_mask": (88, "I"), "eos_id": (92, "I"), "tied_embeddings": (96, "I"),
       "has_mtp": (100, "I"), "rms_eps": (104, "f"), "rope_theta": (108, "f")}
# pq_tensor_entry (128 bytes): name[80] dtype ndim shape[4] offset nbytes reserved[8]
ENTRY = 128
ENT_INT = {"dtype": (80, "I"), "ndim": (84, "I"), "offset": (104, "Q"), "nbytes": (112, "Q")}
F32, BF16, Q8 = 0, 1, 2
E_FORMAT, E_CTX_FULL = -2, -5
SAN_MARKERS = ("ERROR: AddressSanitizer", "runtime error:", "ERROR: LeakSanitizer",
               "SUMMARY: UndefinedBehaviorSanitizer")


class Pq:
    """Header + tensor directory of a .pq file, editable, written out as a sparse file."""

    def __init__(self, path: Path, size: int, buf: bytearray) -> None:
        self.path, self.size, self.buf = path, size, buf

    @classmethod
    def open(cls, path: Path) -> Pq:
        with path.open("rb") as f:
            data_offset = struct.unpack_from("<Q", f.read(32), 24)[0]
            f.seek(0)
            return cls(path, path.stat().st_size, bytearray(f.read(data_offset)))

    def copy(self) -> Pq:
        return Pq(self.path, self.size, bytearray(self.buf))

    # header
    def hdr(self, k: str) -> int:
        off, fmt = HDR[k]
        return int(struct.unpack_from("<" + fmt, self.buf, off)[0])

    def set_hdr(self, k: str, v: float | bytes) -> Pq:
        off, fmt = HDR[k]
        struct.pack_into("<" + fmt, self.buf, off, v)
        return self

    # directory
    def _at(self, name: str) -> int:
        base = self.hdr("dir_offset")
        for i in range(self.hdr("n_tensors")):
            if self.name(base + i * ENTRY) == name:
                return base + i * ENTRY
        raise KeyError(name)

    def name(self, at: int) -> str:
        return bytes(self.buf[at:at + 80]).split(b"\0", 1)[0].decode()

    def names(self) -> list[str]:
        base = self.hdr("dir_offset")
        return [self.name(base + i * ENTRY) for i in range(self.hdr("n_tensors"))]

    def get(self, name: str, k: str) -> int:
        off, fmt = ENT_INT[k]
        return int(struct.unpack_from("<" + fmt, self.buf, self._at(name) + off)[0])

    def shape(self, name: str) -> list[int]:
        return list(struct.unpack_from("<4I", self.buf, self._at(name) + 88))

    def set(self, name: str, k: str, v: int) -> Pq:
        off, fmt = ENT_INT[k]
        struct.pack_into("<" + fmt, self.buf, self._at(name) + off, v)
        return self

    def set_shape(self, name: str, shape: list[int]) -> Pq:
        struct.pack_into("<4I", self.buf, self._at(name) + 88, *(shape + [0] * 4)[:4])
        return self

    def set_name(self, name: str, raw: bytes) -> Pq:
        at = self._at(name)
        self.buf[at:at + 80] = raw[:80].ljust(80, b"\0")
        return self

    def write(self, out: Path, size: int | None = None) -> Path:
        with out.open("wb") as f:
            f.write(self.buf)
            f.truncate(self.size if size is None else size)
        return out


def set_lin_v_dim(pq: Pq, dv: int) -> Pq:
    """A self-consistent file with lin_v_dim = dv: every linear-attention tensor that depends
    on it gets the matching shape and nbytes, so only pinfer_load's lin_v_dim limits can
    refuse it (the format checks all pass)."""
    kd = pq.hdr("lin_k_heads") * pq.hdr("lin_k_dim")
    vd = pq.hdr("lin_v_heads") * dv
    pq.set_hdr("lin_v_dim", dv)
    for name in pq.names():
        shape = pq.shape(name)
        if name.endswith(("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight")):
            shape[0] = 2 * kd + vd
        elif name.endswith("linear_attn.in_proj_z.weight"):
            shape[0] = vd
        elif name.endswith("linear_attn.out_proj.weight"):
            shape[1] = vd
        elif name.endswith("linear_attn.norm.weight"):
            shape[0] = dv
        else:
            continue
        n = 4 if pq.get(name, "dtype") == F32 else 2
        for d in shape[:pq.get(name, "ndim")]:
            n *= d
        pq.set_shape(name, shape).set(name, "nbytes", n)
    return pq


@dataclass
class Env:
    bin: str
    ppl: str
    model: Path
    env: dict[str, str]


@dataclass
class Case:
    name: str
    what: str
    make: Callable[[Path], list[str]]  # tmp dir -> argv after the binary
    expect_rc: int = 1
    expect: list[str] = field(default_factory=list)  # substrings of stdout + stderr
    tool: str = "run"


def model_cases(bf: Pq, q: Pq | None) -> list[Case]:
    cases: list[Case] = []
    fail = f"load failed: {E_FORMAT}"

    def add(name: str, what: str, patch: Callable[[Pq], Pq], expect: list[str], *,
            src: Pq | None = None, size: Callable[[Pq], int] | None = None,
            argv: list[str] | None = None, rc: int = 1) -> None:
        base = src if src is not None else bf

        def make(tmp: Path) -> list[str]:
            p = patch(base.copy())
            out = p.write(tmp / f"{name}.pq", size(p) if size else None)
            return ["-m", str(out)] + (argv if argv is not None else ["-n", "1", "--ids", "1,2,3"])
        cases.append(Case(name, what, make, rc, expect))

    def same(p: Pq) -> Pq:
        return p

    add("control-sparse-bf16", "unpatched header + directory, zero weights: loads and runs",
        same, ["generated:"], rc=0)
    # file
    add("truncated", "file cut 1 MiB short: the last tensors lie outside it", same,
        [fail, "data out of bounds"], size=lambda p: p.size - (1 << 20))
    add("truncated-header", "100-byte file", same, [fail, "too small"], size=lambda p: 100)
    add("bad-magic", "magic PINFERX", lambda p: p.set_hdr("magic", b"PINFERX\0"),
        [fail, "bad magic"])
    # header bounds and uint32 wraps
    add("hdr-huge-hidden", "hidden = 2^32 - 1", lambda p: p.set_hdr("hidden", 0xFFFFFFFF),
        [fail, "bad model dimensions"])
    add("hdr-layers-33", "n_layers = 33 (full_attn_mask has 32 bits)",
        lambda p: p.set_hdr("n_layers", 33), [fail, "bad model dimensions"])
    add("hdr-heads-wrap", "n_heads = 2^23 + 8: n_heads * head_dim * 2 wraps to 4096, the real "
        "q_proj rows, in uint32", lambda p: p.set_hdr("n_heads", (1 << 23) + 8),
        [fail, "bad attention dimensions"])
    add("hdr-lin-wrap", "lin_k_heads = lin_v_heads = 2^25 + 16: kd and vd wrap to the real 2048",
        lambda p: p.set_hdr("lin_k_heads", (1 << 25) + 16).set_hdr("lin_v_heads", (1 << 25) + 16),
        [fail, "bad linear-attention dimensions"])
    add("hdr-eos", "eos_id = vocab", lambda p: p.set_hdr("eos_id", p.hdr("vocab")),
        [fail, "eos_id"])
    add("hdr-theta-inf", "rope_theta = inf", lambda p: p.set_hdr("rope_theta", float("inf")),
        [fail, "bad eps / rope_theta"])
    add("hdr-eps-nan", "rms_eps = NaN", lambda p: p.set_hdr("rms_eps", float("nan")),
        [fail, "bad eps / rope_theta"])
    add("lin-v-dim-80", "consistent file with lin_v_dim 80: kern_lin.c needs a multiple of 32",
        lambda p: set_lin_v_dim(p, 80), [fail, "lin_v_dim 80"])
    add("lin-v-dim-544", "consistent file with lin_v_dim 544 (> the 512-float stack scratch)",
        lambda p: set_lin_v_dim(p, 544), [fail, "lin_v_dim 544"],
        size=lambda p: p.size + (64 << 20))
    # directory
    add("dir-misaligned", "dir_offset + 4: entries' uint64 fields misaligned",
        lambda p: p.set_hdr("dir_offset", p.hdr("dir_offset") + 4),
        [fail, "directory out of bounds or misaligned"])
    add("dir-n-tensors", "n_tensors = 2^32 - 1", lambda p: p.set_hdr("n_tensors", 0xFFFFFFFF),
        [fail, "directory out of bounds"])
    add("name-unterminated", "an 80-byte name without NUL",
        lambda p: p.set_name("norm.weight", b"x" * 80), [fail, "name not terminated"])
    add("ndim-5", "norm.weight ndim 5", lambda p: p.set("norm.weight", "ndim", 5),
        [fail, "bad ndim 5"])
    # sizes, shapes and dtypes of F32 / BF16 tensors
    add("bf16-nbytes", "embed_tokens.weight nbytes 2 short",
        lambda p: p.set("embed_tokens.weight", "nbytes", p.get("embed_tokens.weight", "nbytes") - 2),
        [fail, "embed_tokens.weight: nbytes"])
    add("shape-wrap", "mtp.norm.weight F32 [1024, 14586017, 308761441, 1]: 4 x the product = "
        "2^64 + 4096 wraps to its real nbytes; the MTP head is not mapped at load",
        lambda p: p.set("mtp.norm.weight", "ndim", 4)
        .set_shape("mtp.norm.weight", [1024, 14586017, 308761441, 1]),
        [fail, "mtp.norm.weight: shape larger than the file"])
    add("shape-huge", "norm.weight [2^32-1] x 4", lambda p: p.set("norm.weight", "ndim", 4)
        .set_shape("norm.weight", [0xFFFFFFFF] * 4), [fail, "norm.weight: shape larger"])
    add("shape-zero", "norm.weight [0]", lambda p: p.set_shape("norm.weight", [0]),
        [fail, "norm.weight: zero dimension"])
    add("dtype-unknown", "norm.weight dtype 99", lambda p: p.set("norm.weight", "dtype", 99),
        [fail, "unknown dtype 99"])
    up = "layers.0.mlp.up_proj.weight"
    add("dtype-f32-for-bf16", f"{up} says F32, has BF16 nbytes",
        lambda p: p.set(up, "dtype", F32), [fail, f"{up}: nbytes"])
    add("dtype-q8-for-bf16", f"{up} says Q8_G32, has BF16 nbytes",
        lambda p: p.set(up, "dtype", Q8), [fail, f"{up}: nbytes"])
    add("mtp-shape", "mtp.fc.weight [1024, 1024] with matching nbytes: loads; --spec refuses "
        "the head cleanly", lambda p: p.set_shape("mtp.fc.weight", [1024, 1024])
        .set("mtp.fc.weight", "nbytes", 1024 * 1024 * 2),
        [f"prefill failed: {E_FORMAT}", "mtp.fc.weight: expected shape"],
        argv=["-n", "2", "--ids", "1,2,3", "--spec", "1"])
    if q is None:
        return cases
    # quantized tensors (FORMAT.md "Quantized dtypes")
    gate = "layers.0.mlp.gate_proj.weight"
    sb = 576 if q.get(gate, "dtype") in (3, 4) else 1088
    native = os.uname().machine == "aarch64"
    add("control-sparse-quant", "unpatched quantized header + directory", same,
        ["generated:"] if native else [fail, "not supported by this build"], src=q,
        rc=0 if native else 1)
    add("q-nbytes", f"{gate} nbytes one superblock short",
        lambda p: p.set(gate, "nbytes", p.get(gate, "nbytes") - sb),
        [fail, f"{gate}: nbytes"], src=q)
    short = q.get(gate, "nbytes") - sb  # a multiple of 64, like every superblock size
    add("q-nbytes-at-eof", f"{gate} nbytes one superblock short and its offset moved so that "
        "it ends exactly at the end of the (page-aligned) file: in bounds, so only the size "
        "check stops the first matmul from reading the page past EOF (SIGBUS)",
        lambda p: p.set(gate, "nbytes", short).set(gate, "offset", p.size - short),
        [fail, f"{gate}: nbytes"], src=q)
    rows, cols = q.shape(gate)[:2]
    add("q-rows", f"{gate} rows {rows - 1} (not % 4)",
        lambda p: p.set_shape(gate, [rows - 1, cols]),
        [fail, f"{gate}: quantized dtype"], src=q)
    add("q-cols", f"{gate} cols 1000 (not % 256)", lambda p: p.set_shape(gate, [rows, 1000]),
        [fail, f"{gate}: quantized dtype"], src=q)
    add("q-ndim", f"{gate} ndim 1", lambda p: p.set(gate, "ndim", 1),
        [fail, f"{gate}: quantized dtype"], src=q)
    fc = "mtp.fc.weight"
    if q.get(fc, "dtype") >= Q8:
        fsb = 576 if q.get(fc, "dtype") in (3, 4) else 1088
        add("q-mtp-nbytes", f"{fc} (quantized) nbytes short: refused at load, although the "
            "MTP head is only mapped by --spec",
            lambda p: p.set(fc, "nbytes", p.get(fc, "nbytes") - fsb), [fail, f"{fc}: nbytes"],
            src=q)
    return cases


def state_cases(e: Env, bf: Pq) -> list[Case]:
    """State blobs saved with the real model, loaded into the real model (control), into a
    file with the same dimensions and layout but other weights, and patched."""
    real = str(e.model)

    def saved(tmp: Path) -> Path:
        st = tmp / "state.bin"
        if not st.exists():
            r = subprocess.run([e.bin, "-m", real, "-n", "1", "--ids", "1,2,3,4",
                                "--save-state", str(st)], capture_output=True, text=True,
                               env=e.env, check=False)
            if r.returncode != 0:
                raise RuntimeError(f"state save failed:\n{r.stdout}\n{r.stderr}")
        return st

    def patched(tmp: Path, name: str, fn: Callable[[bytearray], bytes]) -> str:
        out = tmp / f"{name}.bin"
        out.write_bytes(fn(bytearray(saved(tmp).read_bytes())))
        return str(out)

    def version1(b: bytearray) -> bytes:
        struct.pack_into("<I", b, 8, 1)
        return bytes(b)

    def flip_fp(b: bytearray) -> bytes:
        b[64] ^= 1  # state_hdr.model_fp (engine/src/state.c)
        return bytes(b)

    tail = ["-n", "1", "--ids", "5"]
    fail = f"failed ({E_FORMAT})"
    return [
        Case("state-control", "state of the real model, loaded into the real model",
             lambda t: ["-m", real, "--load-state", str(saved(t))] + tail, 0,
             ["state loaded", "generated:"]),
        Case("state-other-weights", "loaded into a file with the same header and directory "
             "but other (zero) weights: refused by the fingerprint",
             lambda t: ["-m", str(bf.write(t / "same-layout.pq")), "--load-state",
                        str(saved(t))] + tail, 1, ["load-state", fail]),
        Case("state-v1", "state format version 1 (before 0.2: dimensions only)",
             lambda t: ["-m", real, "--load-state", patched(t, "v1", version1)] + tail, 1,
             ["load-state", fail]),
        Case("state-fingerprint", "one bit of model_fp flipped",
             lambda t: ["-m", real, "--load-state", patched(t, "fp", flip_fp)] + tail, 1,
             ["load-state", fail]),
        Case("state-truncated", "blob 1 byte short",
             lambda t: ["-m", real, "--load-state",
                        patched(t, "short", lambda b: bytes(b[:-1]))] + tail, 1,
             ["load-state", fail]),
        Case("state-ctx-full", "4-token state into a context of 2 tokens (-c 2)",
             lambda t: ["-m", real, "-c", "2", "--load-state", str(saved(t))] + tail, 1,
             ["load-state", f"failed ({E_CTX_FULL})"]),
    ]


def ppl_cases(e: Env) -> list[Case]:
    real = str(e.model)

    def tokens(tmp: Path, name: str, ids: list[int]) -> str:
        out = tmp / f"{name}.bin"
        out.write_bytes(struct.pack(f"<{len(ids)}i", *ids))
        return str(out)

    def raw(tmp: Path, name: str, data: bytes) -> str:
        out = tmp / name
        out.write_bytes(data)
        return str(out)

    ok = [760, 1336, 3550, 13, 220, 19, 374, 264]
    # -c 256: positions 128..254 are scored in forward calls of 64; the target of position
    # 191 is token 192, the first token of the NEXT call, so no forward pass has checked it
    # when its logit is read
    bad = ok * 32
    bad[192] = 1 << 20
    return [
        Case("ppl-control", "8 valid tokens, -c 8 --chunks 1: a valid RESULT json line",
             lambda t: ["-m", real, "-f", tokens(t, "ok", ok), "-c", "8", "--chunks", "1"], 0,
             ['"ppl":'], tool="ppl"),
        Case("ppl-bad-target", "-c 256, token 192 = 2^20: a target read before any forward "
             "pass saw it", lambda t: ["-m", real, "-f", tokens(t, "bad", bad), "-c", "256",
                                       "--chunks", "1"], 1, ["outside the vocabulary"],
             tool="ppl"),
        Case("ppl-negative", "a negative token id",
             lambda t: ["-m", real, "-f", tokens(t, "neg", [-5] + ok[1:]), "-c", "8",
                        "--chunks", "1"], 1, ["outside the vocabulary"], tool="ppl"),
        Case("ppl-chunks0", "--chunks 0 (printed ppl NaN: invalid JSON)",
             lambda t: ["-m", real, "-f", tokens(t, "ok0", ok), "-c", "8", "--chunks", "0"], 2,
             ["--chunks >= 1"], tool="ppl"),
        Case("ppl-odd-size", "a 7-byte token file",
             lambda t: ["-m", real, "-f", raw(t, "odd.bin", b"\1" * 7), "-c", "4"], 1,
             ["cannot read"], tool="ppl"),
    ]


def run_case(e: Env, c: Case, tmp: Path) -> tuple[bool, str]:
    argv = [e.ppl if c.tool == "ppl" else e.bin] + c.make(tmp)
    r = subprocess.run(argv, capture_output=True, text=True, env=e.env, timeout=900,
                       check=False)
    out = r.stdout + r.stderr
    problems = []
    if r.returncode < 0:
        problems.append(f"killed by signal {-r.returncode}")
    elif r.returncode != c.expect_rc:
        problems.append(f"exit {r.returncode}, expected {c.expect_rc}")
    problems += [f"missing {s!r}" for s in c.expect if s not in out]
    problems += [f"sanitizer report ({m})" for m in SAN_MARKERS if m in out]
    if c.tool == "ppl" and c.expect_rc == 0:
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), "")
        try:
            json.loads(line[len("RESULT "):])
        except ValueError:
            problems.append("RESULT line is not valid JSON")
    if problems:
        report = next((ln.strip() for ln in out.splitlines()
                       if "runtime error:" in ln or "ERROR: AddressSanitizer" in ln), "")
        return False, "; ".join(problems) + (f"\n      {report[:240]}" if report else "")
    lines = [ln for ln in out.splitlines() if ln.startswith("pinfer:")] or out.splitlines()
    return True, (lines[0] if lines else "")[:100]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin", default=str(ROOT / "build" / "host-san" / "engine" / "pinfer_run"))
    p.add_argument("--ppl", default=str(ROOT / "build" / "host-san" / "engine" / "pinfer_ppl"))
    p.add_argument("--model", type=Path, default=ROOT / "models" / "pq" / "qwen35-0.8b-bf16.pq")
    p.add_argument("--qmodel", type=Path,
                   default=ROOT / "models" / "pq" / "qwen35-0.8b-q4k-headq4k.pq")
    p.add_argument("--only", default="", help="comma-separated case names")
    p.add_argument("--list", action="store_true")
    p.add_argument("--tmp", type=Path, default=None,
                   help="directory for the sparse files; not a tmpfs (the Pi's /tmp): mmap "
                        "reads of a hole allocate real pages there, 1.5 GB for a control run")
    a = p.parse_args()
    env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0", "UBSAN_OPTIONS": "print_stacktrace=1"}
    e = Env(a.bin, a.ppl, a.model, env)
    bf = Pq.open(a.model)
    q = Pq.open(a.qmodel) if a.qmodel.is_file() else None
    cases = model_cases(bf, q) + state_cases(e, bf) + ppl_cases(e)
    if a.list:
        for c in cases:
            print(f"{c.name:22} {c.what}")
        return 0
    only = {s for s in a.only.split(",") if s}
    todo = [c for c in cases if not only or c.name in only]
    print(f"{len(todo)} cases; bin {a.bin}; ppl {a.ppl}"
          + ("" if q else "; no quantized model: quantized cases skipped"))
    failed = []
    if a.tmp:
        a.tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pinfer-malformed-", dir=a.tmp) as td:
        for c in todo:
            ok, detail = run_case(e, c, Path(td))
            print(f"{'PASS' if ok else 'FAIL'}  {c.name:22} {detail}", flush=True)
            if not ok:
                failed.append(c.name)
            for f in Path(td).glob("*.pq"):   # one 1.5 GB (sparse) file at a time
                f.unlink()
    print(f"\n{len(todo) - len(failed)}/{len(todo)} passed"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    print(f"OVERALL: {'PASS' if not failed else 'FAIL'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
