"""Shared raw inputs and Python GPU reference for v3d_q4_test (test tool only).

make FIXTURE: PC-side deterministic gate inputs. python FIXTURE PREFIX: Pi-side
reference via the separately installed py-videocore6 driver (never copied here).
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

MAGIC = 0x34465051


def make(path: Path) -> None:
    rng = np.random.default_rng(1)
    r, k, n = 9216, 2560, 64
    u = rng.integers(0, 16, (r, k), dtype=np.uint8)
    d = (rng.standard_normal((r, k // 32)) * .02).astype('<f2')
    x = rng.standard_normal((n, k)).astype('<f4')
    # No assembler needed to prepare the inputs.
    xb = x.reshape(n, k // 32, 32)
    amax = np.abs(xb).max(axis=-1)
    q = np.rint(xb * (np.float32(127) / amax)[..., None]).astype(np.int8)
    dx = (amax / np.float32(127)).astype('<f4')
    with path.open('wb') as f:
        f.write(struct.pack('<8I', MAGIC, 1, r, k, n, 0, 0, 0))
        for a in (u, d, x, q, dx):
            f.write(a.tobytes())
    print(json.dumps({'fixture': str(path), 'bytes': path.stat().st_size, 'rows': r, 'cols': k, 'n': n}))


def reference(path: Path, prefix: Path, iters: int) -> None:
    import q4gpu
    from videocore6.driver import Driver  # external reference only, no implementation copied

    with path.open('rb') as f:
        magic, version, r, k, nmax, *reserved = struct.unpack('<8I', f.read(32))
        assert magic == MAGIC and version == 1 and not any(reserved)
        assert (r, k, nmax) == (9216, 2560, 64)
        u = np.fromfile(f, 'u1', r * k).reshape(r, k)
        d = np.fromfile(f, '<f2', r * k // 32).reshape(r, k // 32)
        x = np.fromfile(f, '<f4', nmax * k).reshape(nmax, k)
        q = np.fromfile(f, 'i1', nmax * k).reshape(nmax, k)
        dx = np.fromfile(f, '<f4', nmax * k // 32).reshape(nmax, k // 32)
        assert not f.read(1)
    qq, dd = q4gpu.quant_act(x)
    assert np.array_equal(qq, q) and np.array_equal(dd.view('u4'), dx.view('u4'))
    codes, scales = q4gpu.layout_weights(u, d)
    t, b = r // 32, k // 32
    for n in (7, 64):
        size = n if n == 7 else 8
        kernel = q4gpu.Kernel(size)
        streams = [q4gpu.build_uniforms(kernel, q[v:v+size], dx[v:v+size]) for v in range(0, n, size)]
        memory = codes.nbytes + scales.nbytes + sum(s.nbytes for s in streams) + n * r * 4 + (4 << 20)
        with Driver(data_area_size=memory) as drv:
            program = drv.program(kernel.program())
            cb = drv.alloc(codes.size + 384, dtype=np.uint32)
            sb = drv.alloc(scales.size + 48, dtype=np.uint32)
            cb[:] = 0; sb[:] = 0
            cb[:codes.size] = codes.reshape(-1); sb[:scales.size] = scales.reshape(-1)
            yb = drv.alloc((n, r), dtype=np.float32)
            yb[:] = np.nan
            uniforms, parameters = [], []
            for v, stream in zip(range(0, n, size), streams):
                ub = drv.alloc(stream.size, dtype=np.uint32); ub[:] = stream
                pb = drv.alloc(len(q4gpu.PARAMS), dtype=np.uint32)
                params = {'codes': cb.addresses()[0], 'scales': sb.addresses()[0],
                          'y': yb.addresses()[v, 0], 'act': ub.addresses()[0], 'nt': t // 8,
                          'nb': b, 'tsc': t // 8 * b * 512, 'tss': t // 8 * b * 64,
                          'tsy': t // 8 * 128, 'r4': r * 4, 'mask': q4gpu.MASK_PAIR,
                          'k65536': 65536, 'k512': 512, 'k64': 64}
                pb[:] = [params[key] for key in q4gpu.PARAMS]
                uniforms.append(ub); parameters.append(pb)

            def call():
                for pb in parameters:
                    drv.execute(program, local_invocation=(16, 1, 1), uniforms=pb.addresses()[0], thread=8)

            call()
            y = np.array(yb)
            checked = np.concatenate([np.arange(tile * 32, tile * 32 + 32) for tile in (0, 71, 143, 287)])
            model = q4gpu.reference_gpu(u[checked], d[checked], q[:n], dx[:n])
            exact = np.array_equal(y[:, checked].view('u4'), model.view('u4'))
            if not np.isfinite(y).all() or not exact:
                raise RuntimeError('Python GPU reference failed; stop before C comparison')
            out = Path(f'{prefix}.python-y{n}.bin')
            out.write_bytes(y.astype('<f4').tobytes())
            samples = []
            for _ in range(iters):
                begin = time.perf_counter(); call(); samples.append(time.perf_counter() - begin)
            sec = float(np.median(samples))
            print('RESULT ' + json.dumps({'bench':'gpu-python-reference', 'n':n, 'rows':r, 'cols':k,
                  'passes': len(streams), 'seconds':sec, 'gops':2*r*k*n/sec/1e9,
                  'all_finite':True, 'sample_exact_vs_model':bool(exact),
                  'checked_words':int(model.size), 'output':str(out)}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('make', 'python'))
    ap.add_argument('fixture', type=Path)
    ap.add_argument('prefix', nargs='?', type=Path)
    ap.add_argument('--iters', type=int, default=10)
    args = ap.parse_args()
    if args.iters < 1 or args.iters > 100:
        ap.error('iters must be 1..100')
    if args.mode == 'make':
        make(args.fixture)
    else:
        if args.prefix is None:
            ap.error('python mode needs an output prefix')
        reference(args.fixture, args.prefix, args.iters)
