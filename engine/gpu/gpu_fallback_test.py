"""Confirm CPU-only identity after simulated GPU setup/submit/wait failures.

Run with fault_ioctl.so preloaded and PF_GPU_TEST_FAULT=unavailable|submit|wait.
No simulated job reaches the GPU; this does not deliberately cause a hardware hang.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pinfer


def drm_fds() -> int:
    n = 0
    for fd in Path('/proc/self/fd').iterdir():
        try:
            n += os.readlink(fd) == '/dev/dri/renderD128'
        except FileNotFoundError:
            pass
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--ids', type=Path, required=True)
    a = ap.parse_args()
    mode = os.environ['PF_GPU_TEST_FAULT']
    ids = [int(x) for x in a.ids.read_text().replace(',', ' ').split()]
    assert len(ids) == 64
    batches = [ids, [760] * 7, [760]]
    before = drm_fds()
    with pinfer.Model.load(a.model, tokenizer=False, max_ctx=512, gpu_split=0) as m:
        expected = [m.eval(x).copy() for x in batches]
    with pinfer.Model.load(a.model, tokenizer=False, max_ctx=512, gpu_split=0.33) as m:
        actual = [m.eval(x).copy() for x in batches]
        assert m.n_past == 72
    checks = [np.isfinite(x).all() and x.tobytes() == y.tobytes()
              for x, y in zip(actual, expected)]
    stats = (ctypes.c_uint32 * 3)()
    get_stats = ctypes.CDLL(None).pf_gpu_fault_stats
    get_stats.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    get_stats.restype = None
    get_stats(stats)
    if mode == 'unavailable':
        assert stats[0] == 1 and stats[1] == 0
    elif mode == 'submit':
        assert stats[1] == 2
    elif mode == 'wait':
        assert stats[1] > 0 and stats[2] >= 1
    else:
        raise ValueError(mode)
    assert drm_fds() == before, 'DRM fd leaked after close'
    assert all(checks), 'CPU fallback differs from CPU-only baseline'
    print('RESULT ' + json.dumps({
        'gpu_fallback': mode, 'simulated': True, 'checks': len(checks),
        'all_bits_identical': True, 'compared_fp32_words': sum(x.size for x in actual),
        'setup_faults': stats[0], 'submit_calls': stats[1], 'wait_faults': stats[2],
        'drm_fds_after_close': drm_fds(),
        'sha256': hashlib.sha256(b''.join(x.tobytes() for x in actual)).hexdigest(),
    }), flush=True)


if __name__ == '__main__':
    main()
