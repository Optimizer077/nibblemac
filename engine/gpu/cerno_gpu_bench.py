"""Run the existing Cerno benchmark with an explicit Model.load GPU share.

Only this isolated benchmark process wraps Model.load; no Cerno checkout is edited.
Both arms keep Decider's default in-engine checkpoint path.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import runpy
import sys


def memory() -> dict:
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'CmaTotal', 'CmaFree', 'Shmem'):
            info[key + '_kib'] = int(value.split()[0])
    info['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    drm = []
    for fd in Path('/proc/self/fd').iterdir():
        try:
            if os.readlink(fd) == '/dev/dri/renderD128':
                drm.append(Path('/proc/self/fdinfo', fd.name).read_text())
        except FileNotFoundError:
            pass
    info['drm_fdinfo'] = drm
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', type=float, required=True)
    ap.add_argument('--source', type=Path, required=True)
    args, rest = ap.parse_known_args()
    import pinfer
    original = pinfer.Model.load

    def load(cls, *positional, **kwargs):
        kwargs['gpu_split'] = args.gpu
        before = memory()
        model = original(*positional, **kwargs)
        print('GPU_SETUP ' + json.dumps({'share': args.gpu, 'before': before, 'after': memory(),
                                        'info': model.info()}), flush=True)
        return model

    pinfer.Model.load = classmethod(load)
    sys.argv = [str(args.source), *rest]
    runpy.run_path(str(args.source), run_name='__main__')


if __name__ == '__main__':
    main()
