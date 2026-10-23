"""Pilot fixtures from real Qwen3.5 Q4K weights; generated data stays outside git.

python q4k_fixture.py ASSEMBLER_SRC MODEL.pq OUTPUT_DIR [--only smoke,qkv,...]
Reference: four distributed 32-row tiles, all 7/64 vectors and every column block.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import sys

p = argparse.ArgumentParser()
p.add_argument('assembler_src')
p.add_argument('model', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--only', default='smoke,qkv,gate,up,down,head')
a = p.parse_args()
sys.path.insert(0, a.assembler_src)
import numpy as np
import q4kgpu as K


def unpack6(b):
    lo, hi = b[..., :16], b[..., 16:24]
    high = np.concatenate([(hi >> (2 * i)) & 3 for i in range(4)], axis=-1)
    return np.concatenate((lo & 15, lo >> 4), axis=-1) | (high << 4)


def components(raw, rows, cols):
    sb = raw.reshape(rows // 4, cols // 256, 576)
    sc = sb[..., :64]
    ds = sc[..., :8].copy().view('<f2').astype(np.float32)
    os = sc[..., 8:16].copy().view('<f2').astype(np.float32)
    ds = ds[:, :, None, :] * unpack6(sc[..., 16:40]).reshape(rows // 4, cols // 256, 8, 4)
    os = os[:, :, None, :] * unpack6(sc[..., 40:64]).reshape(rows // 4, cols // 256, 8, 4)
    codes = sb[..., 64:].reshape(rows // 4, cols // 256, 8, 4, 16)
    u = np.concatenate((codes & 15, codes >> 4), axis=-1)
    u = np.ascontiguousarray(u.transpose(0, 3, 1, 2, 4)).reshape(rows, cols)
    ds = np.ascontiguousarray(ds.transpose(0, 3, 1, 2)).reshape(rows, cols // 32)
    os = np.ascontiguousarray(os.transpose(0, 3, 1, 2)).reshape(rows, cols // 32)
    return u, ds, os


with a.model.open('rb') as f:
    header = struct.unpack('<8sIIQQ' + 'I' * 18 + 'ff', f.read(112))
    f.seek(header[3])
    entries = {}
    for i in range(header[2]):
        item = struct.unpack('<80sII4IQQ8x', f.read(128))
        entries[item[0].split(b'\0')[0].decode()] = item
names = {'smoke': 'layers.0.mlp.gate_proj.weight',
         'qkv': 'layers.0.linear_attn.in_proj_qkv.weight',
         'gate': 'layers.0.mlp.gate_proj.weight', 'up': 'layers.0.mlp.up_proj.weight',
         'down': 'layers.0.mlp.down_proj.weight', 'head': 'embed_tokens.weight'}
a.output.mkdir(parents=True, exist_ok=True)
manifest = []
for shape in a.only.split(','):
    item = entries[names[shape]]
    assert item[1] == 4
    rows, cols = item[3:5]
    if shape == 'smoke':
        rows = 256
    nbytes = rows // 4 * (cols // 256) * 576
    with a.model.open('rb') as f:
        f.seek(item[7])
        raw = f.read(nbytes)
    index = np.concatenate([np.arange(t * 32, (t + 1) * 32, dtype=np.uint32)
                            for t in (0, rows // 32 // 3, 2 * (rows // 32) // 3, rows // 32 - 1)])
    assert len(set(map(int, index))) == 128
    # Only decode reference tiles; the full original packed tensor goes to the C runner.
    tile_bytes = cols // 256 * 576
    pieces = [np.frombuffer(raw, np.uint8)[int(t // 4) * tile_bytes:int(t // 4 + 8) * tile_bytes]
              for t in index[::32]]
    u, ds, os = components(np.concatenate(pieces), 128, cols)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((64, cols)).astype(np.float32)
    q, dx = K.quant_act(x)
    ref7 = K.reference_gpu(u, ds, os, q[:7], dx[:7])
    ref64 = K.reference_gpu(u, ds, os, q, dx)
    path = a.output / (shape + '.bin')
    with path.open('wb') as f:
        f.write(struct.pack('<8I', 0x4b345150, 1, rows, cols, 64, 128, nbytes, 0))
        for data in (raw, x, q, dx, index, ref7, ref64):
            f.write(data if isinstance(data, bytes) else data.tobytes())
    record = {'shape': shape, 'tensor': names[shape], 'dtype': 'Q4K_G32',
              'rows': rows, 'cols': cols, 'bytes': path.stat().st_size,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'reference_rows': list(map(int, index)), 'reference_words_n7': 896,
              'reference_words_n64': 8192}
    manifest.append(record)
    print(shape, rows, cols, path.stat().st_size, flush=True)
(a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
