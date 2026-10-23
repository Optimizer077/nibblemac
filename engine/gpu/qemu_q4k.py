"""Check every scheduled Q4K op against its independent float model before Pi use.

Usage: python qemu_q4k.py ASSEMBLER_SRC [pass sizes, default 1..8]
"""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent), sys.argv[1]]
import numpy as np
from _videocore6.assembler import assemble
import q4kgpu as K
from qemu_check import Emu


def check(n, seed, R=64, cols=512):
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 16, (R, cols), dtype=np.uint8)
    blocks = cols // 32
    super_d = np.abs(rng.standard_normal((R, cols // 256)) * .002).astype(np.float16)
    super_o = np.abs(rng.standard_normal((R, cols // 256)) * .002).astype(np.float16)
    d = super_d.astype(np.float32).repeat(8, axis=1) * rng.integers(0, 64, (R, blocks)).astype(np.float32)
    o = super_o.astype(np.float32).repeat(8, axis=1) * rng.integers(0, 64, (R, blocks)).astype(np.float32)
    x = rng.standard_normal((n, cols)).astype(np.float32)
    if seed:
        u[0] = 0
        u[16] = 15
        x[0] = 1
        if n > 1:
            x[1] = -1
        if n > 2:
            x[2] = 0
    q, dx = K.quant_act(x)
    kernel = K.Kernel(n)
    codes, scales = K.layout_weights(u, d, o)
    cbase, sbase = 0, codes.nbytes + 1536
    ybase = sbase + scales.nbytes + 768
    mem = np.zeros((ybase + n * R * 4) // 4 + 64, np.uint32)
    mem[:codes.size] = codes.reshape(-1)
    mem[sbase // 4:sbase // 4 + scales.size] = scales.reshape(-1)
    uniforms = K.build_uniforms(kernel, q, dx)
    emu = Emu(mem, uniforms)
    g = kernel.g
    lane = np.arange(16, dtype=np.uint32)
    c = lambda v: np.full(16, v & 0xffffffff, np.uint32)
    emu.reg.update({g.mask.name: c(K.MASK_PAIR), g.sh4.name: c(4), g.sh8.name: c(8),
                    g.sh12.name: c(12), g.k16.name: c(-16), g.k65536.name: c(65536),
                    g.k512.name: c(512), g.k64.name: c(64), g.k256.name: c(256),
                    g.ystride.name: c(4 * R), g.nb.name: c(blocks),
                    g.yaddr.name: c(ybase) + 4 * lane})
    ca, sa = c(cbase) + 32 * lane, c(sbase) + 16 * lane
    for k in range(8):
        emu.reg[g.w[k].name] = mem[(ca + 4 * k) // 4]
    for k, reg in enumerate(g.scales):
        emu.reg[reg.name] = mem[(sa + 4 * k) // 4]
    ca += 512
    sa += 256
    emu.fifo = [mem[(ca + 4 * k) // 4].copy() for k in range(4)]
    emu.fifo += [mem[(sa + 4 * k) // 4].copy() for k in range(4)]
    emu.max_fifo = 8
    emu.reg[g.ca.name], emu.reg[g.sa.name] = ca, sa + 256
    for tile in range(R // 32):
        emu.up = 0
        emu.run(kernel.start_slots)
        for block in range(blocks):
            emu.run(kernel.body_slots)
        assert emu.up == len(uniforms), (emu.up, len(uniforms))
        emu.run(kernel.end_slots)
    assert len(emu.fifo) == 8 and emu.max_fifo <= 8
    actual = mem[ybase // 4:ybase // 4 + n * R].view(np.float32).reshape(n, R)
    reference = K.reference_gpu(u, d, o, q, dx)
    assert np.array_equal(actual.view(np.uint32), reference.view(np.uint32)), (n, seed, np.max(np.abs(actual-reference)))
    code = assemble(kernel.program())
    print(f'PASS n={n} seed={seed} words={actual.size} FIFO={emu.max_fifo} instructions={len(code)} body={kernel.stats()}', flush=True)


if __name__ == '__main__':
    for n in [int(a) for a in sys.argv[2:]] or range(1, 9):
        for seed in (0, 1):
            check(n, seed)
    print('ALL Q4K EMULATOR CHECKS PASS')
