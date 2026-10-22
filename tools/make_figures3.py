"""Hardware figures for the README: the Pi 4's memory hierarchy, a roofline of pinfer's passes, and
what the Cortex-A72's instruction set allows. Four PNGs per figure in docs/figures/, desktop and phone
layouts in the light and dark themes (`<name>.png`, `-dark`, `-mobile`, `-mobile-dark`).

Every number is measured in this repo (bench/results/, engine/bench/KERNELS.md, engine/VERIFY.md) or
taken from Arm and Raspberry Pi documentation; each constant names its source. Sources are listed in
docs/ARCHITECTURE.md §9. Style, palette, themes and layouts: tools/figstyle.py.

    uv run --no-project --with matplotlib python tools/make_figures3.py
"""
from __future__ import annotations

import logging
import re
import struct
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from figstyle import (
    DASH,
    LAYOUTS,
    LINE,
    MARKER,
    THEMES,
    Layout,
    Theme,
    hbar_rows,
    header,
    new_fig,
    on_fill,
    ring,
    save,
    setup,
    style,
    wrap,
)
from make_figures2 import BYTES, PQ_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- the board
# bench/results/*.json "state_before" (e.g. memband_20261021_210615): model "Raspberry Pi 4 Model B
# Rev 1.1", freq_khz "1500000" on all four cores, cpu_features "fp asimd evtstrm crc32 cpuid".
CORES = 4
CLOCK_HZ = 1.5e9
# Caches. Cortex-A72 MPCore TRM r0p3 (Arm 100095_0003), §1.3: "Fixed 48K L1 instruction cache and 32K L1
# data cache", "Shared L2 cache of 512KB, 1MB, 2MB or 4MB configurable size"; §2.1: 48KB 3-way and 32KB
# 2-way set-associative, 64-byte lines. Raspberry Pi, BCM2711 page (documentation/computers/processors):
# "32 kB data + 48 kB instruction L1 cache per core. 1 MB L2 cache."
L1D_KB, L1I_KB, L2_BYTES = 32, 48, 1 << 20
# DRAM. Raspberry Pi, BCM2711 page: "Accesses up to 8 GB LPDDR4-2400 SDRAM"; this board has 4 GB.
# Raspberry Pi does not publish the Pi 4's memory bus width; with the 32-bit bus of docs/design.md the
# peak is 2400 MT/s x 4 B = 9.6 GB/s. An ESTIMATE, labelled as one (Wikipedia's "Raspberry Pi 4" article
# gives "LPDDR4 RAM running at 3.2 GHz", which would make it 12.8 GB/s; no conclusion depends on it).
DRAM_PEAK_EST = 2400e6 * 4 / 1e9
# Measured: memband, 256 MB streaming read, GB/s by threads (bench/results/memband_20261021_210615.json)
MEMBAND_GBS = {1: 4.202, 2: 4.081, 4: 4.068}

# ---------------------------------------------------------------- the model's data (Q4K default)
# Bytes a plain decode step streams: every part of the .pq tensor directory but the MTP head
# (make_figures2.BYTES, re-read from the file by its pq_bytes()). Of them, the Q4K matrices hold
# 751,894,528 weights at 0.5625 B each (576-B superblock per 4 x 256 weights); the other 2.0 MB are F32
# norms and conv kernels. Re-read by check_model() when the model file is present.
BYTES_PLAIN = sum(b for _, b in BYTES[:4])  # 424,934,656
BYTES_HEAD = BYTES[0][1]  # tied output head + final norm
WEIGHTS_PLAIN = 751_894_528  # Q4K matrix weights a plain step multiplies (all tensors but mtp.*)
Q4K_G32 = 4  # dtype id of Q4K_G32 in the tensor directory (engine/src/pq_quant.h PQ_Q4K_G32)
WEIGHTS_HEAD = 248_320 * 1024  # the tied head, part of WEIGHTS_PLAIN
STATE_BYTES = 18 * 16 * 128 * 128 * 4  # recurrent state, read and written per token (ARCHITECTURE.md §5)
KV_PER_TOKEN = 6 * 2 * 2 * 256 * 4  # fp32 KV-cache bytes per token of context (ARCHITECTURE.md §5)
STATE_CHUNK = 128 * 32 * 4  # the recurrence's L1-resident chunk, 16 KB (engine/VERIFY.md §2)

# ---------------------------------------------------------------- measured passes (Q4K, 4 threads)
PLAIN_TPS = 7.89  # plain decode, speed-ab-merge4_20261022_103609 (merged build, median of 3)
VERIFY_MS = 152.9  # k=3 verify pass over 4 positions, engine/VERIFY.md "Result" (ver3-ab-p64_20261022_080513)
PREFILL_TPS = 37.1  # 64-token prompt, speed-ab-merge4_20261022_103609
PREFILL_N, GROUP = 64, 16  # one 64-row chunk; pf_matmul streams the weights once per 16 rows (KERNELS.md v13)

# ---------------------------------------------------------------- compute roofs
# Theoretical int8 peak of the instructions pinfer's kernels use. Cortex-A72 Software Optimization Guide
# (Arm UAN 0016A) §3.14, AArch64 "ASIMD multiply long" SMULL(2) and "ASIMD multiply accumulate long"
# SMLAL(2): throughput 1 per cycle, pipeline F0 only; "ASIMD pairwise add and accumulate" SADALP:
# 1 per cycle on F1, so in principle the widening issues alongside. One SMULL/SMLAL multiplies 8 int8
# pairs into 8 int16 lanes (8 MACs = 16 ops): 8 MACs x 2 ops x 4 cores x 1.5 GHz = 96 GOP/s.
# ubench measures 1.00 cycle per SMULL/SMLAL but finds that SADDLP does not overlap with them
# (KERNELS.md "Machine facts"), so the kernels cannot reach it.
MACS_PER_CYCLE = 8
PEAK_INT8 = MACS_PER_CYCLE * 2 * CORES * CLOCK_HZ / 1e9
# Practical roof: the Q4K kernel itself, cold weights, 4 threads, at a batch of 16 (a prefill group),
# on the largest layer matrix: qkv 6144 x 1024 in 4238.94 us (bench/results/kbench-v13-group_20261022_
# 005221.json, variant v13-group16). Not a maximum: the same file gives 39.4-49.6 GOP/s over the four
# layer shapes (qkv, gate, down, out) at batches of 4, 16 and 64 (qkv 49.3 at 4, 47.2 at 64).
KERNEL_PEAK = 2 * 6144 * 1024 * 16 / 4238.94e-6 / 1e9

# ---------------------------------------------------------------- the instruction set
# Multiply-accumulates per 128-bit NEON instruction, from the Arm A64 ISA (DDI 0602, 2024-12):
# - FMLA (vector) .4S: 4 fp32 lanes, FEAT_AdvSIMD (A72 SOG §3.15: Q-form throughput 1 per cycle);
# - FMLA (vector) .8H: 8 fp16 lanes, needs FEAT_FP16 ("OPTIONAL from Armv8.2", Arm feature names 109697);
# - SMLAL/SMLAL2 (vector) .8H: 8 int8 pairs into 16-bit lanes, FEAT_AdvSIMD;
# - SDOT (vector) .4S: "the dot product of the four signed 8-bit elements in each 32-bit element", 4 x 4
#   = 16, FEAT_DotProd ("OPTIONAL" in Armv8.2 and 8.3, mandatory from 8.4);
# - SMMLA (vector) .4S: "the 2x8 matrix ... by the 8x2 matrix ... 2x2 32-bit integer matrix product",
#   2 x 2 x 8 = 32, FEAT_I8MM (optional from 8.2, mandatory from 8.6).
# The A72 has none of the optional ones: Cortex-A72 TRM r0p3 ID_AA64PFR0_EL1 = 0x2222 (AdvSIMD field 0:
# no half precision), ID_AA64ISAR0_EL1 = 0x10000 (DP field 0), and this board's /proc/cpuinfo lists no
# asimdhp / asimddp / i8mm. The Pi 5's Cortex-A76 (Raspberry Pi, BCM2712 page) implements SDOT/UDOT and
# half precision but not I8MM (Cortex-A76 TRM r4p1 §A1.1; ID_AA64ISAR1_EL1 reset 0x100001, I8MM field 0).
# (name on desktop, name on the phone, MACs, available on the A72, desktop note after the value)
ISA = [
    ("fp32 FMLA .4S\nArmv8.0-A", "fp32 FMLA · Armv8.0-A", 4, True, ""),
    ("fp16 FMLA .8H\nArmv8.2-A FP16", "fp16 FMLA · Armv8.2-A · in the Pi 5", 8, False,
     "in the Pi 5's Cortex-A76"),
    ("int8 SMULL / SMLAL .8H\nArmv8.0-A · pinfer's kernels", "int8 SMLAL · Armv8.0-A · pinfer", 8, True,
     "16-bit sums, widened by extra adds"),
    ("int8 SDOT .4S\nArmv8.2-A DotProd", "int8 SDOT · Armv8.2-A · in the Pi 5", 16, False,
     "in the Pi 5's Cortex-A76; 32-bit sums"),
    ("int8 SMMLA .4S\nArmv8.6-A I8MM", "int8 SMMLA · Armv8.6-A · not in the Pi 5", 32, False,
     "not in the Pi 5's Cortex-A76 either"),
]


def pq_q4k_weights(path: Path) -> tuple[int, int]:
    """(Q4K weights, bytes) of every tensor but mtp.* in a .pq file (directory layout: engine/FORMAT.md)."""
    with path.open("rb") as f:
        hdr = f.read(4096)
        n_tensors, dir_offset = struct.unpack_from("<IQ", hdr, 12)
        f.seek(dir_offset)
        directory = f.read(128 * n_tensors)
    weights = nbytes = 0
    for i in range(n_tensors):
        entry = directory[128 * i:128 * (i + 1)]
        if entry[:80].split(b"\0", 1)[0].startswith(b"mtp."):
            continue
        dtype, ndim = struct.unpack_from("<II", entry, 80)
        shape = struct.unpack_from("<4I", entry, 88)[:ndim]
        nbytes += struct.unpack_from("<Q", entry, 112)[0]
        if dtype == Q4K_G32:
            weights += shape[0] * shape[1]
    return weights, nbytes


def check_model() -> None:
    """Warn if the model on disk no longer matches WEIGHTS_PLAIN / BYTES_PLAIN (skipped without it)."""
    if not PQ_MODEL.exists():
        logger.info("roofline: %s not present, using the recorded weight count", PQ_MODEL.name)
        return
    found = pq_q4k_weights(PQ_MODEL)
    if found != (WEIGHTS_PLAIN, BYTES_PLAIN):
        logger.warning("roofline: %s gives %s, the figure records %s", PQ_MODEL.name, found,
                       (WEIGHTS_PLAIN, BYTES_PLAIN))
    else:
        logger.info("roofline: recorded weight count matches %s", PQ_MODEL.name)


def roof_points() -> list[tuple[str, str, float, float]]:
    """(label, phone label, ops per weight byte, achieved GOP/s) of the three measured passes.

    ops = 2 x the multiply-accumulates of the weight matrices; attention, the recurrence and the other
    per-token work add time but are not counted, so achieved = ops / the whole pass's wall time.
    """
    plain_ops = 2 * WEIGHTS_PLAIN
    layer_w, layer_b = WEIGHTS_PLAIN - WEIGHTS_HEAD, BYTES_PLAIN - BYTES_HEAD
    # Prefill: the layers stream once per group of 16 rows; only the last row needs logits (one head pass).
    prefill_ops = 2 * (PREFILL_N * layer_w + WEIGHTS_HEAD)
    prefill_bytes = PREFILL_N // GROUP * layer_b + BYTES_HEAD
    return [("plain decode, 1 position", "plain decode, n = 1", plain_ops / BYTES_PLAIN,
             plain_ops * PLAIN_TPS / 1e9),
            ("verify pass, 4 positions", "verify, n = 4", 4 * plain_ops / BYTES_PLAIN,
             4 * plain_ops / (VERIFY_MS / 1e3) / 1e9),
            ("prefill, 16 positions per read", "prefill, n = 16", prefill_ops / prefill_bytes,
             prefill_ops / (PREFILL_N / PREFILL_TPS) / 1e9)]


# ---------------------------------------------------------------- figure 1: the memory hierarchy
def _hw_text() -> tuple[list[tuple[str, str | None, str, list[str]]], list[str | None], str]:
    """Levels (box heading, box detail, note heading, note bullets), link labels between levels, footer."""
    mb = BYTES_PLAIN / 1e6
    nbsp = "\u00a0"  # keeps two words on one line through wrap()
    levels = [
        (f"{CORES} × Cortex-A72 cores at {CLOCK_HZ / 1e9:.1f} GHz",
         "each with a 128-bit NEON (SIMD) unit and L1 caches",
         "Kept in L1, next to the arithmetic",
         [f"a {STATE_CHUNK // 1024} KB chunk of the recurrent state, reused by every position of a pass",
          "int8 activations, 1 byte per input: 16 KB for a prefill group of 16 positions × 1024 inputs"]),
        (f"L2 cache: {L2_BYTES >> 20} MB, shared", None,
         "Too small for anything model-sized",
         [(f"one token reads {mb:.1f} MB of weights, {BYTES_PLAIN / L2_BYTES:.0f}× the L2: each byte "
           "passes through once and is gone")]),
        ("LPDDR4 DRAM: 4 GB", None,
         "Streamed from DRAM for every generated token",
         [f"all {mb:.1f} MB of weights (the file minus the 11.5 MB draft{nbsp}head)",
          f"the {STATE_BYTES / 1e6:.1f} MB recurrent state, read and written back",
          (f"the KV cache: {KV_PER_TOKEN // 1024} KiB per token of context "
           f"({4096 * KV_PER_TOKEN >> 20} MiB at 4096 tokens)")]),
    ]
    links: list[str | None] = [
        None,
        (f"{MEMBAND_GBS[1]:.2f} GB/s measured\n(1 core; {MEMBAND_GBS[4]:.2f} with 4)\n"
         f"peak ≈ {DRAM_PEAK_EST:.1f} GB/s (estimate)")]
    footer = (f"Decode speed ≈ bandwidth ÷ weight bytes per token: {MEMBAND_GBS[1]:.2f} GB/s ÷ {mb:.1f} MB = "
              f"at most {MEMBAND_GBS[1] * 1e3 / mb:.1f} tokens/s; pinfer reaches {PLAIN_TPS:.2f} tokens/s.")
    bound = [(_bind(h), d and _bind(d), _bind(nh), [_bind(b) for b in bs]) for h, d, nh, bs in levels]
    return bound, links, _bind(footer)


def _bind(s: str) -> str:
    """Join each number to its unit with a no-break space, so wrap() never splits "16 KB" (it turns the
    no-break spaces back into plain ones after breaking the lines)."""
    s = re.sub(r" × (?=\d)", "\u00a0×\u00a0", s)  # "16 positions × 1024 inputs" stays one piece
    return re.sub(r"(\d) (?=[A-Za-z×])", "\\1\u00a0", s)


def fig_hardware(t: Theme, L: Layout) -> None:
    """Block diagram in CSS px: cores, L2 and DRAM as filled boxes, what pinfer keeps at each level beside
    them (desktop) or under them (phone), the measured stream between L2 and DRAM.
    """
    levels, links, footer = _hw_text()
    fills = [t.orange, t.aqua, t.blue]  # compute warm, memory cool (as in the roofline)
    pad, line_h, bullet = 12.0, L.text * 1.35, 14.0
    width = L.css_width - 2 * L.margin
    box_w = width if L.mobile else 330.0
    note_x = L.margin if L.mobile else L.margin + box_w + 26
    note_w = width if L.mobile else width - box_w - 26
    # arrow length between levels: room for its label beside it, or a plain 40 px arrow
    link_h = [(s.count("\n") + 1) * line_h + 16 if s else 40.0 for s in links]
    probe, _ = new_fig(t, L, 10)  # measures the wrapped texts

    def wr(s: str, w: float, weight: str = "normal") -> str:
        return wrap(probe, s, L.text, w, weight)

    def n_lines(s: str) -> int:
        return s.count("\n") + 1

    tiles = 2 if L.mobile else 4  # core tiles per row
    tile_rows = -(-CORES // tiles)
    tile_text = "L1D {} KB · L1I {} KB" if L.mobile else "L1D {} KB\nL1I {} KB"
    tile_h = (1 + n_lines(tile_text)) * line_h + 10
    rows: list[dict[str, Any]] = []
    for i, (head, detail, note_head, bullets) in enumerate(levels):
        box_head = wr(head, box_w - 2 * pad, "semibold")
        box_detail = wr(detail, box_w - 2 * pad) if detail else ""
        box_h = 2 * pad + n_lines(box_head) * line_h + (n_lines(box_detail) * line_h if detail else 0)
        if i == 0:
            box_h += 6 + tile_rows * tile_h + (tile_rows - 1) * 6
        nh = wr(note_head, note_w, "semibold")
        nb = [wr(b, note_w - bullet) for b in bullets]
        note_h = (n_lines(nh) + sum(n_lines(b) for b in nb)) * line_h + 4 * len(nb)
        rows.append({"head": box_head, "detail": box_detail, "box_h": box_h, "note_head": nh,
                     "bullets": nb, "note_h": note_h})
    foot = wr(footer, width, "semibold")
    plt.close(probe)
    y, gap = 0.0, 10.0
    for i, r in enumerate(rows):  # top of each box and of its note
        r["y"] = y
        r["note_y"] = y + r["box_h"] + gap if L.mobile else y + 2
        y = (r["note_y"] + r["note_h"]) if L.mobile else y + max(r["box_h"], r["note_h"])
        r["end"] = y
        y += link_h[i] if i < len(link_h) else 0
    body = rows[-1]["end"] + 18 + n_lines(foot) * line_h + L.margin
    fig, ax = new_fig(t, L, body)
    ax.axis("off")
    rect = header(fig, t, L, "The Pi 4 as the model sees it",
                  "Broadcom BCM2711. Nothing model-sized fits in a cache, so every token is a full trip "
                  "through DRAM")
    ax.set_position((0, 0, 1, rect[3]))
    ax.set_xlim(0, L.css_width)
    ax.set_ylim(body, 0)  # y down, CSS px from the top of the body
    x0 = L.margin
    for i, (r, fc) in enumerate(zip(rows, fills)):
        ax.add_patch(FancyBboxPatch((x0, r["y"]), box_w, r["box_h"], boxstyle="round,pad=0,rounding_size=6",
                                    fc=fc, ec="none"))
        ink = on_fill(t, fc)
        ty = r["y"] + pad
        ax.text(x0 + pad, ty, r["head"], ha="left", va="top", color=ink, fontweight="semibold", linespacing=1.35)
        ty += n_lines(r["head"]) * line_h
        if r["detail"]:
            ax.text(x0 + pad, ty, r["detail"], ha="left", va="top", color=ink, linespacing=1.35)
            ty += n_lines(r["detail"]) * line_h
        if i == 0:  # one surface tile per core
            tw = (box_w - 2 * pad - (tiles - 1) * 6) / tiles
            for c in range(CORES):
                tx, tyy = x0 + pad + (c % tiles) * (tw + 6), ty + 6 + (c // tiles) * (tile_h + 6)
                ax.add_patch(FancyBboxPatch((tx, tyy), tw, tile_h, boxstyle="round,pad=0,rounding_size=4",
                                            fc=t.surface, ec="none"))
                ax.text(tx + tw / 2, tyy + 5, f"core {c}", ha="center", va="top", color=t.ink,
                        fontweight="semibold")
                ax.text(tx + tw / 2, tyy + 5 + line_h, tile_text.format(L1D_KB, L1I_KB), ha="center",
                        va="top", color=t.ink2, linespacing=1.35)
        ny = r["note_y"]
        ax.text(note_x, ny, r["note_head"], ha="left", va="top", color=t.ink, fontweight="semibold",
                linespacing=1.35)
        ny += n_lines(r["note_head"]) * line_h + 4
        for b in r["bullets"]:
            ax.text(note_x, ny, "•", ha="left", va="top", color=t.ink2)
            ax.text(note_x + bullet, ny, b, ha="left", va="top", color=t.ink2, linespacing=1.35)
            ny += n_lines(b) * line_h + 4
        if i + 1 < len(rows):  # data flows up: from the level below to this box (phone: to its note)
            ax_x = x0 + 34
            top = r["end"] if L.mobile else r["y"] + r["box_h"]
            bottom = rows[i + 1]["y"]
            ax.add_patch(FancyArrowPatch((ax_x, bottom - 6), (ax_x, top + 6), arrowstyle="-|>",
                                         mutation_scale=14, color=t.muted, linewidth=1.6, shrinkA=0, shrinkB=0))
            if links[i]:
                ax.text(ax_x + 14, (top + bottom) / 2, links[i], ha="left", va="center", color=t.ink2,
                        linespacing=1.35)
    ax.text(x0, rows[-1]["end"] + 18, foot, ha="left", va="top", color=t.ink, fontweight="semibold",
            linespacing=1.35)
    save(fig, t, L, "pi4_hardware", layout=False)


# ---------------------------------------------------------------- figure 2: roofline
def fig_roofline(t: Theme, L: Layout) -> None:
    """Log-log roofline: the measured memory roof, the kernel's measured compute roof and the
    instruction-level int8 peak, with the three measured passes."""
    fig, ax = new_fig(t, L, 440 if L.mobile else 400)
    style(ax, t, ylabel="achieved GOP/s (log)", xlabel="operations per byte of weights read (log)", grid="y")
    ax.set_xscale("log")
    ax.set_yscale("log")
    x_lo, x_hi, y_lo, y_hi = 1.0, 200.0, 2.0, 200.0
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    xt, yt = [1, 2, 5, 10, 20, 50, 100, 200], [2, 5, 10, 20, 50, 100, 200]
    ax.set_xticks(xt, [str(v) for v in xt])
    ax.set_yticks(yt, [str(v) for v in yt])
    ax.minorticks_off()
    bw = MEMBAND_GBS[1]
    ridge, k_ridge = PEAK_INT8 / bw, KERNEL_PEAK / bw
    ax.plot([x_lo, ridge], [bw * x_lo, PEAK_INT8], color=t.blue, linewidth=LINE, solid_capstyle="butt")
    ax.plot([ridge, x_hi], [PEAK_INT8, PEAK_INT8], color=t.orange, linewidth=LINE, linestyle=DASH)
    ax.plot([k_ridge, x_hi], [KERNEL_PEAK, KERNEL_PEAK], color=t.orange, linewidth=LINE,
            solid_capstyle="butt")
    ax.annotate(f"{PEAK_INT8:.0f}", (x_hi, PEAK_INT8), xytext=(-2, 4), textcoords="offset points", ha="right",
                va="bottom", color=t.ink)
    ax.annotate(f"{KERNEL_PEAK:.1f}", (x_hi, KERNEL_PEAK), xytext=(-2, 4), textcoords="offset points",
                ha="right", va="bottom", color=t.ink)
    if not L.mobile:  # the phone has no room left of the roof; the legend names it
        ax.annotate(f"{bw:.2f} GB/s", (5, bw * 5), xytext=(-8, 4), textcoords="offset points", ha="right",
                    va="bottom", color=t.ink)
    pts = roof_points()
    ax.plot([p[2] for p in pts], [p[3] for p in pts], "o", color=t.aqua, markersize=MARKER, zorder=4, **ring(t))
    # Labels in the free space under the roofs: n = 1 to its right; desktop: the other two under their
    # points; phone (narrower decades): verify under its point, prefill further down so the two clear.
    if L.mobile:
        offsets = [((8, -4), "left", "top"), ((10, -12), "center", "top"), ((4, -58), "center", "top")]
    else:
        offsets = [((10, 0), "left", "center"), ((0, -12), "center", "top"), ((0, -12), "center", "top")]
    for (name, short, x, y), (off, ha, va) in zip(pts, offsets):
        leader = {"arrowstyle": "-", "color": t.muted, "linewidth": 1.0, "shrinkA": 3, "shrinkB": 7}
        ax.annotate(f"{short if L.mobile else name}\n{y:.1f} GOP/s", (x, y), xytext=off, textcoords="offset points",
                    ha=ha, va=va, color=t.ink2, linespacing=1.3, zorder=5,
                    arrowprops=leader if abs(off[1]) > 20 else None)  # a label set far off gets a leader
    keys = [Line2D([], [], color=t.blue, linewidth=LINE, label=f"DRAM roof: {bw:.2f} GB/s × intensity (measured)"),
            Line2D([], [], color=t.orange, linewidth=LINE, label="Q4K kernel, measured (qkv matrix, batch of 16)"),
            Line2D([], [], color=t.orange, linewidth=LINE, linestyle=DASH,
                   label="int8 NEON peak: 8 MACs/cycle × 4 cores × 1.5 GHz"),
            Line2D([], [], color=t.aqua, marker="o", linestyle="none", markersize=MARKER - 2,
                   label="pinfer passes, measured (whole pass)")]
    rect = header(fig, t, L, "Why decode is memory-bound and prefill is compute-bound",
                  "Q4K model on the Pi 4. One position per weight read sits under the DRAM roof; 4 or more "
                  "(a verify pass, prefill) sit under the kernel's compute roof",
                  handles=keys, ncol=1 if L.mobile else 2)
    save(fig, t, L, "roofline", rect)


# ---------------------------------------------------------------- figure 3: the instruction set
def fig_isa(t: Theme, L: Layout) -> None:
    """Horizontal bars: multiply-accumulates per 128-bit instruction, blue where the A72 has it."""
    off = t.muted  # not on this chip: de-emphasised
    fig, ax = new_fig(t, L, 300 if L.mobile else 290)
    style(ax, t, xlabel="multiply-accumulates per 128-bit instruction", grid="x")
    ax.set_xlim(0, 36)
    ax.set_xticks([0, 8, 16, 24, 32])
    keys = [Patch(color=t.blue, label="on the Pi 4 (Cortex-A72, Armv8.0-A)"),
            Patch(color=off, label="not on the Pi 4")]
    rect = header(fig, t, L, "What the A72's instruction set allows",
                  "No dot-product or matrix instructions, so pinfer's int8 kernels use SMULL/SMLAL plus "
                  "widening adds; SDOT would do twice the work per instruction. A count per instruction, "
                  "not a speed comparison between chips", handles=keys, ncol=1 if L.mobile else 2)
    hbar_rows(ax, t, L, [r[1] if L.mobile else r[0] for r in ISA], [[r[2]] for r in ISA],
              [[t.blue if r[3] else off] for r in ISA], "{:.0f}", rect)
    if not L.mobile:  # a desktop row's single bar is centred on its index (hbar_rows)
        box = {"boxstyle": "square,pad=0.15", "fc": t.surface, "ec": "none"}  # breaks the grid under a note
        for i, (_, _, macs, _, note) in enumerate(ISA):
            if not note:
                continue
            if macs > 24:  # no room after the longest bar: the note goes inside it
                ax.annotate(note, (0, i), xytext=(8, 0), textcoords="offset points", va="center",
                            color=on_fill(t, off))
            else:  # after the value label
                ax.annotate(note, (macs, i), xytext=(24, 0), textcoords="offset points", va="center",
                            color=t.ink2, bbox=box)
    save(fig, t, L, "arm_isa", rect)


FIGURES = (fig_hardware, fig_roofline, fig_isa)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_model()
    logger.info("roofs: DRAM %.2f GB/s, int8 peak %.1f GOP/s, Q4K kernel %.1f GOP/s (ridges %.1f / %.1f ops/B)",
                MEMBAND_GBS[1], PEAK_INT8, KERNEL_PEAK, PEAK_INT8 / MEMBAND_GBS[1], KERNEL_PEAK / MEMBAND_GBS[1])
    for name, _, x, y in roof_points():
        logger.info("roofline: %s: %.2f ops/B, %.2f GOP/s (DRAM roof there %.1f)", name, x, y, MEMBAND_GBS[1] * x)
    for L in LAYOUTS:
        for t in THEMES:
            setup(t, L)
            for make in FIGURES:
                make(t, L)


if __name__ == "__main__":
    main()
