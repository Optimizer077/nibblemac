"""'How it works' figures for the README (docs/figures/*.png and *-dark.png), from the logged measurements.

Sources: engine/CORE.md (per-step profile), engine/bench/KERNELS.md (kernel GB/s, quantization
error), engine/SPEC.md (MTP acceptance), notes.md (memband, kernel experiment), bench/results/,
and the tensor directory of the shipped Q4K model (bytes per part).

    uv run --no-project --with matplotlib python tools/make_figures2.py
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from figstyle import (
    BAR_PX,
    DPI,
    GAP_PX,
    ROOT,
    THEMES,
    Theme,
    fig_title,
    new_fig,
    on_fill,
    px_to_data,
    ring,
    save,
    setup,
    slot_lim,
    style,
    title,
)

logger = logging.getLogger(__name__)

# Per-token time, ms, Q4K, 4 threads (engine/CORE.md final profile; verify pass = k=3, 4 positions)
STEP_PLAIN = [("weight streaming (matmul)", 118.6), ("recurrent state (18 linear-attn layers)", 6.5),
              ("everything else (norms, SiLU, attention, argmax)", 3.2)]
STEP_VERIFY = [("weight streaming (matmul)", 129.0), ("recurrent state (18 linear-attn layers)", 22.8),
               ("everything else (norms, SiLU, attention, argmax)", 8.4)]
# Kernel weight-streaming speed, GB/s, cold weights, real shapes (KERNELS.md v9/v11)
KERNEL_GBS = [("Q8 (8.5 bit)", 4.33, 1), ("Q4K (4.5 bit)", 3.91, 3), ("Q4 (4.5 bit)", 3.91, 3),
              ("bf16", 3.94, 2), ("fp32", 3.86, 2)]
MEMBAND = 4.20  # GB/s, bench/results/memband_20261021_210615.json (1 thread, saturated)
# Quantization: relative RMS weight error (%) and perplexity (wikitext-2, 8 chunks)
QUANT = [("bf16", 0.0, 17.93), ("Q8", 0.57, 17.96), ("Q4K", 7.8, 18.50), ("Q4 (symmetric)", 8.9, 21.61)]
# MTP acceptance per draft position (651 positions, engine/SPEC.md) and the two controls
ACCEPT = [("1st draft", 0.873), ("2nd (given 1st ok)", 0.868), ("3rd (given 1–2 ok)", 0.897), ("4th (given 1–3 ok)", 0.937)]
CONTROLS = [("as specified", 0.873), ("hidden state taken\nbefore final norm", 0.545), ("concat order swapped", 0.003)]
# OS experiment, decode tok/s (notes.md "Custom kernel"): same binary and model
OS_EXP = [("stock kernel", 7.26, 13.35), ("custom kernel\n(NO_HZ_FULL built in)", 7.28, 13.51),
          ("+ isolated, tickless\ncores 1–3", 7.21, 13.45), ("+ weights in 2 MB\nhuge pages", 7.30, 13.42)]
# Bytes of the shipped Q4K model by part, summed over the tensor directory of PQ_MODEL (engine/FORMAT.md):
# 436,458,752 tensor bytes; with the 4 KiB header, the directory and alignment the file is 436,507,904.
# A layer's input norm counts with its mixer, its post-attention norm with its MLP, the final norm with
# the head. A plain decode step reads every part but the MTP head once; recomputed by pq_bytes().
BYTES = [("output head (= input embedding, tied)", 143_036_416), ("linear attention, 18 layers", 108_354_816),
         ("full attention, 6 layers", 24_809_472), ("MLPs, 24 layers", 148_733_952),
         ("MTP draft head (read only when drafting)", 11_524_096)]
PQ_MODEL = ROOT / "models" / "pq" / "qwen35-0.8b-q4k-headq4k.pq"
GAP_PT = GAP_PX * 72 / DPI  # surface-coloured edge that opens the gap between stacked segments


def fig_step_breakdown(t: Theme) -> None:
    """Stacked horizontal bars: where a plain decode step and a verify pass spend their time."""
    fig, ax = new_fig(t, 8.6, 2.9)
    colors = [t.blue, t.orange, t.aqua]
    rows = [("plain decode step\n(1 token)", STEP_PLAIN), ("speculative verify pass\n(k=3: 4 positions at once)", STEP_VERIFY)]
    ax.set_yticks([0, 1], [r[0] for r in rows])
    style(ax, t, xlabel="milliseconds (Q4K model, 4 threads)", grid="x")
    ax.set_xlim(0, 260)
    ax.set_ylim(1.4, -0.4)  # inverted: plain step on top
    title(ax, t, "Where a token's time goes", "Reading 437 MB of weights dominates; verifying 4 drafted tokens costs only 1.24× one token")
    ax.legend(handles=[Patch(color=c, label=n) for (n, _), c in zip(STEP_PLAIN, colors)], loc="center right", fontsize=8.5)
    h = px_to_data(ax, BAR_PX, axis="y")
    for i, (_, parts) in enumerate(rows):
        left = 0.0
        for (_, ms), c in zip(parts, colors):
            ax.barh(i, ms, left=left, color=c, height=h, edgecolor=t.surface, linewidth=GAP_PT)
            if ms > 10:
                ax.text(left + ms / 2, i, f"{ms:.0f} ms", ha="center", va="center", color=on_fill(t, c), fontsize=9)
            left += ms
        ax.text(left + 2, i, f"{left:.0f} ms total", va="center", color=t.ink, fontsize=9)
    save(fig, t, "step_breakdown")


def fig_bandwidth(t: Theme) -> None:
    """Bars: effective weight-streaming speed of each kernel against the DRAM ceiling."""
    fig, ax = new_fig(t, 7.5, 3.6)
    x = list(range(len(KERNEL_GBS)))
    ax.set_xticks(x, [f"{n}\n{thr} thr" for n, _, thr in KERNEL_GBS])
    style(ax, t, ylabel="GB/s of weights streamed")
    ax.set_xlim(*slot_lim(0, len(x) - 1))
    ax.set_ylim(0, 5)
    title(ax, t, "Matrix kernels run at the memory wall",
          "Cortex-A72, best thread count per format; one core already saturates DRAM, so decode speed = bandwidth ÷ bytes per token")
    ax.bar(x, [v for _, v, _ in KERNEL_GBS], color=t.blue, width=px_to_data(ax, BAR_PX))
    for i, (_, v, _) in enumerate(KERNEL_GBS):
        ax.text(i, v + 0.06, f"{v:.2f}", ha="center", color=t.ink, fontsize=9)
    ax.axhline(MEMBAND, color=t.axis, linewidth=1, linestyle=(0, (4, 3)), zorder=0.8)  # behind the bars
    ax.text(len(x) - 0.5, MEMBAND + 0.08, f"DRAM read ceiling {MEMBAND:.1f} GB/s (memband)", ha="right", color=t.ink2, fontsize=8.5)
    save(fig, t, "kernel_bandwidth")


def fig_quant(t: Theme) -> None:
    """Two small multiples: weight error (bars from zero) and perplexity (dots: no zero baseline)."""
    fig, (a1, a2) = new_fig(t, 8.4, 3.4, ncols=2)
    x = list(range(len(QUANT)))
    for ax in (a1, a2):
        ax.set_xticks(x, [q[0] for q in QUANT])
        ax.set_xlim(*slot_lim(0, len(x) - 1))
    style(a1, t, ylabel="relative RMS weight error (%)")
    a1.set_ylim(0, 10.5)
    style(a2, t, ylabel="perplexity, wikitext-2 (lower is better)")
    a2.set_ylim(17, 22.5)
    rect = fig_title(fig, t, "Quantization formats: the same bits, very different quality",
                     "Q4K's asymmetric 32-column groups with 6-bit sub-scales keep Q4_K_M quality; symmetric Q4 loses 17% like llama.cpp's Q4_0")
    a1.bar(x, [q[1] for q in QUANT], color=t.blue, width=px_to_data(a1, BAR_PX, rect=rect))
    for i, q in enumerate(QUANT):
        a1.text(i, q[1] + 0.2, f"{q[1]:.1f}%" if q[1] else "exact", ha="center", color=t.ink, fontsize=9)
    a2.plot(x, [q[2] for q in QUANT], "o", ms=8.4, color=t.blue, **ring(t))
    for i, q in enumerate(QUANT):
        a2.annotate(f"{q[2]:.2f}", (i, q[2]), xytext=(0, 8), textcoords="offset points", ha="center", color=t.ink, fontsize=9)
    save(fig, t, "quantization", rect=rect)


def fig_acceptance(t: Theme) -> None:
    """Two small multiples: acceptance per draft position, and the semantics controls."""
    fig, (a1, a2) = new_fig(t, 8.6, 3.4, ncols=2, gridspec_kw={"width_ratios": [4, 3]})
    for ax, rows in ((a1, ACCEPT), (a2, CONTROLS)):
        ax.set_xticks(range(len(rows)), [r[0] for r in rows])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(*slot_lim(0, len(rows) - 1))
        ax.set_ylim(0, 105)
    style(a1, t, ylabel="drafts accepted by the full model (%)")
    style(a2, t, ylabel="1st-draft acceptance (%)")
    rect = fig_title(fig, t, "The built-in MTP draft head is right 87% of the time — and we proved we wired it correctly",
                     "651 positions of prose, code, Q&A and Chinese; later drafts are more often right because accepted text is predictable text")
    a2.legend(handles=[Patch(color=t.blue, label="our implementation"), Patch(color=t.orange, label="deliberately wrong control")],
              loc="upper right")
    a1.bar(range(len(ACCEPT)), [a[1] * 100 for a in ACCEPT], color=t.blue, width=px_to_data(a1, BAR_PX, rect=rect))
    for i, a in enumerate(ACCEPT):
        a1.text(i, a[1] * 100 + 1.5, f"{a[1] * 100:.0f}%", ha="center", color=t.ink, fontsize=9)
    a2.bar(range(len(CONTROLS)), [c[1] * 100 for c in CONTROLS], color=[t.blue, t.orange, t.orange],
           width=px_to_data(a2, BAR_PX, rect=rect))
    for i, c in enumerate(CONTROLS):
        a2.text(i, c[1] * 100 + 1.5, f"{c[1] * 100:.1f}%", ha="center", color=t.ink, fontsize=9)
    save(fig, t, "mtp_acceptance", rect=rect)


def fig_os(t: Theme) -> None:
    """Grouped bars: the custom-kernel experiment changed nothing."""
    fig, ax = new_fig(t, 8.0, 3.4)
    x = list(range(len(OS_EXP)))
    ax.set_xticks(x, [o[0] for o in OS_EXP], fontsize=8.5)
    style(ax, t, ylabel="tokens / s")
    ax.set_xlim(*slot_lim(0, len(x) - 1, half=0.36))
    ax.set_ylim(0, 19)
    title(ax, t, "What did not help: a custom split-core kernel",
          "Same binary and model; three one-time boots of 6.18.50-pinfer — all within 1%, because the engine is DRAM-bound")
    w = px_to_data(ax, BAR_PX)
    off = px_to_data(ax, (BAR_PX + GAP_PX) / 2)
    ax.bar([i - off for i in x], [o[1] for o in OS_EXP], width=w, color=t.blue, label="plain decode")
    ax.bar([i + off for i in x], [o[2] for o in OS_EXP], width=w, color=t.orange, label="speculative (k=3)")
    for i, o in enumerate(OS_EXP):
        ax.text(i - off, o[1] + 0.3, f"{o[1]:.2f}", ha="center", color=t.ink, fontsize=8.5)
        ax.text(i + off, o[2] + 0.3, f"{o[2]:.2f}", ha="center", color=t.ink, fontsize=8.5)
    ax.legend(loc="upper right", ncol=2)
    save(fig, t, "os_experiment")


def fig_spec_schematic(t: Theme) -> None:
    """Block schematic of one speculative round."""
    fig, ax = new_fig(t, 8.6, 3.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=on_fill(t, fc), fontsize=8.6)

    def arrow(x1: float, x2: float, y: float, text: str = "") -> None:
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=12, color=t.muted, linewidth=1.4))
        if text:
            ax.text((x1 + x2) / 2, y + 3.2, text, ha="center", color=t.ink2, fontsize=7.6)

    box(1, 10, 22, 10, "1. Draft\nMTP head guesses 3 tokens\n(1 layer, ~27 ms, draft vocab 32K)", t.orange)
    arrow(23.6, 29.4, 15, "3 drafts")
    box(30, 10, 30, 10, "2. Verify in ONE weight pass\nfull model scores 4 positions\n(437 MB read once, ~160 ms)", t.blue)
    arrow(60.6, 66.4, 15, "verdicts")
    box(67, 10, 32, 10, "3. Accept the matching prefix\n+ 1 token of the full model's own\n→ 3.3 tokens per round on average", t.aqua)
    ax.text(50, 3.5, "Output is token-for-token identical to plain greedy decoding (every kept token is the full model's own choice); only the speed changes.",
            ha="center", color=t.ink2, fontsize=8.6)
    ax.text(0, 27.5, "One round of MTP speculative decoding", color=t.ink, fontsize=11.5, fontweight="semibold")
    save(fig, t, "speculative_schematic")


def fig_bytes(t: Theme) -> None:
    """One stacked bar: the Q4K model's bytes by part, i.e. what every decode step has to stream."""
    fig, ax = new_fig(t, 8.0, 3.0)
    mb = [b / 1e6 for _, b in BYTES]
    total, plain = sum(mb), sum(mb[:-1])
    colors = [t.blue, t.orange, t.aqua, t.yellow, t.magenta]  # slot order = stack order (validated adjacent pairs)
    style(ax, t, xlabel="MB (10⁶ bytes)", grid="x")
    ax.set_xlim(0, 460)
    ax.set_ylim(-0.5, 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    title(ax, t, "Where a token's bytes go",
          "Tensor bytes of the 437 MB Q4K model by part (from the .pq tensor directory); decode speed = bandwidth ÷ bytes read per token")
    ax.legend(handles=[Patch(color=c, label=f"{name}: {v:.1f} MB") for (name, _), v, c in zip(BYTES, mb, colors)],
              loc="upper left", bbox_to_anchor=(0, -0.42), ncol=2, columnspacing=2.4)
    h = px_to_data(ax, BAR_PX, axis="y")
    per_px = px_to_data(ax, 1)
    renderer = fig.canvas.get_renderer()
    left = 0.0
    for i, (v, c) in enumerate(zip(mb, colors)):
        ax.barh(0, v, left=left, height=h, color=c, edgecolor=t.surface, linewidth=GAP_PT)
        label = ax.text(left + v / 2, 0, f"{100 * v / total:.1f}%", ha="center", va="center", color=on_fill(t, c), fontsize=9)
        if label.get_window_extent(renderer).width + 12 > v / per_px:  # does not fit with 6 px a side
            label.remove()  # an interior segment leaves its value to the legend
            if i == len(mb) - 1:  # the last segment has a free end: label it outside
                ax.text(left + v + 4 * per_px, 0, f"{100 * v / total:.1f}%", va="center", color=t.ink, fontsize=9)
        left += v
    y0, y1 = h / 2 + px_to_data(ax, 7, axis="y"), h / 2 + px_to_data(ax, 14, axis="y")
    ax.plot([0, 0, plain, plain], [y0, y1, y1, y0], color=t.muted, linewidth=1, solid_joinstyle="miter")
    ax.text(plain / 2, y1 + px_to_data(ax, 4, axis="y"), f"every plain decode step reads these {plain:.1f} MB",
            ha="center", va="bottom", color=t.ink2, fontsize=8.5)
    save(fig, t, "bytes_per_token")


def _part(name: str, full_mask: int) -> int:
    """Index into BYTES of the part a .pq tensor belongs to."""
    if name.startswith("mtp."):
        return 4
    if not name.startswith("layers."):
        return 0  # embed_tokens (= the tied output head) and the final norm
    _, layer, block = name.split(".")[:3]
    if block in ("mlp", "post_attention_layernorm"):
        return 3
    return 2 if full_mask >> int(layer) & 1 else 1  # self_attn / linear_attn and the layer's input norm


def pq_bytes(path: Path) -> list[int]:
    """Bytes per BYTES part, summed over a .pq tensor directory (header and 128-byte entries: engine/FORMAT.md)."""
    with path.open("rb") as f:
        header = f.read(4096)
        n_tensors, dir_offset = struct.unpack_from("<IQ", header, 12)
        full_mask = struct.unpack_from("<I", header, 88)[0]
        f.seek(dir_offset)
        directory = f.read(128 * n_tensors)
    sums = [0] * len(BYTES)
    for i in range(n_tensors):
        entry = directory[128 * i:128 * (i + 1)]
        name = entry[:80].split(b"\0", 1)[0].decode()
        sums[_part(name, full_mask)] += struct.unpack_from("<Q", entry, 112)[0]
    return sums


def check_bytes() -> None:
    """Warn if the model on disk no longer matches the recorded BYTES (skipped without the model)."""
    if not PQ_MODEL.exists():
        logger.info("bytes_per_token: %s not present, using the recorded sizes", PQ_MODEL.name)
        return
    found, recorded = pq_bytes(PQ_MODEL), [b for _, b in BYTES]
    if found != recorded:
        logger.warning("bytes_per_token: %s gives %s, the figure records %s", PQ_MODEL.name, found, recorded)
    else:
        logger.info("bytes_per_token: recorded sizes match %s", PQ_MODEL.name)


FIGURES = (fig_step_breakdown, fig_bandwidth, fig_quant, fig_acceptance, fig_os, fig_spec_schematic, fig_bytes)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_bytes()
    for t in THEMES:
        setup(t)
        for make in FIGURES:
            make(t)


if __name__ == "__main__":
    main()
