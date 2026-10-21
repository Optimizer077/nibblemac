"""'How it works' figures for the README (docs/figures/*.png), from the logged measurements.

Sources: engine/CORE.md (per-step profile), engine/bench/KERNELS.md (kernel GB/s, quantization
error), engine/SPEC.md (MTP acceptance), notes.md (memband, kernel experiment), bench/results/.

    uv run --no-project --with matplotlib python tools/make_figures2.py
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from figstyle import AQUA, AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, YELLOW, new_fig, save, setup, style, title

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


def fig_step_breakdown() -> None:
    """Stacked horizontal bars: where a plain decode step and a verify pass spend their time."""
    fig, ax = new_fig(8.6, 3.4)
    colors = [BLUE, ORANGE, AQUA]
    rows = [("plain decode step\n(1 token)", STEP_PLAIN), ("speculative verify pass\n(k=3: 4 positions at once)", STEP_VERIFY)]
    for i, (label, parts) in enumerate(rows):
        left = 0.0
        for (name, ms), c in zip(parts, colors):
            ax.barh(i, ms, left=left, color=c, height=0.5, edgecolor=SURFACE, linewidth=2)
            if ms > 10:
                ax.text(left + ms / 2, i, f"{ms:.0f} ms", ha="center", va="center", color=SURFACE, fontsize=9)
            left += ms
        ax.text(left + 2, i, f"{left:.0f} ms total", va="center", color=INK, fontsize=9)
    ax.set_yticks([0, 1], [r[0] for r in rows])
    ax.invert_yaxis()
    style(ax, xlabel="milliseconds (Q4K model, 4 threads)", grid="x")
    ax.set_xlim(0, 260)
    ax.legend(handles=[Patch(color=c, label=n) for (n, _), c in zip(STEP_PLAIN, colors)],
              frameon=False, loc="center right", labelcolor=INK2, fontsize=8.5)
    title(ax, "Where a token's time goes", "Reading 437 MB of weights dominates; verifying 4 drafted tokens costs only 1.24× one token")
    save(fig, "step_breakdown.png")


def fig_bandwidth() -> None:
    """Bars: effective weight-streaming speed of each kernel against the DRAM ceiling."""
    fig, ax = new_fig(7.5, 3.6)
    names = [k[0] for k in KERNEL_GBS]
    vals = [k[1] for k in KERNEL_GBS]
    ax.bar(names, vals, color=BLUE, width=0.6)
    for i, (n, v, t) in enumerate(KERNEL_GBS):
        ax.text(i, v + 0.06, f"{v:.2f}", ha="center", color=INK, fontsize=9)
        ax.text(i, 0.15, f"{t} thr", ha="center", color=SURFACE, fontsize=8.5)
    ax.axhline(MEMBAND, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(names) - 0.5, MEMBAND + 0.08, f"DRAM read ceiling {MEMBAND:.1f} GB/s (memband)", ha="right", color=MUTED, fontsize=8.5)
    style(ax, ylabel="GB/s of weights streamed")
    ax.set_ylim(0, 5)
    title(ax, "Matrix kernels run at the memory wall",
          "Cortex-A72, best thread count per format; one core already saturates DRAM, so decode speed = bandwidth ÷ bytes per token")
    save(fig, "kernel_bandwidth.png")


def fig_quant() -> None:
    """Two small multiples: weight error and perplexity per format."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    names = [q[0] for q in QUANT]
    a1.bar(names, [q[1] for q in QUANT], color=BLUE, width=0.6)
    for i, q in enumerate(QUANT):
        a1.text(i, q[1] + 0.2, f"{q[1]:.1f}%" if q[1] else "exact", ha="center", color=INK, fontsize=9)
    style(a1, ylabel="relative RMS weight error (%)")
    a1.set_ylim(0, 10.5)
    a2.bar(names, [q[2] for q in QUANT], color=ORANGE, width=0.6)
    for i, q in enumerate(QUANT):
        a2.text(i, q[2] + 0.1, f"{q[2]:.2f}", ha="center", color=INK, fontsize=9)
    style(a2, ylabel="perplexity, wikitext-2 (lower is better)")
    a2.set_ylim(17, 22.5)
    fig.suptitle("Quantization formats: the same bits, very different quality", color=INK, x=0.02, ha="left", fontweight="semibold")
    fig.text(0.02, 0.90, "Q4K's asymmetric 32-column groups with 6-bit sub-scales keep Q4_K_M quality; symmetric Q4 loses 17% like llama.cpp's Q4_0",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "quantization.png")


def fig_acceptance() -> None:
    """Two small multiples: acceptance per draft position, and the semantics controls."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=160, gridspec_kw={"width_ratios": [4, 3]})
    fig.patch.set_facecolor(SURFACE)
    a1.bar([a[0] for a in ACCEPT], [a[1] * 100 for a in ACCEPT], color=BLUE, width=0.6)
    for i, a in enumerate(ACCEPT):
        a1.text(i, a[1] * 100 + 1.5, f"{a[1] * 100:.0f}%", ha="center", color=INK, fontsize=9)
    a1.tick_params(axis="x", labelsize=8)
    style(a1, ylabel="drafts accepted by the full model (%)")
    a1.set_ylim(0, 105)
    a2.bar([c[0] for c in CONTROLS], [c[1] * 100 for c in CONTROLS], color=[BLUE, ORANGE, ORANGE], width=0.6)
    for i, c in enumerate(CONTROLS):
        a2.text(i, c[1] * 100 + 1.5, f"{c[1] * 100:.1f}%", ha="center", color=INK, fontsize=9)
    a2.tick_params(axis="x", labelsize=8)
    style(a2, ylabel="1st-draft acceptance (%)")
    a2.set_ylim(0, 105)
    a2.legend(handles=[Patch(color=BLUE, label="our implementation"), Patch(color=ORANGE, label="deliberately wrong control")],
              frameon=False, loc="upper right", labelcolor=INK2)
    fig.suptitle("The built-in MTP draft head is right 87% of the time — and we proved we wired it correctly",
                 color=INK, x=0.02, ha="left", fontweight="semibold")
    fig.text(0.02, 0.90, "651 positions of prose, code, Q&A and Chinese; later drafts are more often right because accepted text is predictable text",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "mtp_acceptance.png")


def fig_os() -> None:
    """Grouped bars: the custom-kernel experiment changed nothing."""
    fig, ax = new_fig(8.0, 3.4)
    x = range(len(OS_EXP))
    w = 0.36
    ax.bar([i - w / 2 for i in x], [o[1] for o in OS_EXP], width=w, color=BLUE, label="plain decode")
    ax.bar([i + w / 2 for i in x], [o[2] for o in OS_EXP], width=w, color=ORANGE, label="speculative (k=3)")
    for i, o in enumerate(OS_EXP):
        ax.text(i - w / 2, o[1] + 0.3, f"{o[1]:.2f}", ha="center", color=INK, fontsize=8.5)
        ax.text(i + w / 2, o[2] + 0.3, f"{o[2]:.2f}", ha="center", color=INK, fontsize=8.5)
    ax.set_xticks(list(x), [o[0] for o in OS_EXP], fontsize=8.5)
    style(ax, ylabel="tokens / s")
    ax.set_ylim(0, 19)
    ax.legend(frameon=False, loc="upper right", ncol=2, labelcolor=INK2)
    title(ax, "What did not help: a custom split-core kernel",
          "Same binary and model; three one-time boots of 6.18.50-pinfer — all within 1%, because the engine is DRAM-bound")
    save(fig, "os_experiment.png")


def fig_spec_schematic() -> None:
    """Block schematic of one speculative round."""
    fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, tc: str = SURFACE) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=8.6)

    def arrow(x1: float, x2: float, y: float, text: str = "") -> None:
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=12, color=AXIS, linewidth=1.4))
        if text:
            ax.text((x1 + x2) / 2, y + 3.2, text, ha="center", color=MUTED, fontsize=7.6)

    box(1, 10, 22, 10, "1. Draft\nMTP head guesses 3 tokens\n(1 layer, ~27 ms, draft vocab 32K)", ORANGE)
    arrow(23.6, 29.4, 15, "3 drafts")
    box(30, 10, 30, 10, "2. Verify in ONE weight pass\nfull model scores 4 positions\n(437 MB read once, ~160 ms)", BLUE)
    arrow(60.6, 66.4, 15, "verdicts")
    box(67, 10, 32, 10, "3. Accept the matching prefix\n+ 1 token of the full model's own\n→ 3.3 tokens per round on average", AQUA)
    ax.text(50, 3.5, "Output is token-for-token identical to plain greedy decoding (every kept token is the full model's own choice); only the speed changes.",
            ha="center", color=INK2, fontsize=8.6)
    ax.text(0, 27.5, "One round of MTP speculative decoding", color=INK, fontsize=11.5, fontweight="semibold")
    save(fig, "speculative_schematic.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup()
    fig_step_breakdown()
    fig_bandwidth()
    fig_quant()
    fig_acceptance()
    fig_os()
    fig_spec_schematic()


if __name__ == "__main__":
    main()
