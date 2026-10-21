"""Render the result figures for the README into docs/figures/*.png.

Numbers are the verified ones from notes.md ("Scoreboard", "Context length") with their
result files in bench/results/; each dataset below names its source so a figure can be
traced back. Colors follow the validated dataviz palette (light surface #fcfcfb):
series 1 blue #2a78d6, 2 orange #eb6834, 3 aqua #1baf7a; ink #0b0b0b / #52514e / #898781.

    uv run --no-project --with matplotlib python tools/make_figures.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FONT = {"family": ["Segoe UI", "DejaVu Sans", "sans-serif"]}

# ---------------------------------------------------------------- data (Pi 4, 4 threads, 64+64 tokens)
# (label, plain tok/s, speculative tok/s or None, perplexity or None, source)
ENGINES = [
    ("pinfer Q4K (ours)", 7.88, 16.49, 18.50, "pinfer-final2-q4k_20261022_023950"),
    ("pinfer Q8 (ours)", 4.22, 8.73, 17.96, "pinfer-fast-q8-headq8_20261021_235144"),
    ("MNN 3.6 int4", 6.67, None, None, "mnn-qwen35-0.8b-int4_20261021_221038"),
    ("llama.cpp Q4_K_M", 5.41, None, 18.59, "llamacpp-qwen35-0.8b-Q4_K_M_20261021_213929"),
    ("llama.cpp Q4_0", 5.59, None, 21.25, "llamacpp-qwen35-0.8b-Q4_0_20261021_213343"),
    ("llama.cpp Q8_0", 4.12, None, 17.97, "llamacpp-qwen35-0.8b-Q8_0_20261021_214506"),
    ("ONNX Runtime GenAI int4", 3.10, None, None, "ort-qwen35-0.8b-int4-acc4_20261021_222410"),
]
PREFILL = {"pinfer Q4K (ours)": 37.4, "MNN 3.6 int4": 30.9, "llama.cpp Q8_0": 17.1,
           "llama.cpp Q4_0": 15.7, "llama.cpp Q4_K_M": 14.8, "ONNX Runtime GenAI int4": 9.9}
# Optimization steps in order (plain, speculative); 2026-10-21/22
PROGRESS = [
    ("scalar C\nbf16", 1.55, None),
    ("NEON\nbf16", 2.27, 4.13),
    ("Q4K\nweights", 6.83, 11.57),
    ("bit-exact kernels\n+ NEON recurrence", 7.26, 13.35),
    ("vectorized\nper-token ops", 7.88, 16.49),
]
# Context length sweep (plain, spec k=3, prefill) — pre-optimization, mixed binaries for 2048
CONTEXT = [(64, 7.88, 16.49, 37.4), (512, 6.20, 9.04, 21.6), (2048, 4.12, 2.39, 11.6),
           (4096, 5.56, 5.69, 18.1)]
# Speculative k sweep on the Q4K default (pinfer-final2-q4k): k, tokens per pass, tok/s
KSWEEP = [(0, 1.00, 7.88), (2, 2.63, 14.51), (3, 3.32, 16.49), (4, 3.94, 15.65)]


def style(ax: plt.Axes, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_xlabel(xlabel, color=INK2)


def new_fig(w: float = 8.0, h: float = 4.2) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    logger.info("wrote %s", OUT / name)


def fig_decode() -> None:
    """Grouped horizontal bars: plain vs speculative decode, fastest at the top."""
    rows = sorted(ENGINES, key=lambda r: r[1])
    fig, ax = new_fig(8.0, 4.6)
    y = range(len(rows))
    h = 0.36
    ax.barh([i + h / 2 for i in y], [r[1] for r in rows], height=h, color=BLUE, label="plain greedy")
    spec = [r[2] or 0 for r in rows]
    ax.barh([i - h / 2 for i in y], spec, height=h, color=ORANGE, label="speculative (MTP, k=3)")
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.2, i + h / 2, f"{r[1]:.1f}", va="center", color=INK, fontsize=9)
        if r[2]:
            ax.text(r[2] + 0.2, i - h / 2, f"{r[2]:.1f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(list(y), [r[0] for r in rows])
    style(ax, xlabel="tokens / s (higher is better)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 19)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
    ax.set_title("Qwen3.5-0.8B decode speed on a Raspberry Pi 4 (4 threads)", color=INK, loc="left")
    save(fig, "decode_speed.png")


def fig_speed_quality() -> None:
    """Scatter: perplexity (lower is better) against plain decode speed."""
    fig, ax = new_fig(7.0, 4.4)
    # Hand-placed label offsets where points sit on top of each other (Q8 pair at ~17.96).
    offsets = {"pinfer Q8 (ours)": (8, 6), "llama.cpp Q8_0": (8, -12), "pinfer Q4K (ours)": (8, -4),
               "pinfer Q8 (ours) + speculative": (8, 2), "pinfer Q4K (ours) + speculative": (8, -4)}
    # Baselines first, ours last, so an overlapping pinfer point (Q8 vs Q8_0) stays visible.
    for label, plain, spec, ppl, _ in sorted(ENGINES, key=lambda r: r[0].startswith("pinfer")):
        if ppl is None:
            continue
        ours = label.startswith("pinfer")
        ax.scatter(ppl, plain, s=70, color=BLUE if ours else ORANGE, zorder=4 if ours else 3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(label, (ppl, plain), xytext=offsets.get(label, (6, 4)), textcoords="offset points",
                    fontsize=8.5, color=INK2)
        if spec:
            name = f"{label} + speculative"
            ax.scatter(ppl, spec, s=70, color=BLUE, marker="^", zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.annotate(name, (ppl, spec), xytext=offsets.get(name, (6, 4)),
                        textcoords="offset points", fontsize=8.5, color=INK2)
    ax.scatter([], [], color=BLUE, label="pinfer (ours)")
    ax.scatter([], [], color=ORANGE, label="llama.cpp")
    ax.scatter([], [], color=BLUE, marker="^", label="pinfer, speculative")
    style(ax, ylabel="decode tokens / s", xlabel="perplexity on wikitext-2 (lower is better)")
    ax.set_xlim(17.5, 22.5)
    ax.set_ylim(0, 18.5)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2)
    ax.set_title("Speed vs quality at 4- and 8-bit weights", color=INK, loc="left")
    save(fig, "speed_vs_quality.png")


def fig_progress() -> None:
    """Line: decode speed after each optimization step."""
    fig, ax = new_fig(8.0, 4.2)
    x = list(range(len(PROGRESS)))
    plain = [p[1] for p in PROGRESS]
    ax.plot(x, plain, color=BLUE, linewidth=2, marker="o", markersize=7, label="plain greedy")
    sx = [i for i, p in enumerate(PROGRESS) if p[2]]
    ax.plot(sx, [PROGRESS[i][2] for i in sx], color=ORANGE, linewidth=2, marker="o", markersize=7,
            label="speculative (k=3)")
    for i, p in enumerate(PROGRESS):
        ax.annotate(f"{p[1]:.1f}", (i, p[1]), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
        if p[2]:
            ax.annotate(f"{p[2]:.1f}", (i, p[2]), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
    ax.axhline(5.41, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(PROGRESS) - 1.02, 5.41 + 0.3, "llama.cpp Q4_K_M 5.4", ha="right", color=MUTED, fontsize=8.5)
    ax.set_xticks(x, [p[0] for p in PROGRESS], fontsize=8.5)
    style(ax, ylabel="tokens / s")
    ax.set_ylim(0, 19)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    ax.set_title("pinfer decode speed after each optimization step (Q4K unless noted)", color=INK, loc="left")
    save(fig, "progress.png")


def fig_context() -> None:
    """Lines: speed against context length (log-2 x axis)."""
    fig, ax = new_fig(7.5, 4.2)
    xs = [c[0] for c in CONTEXT]
    end_offsets = [(8, -10), (8, 4), (8, -3)]  # plain below, spec above: 5.6 vs 5.7 at 4096
    for idx, (label, color) in enumerate([("plain decode", BLUE), ("speculative decode (k=3)", ORANGE),
                                          ("prefill", AQUA)]):
        ys = [c[idx + 1] for c in CONTEXT]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, label=label)
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=end_offsets[idx], textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(v) for v in xs])
    style(ax, ylabel="tokens / s", xlabel="context length (tokens in the prompt)")
    ax.set_ylim(0, 40)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2)
    ax.set_title("Speed vs context length — before the context-window work", color=INK, loc="left")
    save(fig, "context_length.png")


def fig_kspeed() -> None:
    """Two small multiples (not a dual axis): tokens per pass and tok/s against k."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ks = [k for k, _, _ in KSWEEP]
    a1.bar(ks, [t for _, t, _ in KSWEEP], color=BLUE, width=0.6)
    for k, t, _ in KSWEEP:
        a1.text(k, t + 0.08, f"{t:.2f}", ha="center", color=INK, fontsize=9)
    a1.set_xticks(ks, ["plain", "k=2", "k=3", "k=4"])
    style(a1, ylabel="tokens accepted per weight pass")
    a1.set_ylim(0, 4.6)
    a2.bar(ks, [s for _, _, s in KSWEEP], color=ORANGE, width=0.6)
    for k, _, s in KSWEEP:
        a2.text(k, s + 0.3, f"{s:.1f}", ha="center", color=INK, fontsize=9)
    a2.set_xticks(ks, ["plain", "k=2", "k=3", "k=4"])
    style(a2, ylabel="decode tokens / s")
    a2.set_ylim(0, 19)
    fig.suptitle("MTP speculative decoding: drafts per round (k) — Q4K model", color=INK, x=0.02, ha="left")
    save(fig, "speculative_k.png")


def fig_prefill() -> None:
    fig, ax = new_fig(7.0, 3.4)
    rows = sorted(PREFILL.items(), key=lambda kv: kv[1])
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[BLUE if r[0].startswith("pinfer") else ORANGE for r in rows], height=0.55)
    for i, (label, v) in enumerate(rows):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", color=INK, fontsize=9)
    style(ax, xlabel="prompt tokens / s (64-token prompt)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 42)
    ax.legend(handles=[Patch(color=BLUE, label="pinfer (ours)"), Patch(color=ORANGE, label="standard runtimes")],
              frameon=False, loc="lower right", labelcolor=INK2)
    ax.set_title("Prefill (prompt processing) speed", color=INK, loc="left")
    save(fig, "prefill_speed.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plt.rcParams["font.family"] = FONT["family"]
    plt.rcParams["text.color"] = INK
    plt.rcParams["axes.titleweight"] = "semibold"
    fig_decode()
    fig_speed_quality()
    fig_progress()
    fig_context()
    fig_kspeed()
    fig_prefill()


if __name__ == "__main__":
    main()
