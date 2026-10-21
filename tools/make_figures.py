"""Render the result figures for the README into docs/figures/*.png (light) and *-dark.png.

Numbers are the verified ones from notes.md ("Scoreboard", "Context length") with their
result files in bench/results/; each dataset below names its source so a figure can be
traced back. Style, palette and the two themes: tools/figstyle.py.

    uv run --no-project --with matplotlib python tools/make_figures.py
"""
from __future__ import annotations

import logging

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figstyle import (
    BAR_PX,
    GAP_PX,
    LINE_PT,
    THEMES,
    Theme,
    fig_title,
    line_key,
    new_fig,
    px_to_data,
    ring,
    save,
    setup,
    slot_lim,
    style,
    title,
)

logger = logging.getLogger(__name__)

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
# Context length sweep, Q4K, wikitext prompt of N tokens then 32 generated (engine/CONTEXT.md Summary).
# After: final binary (ctx, plain fp32 KV, plain fp16 KV or None, prefill) — pinfer-ctx7-*, pinfer-final-kv*,
# 8192 decode = mean of 3 decode-only runs from a saved state (CONTEXT.md §3d).
CONTEXT = [(64, 7.88, 7.90, 37.3), (512, 7.69, 7.67, 36.8), (2048, 7.13, None, 32.3),
           (4096, 6.38, 6.65, 28.0), (8192, 5.30, 5.72, 22.0)]
# Before: the vectorized binary before the context work (ctx, plain, prefill) — pinfer-final2-q4k (64),
# pinfer-ctx2048-profile-q4k (2048), pinfer-ctx4096-q4k (4096); 512 was only run with an older binary.
CONTEXT_BEFORE = [(64, 7.88, 37.4), (2048, 6.57, 25.7), (4096, 5.56, 18.1)]
# llama.cpp Q4_K_M token generation at the same KV depth (f16 KV, its default) — llamacpp-ctx-d*
LLAMA_CTX = [(64, 5.48), (512, 5.23), (2048, 4.32), (4096, 3.43)]
# Speculative k sweep on the Q4K default (pinfer-final2-q4k): k, tokens per pass, tok/s
KSWEEP = [(0, 1.00, 7.88), (2, 2.63, 14.51), (3, 3.32, 16.49), (4, 3.94, 15.65)]


def fig_decode(t: Theme) -> None:
    """Grouped horizontal bars: plain vs speculative decode, fastest at the top."""
    rows = sorted(ENGINES, key=lambda r: r[1])
    fig, ax = new_fig(t, 8.0, 4.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    style(ax, t, xlabel="tokens / s (higher is better)", grid="x")
    ax.set_xlim(0, 19)
    ax.set_ylim(*slot_lim(0, len(rows) - 1, half=0.36))
    title(ax, t, "Decode speed: pinfer vs standard runtimes", "Qwen3.5-0.8B, Raspberry Pi 4 (4 GB), best thread count per engine, 64-token prompt + 64 tokens")
    bar = 28  # px; a pair is two bars and a surface gap, a single bar sits on its label
    h = px_to_data(ax, bar, axis="y")
    off = px_to_data(ax, (bar + GAP_PX) / 2, axis="y")
    for i, (_, plain, spec, _, _) in enumerate(rows):
        y_plain = i + off if spec else i
        ax.barh(y_plain, plain, height=h, color=t.blue)
        ax.text(plain + 0.2, y_plain, f"{plain:.1f}", va="center", color=t.ink, fontsize=9)
        if spec:
            ax.barh(i - off, spec, height=h, color=t.orange)
            ax.text(spec + 0.2, i - off, f"{spec:.1f}", va="center", color=t.ink, fontsize=9)
    ax.legend(handles=[Patch(color=t.blue, label="plain greedy"), Patch(color=t.orange, label="speculative (MTP, k=3)")],
              loc="lower right")
    save(fig, t, "decode_speed")


def fig_speed_quality(t: Theme) -> None:
    """Scatter: perplexity (lower is better) against plain decode speed."""
    fig, ax = new_fig(t, 7.0, 4.4)
    # Hand-placed label offsets (points) where points crowd: the Q8 pair, the Q4K pair.
    offsets = {"pinfer Q8 (ours)": (8, -4), "llama.cpp Q8_0": (8, -16), "pinfer Q4K (ours)": (8, -4),
               "pinfer Q8 (ours) + speculative": (8, 2), "pinfer Q4K (ours) + speculative": (8, -4)}
    # pinfer Q8 (17.96, 4.22) and llama.cpp Q8_0 (17.97, 4.12) coincide at this scale: each is drawn
    # as one half of a disc at its own position (no ring, which would cut the other half), so both
    # colours stay visible.
    halves = {"pinfer Q8 (ours)": "left", "llama.cpp Q8_0": "right"}
    for label, plain, spec, ppl, _ in sorted(ENGINES, key=lambda r: r[0].startswith("pinfer")):
        if ppl is None:
            continue
        color = t.blue if label.startswith("pinfer") else t.orange
        edge = {"markeredgewidth": 0} if label in halves else ring(t)
        ax.plot(ppl, plain, "o", ms=8.4, color=color, fillstyle=halves.get(label, "full"),
                markerfacecoloralt="none", zorder=4, **edge)
        ax.annotate(label, (ppl, plain), xytext=offsets.get(label, (6, 4)), textcoords="offset points",
                    fontsize=8.5, color=t.ink2)
        if spec:
            name = f"{label} + speculative"
            ax.plot(ppl, spec, "^", ms=8.4, color=t.blue, zorder=3, **ring(t))
            ax.annotate(name, (ppl, spec), xytext=offsets.get(name, (6, 4)),
                        textcoords="offset points", fontsize=8.5, color=t.ink2)
    ax.scatter([], [], color=t.blue, label="pinfer (ours)")
    ax.scatter([], [], color=t.orange, label="llama.cpp")
    ax.scatter([], [], color=t.blue, marker="^", label="pinfer, speculative")
    style(ax, t, ylabel="decode tokens / s", xlabel="perplexity on wikitext-2 (lower is better)")
    ax.set_xlim(17.5, 22.5)
    ax.set_ylim(0, 18.5)
    ax.legend(loc="upper right")
    title(ax, t, "Speed vs quality", "Perplexity on wikitext-2 (8 × 512 tokens, llama-perplexity protocol); pinfer Q4K sits at Q4_K_M quality")
    save(fig, t, "speed_vs_quality")


def fig_progress(t: Theme) -> None:
    """Line: decode speed after each optimization step."""
    fig, ax = new_fig(t, 8.0, 4.2)
    x = list(range(len(PROGRESS)))
    plain = [p[1] for p in PROGRESS]
    ax.plot(x, plain, color=t.blue, linewidth=LINE_PT, marker="o", markersize=7, label="plain greedy", **ring(t))
    sx = [i for i, p in enumerate(PROGRESS) if p[2]]
    ax.plot(sx, [PROGRESS[i][2] for i in sx], color=t.orange, linewidth=LINE_PT, marker="o", markersize=7,
            label="speculative (k=3)", **ring(t))
    for i, p in enumerate(PROGRESS):
        ax.annotate(f"{p[1]:.1f}", (i, p[1]), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=9, color=t.ink)
        if p[2]:
            ax.annotate(f"{p[2]:.1f}", (i, p[2]), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, color=t.ink)
    ax.axhline(5.41, color=t.axis, linewidth=1, linestyle=(0, (4, 3)), zorder=1.5)  # behind the series
    ax.text(len(PROGRESS) - 1.02, 5.41 + 0.3, "llama.cpp Q4_K_M 5.4", ha="right", color=t.ink2, fontsize=8.5)
    ax.set_xticks(x, [p[0] for p in PROGRESS], fontsize=8.5)
    style(ax, t, ylabel="tokens / s")
    ax.set_ylim(0, 19)
    ax.legend(handles=[line_key(t.blue, "plain greedy"), line_key(t.orange, "speculative (k=3)")], loc="upper left")
    title(ax, t, "How the speed was built up", "Each step verified against the fp32 reference (per-layer error ≤ 3e-6) and, for speculation, token-identical to plain greedy")
    save(fig, t, "progress")


def _ctx_line(ax, t: Theme, pts: list, color: str, dashed: bool = False, end_dy: float = 0) -> None:
    """One series of the context figure, its last value labelled to the right of the last marker."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, color=color, linewidth=LINE_PT, marker="o", markersize=6.5,
            linestyle=(0, (4, 3)) if dashed else "-", **ring(t))
    ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(9, end_dy), textcoords="offset points",
                va="center", fontsize=9, color=t.ink)


def fig_context(t: Theme) -> None:
    """Two panels against context length (log-2 x axis): plain decode, prefill; before vs after."""
    fig, (a1, a2) = new_fig(t, 10.0, 4.3, ncols=2, gridspec_kw={"width_ratios": [3, 2]})
    xs = [c[0] for c in CONTEXT]
    before = t.muted  # the old binary is context, not a series to compare hues against
    _ctx_line(a1, t, [(c[0], c[1]) for c in CONTEXT], t.blue, end_dy=-7)
    _ctx_line(a1, t, [(c[0], c[2]) for c in CONTEXT if c[2]], t.aqua, end_dy=7)
    _ctx_line(a1, t, [(c[0], c[1]) for c in CONTEXT_BEFORE], before, dashed=True)
    _ctx_line(a1, t, LLAMA_CTX, t.orange)
    a1.legend(handles=[line_key(t.blue, "pinfer, fp32 KV cache (default, exact)"),
                       line_key(t.aqua, "pinfer, fp16 KV cache (opt-in)"),
                       Line2D([], [], color=before, linewidth=LINE_PT, linestyle=(0, (4, 3)), marker="o",
                              markersize=7, label="pinfer before"),
                       line_key(t.orange, "llama.cpp Q4_K_M")], loc="lower left")
    style(a1, t, ylabel="decode tokens / s (plain greedy)", xlabel="context length (tokens)")
    a1.set_ylim(0, 9)
    a1.set_title("Decode", color=t.ink, loc="left", fontsize=10.5)
    _ctx_line(a2, t, [(c[0], c[3]) for c in CONTEXT], t.blue)
    _ctx_line(a2, t, [(c[0], c[2]) for c in CONTEXT_BEFORE], before, dashed=True)
    style(a2, t, ylabel="prefill tokens / s", xlabel="context length (tokens)")
    a2.set_ylim(0, 42)
    a2.set_title("Prefill (whole prompt)", color=t.ink, loc="left", fontsize=10.5)
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(v) for v in xs])
        ax.set_xlim(48, 8192 * 1.6)
    rect = fig_title(fig, t, "Speed vs context length, before and after the attention rewrite",
                     "Qwen3.5-0.8B Q4K, Pi 4, 4 threads: an N-token wikitext prompt, then 32 generated tokens. "
                     "At 4096 tokens decode keeps 81% (fp32) / 84% (fp16) of its short-context speed")
    save(fig, t, "context_length", rect=rect)


def fig_kspeed(t: Theme) -> None:
    """Two small multiples (not a dual axis): tokens per pass and tok/s against k."""
    fig, (a1, a2) = new_fig(t, 8.0, 3.6, ncols=2)
    ks = [k for k, _, _ in KSWEEP]
    colors = [t.blue if k == 0 else t.orange for k in ks]  # plain = blue, speculative = orange, as in every figure
    for ax, ylabel, top in ((a1, "tokens accepted per weight pass", 4.6), (a2, "decode tokens / s", 19)):
        ax.set_xticks(ks, ["plain", "k=2", "k=3", "k=4"])
        style(ax, t, ylabel=ylabel)
        ax.set_xlim(*slot_lim(0, 4))
        ax.set_ylim(0, top)
    rect = fig_title(fig, t, "Drafts per round (k): more drafts raise tokens per pass, but verifying them costs more")
    w = px_to_data(a1, BAR_PX, rect=rect)
    a1.bar(ks, [p for _, p, _ in KSWEEP], color=colors, width=w)
    for k, p, _ in KSWEEP:
        a1.text(k, p + 0.08, f"{p:.2f}", ha="center", color=t.ink, fontsize=9)
    a2.bar(ks, [s for _, _, s in KSWEEP], color=colors, width=w)
    for k, _, s in KSWEEP:
        a2.text(k, s + 0.3, f"{s:.1f}", ha="center", color=t.ink, fontsize=9)
    a1.legend(handles=[Patch(color=t.blue, label="plain greedy"), Patch(color=t.orange, label="speculative (MTP)")],
              loc="upper left")
    save(fig, t, "speculative_k", rect=rect)


def fig_prefill(t: Theme) -> None:
    fig, ax = new_fig(t, 7.0, 3.4)
    rows = sorted(PREFILL.items(), key=lambda kv: kv[1])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    style(ax, t, xlabel="prompt tokens / s (64-token prompt)", grid="x")
    ax.set_xlim(0, 42)
    ax.set_ylim(*slot_lim(0, len(rows) - 1, half=0.275))
    title(ax, t, "Prefill (prompt processing) speed", "64-token prompt; pinfer reads the weights once per group of up to 16 prompt positions")
    ax.barh(range(len(rows)), [r[1] for r in rows], height=px_to_data(ax, BAR_PX, axis="y"),
            color=[t.blue if r[0].startswith("pinfer") else t.orange for r in rows])
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", color=t.ink, fontsize=9)
    ax.legend(handles=[Patch(color=t.blue, label="pinfer (ours)"), Patch(color=t.orange, label="standard runtimes")],
              loc="lower right")
    save(fig, t, "prefill_speed")


FIGURES = (fig_decode, fig_speed_quality, fig_progress, fig_context, fig_kspeed, fig_prefill)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for t in THEMES:
        setup(t)
        for make in FIGURES:
            make(t)


if __name__ == "__main__":
    main()
