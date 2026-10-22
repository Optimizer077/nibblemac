"""Render the result figures for the README into docs/figures/: four PNGs per figure, desktop and
phone layouts in the light and dark themes (`<name>.png`, `-dark`, `-mobile`, `-mobile-dark`).

Numbers are the verified ones from notes.md ("Scoreboard", "Context length") with their
result files in bench/results/; each dataset below names its source so a figure can be
traced back. Style, palette, themes and layouts: tools/figstyle.py.

    uv run --no-project --with matplotlib python tools/make_figures.py
"""
from __future__ import annotations

import logging

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figstyle import (
    DASH,
    LAYOUTS,
    LINE,
    MARKER,
    RING,
    THEMES,
    Layout,
    Theme,
    hbar_rows,
    header,
    line_key,
    new_fig,
    panel_title,
    px_to_data,
    ring,
    save,
    setup,
    slot_lim,
    style,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- data (Pi 4, 4 threads, 64+64 tokens)
# (label, plain tok/s, speculative tok/s or None, perplexity or None, source)
ENGINES = [
    ("pinfer Q4K (ours)", 7.89, 20.22, 18.50, "kcap-merge6_20261022_145226"),  # default: adaptive k<=4 + ngram 7
    ("pinfer Q8 (ours)", 4.22, 8.73, 17.96, "pinfer-fast-q8-headq8_20261021_235144"),
    ("MNN 3.6 int4", 6.67, None, None, "mnn-qwen35-0.8b-int4_20261021_221038"),
    ("llama.cpp Q4_K_M", 5.41, None, 18.59, "llamacpp-qwen35-0.8b-Q4_K_M_20261021_213929"),
    ("llama.cpp Q4_0", 5.59, None, 21.25, "llamacpp-qwen35-0.8b-Q4_0_20261021_213343"),
    ("llama.cpp Q8_0", 4.12, None, 17.97, "llamacpp-qwen35-0.8b-Q8_0_20261021_214506"),
    ("ONNX Runtime GenAI int4", 3.10, None, None, "ort-qwen35-0.8b-int4-acc4_20261021_222410"),
]
PREFILL = {"pinfer Q4K (ours)": 43.1,  # mm2-e2e-prefill_20261022_140719
 "MNN 3.6 int4": 30.9, "llama.cpp Q8_0": 17.1,
           "llama.cpp Q4_0": 15.7, "llama.cpp Q4_K_M": 14.8, "ONNX Runtime GenAI int4": 9.9}
# Optimization steps in order (desktop label, phone label, plain, speculative); 2026-10-21/22
# Phone labels stay within ~7 characters: six slots share a 360 px column.
PROGRESS = [
    ("scalar C\nbf16", "scalar\nbf16", 1.55, None),
    ("NEON\nbf16", "NEON\nbf16", 2.27, 4.13),
    ("Q4K\nweights", "Q4K", 6.83, 11.57),
    ("bit-exact kernels\n+ NEON recurrence", "exact\nkernels", 7.26, 13.35),
    ("vectorized\nper-token ops", "vector\nops", 7.88, 16.49),
    ("cheaper\nverify pass", "faster\nverify", 7.89, 18.43),  # speed-ab-merge4_20261022_103609
    ("adaptive k +\nn-gram drafts", "n-gram\ndrafts", 7.88, 20.22),  # kcap-merge6_20261022_145226
]
# Context length sweep, Q4K, wikitext prompt of N tokens then 32 generated (engine/CONTEXT.md Summary).
# After: attention round 3 (ctx, plain fp32 KV, plain fp16 KV, fp32 prefill) — decode = medians of 3 rounds
# from saved states (att3-final-ab-ctx{64,2048-4096,8192}_20261022_10*), prefill single runs (CONTEXT.md §9);
# 512 from the first rewrite (pinfer-ctx512-new-q4k, pinfer-final-kv{0,1}-ctx512): not re-measured.
CONTEXT = [(64, 7.87, 7.88, 36.3), (512, 7.69, 7.67, 36.8), (2048, 7.10, 7.36, 32.7),
           (4096, 6.43, 6.86, 29.9), (8192, 5.34, 6.07, 24.3)]
# Before: the vectorized binary before the context work (ctx, plain, prefill) — pinfer-final2-q4k (64),
# pinfer-ctx2048-profile-q4k (2048), pinfer-ctx4096-q4k (4096); 512 was only run with an older binary.
CONTEXT_BEFORE = [(64, 7.88, 37.4), (2048, 6.57, 25.7), (4096, 5.56, 18.1)]
# llama.cpp Q4_K_M token generation at the same KV depth (f16 KV, its default) — llamacpp-ctx-d*
LLAMA_CTX = [(64, 5.48), (512, 5.23), (2048, 4.32), (4096, 3.43)]
# Speculative k sweep, fixed k, after the batched-kernels round (mm2-e2e-decode_20261022_140420, prompt64,
# medians of 3; tokens per pass unchanged, output identical): k, tokens per pass, tok/s
KSWEEP = [(0, 1.00, 7.88), (2, 2.63, 16.18), (3, 3.32, 18.15), (4, 3.94, 18.64)]


def fig_decode(t: Theme, L: Layout) -> None:
    """Grouped horizontal bars: plain vs speculative decode, fastest at the top."""
    rows = sorted(ENGINES, key=lambda r: -r[1])
    fig, ax = new_fig(t, L, 460 if L.mobile else 410)
    style(ax, t, xlabel="tokens / s (higher is better)", grid="x")
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 5, 10, 15, 20])
    keys = [Patch(color=t.blue, label="plain greedy"), Patch(color=t.orange, label="speculative (exact)")]
    rect = header(fig, t, L, "Decode speed: pinfer vs standard runtimes",
                  "Qwen3.5-0.8B, Raspberry Pi 4 (4 GB), best thread count per engine, 64-token prompt + 64 tokens",
                  handles=keys if L.mobile else None, ncol=2)
    hbar_rows(ax, t, L, [r[0] for r in rows], [[r[1], r[2]] for r in rows], [[t.blue, t.orange]] * len(rows),
              "{:.1f}", rect)
    if not L.mobile:
        ax.legend(handles=keys, loc="lower right")
    save(fig, t, L, "decode_speed", rect)


def _dot_key(color: str, label: str, marker: str = "o") -> Line2D:
    """Legend handle for one scatter group."""
    return Line2D([], [], color=color, marker=marker, linestyle="none", markersize=MARKER - RING, label=label)


def fig_speed_quality(t: Theme, L: Layout) -> None:
    """Scatter: perplexity (lower is better) against plain decode speed."""
    fig, ax = new_fig(t, L, 400 if L.mobile else 430)
    # Point labels: text, offset in CSS px, horizontal alignment. The Q8 pair coincides at this scale,
    # so the phone names both under it in one label.
    if L.mobile:
        labels = {"pinfer Q4K (ours)": ("pinfer Q4K", (9, -3), "left"),
                  "pinfer Q4K (ours) + speculative": ("pinfer Q4K + speculative", (9, 0), "left"),
                  "pinfer Q8 (ours)": ("pinfer Q8 / llama.cpp Q8_0", (-6, -17), "left"),
                  "pinfer Q8 (ours) + speculative": ("pinfer Q8 + speculative", (9, 5), "left"),
                  "llama.cpp Q4_K_M": ("llama.cpp Q4_K_M", (9, 2), "left"),
                  "llama.cpp Q4_0": ("llama.cpp Q4_0", (0, 15), "center")}
    else:
        labels = {"pinfer Q4K (ours)": ("pinfer Q4K (ours)", (10, -4), "left"),
                  "pinfer Q4K (ours) + speculative": ("pinfer Q4K (ours) + speculative", (10, 0), "left"),
                  "pinfer Q8 (ours)": ("pinfer Q8 (ours)", (10, 4), "left"),
                  "pinfer Q8 (ours) + speculative": ("pinfer Q8 (ours) + speculative", (10, 5), "left"),
                  "llama.cpp Q4_K_M": ("llama.cpp Q4_K_M", (10, 3), "left"),
                  "llama.cpp Q4_0": ("llama.cpp Q4_0", (10, 0), "left"),
                  "llama.cpp Q8_0": ("llama.cpp Q8_0", (10, -12), "left")}
    # pinfer Q8 (17.96, 4.22) and llama.cpp Q8_0 (17.97, 4.12) coincide at this scale: each is drawn
    # as one half of a disc at its own position (no ring, which would cut the other half), so both
    # colours stay visible.
    halves = {"pinfer Q8 (ours)": "left", "llama.cpp Q8_0": "right"}

    def label(key: str, x: float, y: float) -> None:
        if key in labels:
            text, off, ha = labels[key]
            ax.annotate(text, (x, y), xytext=off, textcoords="offset points", ha=ha, va="center", color=t.ink2)

    for name, plain, spec, ppl, _ in sorted(ENGINES, key=lambda r: r[0].startswith("pinfer")):
        if ppl is None:
            continue
        color = t.blue if name.startswith("pinfer") else t.orange
        edge = {"markeredgewidth": 0, "markersize": MARKER - RING} if name in halves else {**ring(t), "markersize": MARKER}
        ax.plot(ppl, plain, "o", color=color, fillstyle=halves.get(name, "full"), markerfacecoloralt="none",
                zorder=4, **edge)
        label(name, ppl, plain)
        if spec:
            ax.plot(ppl, spec, "^", ms=MARKER + 1, color=t.blue, zorder=3, **ring(t))
            label(f"{name} + speculative", ppl, spec)
    keys = [_dot_key(t.blue, "pinfer (ours)"), _dot_key(t.orange, "llama.cpp"),
            _dot_key(t.blue, "pinfer, speculative", "^")]
    style(ax, t, ylabel="decode tokens / s", xlabel="perplexity, wikitext-2 (lower is better)")
    ax.set_xlim(*((17.6, 21.9) if L.mobile else (17.5, 22.5)))
    ax.set_ylim(0, 23)
    ax.set_yticks([0, 5, 10, 15, 20])
    rect = header(fig, t, L, "Speed vs quality",
                  "Perplexity on wikitext-2 (8 × 512 tokens, llama-perplexity protocol); pinfer Q4K sits at "
                  "Q4_K_M quality", handles=keys if L.mobile else None, ncol=2)
    if not L.mobile:
        ax.legend(handles=keys, loc="upper right")
    save(fig, t, L, "speed_vs_quality", rect)


def fig_progress(t: Theme, L: Layout) -> None:
    """Line: decode speed after each optimization step."""
    fig, ax = new_fig(t, L, 330 if L.mobile else 360)
    x = list(range(len(PROGRESS)))
    plain = [p[2] for p in PROGRESS]
    ax.plot(x, plain, color=t.blue, linewidth=LINE, marker="o", markersize=MARKER, **ring(t))
    sx = [i for i, p in enumerate(PROGRESS) if p[3]]
    ax.plot(sx, [PROGRESS[i][3] for i in sx], color=t.orange, linewidth=LINE, marker="o", markersize=MARKER,
            **ring(t))
    # Value labels: plain under its point, speculative above, except the speculative point at NEON bf16
    # (i == 1), whose steep segment to Q4K passes over it: that label goes to its left. The surface box
    # breaks the dashed reference rule where a label crosses it (zorder: over the rule, under the series).
    box = {"boxstyle": "square,pad=0.15", "fc": t.surface, "ec": "none"}
    for i, p in enumerate(PROGRESS):
        ax.annotate(f"{p[2]:.1f}", (i, p[2]), xytext=(0, -9), textcoords="offset points", ha="center",
                    va="top", color=t.ink, bbox=box, zorder=1.8)
        if p[3]:
            off, ha = ((-10, 0), "right") if i == 1 else ((0, 9), "center")
            ax.annotate(f"{p[3]:.1f}", (i, p[3]), xytext=off, textcoords="offset points", ha=ha,
                        va="center" if ha == "right" else "bottom", color=t.ink, bbox=box, zorder=1.8)
    ax.axhline(5.41, color=t.muted, linewidth=1.2, linestyle=DASH, zorder=1.5)  # behind the series
    ax.annotate("llama.cpp Q4_K_M 5.4", (len(PROGRESS) - 0.6, 5.41), xytext=(0, -5),
                textcoords="offset points", ha="right", va="top", color=t.ink2)
    ax.set_xticks(x, [p[1] if L.mobile else p[0] for p in PROGRESS])
    style(ax, t, ylabel="tokens / s")
    ax.set_xlim(-0.45, len(PROGRESS) - 0.55)
    ax.set_ylim(0, 23)
    ax.set_yticks([0, 5, 10, 15, 20])
    keys = [line_key(t.blue, "plain greedy"), line_key(t.orange, "speculative (exact)")]
    rect = header(fig, t, L, "How the speed was built up",
                  "Each step verified against the fp32 reference (per-layer error ≤ 3e-6) and, for speculation, "
                  "token-identical to plain greedy", handles=keys if L.mobile else None, ncol=2)
    if not L.mobile:
        ax.legend(handles=keys, loc="upper left")
    save(fig, t, L, "progress", rect)


def _ctx_line(ax, t: Theme, pts: list, color: str, dashed: bool = False, end: tuple = (8, 0)) -> None:
    """One series of the context figure, its last value labelled beside the last marker (offset `end`, CSS px)."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, color=color, linewidth=LINE, marker="o", markersize=MARKER - 1,
            linestyle=DASH if dashed else "-", **ring(t))
    ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=end, textcoords="offset points",
                va="center", color=t.ink)


def fig_context(t: Theme, L: Layout) -> None:
    """Two panels against context length (log-2 x axis): plain decode, prefill; before vs after.

    Desktop: side by side. Phone: stacked, with the long tick values written as 2K / 4K / 8K.
    """
    if L.mobile:
        fig, (a1, a2) = new_fig(t, L, 560, nrows=2, gridspec_kw={"height_ratios": [4, 3]})
    else:
        fig, (a1, a2) = new_fig(t, L, 340, ncols=2)
    xs = [c[0] for c in CONTEXT]
    before = t.muted  # the old binary is context, not a series to compare hues against
    _ctx_line(a1, t, [(c[0], c[1]) for c in CONTEXT], t.blue, end=(8, -7))
    _ctx_line(a1, t, [(c[0], c[2]) for c in CONTEXT if c[2]], t.aqua, end=(8, 7))
    _ctx_line(a1, t, [(c[0], c[1]) for c in CONTEXT_BEFORE], before, dashed=True, end=(-6, -14))
    _ctx_line(a1, t, LLAMA_CTX, t.orange)
    style(a1, t, ylabel="decode tokens / s", xlabel="context length (tokens)")
    a1.set_ylim(0, 9)
    panel_title(a1, t, "Decode (plain greedy)")
    _ctx_line(a2, t, [(c[0], c[3]) for c in CONTEXT], t.blue)
    _ctx_line(a2, t, [(c[0], c[2]) for c in CONTEXT_BEFORE], before, dashed=True)
    style(a2, t, ylabel="prefill tokens / s", xlabel="context length (tokens)")
    a2.set_ylim(0, 42)
    panel_title(a2, t, "Prefill (whole prompt)")
    ticks = [f"{v // 1024}K" if L.mobile and v >= 2048 else str(v) for v in xs]
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, ticks)
        ax.minorticks_off()
        ax.set_xlim(48, 8192 * 1.9)
        if L.mobile:
            ax.set_xlabel("context length (tokens, K = 1024)", color=t.ink2)
    keys = [line_key(t.blue, "pinfer, fp32 KV cache (default, exact)"),
            line_key(t.aqua, "pinfer, fp16 KV cache (opt-in)"),
            line_key(before, "pinfer before the rewrite", dashed=True),
            line_key(t.orange, "llama.cpp Q4_K_M")]
    rect = header(fig, t, L, "Speed vs context length, before and after the attention rewrite",
                  "Qwen3.5-0.8B Q4K, Pi 4, 4 threads: an N-token wikitext prompt, then 32 generated tokens. "
                  "At 4096 tokens decode keeps 82% (fp32) / 87% (fp16) of its short-context speed",
                  handles=keys, ncol=1 if L.mobile else 2)
    save(fig, t, L, "context_length", rect)


def fig_kspeed(t: Theme, L: Layout) -> None:
    """Two small multiples (not a dual axis): tokens per pass and tok/s against k.

    Desktop: side by side. Phone: stacked.
    """
    if L.mobile:
        fig, (a1, a2) = new_fig(t, L, 470, nrows=2)
    else:
        fig, (a1, a2) = new_fig(t, L, 290, ncols=2)
    ks = [k for k, _, _ in KSWEEP]
    colors = [t.blue if k == 0 else t.orange for k in ks]  # plain = blue, speculative = orange, as in every figure
    for ax, head, top, yt in ((a1, "Tokens accepted per weight pass", 4.6, [0, 1, 2, 3, 4]),
                              (a2, "Decode tokens / s", 22, [0, 5, 10, 15, 20])):
        ax.set_xticks(ks, ["plain", "k=2", "k=3", "k=4"])
        style(ax, t)
        panel_title(ax, t, head)
        ax.set_xlim(*slot_lim(0, 4))
        ax.set_ylim(0, top)
        ax.set_yticks(yt)
    keys = [Patch(color=t.blue, label="plain greedy"), Patch(color=t.orange, label="speculative (MTP)")]
    rect = header(fig, t, L, "Drafts per round (k): more drafts raise tokens per pass, but verifying them costs more",
                  handles=keys, ncol=2)
    w = px_to_data(a1, L.bar + 4, rect=rect)
    a1.bar(ks, [p for _, p, _ in KSWEEP], color=colors, width=w)
    a2.bar(ks, [s for _, _, s in KSWEEP], color=colors, width=w)
    for k, p, s in KSWEEP:
        a1.annotate(f"{p:.2f}", (k, p), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    color=t.ink)
        a2.annotate(f"{s:.1f}", (k, s), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    color=t.ink)
    save(fig, t, L, "speculative_k", rect)


def fig_prefill(t: Theme, L: Layout) -> None:
    """Horizontal bars: prompt-processing speed by engine, fastest at the top."""
    rows = sorted(PREFILL.items(), key=lambda kv: -kv[1])
    fig, ax = new_fig(t, L, 330 if L.mobile else 300)
    style(ax, t, xlabel="prompt tokens / s (64-token prompt)", grid="x")
    ax.set_xlim(0, 42)
    keys = [Patch(color=t.blue, label="pinfer (ours)"), Patch(color=t.orange, label="standard runtimes")]
    rect = header(fig, t, L, "Prefill (prompt processing) speed",
                  "64-token prompt; pinfer reads the weights once per group of up to 16 prompt positions",
                  handles=keys if L.mobile else None, ncol=2)
    hbar_rows(ax, t, L, [r[0] for r in rows], [[r[1]] for r in rows],
              [[t.blue if r[0].startswith("pinfer") else t.orange] for r in rows], "{:.1f}", rect)
    if not L.mobile:
        ax.legend(handles=keys, loc="lower right")
    save(fig, t, L, "prefill_speed", rect)


FIGURES = (fig_decode, fig_speed_quality, fig_progress, fig_context, fig_kspeed, fig_prefill)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for L in LAYOUTS:
        for t in THEMES:
            setup(t, L)
            for make in FIGURES:
                make(t, L)


if __name__ == "__main__":
    main()
