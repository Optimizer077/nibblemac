"""'How it works' figures for the README, from the logged measurements: four PNGs per figure in
docs/figures/, desktop and phone layouts in the light and dark themes (`<name>.png`, `-dark`,
`-mobile`, `-mobile-dark`).

Sources: engine/CORE.md (per-step profile), engine/bench/KERNELS.md (kernel GB/s, quantization
error), engine/SPEC.md (MTP acceptance), notes.md (memband, kernel experiment), bench/results/,
and the tensor directory of the shipped Q4K model (bytes per part).

    uv run --no-project --with matplotlib python tools/make_figures2.py
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from figstyle import (
    DASH,
    GAP,
    LAYOUTS,
    MARKER,
    ROOT,
    THEMES,
    Layout,
    Theme,
    get_renderer,
    hbar_axis,
    hbar_rows,
    header,
    label_inside,
    name_above,
    new_fig,
    on_fill,
    panel_title,
    px_to_data,
    ring,
    save,
    setup,
    slot_lim,
    style,
    wrap,
)

logger = logging.getLogger(__name__)

# Per-token time, ms, Q4K, 4 threads (plain: engine/CORE.md final profile; verify pass = k=3, 4 positions,
# after the cheaper verify pass: 152.9 ms per pass, recurrence 13.7, other 8.5 — engine/VERIFY.md "Result" and §5,
# ver3-profile_20261022_081052; the deferred commit adds 1.9 ms per round)
STEP_PLAIN = [("weight streaming (matmul)", 118.6), ("recurrent state (18 linear-attn layers)", 6.5),
              ("everything else (norms, SiLU, attention, argmax)", 3.2)]
STEP_VERIFY = [("weight streaming (matmul)", 130.7), ("recurrent state (18 linear-attn layers)", 13.7),
               ("everything else (norms, SiLU, attention, argmax)", 8.5)]
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
STACK_BAR = 24.0  # CSS px: a single stacked bar carries labels inside, so it takes the dataviz cap


def fig_step_breakdown(t: Theme, L: Layout) -> None:
    """Stacked horizontal bars: where a plain decode step and a verify pass spend their time.

    Desktop: names as y labels, totals at the bar ends. Phone: name and total above each bar.
    """
    colors = [t.blue, t.orange, t.aqua]
    rows = [("plain decode step\n(1 token)", STEP_PLAIN), ("speculative verify pass\n(k=3: 4 positions at once)", STEP_VERIFY)]
    totals = [sum(ms for _, ms in parts) for _, parts in rows]
    if L.mobile:
        names = [f"plain decode step (1 token): {totals[0]:.0f} ms",
                 f"verify pass (k=3: 4 positions): {totals[1]:.0f} ms"]
    else:
        names = [name for name, _ in rows]
    fig, ax = new_fig(t, L, 190 if L.mobile else 180)
    style(ax, t, xlabel="milliseconds (Q4K model, 4 threads)", grid="x")
    ax.set_xlim(0, 166 if L.mobile else 200)
    keys = [Patch(color=c, label=n) for (n, _), c in zip(STEP_PLAIN, colors)]
    rect = header(fig, t, L, "Where a token's time goes",
                  "Reading 437 MB of weights dominates; verifying 3 drafts (4 positions) costs only 1.20× one token",
                  handles=keys, ncol=1 if L.mobile else 2)
    unit, name_h = hbar_axis(ax, L, names, rect)
    for i, (_, parts) in enumerate(rows):
        y = i + name_h / 2 * unit
        left = 0.0
        for (_, ms), c in zip(parts, colors):
            ax.barh(y, ms, left=left, color=c, height=STACK_BAR * unit, edgecolor=t.surface, linewidth=GAP)
            label_inside(ax, t, left + ms / 2, y, f"{ms:.0f} ms", c, ms)
            left += ms
        if L.mobile:
            name_above(ax, t, names[i], y - STACK_BAR / 2 * unit)
        else:
            ax.annotate(f"{left:.0f} ms total", (left, y), xytext=(6, 0), textcoords="offset points",
                        va="center", color=t.ink)
    save(fig, t, L, "step_breakdown", rect)


def fig_bandwidth(t: Theme, L: Layout) -> None:
    """Bars: effective weight-streaming speed of each kernel against the DRAM ceiling."""
    fig, ax = new_fig(t, L, 300 if L.mobile else 290)
    x = list(range(len(KERNEL_GBS)))
    if L.mobile:  # "Q8 (8.5 bit)" -> "Q8 / 8.5 bit" on two lines: a phone slot is ~60 px
        ticks = [f"{n.replace(' (', chr(10)).rstrip(')')}\n{thr} thr" for n, _, thr in KERNEL_GBS]
    else:
        ticks = [f"{n}\n{thr} thr" for n, _, thr in KERNEL_GBS]
    ax.set_xticks(x, ticks)
    style(ax, t, ylabel="GB/s of weights streamed")
    ax.set_xlim(*slot_lim(0, len(x) - 1))
    ax.set_ylim(0, 5)
    ceiling = Line2D([], [], color=t.muted, linewidth=1.2, linestyle=DASH,
                     label=f"DRAM read ceiling {MEMBAND:.2f} GB/s (memband, 1 thread)")
    rect = header(fig, t, L, "Matrix kernels run at the memory wall",
                  "Cortex-A72, best thread count per format; one core already saturates DRAM, so decode speed = "
                  "bandwidth ÷ bytes per token", handles=[ceiling])
    ax.bar(x, [v for _, v, _ in KERNEL_GBS], color=t.blue, width=px_to_data(ax, L.bar + 4, rect=rect))
    box = {"boxstyle": "square,pad=0.15", "fc": t.surface, "ec": "none"}  # breaks the rule under a label
    for i, (_, v, _) in enumerate(KERNEL_GBS):
        ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    color=t.ink, bbox=box)
    ax.axhline(MEMBAND, color=t.muted, linewidth=1.2, linestyle=DASH, zorder=0.8)  # behind the bars
    save(fig, t, L, "kernel_bandwidth", rect)


def fig_quant(t: Theme, L: Layout) -> None:
    """Two small multiples: weight error (bars from zero) and perplexity (dots: no zero baseline).

    Desktop: side by side. Phone: stacked.
    """
    if L.mobile:
        fig, (a1, a2) = new_fig(t, L, 470, nrows=2)
    else:
        fig, (a1, a2) = new_fig(t, L, 280, ncols=2)
    x = list(range(len(QUANT)))
    for ax in (a1, a2):
        ax.set_xticks(x, [q[0].replace(" (", "\n(") for q in QUANT])
        ax.set_xlim(*slot_lim(0, len(x) - 1))
    style(a1, t)
    panel_title(a1, t, "Relative RMS weight error (%)")
    a1.set_ylim(0, 10.5)
    style(a2, t)
    panel_title(a2, t, "Perplexity, wikitext-2 (lower is better)")
    a2.set_ylim(17, 22.5)
    rect = header(fig, t, L, "Quantization formats: the same bits, very different quality",
                  "Q4K's asymmetric 32-column groups with 6-bit sub-scales keep Q4_K_M quality; symmetric Q4 "
                  "loses 17% like llama.cpp's Q4_0")
    a1.bar(x, [q[1] for q in QUANT], color=t.blue, width=px_to_data(a1, L.bar + 4, rect=rect))
    a2.plot(x, [q[2] for q in QUANT], "o", ms=MARKER + 1, color=t.blue, **ring(t))
    for i, q in enumerate(QUANT):
        a1.annotate(f"{q[1]:.1f}%" if q[1] else "exact", (i, q[1]), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", color=t.ink)
        a2.annotate(f"{q[2]:.2f}", (i, q[2]), xytext=(0, 9), textcoords="offset points", ha="center",
                    va="bottom", color=t.ink)
    save(fig, t, L, "quantization", rect)


# Tick labels of the acceptance figure, keyed by Layout.mobile: the names of ACCEPT / CONTROLS (keep
# them in step, same order) re-broken for each layout's slot width.
ACCEPT_TICKS = {False: ["1st draft", "2nd\n(given 1st ok)", "3rd\n(given 1–2 ok)", "4th\n(given 1–3 ok)"],
                True: ["1st\ndraft", "2nd\ngiven\n1st ok", "3rd\ngiven\n1–2 ok", "4th\ngiven\n1–3 ok"]}
CONTROL_TICKS = {False: ["as specified", "hidden state taken\nbefore final norm", "concat order\nswapped"],
                 True: ["as\nspecified", "hidden state\ntaken before\nfinal norm", "concat\norder\nswapped"]}


def fig_acceptance(t: Theme, L: Layout) -> None:
    """Two small multiples: acceptance per draft position, and the semantics controls.

    Desktop: side by side. Phone: stacked.
    """
    if L.mobile:
        fig, (a1, a2) = new_fig(t, L, 520, nrows=2)
    else:
        fig, (a1, a2) = new_fig(t, L, 300, ncols=2, gridspec_kw={"width_ratios": [4, 3]})
    for ax, ticks, n in ((a1, ACCEPT_TICKS[L.mobile], len(ACCEPT)), (a2, CONTROL_TICKS[L.mobile], len(CONTROLS))):
        ax.set_xticks(range(n), ticks)
        ax.set_xlim(*slot_lim(0, n - 1))
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        style(ax, t)
    panel_title(a1, t, "Drafts accepted by the full model (%)")
    panel_title(a2, t, "1st-draft acceptance (%)")
    keys = [Patch(color=t.blue, label="our implementation"), Patch(color=t.orange, label="deliberately wrong control")]
    rect = header(fig, t, L, "The built-in MTP draft head is right 87% of the time — and we proved we wired it correctly",
                  "651 positions of prose, code, Q&A and Chinese; later drafts are more often right because "
                  "accepted text is predictable text", handles=keys, ncol=1 if L.mobile else 2)
    w = px_to_data(a1, L.bar + 4, rect=rect)
    a1.bar(range(len(ACCEPT)), [a[1] * 100 for a in ACCEPT], color=t.blue, width=w)
    a2.bar(range(len(CONTROLS)), [c[1] * 100 for c in CONTROLS], color=[t.blue, t.orange, t.orange],
           width=px_to_data(a2, L.bar + 4, rect=rect))
    for ax, rows, fmt in ((a1, ACCEPT, "{:.0f}%"), (a2, CONTROLS, "{:.1f}%")):
        for i, (_, v) in enumerate(rows):
            ax.annotate(fmt.format(v * 100), (i, v * 100), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", color=t.ink)
    save(fig, t, L, "mtp_acceptance", rect)


def fig_os(t: Theme, L: Layout) -> None:
    """Grouped bars: the custom-kernel experiment changed nothing.

    Desktop: vertical pairs per setup. Phone: horizontal pairs with each setup's name above them.
    """
    keys = [Patch(color=t.blue, label="plain decode"), Patch(color=t.orange, label="speculative (k=3)")]
    title = "What did not help: a custom split-core kernel"
    sub = ("Same binary and model; three one-time boots of 6.18.50-pinfer — all within 1%, because the engine "
           "is DRAM-bound")
    if L.mobile:
        fig, ax = new_fig(t, L, 330)
        style(ax, t, xlabel="tokens / s", grid="x")
        ax.set_xlim(0, 19)
        ax.set_xticks([0, 5, 10, 15])
        rect = header(fig, t, L, title, sub, handles=keys, ncol=2)
        hbar_rows(ax, t, L, [o[0].replace("\n", " ") for o in OS_EXP], [[o[1], o[2]] for o in OS_EXP],
                  [[t.blue, t.orange]] * len(OS_EXP), "{:.2f}", rect)
        save(fig, t, L, "os_experiment", rect)
        return
    fig, ax = new_fig(t, L, 280)
    x = list(range(len(OS_EXP)))
    ax.set_xticks(x, [o[0] for o in OS_EXP])
    style(ax, t, ylabel="tokens / s")
    ax.set_xlim(*slot_lim(0, len(x) - 1, half=0.36))
    ax.set_ylim(0, 17)
    ax.set_yticks([0, 5, 10, 15])
    rect = header(fig, t, L, title, sub, handles=keys, ncol=2)
    w = px_to_data(ax, L.bar, rect=rect)
    off = px_to_data(ax, (L.bar + GAP) / 2, rect=rect)
    ax.bar([i - off for i in x], [o[1] for o in OS_EXP], width=w, color=t.blue)
    ax.bar([i + off for i in x], [o[2] for o in OS_EXP], width=w, color=t.orange)
    for i, o in enumerate(OS_EXP):
        for dx, v in ((-off, o[1]), (off, o[2])):
            ax.annotate(f"{v:.2f}", (i + dx, v), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", color=t.ink)
    save(fig, t, L, "os_experiment", rect)


# The three stages of a speculative round: heading, then two lines of detail (box text).
SPEC_STEPS = [("1. Draft", "MTP head guesses 3 tokens\n(1 layer, ~27 ms, draft vocab 32K)"),
              ("2. Verify in ONE weight pass", "full model scores 4 positions\n(437 MB read once, ~160 ms)"),
              ("3. Accept the matching prefix", "+ 1 token of the full model's own\n→ 3.3 tokens per round on average")]
SPEC_ARROWS = ["3 drafts", "verdicts"]
SPEC_NOTE = ("Output is token-for-token identical to plain greedy decoding (every kept token is the full "
             "model's own choice); only the speed changes.")


def fig_spec_schematic(t: Theme, L: Layout) -> None:
    """Block schematic of one speculative round, drawn in CSS px on an axes that fills the body.

    Desktop: the three boxes in a row, each as wide as its text. Phone: stacked, arrows pointing down.
    """
    fills = [t.orange, t.blue, t.aqua]
    pad, line_h = 14.0, L.text * 1.35  # box padding and text line height, CSS px
    box_h = 3 * line_h + 2 * pad
    width = L.css_width - 2 * L.margin
    probe, _ = new_fig(t, L, 10)  # a scratch figure to measure the box texts and the note
    note = wrap(probe, SPEC_NOTE, L.text, width)
    renderer = get_renderer(probe)

    def text_w(s: str, weight: str = "normal") -> float:
        txt = probe.text(0, 0, s, fontsize=L.text, fontweight=weight, linespacing=1.35)
        return txt.get_window_extent(renderer).width * 72 / probe.dpi

    widths = [max(text_w(h, "semibold"), text_w(b)) + 2 * pad for h, b in SPEC_STEPS]
    note_lines = note.count("\n") + 1
    plt.close(probe)
    arrow_v = 46.0  # phone: vertical arrow length between boxes
    if L.mobile:
        body = 3 * box_h + 2 * arrow_v + 16 + note_lines * line_h + L.margin
    else:
        body = box_h + 18 + note_lines * line_h + L.margin
    fig, ax = new_fig(t, L, body)
    ax.axis("off")
    rect = header(fig, t, L, "One round of MTP speculative decoding")
    ax.set_position((0, 0, 1, rect[3]))
    ax.set_xlim(0, L.css_width)
    ax.set_ylim(body, 0)  # y down, in CSS px from the top of the body

    def box(x: float, y: float, w: float, step: tuple, fc: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, box_h, boxstyle="round,pad=0,rounding_size=6", fc=fc, ec="none"))
        ink = on_fill(t, fc)
        ax.text(x + w / 2, y + pad, step[0], ha="center", va="top", color=ink, fontweight="semibold")
        ax.text(x + w / 2, y + pad + line_h, step[1], ha="center", va="top", color=ink, linespacing=1.35)

    def arrow(p: tuple, q: tuple) -> None:
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=t.muted, linewidth=1.6,
                                     shrinkA=0, shrinkB=0))

    x0 = L.margin
    if L.mobile:
        y = 0.0
        for i, (step, fc) in enumerate(zip(SPEC_STEPS, fills)):
            box(x0, y, width, step, fc)
            y += box_h
            if i < 2:
                arrow((L.css_width / 2, y + 6), (L.css_width / 2, y + arrow_v - 6))
                ax.text(L.css_width / 2 + 12, y + arrow_v / 2, SPEC_ARROWS[i], ha="left", va="center",
                        color=t.ink2)
                y += arrow_v
        y_note = y + 16
    else:
        span = (width - sum(widths)) / 2  # the two arrow gaps share what the boxes leave
        x = x0
        for i, (step, fc, w) in enumerate(zip(SPEC_STEPS, fills, widths)):
            box(x, 0, w, step, fc)
            x += w
            if i < 2:
                arrow((x + 6, box_h / 2), (x + span - 6, box_h / 2))
                ax.text(x + span / 2, box_h / 2 - 6, SPEC_ARROWS[i], ha="center", va="bottom", color=t.ink2)
                x += span
        y_note = box_h + 18
    ax.text(x0, y_note, note, ha="left", va="top", color=t.ink2, linespacing=1.35)
    save(fig, t, L, "speculative_schematic", layout=False)


def fig_bytes(t: Theme, L: Layout) -> None:
    """One stacked bar: the Q4K model's bytes by part, i.e. what every decode step has to stream."""
    fig, ax = new_fig(t, L, 140)
    mb = [b / 1e6 for _, b in BYTES]
    total, plain = sum(mb), sum(mb[:-1])
    colors = [t.blue, t.orange, t.aqua, t.yellow, t.magenta]  # slot order = stack order (validated adjacent pairs)
    style(ax, t, xlabel="MB (10⁶ bytes)", grid="x")
    ax.set_xlim(0, 480 if L.mobile else 460)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    keys = [Patch(color=c, label=f"{name}: {v:.1f} MB") for (name, _), v, c in zip(BYTES, mb, colors)]
    rect = header(fig, t, L, "Where a token's bytes go",
                  "Tensor bytes of the 437 MB Q4K model by part (from the .pq tensor directory); decode speed = "
                  "bandwidth ÷ bytes read per token", handles=keys, ncol=1 if L.mobile else 2)
    # y in CSS px: the bar centred at 0, the bracket and its caption above it.
    ax.set_ylim(0, 1)
    h_css = 1 / px_to_data(ax, 1, axis="y", rect=rect)
    above = STACK_BAR / 2 + 7 + 7 + 4 + L.text * 1.35 + 6  # bar half, gap, bracket, gap, caption, air
    ax.set_ylim(above - h_css, above)
    per_px = px_to_data(ax, 1, rect=rect)
    left = 0.0
    for i, (v, c) in enumerate(zip(mb, colors)):
        ax.barh(0, v, left=left, height=STACK_BAR, color=c, edgecolor=t.surface, linewidth=GAP)
        # an interior segment too narrow for its value leaves it to the legend; the last one has a free end
        if not label_inside(ax, t, left + v / 2, 0, f"{100 * v / total:.1f}%", c, v) and i == len(mb) - 1:
            ax.text(left + v + 4 * per_px, 0, f"{100 * v / total:.1f}%", va="center", color=t.ink,
                    in_layout=False)  # may overhang the axes into the outer pad; must not move the axes
        left += v
    y0, y1 = STACK_BAR / 2 + 7, STACK_BAR / 2 + 14
    ax.plot([0, 0, plain, plain], [y0, y1, y1, y0], color=t.muted, linewidth=1.2, solid_joinstyle="miter")
    ax.text(plain / 2, y1 + 4, f"every plain decode step reads these {plain:.1f} MB", ha="center", va="bottom",
            color=t.ink2, bbox={"boxstyle": "square,pad=0.2", "fc": t.surface, "ec": "none"})  # breaks the grid
    save(fig, t, L, "bytes_per_token", rect)


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
        header_ = f.read(4096)
        n_tensors, dir_offset = struct.unpack_from("<IQ", header_, 12)
        full_mask = struct.unpack_from("<I", header_, 88)[0]
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
    for L in LAYOUTS:
        for t in THEMES:
            setup(t, L)
            for make in FIGURES:
                make(t, L)


if __name__ == "__main__":
    main()
