"""Shared style for the README figures (tools/make_figures*.py).

Every figure is rendered twice from the validated dataviz reference palette: `<name>.png` on the
light surface and `<name>-dark.png` on the dark surface. The dark tokens are the palette's own
dark-surface steps (selected, not an inverted light figure); README.md shows the file matching the
reader's GitHub theme through a <picture> element. Categorical hues are used in fixed slot order.
Text wears ink tokens, never a series colour; the one exception is a label set inside a filled
mark, which takes whichever theme extreme (surface or ink) contrasts more with that fill.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.use("Agg")  # files only; switching after the pyplot import is fine while no figure exists
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
DPI = 160
# Mark sizes, in saved pixels. GitHub fits README images to its ~860 px column, i.e. scales these
# 1100-1400 px wide figures by ~0.6-0.75: 34 px renders at ~22 CSS px (the bar cap is 24 px).
BAR_PX = 34
GAP_PX = 3  # surface gap between touching marks (~2 CSS px)
LINE_PT = 1.4  # line width in points (~3 px saved, ~2 CSS px)
RING_PT = 1.4  # surface-coloured ring around dots and line markers


@dataclass(frozen=True)
class Theme:
    """One mode's tokens from the dataviz reference palette (chart chrome, ink, slots 1-5)."""

    name: str
    suffix: str  # file-name suffix: "" (light) or "-dark"
    surface: str
    ink: str  # primary text: titles, values
    ink2: str  # secondary text: subtitles, tick and legend labels, annotations
    muted: str  # connectors and brackets that must stay visible
    grid: str
    axis: str  # spines and reference rules
    blue: str  # categorical slots 1-5, validated order
    orange: str
    aqua: str
    yellow: str
    magenta: str


LIGHT = Theme("light", "", surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
              grid="#e1e0d9", axis="#c3c2b7", blue="#2a78d6", orange="#eb6834", aqua="#1baf7a",
              yellow="#eda100", magenta="#e87ba4")
DARK = Theme("dark", "-dark", surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
             grid="#2c2c2a", axis="#383835", blue="#3987e5", orange="#d95926", aqua="#199e70",
             yellow="#c98500", magenta="#d55181")
THEMES = (LIGHT, DARK)


def setup(t: Theme) -> None:
    """Matplotlib defaults for theme `t`; call before drawing that theme's figures."""
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "text.color": t.ink,
        "axes.titleweight": "semibold",
        "axes.titlesize": 11.5,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "legend.labelcolor": t.ink2,
        "figure.facecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
    })


def new_fig(t: Theme, w: float = 8.0, h: float = 4.2, ncols: int = 1, **kw) -> tuple:
    """Figure on the theme surface: (fig, ax), or (fig, axes) when ncols > 1."""
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), dpi=DPI, **kw)
    fig.patch.set_facecolor(t.surface)
    return fig, axes


def style(ax: plt.Axes, t: Theme, ylabel: str = "", xlabel: str = "", grid: str = "y") -> None:
    """Recessive chrome: no top/right spines, hairline grid on one axis, secondary-ink labels."""
    ax.set_facecolor(t.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.axis)
    ax.tick_params(colors=t.muted, labelcolor=t.ink2, length=0)
    ax.grid(visible=False)
    if grid:
        ax.grid(axis=grid, color=t.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=t.ink2)
    ax.set_xlabel(xlabel, color=t.ink2)


def title(ax: plt.Axes, t: Theme, text: str, sub: str | None = None) -> None:
    """Left-aligned title with an optional subtitle line in secondary ink."""
    ax.set_title(text, color=t.ink, loc="left", pad=16 if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=t.ink2, fontsize=8.5, va="bottom")


def fig_title(fig: plt.Figure, t: Theme, text: str, sub: str | None = None) -> tuple | None:
    """Figure-level title for small multiples, with an optional subtitle line under it.

    tight_layout keeps the axes clear of a suptitle by itself but ignores fig.text, so with a
    subtitle this returns the layout rect (12 px lower) to pass to px_to_data() and save().
    """
    fig.suptitle(text, color=t.ink, x=0.02, y=0.98, ha="left", va="top", fontweight="semibold")
    if not sub:
        return None
    h_px = fig.get_figheight() * fig.dpi
    fig.text(0.02, 1 - 44 / h_px, sub, color=t.ink2, fontsize=8.5, va="top")
    return (0, 0, 1, 1 - 12 / h_px)


def slot_lim(first: float, last: float, half: float = 0.3) -> tuple[float, float]:
    """Axis limits framing bar slots first..last the way autoscale framed the original 0.6-wide bars."""
    lo, hi = first - half, last + half
    margin = 0.05 * (hi - lo)
    return lo - margin, hi + margin


def px_to_data(ax: plt.Axes, n_px: float, axis: str = "x", rect: tuple | None = None) -> float:
    """Data units that render as `n_px` saved pixels along `axis`.

    Lays the figure out exactly as save() will (pass the same rect), so call it after the
    limits, ticks, labels and titles are set and before drawing the marks it sizes.
    """
    ax.figure.tight_layout(rect=rect)
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(0.0, 0.0), (n_px, n_px)])
    return abs(x1 - x0) if axis == "x" else abs(y1 - y0)


def _luminance(color: str) -> float:
    """WCAG relative luminance of a #rrggbb colour."""
    rgb = [int(color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two #rrggbb colours."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def on_fill(t: Theme, fill: str) -> str:
    """Text colour for a label inside a filled mark: the theme extreme that contrasts more.

    Near-ties (within 0.25) go to the lighter extreme, which keeps white on the light-mode blue
    (4.3:1 white vs 4.5:1 ink) while orange, aqua, yellow and every dark-mode fill take dark text.
    """
    light, dark = sorted((t.surface, t.ink), key=_luminance, reverse=True)
    return light if contrast(light, fill) >= contrast(dark, fill) - 0.25 else dark


def ring(t: Theme) -> dict:
    """Marker kwargs for ax.plot: the surface ring that keeps overlapping markers legible."""
    return {"markeredgecolor": t.surface, "markeredgewidth": RING_PT}


def line_key(color: str, label: str) -> Line2D:
    """Legend handle for a line series: line and marker without the ring, which would cut the short key."""
    return Line2D([], [], color=color, linewidth=LINE_PT, marker="o", markersize=7, label=label)


def save(fig: plt.Figure, t: Theme, name: str, rect: tuple | None = None) -> None:
    """Lay out and write docs/figures/<name><suffix>.png."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    path = OUT / f"{name}{t.suffix}.png"
    fig.savefig(path, facecolor=t.surface)
    plt.close(fig)
    logger.info("wrote %s", path)
