"""Shared style for the README figures (tools/make_figures*.py).

Every figure is rendered in two layouts and two themes, four PNGs:

    <name>.png, <name>-dark.png                 desktop: GitHub's 838 CSS px README column, at 2x (1676 px)
    <name>-mobile.png, <name>-mobile-dark.png   phones: a 360 CSS px column, at 3x (1080 px)

README.md picks one with <picture> sources on `(max-width: 600px)` and `(prefers-color-scheme: dark)`.
Each layout renders at DPI = 72 x its scale, so one point is one CSS px at display size: the font
sizes, line widths, marker sizes and bar thicknesses below are CSS px as the reader sees them. A
figure function draws both layouts; where the phone needs a different arrangement (stacked panels,
names above bars, legend in the header) it branches on `L.mobile`.

Colours come from the validated dataviz reference palette. The dark tokens are the palette's own
dark-surface steps (selected, not an inverted light figure). Categorical hues are used in fixed slot
order. Text wears ink tokens, never a series colour; the one exception is a label set inside a filled
mark, which takes whichever theme extreme (surface or ink) contrasts more with that fill. One axis per
panel, never a dual axis; thin marks; a legend for two or more series; direct labels used selectively.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast
from weakref import WeakKeyDictionary

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image  # a matplotlib dependency

matplotlib.use("Agg")  # files only; switching after the pyplot import is fine while no figure exists
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
FONTS = ["Segoe UI", "DejaVu Sans", "sans-serif"]
# Mark sizes in CSS px at display size (= points, see the docstring); dataviz mark specs.
GAP = 2.0  # surface gap between touching marks
NAME_GAP = 4.0  # between a phone row's name and its bars
LINE = 2.0  # line width
MARKER = 10.0  # marker diameter; the surface ring covers 1 px of it, leaving an 8 px dot
RING = 2.0  # surface-coloured ring around dots and line markers
DASH = (0, (4, 3))  # the one dash pattern: reference rules and the "before" series
PAD = 0.9  # tight_layout outer pad, in font sizes (~12 CSS px)
PANEL_PAD = 1.8  # gap between small multiples, in font sizes (~23 CSS px)


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


@dataclass(frozen=True)
class Layout:
    """One display target: the column width it is drawn for, its pixel density and type sizes (CSS px)."""

    name: str
    suffix: str  # file-name suffix before the theme's: "" (desktop) or "-mobile"
    css_width: int  # width of the column the figure fills, CSS px
    scale: int  # PNG pixels per CSS px (2 = retina desktop, 3 = phone)
    text: float  # ticks, axis labels, legends, value labels
    title: float
    sub: float  # subtitle under the title
    panel: float  # panel titles of small multiples
    bar: float  # bar thickness (dataviz cap: 24)

    @property
    def mobile(self) -> bool:
        return self.name == "mobile"

    @property
    def dpi(self) -> float:
        return 72.0 * self.scale

    @property
    def width_px(self) -> int:
        return self.css_width * self.scale

    @property
    def margin(self) -> float:
        """Outer margin in CSS px: the tight_layout pad, so the header lines up with the axes' outer edge."""
        return PAD * self.text


DESKTOP = Layout("desktop", "", css_width=838, scale=2, text=13, title=18, sub=14, panel=15, bar=20)
MOBILE = Layout("mobile", "-mobile", css_width=360, scale=3, text=13, title=16, sub=13, panel=14, bar=16)
LAYOUTS = (DESKTOP, MOBILE)


def setup(t: Theme, L: Layout) -> None:
    """Matplotlib defaults for theme `t` in layout `L`; call before drawing that pair's figures."""
    plt.rcParams.update({
        "font.family": FONTS,
        "font.size": L.text,
        "axes.labelsize": L.text,
        "xtick.labelsize": L.text,
        "ytick.labelsize": L.text,
        "legend.fontsize": L.text,
        "axes.titlesize": L.panel,
        "axes.titleweight": "semibold",
        "axes.titlepad": 8,
        "axes.labelpad": 6,
        "axes.linewidth": 1.0,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "text.color": t.ink,
        "legend.frameon": False,
        "legend.labelcolor": t.ink2,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "figure.facecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
    })


def new_fig(t: Theme, L: Layout, body: float, nrows: int = 1, ncols: int = 1, **kw) -> tuple[Figure, Any]:
    """Figure the width of `L`'s column with a body (everything under the header) `body` CSS px tall.

    Returns (fig, ax), or (fig, axes) for small multiples. header() then grows it by the header.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(L.css_width / 72, body / 72), dpi=L.dpi, **kw)
    fig.patch.set_facecolor(t.surface)
    return fig, axes


def get_renderer(fig: Figure) -> RendererAgg:
    """The figure's Agg renderer, for measuring text in display pixels."""
    return cast(FigureCanvasAgg, fig.canvas).get_renderer()


def _root(ax: plt.Axes) -> Figure:
    """The top-level figure of `ax` (these figures have no subfigures)."""
    fig = ax.figure
    assert isinstance(fig, Figure)
    return fig


def style(ax: plt.Axes, t: Theme, ylabel: str = "", xlabel: str = "",
          grid: Literal["x", "y"] | None = "y") -> None:
    """Recessive chrome: no top/right spines, hairline grid on one axis, secondary-ink labels."""
    ax.set_facecolor(t.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.axis)
    ax.tick_params(colors=t.muted, labelcolor=t.ink2, length=0)
    ax.grid(visible=False)
    if grid:
        ax.grid(axis=grid, color=t.grid, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=t.ink2)
    ax.set_xlabel(xlabel, color=t.ink2)


def panel_title(ax: plt.Axes, t: Theme, text: str) -> None:
    """Left-aligned title of one small multiple."""
    ax.set_title(text, color=t.ink, loc="left")


def wrap(fig: plt.Figure, text: str, size: float, width: float, weight: str = "normal") -> str:
    """`text` broken at spaces into lines no wider than `width` CSS px, measured in this figure's font.

    A " / " between alternatives never starts or ends a line.
    """
    text = text.replace(" / ", "\u00a0/\u00a0")
    renderer = get_renderer(fig)
    prop = FontProperties(family=FONTS, size=size, weight=weight)
    limit = width * fig.dpi / 72

    def fits(s: str) -> bool:
        return renderer.get_text_width_height_descent(s, prop, ismath=False)[0] <= limit

    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            trial = f"{cur} {word}" if cur else word
            if cur and not fits(trial):
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
    return "\n".join(lines).replace("\u00a0", " ")


def header(fig: plt.Figure, t: Theme, L: Layout, text: str, sub: str | None = None,
           handles: list | None = None, ncol: int = 1) -> tuple:
    """Title, optional subtitle and optional legend stacked at the top left, wrapped to the column.

    new_fig() sized only the body; this grows the figure by the header's measured height and returns
    the layout rect under it, to pass to px_to_data() and save().
    """
    renderer = get_renderer(fig)
    width = L.css_width - 2 * L.margin
    x = L.margin / L.css_width
    blocks: list[Text | Legend] = [fig.text(x, 1, wrap(fig, text, L.title, width, "semibold"), color=t.ink, fontsize=L.title,
                       fontweight="semibold", va="top", ha="left", linespacing=1.25)]
    gaps = [L.margin - 2]  # above the title (its line box adds ~2 px)
    if sub:
        blocks.append(fig.text(x, 1, wrap(fig, sub, L.sub, width), color=t.ink2, fontsize=L.sub,
                               va="top", ha="left", linespacing=1.35))
        gaps.append(5)
    if handles:
        blocks.append(fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, 1),
                                 bbox_transform=fig.transFigure, ncol=ncol, borderpad=0, borderaxespad=0,
                                 columnspacing=1.6))
        gaps.append(10)
    heights = [b.get_window_extent(renderer).height * 72 / fig.dpi for b in blocks]  # CSS px
    body = fig.get_figheight() * 72
    head = sum(gaps) + sum(heights) + 4  # tight_layout adds its own pad under this
    total = body + head
    fig.set_figheight(total / 72)
    y = 0.0
    for b, gap, h in zip(blocks, gaps, heights):
        y += gap
        if isinstance(b, Legend):
            b.set_bbox_to_anchor((x, 1 - y / total), transform=fig.transFigure)
        else:
            b.set_y(1 - y / total)
        y += h
    return (0, 0, 1, 1 - head / total)


def slot_lim(first: float, last: float, half: float = 0.3) -> tuple[float, float]:
    """Axis limits framing bar slots first..last the way autoscale framed the original 0.6-wide bars."""
    lo, hi = first - half, last + half
    margin = 0.05 * (hi - lo)
    return lo - margin, hi + margin


def tight(fig: plt.Figure, rect: tuple | None = None) -> None:
    """The one layout call, shared by px_to_data() and save() so both see the same axes boxes."""
    fig.tight_layout(pad=PAD, h_pad=PANEL_PAD, w_pad=PANEL_PAD, rect=rect)


# Axes boxes of the last px_to_data() layout per figure: save() checks its own layout still matches, so
# a mark sized in px cannot silently drift (an artist added later must stay inside the axes or set
# in_layout=False).
_SIZED: WeakKeyDictionary[Figure, list] = WeakKeyDictionary()


def _boxes(fig: Figure) -> list[tuple[float, ...]]:
    """Every axes box (x0, y0, width, height) in PNG pixels."""
    w, h = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
    return [(x0 * w, y0 * h, bw * w, bh * h) for x0, y0, bw, bh in (a.get_position().bounds for a in fig.axes)]


def px_to_data(ax: plt.Axes, css: float, axis: str = "x", rect: tuple | None = None) -> float:
    """Data units that render as `css` CSS px along `axis` at display size.

    Lays the figure out exactly as save() will (pass the same rect), so call it after the
    limits, ticks, labels, header and legends are set and before drawing the marks it sizes.
    """
    fig = _root(ax)
    tight(fig, rect)
    _SIZED[fig] = _boxes(fig)
    n_px = css * fig.dpi / 72
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(0.0, 0.0), (n_px, n_px)])
    return abs(x1 - x0) if axis == "x" else abs(y1 - y0)


def name_above(ax: plt.Axes, t: Theme, name: str, y_top: float) -> None:
    """Category name over its bars (phone layout of horizontal bars): left-aligned at x = 0.

    `y_top` is the data y of the upper edge of the row's top bar; the axis must be inverted
    (first row on top) or the name lands under the bar.
    """
    ax.annotate(name, (0, y_top), xytext=(0, NAME_GAP), textcoords="offset points", ha="left", va="bottom",
                color=t.ink2, annotation_clip=False)


def hbar_axis(ax: plt.Axes, L: Layout, names: list[str], rect: tuple | None) -> tuple[float, float]:
    """Row axis for horizontal bars, first row on top: (data units per CSS px on y, name line height).

    Desktop: the names are y tick labels. Phone: they go above each row (name_above), so the y ticks,
    the left spine and the grid go (every bar there carries its value). Call after the x limits,
    chrome and header are set; this lays the figure out.
    """
    n = len(names)
    ax.set_ylim(n - 0.5, -0.5)  # inverted: first row on top
    ax.set_yticks([] if L.mobile else range(n), [] if L.mobile else names)
    if L.mobile:
        ax.spines["left"].set_visible(False)
        ax.grid(visible=False)
    return px_to_data(ax, 1, axis="y", rect=rect), (L.text * 1.35 + NAME_GAP if L.mobile else 0)


def hbar_rows(ax: plt.Axes, t: Theme, L: Layout, names: list[str], values: list[list], colors: list[list],
              fmt: str, rect: tuple | None) -> None:
    """Horizontal bars, one row per name, first row on top; a row's bars stack downward 2 px apart.

    values[i] lists row i's bars (None skips one) and colors[i] their fills; each bar end carries
    its value. Names as in hbar_axis().
    """
    unit, name_h = hbar_axis(ax, L, names, rect)
    for i, (name, vals, cols) in enumerate(zip(names, values, colors)):
        bars = [(v, c) for v, c in zip(vals, cols) if v is not None]
        group = len(bars) * L.bar + (len(bars) - 1) * GAP
        top = i + (name_h - group) / 2 * unit  # name + bars centred on the slot
        for j, (v, c) in enumerate(bars):
            y = top + (j * (L.bar + GAP) + L.bar / 2) * unit
            ax.barh(y, v, height=L.bar * unit, color=c)
            ax.annotate(fmt.format(v), (v, y), xytext=(4, 0), textcoords="offset points", va="center",
                        color=t.ink)
        if L.mobile:
            name_above(ax, t, name, top)


def label_inside(ax: plt.Axes, t: Theme, x: float, y: float, text: str, fill: str, span: float,
                 pad: float = 5) -> bool:
    """Label centred in a bar segment `span` data units wide; dropped (False) unless it fits with `pad` CSS px a side.

    Call after the layout is final (px_to_data), since it measures the segment in pixels.
    """
    fig = _root(ax)
    label = ax.text(x, y, text, ha="center", va="center", color=on_fill(t, fill))
    span_px = abs(ax.transData.transform((span, 0))[0] - ax.transData.transform((0, 0))[0])
    if label.get_window_extent(get_renderer(fig)).width + 2 * pad * fig.dpi / 72 > span_px:
        label.remove()
        return False
    return True


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
    return {"markeredgecolor": t.surface, "markeredgewidth": RING}


def line_key(color: str, label: str, dashed: bool = False) -> Line2D:
    """Legend handle for a line series: line and marker without the ring, which would cut the short key."""
    return Line2D([], [], color=color, linewidth=LINE, marker="o", markersize=MARKER - RING,
                  linestyle=DASH if dashed else "-", label=label)


def save(fig: plt.Figure, t: Theme, L: Layout, name: str, rect: tuple | None = None, layout: bool = True) -> None:
    """Lay out (unless `layout` is False: a schematic placed by hand) and write
    docs/figures/<name><layout suffix><theme suffix>.png."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}{L.suffix}{t.suffix}.png"
    if layout:
        tight(fig, rect)
        sized = _SIZED.get(fig)
        if sized and any(abs(a - b) > 0.5 for old, new in zip(sized, _boxes(fig)) for a, b in zip(old, new)):
            raise ValueError(f"{path.name}: the final layout moved the axes after marks were sized in px")
    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor=t.surface)
    plt.close(fig)
    img = Image.open(buf).convert("RGB")
    if img.width != L.width_px:
        raise ValueError(f"{path.name}: {img.width} px wide, the {L.name} layout needs {L.width_px}")
    # 256-colour palette, no dithering: flat fills and anti-aliased text survive it (on the 2026-10-22
    # render no file's dominant colours moved and text crops looked identical) and the files shrink ~60%.
    img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
    logger.info("wrote %s (%d x %d)", path.name, img.width, img.height)
