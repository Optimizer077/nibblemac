"""Render the nibblepi logo (docs/assets/logo.svg, logo-dark.svg, logo.png) and the social banner.

The mark is a white pi on a rounded square with three stepped bars: "pi" + "quant" (quantization
levels). Shapes only, no text, so the SVG renders the same everywhere; colours are the figure
palette's blue and orange (tools/figstyle.py). The banner (docs/assets/social-preview.png,
1280x640, GitHub's social-preview size) adds the name and tagline.

    uv run --no-project --with matplotlib python tools/make_logo.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

logger = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
SVG_META = {"Date": None, "Creator": None}  # deterministic SVG output
PNG_META = {"Software": None}
# (tile, pi, steps): light tile matches the light figures' blue, dark a step brighter for dark pages
THEMES = {"": ("#2a78d6", "#ffffff", "#eb6834"), "-dark": ("#3987e5", "#ffffff", "#f08a5d")}


def draw_mark(ax: plt.Axes, tile: str, ink: str, accent: str, x0: float = 0.0, y0: float = 0.0,
              s: float = 1.0) -> None:
    """The mark in a 160 x 160 unit box at (x0, y0), scaled by s; y points up."""
    def r(x: float, y: float, w: float, h: float, color: str, rad: float) -> None:
        ax.add_patch(FancyBboxPatch((x0 + s * x, y0 + s * y), s * w, s * h, color=color, linewidth=0,
                                    boxstyle=f"round,pad=0,rounding_size={s * rad}"))

    def stroke(points: list[tuple[float, float]], codes: list[int]) -> None:
        verts = [(x0 + s * x, y0 + s * y) for x, y in points]
        ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, edgecolor=ink, capstyle="round",
                               joinstyle="round", linewidth=16 * s * 72 / 100))  # 16 units, 100-dpi canvas

    r(0, 0, 160, 160, tile, 36)  # tile
    r(30, 104, 96, 16, ink, 8)  # pi: top bar
    # left leg: one smooth quadratic curve from under the bar down and out to the left
    stroke([(58, 110), (58, 64), (44, 38)], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    stroke([(96, 110), (96, 40)], [MplPath.MOVETO, MplPath.LINETO])  # right leg: straight stem
    for i, h in enumerate((10, 19, 28)):  # three quantization steps, bottom right, clear of the stem
        r(113 + 11 * i, 26, 8, h, accent, 1.5)


def logo(suffix: str) -> None:
    tile, ink, accent = THEMES[suffix]
    fig = plt.figure(figsize=(1.6, 1.6), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 160)
    ax.axis("off")
    fig.patch.set_alpha(0)
    draw_mark(ax, tile, ink, accent)
    fig.savefig(OUT / f"logo{suffix}.svg", transparent=True, metadata=SVG_META)
    if not suffix:
        fig.savefig(OUT / "logo.png", dpi=320, transparent=True, metadata=PNG_META)  # 512 px
    plt.close(fig)


def banner() -> None:
    """1280 x 640 social preview: mark, name, tagline (text is fine in a PNG)."""
    tile, ink, accent = THEMES[""]
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 640)
    ax.axis("off")
    fig.patch.set_facecolor("#fcfcfb")
    draw_mark(ax, tile, ink, accent, x0=110, y0=200, s=1.5)
    font = {"family": ["Segoe UI", "DejaVu Sans", "sans-serif"]}
    ax.text(410, 380, "nibblepi", fontsize=78, fontweight="semibold", color="#0b0b0b", va="center", **font)
    ax.text(414, 290, "Qwen3.5 at 20 tokens/s on a Raspberry Pi 4", fontsize=27, color="#0b0b0b",
            va="center", **font)
    ax.text(414, 240, "From-scratch C + NEON engine · exact speculative decoding · long context",
            fontsize=19, color="#52514e", va="center", **font)
    fig.savefig(OUT / "social-preview.png", facecolor=fig.get_facecolor(), metadata=PNG_META)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in THEMES:
        logo(suffix)
    banner()
    logger.info("wrote %s", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
