"""Shared style for the README figures (tools/make_figures*.py).

Validated dataviz reference palette on the light surface: series 1 blue, 2 orange, 3 aqua;
ink tokens for text; recessive grid and axes. Categorical hues are assigned in fixed order.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]  # blue sequential ramp


def setup() -> None:
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    plt.rcParams["text.color"] = INK
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.titlesize"] = 11.5
    plt.rcParams["legend.fontsize"] = 9


def style(ax: plt.Axes, ylabel: str = "", xlabel: str = "", grid: str = "y") -> None:
    """Recessive chrome: no top/right spines, hairline grid on one axis, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    ax.grid(visible=False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
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


def title(ax: plt.Axes, text: str, sub: str | None = None) -> None:
    """Left-aligned title with an optional secondary line in muted ink."""
    ax.set_title(text, color=INK, loc="left", pad=16 if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=MUTED, fontsize=8.5, va="bottom")
