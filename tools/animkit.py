"""Drawing and GIF helpers for the README animations (tools/make_animations.py).

A scene is one figure in figstyle's layout and theme whose body is an axes in CSS px (y down, as in
make_figures3.fig_hardware); the scene moves a few artists per frame and grabs the canvas. write_gif()
maps the frames to one palette that holds the theme's tokens exactly, without dithering, and lets
Pillow store each frame as the rectangle that changed with its unchanged pixels transparent; it then
decodes the file again and checks every displayed frame pixel for pixel.
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text
from PIL import Image, ImageSequence

from figstyle import Layout, Theme, header, new_fig, on_fill, wrap

FRAME_MS = 80  # 12.5 frames per second
COLORS = 255  # palette entries per file, the theme's tokens first; one index stays free for transparency


def n_lines(s: str) -> int:
    """Lines in a wrapped text."""
    return s.count("\n") + 1


def ease(u: float) -> float:
    """Smoothstep of u clipped to [0, 1]."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Probe:
    """Measures and wraps text in layout L's font before the figure exists (its height depends on it)."""

    def __init__(self, t: Theme, L: Layout) -> None:
        self.fig, _ = new_fig(t, L, 10)
        self.L = L

    def wrap(self, s: str, width: float, size: float | None = None, weight: str = "normal") -> str:
        """`s` broken into lines no wider than `width` CSS px (figstyle.wrap)."""
        return wrap(self.fig, s, size or self.L.text, width, weight)

    def lines(self, s: str, width: float, size: float | None = None, weight: str = "normal") -> int:
        """How many lines `s` takes at `width` CSS px."""
        return n_lines(self.wrap(s, width, size, weight))

    def close(self) -> None:
        """Release the scratch figure."""
        plt.close(self.fig)


class Board:
    """A figure whose body is one axes in CSS px, y down from the top of the body."""

    def __init__(self, t: Theme, L: Layout, body: float, title: str, sub: str) -> None:
        self.t, self.L = t, L
        self.fig, self.ax = new_fig(t, L, body)
        self.ax.axis("off")
        rect = header(self.fig, t, L, title, sub)
        self.ax.set_position((0, 0, 1, rect[3]))
        self.ax.set_xlim(0, L.css_width)
        self.ax.set_ylim(body, 0)

    def box(self, x: float, y: float, w: float, h: float, fc: str, r: float = 6.0, z: float = 1.0,
            **kw: Any) -> FancyBboxPatch:
        """A rounded box with its top-left corner at (x, y); `kw` goes to FancyBboxPatch (ec, lw, ls)."""
        kw.setdefault("ec", "none")
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, zorder=z, **kw)
        self.ax.add_patch(p)
        return p

    def slot(self, x: float, y: float, w: float, h: float) -> FancyBboxPatch:
        """An empty token slot: a dashed outline."""
        return self.box(x, y, w, h, "none", r=4, z=3, ec=self.t.muted, lw=1.2, ls=(0, (3, 2)))

    def text(self, x: float, y: float, s: str, color: str | None = None, size: float | None = None,
             weight: str = "normal", ha: str = "left", va: str = "top", z: float = 6.0) -> Text:
        """Text anchored at its top-left corner by default, in ink unless `color` is given."""
        return self.ax.text(x, y, s, color=color or self.t.ink, fontsize=size or self.L.text, fontweight=weight,
                            ha=ha, va=va, zorder=z, linespacing=1.35)

    def wrap(self, s: str, width: float) -> str:
        """wrap() in this figure, for text set per frame."""
        return wrap(self.fig, s, self.L.text, width)

    def arrow(self, p: tuple[float, float], q: tuple[float, float], **kw: Any) -> FancyArrowPatch:
        """A muted arrow from p to q; `scale` and `lw` size it, other `kw` go to FancyArrowPatch."""
        a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=kw.pop("scale", 14), color=self.t.muted,
                            linewidth=kw.pop("lw", 1.6), shrinkA=0, shrinkB=0, zorder=2, **kw)
        self.ax.add_patch(a)
        return a

    def grab(self) -> Image.Image:
        """The current frame as an RGB image, checked against the layout's pixel width."""
        canvas = cast(FigureCanvasAgg, self.fig.canvas)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        img = Image.fromarray(rgba[..., :3].copy())
        if img.width != self.L.width_px:
            raise ValueError(f"frame is {img.width} px wide, the {self.L.name} layout needs {self.L.width_px}")
        return img


class Chip:
    """A token: a rounded box with an optional label, moved and recoloured per frame."""

    def __init__(self, b: Board, w: float, h: float, z: float = 5.0) -> None:
        self.b, self.w, self.h = b, w, h
        self.patch = b.box(0, 0, w, h, b.t.muted, r=4, z=z)
        self.label = b.text(0, 0, "", weight="semibold", ha="center", va="center", z=z + 0.01)  # under the next chip

    def set(self, x: float, y: float, fill: str | None, label: str = "", alpha: float = 1.0,
            scale: float = 1.0) -> None:
        """Top-left corner at (x, y), scaled about its centre; `fill` None (or alpha ~0) hides it."""
        if fill is None or alpha <= 0.02:
            self.patch.set_visible(False)
            self.label.set_visible(False)
            return
        self.patch.set_visible(True)
        self.label.set_visible(bool(label))
        w, h = self.w * scale, self.h * scale
        self.patch.set_bounds(x + (self.w - w) / 2, y + (self.h - h) / 2, w, h)
        self.patch.set_facecolor(fill)
        self.patch.set_alpha(alpha)
        self.label.set_text(label)
        self.label.set_position((x + self.w / 2, y + self.h / 2))
        self.label.set_color(on_fill(self.b.t, fill))
        self.label.set_alpha(alpha)


class Tape:
    """Bytes in reading order, one segment each (the model's layers, then its head), with a read overlay
    and a cursor; drawn on a blue box, so the unread bytes are surface-coloured."""

    def __init__(self, b: Board, x: float, y: float, w: float, h: float, segments: list[int]) -> None:
        self.x, self.w = x, w
        edges = np.cumsum([0, *segments]) / sum(segments) * w + x
        for a, c in pairwise(edges):
            b.ax.add_patch(Rectangle((a + 0.75, y), c - a - 1.5, h, fc=b.t.surface, ec="none", zorder=2))
        self.split = float(edges[-2])  # where the last segment starts
        self.read = Rectangle((x, y), 0, h, fc=b.t.blue, alpha=0.6, ec="none", zorder=2.5)
        self.cursor = Rectangle((x, y - 4), 3, h + 8, fc=b.t.yellow, ec="none", zorder=3)
        b.ax.add_patch(self.read)
        b.ax.add_patch(self.cursor)

    def set(self, frac: float, cursor: bool) -> None:
        """Mark the first `frac` of the bytes as read; show the cursor at that point if `cursor`."""
        self.read.set_width(frac * self.w)
        self.cursor.set_x(self.x + frac * self.w - 1.5)
        self.cursor.set_visible(cursor)


# ---------------------------------------------------------------- GIF writing
def theme_colors(t: Theme) -> list[int]:
    """The theme's tokens as 0xRRGGBB, each once, in token order."""
    out: list[int] = []
    for h in (t.surface, t.ink, t.ink2, t.muted, t.grid, t.axis, t.blue, t.orange, t.aqua, t.yellow, t.magenta):
        if int(h[1:], 16) not in out:
            out.append(int(h[1:], 16))
    return out


def rgb_codes(a: np.ndarray) -> np.ndarray:
    """An RGB array as one 0xRRGGBB integer per pixel."""
    a = a.astype(np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def build_palette(frames: list[Image.Image], t: Theme) -> np.ndarray:
    """Up to COLORS entries: the theme tokens exactly (flat fills keep their colour), then a median cut of
    every other colour in `frames` (anti-aliased edges, the read overlay), each weighted by 1 + log2 of its
    pixel count so that rare edge blends get entries too. On anim_speculative (desktop, light) this keeps
    99.9% of pixels within 6 levels of the render; square-root weights over every third frame with 96
    entries left 18."""
    fixed = theme_colors(t)
    counts: dict[int, int] = {}
    for fr in frames:
        u, c = np.unique(rgb_codes(np.asarray(fr)), return_counts=True)
        for k, v in zip(u.tolist(), c.tolist()):
            counts[k] = counts.get(k, 0) + v
    rest = {k: v for k, v in counts.items() if k not in fixed}
    free = COLORS - len(fixed)
    if len(rest) <= free:
        extra = list(rest)
    else:
        keys = np.fromiter(rest.keys(), dtype=np.uint32, count=len(rest))
        weights = np.ceil(1 + np.log2(np.fromiter(rest.values(), dtype=np.float64, count=len(rest)))).astype(np.int64)
        px = np.repeat(keys, weights)
        img = Image.fromarray(np.stack([px >> 16 & 255, px >> 8 & 255, px & 255], -1).astype(np.uint8)[None])
        q = img.quantize(colors=free, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        pal = (q.getpalette() or [])[: 3 * free]
        extra = [pal[i] << 16 | pal[i + 1] << 8 | pal[i + 2] for i in range(0, len(pal), 3)]
    codes = list(dict.fromkeys(fixed + extra))
    return np.array([[c >> 16 & 255, c >> 8 & 255, c & 255] for c in codes], dtype=np.int32)


def indexed(frame: Image.Image, pal: np.ndarray) -> tuple[Image.Image, int]:
    """`frame` as a P image on `pal` (every colour to its nearest entry, no dithering), and how many of its
    pixels that moved by more than 16 levels in some channel."""
    a = np.asarray(frame)
    u, inv, cnt = np.unique(rgb_codes(a).ravel(), return_inverse=True, return_counts=True)
    rgb = np.stack([u >> 16 & 255, u >> 8 & 255, u & 255], -1).astype(np.int32)
    near = ((rgb[:, None, :] - pal[None]) ** 2).sum(-1).argmin(1)
    off = int(cnt[np.abs(pal[near] - rgb).max(-1) > 16].sum())
    img = Image.fromarray(near.astype(np.uint8)[inv.ravel()].reshape(a.shape[:2]))
    img.putpalette(pal.astype(np.uint8).ravel().tolist())
    return img, off


def write_gif(frames: list[Image.Image], t: Theme, path: Path) -> tuple[list[Image.Image], float]:
    """Write `frames` as a GIF that loops forever at FRAME_MS per frame; return the indexed frames and the
    share of pixels the palette moved by more than 16 levels.

    The file is decoded again and must show exactly the indexed frames (Pillow merges a run of identical
    frames into one longer frame), last for the right total time and loop forever.
    """
    pal = build_palette(frames, t)
    pairs = [indexed(fr, pal) for fr in frames]
    idx = [p[0] for p in pairs]
    idx[0].save(path, save_all=True, append_images=idx[1:], duration=FRAME_MS, loop=0, optimize=True, disposal=1)
    shown = []  # indices of the frames a viewer sees: each run of identical frames counts once
    for i, fr in enumerate(idx):
        if i == 0 or not np.array_equal(np.asarray(fr), np.asarray(idx[i - 1])):
            shown.append(i)
    total = n = 0
    with Image.open(path) as im:
        if im.info.get("loop") != 0:
            raise ValueError(f"{path.name}: the GIF does not loop forever")
        for n, fr in enumerate(ImageSequence.Iterator(im), start=1):
            total += fr.info.get("duration", 0)
            want = pal[np.asarray(idx[shown[n - 1]])] if n <= len(shown) else None
            if want is None or not np.array_equal(np.asarray(fr.convert("RGB")), want):
                raise ValueError(f"{path.name}: frame {n} does not decode to its source frame")
    if n != len(shown) or total != len(frames) * FRAME_MS:
        raise ValueError(f"{path.name}: {n} frames over {total} ms, expected {len(shown)} over "
                         f"{len(frames) * FRAME_MS} ms")
    return idx, sum(p[1] for p in pairs) / (len(frames) * frames[0].width * frames[0].height)


def save_qa(frames: list[Image.Image], L: Layout, stem: str, qa: Path, picks: list[int]) -> None:
    """Sampled frames as the GIF shows them: full size and downscaled to the CSS column width."""
    qa.mkdir(parents=True, exist_ok=True)
    for i in picks:
        fr = frames[i].convert("RGB")
        fr.save(qa / f"{stem}_f{i:03d}.png")
        w = L.css_width
        fr.resize((w, round(fr.height * w / fr.width)), Image.Resampling.LANCZOS).save(qa / f"{stem}_f{i:03d}_{w}.png")
