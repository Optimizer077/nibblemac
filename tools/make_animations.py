"""Animated README figures: one token's trip through the Pi 4 (anim_decode) and one speculative round
(anim_speculative). Four looping GIFs per animation in docs/figures/, desktop and phone layouts in the
light and dark themes (`<name>.gif`, `-dark`, `-mobile`, `-mobile-dark`), 12.5 frames/s, under 1.5 MB each.

Every number is measured in this repository and names its source next to the constant (the hardware
and model constants come from make_figures2/3, which cite theirs); only the pace is slowed down for
viewing, and each animation says so. Drawing and GIF helpers: tools/animkit.py; style: tools/figstyle.py.

    uv run --no-project --with matplotlib python tools/make_animations.py [--only decode] [--qa DIR]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from animkit import (
    FRAME_MS,
    Board,
    Chip,
    Probe,
    Tape,
    ease,
    n_lines,
    save_qa,
    write_gif,
)
from figstyle import LAYOUTS, OUT, THEMES, Layout, Theme, on_fill, setup
from make_figures2 import BYTES, STEP_PLAIN
from make_figures3 import (
    BYTES_PLAIN,
    CLOCK_HZ,
    CORES,
    L1D_KB,
    L2_BYTES,
    MEMBAND_GBS,
    PLAIN_TPS,
    STATE_BYTES,
    STATE_CHUNK,
)

logger = logging.getLogger(__name__)
MAX_BYTES = 1_500_000  # size budget per file

# ---------------------------------------------------------------- data: one plain decode step (Q4K)
# Bytes a plain step reads: make_figures2.BYTES, summed over the .pq tensor directory of the shipped
# model. The per-layer shares divide exactly: 18 linear-attention mixers of 6,019,712 B, 6 full-attention
# mixers of 4,134,912 B, 24 MLPs of 6,197,248 B (each layer's norms counted with them, as in BYTES).
N_LAYERS = 24
FULL_ATTN = frozenset({3, 7, 11, 15, 19, 23})  # 0-based ids of the full-attention layers (ARCHITECTURE.md §1)
N_FULL = len(FULL_ATTN)
MIX_LIN, MIX_FULL, MLP = BYTES[1][1] // (N_LAYERS - N_FULL), BYTES[2][1] // N_FULL, BYTES[3][1] // N_LAYERS
HEAD = BYTES[0][1]  # tied output head + final norm
VOCAB = 248_320  # rows of the tied head (ARCHITECTURE.md §1)
SEGMENTS = [(MIX_FULL if i in FULL_ATTN else MIX_LIN) + MLP for i in range(N_LAYERS)] + [HEAD]
assert sum(SEGMENTS) == BYTES_PLAIN, "per-layer shares must add up to the bytes of a plain step"
PLAIN_MS = 1e3 / PLAIN_TPS  # 126.7 ms: plain decode at 7.89 tok/s (speed-ab-merge4_20261022_103609)
CEILING_TPS = MEMBAND_GBS[1] * 1e3 / (BYTES_PLAIN / 1e6)  # 4.20 GB/s / 424.9 MB = 9.9 tok/s
# Share of a plain step spent streaming weights (matmuls): 118.6 of 128.3 ms in the per-step profile
# (make_figures2.STEP_PLAIN, from engine/CORE.md). The clock runs at this share while the weights stream
# and spends the rest (recurrence, norms, attention, argmax) as the last blocks arrive.
STREAM_SHARE = STEP_PLAIN[0][1] / sum(ms for _, ms in STEP_PLAIN)

# ---------------------------------------------------------------- data: one speculative round
# Fixed k = 4 on prompt64 (prose) with the release kernels, which verify 5 positions in one register pass:
# bench/results/mm2-e2e-decode_20261022_140420.json, binary pinfer_run_mm2, medians of its 3 interleaved
# runs (engine/bench/KERNELS.md Part 3, "End to end").
PLAIN_TPS_SAME_JOB = 7.877  # plain decode in the same job
SPEC_PLAIN_MS = 1e3 / PLAIN_TPS_SAME_JOB  # 127.0 ms
VERIFY_MS = 176.45  # one verify pass over 5 positions: the newest token + 4 drafts
MTP_ROUND_MS = 32.87  # 4 MTP passes per round: 3 chained drafts + the pass after the commit (the next d1)
COMMIT_MS = 2.07  # acceptance argmaxes + deferred state commit (engine/VERIFY.md §2)
TOKENS_PER_ROUND = 3.9375  # 63 tokens in 16 rounds
K4_TPS = 18.64
MTP_MS = MTP_ROUND_MS / 4  # 8.2 ms per draft (engine/DRAFT.md §1b: 8.4 ms in an earlier job)
ROUND_MS = MTP_ROUND_MS + VERIFY_MS + COMMIT_MS  # 211.4 ms, = 3.94 tokens / 18.64 tok/s
VERIFY_RATIO = VERIFY_MS / SPEC_PLAIN_MS  # 1.39 (4 positions: 155.6 ms, 1.23x, same file)
# Drafts in the animated round: the measured fixed k above, and also the default's cap
# (pinfer_default_spec_params() = adaptive k <= 4 + n-gram 7, docs/release-0.2.0.md).
K_MAX = 4
CONFIDENCE = 0.6  # adaptive k drafts again while the newest draft's top-1 probability is >= 0.6 (DRAFT.md §2d)
# Prompt lookup (DRAFT.md §3b): a match of >= 16 tokens drafts up to `ngram` (7 by default); a match of
# 4-15 drafts at most 3 tokens, and only when its first token equals the MTP head's d1; shorter is ignored.
NGRAM_MAX, NGRAM_LONG, NGRAM_SHORT, NGRAM_SHORT_MAX = 7, 16, 4, 3
DEFAULT_TPS = 20.22  # the default on prompt64 (kcap-merge6_20261022_145226)
CHAT_TPS = (11.0, 14.8)  # six chat answers, adaptive with a cap of 3; a cap of 4 is within ±1% (README §3)
# The animated round drafts 4 tokens and the model keeps 3 of them plus 1 of its own: 4 tokens, close to
# the measured average of 3.94 per round. It is an example round, not a recorded trace.
ACCEPTED = 3
assert ACCEPTED == K_MAX - 1, "the round is drawn with exactly one rejected draft, the last"


# ---------------------------------------------------------------- animation 1: a plain decode step
D_PRE, D_SWEEP, D_DRAIN, D_EMIT, D_HOLD, D_RESET = 3, 72, 12, 3, 16, 7


class Decode:
    """Weight blocks stream from DRAM through the L2 into the four cores, layer after layer; a token comes out."""

    name = "anim_decode"

    def __init__(self, t: Theme, L: Layout) -> None:
        self.t, self.L = t, L
        desk = not L.mobile
        m, W, lh, pad = L.margin, L.css_width, L.text * 1.35, 12.0
        mb = BYTES_PLAIN / 1e6
        vsize = 20.0 if desk else 17.0
        vh = vsize * 1.3
        self.f_sweep = D_PRE
        self.f_emit = D_PRE + D_SWEEP + D_DRAIN
        self.f_reset = self.f_emit + D_EMIT + D_HOLD
        self.n = self.f_reset + D_RESET
        slow = (D_SWEEP + D_DRAIN) * FRAME_MS / PLAIN_MS
        title = "One token's trip through the Pi 4"
        sub = (f"Every generated token streams all {mb:.1f} MB of weights from DRAM through the {L2_BYTES >> 20} MB "
               f"L2 into the four cores, layer after layer (shown about {round(slow, -1):.0f}× slower)")
        if desk:
            labels = {"out": "generated text", "layer": "layer being read",
                      "read": f"weights read · {mb:.1f} MB per token", "time": f"time · {PLAIN_MS:.1f} ms per token"}
            self.details = {"lin": "Gated DeltaNet (linear attention) + MLP, {:.1f} MB",
                            "full": "full attention + MLP, {:.1f} MB",
                            "head": f"{VOCAB:,} × 1024, tied to the embedding, {HEAD / 1e6:.1f} MB",
                            "argmax": f"over {VOCAB:,} logits picks the next token"}
            out_note = f"+1 token after {PLAIN_MS:.1f} ms: {PLAIN_TPS:.2f} tokens/s"
        else:
            labels = {"out": "output", "layer": "layer", "read": f"read · {mb:.0f} MB/token",
                      "time": f"time · {PLAIN_MS:.0f} ms/token"}
            self.details = {"lin": "linear attention + MLP", "full": "full attention + MLP",
                            "head": f"{VOCAB:,} × 1024", "argmax": f"over {VOCAB:,} logits"}
            out_note = "+1 token"
        lw = 470.0 if desk else W - 2 * m  # the hardware column
        x0 = m
        pr = Probe(t, L)
        # ---- counters: right of the hardware on desktop, above it on the phone (2 x 2)
        if desk:
            cx = x0 + lw + 34
            cw = W - m - cx
        else:
            cw = (W - 2 * m - 16) / 2
        n_detail = max(pr.lines(d.format(12.2), cw) for d in self.details.values())
        chip_w, chip_h, chip_gap, n_chips = (30.0, 22.0, 8.0, 6) if desk else (22.0, 18.0, 6.0, 5)
        out_note = pr.wrap(out_note, cw)

        def cell_h(key: str) -> float:
            if key == "out":
                return lh + 6 + chip_h + 6 + (out_note.count("\n") + 1) * lh
            return lh + 2 + vh + 2 + (n_detail * lh if key == "layer" else 0)

        cells: dict[str, tuple[float, float]] = {}
        y = 0.0
        if desk:
            for k in ("out", "layer", "read", "time"):
                cells[k] = (cx, y)
                y += cell_h(k) + 16
            stats_h = 0.0
        else:
            y = 10.0  # air between the subtitle and the counters' labels
            for row in (("layer", "read"), ("time", "out")):
                for j, k in enumerate(row):
                    cells[k] = (x0 + j * (cw + 16), y)
                y += max(cell_h(k) for k in row) + 12
            stats_h = y + 4
        # ---- the hardware column: cores, L2, DRAM; data flows up
        c_head = f"{CORES} × Cortex-A72 cores at {CLOCK_HZ / 1e9:.1f} GHz"
        c_sub = pr.wrap(f"kept in each core's {L1D_KB} KB L1: a {STATE_CHUNK // 1024} KB chunk of the recurrent "
                        "state and the int8 activations", lw - 2 * pad)
        l2_txt = pr.wrap(f"L2 cache · {L2_BYTES >> 20} MB, shared · each weight byte passes through once",
                         lw - 2 * pad, weight="semibold")
        d_head = pr.wrap(f"LPDDR4 DRAM · 4 GB · read at {MEMBAND_GBS[1]:.2f} GB/s (measured)", lw - 2 * pad,
                         weight="semibold")
        d_note = pr.wrap(f"plus the {STATE_BYTES / 1e6:.1f} MB recurrent state, read and written back per token",
                         lw - 2 * pad)
        footer = pr.wrap(f"Speed ≈ bandwidth ÷ bytes per token: {MEMBAND_GBS[1]:.2f} GB/s ÷ {mb:.1f} MB = at most "
                         f"{CEILING_TPS:.1f} tokens/s; pinfer reaches {PLAIN_TPS:.2f} tokens/s "
                         f"({PLAIN_MS:.1f} ms per token).", W - 2 * m, weight="semibold")
        pr.close()
        tiles = (f"state {STATE_CHUNK // 1024} KB", "int8 acts") if desk else ("state", "int8")
        tile_chip = 20.0
        tile_h = 5 + lh + 4 + 2 * tile_chip + 4 + 6
        y_c = stats_h
        cores_h = pad + lh + n_lines(c_sub) * lh + 8 + tile_h + pad
        y_l2 = y_c + cores_h + 48
        l2_h = 2 * pad + n_lines(l2_txt) * lh
        y_d = y_l2 + l2_h + 56
        tape_h = 24.0
        dram_h = pad + n_lines(d_head) * lh + 8 + tape_h + 6 + lh + 2 + n_lines(d_note) * lh + pad
        stats_end = max(cells[k][1] + cell_h(k) for k in cells)
        y_foot = max(y_d + dram_h, stats_end) + 18
        body = y_foot + n_lines(footer) * lh + L.margin

        b = self.b = Board(t, L, body, title, sub)
        on_o, on_a, on_b = on_fill(t, t.orange), on_fill(t, t.aqua), on_fill(t, t.blue)
        b.box(x0, y_c, lw, cores_h, t.orange)
        b.text(x0 + pad, y_c + pad, c_head, color=on_o, weight="semibold")
        b.text(x0 + pad, y_c + pad + lh, c_sub, color=on_o)
        tw = (lw - 2 * pad - 3 * 8) / CORES
        ty = y_c + pad + lh + n_lines(c_sub) * lh + 8
        self.lanes = []  # packets rise under the middle of each core
        for c in range(CORES):
            tx = x0 + pad + c * (tw + 8)
            b.box(tx, ty, tw, tile_h, t.surface, r=4, z=1.5)
            b.text(tx + tw / 2, ty + 5, f"core {c}", weight="semibold", ha="center")
            for j, s in enumerate(tiles):
                cy = ty + 5 + lh + 4 + j * (tile_chip + 4)
                b.box(tx + 5, cy, tw - 10, tile_chip, t.yellow, r=3, z=2)
                b.text(tx + tw / 2, cy + tile_chip / 2, s, color=on_fill(t, t.yellow), ha="center", va="center")
            self.lanes.append(tx + tw / 2)
        b.box(x0, y_l2, lw, l2_h, t.aqua)
        b.text(x0 + pad, y_l2 + pad, l2_txt, color=on_a, weight="semibold")
        b.box(x0, y_d, lw, dram_h, t.blue)
        b.text(x0 + pad, y_d + pad, d_head, color=on_b, weight="semibold")
        y_tape = y_d + pad + n_lines(d_head) * lh + 8
        self.tape = Tape(b, x0 + pad, y_tape, lw - 2 * pad, tape_h, SEGMENTS)
        y_tl = y_tape + tape_h + 6
        b.text(x0 + pad, y_tl, f"{N_LAYERS} layers · {(BYTES_PLAIN - HEAD) / 1e6:.1f} MB", color=on_b)
        b.text((self.tape.split + x0 + lw - pad) / 2, y_tl, f"head · {HEAD / 1e6:.1f} MB", color=on_b, ha="center")
        b.text(x0 + pad, y_tl + lh + 2, d_note, color=on_b)
        # packets: visible only between the boxes; they leave DRAM and enter the cores at a constant speed
        self.gaps = [(y_c + cores_h, y_l2), (y_l2 + l2_h, y_d)]
        self.y_d, self.speed = y_d, (y_d - (y_c + cores_h)) / D_DRAIN
        self.packets = b.ax.scatter(np.empty(0), np.empty(0), s=81, marker="s", c=t.blue, edgecolors=t.surface,
                                    linewidths=1.5, zorder=4)
        b.text(x0, y_foot, footer, weight="semibold")
        # counters
        self.values: dict[str, Text] = {}
        for k, (x, y) in cells.items():
            b.text(x, y, labels[k], color=t.ink2)
            if k == "out":
                self.y_chip = y + lh + 6
                self.slots = [x + i * (chip_w + chip_gap) for i in range(n_chips)]
                self.empty = b.slot(self.slots[-1], self.y_chip, chip_w, chip_h)
                self.chips = [Chip(b, chip_w, chip_h, z=5 + 0.02 * i) for i in range(n_chips)]
                self.note = b.text(x, self.y_chip + chip_h + 6, out_note)
                if desk:  # the token comes out of the cores
                    b.arrow((x0 + lw + 4, self.y_chip + chip_h / 2), (x - 8, self.y_chip + chip_h / 2))
                continue
            self.values[k] = b.text(x, y + lh + 2, "", size=vsize, weight="semibold")
            if k == "layer":
                self.detail = b.text(x, y + lh + 2 + vh + 2, "", color=t.ink2)
                self.detail_w = cw

    def draw(self, f: int) -> None:
        """Set every moving artist for frame `f`."""
        t = self.t
        k = f - self.f_sweep  # sweep frame
        reset = f >= self.f_reset
        frac = 0.0 if reset else min(max((k + 1) / D_SWEEP, 0.0), 1.0)  # of the weights requested
        drain = min(max((k + 1 - D_SWEEP) / D_DRAIN, 0.0), 1.0)  # the last blocks arriving, then the argmax
        ms = 0.0 if reset else (frac * STREAM_SHARE + drain * (1 - STREAM_SHARE)) * PLAIN_MS
        self.tape.set(frac, 0 <= k < D_SWEEP)
        read = frac * BYTES_PLAIN
        if self.f_emit - 2 <= f < self.f_reset:  # the last block has arrived
            value, detail = "argmax", self.details["argmax"]
        else:
            i = int(np.searchsorted(np.cumsum(SEGMENTS), read, side="right")) if read > 0 else 0
            if i >= N_LAYERS:
                value, detail = "output head", self.details["head"]
            else:
                value = f"{i + 1} of {N_LAYERS}"
                detail = self.details["full" if i in FULL_ATTN else "lin"].format(SEGMENTS[i] / 1e6)
        self.values["layer"].set_text(value)
        self.detail.set_text(self.b.wrap(detail, self.detail_w))
        self.values["read"].set_text(f"{read / 1e6:.1f} MB")
        emitted = self.f_emit <= f < self.f_reset
        self.values["time"].set_text(f"{PLAIN_MS:.1f} ms" if emitted else f"≈ {ms:.0f} ms")
        # packets: two per sweep frame, in lanes (0, 2) and (1, 3) alternately
        pts = []
        if not reset:
            for s in range(max(0, k - D_DRAIN), min(k, D_SWEEP - 1) + 1):
                y = self.y_d - (k - s) * self.speed
                if any(lo + 6 < y < hi - 6 for lo, hi in self.gaps):
                    pts += [(self.lanes[s % 2], y), (self.lanes[s % 2 + 2], y)]
        self.packets.set_offsets(np.array(pts) if pts else np.empty((0, 2)))
        # generated tokens: the new one pops in, then the row shifts a slot left and the oldest fades out
        u = ease((f - self.f_reset + 1) / (D_RESET + 1)) if reset else 0.0
        pitch = self.slots[1] - self.slots[0]
        for i, chip in enumerate(self.chips):
            if i == 0:
                chip.set(self.slots[0], self.y_chip, t.muted, alpha=max(0.0, 1 - 2 * u), scale=1 - 0.4 * u)
            elif i < len(self.chips) - 1:
                chip.set(self.slots[i] - u * pitch, self.y_chip, t.muted)
            elif emitted or reset:
                pop = {0: 0.7, 1: 1.1}.get(f - self.f_emit, 1.0) if emitted else 1.0
                chip.set(self.slots[i] - u * pitch, self.y_chip, t.muted if reset else t.magenta, scale=pop)
            else:
                chip.set(0, 0, None)
        self.empty.set_visible(not emitted and not (reset and u < 0.3))  # hidden while the new token covers it
        self.note.set_visible(emitted)


# ---------------------------------------------------------------- animation 2: one speculative round
S_PRE, S_FLY, S_LAND, S_FAN, S_SWEEP, S_VHOLD = 4, 5, 3, 2, 36, 6
S_VERDICT, S_DROP, S_AHOLD, S_SCROLL, S_NEXT, S_HOLD = 4, 6, 10, 9, 6, 14


class Speculative:
    """The MTP head drafts, the full model verifies every draft in one weight sweep, the matching prefix stays."""

    name = "anim_speculative"

    def __init__(self, t: Theme, L: Layout) -> None:
        self.t, self.L = t, L
        desk = not L.mobile
        m, W, lh, pad = L.margin, L.css_width, L.text * 1.35, 12.0
        mb = BYTES_PLAIN / 1e6
        self.f_draft = S_PRE
        self.f_verify = self.f_draft + (K_MAX - 1) * (S_FLY + S_LAND)
        self.f_sweep = self.f_verify + S_FAN
        self.f_accept = self.f_sweep + S_SWEEP + S_VHOLD
        self.f_drop = self.f_accept + K_MAX * S_VERDICT
        self.f_commit = self.f_drop + S_DROP + S_AHOLD
        self.f_next = self.f_commit + S_SCROLL
        self.f_hold = self.f_next + S_NEXT
        self.n = self.f_hold + S_HOLD
        kept = ACCEPTED + 1
        title = "One round of exact speculative decoding"
        sub = (f"Several tokens for one read of the weights ({ROUND_MS / SPEC_PLAIN_MS:.1f}× the time of a plain "
               "step), with output identical to plain greedy decoding. An example round, slowed down; the bars "
               "show real time")
        self.captions = [
            ("1 · Draft", (f"The MTP head, one extra layer, guesses the next tokens, about {MTP_MS:.0f} ms each. "
                           f"By default it drafts again while its top-1 probability is at least {CONFIDENCE}, up "
                           f"to {K_MAX} drafts; here all {K_MAX}.")),
            ("2 · Verify in ONE weight sweep", (f"The full model scores the newest token and all {K_MAX} drafts "
                                                f"together: one read of its {mb:.1f} MB of weights, "
                                                f"{VERIFY_MS:.0f} ms, {VERIFY_RATIO:.2f}× a plain step.")),
            ("3 · Accept the matching prefix", (f"d1–d{ACCEPTED} equal the full model's own greedy choices and "
                                                f"are kept. At d{K_MAX} it disagrees: its own token (+1) is kept "
                                                f"instead and d{K_MAX} is rolled back.")),
            ("4 · Commit and draft again", (f"{kept} tokens for one weight sweep. The rejected draft leaves no "
                                            "trace (its state updates ran on a copy), and one MTP pass over the "
                                            "kept rows drafts the next d1.")),
        ]
        pr = Probe(t, L)
        self.cap_w = W - 2 * m
        cap_h = L.panel * 1.35 + 2 + max(pr.lines(d, self.cap_w) for _, d in self.captions) * lh
        # token strip: committed tokens, the newest token, the drafts
        self.cw, self.ch, self.pitch = (60.0, 30.0, 70.0) if desk else (41.0, 28.0, 47.0)
        self.n_c = 6 if desk else 2
        y_lab = cap_h + 16
        self.y_chip = y_lab + lh + 4
        y_mach = self.y_chip + self.ch + 54
        # machine row: MTP head | full model
        gap = 20.0 if desk else 10.0
        mtp_w = 236.0 if desk else round((W - 2 * m - gap) * 0.4)
        mod_x = m + mtp_w + gap
        mod_w = W - m - mod_x
        mtp_lines = [pr.wrap("MTP draft head", mtp_w - 2 * pad, weight="semibold"),
                     pr.wrap(f"≈ {MTP_MS:.0f} ms per draft", mtp_w - 2 * pad),
                     pr.wrap(f"drafts again while p ≥ {CONFIDENCE}, up to {K_MAX}", mtp_w - 2 * pad)]
        mod_head = pr.wrap(f"Full model: {N_LAYERS} layers + head", mod_w - 2 * pad, weight="semibold")
        self.mod_notes = [f"{mb:.1f} MB of weights", f"one sweep for all {K_MAX + 1} positions",
                          f"one sweep: {VERIFY_MS:.0f} ms = {VERIFY_RATIO:.2f}× a plain step ({SPEC_PLAIN_MS:.0f} ms)"]
        self.mod_note_w = mod_w - 2 * pad
        tape_h = 22.0
        mtp_h = 2 * pad + sum(n_lines(s) for s in mtp_lines) * lh + 4
        mod_h = (2 * pad + n_lines(mod_head) * lh + 8 + tape_h + 6
                 + max(pr.lines(s, self.mod_note_w) for s in self.mod_notes) * lh)
        mach_h = max(mtp_h, mod_h)
        # cost bars: desktop names left of the bars, phone names above them
        y_bars = y_mach + mach_h + 28
        bar_h = float(L.bar)
        if desk:
            bx0, bx1 = m + 262, W - m - 118
            rows_y = [y_bars, y_bars + bar_h + 14]
        else:
            bx0, bx1 = m, W - m - 92
            rows_y = [y_bars + lh + 4, y_bars + 2 * (lh + 4) + bar_h + 12]
        self.scale = (bx1 - bx0) / (K_MAX * SPEC_PLAIN_MS)
        # inset (prompt lookup) and the measured result
        y_low = rows_y[1] + bar_h + 26
        in_w = 400.0 if desk else W - 2 * m
        in_title = pr.wrap("Also: prompt lookup (n-gram drafts)", in_w - 2 * pad, weight="semibold")
        in_txt = pr.wrap(f"When the newest tokens (yellow, drawn short) also occur earlier in the context, the draft "
                         f"(orange) copies what followed them there, with no MTP pass, and is verified in the same "
                         f"single sweep: up to {NGRAM_MAX} tokens after a match of {NGRAM_LONG} or more, at most "
                         f"{NGRAM_SHORT_MAX} after a match of {NGRAM_SHORT}–{NGRAM_LONG - 1} that agrees with the "
                         "MTP head's d1.", in_w - 2 * pad)
        chip_s = 24.0 if desk else 21.0
        in_h = pad + n_lines(in_title) * lh + 22 + chip_s + 8 + n_lines(in_txt) * lh + pad
        note_x = m + in_w + 24 if desk else m
        note = pr.wrap(f"Exact: every kept token is the full model's own greedy choice, so the output is identical "
                       f"to plain greedy decoding. Measured on prose: {TOKENS_PER_ROUND:.2f} tokens per round and "
                       f"{K4_TPS:.2f} tokens/s at k = {K_MAX}, against {PLAIN_TPS_SAME_JOB:.2f} plain; the default "
                       f"(adaptive k ≤ {K_MAX} with prompt lookup) reaches {DEFAULT_TPS:.1f} on prose. Chat answers: "
                       f"{CHAT_TPS[0]:.1f}–{CHAT_TPS[1]:.1f} (measured with a cap of 3).", W - m - note_x)
        pr.close()
        y_note = y_low if desk else y_low + in_h + 16
        body = max(y_low + in_h, y_note + n_lines(note) * lh) + L.margin

        b = self.b = Board(t, L, body, title, sub)
        xs, n_c = self.slot_x, self.n_c
        self.cap_title = b.text(m, 0, "", size=L.panel, weight="semibold")
        self.cap_text = b.text(m, L.panel * 1.35 + 2, "", color=t.ink2)
        b.text(xs(0), y_lab, "text so far", color=t.ink2)
        b.text(xs(n_c) + self.cw / 2, y_lab, "newest token" if desk else "newest", color=t.ink2, ha="center")
        b.text((xs(n_c + 1) + xs(n_c + K_MAX) + self.cw) / 2, y_lab, f"drafts d1–d{K_MAX}", color=t.ink2,
               ha="center")
        self.empty = [b.slot(xs(n_c + j), self.y_chip, self.cw, self.ch) for j in range(1, K_MAX + 1)]
        # machine row
        self.mtp_box = (m, y_mach, mtp_w, mach_h)
        b.box(m, y_mach, mtp_w, mach_h, t.orange)
        on_o, on_b = on_fill(t, t.orange), on_fill(t, t.blue)
        ty = y_mach + pad
        for i, s in enumerate(mtp_lines):
            b.text(m + pad, ty, s, color=on_o, weight="semibold" if i == 0 else "normal")
            ty += n_lines(s) * lh + (4 if i == 0 else 0)
        b.box(mod_x, y_mach, mod_w, mach_h, t.blue)
        b.text(mod_x + pad, y_mach + pad, mod_head, color=on_b, weight="semibold")
        y_tape = y_mach + pad + n_lines(mod_head) * lh + 8
        self.tape = Tape(b, mod_x + pad, y_tape, mod_w - 2 * pad, tape_h, SEGMENTS)
        self.mod_note = b.text(mod_x + pad, y_tape + tape_h + 6, "", color=on_b)
        self.ring = {k: b.box(x - 3, y - 3, w + 6, h + 6, "none", r=8, z=0.9, ec=t.ink, lw=2.0)
                     for k, (x, y, w, h) in (("mtp", self.mtp_box), ("mod", (mod_x, y_mach, mod_w, mach_h)))}
        self.fan = []  # verify: every position goes into the one sweep
        for j in range(K_MAX + 1):
            x_a, x_b = xs(n_c + j) + self.cw / 2, mod_x + (j + 0.5) / (K_MAX + 1) * mod_w
            line = Line2D([x_a, x_b], [self.y_chip + self.ch + 2, y_mach - 3], color=t.blue, linewidth=2.0,
                          zorder=2)
            b.ax.add_line(line)
            self.fan.append(line)
        self.marks = [b.text(xs(n_c + j) + self.cw / 2, self.y_chip + self.ch + 3, "✓" if j <= ACCEPTED else "✗",
                             size=L.panel, weight="semibold", ha="center") for j in range(1, K_MAX + 1)]
        # cost bars: plain decoding of the same tokens (reference) and this round as it runs
        for name, y in zip((f"plain decoding: {kept} tokens = {kept} sweeps", f"speculative: {kept} tokens = 1 sweep"),
                           rows_y):
            if desk:
                b.text(bx0 - 12, y + bar_h / 2, name, color=t.ink2, ha="right", va="center")
            else:
                b.text(m, y - lh - 2, name, color=t.ink2)
        for i in range(kept):
            b.ax.add_patch(Rectangle((bx0 + i * SPEC_PLAIN_MS * self.scale, rows_y[0]),
                                     SPEC_PLAIN_MS * self.scale - 2, bar_h, fc=t.blue, ec="none", zorder=2))
        b.text(bx0 + kept * SPEC_PLAIN_MS * self.scale + 6, rows_y[0] + bar_h / 2, f"{kept * SPEC_PLAIN_MS:.0f} ms",
               va="center")
        self.bx0, self.y_bar, self.bar_h = bx0, rows_y[1], bar_h
        self.segs = [Rectangle((bx0, rows_y[1]), 0, bar_h, fc=c, ec="none", zorder=2)
                     for c in (t.orange, t.blue, t.orange)]  # chained drafts, verify sweep, commit + next d1
        for r in self.segs:
            b.ax.add_patch(r)
        self.clock = b.text(bx0, rows_y[1] + bar_h / 2, "", va="center")
        self._inset(b, m, y_low, in_w, in_h, in_title, in_txt, chip_s)
        b.text(note_x, y_note, note, color=t.ink2)
        # pool of tokens; a later one in the pool is drawn on top, so place them back to front
        self.chips = [Chip(b, self.cw, self.ch, z=5 + 0.02 * i) for i in range(n_c + 2 + K_MAX + 1)]

    def _inset(self, b: Board, x: float, y: float, w: float, h: float, title: str, text: str, s: float) -> None:
        """Static inset: the newest tokens (yellow) found earlier in the context; the draft copies what followed."""
        t, lh, pad = self.t, self.L.text * 1.35, 12.0
        b.box(x, y, w, h, "none", r=6, z=1, ec=t.axis, lw=1.0)
        b.text(x + pad, y + pad, title, weight="semibold")
        seq = ["…", "A", "B", "C", "D", "E", "…", "A", "B", "C", "D", "E"]
        kinds = ["gap", "match", "match", "match", "src", "src", "gap", "match", "match", "match", "draft", "draft"]
        fills = {"match": t.yellow, "src": t.muted, "draft": t.orange}
        cx, y_seq = x + pad, y + pad + (title.count("\n") + 1) * lh + 22
        centers = []
        for ch, kind in zip(seq, kinds):
            if kind == "gap":
                b.text(cx + 6, y_seq + s / 2, ch, color=t.ink2, ha="center", va="center")
                centers.append(cx + 6)
                cx += 16
                continue
            b.box(cx, y_seq, s, s, fills[kind], r=3, z=2)
            b.text(cx + s / 2, y_seq + s / 2, ch, color=on_fill(t, fills[kind]), weight="semibold", ha="center",
                   va="center")
            centers.append(cx + s / 2)
            cx += s + 4
        b.arrow(((centers[4] + centers[5]) / 2, y_seq - 3), ((centers[10] + centers[11]) / 2, y_seq - 3),
                connectionstyle="arc3,rad=-0.2", scale=12, lw=1.4)
        b.text(x + pad, y_seq + s + 8, text, color=t.ink2)

    def slot_x(self, i: float) -> float:
        """Left edge of token slot `i` of the strip, CSS px."""
        return self.L.margin + i * self.pitch

    def ms_at(self, f: int) -> float:
        """The round's clock: chained drafts, the verify sweep, then commit and the next d1."""
        chain = (K_MAX - 1) * MTP_MS
        if f < self.f_verify:
            return (f - self.f_draft + 1) / (self.f_verify - self.f_draft) * chain if f >= self.f_draft else 0.0
        if f < self.f_accept:
            return chain + min(max((f - self.f_sweep + 1) / S_SWEEP, 0.0), 1.0) * VERIFY_MS
        tail = ease((f - self.f_commit + 1) / (self.f_hold - self.f_commit)) if f >= self.f_commit else 0.0
        return chain + VERIFY_MS + tail * (COMMIT_MS + MTP_MS)

    def draw(self, f: int) -> None:
        """Set every moving artist for frame `f`."""
        t, xs, n_c, y, m = self.t, self.slot_x, self.n_c, self.y_chip, self.L.margin
        phase = 0 if f < self.f_verify else 1 if f < self.f_accept else 2 if f < self.f_commit else 3
        self.cap_title.set_text(self.captions[phase][0])
        self.cap_text.set_text(self.b.wrap(self.captions[phase][1], self.cap_w))
        committing = f >= self.f_commit
        scroll = ease((f - self.f_commit + 1) / S_SCROLL) * (ACCEPTED + 1) if committing else 0.0
        mx, my, mw, _ = self.mtp_box
        launch = (mx + mw / 2 - self.cw / 2, my - self.ch / 2)  # drafts leave the MTP head's top edge
        placed: list[tuple[float, float, str | None, str, float, float]] = []  # x, y, fill, label, alpha, scale
        occupied = set()  # draft slots with a token in them

        def at(slot: int, fill: str, label: str = "", alpha: float = 1.0, dy: float = 0.0,
               scale: float = 1.0) -> None:
            x = xs(slot - scroll)
            if x < m:  # scrolled off the left edge: fade out over one slot
                alpha *= min(1.0, max(0.0, 1 + (x - m) / self.pitch))
            placed.append((x, y + dy, fill, label, alpha, scale))
            if not committing and dy == 0:
                occupied.add(slot - n_c)

        def flying(u: float, slot: int, label: str) -> None:
            u = ease(u)
            placed.append((launch[0] + (xs(slot) - launch[0]) * u, launch[1] + (y - launch[1]) * u, t.orange,
                           label, 1.0, 1.0))

        for i in range(n_c):
            at(i, t.muted)
        at(n_c, t.muted if committing else t.magenta)  # the newest token, fed in this round's verify pass
        for j in range(1, K_MAX + 1):
            if j >= 2:  # d1 came from the previous round's last MTP pass; d2.. are chained now
                fly = self.f_draft + (j - 2) * (S_FLY + S_LAND)
                if f < fly:
                    continue
                if f < fly + S_FLY:
                    flying((f - fly + 1) / S_FLY, n_c + j, f"d{j}")
                    continue
            verdict = f >= self.f_accept + (j - 1) * S_VERDICT
            if j <= ACCEPTED:
                fill = t.muted if committing else t.aqua if verdict else t.orange
                at(n_c + j, fill, "" if committing else f"d{j}")
                continue
            drop = min((f - self.f_drop + 1) / S_DROP, 1.0) if f >= self.f_drop else 0.0  # rejected: falls out
            if drop < 1:
                at(n_c + j, t.orange, f"d{j}", alpha=1 - drop, dy=18 * drop)
            mark = self.marks[j - 1]
            mark.set_position((xs(n_c + j) + self.cw / 2, y + self.ch + 3 + 18 * drop))
            mark.set_alpha(1 - drop)
            if f >= self.f_drop:  # the full model's own token takes the slot, in front of the falling draft
                fade = ease((f - self.f_commit + 1) / S_SCROLL) if committing else 0.0
                pop = {0: 0.7, 1: 1.1}.get(f - self.f_drop, 1.0)
                at(n_c + j, t.magenta, "+1" if fade < 0.5 else "", scale=pop)
        if f >= self.f_next:  # the next round's d1
            flying((f - self.f_next + 1) / S_FLY, n_c + 1, "d1")
            if f >= self.f_next + S_FLY - 1:
                occupied.add(1)
        assert len(placed) <= len(self.chips), "the token pool is too small"
        for chip, pl in zip(self.chips, placed + [(0, 0, None, "", 0.0, 1.0)] * len(self.chips)):
            chip.set(pl[0], pl[1], pl[2], pl[3], alpha=pl[4], scale=pl[5])
        for j, slot in enumerate(self.empty, start=1):
            slot.set_visible(j not in occupied)
        for j, mk in enumerate(self.marks, start=1):
            end = self.f_commit if j <= ACCEPTED else self.f_drop + S_DROP
            mk.set_visible(self.f_accept + (j - 1) * S_VERDICT <= f < end)
        # machine: active outlines, the verify fan, the sweep
        self.ring["mtp"].set_visible(self.f_draft <= f < self.f_verify or self.f_next <= f < self.f_hold)
        self.ring["mod"].set_visible(self.f_verify <= f < self.f_accept)
        for line in self.fan:
            line.set_visible(self.f_verify <= f < self.f_accept)
        k = f - self.f_sweep
        self.tape.set(min(max((k + 1) / S_SWEEP, 0.0), 1.0) if f < self.f_accept else 0.0, 0 <= k < S_SWEEP)
        note = 0 if f < self.f_verify else 1 if k < S_SWEEP else 2
        self.mod_note.set_text(self.b.wrap(self.mod_notes[note], self.mod_note_w))
        # this round's bar and clock
        ms = self.ms_at(f)
        chain = (K_MAX - 1) * MTP_MS
        parts = [min(ms, chain), min(max(ms - chain, 0.0), VERIFY_MS), max(ms - chain - VERIFY_MS, 0.0)]
        x = self.bx0
        for i, (r, p) in enumerate(zip(self.segs, parts)):
            w = p * self.scale
            r.set_x(x)
            r.set_width(max(w - (2.0 if i < 2 and parts[i + 1] > 0 else 0.0), 0.0))  # 2 px gap to the next part
            x += w
        self.clock.set_position((x + 6, self.y_bar + self.bar_h / 2))
        self.clock.set_text(f"{ROUND_MS:.0f} ms" if f >= self.f_hold else f"{ms:.0f} ms" if ms > 0 else "")


ANIMATIONS: dict[str, type[Decode | Speculative]] = {"decode": Decode, "speculative": Speculative}


def main() -> None:
    """Render the selected animations (all by default), check them, and fail if one is over budget."""
    ap = argparse.ArgumentParser(description="Render the README animations into docs/figures/.")
    ap.add_argument("--only", choices=sorted(ANIMATIONS))
    ap.add_argument("--layout", choices=[L.name for L in LAYOUTS])
    ap.add_argument("--theme", choices=[t.name for t in THEMES])
    ap.add_argument("--qa", type=Path, help="also write sampled frames, full size and at CSS width, here")
    ap.add_argument("--frames", help="comma-separated frame indices for --qa (default: start, quarters, end)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # semibold falls back to bold in DejaVu
    over = []
    for key, cls in ANIMATIONS.items():
        for L in LAYOUTS:
            for t in THEMES:
                if ((args.only and key != args.only) or (args.layout and L.name != args.layout)
                        or (args.theme and t.name != args.theme)):
                    continue
                setup(t, L)
                scene = cls(t, L)
                frames = []
                for f in range(scene.n):
                    scene.draw(f)
                    frames.append(scene.b.grab())
                plt.close(scene.b.fig)
                OUT.mkdir(parents=True, exist_ok=True)
                path = OUT / f"{cls.name}{L.suffix}{t.suffix}.gif"
                width, height = frames[0].size
                idx, off = write_gif(frames, t, path)
                del frames
                size = path.stat().st_size
                logger.info("wrote %s (%d x %d, %d frames, %.1f s, %.2f MB; palette moved %.4f%% of pixels by "
                            "> 16 levels)", path.name, width, height, scene.n, scene.n * FRAME_MS / 1e3,
                            size / 1e6, 100 * off)
                if size > MAX_BYTES:
                    over.append(path.name)
                if args.qa:
                    n = scene.n
                    picks = ([int(s) for s in args.frames.split(",")] if args.frames
                             else [0, n // 4, n // 2, 3 * n // 4, n - 1])
                    save_qa(idx, L, path.stem, args.qa, [p for p in picks if 0 <= p < n])
    if over:
        raise SystemExit(f"over the {MAX_BYTES / 1e6:.1f} MB budget: {', '.join(over)}")


if __name__ == "__main__":
    main()
