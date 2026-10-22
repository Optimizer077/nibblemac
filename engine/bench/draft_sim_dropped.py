"""Draft policies tried and dropped (engine/DRAFT.md §2d): kept so that the tables there can
be reproduced with engine/bench/draft_sim.py --policies new3,dyn3,rate3,...

  new<k>[g<n>][d<decay>]       NewAdaptive: per-width verify costs, first / later draft rates
  dyn<k>[g<n>][d<decay>][last][nf][r<recal>]   Dyn: calibrated confidence, ratio rule
  rate<k>...[R<decay>]         Rate: the same with the average-reward rule
"""
from __future__ import annotations

import math
import re

from draft_sim import Policy

class NewAdaptive(Policy):
    """Candidate policy (to be ported to spec_adapt.c): verify cost per pass width, two-rate
    acceptance (first draft, continuation) per draft source with longer memory, n-gram vs MTP
    per round, optional confidence-gated chain."""

    def __init__(self, kmax: int = 3, ngram: int = 0, decay: float = 0.95, hyst: float = 0.98,
                 conf_gate: float = 0.0, recal: int = 0, ng_min: int = 2,
                 calib_first: bool = True, name: str = "") -> None:
        self.kmax, self.ngram, self.decay, self.hyst = kmax, ngram, decay, hyst
        self.conf_gate, self.recal, self.ng_min, self.calib_first = conf_gate, recal, ng_min, \
            calib_first
        self.name = name or f"new-a{kmax}" + (f"g{ngram}" if ngram else "") + \
            (f"-c{conf_gate}" if conf_gate else "")
        self.reset()

    def reset(self) -> None:
        self.t_step = 0.0
        self.tv: dict[int, float] = {}
        self.t_chain = self.t_pend = self.t_commit = 0.0
        # decayed (hits, trials): MTP first / continuation, n-gram first / continuation
        self.acc = {"m1": [1.3, 2.0], "mc": [1.5, 2.0], "g1": [1.0, 2.0], "gc": [1.6, 2.0]}
        self.k_cur = -1   # last MTP k chosen (hysteresis)
        self.plain_mode = False
        self.plain_run, self.probe_after = 0, 8
        self.rounds_since_cal = 0
        self.calib = False

    # -- model
    def _p(self, key: str) -> float:
        h, t = self.acc[key]
        return h / t

    def _tv(self, n: int) -> float:
        if n in self.tv:
            return self.tv[n]
        known = sorted(self.tv)
        if not known:
            return self.t_step * (1 + 0.1 * (n - 1)) if self.t_step else 100.0 * (1 + 0.1 * (n - 1))
        lo = [m for m in known if m < n]
        hi = [m for m in known if m > n]
        if lo and hi:   # interpolate
            a, b = lo[-1], hi[0]
            return self.tv[a] + (self.tv[b] - self.tv[a]) * (n - a) / (b - a)
        if len(known) >= 2:   # extrapolate linearly from the two nearest
            a, b = (known[-2], known[-1]) if not hi else (known[0], known[1])
            slope = max(0.0, (self.tv[b] - self.tv[a]) / (b - a))
            ref = b if not hi else a
            return self.tv[ref] + slope * (n - ref)
        m = known[0]
        base = self.t_step if self.t_step else self.tv[m] / (1 + 0.1 * (m - 1))
        return self.tv[m] + 0.1 * base * (n - m)

    def _yield(self, p1: float, pc: float, k: int) -> float:
        y, run = 1.0, p1
        for _ in range(k):
            y += run
            run *= pc
        return y

    def _mtp_cost(self, k: int) -> float:
        return self._tv(k + 1) + (k - 1) * self.t_chain + self.t_pend + self.t_commit

    def _ng_cost(self, L: int) -> float:
        return self._tv(L + 1) + self.t_commit

    def _best_mtp(self) -> tuple[int, float]:
        p1, pc = self._p("m1"), self._p("mc")
        best, best_t = 0, self.t_step if self.t_step else math.inf
        for k in range(1, self.kmax + 1):
            t = self._mtp_cost(k) / self._yield(p1, pc, k)
            if t < best_t:
                best, best_t = k, t
        if self.k_cur > 0 and best != self.k_cur:
            t_cur = self._mtp_cost(self.k_cur) / self._yield(p1, pc, self.k_cur)
            if best_t > self.hyst * t_cur:
                best, best_t = self.k_cur, t_cur
        return best, best_t

    def decide(self, tr, P, left, ng, ng_n, d1_known):
        if self.t_step <= 0 and self.calib_first:
            self.calib = True
            return "plain", 0
        if self.plain_mode:
            if self.plain_run < self.probe_after:
                return "plain", 0
        elif self.recal and self.rounds_since_cal >= self.recal:
            k, t = self._best_mtp()
            if k == 0 or t > self.t_step / 1.3:
                self.calib = True
                return "plain", 0
        k, t_mtp = self._best_mtp()
        if self.plain_mode:
            k = max(k, 1)   # probe
        choice = ("mtp", min(k, left)) if k else ("plain", 0)
        best_t = t_mtp if k else (self.t_step or math.inf)
        if self.ngram and ng and ng_n >= self.ng_min:
            g1, gc = self._p("g1"), self._p("gc")
            for L in range(1, min(len(ng), self.ngram, left) + 1):
                t = self._ng_cost(L) / self._yield(g1, gc, L)
                if t < best_t:
                    best_t, choice = t, ("ngram", L)
        if choice[0] == "mtp" and choice[1] > 0 or choice[0] == "ngram":
            return choice
        return "plain", 0

    def chain_more(self, s, conf):
        return not self.conf_gate or conf[s - 1] >= self.conf_gate

    def _obs_acc(self, k1: str, kc: str, k: int, a: int) -> None:
        d = self.decay
        for key in (k1, kc):
            self.acc[key][0] *= d
            self.acc[key][1] *= d
        self.acc[k1][0] += 1 if a >= 1 else 0
        self.acc[k1][1] += 1
        if a >= 1 and k >= 2:
            self.acc[kc][0] += a - 1
            self.acc[kc][1] += (a - 1) + (1 if a < k else 0)

    @staticmethod
    def _ema(old: float, x: float, w: float = 0.25) -> float:
        return x if old <= 0 else old + w * (x - old)

    def observe(self, kind, k, a, t):
        if kind == "plain":
            self.t_step = self._ema(self.t_step, t["plain"])
            if self.calib:
                self.calib = False
                self.rounds_since_cal = 0
                return
            self.plain_run += 1
            return
        n = k + 1
        self.tv[n] = self._ema(self.tv.get(n, 0.0), t["verify"])
        self.t_commit = self._ema(self.t_commit, t["commit"])
        self.rounds_since_cal += 1
        if kind == "mtp":
            if k > 1:
                self.t_chain = self._ema(self.t_chain, t["chain"] / (k - 1))
            self.t_pend = self._ema(self.t_pend, t["pend"])
            self._obs_acc("m1", "mc", t.get("k_req", k), a)
        else:
            self._obs_acc("g1", "gc", k, a)
        # plain mode: enter when the best option is slower than a plain step
        kb, tb = self._best_mtp()
        if self.plain_mode:
            self.plain_mode = kb == 0
            self.probe_after = 8 if kb else min(self.probe_after * 2, 64)
            self.plain_run = 0
        elif kb == 0:
            self.plain_mode, self.plain_run, self.probe_after = True, 0, 8
        if kind == "mtp":
            self.k_cur = kb


class Dyn(Policy):
    """Mirror of spec_adapt.c (this work): per round, the draft source and length with the
    best expected tokens per second.
      * MTP: each draft's acceptance from its top-1 probability through an online calibration
        (bins CONF_EDGES, priors CONF_PRIOR); the chain continues while the best reachable
        length (lookahead with the mean continuation rate) beats stopping, and a new draft
        is dropped if it lowers the expected rate. Pass costs per verify width are measured.
      * n-gram: first-draft acceptance per (extended match length bucket, agreement of the
        first n-gram token with the MTP's free first draft d1), continuation rate per bucket;
        short matches flush the pending MTP rows first so that d1 is known.
      * plain mode when even the best option is slower than a plain step (probed again)."""

    info = True
    chain_hooks = True
    CONF_EDGES = (0.2, 0.4, 0.6, 0.8, 0.9, 0.97)
    CONF_PRIOR = (0.2, 0.35, 0.55, 0.72, 0.86, 0.93, 0.98)
    NG_PRIOR1 = ((0.63, 0.95, 0.10), (0.69, 0.92, 0.32), (0.92, 0.98, 0.3), (0.98, 1.0, 0.67))
    NG_PRIORC = (0.80, 0.85, 0.96, 0.97)
    W = 4.0   # prior weight in trials

    def __init__(self, kmax: int = 3, ngram: int = 0, decay: float = 0.98, recal: int = 32,
                 la: str = "avg", flush_short: bool = True, name: str = "") -> None:
        self.kmax, self.ngram, self.decay, self.recal = kmax, ngram, decay, recal
        self.la, self.flush_short = la, flush_short
        self.name = name or f"dyn{kmax}" + (f"g{ngram}" if ngram else "")
        self.reset()

    def reset(self) -> None:
        self.t_step = 0.0
        self.tv: dict[int, float] = {}
        self.t_chain = self.t_pend = self.t_commit = 0.0
        self.cb = [[p * self.W, self.W] for p in self.CONF_PRIOR]
        self.mc = [0.75 * self.W, self.W]      # MTP continuation (any depth >= 2)
        self.m1 = [0.6 * self.W, self.W]       # MTP first draft (mean, for plain-mode test)
        self.g1 = [[[p * self.W, self.W] for p in row] for row in self.NG_PRIOR1]
        self.gc = [[p * self.W, self.W] for p in self.NG_PRIORC]
        self.calib = False
        self.plain_mode = False
        self.plain_run, self.probe_after, self.since_cal = 0, 8, 0
        self.q: list[float] = []   # calibrated acceptance of this round's MTP drafts
        self.bins: list[int] = []
        self.ng_cell = (0, 0)

    # -- cost model
    def _tv(self, n: int) -> float:
        if n in self.tv:
            return self.tv[n]
        known = sorted(self.tv)
        step = self.t_step or 100.0
        if not known:
            return step * (1 + 0.1 * (n - 1))
        lo = [m for m in known if m < n]
        hi = [m for m in known if m > n]
        if lo and hi:
            a, b = lo[-1], hi[0]
            return self.tv[a] + (self.tv[b] - self.tv[a]) * (n - a) / (b - a)
        ref = lo[-1] if lo else hi[0]
        return max(self.tv[ref] + 0.1 * step * (n - ref), 0.5 * self.tv[ref])

    def _cost(self, k: int, mtp: bool) -> float:
        if not mtp:
            return self._tv(k + 1) + self.t_commit
        return self._tv(k + 1) + (k - 1) * self.t_chain + self.t_pend + self.t_commit

    @staticmethod
    def _r(h: list[float]) -> float:
        return h[0] / h[1]

    def _bin(self, c: float) -> int:
        return next((i for i, e in enumerate(self.CONF_EDGES) if c < e), len(self.CONF_EDGES))

    def _mtp_rate(self, q1: float, k: int) -> float:
        """expected seconds per token of an MTP round of k drafts, first acceptance q1"""
        pc = self._r(self.mc)
        y, run = 1.0, q1
        for _ in range(k):
            y += run
            run *= pc
        return self._cost(k, True) / y

    def _best_mtp_rate(self, q1: float) -> float:
        return min(self._mtp_rate(q1, k) for k in range(1, self.kmax + 1)) if self.kmax else \
            math.inf

    @staticmethod
    def _bucket(m: int) -> int:
        return 0 if m < 4 else 1 if m < 8 else 2 if m < 16 else 3

    def _ng_rate(self, bkt: int, agree: int, L: int) -> float:
        p1, pc = self._r(self.g1[bkt][agree]), self._r(self.gc[bkt])
        y, run = 1.0, p1
        for _ in range(L):
            y += run
            run *= pc
        return self._cost(L, False) / y

    # -- decisions
    def decide(self, tr, P, left, ng, ng_n, d1_known, ng_m=0, d1=-1, c1=0.0):
        if self.t_step <= 0:
            self.calib = True
            return "plain", 0
        if self.plain_mode and self.plain_run < self.probe_after:
            return "plain", 0
        best = ("plain", 0)
        best_t = self.t_step
        q1 = self.cal(c1) if d1_known else self._r(self.m1)
        if self.kmax:
            t = self._best_mtp_rate(q1)
            if t < best_t or self.plain_mode:   # a probe speculates
                best, best_t = ("mtp", min(self.kmax, left)), t
        if self.recal and self.since_cal >= self.recal and best[0] == "mtp" and \
                best_t > self.t_step / 1.3:
            self.calib = True
            return "plain", 0
        if ng and self.ngram:
            bkt = self._bucket(ng_m)
            agree = 0 if not d1_known else (1 if ng[0] == d1 else 2)
            if agree == 0 and bkt < 2 and self.flush_short and self.kmax:
                return "flush", 0   # learn d1 first (simulate() flushes and asks again)
            for L in range(1, min(len(ng), self.ngram, left) + 1):
                t = self._ng_rate(bkt, agree, L)
                if t < best_t:
                    best, best_t = ("ngram", L), t
            self.ng_cell = (bkt, agree)
        return best

    def cal(self, c: float) -> float:
        return self._r(self.cb[self._bin(c)])

    def start_chain(self, c1: float) -> None:
        self.q = [self.cal(c1)]
        self.bins = [self._bin(c1)]

    def chain_more(self, s: int, conf: list[float]) -> bool:
        """after s drafts (q known): compute draft s+1?"""
        P_s = math.prod(self.q[:s])
        Y = 1.0 + sum(math.prod(self.q[:j + 1]) for j in range(s))
        C = self._cost(s, True)
        here = Y / C
        pc = self._r(self.mc) if self.la == "avg" else self.q[s - 1]
        for L in range(s + 1, self.kmax + 1):
            y, run = Y, P_s
            for _ in range(L - s):
                run *= pc
                y += run
            if y / self._cost(L, True) > here:
                return True
        return False

    def keep_last(self, s: int, conf: list[float]) -> bool:
        """draft s was just computed (conf[s-1]): keep it, or verify only s-1 drafts?"""
        self.q.append(self.cal(conf[s - 1]))
        self.bins.append(self._bin(conf[s - 1]))
        Y0 = 1.0 + sum(math.prod(self.q[:j + 1]) for j in range(s - 1))
        Y1 = Y0 + math.prod(self.q[:s])
        sunk = self.t_chain   # the step is paid either way
        c0 = self._cost(s - 1, True) + sunk
        c1 = self._cost(s, True)
        return Y1 / c1 >= Y0 / c0

    def _upd(self, h: list[float], hit: float, trial: float) -> None:
        h[0] = self.decay * h[0] + hit
        h[1] = self.decay * h[1] + trial

    def observe(self, kind, k, a, t):
        if kind == "plain":
            self.t_step = t["plain"] if self.t_step <= 0 else self.t_step + 0.25 * (t["plain"] - self.t_step)
            if self.calib:
                self.calib = False
                self.since_cal = 0
                return
            self.plain_run += 1
            return
        ema = lambda o, x: x if o <= 0 else o + 0.25 * (x - o)  # noqa: E731
        self.tv[k + 1] = ema(self.tv.get(k + 1, 0.0), t["verify"])
        self.t_commit = ema(self.t_commit, t["commit"])
        self.since_cal += 1
        if kind == "mtp":
            if k > 1:
                self.t_chain = ema(self.t_chain, t["chain"] / max(1, t.get("steps", k - 1)))
            self.t_pend = ema(self.t_pend, t["pend"])
            for i in range(min(k, a + 1)):   # drafts 1..min(k, a+1) were tried
                hit = 1.0 if i < a else 0.0
                self._upd(self.cb[self.bins[i]], hit, 1.0)
                if i == 0:
                    self._upd(self.m1, hit, 1.0)
                else:
                    self._upd(self.mc, hit, 1.0)
        else:
            bkt, agree = self.ng_cell
            self._upd(self.g1[bkt][agree], 1.0 if a >= 1 else 0.0, 1.0)
            if a >= 1 and k >= 2:
                self._upd(self.gc[bkt], a - 1, (a - 1) + (1 if a < k else 0))
        self._mode(kind)

    def _mode(self, kind: str) -> None:
        best_t = self._best_mtp_rate(self._r(self.m1))
        if self.plain_mode:
            self.plain_mode = best_t >= self.t_step
            self.probe_after = 8 if not self.plain_mode else min(self.probe_after * 2, 64)
            self.plain_run = 0
        elif kind == "mtp" and best_t >= self.t_step:
            self.plain_mode, self.plain_run, self.probe_after = True, 0, 8


class Rate(Dyn):
    """Dyn with the average-reward rule: a round's choice maximizes E[tokens] - R * E[time],
    R = the realized long-run rate (decayed tokens / decayed seconds), instead of the round's
    own tokens / time; a draft is worth adding iff its probability of being accepted exceeds
    R times its extra cost. (Maximizing each round's ratio over-drafts at low confidence and
    under-drafts at high confidence.)"""

    def __init__(self, *a, rdecay: float = 0.95, **kw) -> None:
        self.rdecay = rdecay
        super().__init__(*a, **kw)
        if not kw.get("name"):
            self.name = self.name.replace("dyn", "rate")

    def reset(self) -> None:
        super().reset()
        self.r_tok = self.r_sec = 0.0

    def _R(self) -> float:
        if self.r_sec > 0:
            return self.r_tok / self.r_sec
        return 1.4 / (self.t_step or 100.0)

    def _val(self, y: float, cost: float) -> float:
        return y - self._R() * cost

    def _mtp_val(self, q1: float, k: int) -> float:
        pc = self._r(self.mc)
        y, run = 1.0, q1
        for _ in range(k):
            y += run
            run *= pc
        return self._val(y, self._cost(k, True))

    def _ng_val(self, bkt: int, agree: int, L: int) -> float:
        p1, pc = self._r(self.g1[bkt][agree]), self._r(self.gc[bkt])
        y, run = 1.0, p1
        for _ in range(L):
            y += run
            run *= pc
        return self._val(y, self._cost(L, False))

    def decide(self, tr, P, left, ng, ng_n, d1_known, ng_m=0, d1=-1, c1=0.0):
        if self.t_step <= 0:
            self.calib = True
            return "plain", 0
        if self.plain_mode and self.plain_run < self.probe_after:
            return "plain", 0
        q1 = self.cal(c1) if d1_known else self._r(self.m1)
        best, best_v = ("plain", 0), self._val(1.0, self.t_step)
        if self.kmax:
            v = max(self._mtp_val(q1, k) for k in range(1, self.kmax + 1))
            if v > best_v or self.plain_mode:
                best, best_v = ("mtp", min(self.kmax, left)), v
        if ng and self.ngram:
            bkt = self._bucket(ng_m)
            agree = 0 if not d1_known else (1 if ng[0] == d1 else 2)
            if agree == 0 and bkt < 2 and self.flush_short and self.kmax:
                return "flush", 0
            for L in range(1, min(len(ng), self.ngram, left) + 1):
                v = self._ng_val(bkt, agree, L)
                if v > best_v:
                    best, best_v = ("ngram", L), v
            self.ng_cell = (bkt, agree)
        return best

    def chain_more(self, s, conf):
        P_s = math.prod(self.q[:s])
        Y = 1.0 + sum(math.prod(self.q[:j + 1]) for j in range(s))
        here = self._val(Y, self._cost(s, True))
        pc = self._r(self.mc) if self.la == "avg" else self.q[s - 1]
        for L in range(s + 1, self.kmax + 1):
            y, run = Y, P_s
            for _ in range(L - s):
                run *= pc
                y += run
            if self._val(y, self._cost(L, True)) > here:
                return True
        return False

    def keep_last(self, s, conf):
        self.q.append(self.cal(conf[s - 1]))
        self.bins.append(self._bin(conf[s - 1]))
        gain = math.prod(self.q[:s])
        return gain >= self._R() * (self._tv(s + 1) - self._tv(s))

    def observe(self, kind, k, a, t):
        tok = 1 if kind == "plain" else a + 1
        sec = sum(v for key, v in t.items() if key in ("plain", "verify", "commit", "chain", "pend"))
        if not (kind == "plain" and self.calib):
            self.r_tok = self.rdecay * self.r_tok + tok
            self.r_sec = self.rdecay * self.r_sec + sec
        super().observe(kind, k, a, t)

    def _mode(self, kind: str) -> None:
        """plain mode by value: a plain step beats the best speculative round"""
        q1 = self._r(self.m1)
        v_spec = max(self._mtp_val(q1, kk) for kk in range(1, self.kmax + 1)) \
            if self.kmax else -math.inf
        v_plain = self._val(1.0, self.t_step)
        if self.plain_mode:
            self.plain_mode = v_spec < v_plain
            self.probe_after = 8 if not self.plain_mode else min(self.probe_after * 2, 64)
            self.plain_run = 0
        elif kind == "mtp" and v_spec < v_plain:
            self.plain_mode, self.plain_run, self.probe_after = True, 0, 8


def by_name(nm: str) -> Policy:
    if m := re.fullmatch(r"new(\d)(?:g(\d))?(?:c([\d.]+))?(?:d([\d.]+))?(?:n(\d))?", nm):
        return NewAdaptive(int(m[1]), int(m[2] or 0), float(m[4] or 0.95),
                           conf_gate=float(m[3] or 0), ng_min=int(m[5] or 2), name=nm)
    if m := re.fullmatch(r"(dyn|rate)(\d)(?:g(\d))?(?:d([\d.]+))?(last)?(nf)?(?:r(\d+))?"
                         r"(?:R([\d.]+))?", nm):
        kw = dict(recal=int(m[7]) if m[7] else 32, la="last" if m[5] else "avg",
                  flush_short=not m[6], name=nm)
        if m[1] == "rate":
            return Rate(int(m[2]), int(m[3] or 0), float(m[4] or 0.98),
                        rdecay=float(m[8] or 0.95), **kw)
        return Dyn(int(m[2]), int(m[3] or 0), float(m[4] or 0.98), **kw)
    raise SystemExit(f"unknown policy {nm}")
