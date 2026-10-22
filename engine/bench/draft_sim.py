"""Offline simulator of speculative draft policies (engine/DRAFT.md).

Input: `draft_trace` output (engine/bench/draft_trace.c) for each prompt, i.e. at every
position P the MTP chain drafts d1..dK and their top-1 probabilities, plus the greedy tokens.
Because the drafts at P do not depend on how earlier rounds were cut, a round at P with k
MTP drafts accepts exactly min(k, A(P)); n-gram (prompt-lookup) drafts depend only on the
token history, so they are computed here from the prompt + greedy tokens. Costs per pass
come from Pi measurements (COSTS); measured times fed to adaptive policies get noise.

    python engine/bench/draft_sim.py TRACE.json [TRACE.json ...] [--policies a,b] [--seeds 20]

TRACE.json: a bench/results file of a `draft_trace` run (its stdout holds the T / G lines
of several prompts, separated by their RESULT lines), or a plain text file of one trace.
Policies (--policies, comma-separated):
  k<k>[g<n>n<min>[G<gate>]]   fixed k; n-gram drafts of up to n (early gating variants)
  old<k>                      spec_adapt.c at main 792676c
  s<k>[g<n>][t<tau>][T<trust>][L<short>][m<margin>][S<ask>][M<mid>]
                              the final policy (spec_adapt.c + spec.c ngram_pick); f<k>... =
                              the same n-gram rule with fixed k (spec.c without adaptive)
  new.., dyn.., rate..        dropped candidates (draft_sim_dropped.py)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_K = 8


@dataclass
class Trace:
    name: str
    prompt: list[int]
    gen: list[int]                 # greedy tokens; gen[0] = first generated (position len(prompt))
    drafts: dict[int, list[int]]   # P -> d1..dK
    conf: dict[int, list[float]]   # P -> top-1 probabilities
    plain_ms: float = 0.0

    def tok(self, pos: int) -> int:
        n = len(self.prompt)
        return self.prompt[pos] if pos < n else self.gen[pos - n]

    def accept(self, P: int, draft: list[int]) -> int:
        a = 0
        while a < len(draft) and draft[a] == self.tok(P + 1 + a):
            a += 1
        return a


def load_traces(path: Path) -> list[Trace]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        text = json.loads(text)["stdout"]
    out: list[Trace] = []
    drafts: dict[int, list[int]] = {}
    conf: dict[int, list[float]] = {}
    gen: list[int] = []
    for line in text.splitlines():
        if line.startswith("T "):
            f = line.split()
            P, K = int(f[1]), (len(f) - 3) // 2
            drafts[P] = [int(x) for x in f[3:3 + K]]
            conf[P] = [float(x) for x in f[3 + K:3 + 2 * K]]
        elif line.startswith("G "):
            gen = [int(x) for x in line[2:].split(",")]
        elif line.startswith("RESULT "):
            r = json.loads(line[7:])
            ids_file = r["ids_file"]
            name = Path(ids_file).stem
            src = ROOT / "bench" / "common" / f"{name}.ids"
            prompt = [int(x) for x in re.findall(r"\d+", src.read_text())]
            assert len(prompt) == r["n_prompt"], (name, len(prompt), r["n_prompt"])
            out.append(Trace(name, prompt, gen, drafts, conf, r.get("plain_ms", 0.0)))
            drafts, conf, gen = {}, {}, []
    return out


# ----------------------------------------------------------------------------- n-gram lookup
@dataclass
class NgramIndex:
    """Mirror of engine/src/ngram.c: per n in [nmin, nmax] the most recent end position of
    every n-gram of the history (exact here, hashed with verification in C)."""
    nmin: int = 2
    nmax: int = 4
    last: dict[tuple[int, ...], int] = field(default_factory=dict)
    upto: int = 0   # positions < upto are indexed

    def draft(self, tr: Trace, P: int, L: int) -> tuple[list[int], int]:
        """Draft up to L tokens after position P (tokens 0..P known). Returns (draft, n)."""
        for q in range(self.upto, P):     # index n-grams ending at q < P
            for n in range(self.nmin, self.nmax + 1):
                if q - n + 1 >= 0:
                    self.last[tuple(tr.tok(q - n + 1 + i) for i in range(n))] = q
        self.upto = max(self.upto, P)
        for n in range(self.nmax, self.nmin - 1, -1):
            if P - n + 1 < 0:
                continue
            q = self.last.get(tuple(tr.tok(P - n + 1 + i) for i in range(n)))
            if q is not None:
                m = min(L, P - q)
                return [tr.tok(q + 1 + i) for i in range(m)], n
        return [], 0


# ----------------------------------------------------------------------------- costs (ms)
@dataclass
class Costs:
    step: float          # plain decode step
    tv: dict[int, float]  # verify pass of n tokens (cur + n-1 drafts)
    chain: float = 8.4   # one MTP chain step (1 row, draft vocab 32768)
    pend: float = 8.4    # MTP pass over the committed rows of a round
    commit: float = 0.9  # + commit_row per verified row's argmax (a + 1 of them)
    flush: float = 13.0  # final pf_lin_flush of a speculative call
    commit_row: float = 0.25

    def verify(self, n: int) -> float:
        return self.tv[n]


def costs_from(step: float, tv: dict[int, float], **kw: float) -> Costs:
    return Costs(step, dict(tv), **kw)


# ----------------------------------------------------------------------------- policies
class Policy:
    """decide(P, ...) -> ("plain", 0) | ("mtp", k) | ("ngram", L) | ("flush", 0);
    observe(...) after it. info: decide() also takes (ng_m, d1, c1); chain_hooks: the
    policy has start_chain / keep_last (draft_sim_dropped.Dyn)."""
    name = "?"
    info = False
    chain_hooks = False

    def reset(self) -> None: ...

    def decide(self, tr: Trace, P: int, left: int, ng: list[int], ng_n: int,
               d1_known: bool) -> tuple[str, int]:
        raise NotImplementedError

    def chain_more(self, s: int, conf: list[float]) -> bool:
        """MTP round: after s drafts (conf of d1..ds known), draft one more?"""
        return True

    def observe(self, kind: str, k: int, a: int, t: dict[str, float]) -> None: ...


class Fixed(Policy):
    """Fixed k MTP drafts; with ngram, the n-gram draft when there is a match (ng_min: the
    shortest n-gram accepted). gate = G: only when the extended match is >= G tokens or the
    first n-gram token equals the MTP's d1 (pending MTP rows are flushed to learn d1) —
    an early version of the rule; Simple has the one spec.c uses."""
    info = True

    def __init__(self, k: int, ngram: int = 0, ng_min: int = 2, gate: int = 0) -> None:
        self.k, self.ngram, self.ng_min, self.gate = k, ngram, ng_min, gate
        self.name = f"k{k}" + (f"g{ngram}n{ng_min}" if ngram else "") + \
            (f"G{gate}" if gate else "")

    def decide(self, tr, P, left, ng, ng_n, d1_known, ng_m=0, d1=-1, c1=0.0):
        if self.ngram and ng and ng_n >= self.ng_min:
            if not self.gate or ng_m >= self.gate or not self.k:
                return "ngram", min(len(ng), self.ngram, left)
            if not d1_known:
                return "flush", 0
            if ng[0] == d1:
                return "ngram", min(len(ng), self.ngram, left)
        return ("mtp", min(self.k, left)) if self.k else ("plain", 0)


class OldAdaptive(Policy):
    """Port of spec_adapt.c at main 792676c."""
    name = "old-adaptive"

    def __init__(self, kmax: int = 3) -> None:
        self.kmax = kmax
        self.reset()

    def reset(self) -> None:
        self.t_step = self.t_draft = self.t_commit = 0.0
        self.cv = -1.0
        self.hits, self.trials = 2.0, 3.0
        self.rounds = self.since_cal = self.plain_run = 0
        self.k_cur, self.calib, self.probe_after = self.kmax, 0, 8

    @staticmethod
    def _ema(old: float, x: float) -> float:
        return x if old <= 0 else old + 0.25 * (x - old)

    def _per_token(self, k: int) -> float:
        if k == 0:
            return self.t_step
        cv = self.cv if self.cv >= 0 else 0.1
        p = self.hits / self.trials
        y = sum(p ** i for i in range(k + 1))
        return (self.t_step * (1 + cv * k) + k * self.t_draft + self.t_commit) / y

    def _choose(self) -> int:
        best, best_c = 0, self._per_token(0)
        for k in range(1, self.kmax + 1):
            c = self._per_token(k)
            if c < best_c:
                best, best_c = k, c
        if 0 < self.k_cur <= self.kmax and best != self.k_cur and \
                best_c > 0.97 * self._per_token(self.k_cur):
            best = self.k_cur
        return best

    def decide(self, tr, P, left, ng, ng_n, d1_known):
        if self.t_step <= 0:
            self.calib = 1
            k = 0
        elif self.rounds == 0:
            k = self.kmax
        elif self.k_cur == 0:
            k = self.kmax if self.plain_run >= self.probe_after else 0
        elif self.since_cal >= 32:
            self.calib = 1
            k = 0
        else:
            k = self.k_cur
        k = min(k, left)
        return ("mtp", k) if k else ("plain", 0)

    def observe(self, kind, k, a, t):
        if kind == "plain":
            self.t_step = self._ema(self.t_step, t["plain"])
            self.since_cal = 0
            if self.calib:
                self.calib = 0
                return
            self.plain_run += 1
            return
        if self.t_step > 0:
            cv = max(0.0, (t["verify"] / self.t_step - 1.0) / k)
            self.cv = cv if self.cv < 0 else self.cv + 0.25 * (cv - self.cv)
        self.t_draft = self._ema(self.t_draft, t["mtp"] / k)
        self.t_commit = self._ema(self.t_commit, t["commit"])
        self.hits = 0.9 * self.hits + a
        self.trials = 0.9 * self.trials + a + (1 if a < k else 0)
        self.rounds += 1
        self.since_cal += 1
        best = self._choose()
        if self.k_cur == 0:
            self.probe_after = 8 if best > 0 else min(self.probe_after * 2, 64)
            self.plain_run = 0
        elif best == 0:
            self.plain_run = 0
            self.probe_after = 8
        self.k_cur = best


class Simple(Policy):
    """Mirror of spec_adapt.c (final). MTP chain: draft d(s+1) while the MTP's top-1
    probability of d(s) is >= tau (up to kmax); d1 always. N-gram drafts (shared with the
    fixed-k mode, spec.c): taken when the match extends over >= trust tokens, or when the
    first n-gram token equals the MTP's d1 (pending MTP rows are run first to learn d1);
    length: all available (<= ngram) when the match is >= trust tokens, else <= short_len.
    Plain mode: after MIN_ROUNDS speculative rounds (2 if a 2-token verify costs >= 1.5 plain
    steps), when the realized rate of speculative rounds (decayed tokens / seconds) is below
    `margin` x the plain-step rate. The plain step's time: measured plain steps if there
    were any, else extrapolated from verify passes of widths 2 and 3 (2 tv(2) - tv(3); no
    calibration step). Without either, only the sure case: a rate below 1 / (cheapest verify
    pass) is slower than plain decoding on any machine (a pass of n >= 2 tokens costs at
    least one plain step). Probes of PROBE_ROUNDS rounds, judged on their own, after 8, 16,
    .. 64 plain steps."""

    info = True
    MIN_ROUNDS = 8   # speculative rounds before a switch to plain mode
    PROBE_ROUNDS = 2  # speculative rounds of a probe out of plain mode, judged on their own

    def __init__(self, kmax: int = 3, ngram: int = 0, tau: float = 0.6, trust: int = 16,
                 short_len: int = 3, decay: float = 0.95, margin: float = 0.95,
                 adaptive: bool = True, min_short: int = 4, mid: int = 99,
                 name: str = "") -> None:
        self.kmax, self.ngram, self.tau, self.trust, self.short_len = kmax, ngram, tau, trust, \
            short_len
        self.min_short, self.mid = min_short, mid
        self.decay, self.margin, self.adaptive = decay, margin, adaptive
        self.name = name or f"s{kmax}" + (f"g{ngram}" if ngram else "")
        self.reset()

    def reset(self) -> None:
        self.tv: dict[int, float] = {}
        self.r_tok = self.r_sec = 0.0
        self.plain_mode, self.plain_run, self.probe_after = False, 0, 8
        self.t_plain = 0.0
        self.n_spec = 0   # speculative rounds since the last mode change (or probe start)
        self.probing = False

    def _t_step(self) -> float:
        if self.t_plain > 0:
            return self.t_plain
        if 2 in self.tv and 3 in self.tv:
            return max(0.5 * self.tv[2], 2 * self.tv[2] - self.tv[3])
        return 0.0

    def ng_choice(self, ng: list[int], ng_m: int, d1_known: bool, d1: int, left: int
                  ) -> tuple[str, int] | None:
        if not (ng and self.ngram):
            return None
        L = min(len(ng), self.ngram, left)
        if ng_m >= self.trust:
            return "ngram", L
        if not self.kmax:   # no MTP head to ask
            return "ngram", min(L, self.short_len)
        if ng_m < self.min_short:
            return None
        if not d1_known:
            return "flush", 0
        if ng[0] == d1:
            return "ngram", L if ng_m >= self.mid else min(L, self.short_len)
        return None

    def decide(self, tr, P, left, ng, ng_n, d1_known, ng_m=0, d1=-1, c1=0.0):
        if self.adaptive and self.plain_mode:
            if self.plain_run < self.probe_after:
                return "plain", 0
            if not self.probing:   # start a probe: judged on its own rounds
                self.probing, self.n_spec, self.r_tok, self.r_sec = True, 0, 0.0, 0.0
        ch = self.ng_choice(ng, ng_m, d1_known, d1, left)
        if ch:
            return ch
        return ("mtp", min(self.kmax, left)) if self.kmax and left else ("plain", 0)

    def chain_more(self, s, conf):
        return not self.adaptive or conf[s - 1] >= self.tau

    def observe(self, kind, k, a, t):
        if kind == "plain":
            self.t_plain = t["plain"] if self.t_plain <= 0 else \
                self.t_plain + 0.25 * (t["plain"] - self.t_plain)
            if self.plain_mode:
                self.plain_run += 1
            return
        self.tv[k + 1] = t["verify"] if k + 1 not in self.tv else \
            self.tv[k + 1] + 0.25 * (t["verify"] - self.tv[k + 1])
        sec = t["verify"] + t["commit"] + t.get("chain", 0.0) + t.get("pend", 0.0)
        self.r_tok = self.decay * self.r_tok + a + 1
        self.r_sec = self.decay * self.r_sec + sec
        if not self.adaptive:
            return
        self.n_spec += 1
        ts = self._t_step()
        rate = self.r_tok / self.r_sec
        slow = rate < self.margin / min(self.tv.values()) or (ts > 0 and rate < self.margin / ts)
        if self.plain_mode:   # a probe round
            if self.n_spec < self.PROBE_ROUNDS:
                return
            self.probing = False
            self.plain_mode = slow
            self.probe_after = min(self.probe_after * 2, 64) if slow else 8
            self.plain_run = 0
            self.n_spec = 0
            if slow:
                self.r_tok = self.r_sec = 0.0
        elif slow and (self.n_spec >= self.MIN_ROUNDS or
                       (ts > 0 and 2 in self.tv and self.tv[2] >= 1.5 * ts and self.n_spec >= 2)):
            # compute-bound machines (a 2-token verify >= 1.5 plain steps) switch early
            self.plain_mode, self.plain_run, self.probe_after = True, 0, 8


# ----------------------------------------------------------------------------- simulation
def ext_match(tr: Trace, P: int, ng: list[int], idx: "NgramIndex", n: int, cap: int = 32) -> int:
    """Extended match length of the n-gram draft at P (tokens equal backwards from the end of
    the earlier occurrence and from P)."""
    q = idx.last.get(tuple(tr.tok(P - n + 1 + i) for i in range(n)))
    if q is None:
        return 0
    m = 0
    while m < cap and q - m >= 0 and tr.tok(P - m) == tr.tok(q - m):
        m += 1
    return m


def simulate(tr: Trace, pol: Policy, c: Costs, n_gen: int, rng: random.Random,
             noise: float = 0.03, ngram_cfg: tuple[int, int] = (2, 4)) -> dict[str, float]:
    """Decode n_gen tokens (the first comes from the prefill); returns tok/s etc."""
    pol.reset()
    idx = NgramIndex(*ngram_cfg)
    P = len(tr.prompt)
    count = 1
    T = 0.0
    rounds = drafted = accepted = plain = ng_rounds = 0
    pending = 0   # plain / n-gram rows whose MTP pass is deferred

    def noisy(x: float) -> float:
        return x * max(0.5, 1 + noise * rng.gauss(0, 1))

    dyn = getattr(pol, "info", False)   # decide() takes the n-gram / d1 arguments
    while count < n_gen:
        left = n_gen - count - 1
        ng, ng_n = idx.draft(tr, P, MAX_K) if getattr(pol, "ngram", 0) else ([], 0)
        if ng and getattr(pol, "ngram", 0):
            ng = ng[:min(pol.ngram, left)] if left > 0 else []
        if dyn:
            ng_m = ext_match(tr, P, ng, idx, ng_n) if ng else 0
            kind, k = pol.decide(tr, P, left, ng, ng_n, pending == 0, ng_m, tr.drafts[P][0],
                                 tr.conf[P][0])
            if kind == "flush":
                T += c.pend
                pending = 0
                kind, k = pol.decide(tr, P, left, ng, ng_n, True, ng_m, tr.drafts[P][0],
                                     tr.conf[P][0])
        else:
            kind, k = pol.decide(tr, P, left, ng, ng_n, pending == 0)
        if kind == "plain" or k == 0:
            dt = noisy(c.step)
            T += dt
            pending += 1
            if pending == 64:
                T += c.pend
                pending = 0
            P += 1
            count += 1
            plain += 1
            pol.observe("plain", 0, 0, {"plain": dt})
            continue
        t: dict[str, float] = {}
        if kind == "mtp":
            flush = c.pend if pending else 0.0
            pending = 0
            d = tr.drafts[P]
            ke, steps = 1, 0
            isdyn = getattr(pol, "chain_hooks", False)   # start_chain / keep_last
            if isdyn:
                pol.start_chain(tr.conf[P][0])
            while ke < k and pol.chain_more(ke, tr.conf[P]):
                steps += 1
                ke += 1
                if isdyn and not pol.keep_last(ke, tr.conf[P]):
                    ke -= 1
                    break
            a = tr.accept(P, d[:ke])
            t["chain"] = noisy(steps * c.chain) if steps else 0.0
            t["steps"] = steps
            t["pend"] = noisy(c.pend)
            t["k_req"] = ke
            T += flush + t["chain"] + t["pend"]
            t["mtp"] = t["chain"] + t["pend"]
        else:
            ke = k
            a = tr.accept(P, ng[:ke])
            pending += a + 1
            if pending >= 64:
                T += c.pend
                pending = 0
            ng_rounds += 1
        t["verify"] = noisy(c.verify(ke + 1))
        t["commit"] = noisy(c.commit + c.commit_row * (a + 1))
        T += t["verify"] + t["commit"]
        e = min(a + 1, n_gen - count)
        P += e
        count += e
        rounds += 1
        drafted += ke
        accepted += a
        pol.observe(kind, ke, a, t)
    if rounds:
        T += c.flush + (c.pend if pending else 0.0)
    return {"tps": (n_gen - 1) / (T / 1e3), "rounds": rounds, "drafted": drafted,
            "accepted": accepted, "plain": plain, "ng_rounds": ng_rounds,
            "tpp": 1 + accepted / rounds if rounds else 1.0}


# Pi Q4K, 4 threads, pinfer_run_fast 23a76242: verify ms per pass width n = drafts + 1,
# prompt64 k sweep (bench/results/ng-base-p64_20261022_111810.json, medians of 3); MTP pass
# 8.4 ms; plain step 126.1 ms. ctx2048: plain 141.7 ms (draft trace), verify = short +
# 15.6 + 6.5 (n - 1) ms (fits k=3 187.3 ms of engine/VERIFY.md), replaced by measurements.
_TV64 = {2: 136.1, 3: 142.9, 4: 152.0, 5: 199.6, 6: 258.0, 7: 266.0, 8: 285.0, 9: 360.0}
PI_COSTS = {
    "short": costs_from(126.1, _TV64),
    "ctx2048": costs_from(141.7, {n: t + 15.6 + 6.5 * (n - 1) for n, t in _TV64.items()}),
}


def policies_by_name(names: list[str]) -> list[Policy]:
    out: list[Policy] = []
    for nm in names:
        if m := re.fullmatch(r"k(\d)(?:g(\d)n(\d))?(?:G(\d+))?", nm):
            out.append(Fixed(int(m[1]), int(m[2] or 0), int(m[3] or 2), int(m[4] or 0)))
        elif m := re.fullmatch(r"old(\d)", nm):
            out.append(OldAdaptive(int(m[1])))
        elif nm.startswith(("new", "dyn", "rate")):
            from draft_sim_dropped import by_name
            out.append(by_name(nm))
        elif m := re.fullmatch(r"(s|f)(\d)(?:g(\d))?(?:t([\d.]+))?(?:T(\d+))?(?:L(\d))?"
                               r"(?:m([\d.]+))?(?:S(\d+))?(?:M(\d+))?", nm):
            out.append(Simple(int(m[2]), int(m[3] or 0), float(m[4] or 0.6), int(m[5] or 16),
                              int(m[6] or 3), margin=float(m[7] or 0.95),
                              adaptive=m[1] == "s", min_short=int(m[8] or 4),
                              mid=int(m[9] or 99), name=nm))
        else:
            raise SystemExit(f"unknown policy {nm}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--policies", default="k1,k2,k3,k4,old3,new3")
    ap.add_argument("--costs", default="short")
    ap.add_argument("--costs-json", type=Path, help="{name: {step, tv: {n: ms}, ...}}")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.03)
    a = ap.parse_args()
    costs = dict(PI_COSTS)
    if a.costs_json:
        for k, v in json.loads(a.costs_json.read_text()).items():
            costs[k] = costs_from(v["step"], {int(n): t for n, t in v["tv"].items()},
                                  **{x: v[x] for x in ("chain", "pend", "commit", "flush") if x in v})
    traces = [t for p in a.traces for t in load_traces(p)]
    pols = policies_by_name(a.policies.split(","))
    print(f"{'prompt':>14} " + " ".join(f"{p.name:>12}" for p in pols))
    for tr in traces:
        cname = "ctx2048" if tr.name.startswith("ctx") and "ctx2048" in costs else a.costs
        c = costs[cname]
        n = min(64 if tr.name == "prompt64" else a.n, len(tr.drafts))   # as chat_spec.sh runs
        row = []
        for pol in pols:
            rs = [simulate(tr, pol, c, n, random.Random(s), a.noise) for s in range(a.seeds)]
            tps = sum(r["tps"] for r in rs) / len(rs)
            tpp = sum(r["tpp"] for r in rs) / len(rs)
            row.append(f"{tps:6.2f} ({tpp:3.2f})")
        print(f"{tr.name:>14} " + " ".join(f"{x:>12}" for x in row))


if __name__ == "__main__":
    main()
