"""Prompt-layout variants of the Cerno decision protocol: token ids, answer slots, token counts.

Pilot "prompt economy" (engine/PILOTS.md): can a decision be made with fewer prompt tokens while
the probabilities stay (near-)identical to cerno-pi's one-row-per-question layout?

Variants (one `Plan` per request: a prefix evaluated once per decision, then rows; every row is
evaluated after restoring the prefix, except chains, whose rows continue from each other):

* `base`      cerno-pi / Cerno's engine_llama rows: ``Context:\\n<state>\\n\\nQuestion: <q>\\nOptions:
              \\n(A) <o>…\\nAnswer: (``, one row per question, the rows' common prefix evaluated once.
* `multi`     decider's state-first row with all questions (``Question k: … Answer k: (``, numbered
              when there is more than one question; no answer letters inserted; one slot per
              question read from one pass).  decider/prompt.py `build`.
* `multi_ans` the same row as a chain: after slot k the 4B's own top letter and ")" are inserted
              before question k + 1.
* `schema`    decider's schema-first layout (`build_schema_first`): the question blocks first
              (cacheable per schema), then ``\\n\\nContext:\\n<state>``, then the answer slots
              ``\\n\\nAnswer 1: (``, ``\\nAnswer 2: (`` …
* `compact_a` base without the ``\\nOptions:`` line.
* `compact_b` compact_a with bare letters: ``\\nA <o>`` instead of ``\\n(A) <o>``; slot ``\\nAnswer: (``.
* `compact_c` compact_b with the slot ``\\nAnswer:`` and the labels read as " A", " B" … tokens.
* `compact_d` base with the options on the Options line: ``\\nOptions: (A) <o> (B) <o>\\nAnswer: (``.
* `shared`    base, but a request with >= 2 yes/no questions lists their options once after the
              state (``\\n\\nOptions for yes/no questions:\\n(A) no\\n(B) yes``) and its yes/no rows
              are ``\\n\\nQuestion: <q>\\nAnswer: (``.

Question rendering, neutralisation and the label table follow cerno_pi/decide.py (decider's
systemone.py / prompt.py, Apache-2.0).  ``python variants.py`` prints the token counts.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]


def _first(*cands: str | Path) -> Path:
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return Path([c for c in cands if c][0])


# WSL and Windows paths (the worktree has no models/ link; env overrides first)
MODELS = _first(os.environ.get("PINFER_MODELS", ""), ROOT / "models",
                "/mnt/e/Personal_tries/Raspberry_Pi/models", "E:/Personal_tries/Raspberry_Pi/models")
CERNO_PI = _first(os.environ.get("CERNO_PI", ""), "/mnt/e/Personal_tries/cerno-pi",
                  "E:/Personal_tries/cerno-pi", ROOT.parent / "cerno-pi")
TOKENIZER = MODELS / "cerno-4b" / "cerno-4b-q4_0.tokenizer" / "tokenizer.json"
REQUESTS = CERNO_PI / "requests" / "correctness.json"
REF = CERNO_PI / "results" / "ref_llama_pc.jsonl"
LETTERS = "ABCDEFGHIJ"
NEUTRAL_NONE = "not listed here"
ANNOTATE_MIN = 8
VARIANTS = ["base", "multi", "multi_ans", "schema", "schema_q", "compact_a", "compact_b",
            "compact_c", "compact_d", "shared"]


# ------------------------------------------------------------------ rendering (cerno_pi.decide)
def _txt(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def annotate_indices(x: Any, min_len: int = ANNOTATE_MIN) -> Any:
    if isinstance(x, list):
        if len(x) >= min_len:
            return [({"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                     else {"_index": i, "value": annotate_indices(v, min_len)})
                    for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state), ensure_ascii=False)


@dataclass(frozen=True)
class Question:
    key: str
    type: str
    text: str
    options: tuple[str, ...]


def render_question(key: str, spec: dict[str, Any]) -> Question:
    t = spec.get("type", "choice")
    ins = _txt(spec.get("instructions", spec.get("question", "")))
    crit = spec.get("criteria", spec.get("options"))
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        opts = [n if crit[n] in (None, "") else f"{n}: {_txt(crit[n])}" for n in crit]
    elif t == "score":
        if isinstance(crit, dict):
            crit = [crit[k] for k in sorted(crit, key=float)]
        opts = [f"{i}: {_txt(c)}" for i, c in enumerate(crit)]
    elif t in ("noul", "bool"):
        c = crit or {}
        f, tr = c.get("false", c.get(False)), c.get("true", c.get(True))
        opts = ["no" if f in (None, "") else f"no: {_txt(f)}",
                "yes" if tr in (None, "") else f"yes: {_txt(tr)}"]
        t = "noul"
    else:
        raise ValueError(t)
    return Question(key, t, ins, tuple(neutralize_options(opts)))


def neutralize_options(options: list[str]) -> list[str]:
    out = []
    for o in options:
        key = o.strip().lower()
        if key.startswith("none of the above") or key in ("none of the above", "none", "n/a",
                                                          "none of these"):
            out.append(NEUTRAL_NONE)
        else:
            out.append(o)
    return out


# ------------------------------------------------------------------ plans
@dataclass
class Slot:
    pos: int                     # index in the row's ids; -1 = last token
    labels: list[int]            # label token ids, in option order
    key: str                     # question id


@dataclass
class Row:
    ids: list[int]
    slots: list[Slot]
    # chain rows: ids are built from the previous slot's top letter (None: `ids` as is)
    ids_fn: Callable[[str], list[int]] | None = None


@dataclass
class Plan:
    request: str
    variant: str
    prefix: list[int]            # evaluated once per decision; restored before every row
    rows: list[Row]
    chain: bool = False          # rows continue from each other (no restore)
    cacheable_prefix: bool = False   # the prefix depends only on the questions (schema-first)
    tokens: int = 0              # tokens evaluated per decision (prefix + rows)
    tokens_cached: int = 0       # the same with the prefix cached across decisions (schema-first)
    nq: int = 0


class Tok:
    def __init__(self, path: Path = TOKENIZER):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(path))
        self.cache: dict[str, list[int]] = {}
        self.labels = self.label_table()      # A..Z then two-letter single tokens (decide.py)

    def __call__(self, s: str) -> list[int]:
        v = self.cache.get(s)
        if v is None:
            v = self.t.encode(s, add_special_tokens=False).ids
            self.cache[s] = v
        return list(v)

    def one(self, s: str) -> int:
        ids = self(s)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is {len(ids)} tokens")
        return ids[0]

    def label_table(self, n: int = 255) -> list[str]:
        import string
        upper = string.ascii_uppercase
        out = []
        for name in list(upper) + [a + b for a in upper for b in upper]:
            if len(self(name)) == 1:
                out.append(name)
            if len(out) == n:
                break
        assert out[:10] == list(LETTERS)
        return out


def common_prefix(rows: list[list[int]]) -> int:
    short = min(len(r) for r in rows) - 1
    n = 0
    while n < short and all(r[n] == rows[0][n] for r in rows):
        n += 1
    return n


def _finish_rows(request: str, variant: str, keys_rows: list[tuple[str, list[int], list[int]]]
                 ) -> Plan:
    """Rows that end with their slot -> a plan with the rows' common prefix split off."""
    ids = [r for _, r, _ in keys_rows]
    lcp = common_prefix(ids) if len(ids) > 1 else 0
    rows = [Row(r[lcp:], [Slot(-1, lab, k)]) for k, r, lab in keys_rows]
    p = Plan(request, variant, ids[0][:lcp], rows, nq=len(rows))
    p.tokens = lcp + sum(len(r.ids) for r in rows)
    p.tokens_cached = p.tokens
    return p


def labels_for(tok: Tok, n: int, space: bool = False) -> list[int]:
    return [tok.one((" " if space else "") + tok.labels[j]) for j in range(n)]


def option_lines(tok: Tok, qs: Question, style: str = "paren") -> str:
    """Option lines; > 10 options use the label table (A..Z, AA..) as decide.py's wide rows,
    whose decoded text is tokenized again (the reference's "text" tokenization)."""
    lab = tok.labels
    if style == "paren":
        return "".join(f"\n({lab[j]}) {o}" for j, o in enumerate(qs.options))
    if style == "bare":
        return "".join(f"\n{lab[j]} {o}" for j, o in enumerate(qs.options))
    if style == "inline":
        return " " + " ".join(f"({lab[j]}) {o}" for j, o in enumerate(qs.options))
    raise ValueError(style)


def plan_base(tok: Tok, name: str, state: str, qs: list[Question], variant: str = "base",
              header: str = "\nOptions:", style: str = "paren", tail: str = "\nAnswer: (",
              space_labels: bool = False) -> Plan:
    rows = []
    for q in qs:
        text = ("Context:\n" + state + f"\n\nQuestion: {q.text}" + header + option_lines(tok, q, style)
                + tail)
        rows.append((q.key, tok(text), labels_for(tok, len(q.options), space_labels)))
    return _finish_rows(name, variant, rows)


def plan_shared(tok: Tok, name: str, state: str, qs: list[Question]) -> Plan:
    nouls = [q for q in qs if q.type == "noul" and q.options == ("no", "yes")]
    if len(nouls) < 2:
        return plan_base(tok, name, state, qs, "shared")
    ctx = "Context:\n" + state + "\n\nOptions for yes/no questions:\n(A) no\n(B) yes"
    rows = []
    for q in qs:
        if q in nouls:
            text = ctx + f"\n\nQuestion: {q.text}\nAnswer: ("
        else:
            text = ctx + f"\n\nQuestion: {q.text}\nOptions:" + option_lines(tok, q) + "\nAnswer: ("
        rows.append((q.key, tok(text), labels_for(tok, len(q.options))))
    return _finish_rows(name, "shared", rows)


def plan_multi(tok: Tok, name: str, state: str, qs: list[Question], chain: bool = False) -> Plan:
    """decider `build` (state-first, all questions in one row; numbered when > 1)."""
    multi = len(qs) > 1
    ctx = "Context:\n" + state
    pieces: list[str] = []
    for k, q in enumerate(qs):
        n = f" {k + 1}" if multi else ""
        pieces.append(f"\n\nQuestion{n}: {q.text}\nOptions:" + option_lines(tok, q) + f"\nAnswer{n}: (")
    labs = [labels_for(tok, len(q.options)) for q in qs]
    if not chain:
        ids = tok(ctx + pieces[0])
        slots = [Slot(len(ids) - 1, labs[0], qs[0].key)]
        for k in range(1, len(qs)):
            ids += tok(pieces[k])
            slots.append(Slot(len(ids) - 1, labs[k], qs[k].key))
        p = Plan(name, "multi", [], [Row(ids, slots)], nq=len(qs))
        p.tokens = p.tokens_cached = len(ids)
        return p
    rows = [Row(tok(ctx + pieces[0]), [Slot(-1, labs[0], qs[0].key)])]
    for k in range(1, len(qs)):
        piece = pieces[k]
        rows.append(Row([], [Slot(-1, labs[k], qs[k].key)],
                        ids_fn=(lambda prev, piece=piece: tok(prev + ")" + piece))))
    p = Plan(name, "multi_ans", [], rows, chain=True, nq=len(qs))
    p.tokens = len(rows[0].ids) + sum(len(r.ids_fn("A")) for r in rows[1:])
    p.tokens_cached = p.tokens
    return p


def plan_schema(tok: Tok, name: str, state: str, qs: list[Question],
                max_ctx_tokens: int = 16384, repeat_question: bool = False) -> Plan:
    """decider `build_schema_first` (question blocks, then the context, then one slot per
    question), without shuffling.  `repeat_question` (variant `schema_q`, pilot): every slot is
    ``\\n\\nQuestion k: <q>\\nAnswer k: (`` so the slot sees its question again (options only up front)."""
    multi = len(qs) > 1
    pre: list[int] = []
    for k, q in enumerate(qs):
        n = f" {k + 1}" if multi else ""
        pre += tok(f"{chr(10) * 2 if k else ''}Question{n}: {q.text}\nOptions:") + tok(option_lines(tok, q))
    suf = tok("\n\nContext:\n") + tok(state)[:max_ctx_tokens]
    slots = []
    for k, q in enumerate(qs):
        n = f" {k + 1}" if multi else ""
        if repeat_question:
            suf += tok(f"\n\nQuestion{n}: {q.text}\nAnswer{n}: (")
        else:
            suf += tok(f"{chr(10) * 2 if k == 0 else chr(10)}Answer{n}: (")
        slots.append(Slot(len(suf) - 1, labels_for(tok, len(q.options)), q.key))
    p = Plan(name, "schema_q" if repeat_question else "schema", pre, [Row(suf, slots)],
             cacheable_prefix=True, nq=len(qs))
    p.tokens = len(pre) + len(suf)
    p.tokens_cached = len(suf)
    return p


def build(tok: Tok, variant: str, name: str, state: str, qs: list[Question]) -> Plan:
    if variant == "base":
        return plan_base(tok, name, state, qs)
    if variant == "multi":
        return plan_multi(tok, name, state, qs)
    if variant == "multi_ans":
        return plan_multi(tok, name, state, qs, chain=True)
    if variant == "schema":
        return plan_schema(tok, name, state, qs)
    if variant == "schema_q":
        return plan_schema(tok, name, state, qs, repeat_question=True)
    if variant == "compact_a":
        return plan_base(tok, name, state, qs, variant, header="")
    if variant == "compact_b":
        return plan_base(tok, name, state, qs, variant, header="", style="bare")
    if variant == "compact_c":
        return plan_base(tok, name, state, qs, variant, header="", style="bare",
                         tail="\nAnswer:", space_labels=True)
    if variant == "compact_d":
        return plan_base(tok, name, state, qs, variant, header="\nOptions:", style="inline")
    if variant == "shared":
        return plan_shared(tok, name, state, qs)
    raise ValueError(variant)


# ------------------------------------------------------------------ inputs
def load_requests() -> list[tuple[str, str, list[Question]]]:
    reqs = json.loads(REQUESTS.read_text(encoding="utf-8"))
    out = []
    for r in reqs:
        qs = [render_question(k, v) for k, v in r["questions"].items()]
        out.append((r["name"], render_state(r["state"]), qs))
    return out


def load_reference() -> dict[tuple[str, str], dict]:
    """(request, question) -> {"ids": server ids, "probs": reference probabilities}."""
    out = {}
    for line in REF.read_text(encoding="utf-8").splitlines():
        req = json.loads(line)
        for row in req["rows"]:
            out[(req["name"], row["question"])] = {"ids": row["server_ids"], "probs": row["probs"],
                                                   "labels": row["labels"]}
    return out


def check_base(tok: Tok) -> int:
    """The base rows must equal the reference's server ids; returns the number of rows checked."""
    ref = load_reference()
    n = 0
    for name, state, qs in load_requests():
        p = plan_base(tok, name, state, qs)
        for row in p.rows:
            ids = p.prefix + row.ids
            k = row.slots[0].key
            if ids != ref[(name, k)]["ids"]:
                raise AssertionError(f"{name}/{k}: base row differs from the reference")
            n += 1
    return n


def main() -> None:
    tok = Tok()
    print(f"base rows == reference server ids: {check_base(tok)} rows")
    reqs = load_requests()
    totals: dict[str, tuple[int, int]] = {}
    per: dict[str, dict[str, int]] = {}
    for v in VARIANTS:
        t = tc = 0
        per[v] = {}
        for name, state, qs in reqs:
            p = build(tok, v, name, state, qs)
            t += p.tokens
            tc += p.tokens_cached
            per[v][name] = p.tokens
        totals[v] = (t, tc)
    b = totals["base"][0]
    print(f"{'variant':10s} {'tokens':>7s} {'saved':>7s} {'cached':>7s} {'saved':>7s}")
    for v, (t, tc) in totals.items():
        print(f"{v:10s} {t:7d} {100 * (b - t) / b:6.1f}% {tc:7d} {100 * (b - tc) / b:6.1f}%")
    if "--per-request" in sys.argv:
        names = [r[0] for r in reqs]
        print("request," + ",".join(VARIANTS))
        for n in names:
            print(n + "," + ",".join(str(per[v][n]) for v in VARIANTS))
    out = ROOT / "bench" / "results" / "pilot" / "prompt" / "tokens.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"totals": {v: {"tokens": t, "tokens_cached": tc}
                                          for v, (t, tc) in totals.items()},
                               "per_request": per}, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
