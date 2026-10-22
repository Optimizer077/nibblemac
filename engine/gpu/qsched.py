"""A small list scheduler for V3D 4.2 QPU code (used with py-videocore6's assembler).

Ops are given in program order with explicit registers; `schedule` packs them into
dual-issue instructions (one add-ALU op + one mul-ALU op + signals) under the V3D 4.2
rules we rely on:
  - every ALU result, ldunif (-> r5) and ldtmu result is readable by the next instruction;
    an instruction reads its sources before it writes, so a write may share the instruction
    of the last read of the old value (WAR distance 0), RAW / WAW distance >= 1;
  - at most two distinct register-file (rf) reads per instruction, one if a small immediate
    is used (it takes the raddr_b slot), and all small immediates of an instruction equal;
  - signal sets limited to the encodable ones; a small immediate allows only ldtmu with it;
  - TMU writes (tmuc / tmud / tmua) and ldtmu keep program order, one per instruction;
  - ldunif keeps program order (the uniform stream is consumed in program order).
Pseudo registers: "TMU" (TMU ordering), "UNIF" (uniform order), "FLAGS" (cond pushes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

TMU_REGS = {"tmua", "tmud", "tmuc", "tmuau"}
# signal combinations that the instruction encoding allows (py-videocore6 Signals.pack)
SIG_OK = [frozenset(s) for s in ([], ["thrsw"], ["ldunif"], ["ldtmu"], ["ldtmu", "ldunif"],
                                 ["ldunifrf"], ["ldunifa"], ["ldunifarf"])]
SIG_OK_SMIMM = [frozenset(), frozenset(["ldtmu"])]


def _name(r) -> str | None:
    return getattr(r, "name", None)


def _is_rf(r) -> bool:
    return getattr(r, "magic", None) == 0


@dataclass
class Op:
    """One ALU op (alus = allowed ALUs) or one signal (kind == 'sig')."""
    kind: str                      # "alu" or "sig"
    name: str                      # ALU op name, or signal name (ldunif / ldtmu / ldunifrf)
    dst: object = None
    srcs: tuple = ()
    alus: tuple = ("add",)
    cond: str | None = None
    tag: object = None             # for ldunif: what the uniform holds (host fills it)
    reads: set = field(default_factory=set)
    writes: set = field(default_factory=set)
    imm: object = None
    thrsw: bool = False            # ALU op carries the thrsw signal (TMU load requests)

    def finish(self) -> "Op":
        for s in self.srcs:
            if isinstance(s, (int, float)) and not isinstance(s, bool):
                if self.imm is not None and self.imm != s:
                    raise ValueError(f"two immediates in one op {self}")
                self.imm = s
            elif _name(s):
                self.reads.add(_name(s))
        if self.kind == "sig":
            if self.name == "ldunif":
                self.writes |= {"r5", "UNIF"}
                self.reads.add("UNIF")
            else:  # ldtmu / ldunifrf write dst
                self.writes.add(_name(self.dst))
                if self.name == "ldtmu":
                    self.reads.add("TMU"), self.writes.add("TMU")
                else:
                    self.reads.add("UNIF"), self.writes.add("UNIF")
        else:
            d = _name(self.dst)
            if d and d != "null":
                self.writes.add(d)
            if d in TMU_REGS:
                self.reads.add("TMU"), self.writes.add("TMU")
        if self.cond and self.cond.startswith("push"):
            self.writes.add("FLAGS")
        return self

    def rf_reads(self) -> set:
        return {_name(s) for s in self.srcs if _is_rf(s)} if self.kind == "alu" else set()


def alu(name, dst, *srcs, alus=("add",), cond=None, thrsw=False) -> Op:
    return Op("alu", name, dst, tuple(srcs), alus, cond, thrsw=thrsw).finish()


def sig(name, dst=None, tag=None) -> Op:
    return Op("sig", name, dst, (), (), None, tag).finish()


class Slot:
    __slots__ = ("add", "mul", "sigs", "no_sigs")

    def __init__(self):
        self.add = None
        self.mul = None
        self.sigs: list[Op] = []
        self.no_sigs = False      # one of the two instructions after a thrsw: no signals

    def ops(self):
        return [o for o in (self.add, self.mul) if o is not None]

    def has_thrsw(self) -> bool:
        return any(o.thrsw for o in self.ops())

    def fits(self, op: Op, unit: str | None) -> bool:
        if unit is not None and getattr(self, unit) is not None:
            return False
        if self.no_sigs and (op.kind == "sig" or op.thrsw):
            return False
        alu_ops = self.ops() + ([op] if op.kind == "alu" else [])
        sigs = self.sigs + ([op] if op.kind == "sig" else [])
        imms = {o.imm for o in alu_ops if o.imm is not None}
        if len(imms) > 1:
            return False
        rfr = set().union(*[o.rf_reads() for o in alu_ops]) if alu_ops else set()
        names = [s.name for s in sigs] + ["thrsw"] * sum(1 for o in alu_ops if o.thrsw)
        sset = frozenset(names)
        if len(sset) != len(names):
            return False
        if imms:
            if len(rfr) > 1 or sset not in SIG_OK_SMIMM:
                return False
        elif len(rfr) > 2 or sset not in SIG_OK:
            return False
        # one write per register per instruction
        dsts = [_name(o.dst) for o in alu_ops if _name(o.dst) not in (None, "null")]
        dsts += [_name(s.dst) for s in sigs if s.name != "ldunif"]
        dsts += ["r5" for s in sigs if s.name == "ldunif"]
        if len(dsts) != len(set(dsts)):
            return False
        # at most one magic TMU write per instruction
        if sum(1 for d in dsts if d in TMU_REGS) > 1:
            return False
        return True


def schedule(ops: list[Op], prefer_mul: bool = False) -> list[Slot]:
    """Greedy in-order list scheduling: each op goes to the earliest legal slot."""
    slots: list[Slot] = []
    last_w: dict[str, int] = {}
    last_r: dict[str, int] = {}
    for op in ops:
        t = 0
        for r in op.reads:
            if r in last_w:
                t = max(t, last_w[r] + 1)
        for w in op.writes:
            if w in last_r:
                # ALU writes and ldunif -> r5 land after the reads of the same instruction
                # (WAR 0, py-videocore6 test_signals); an ldtmu write is kept one instruction
                # after the last read of the old value (not covered by any test)
                t = max(t, last_r[w] + (1 if op.kind == "sig" and op.name == "ldtmu" else 0))
            if w in last_w:
                t = max(t, last_w[w] + 1)
        units = [None] if op.kind == "sig" else (
            sorted(op.alus, key=lambda u: (u != "mul") if prefer_mul else (u != "add")))
        while True:
            while len(slots) <= t + 2:
                slots.append(Slot())
            placed = None
            for u in units:
                if slots[t].fits(op, u):
                    placed = u
                    break
            ok = placed is not None or (op.kind == "sig" and slots[t].fits(op, None))
            if ok and op.thrsw:   # the two instructions after a thrsw carry no signals
                ok = not any(slots[t + d].sigs or slots[t + d].has_thrsw() for d in (1, 2))
            if ok:
                break
            t += 1
        s = slots[t]
        if op.kind == "sig":
            s.sigs.append(op)
        else:
            setattr(s, placed, op)
            op.alus = (placed,)
            if op.thrsw:
                slots[t + 1].no_sigs = slots[t + 2].no_sigs = True
        for r in op.reads:
            last_r[r] = max(last_r.get(r, -1), t)
        for w in op.writes:
            last_w[w] = t
    while slots and not slots[-1].ops() and not slots[-1].sigs:   # trailing padding
        if any(s.has_thrsw() for s in slots[-3:-1]):
            break
        slots.pop()
    return slots


def uniform_tags(ops: list[Op]) -> list:
    """Tags of the ldunif ops in program order (= order the hardware consumes them)."""
    return [o.tag for o in ops if o.kind == "sig" and o.name == "ldunif"]


def emit(g: dict, slots: list[Slot]) -> None:
    """Emit scheduled slots with the assembler functions in g (globals of a @qpu function)."""
    for s in slots:
        sigs = []
        for o in s.sigs:
            if o.name == "ldunif":
                sigs.append(g["ldunif"])
            else:
                sigs.append(g[o.name](o.dst))
        if s.has_thrsw():
            sigs.append(g["thrsw"])
        sig_arg = sigs if sigs else None
        a, m = s.add, s.mul
        if a is not None:
            name, srcs = a.name, a.srcs
            if name == "mov":
                name, srcs = "bor", (srcs[0], srcs[0])
            kw = {"cond": a.cond} if a.cond else {}
            inst = g[name](a.dst, *srcs, sig=sig_arg, **kw) if srcs or name not in ("tidx", "eidx") \
                else g[name](a.dst, sig=sig_arg, **kw)
        else:
            inst = g["nop"](sig=sig_arg)
        if m is not None:
            kw = {"cond": m.cond} if m.cond else {}
            getattr(inst, m.name)(m.dst, *m.srcs, **kw)


def stats(slots: list[Slot]) -> dict:
    n = len(slots)
    return {"instr": n, "add": sum(1 for s in slots if s.add), "mul": sum(1 for s in slots if s.mul)}
