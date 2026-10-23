"""Instruction count per block for the schedule knobs of q4gpu (PC side, no GPU needed).
usage: python sweep_sched.py PYVC6_SRC [sweep|prof]"""
import itertools
import os
import sys

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), sys.argv[1]]
import q4gpu  # noqa: E402


def sweep():
    for nprod, mul_adds, acc in itertools.product(
            [2, 3, 4], [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [1, 2, 2, 2]],
            [("add",), ("add", "mul")]):
        q4gpu.NPROD, q4gpu.MUL_ADDS, q4gpu.ACC_ALUS = nprod, mul_adds, acc
        row = []
        for n in (7, 8, 11):
            try:
                st = q4gpu.Kernel(n).stats()
                row.append(f"n={n}: {st['instr']:4d} ({st['mac_per_instr']:.3f})")
            except ValueError as e:
                row.append(f"n={n}: {e}")
        print(f"nprod={nprod} mul_adds={mul_adds} acc={'+'.join(acc):7s} | " + " | ".join(row))


def prof():
    for n in (7, 8, 11):
        sl = q4gpu.Kernel(n).body_slots
        sm = [i for i, s in enumerate(sl) if s.mul is not None and s.mul.name == "smul24"
              and s.mul.srcs[1] is q4gpu.r5]
        mac = sl[sm[0]:sm[-1] + 1]
        post = sl[sm[-1] + 1:]
        idle = lambda ss, u: sum(1 for s in ss if getattr(s, u) is None)  # noqa: E731
        print(f"n={n}: total {len(sl)}; pre {sm[0]}; MAC {len(mac)} (32n+28 = {32 * n + 28}) "
              f"idle add {idle(mac, 'add')} mul {idle(mac, 'mul')}; post {len(post)} "
              f"idle add {idle(post, 'add')} mul {idle(post, 'mul')}")
        print("   MAC[0:48]: " + " ".join(("A" if s.add else ".") + ("M" if s.mul else ".")
                                        for s in mac[:48]))
        for i, s in enumerate(sl[:10]):
            print(f"   {i:3d} add={s.add.name if s.add else '-':7s} mul={s.mul.name if s.mul else '-':7s} "
                  f"sigs={[g.name for g in s.sigs]}")
        print("   post:      " + " ".join(("A" if s.add else ".") + ("M" if s.mul else ".")
                                        for s in post))


if __name__ == "__main__":
    (prof if sys.argv[2:] == ["prof"] else sweep)()
