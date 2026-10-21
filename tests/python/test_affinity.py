"""The engine must not change the calling thread's CPU affinity (issue #11).

Up to 0.1.0 the NEON build's thread pool (engine/src/matvec_fast.c pf_pool_create) pinned the
thread that called pinfer_load to CPU 0, and every thread the application created afterwards
inherited that mask. The scalar build (MATVEC=ref, the x86 host) has no pool and passes
trivially; the Pi's NEON library is the real test. With the NEON build the workers must
still be pinned, one per CPU of the caller's mask after the first.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def _threads() -> dict[int, set[int]]:
    """tid -> allowed CPUs of every thread of this process."""
    out: dict[int, set[int]] = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            out[int(tid)] = os.sched_getaffinity(int(tid))
        except OSError:   # thread exited meanwhile
            pass
    return out


def _is_fast(model: Any) -> bool:
    return bool(model.info().get("build", {}).get("matvec") == "fast")


def _in_thread(fn: Callable[[], Any]) -> Any:
    """Run fn on a new thread (so that affinity changes stay there); re-raise its error."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            box["error"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_load_and_eval_keep_caller_affinity(lib: Any, model_path: Path) -> None:
    import pinfer

    def body() -> dict[str, Any]:
        before = os.sched_getaffinity(0)
        tids = set(_threads())
        m = pinfer.Model.load(model_path, tokenizer=False, n_threads=4, max_ctx=64)
        try:
            after_load = os.sched_getaffinity(0)
            m.eval([1, 2, 3])
            after_eval = os.sched_getaffinity(0)
            child = _in_thread(lambda: os.sched_getaffinity(0))   # created after the load
            workers = {t: c for t, c in _threads().items() if t not in tids}
            fast = _is_fast(m)
        finally:
            m.close()
        return {"before": before, "after_load": after_load, "after_eval": after_eval,
                "child": child, "workers": workers, "fast": fast}

    r = _in_thread(body)
    assert r["after_load"] == r["before"], "pinfer_load changed the caller's affinity"
    assert r["after_eval"] == r["before"], "pinfer_eval changed the caller's affinity"
    assert r["child"] == r["before"], "a thread created after pinfer_load lost CPUs"
    if (r["fast"] or r["workers"]) and len(r["before"]) >= 4:
        want = sorted(r["before"])[1:4]   # workers 1..3 on the 2nd..4th allowed CPU
        pinned = sorted(next(iter(c)) for c in r["workers"].values() if len(c) == 1)
        assert pinned == want, f"workers pinned to {pinned}, expected {want}"


def test_restricted_caller_mask(lib: Any, model_path: Path) -> None:
    """A caller limited to two CPUs: its mask is kept, and the NEON pool runs one worker,
    pinned to the second of those CPUs (more threads than allowed cores would share one)."""
    import pinfer

    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 3:
        pytest.skip("needs >= 3 CPUs")
    mask = set(cpus[-2:])

    def body() -> dict[str, Any]:
        os.sched_setaffinity(0, mask)
        tids = set(_threads())
        m = pinfer.Model.load(model_path, tokenizer=False, n_threads=4, max_ctx=64)
        try:
            logits = m.eval([1, 2, 3])
            after = os.sched_getaffinity(0)
            workers = {t: c for t, c in _threads().items() if t not in tids}
            fast = _is_fast(m)
        finally:
            m.close()
        return {"after": after, "workers": workers, "fast": fast, "ok": bool(logits.size)}

    r = _in_thread(body)
    assert r["after"] == mask and r["ok"]
    if r["fast"] or r["workers"]:   # the NEON pool (the scalar build starts no threads)
        assert list(r["workers"].values()) == [{cpus[-1]}]
