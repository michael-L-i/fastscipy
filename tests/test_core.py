import importlib
import os
import sys
import time

import numpy as np
import pytest
import scipy.stats

import fastscipy
from fastscipy import _patch, _threads
from fastscipy.testing import identical


def _on():
    return [c for c, s in fastscipy.status().items() if s == "on"]


def _scipy_bindings():
    return {
        (m.__name__, k): id(v)
        for m in list(sys.modules.values())
        if (getattr(m, "__name__", None) or "").startswith("scipy")
        for k, v in list(vars(m).items())
    }


def test_unpatch_restores_every_binding():
    fastscipy.unpatch()
    before = _scipy_bindings()
    fastscipy.patch()
    fastscipy.unpatch()
    after = _scipy_bindings()
    assert {k: v for k, v in after.items() if k in before} == before
    fastscipy.patch()


def test_wrap_declines_and_counts():
    w = _patch.wrap(len, lambda x: NotImplemented if x == "" else 42, name="t_len")
    assert w("abc") == 42 and w("") == 0
    with fastscipy.disabled():
        assert w("abc") == 3
    assert _patch.counts["t_len", "fast"] == 1 and _patch.counts["t_len", "scipy"] == 2


def test_threads_roundtrip():
    n = fastscipy.get_num_threads()
    fastscipy.set_num_threads(3)
    assert fastscipy.get_num_threads() == 3
    assert _threads.parallel_map(lambda i: i * i, range(10)) == [i * i for i in range(10)]
    fastscipy.set_num_threads(n)


def test_chunks_cover_range():
    for n in (0, 1, 7, 1000, 12345):
        s = _threads.chunks(n, 100, threads=4)
        assert s[0].start == 0 and s[-1].stop == n
        assert all(a.stop == b.start for a, b in zip(s, s[1:]))


def test_identical():
    a = np.arange(6.0)
    assert identical(a, a.copy()) and not identical(a, a.astype(np.float32))
    assert not identical(np.float64(0.0), np.float64(-0.0))
    assert identical(scipy.stats.ttest_ind(a, a + 1), scipy.stats.ttest_ind(a, a + 1))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_forked_child_does_not_hang():
    """Each component's smoke() runs in the parent (starting every pool), then in a forked child."""
    smokes = [importlib.import_module(f"fastscipy._{c}").smoke for c in _on()]
    for s in smokes:
        s()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            for s in smokes:
                s()
            code = 0
        finally:
            os._exit(code)
    deadline = time.time() + 60
    while time.time() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            assert os.waitstatus_to_exitcode(status) == 0
            return
        time.sleep(0.05)
    os.kill(pid, 9)
    pytest.fail("forked child hung")


def test_unvalidated_platform_is_refused(monkeypatch):
    fastscipy.unpatch(["integrate"])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("FASTSCIPY_ANY_PLATFORM", raising=False)
    assert fastscipy.patch(["integrate"])["integrate"].startswith("off: RuntimeError: validated on macOS arm64")
    fastscipy.unpatch(["integrate"])
    monkeypatch.undo()
    assert fastscipy.patch(["integrate"])["integrate"] == "on"
