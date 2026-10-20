import importlib
import os
import subprocess
import sys
import threading
import time
import types
import weakref

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


def test_restore_releases_identity_registry_wrappers(monkeypatch):
    from scipy._lib._array_api import xp_capabilities_table as table

    original = lambda: None
    metadata = object()
    monkeypatch.setitem(table, original, metadata)
    monkeypatch.setattr(_patch, "current", "test_registry")
    owner = types.SimpleNamespace(fn=original, alias=original)
    before = dict(table)
    for _ in range(3):
        wrapped = _patch.wrap(original, lambda: None)
        reference = weakref.ref(wrapped)
        try:
            _patch.replace(owner, "fn", wrapped)
            _patch.replace(owner, "alias", wrapped)
            assert table[wrapped] is metadata
        finally:
            _patch.restore("test_registry")
        assert owner.fn is original and owner.alias is original
        assert table == before
        del wrapped
        assert reference() is None


def test_restore_preserves_existing_identity_registry_entry(monkeypatch):
    from scipy._lib._array_api import xp_capabilities_table as table

    original, replacement = lambda: None, lambda: None
    old_metadata = object()
    monkeypatch.setitem(table, original, object())
    monkeypatch.setitem(table, replacement, old_metadata)
    monkeypatch.setattr(_patch, "current", "test_registry_existing")
    owner = types.SimpleNamespace(fn=original)
    try:
        _patch.replace(owner, "fn", replacement)
    finally:
        _patch.restore("test_registry_existing")
    assert table[replacement] is old_metadata


def test_disabled_overlapping_threads():
    entered = threading.Event()
    release = threading.Event()
    observed = []

    def other_thread():
        with fastscipy.disabled():
            entered.set()
            if release.wait(timeout=5):
                observed.append(_patch.enabled)

    worker = threading.Thread(target=other_thread)
    try:
        with fastscipy.disabled():
            worker.start()
            assert entered.wait(timeout=5)
        # The worker's context must remain disabled after this one exits.
        assert not _patch.enabled
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert observed == [False]
    assert _patch.enabled


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_disabled_lock_reset_after_fork():
    # The child inherits this locked mutex, but the thread that owned it may
    # not exist there. Its inherited disabled context must still be usable.
    with fastscipy.disabled():
        with fastscipy._disabled_lock:
            pid = os.fork()
        if pid == 0:
            code = 1
            try:
                with fastscipy.disabled():
                    assert not _patch.enabled
                assert not _patch.enabled
                assert fastscipy._disabled_count == 1
                code = 0
            finally:
                os._exit(code)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                assert os.waitstatus_to_exitcode(status) == 0
                break
            time.sleep(0.01)
        else:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            pytest.fail("forked child hung entering disabled()")
    assert _patch.enabled


def test_threads_roundtrip():
    n = fastscipy.get_num_threads()
    fastscipy.set_num_threads(3)
    assert fastscipy.get_num_threads() == 3
    assert _threads.parallel_map(lambda i: i * i, range(10)) == [i * i for i in range(10)]
    fastscipy.set_num_threads(n)


def test_parallel_map_drains_before_fallback(monkeypatch):
    old_threads = fastscipy.get_num_threads()
    fastscipy.set_num_threads(2)
    started, release, finished, draining, returned = (threading.Event() for _ in range(5))
    real_wait = _threads.wait
    nested_threads = []
    cancelled = []
    visited = []
    fallback_observations = []
    error = ValueError("failed chunk")
    caught = []

    def watch_wait(futures):
        cancelled.append(futures[-1].cancelled())
        draining.set()
        return real_wait(futures)

    monkeypatch.setattr(_threads, "wait", watch_wait)

    def chunk(i):
        visited.append(i)
        if i == 0:
            assert started.wait(timeout=5)
            raise error
        started.set()
        assert release.wait(timeout=5)
        if i != 1:
            return
        worker = threading.get_ident()
        nested_threads.extend(_threads.parallel_map(lambda _: threading.get_ident(), range(2)))
        assert nested_threads == [worker, worker]
        finished.set()

    def run():
        try:
            _threads.parallel_map(chunk, range(4))
        except ValueError as exc:
            caught.append(exc)
            # This is where _patch.wrap can call SciPy on the original arguments.
            fallback_observations.append(finished.is_set())
        finally:
            returned.set()

    caller = threading.Thread(target=run)
    try:
        caller.start()
        assert draining.wait(timeout=5)
        assert _threads._busy.locked()
        assert not returned.is_set()
        assert not finished.is_set()
    finally:
        release.set()
        caller.join(timeout=5)
        fastscipy.set_num_threads(old_threads)
    assert not caller.is_alive()
    assert caught == [error]
    assert cancelled == [True]
    assert 3 not in visited
    assert fallback_observations == [True]
    assert not _threads._busy.locked()


def test_parallel_map_nested_calls():
    assert _threads.parallel_map(lambda i: _threads.parallel_map(lambda j: i + j, range(2)), range(2)) == [
        [0, 1],
        [1, 2],
    ]


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


def test_import_without_fork_hooks():
    """Windows has no register_at_fork; imports and patching must still work."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; "
            "getattr(os, 'register_at_fork', None) and delattr(os, 'register_at_fork'); "
            "import fastscipy; "
            "assert all(v == 'on' for v in fastscipy.patch(threads=1).values()), fastscipy.status()",
        ],
        check=True,
        timeout=30,
    )


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
