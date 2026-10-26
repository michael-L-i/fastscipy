"""Executor ownership while callers repeat or change the shared budget."""

import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import fastscipy
from fastscipy import _threads


@pytest.fixture
def two_threads():
    previous = fastscipy.get_num_threads()
    fastscipy.set_num_threads(2)
    try:
        yield
    finally:
        fastscipy.set_num_threads(previous)


def test_unchanged_budget_keeps_warm_executor(two_threads):
    assert _threads.parallel_map(abs, [-1, -2]) == [1, 2]
    executor = _threads._pool
    fastscipy.set_num_threads("2")  # Preserve the public integer conversion.
    assert _threads._pool is executor
    assert _threads.parallel_map(abs, [-3, -4]) == [3, 4]


@pytest.mark.parametrize("value", [0, -1, "invalid", 1 << 100])
def test_invalid_budget_keeps_executor_and_setting(two_threads, value):
    _threads.parallel_map(abs, [-1, -2])
    executor = _threads._pool
    with pytest.raises((ValueError, OverflowError)):
        fastscipy.set_num_threads(value)
    assert fastscipy.get_num_threads() == 2
    assert _threads._pool is executor
    assert _threads.parallel_map(abs, [-3, -4]) == [3, 4]


def test_reconfiguration_cannot_interrupt_submission(two_threads, monkeypatch):
    first_submission, setter_attempt = threading.Event(), threading.Event()
    real_executor = _threads.ThreadPoolExecutor
    real_lock = _threads._pool_lock
    errors = []

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread() is setter:
                setter_attempt.set()
            return real_lock.__enter__()

        def __exit__(self, *args):
            return real_lock.__exit__(*args)

    class ObservedExecutor(real_executor):
        def submit(self, fn, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            if not first_submission.is_set():
                first_submission.set()
                assert setter_attempt.wait(timeout=5)
            return future

    def change_budget():
        try:
            assert first_submission.wait(timeout=5)
            fastscipy.set_num_threads(3)
        except BaseException as error:
            errors.append(error)

    # Start without an existing executor so every submission uses the observer.
    fastscipy.set_num_threads(1)
    fastscipy.set_num_threads(2)
    monkeypatch.setattr(_threads, "_pool_lock", ObservedLock())
    monkeypatch.setattr(_threads, "ThreadPoolExecutor", ObservedExecutor)
    setter = threading.Thread(target=change_budget)
    setter.start()
    try:
        assert _threads.parallel_map(abs, [-1, -2, -3, -4]) == [1, 2, 3, 4]
    finally:
        setter.join(timeout=5)
    assert not setter.is_alive()
    assert not errors
    assert fastscipy.get_num_threads() == 3
    assert _threads.parallel_map(abs, [-5, -6]) == [5, 6]


def test_worker_can_change_budget_and_run_nested_map():
    # Isolate the deadlock guard: a broken lifecycle lock must not also hang
    # fixture teardown or the interpreter's executor shutdown in this process.
    code = textwrap.dedent("""\
        from fastscipy import _threads

        _threads.set_num_threads(2)

        def chunk(i):
            if i == 0:
                _threads.set_num_threads(3)
            return _threads.parallel_map(lambda j: i + j, [0, 1])

        assert _threads.parallel_map(chunk, [0, 1, 2, 3]) == [[0, 1], [1, 2], [2, 3], [3, 4]]
        assert _threads.get_num_threads() == 3
        assert _threads.parallel_map(abs, [-1, -2]) == [1, 2]
    """)
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=10)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_fork_resets_held_executor_lock(two_threads):
    with _threads._pool_lock:
        pid = os.fork()
        if pid == 0:
            os._exit(0 if _threads._pool_lock.acquire(blocking=False) else 1)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            assert os.waitstatus_to_exitcode(status) == 0
            return
        time.sleep(0.01)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    pytest.fail("forked child inherited the executor lifecycle lock")
