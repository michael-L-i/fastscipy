"""The thread budget, and a Python pool for running SciPy's own GIL-releasing
kernels on several chunks at once.

Rust kernels use the Rust pool in ``_native``; Python wrappers use
``parallel_map``. Both read the same thread count. Only one ``parallel_map``
runs at a time; a second caller (another user thread, or a nested call) runs
serially instead of oversubscribing. Both pools are rebuilt after ``fork``.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait

from . import _native

_pool = None
_busy = threading.Lock()


def get_num_threads():
    return _native.get_num_threads()


def set_num_threads(threads):
    """Set the thread budget for every component (1 = single-threaded)."""
    global _pool
    _native.set_num_threads(int(threads))
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None


def _after_fork():
    global _pool, _busy
    _pool, _busy = None, threading.Lock()


os.register_at_fork(after_in_child=_after_fork)


def chunks(n, min_size, threads=None):
    """Split ``range(n)`` into at most ``4 * threads`` slices of at least ``min_size``."""
    threads = threads or get_num_threads()
    k = max(1, min(4 * threads, n // max(1, min_size)))
    bounds = [n * i // k for i in range(k + 1)]
    return [slice(a, b) for a, b in zip(bounds, bounds[1:])]


def parallel_map(fn, items):
    """``[fn(x) for x in items]``, on the shared pool when it is free."""
    global _pool
    items = list(items)
    threads = get_num_threads()
    if threads <= 1 or len(items) <= 1 or not _busy.acquire(blocking=False):
        return [fn(x) for x in items]
    try:
        if _pool is None:
            _pool = ThreadPoolExecutor(threads, thread_name_prefix="fastscipy")
        futures = []
        completed = False
        try:
            for item in items:
                futures.append(_pool.submit(fn, item))
            results = [future.result() for future in futures]
            completed = True
            return results
        finally:
            if not completed:
                # Running chunks can still write their outputs. Drain them before
                # a wrapper falls back to SciPy or another call reuses the pool.
                for future in futures:
                    future.cancel()
                wait(futures)
    finally:
        _busy.release()
