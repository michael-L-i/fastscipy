"""Drop-in acceleration for SciPy.

    import fastscipy
    fastscipy.patch()    # SciPy calls made after this are faster, with the same results

Each component replaces SciPy internals with faster code that returns the same
bytes as SciPy, and hands any call it does not cover back to SciPy unchanged.
Call ``patch()`` before importing libraries that do ``from scipy.x import f``.
"""

import contextlib
import importlib
import logging
import os
import threading

from . import _patch
from ._patch import stats
from ._threads import get_num_threads, set_num_threads

__version__ = "0.1.0a1"
__all__ = ["patch", "unpatch", "status", "stats", "disabled", "set_num_threads", "get_num_threads"]

COMPONENTS = (
    "ndimage",
    "special",
    "stats",
    "signal",
    "sparse",
    "sparse_linalg",
    "linalg",
    "interpolate",
    "spatial",
    "csgraph",
    "cluster",
    "integrate",
    "optimize",
)
_status = {}
_disabled_lock = threading.Lock()
_disabled_count = 0
_disabled_old = True
log = logging.getLogger("fastscipy")


def _after_fork():
    global _disabled_lock
    _disabled_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def patch(components=None, threads=None):
    """Make SciPy faster in this process.

    ``components`` limits which SciPy areas are patched (default: all);
    ``threads`` sets the thread budget (default: all cores, or
    ``FASTSCIPY_NUM_THREADS``). Returns ``status()``. A component that cannot
    be patched (unsupported SciPy version, missing hook) is left off and its
    reason is recorded; ``patch()`` never raises for that.
    """
    if threads is not None:
        set_num_threads(threads)
    for name in components or COMPONENTS:
        if _status.get(name) == "on":
            continue
        _patch.current = name
        try:
            importlib.import_module(f"{__name__}._{name}").patch()
            _status[name] = "on"
        except Exception as e:
            _patch.restore(name)
            _status[name] = f"off: {type(e).__name__}: {e}"
            log.info("fastscipy: %s not patched: %s", name, _status[name])
        finally:
            _patch.current = None
    return status()


def unpatch(components=None):
    """Restore SciPy's own code (all components, or the ones named)."""
    for name in components or list(_status):
        _patch.restore(name)
        _status.pop(name, None)


def status():
    """``{component: "on" | "off: <reason>"}`` for every component ``patch()`` tried."""
    return dict(_status)


@contextlib.contextmanager
def disabled():
    """Run SciPy's own code inside the block without unpatching (for comparisons)."""
    global _disabled_count, _disabled_old
    with _disabled_lock:
        if _disabled_count == 0:
            _disabled_old = _patch.enabled
            _patch.enabled = False
        _disabled_count += 1
    try:
        yield
    finally:
        with _disabled_lock:
            _disabled_count -= 1
            if _disabled_count == 0:
                _patch.enabled = _disabled_old
