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

from . import _patch
from ._patch import stats
from ._threads import get_num_threads, set_num_threads

__version__ = "0.1.0"
__all__ = ["patch", "unpatch", "status", "stats", "disabled", "set_num_threads", "get_num_threads"]

COMPONENTS = (
    "ndimage",
    "special",
    "stats",
    "signal",
    "sparse",
    "linalg",
    "interpolate",
    "spatial",
    "csgraph",
    "integrate",
    "optimize",
)
_status = {}
log = logging.getLogger("fastscipy")


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
    old, _patch.enabled = _patch.enabled, False
    try:
        yield
    finally:
        _patch.enabled = old
