"""The `_nd_image` seam: every ndimage fast path is an override of one function
of SciPy's private C module. They are composed into a single stand-in module
(SciPy's functions plus the overrides; overlapping names are an error) that is
installed in each SciPy wrapper module.
"""

import functools
import types

from scipy.ndimage import _nd_image as orig

from . import _native, _patch

core = _native.ndimage
OVERRIDES = {}  # _nd_image function name -> wrapper that tries Rust first


def accelerated(name):
    """Register ``fn`` as the fast path for ``_nd_image.<name>``.

    ``fn`` takes the C function's arguments and returns ``False`` or
    ``NotImplemented`` to decline (without touching any output), ``True`` after
    writing the output in place, or the call's result.
    """
    original = getattr(orig, name)
    if name in OVERRIDES:
        raise RuntimeError(f"_nd_image.{name} is overridden twice")

    def decorate(fn):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            if _patch.enabled:
                try:
                    r = fn(*args, **kwargs)
                except Exception:
                    if _patch.STRICT:
                        raise
                    _patch.counts[name, "error"] += 1
                else:
                    if r is not False and r is not NotImplemented:
                        _patch.counts[name, "fast"] += 1
                        return None if r is True else r
            _patch.counts[name, "scipy"] += 1
            return original(*args, **kwargs)

        OVERRIDES[name] = wrapper
        return fn

    return decorate


def standin():
    m = types.ModuleType("fastscipy._nd_image")
    for name in dir(orig):
        if not name.startswith("__"):
            setattr(m, name, getattr(orig, name))
    for name, wrapper in OVERRIDES.items():
        setattr(m, name, wrapper)
    return m
