"""Rebinding helpers shared by every component.

Every replacement goes through ``replace`` so ``restore`` can undo it, and
every fast path goes through ``wrap`` (or checks ``enabled``) so
``fastscipy.disabled()`` and the fallback counters work everywhere.
"""

import functools
import os
import sys


class _CallCounts(dict):
    """Dictionary increments with zero for unseen paths, without inserting on reads.

    Call accounting needs neither Counter's multiset operations nor its numeric
    protocols. A plain dict subclass keeps the per-call update inexpensive.
    """

    def __missing__(self, key):
        return 0


enabled = True  # False inside fastscipy.disabled()
current = None  # component being patched; set by fastscipy.patch()
counts = _CallCounts()  # (function, "fast" | "scipy" | "error") -> calls
STRICT = os.environ.get("FASTSCIPY_STRICT", "") not in ("", "0")  # re-raise fast-path errors
_records = []  # (component, owner, attr, old value)
_identity_records = []  # (component, table, key, old value)
_MISSING = object()


def stats():
    """Call counts per (function, path) since the process started."""
    return dict(counts)


def require_validated(*prefixes):
    """Gate component versions and candidate 64-bit platform families.

    Per-kernel arithmetic calibration and release validation remain necessary:
    accepting a platform family does not certify a particular SciPy wheel.
    ``FASTSCIPY_ANY_VERSION=1`` and ``FASTSCIPY_ANY_PLATFORM=1`` override.
    """
    import platform

    import scipy

    if not scipy.__version__.startswith(prefixes) and not os.environ.get("FASTSCIPY_ANY_VERSION"):
        raise RuntimeError(f"validated on SciPy {', '.join(prefixes)}, found {scipy.__version__}")
    system, machine = sys.platform.lower(), platform.machine().lower()
    candidates = {
        "darwin": {"arm64", "x86_64"},
        "linux": {"aarch64", "arm64", "x86_64", "amd64"},
        "win32": {"arm64", "amd64", "x86_64"},
    }
    if machine not in candidates.get(system, ()) and not os.environ.get("FASTSCIPY_ANY_PLATFORM"):
        raise RuntimeError(f"outside candidate 64-bit platforms, found {system}-{machine}")


def replace(owner, attr, new):
    """Set ``owner.attr = new`` and remember the old value. Returns the old value."""
    old = getattr(owner, attr)
    _records.append((current, owner, attr, old))
    setattr(owner, attr, new)
    _copy_identity_entries(old, new)
    return old


def rebind(old, new, prefix="scipy"):
    """Replace ``old`` by ``new`` wherever a loaded module under ``prefix`` binds it
    (for example ``from ._ufuncs import erf`` inside SciPy)."""
    for mod in list(sys.modules.values()):
        name = getattr(mod, "__name__", None) or ""
        if name != prefix and not name.startswith(prefix + "."):
            continue
        for attr, val in list(getattr(mod, "__dict__", {}).items()):
            if val is old:
                replace(mod, attr, new)


def restore(component):
    """Undo every replacement made while patching ``component``."""
    keep = []
    for rec in reversed(_records):
        comp, owner, attr, old = rec
        if comp == component:
            setattr(owner, attr, old)
        else:
            keep.append(rec)
    _records[:] = reversed(keep)
    keep = []
    for rec in reversed(_identity_records):
        comp, table, key, old = rec
        if comp == component:
            if old is _MISSING:
                table.pop(key, None)
            else:
                table[key] = old
        else:
            keep.append(rec)
    _identity_records[:] = reversed(keep)


def wrap(original, fast, name=None):
    """A stand-in for ``original`` that tries ``fast`` first.

    ``fast`` takes the same arguments and returns the result, or
    ``NotImplemented`` to decline; SciPy then runs on the untouched arguments.
    A fast path that raises also falls back (and is counted as "error"), unless
    ``FASTSCIPY_STRICT=1``.
    """
    key = name or getattr(original, "__name__", repr(original))

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        if enabled:
            try:
                result = fast(*args, **kwargs)
            except Exception:
                if STRICT:
                    raise
                counts[key, "error"] += 1
            else:
                if result is not NotImplemented:
                    counts[key, "fast"] += 1
                    return result
        counts[key, "scipy"] += 1
        return original(*args, **kwargs)

    wrapper.fast = fast
    return wrapper


def _copy_identity_entries(old, new):
    """SciPy keys some registries by function object; give the stand-in the same entry."""
    try:
        from scipy._lib._array_api import xp_capabilities_table as table

        if old in table:
            _identity_records.append((current, table, new, table.get(new, _MISSING)))
            table[new] = table[old]
    except (ImportError, TypeError):
        pass
