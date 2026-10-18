"""Rebinding helpers shared by every component.

Every replacement goes through ``replace`` so ``restore`` can undo it, and
every fast path goes through ``wrap`` (or checks ``enabled``) so
``fastscipy.disabled()`` and the fallback counters work everywhere.
"""
import collections
import functools
import os
import sys

enabled = True        # False inside fastscipy.disabled()
current = None        # component being patched; set by fastscipy.patch()
counts = collections.Counter()   # (function, "fast" | "scipy" | "error") -> calls
STRICT = os.environ.get("FASTSCIPY_STRICT", "") not in ("", "0")   # re-raise fast-path errors
_records = []         # (component, owner, attr, old value)


def stats():
    """Call counts per (function, path) since the process started."""
    return dict(counts)


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
            table[new] = table[old]
    except (ImportError, TypeError):
        pass
