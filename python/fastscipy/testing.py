"""Exact comparison of SciPy results, for the tests and benchmarks."""

import numpy as np


def identical(a, b):
    """True if ``a`` and ``b`` have the same type and the same bytes, recursively.

    Arrays must match in dtype, shape, strides and bytes (so NaN payloads and
    signed zeros count); result objects match attribute by attribute.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.strides == b.strides and a.tobytes() == b.tobytes()
    if isinstance(a, (np.generic, float, complex)):
        return np.asarray(a).tobytes() == np.asarray(b).tobytes()
    if isinstance(a, (tuple, list)):
        if len(a) != len(b) or not all(map(identical, a, b)):
            return False
    elif isinstance(a, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    elif callable(a) and not hasattr(a, "__dict__"):
        return True
    if hasattr(a, "__dict__") and not isinstance(a, type) and type(a).__module__ != "builtins":
        return identical(vars(a), vars(b))
    return isinstance(a, (tuple, list)) or a is b or a == b


def assert_identical(a, b):
    assert identical(a, b), f"results differ:\n{a!r}\n{b!r}"


def fast_calls():
    """Total fast-path calls so far."""
    import fastscipy

    return sum(v for (_, path), v in fastscipy.stats().items() if path == "fast")


def check(call, fast=True):
    """Assert ``call()`` gives the same result patched as on plain SciPy, and
    (if ``fast``) that it took a fast path. ``call`` must look SciPy functions
    up at call time (``scipy.special.erf(x)``, not a name bound before patching).
    If plain SciPy raises, the patched call must raise the same exception type.
    """
    import fastscipy

    expected_error = None
    with fastscipy.disabled():
        try:
            expected = call()
        except Exception as error:
            expected_error = error
    if expected_error is not None:
        try:
            call()
        except Exception as error:
            assert type(error) is type(expected_error), (
                f"SciPy raised {expected_error!r}; the patched call raised {error!r}"
            )
            return None
        raise AssertionError(f"SciPy raised {expected_error!r}; the patched call did not") from None
    before = fast_calls()
    got = call()
    assert_identical(expected, got)
    if fast:
        assert fast_calls() > before, "fast path did not run"
    return got
