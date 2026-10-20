"""Exact comparison of SciPy results, for the tests and benchmarks."""

import functools
import hashlib
import json
import os
import pickle
import platform
import sys
import warnings

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


class ReferenceVarianceWarning(RuntimeWarning):
    """An explicitly checked Intel Accelerate reference was not byte-stable."""


@functools.cache
def _alignment_sensitive_reference():
    return (
        sys.platform == "darwin"
        and platform.machine().lower() == "x86_64"
        and np.__config__.show(mode="dicts").get("Build Dependencies", {}).get("blas", {}).get("name") == "accelerate"
    )


def reference_variance(call, expected, actual, validate):
    """Record a narrow, numerically validated exception to an unstable oracle.

    Callers must explicitly supply an independent numerical validator, which
    raises on failure and returns ``None`` or ``True`` on success. This
    never accepts a deterministic mismatch, changes ``identical``, or proves
    byte identity. Extra reference calls are outside benchmark timing.
    """
    if validate is None or not _alignment_sensitive_reference():
        return None
    import fastscipy

    def validated(a, b):
        result = validate(a, b)
        assert result is None or result is True, "reference validator rejected result"

    validated(expected, actual)
    references = [expected]
    for trial in range(1, 9):
        with fastscipy.disabled():
            repeated = call()
        references.append(repeated)  # Do not repeatedly recycle the same aligned allocation.
        validated(expected, repeated)
        if identical(expected, repeated):
            continue
        validated(repeated, actual)

        def fingerprint(value):
            return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()

        record = {
            "policy": "intel-accelerate-reference-variance",
            "test": os.environ.get("PYTEST_CURRENT_TEST"),
            "validator": f"{validate.__module__}.{validate.__qualname__}",
            "reference_trials": trial,
            "reference_sha256": fingerprint(expected),
            "repeated_reference_sha256": fingerprint(repeated),
            "actual_sha256": fingerprint(actual),
            "numpy": np.__version__,
            "platform": platform.platform(),
        }
        destination = os.environ.get("FASTSCIPY_REFERENCE_REPORT")
        if destination:
            with open(destination, "a", encoding="utf-8") as output:
                output.write(json.dumps(record) + "\n")
        warnings.warn(
            "Intel Accelerate reference changed on repeated unpatched calls; "
            f"{record['validator']} passed. Byte identity is not established for this comparison.",
            ReferenceVarianceWarning,
            stacklevel=2,
        )
        return record
    return None


def check(call, fast=True, *, reference_validation=None):
    """Assert ``call()`` gives the same result patched as on plain SciPy, and
    (if ``fast``) that it took a fast path. ``call`` must look SciPy functions
    up at call time (``scipy.special.erf(x)``, not a name bound before patching).
    If plain SciPy raises, the patched call must raise the same exception type.
    Optional ``reference_validation(expected, actual)`` must raise on a bad
    numerical result; only proven Intel Accelerate reference variation can
    use it in place of byte identity, with an explicit warning and report.
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
    if fast:
        assert fast_calls() > before, "fast path did not run"
    if not identical(expected, got):
        if reference_variance(call, expected, got, reference_validation) is None:
            assert_identical(expected, got)
    return got
