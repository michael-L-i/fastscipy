"""Diagnostic lower-overhead wrapper; not a complete dispatch implementation.

This deliberately omits threaded dispatch and the .fast testing hook to assess
whether fusing guards and accounting can beat SciPy even for a trivial input.
"""

import functools
import json
import pathlib
import platform
import statistics
import time

import numpy as np
import scipy
import scipy.spatial.distance as dist

from fastscipy import _patch, _spatial


def main():
    original = dist.cdist
    direct = {name: dist._METRIC_ALIAS[name].cdist_func for name in _spatial._DIRECT_PDIST}

    @functools.wraps(original)
    def fused(XA, XB, metric="euclidean", *, out=None, **kwargs):
        if _patch.enabled:
            try:
                if (
                    not kwargs
                    and out is None
                    and type(metric) is str
                    and metric in direct
                    and type(XA) is np.ndarray
                    and type(XB) is np.ndarray
                    and XA.ndim == XB.ndim == 2
                    and XA.dtype == XB.dtype == np.float64
                    and XA.shape[1] == XB.shape[1]
                    and XA.size * XB.shape[0] < _spatial._WORK
                ):
                    result = direct[metric](XA, XB, out=None)
                    _patch.counts["cdist", "fast"] += 1
                    return result
            except Exception:
                if _patch.STRICT:
                    raise
                _patch.counts["cdist", "error"] += 1
        _patch.counts["cdist", "scipy"] += 1
        return original(XA, XB, metric, out=out, **kwargs)

    a = np.random.default_rng(0).normal(size=(4, 3))
    b = a.copy()
    assert np.array_equal(original(a, b).view(np.uint64), fused(a, b).view(np.uint64))
    calls = {"scipy": original, "fused_diagnostic": fused}
    samples = {name: [] for name in calls}
    for call in calls.values():
        for _ in range(100):
            call(a, b)
    for repeat in range(9):
        names = list(calls)
        if repeat % 2:
            names.reverse()
        for name in names:
            call = calls[name]
            start = time.perf_counter_ns()
            for _ in range(10_000):
                call(a, b)
            samples[name].append((time.perf_counter_ns() - start) / 10_000)
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "batch_calls": 10_000,
        "median_ns": {name: statistics.median(values) for name, values in samples.items()},
        "samples_ns": samples,
    }
    pathlib.Path(__file__).with_name("fused-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["median_ns"])


if __name__ == "__main__":
    main()
