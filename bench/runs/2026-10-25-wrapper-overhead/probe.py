"""Compare accounting implementations on identical warmed calls, in alternating order."""

import collections
import json
import platform
import statistics
import timeit

import numpy as np
import scipy
from scipy import fft
from scipy.sparse.linalg import aslinearoperator
from scipy.spatial import distance

import fastscipy
from fastscipy import _patch


def main():
    fastscipy.patch(threads=1)
    x = np.arange(48.0).reshape(16, 3)
    image = np.arange(64.0).reshape(8, 8)
    integers = image.astype(np.int64)
    operator = aslinearoperator(np.eye(32))
    batched = np.ones((2, 32))
    accepted = _patch.wrap(len, lambda x: 42, "probe.accepted")
    declined = _patch.wrap(len, lambda x: NotImplemented, "probe.declined")
    cases = {
        "wrapper accepted": lambda: accepted("abc"),
        "wrapper declined": lambda: declined("abc"),
        "pdist 16x3": lambda: distance.pdist(x),
        "fftn 8x8": lambda: fft.fftn(image),
        "dctn 8x8": lambda: fft.dctn(image),
        "dctn integer fallback": lambda: fft.dctn(integers),
        "LinearOperator batched fallback": lambda: operator.matvec(batched),
    }
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threads": fastscipy.get_num_threads(),
        },
        "method": "9 alternating paired batches; 10000 calls per batch; median ns/call",
        "cases": {},
    }
    original_counts = _patch.counts
    try:
        for name, call in cases.items():
            samples = {"counter": [], "dictionary": []}
            for repeat in range(9):
                order = ("counter", "dictionary") if repeat % 2 == 0 else ("dictionary", "counter")
                for label in order:
                    _patch.counts = collections.Counter() if label == "counter" else _patch._CallCounts()
                    for _ in range(100):
                        call()
                    samples[label].append(timeit.timeit(call, number=10000) * 1e5)
            before, after = (statistics.median(samples[label]) for label in ("counter", "dictionary"))
            report["cases"][name] = {
                "before_ns": before,
                "after_ns": after,
                "speedup": before / after,
                "samples_ns": samples,
            }
    finally:
        _patch.counts = original_counts
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
