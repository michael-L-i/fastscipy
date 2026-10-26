"""Alternate genuine SciPy, previous fastscipy and candidate dispatch timings."""

import json
import platform
import statistics
import subprocess
import timeit
import types

import numpy as np
import scipy
from scipy import fft
from scipy.sparse.linalg import LinearOperator

import fastscipy
from fastscipy import _fft, _sparse_linalg
from fastscipy.testing import assert_identical


def main():
    baseline = types.ModuleType("fastscipy._baseline_sparse_linalg")
    baseline.__package__ = "fastscipy"
    source = subprocess.check_output(["git", "show", "a5961a5:python/fastscipy/_sparse_linalg.py"], text=True)
    exec(compile(source, "baseline_sparse_linalg.py", "exec"), baseline.__dict__)
    baseline_fft = types.ModuleType("fastscipy._baseline_fft")
    baseline_fft.__package__ = "fastscipy"
    source = subprocess.check_output(["git", "show", "a5961a5:python/fastscipy/_fft.py"], text=True)
    exec(compile(source, "baseline_fft.py", "exec"), baseline_fft.__dict__)
    candidate_patch = _sparse_linalg.patch
    candidate_fft_patch = _fft.patch
    x = np.arange(32.0)
    image = np.arange(64.0).reshape(8, 8)
    data = {
        "vector": x,
        "integer": x.astype(np.int64),
        "batched": np.tile(x, (2, 1)),
        "column": x[:, None],
    }
    operator = LinearOperator((32, 32), matvec=lambda a: a * 2, dtype=np.float64)
    cases = {f"LinearOperator {name}": lambda a=a: operator.matvec(a) for name, a in data.items()}
    # Conversion is fixture preparation, not part of the integer FFT timing.
    integers = image.astype(np.int64)
    cases.update(
        {
            "dctn integer fallback": lambda: fft.dctn(integers),
            "dctn float fast": lambda: fft.dctn(image),
        }
    )
    labels = ("scipy", "previous", "candidate")
    samples = {name: {label: [] for label in labels} for name in cases}
    fastscipy.unpatch()
    refs = {name: call() for name, call in cases.items()}
    try:
        for repeat in range(9):
            for label in labels[repeat % 3 :] + labels[: repeat % 3]:
                fastscipy.unpatch()
                _sparse_linalg.patch = baseline.patch if label == "previous" else candidate_patch
                _fft.patch = baseline_fft.patch if label == "previous" else candidate_fft_patch
                if label != "scipy":
                    fastscipy.patch(["fft", "sparse_linalg"], threads=1)
                for name, call in cases.items():
                    assert_identical(refs[name], call())
                    for _ in range(100):
                        call()
                    samples[name][label].append(timeit.timeit(call, number=10000) * 1e5)
    finally:
        fastscipy.unpatch()
        _sparse_linalg.patch = candidate_patch
        _fft.patch = candidate_fft_patch
    results = {}
    for name, values in samples.items():
        medians = {label: statistics.median(v) for label, v in values.items()}
        results[name] = {
            "median_ns": medians,
            "previous_over_candidate": medians["previous"] / medians["candidate"],
            "scipy_over_candidate": medians["scipy"] / medians["candidate"],
            "samples_ns": values,
        }
    print(
        json.dumps(
            {
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "threads": 1,
                },
                "method": "9 rotated state batches, 100 warmups then 10000 calls; median ns per call",
                "baseline_commit": "a5961a5",
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
