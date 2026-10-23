"""Compare true unpatched SciPy, the previous spatial wrapper, and this checkout.

Run in an environment with the source checkout on PYTHONPATH and numerical
thread limits set. The previous implementation is read from an explicit git
revision; no installed-package or disabled-wrapper timing is used as SciPy.
"""

import importlib.util
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import timeit

import numpy as np
import scipy
import scipy.spatial.distance as dist

import fastscipy
from fastscipy import _patch, _spatial
from fastscipy.testing import assert_identical

PREVIOUS = "f60dca0"
source = subprocess.check_output(["git", "show", f"{PREVIOUS}:python/fastscipy/_spatial.py"], text=True)
spec = importlib.util.spec_from_loader("fastscipy._spatial_previous", loader=None)
previous = importlib.util.module_from_spec(spec)
exec(compile(source, "<previous spatial>", "exec"), previous.__dict__)
original = dist.pdist
assert not hasattr(original, "__wrapped__"), "Reference must be true unpatched SciPy"
functions = {
    "scipy": original,
    "previous": _patch.wrap(original, previous._pdist_fast(dist.cdist), "pdist"),
    "candidate": _patch.wrap(original, _spatial._pdist_fast(dist.cdist), "pdist"),
}
rng = np.random.default_rng(905)
cases = []
for shape in ((4, 3), (16, 3), (32, 16), (100, 16)):
    x = rng.random(shape)
    for metric in ("euclidean", "cityblock", "sqeuclidean"):
        cases.append((f"{metric} {shape[0]}x{shape[1]}", x, (metric,), {}, 1000))
x = rng.random((16, 3))
cases.extend(
    [
        ("default 16x3", x, (), {}, 1000),
        ("keyword metric 16x3", x, (), {"metric": "euclidean"}, 1000),
        ("Fortran 16x3", np.asfortranarray(x), (), {}, 1000),
        ("reversed 16x3", x[::-1], (), {}, 1000),
        ("float32 control 16x3", x.astype(np.float32), (), {}, 1000),
        ("list control 16x3", x.tolist(), (), {}, 1000),
        ("weights control 16x3", x, (), {"w": np.ones(3)}, 1000),
        ("out control 16x3", x, (), {"out": np.empty(120)}, 1000),
        ("cosine control 16x3", x, ("cosine",), {}, 1000),
        ("large control 1500x16", rng.random((1500, 16)), (), {}, 4),
    ]
)
report = {
    "previous_revision": PREVIOUS,
    "platform": platform.platform(),
    "python": sys.version,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "timing": "median of 7 batches; cyclic interleaving; true unpatched SciPy reference; microseconds per call",
    "cases": [],
}
for threads in (1, 2):
    fastscipy.set_num_threads(threads)
    for name, x, args, kwargs, number in cases:
        calls = {label: lambda f=f: f(x, *args, **kwargs) for label, f in functions.items()}
        expected = calls["scipy"]().copy()
        for call in calls.values():
            assert_identical(expected, call())
        labels = list(calls)
        samples = {label: [] for label in labels}
        for repeat in range(7):
            for label in labels[repeat % 3 :] + labels[: repeat % 3]:
                samples[label].append(timeit.timeit(calls[label], number=number) / number * 1e6)
        medians = {label: statistics.median(values) for label, values in samples.items()}
        row = {
            "name": name,
            "threads": threads,
            "identical": True,
            "iterations_per_batch": number,
            "median_us": medians,
            "samples_us": samples,
            "scipy_over_candidate": medians["scipy"] / medians["candidate"],
            "previous_over_candidate": medians["previous"] / medians["candidate"],
        }
        report["cases"].append(row)
        print(
            f"{threads}T {name}: us SciPy/previous/candidate "
            f"{medians['scipy']:.3f}/{medians['previous']:.3f}/{medians['candidate']:.3f}; "
            f"{row['scipy_over_candidate']:.2f}x true SciPy, {row['previous_over_candidate']:.2f}x previous",
            flush=True,
        )
Path(__file__).with_name("comparison.json").write_text(json.dumps(report, indent=2) + "\n")
