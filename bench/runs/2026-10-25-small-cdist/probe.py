"""Compare actual SciPy, previous fastscipy and candidate on identical inputs."""

import json
import pathlib
import platform
import statistics
import subprocess
import time
import tempfile
import types

import numpy as np
import scipy
import scipy.spatial.distance as dist

import fastscipy
from fastscipy import _patch
from fastscipy.testing import assert_identical

BASELINE = "a5961a5"


def measure(call, batch=1000):
    start = time.perf_counter_ns()
    for _ in range(batch):
        call()
    return (time.perf_counter_ns() - start) / batch


def main():
    source = subprocess.check_output(["git", "show", f"{BASELINE}:python/fastscipy/_spatial.py"], text=True)
    baseline = types.ModuleType("fastscipy._spatial_before_cdist")
    exec(compile(source, f"{BASELINE}:_spatial.py", "exec"), baseline.__dict__)
    original = dist.cdist
    previous = _patch.wrap(original, baseline._cdist_fast(original), "before.cdist")
    fastscipy.patch(["spatial"])
    # Reconstruct the rejected experiment outside the checkout; never mutate
    # production sources or import paths to reproduce its measurements.
    patch = pathlib.Path(__file__).with_name("initial-prototype.patch").resolve()
    with tempfile.TemporaryDirectory(prefix="fastscipy-cdist-probe-") as tmp:
        target = pathlib.Path(tmp) / "python/fastscipy/_spatial.py"
        target.parent.mkdir(parents=True)
        target.write_text(source)
        subprocess.run(["git", "apply", str(patch)], cwd=tmp, check=True)
        candidate_module = types.ModuleType("fastscipy._spatial_cdist_experiment")
        exec(compile(target.read_text(), str(patch), "exec"), candidate_module.__dict__)
    candidate = _patch.wrap(original, candidate_module._cdist_fast(original), "candidate.cdist")
    rng = np.random.default_rng(929)
    fixtures = []
    for rows, dims in ((4, 3), (16, 3), (32, 16), (100, 16)):
        a, b = rng.normal(size=(rows, dims)), rng.normal(size=(rows + 1, dims))
        for metric in ("euclidean", "cityblock", "sqeuclidean", "minkowski", "hamming"):
            fixtures.append((f"{metric} {rows}x{rows + 1}x{dims}", a, b, (metric,), {}))
    a, b = rng.random((4, 3)), rng.random((5, 3))
    fixtures.extend(
        [
            ("default 4x5x3", a, b, (), {}),
            ("keyword metric 4x5x3", a, b, (), {"metric": "euclidean"}),
            ("float32 control", a.astype(np.float32), b.astype(np.float32), (), {}),
            ("list control", a.tolist(), b.tolist(), (), {}),
            ("alias control", a, b, ("eu",), {}),
            ("weighted control", a, b, (), {"w": np.ones(3)}),
            ("out control", a, b, (), {"out": np.empty((4, 5))}),
            ("cosine control", a, b, ("cosine",), {}),
        ]
    )
    rows = []
    for threads in (1, 2):
        fastscipy.set_num_threads(threads)
        for name, a, b, args, kw in fixtures:
            calls = {
                label: lambda fn=fn: fn(a, b, *args, **kw)
                for label, fn in (("scipy", original), ("before", previous), ("candidate", candidate))
            }
            reference = calls["scipy"]()
            for call in calls.values():
                assert_identical(reference, call())
                for _ in range(50):
                    call()
            samples = {label: [] for label in calls}
            for repeat in range(7):
                labels = list(calls)
                if repeat % 2:
                    labels.reverse()
                for label in labels:
                    samples[label].append(measure(calls[label]))
            medians = {label: statistics.median(values) for label, values in samples.items()}
            rows.append({"case": name, "threads": threads, "median_ns": medians, "samples_ns": samples})
    result = {
        "baseline": BASELINE,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "batch_calls": 1000,
        "repeats": 7,
        "cases": rows,
    }
    path = pathlib.Path(__file__).with_name("results.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        times = row["median_ns"]
        print(row["threads"], row["case"], *(f"{times[k]:.1f}" for k in ("scipy", "before", "candidate")))


if __name__ == "__main__":
    main()
