"""Paired public-call timings for the conservative two-thread pdist policy.

Pass a saved baseline _spatial.py with --baseline-source. SciPy callables are
captured before patching, and each fast implementation uses the normal counted
wrapper. Output comparisons use bit views rather than byte-string copies.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.distance import cdist, pdist

import fastscipy
from fastscipy import _native, _patch, _spatial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("fastscipy._spatial_baseline", args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    implementations = {
        "scipy": pdist,
        "baseline": _patch.wrap(pdist, baseline._pdist_fast(cdist), "probe_baseline"),
        "candidate": _patch.wrap(pdist, _spatial._pdist_fast(cdist), "probe_candidate"),
    }
    cases = [
        (n, d, m, 2, "contiguous")
        for n in (1500, 4000)
        for d in (1, 2, 3)
        for m in ("euclidean", "sqeuclidean", "cityblock")
    ]
    cases += [
        (1500, 3, m, 2, layout) for m in ("euclidean", "sqeuclidean", "cityblock") for layout in ("float32", "strided")
    ]
    cases += [(3000, 3, m, 4, "contiguous") for m in ("euclidean", "sqeuclidean", "cityblock")]
    cases += [(1500, d, m, 2, "contiguous") for d in (4, 16) for m in ("euclidean", "minkowski")]
    rows = []
    rng = np.random.default_rng(459)
    for n, d, metric, threads, layout in cases:
        fastscipy.set_num_threads(threads)
        x = rng.random((n, d * (2 if layout == "strided" else 1)))
        if layout == "strided":
            x = x[:, ::2]
        elif layout == "float32":
            x = x.astype(np.float32)
        options = {"p": 3.5} if metric == "minkowski" else {}
        expected = pdist(x, metric, **options)
        for call in implementations.values():
            observed = call(x, metric, **options)
            assert np.array_equal(expected.view(np.uint64), observed.view(np.uint64))
            del observed
        del expected
        samples = {name: [] for name in implementations}
        names = list(implementations)
        for repeat in range(args.repeat):
            # Rotate order to avoid always favoring the first/last implementation.
            for name in names[repeat % 3 :] + names[: repeat % 3]:
                start = time.perf_counter()
                result = implementations[name](x, metric, **options)
                samples[name].append((time.perf_counter() - start) * 1000)
                del result
        medians = {name: statistics.median(values) for name, values in samples.items()}
        rows.append(
            dict(
                rows=n,
                dims=d,
                metric=metric,
                threads=threads,
                layout=layout,
                samples_ms=samples,
                median_ms=medians,
                before_after=medians["baseline"] / medians["candidate"],
                scipy_candidate=medians["scipy"] / medians["candidate"],
                byte_identical=True,
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                meta=dict(
                    python=platform.python_version(),
                    numpy=np.__version__,
                    scipy=scipy.__version__,
                    platform=platform.platform(),
                    repeat=args.repeat,
                    baseline_source_sha256=hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
                    candidate_source_sha256=hashlib.sha256(Path(_spatial.__file__).read_bytes()).hexdigest(),
                    native_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
                ),
                cases=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
