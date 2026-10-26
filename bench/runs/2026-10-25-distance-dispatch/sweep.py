"""Bounded, paired distance-dispatch measurements against untouched SciPy."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.distance import cdist, pdist

import fastscipy
from fastscipy import _spatial


def sample(call, repeat):
    call()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
        del result
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    rows = []
    rng = np.random.default_rng(456)
    for n in (500, 1500, 3000):
        for d in (1, 3, 8, 16, 32):
            x = rng.random((n, d))
            for metric in ("euclidean", "cityblock", "cosine", "minkowski"):
                options = {"p": 3.5} if metric == "minkowski" else {}
                for function, original, factory, inputs in (
                    ("pdist", pdist, _spatial._pdist_fast, (x,)),
                    ("cdist", cdist, _spatial._cdist_fast, (x, x[: n // 2])),
                ):
                    ref = lambda: original(*inputs, metric, **options)
                    reference = sample(ref, args.repeat)
                    expected = ref()
                    for threads in (2, 4):
                        fastscipy.set_num_threads(threads)
                        fast = factory(cdist)
                        hit = fast(*inputs, metric, **options) is not NotImplemented

                        def call():
                            result = fast(*inputs, metric, **options)
                            return original(*inputs, metric, **options) if result is NotImplemented else result

                        observed = call()
                        assert np.array_equal(expected.view(np.uint64), observed.view(np.uint64))
                        elapsed = sample(call, args.repeat)
                        rows.append(
                            dict(
                                function=function,
                                rows=n,
                                dims=d,
                                metric=metric,
                                threads=threads,
                                fast_path=hit,
                                scipy_ms=reference * 1000,
                                fastscipy_ms=elapsed * 1000,
                                ratio=reference / elapsed,
                            )
                        )
                    del expected, observed
    args.output.write_text(
        json.dumps(
            dict(
                meta=dict(
                    python=platform.python_version(),
                    numpy=np.__version__,
                    scipy=scipy.__version__,
                    platform=platform.platform(),
                    repeat=args.repeat,
                ),
                cases=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
