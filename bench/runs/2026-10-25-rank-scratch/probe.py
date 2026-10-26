"""Compare source builds by running this probe once with each package on PYTHONPATH.

Use FASTSCIPY_NUM_THREADS=2 and limit BLAS/OpenMP to one thread. The output
contains seven batch samples per case; compare medians, not single best calls.
"""

import argparse
import json
import platform
import statistics
import timeit

import numpy as np
import scipy
import scipy.stats as st
import fastscipy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    rng = np.random.default_rng(752)
    cases = {}
    for rows, n in [(1, 300), (1, 4096), (1, 200_000), (3000, 300), (512, 2048), (256, 4096), (64, 16384)]:
        for kind in ("random", "ties"):
            x = rng.normal(size=(rows, n)) if kind == "random" else rng.integers(0, 16, size=(rows, n)).astype(float)
            cases[f"{rows}x{n} {kind}"] = x
    skewed = np.zeros((512, 2048))
    skewed[:64] = rng.normal(size=(64, 2048))
    cases["512x2048 skewed"] = skewed
    fastscipy.patch(["stats"])
    results = {}
    for threads in (1, 2):
        fastscipy.set_num_threads(threads)
        for name, x in cases.items():
            if args.only not in name:
                continue
            fn = lambda: st.rankdata(x, axis=-1)
            fn()
            timer = timeit.Timer(fn)
            number = max(1, min(10_000, int(0.05 / timer.timeit(1))))
            samples = [v / number for v in timer.repeat(7, number)]
            results[f"{threads}t {name}"] = {
                "calls_per_batch": number,
                "median_ms": 1000 * statistics.median(samples),
                "samples_ms": [v * 1000 for v in samples],
            }
    print(json.dumps(results, indent=2))
    with open(args.output, "w") as f:
        json.dump(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "cases": results,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
