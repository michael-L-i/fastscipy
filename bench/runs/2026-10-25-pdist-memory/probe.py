"""Measure one pdist implementation in a fresh process.

Use --source for a saved baseline _spatial.py; omit it for the current source.
Peak RSS covers imports, warmup and timing, before allocating the reference.
macOS/Linux ru_maxrss units differ. Inputs and outputs stay below 512 MiB for
our recorded 1500/3000/4000-row cases; larger user-specified cases may not.
"""

import argparse
import importlib.util
import json
import resource
import statistics
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist, pdist

import fastscipy
from fastscipy import _spatial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--dims", type=int, default=3)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--repeat", type=int, default=15)
    args = parser.parse_args()
    module = _spatial
    if args.source:
        spec = importlib.util.spec_from_file_location("fastscipy._spatial_baseline", args.source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    fastscipy.set_num_threads(2)
    call = module._pdist_fast(cdist)
    x = np.random.default_rng(845).random((args.rows, args.dims))
    call(x, args.metric)
    timings = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = call(x, args.metric)
        timings.append(time.perf_counter() - start)
        del result
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
    result = call(x, args.metric)
    reference = pdist(x, args.metric)
    # Compare the bit patterns without cloning either array into a bytes object.
    identical = np.array_equal(result.view(np.uint64), reference.view(np.uint64))
    assert identical
    print(
        json.dumps(
            {
                "source": args.source or "candidate",
                "rows": args.rows,
                "dims": args.dims,
                "metric": args.metric,
                "threads": 2,
                "repeat": args.repeat,
                "median_ms": statistics.median(timings) * 1000,
                "min_ms": min(timings) * 1000,
                "peak_mib": peak_mib,
                "byte_identical": bool(identical),
            }
        )
    )


if __name__ == "__main__":
    main()
