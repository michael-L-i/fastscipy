"""Compare truly unpatched SciPy, previous components, and the FFT candidate.

Run under tools/platform_suite.run with explicit process/thread budgets, e.g.
python bench/audit_fft.py --output artifacts/fft-comparison.json
The existing native extension is sufficient; no new compiled code is used.
"""

import argparse
import json
import platform
import timeit
from pathlib import Path

import numpy as np
import scipy

import fastscipy
from bench_fft import cases
from fastscipy.testing import assert_identical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    workloads = list(cases())
    rows = {name: {} for name, *_ in workloads}
    reference = {}
    previous = tuple(component for component in fastscipy.COMPONENTS if component != "fft")
    # Compare adjacent phases per input, then reverse their order. Whole-suite
    # phases can hide small dispatch costs behind CPU frequency/thermal changes.
    for name, call, *_ in workloads:
        for mode in ("scipy", "previous", "candidate", "candidate", "previous", "scipy"):
            fastscipy.unpatch()
            if mode != "scipy":
                status = fastscipy.patch(previous if mode == "previous" else None)
                assert all(value == "on" for value in status.values()), status
            result = call()
            if name not in reference:
                reference[name] = result
            assert_identical(reference[name], result)
            seconds = min(timeit.repeat(call, number=100, repeat=3)) / 100
            rows[name].setdefault(f"{mode}_us", []).append(seconds * 1e6)
    for name, row in rows.items():
        for mode in ("scipy", "previous", "candidate"):
            row[f"{mode}_best_us"] = min(row[f"{mode}_us"])
        row["scipy_speedup"] = row["scipy_best_us"] / row["candidate_best_us"]
        row["previous_speedup"] = row["previous_best_us"] / row["candidate_best_us"]
        row["byte_identical"] = True
        print(
            f"{name}: {row['scipy_best_us']:.3f} / {row['previous_best_us']:.3f} / "
            f"{row['candidate_best_us']:.3f} us; {row['scipy_speedup']:.2f}x / {row['previous_speedup']:.2f}x"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "threads": fastscipy.get_num_threads(),
                "previous": "base f60dca0 Python components, with fft absent; unchanged published 0.1.0a4 native",
                "method": "best of 3 x 100 calls per input/phase; scipy/previous/candidate/candidate/previous/scipy",
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
