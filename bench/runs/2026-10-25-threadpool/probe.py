"""Measure executor reuse separately from numerical kernel performance."""

import argparse
import importlib.util
import json
import platform
import statistics
import time

from fastscipy import _threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="baseline _threads.py; default is the installed candidate")
    args = parser.parse_args()
    module = _threads
    if args.source:
        spec = importlib.util.spec_from_file_location("fastscipy._thread_probe", args.source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    module.set_num_threads(2)
    items = list(range(-8, 0))
    expected = list(range(8, 0, -1))

    def call(reconfigure):
        if reconfigure:
            module.set_num_threads(2)
        assert module.parallel_map(abs, items) == expected

    results = {}
    try:
        for name, reconfigure in (("warm_map", False), ("same_budget_then_map", True)):
            for _ in range(10):
                call(reconfigure)
            batches = []
            for _ in range(7):
                start = time.perf_counter()
                for _ in range(100):
                    call(reconfigure)
                batches.append((time.perf_counter() - start) * 1e6 / 100)
            results[name] = {"median_us": statistics.median(batches), "batch_us": batches}
    finally:
        module.set_num_threads(1)
    print(
        json.dumps({"python": platform.python_version(), "platform": platform.platform(), "cases": results}, indent=2)
    )


if __name__ == "__main__":
    main()
