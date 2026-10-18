"""Benchmark plain SciPy against fastscipy on 1 thread and on all threads.

    python bench/run.py [area ...] [--budget 0.3] [--only TEXT] [--check]

Each ``bench/bench_<area>.py`` defines ``cases()``, yielding ``(name, call)``
where ``call()`` makes one SciPy call through module attributes (so the patch
applies) and returns its result. A third item says where the fast path must
run: ``True`` (default: on 1 and all threads), ``"threads"`` (only with
threads; a module sets ``THREADED_ONLY = True`` to make that its default) or
``False`` (a control it should decline, which must just not be slower).
Every case checks that the patched result is byte-identical to SciPy's.
Writes ``bench/results/<area>.json`` and prints a markdown table; ``--check`` exits non-zero if a case differs, never took the
fast path, or is slower than SciPy on one thread.
"""
import argparse
import importlib
import json
import math
import os
import pathlib
import sys
import time

import fastscipy
from fastscipy.testing import fast_calls, identical

HERE = pathlib.Path(__file__).parent


def timeit(call, budget):
    """Best of repeated calls within ``budget`` seconds (at least 3 calls)."""
    best, total, n = math.inf, 0.0, 0
    while n < 3 or (total < budget and n < 10_000):
        t = time.perf_counter()
        call()
        dt = time.perf_counter() - t
        best, total, n = min(best, dt), total + dt, n + 1
    return best


def run_case(call, budget, threads, expect=True):
    with fastscipy.disabled():
        ref = call()
        t_scipy = timeit(call, budget)
    row = {"scipy_ms": t_scipy * 1e3, "identical": True, "fast_path": True}
    for label, n in (("1t", 1), ("nt", threads)):
        fastscipy.set_num_threads(n)
        before = fast_calls()
        out = call()
        if expect is True or (expect == "threads" and n > 1):
            row["fast_path"] &= fast_calls() > before
        row["identical"] &= identical(ref, out)
        row[f"{label}_ms"] = timeit(call, budget) * 1e3
        row[f"x{label}"] = t_scipy / (row[f"{label}_ms"] / 1e3)
    return row


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("areas", nargs="*")
    ap.add_argument("--budget", type=float, default=0.3)
    ap.add_argument("--only", default="")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    areas = args.areas or sorted(p.stem[6:] for p in HERE.glob("bench_*.py"))
    threads = fastscipy.get_num_threads()
    print(fastscipy.patch())
    sys.path.insert(0, str(HERE))
    (HERE / "results").mkdir(exist_ok=True)
    bad = []
    print(f"\n| case | SciPy ms | 1 thread ms | {threads} threads ms | x 1T | x {threads}T | identical |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for area in areas:
        rows = {}
        mod = importlib.import_module(f"bench_{area}")
        default = "threads" if getattr(mod, "THREADED_ONLY", False) else True
        for name, call, *expect in mod.cases():
            if args.only not in name:
                continue
            r = rows[name] = run_case(call, args.budget, threads, *(expect or [default]))
            flag = "yes" if r["identical"] else "**NO**"
            if not r["fast_path"]:
                flag += " (no fast path)"
            print(f"| {area}: {name} | {r['scipy_ms']:.3f} | {r['1t_ms']:.3f} | {r['nt_ms']:.3f} "
                  f"| {r['x1t']:.1f} | {r['xnt']:.1f} | {flag} |", flush=True)
            if not (r["identical"] and r["fast_path"] and r["x1t"] >= 0.9):
                bad.append(f"{area}: {name}")
        fastscipy.set_num_threads(threads)
        if rows:
            print(f"| **{area} geomean** | | | | **{geomean([r['x1t'] for r in rows.values()]):.1f}** "
                  f"| **{geomean([r['xnt'] for r in rows.values()]):.1f}** | |")
            meta = {"threads": threads, "cpu": os.cpu_count(), "platform": sys.platform}
            (HERE / "results" / f"{area}.json").write_text(json.dumps({"meta": meta, "cases": rows}, indent=1))
    if bad:
        print("\nNot identical, no fast path, or slower than SciPy on 1 thread:\n  " + "\n  ".join(bad))
    if args.check and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
