"""Benchmark plain SciPy against fastscipy on 1 thread and on all threads.

    python bench/run.py [area ...] [--budget 0.3] [--only TEXT] [--check]

Each ``bench/bench_<area>.py`` defines ``cases()``, yielding ``(name, call)``
where ``call()`` makes one SciPy call through module attributes (so the patch
applies) and returns its result. A third item says where the fast path must
run: ``True`` (default: on 1 and all threads), ``"threads"`` (only with
threads; a module sets ``THREADED_ONLY = True`` to make that its default) or
``False`` (a control it should decline, which must just not be slower).
Every case checks byte identity. An optional fourth item supplies independent
numerical validation for proven Intel Accelerate reference variation; such
cases remain marked nonidentical and are excluded from headline averages.
Writes ``bench/results/<area>.json`` and prints a markdown table; ``--check``
exits non-zero for unexplained differences, missing fast paths, or a one-thread
regression. Numerical validation never excuses a deterministic mismatch.
"""

import argparse
import contextlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import sys
import time

import fastscipy
from fastscipy.testing import fast_calls, identical, reference_variance

HERE = pathlib.Path(__file__).parent


@contextlib.contextmanager
def unpatched():
    """Restore actual SciPy bindings, then reinstall the active components.

    This is a serial benchmark operation, not a thread-local runtime API.
    Disabling dispatch alone would still time wrappers and their counters.
    """
    active = [name for name, state in fastscipy.status().items() if state == "on"]
    fastscipy.unpatch()
    try:
        with fastscipy.disabled():
            yield
    finally:
        if active:
            fastscipy.patch(active)


def load_cases(area, label):
    """Load independent module state, including module-level seeded RNGs."""
    spec = importlib.util.spec_from_file_location(f"bench_{area}_{label}", HERE / f"bench_{area}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return iter(module.cases()), "threads" if getattr(module, "THREADED_ONLY", False) else True


def paired_cases(area):
    """Construct each side's fixtures in its own patch state, one pair at a time.

    Captured functions and objects such as cKDTree retain their implementation
    after unpatch(), so the reference needs its own independently built cases.
    """
    with unpatched():
        references, reference_default = load_cases(area, "reference")
    candidates, default = load_cases(area, "candidate")
    if reference_default != default:
        raise ValueError(f"{area}: inconsistent default fast-path expectation")
    sentinel = object()
    while True:
        with unpatched():
            reference = next(references, sentinel)
        candidate = next(candidates, sentinel)
        if reference is sentinel and candidate is sentinel:
            return
        if reference is sentinel or candidate is sentinel or reference[0] != candidate[0]:
            raise ValueError(f"{area}: reference and candidate cases do not match")
        name, call, *options = candidate
        expected = options[0] if options else default
        reference_expected = reference[2] if len(reference) > 2 else reference_default
        if expected != reference_expected or len(reference) != len(candidate):
            raise ValueError(f"{area}: inconsistent options for {name}")
        yield name, call, reference[1], expected, options[1] if len(options) > 1 else None


def timeit(call, budget):
    """Median per-call time from at least five calibrated batches.

    Batch tiny calls to amortize timer overhead. Calibration and warmup are
    excluded; the budget is a sampling target, not a hard timeout.
    """
    batch = 1
    target = min(0.001, max(0.00001, budget / 10))
    while True:
        start = time.perf_counter()
        for _ in range(batch):
            call()
        elapsed = time.perf_counter() - start
        if elapsed >= target or batch >= 10_000:
            break
        batch = min(10_000, batch * 10)
    samples, total = [], 0.0
    while len(samples) < 5 or (total < budget and len(samples) < 100):
        start = time.perf_counter()
        for _ in range(batch):
            call()
        elapsed = time.perf_counter() - start
        samples.append(elapsed / batch)
        total += elapsed
    return statistics.median(samples)


def run_case(call, budget, threads, expect=True, reference_validation=None, *, reference_call):
    with unpatched():
        before_reference = fastscipy.stats()
        ref = reference_call()
        t_scipy = timeit(reference_call, budget)
        if fastscipy.stats() != before_reference:
            raise RuntimeError("reference reached a fastscipy wrapper; construct its fixture while unpatched")
    row = {"scipy_ms": t_scipy * 1e3, "identical": True, "fast_path": True}
    exceptions = {}
    unexplained = False
    for label, n in (("1t", 1), ("nt", threads)):
        fastscipy.set_num_threads(n)
        before = fast_calls()
        out = call()
        if expect is True or (expect == "threads" and n > 1):
            row["fast_path"] &= fast_calls() > before
        same = identical(ref, out)
        row["identical"] &= same
        if not same:
            with unpatched():
                exception = reference_variance(reference_call, ref, out, reference_validation)
            if exception is None:
                unexplained = True
            else:
                exceptions[label] = exception
        row[f"{label}_ms"] = timeit(call, budget) * 1e3
        row[f"x{label}"] = t_scipy / (row[f"{label}_ms"] / 1e3)
    if exceptions:
        row["reference_variance"] = exceptions
    row["validation"] = "byte_identical" if row["identical"] else "mismatch" if unexplained else "reference_variance"
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
    ap.add_argument("--output-dir", type=pathlib.Path, default=HERE / "results")
    args = ap.parse_args()
    if not math.isfinite(args.budget) or args.budget < 0:
        ap.error("--budget must be finite and nonnegative")
    areas = args.areas or sorted(p.stem[6:] for p in HERE.glob("bench_*.py"))
    threads = fastscipy.get_num_threads()
    print(fastscipy.patch())
    sys.path.insert(0, str(HERE))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bad = []
    print(f"\n| case | SciPy ms | 1 thread ms | {threads} threads ms | x 1T | x {threads}T | identical |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for area in areas:
        rows = {}
        for name, call, reference_call, expected, validation in paired_cases(area):
            if args.only not in name:
                continue
            r = rows[name] = run_case(call, args.budget, threads, expected, validation, reference_call=reference_call)
            r["expected_fast_path"] = expected
            flag = "yes" if r["identical"] else "**NO**"
            if r["validation"] == "reference_variance":
                flag = "reference varies; numerical checks passed"
            if not r["fast_path"]:
                flag += " (no fast path)"
            print(
                f"| {area}: {name} | {r['scipy_ms']:.3f} | {r['1t_ms']:.3f} | {r['nt_ms']:.3f} "
                f"| {r['x1t']:.1f} | {r['xnt']:.1f} | {flag} |",
                flush=True,
            )
            if not (r["validation"] != "mismatch" and r["fast_path"] and r["x1t"] >= 0.9):
                bad.append(f"{area}: {name}")
        fastscipy.set_num_threads(threads)
        if rows:
            stable = [r for r in rows.values() if r["identical"] and r["expected_fast_path"] is not False]
            print(
                f"| **{area} geomean (byte-identical non-controls)** | | | | "
                f"**{geomean([r['x1t'] for r in stable]):.1f}** "
                f"| **{geomean([r['xnt'] for r in stable]):.1f}** | |"
            )
            meta = {
                "threads": threads,
                "cpu": os.cpu_count(),
                "platform": sys.platform,
                "reference": "unpatched_scipy",
                "timing": "median_calibrated_batches",
                "budget_seconds": args.budget,
                "headline_cases": len(stable),
            }
            (args.output_dir / f"{area}.json").write_text(json.dumps({"meta": meta, "cases": rows}, indent=1))
    if bad:
        print("\nNot identical, no fast path, or slower than SciPy on 1 thread:\n  " + "\n  ".join(bad))
    if args.check and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
