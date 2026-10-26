"""Bounded downstream application pilots with fresh-process SciPy baselines.

Optional dependencies: scikit-image, scikit-learn, statsmodels. Run with
PYTHONPATH=python python tools/pilot_downstream.py --output /tmp/downstream.
Imports and startup are excluded from timings; no library-wide claim is made.
Use --size small for overhead-sensitive inputs, with a separate output directory
for each size/order. Counts describe one execution, not timing batches; they
cover instrumented dispatch only, not all SciPy or downstream work. Empty
counts do not establish that no acceleration occurred (some replacements do
not use dispatch counters). Negative seconds_saved means an observed slowdown.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import warnings


CASES = ("segmentation", "gaussian_process", "seasonal")


def workload(name, size="standard"):
    import numpy as np

    rng = np.random.default_rng(20261025)
    if name == "segmentation":
        from skimage import filters, measure

        side = 128 if size == "small" else 512
        y, x = np.mgrid[:side, :side]
        image = rng.normal(0, 0.12, (side, side))
        for cy, cx in ((100, 120), (290, 280), (380, 100)):
            cy, cx = cy * side / 512, cx * side / 512
            image += np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (1600 * (side / 512) ** 2))

        def run():
            clean = filters.median(image, footprint=np.ones((7, 7), dtype=bool))
            smooth = filters.gaussian(clean, sigma=1.2, preserve_range=True)
            labels = measure.label(smooth > filters.threshold_otsu(smooth))
            regions = measure.regionprops_table(
                labels, smooth, properties=("label", "area", "centroid", "intensity_mean")
            )
            return {"clean": clean, "smooth": smooth, "labels": labels, **regions}

        inputs = [image]
    elif name == "gaussian_process":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF

        n, nq = (32, 64) if size == "small" else (128, 256)
        x = rng.uniform(-3, 3, (n, 2))
        y = np.sin(x[:, 0]) + np.cos(x[:, 1])
        query = rng.uniform(-3, 3, (nq, 2))

        def run():
            model = GaussianProcessRegressor(kernel=RBF(0.7), optimizer=None, alpha=1e-5)
            model.fit(x, y)
            mean, std = model.predict(query, return_std=True)
            return {"mean": mean, "std": std, "alpha": model.alpha_, "factor": model.L_}

        inputs = [x, y, query]
    else:
        from statsmodels.tsa.seasonal import seasonal_decompose

        n, channels = (240, 4) if size == "small" else (2400, 24)
        t = np.arange(n)
        data = 0.002 * t[:, None] + np.sin(2 * np.pi * t[:, None] / 12) + rng.normal(0, 0.2, (n, channels))

        def run():
            result = seasonal_decompose(data, period=12, extrapolate_trend="period")
            return {key: np.asarray(getattr(result, key)) for key in ("observed", "trend", "seasonal", "resid")}

        inputs = [data]
    return run, inputs


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def child(args):
    import numpy as np
    import fastscipy

    if args.mode == "patched":
        fastscipy.patch(threads=2)
    run, inputs = workload(args.case, args.size)
    before = [digest(value) for value in inputs]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run()
    warning_records = [(item.category.__name__, str(item.message)) for item in caught]
    warmup_end = time.perf_counter() + 0.1
    while time.perf_counter() < warmup_end:
        run()
    counts_before = fastscipy.stats()
    run()
    counts = {key: count - counts_before.get(key, 0) for key, count in fastscipy.stats().items()}
    counts = {key: count for key, count in counts.items() if count}
    samples = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    assert before == [digest(value) for value in inputs], "input mutation"
    root = Path(args.output)
    np.savez(root.with_suffix(".npz"), **result)
    report = {
        "case": args.case,
        "size": args.size,
        "mode": args.mode,
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
        "warnings": warning_records,
        "inputs_unchanged": True,
        "structure": {
            key: {"dtype": str(value.dtype), "shape": list(value.shape), "strides": list(value.strides)}
            for key, value in result.items()
        },
        "counts": {f"{key[0]}:{key[1]}": value for key, value in counts.items()},
        "input_array_bytes": sum(value.nbytes for value in inputs),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-image", "scikit-learn", "statsmodels")
        },
    }
    root.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--size", choices=("small", "standard"), default="standard")
    parser.add_argument("--reverse-order", action="store_true", help="Run patched child first")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--mode", choices=("baseline", "patched"))
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.mode:
        if not args.case:
            parser.error("--mode requires --case")
        child(args)
        return
    import numpy as np
    from check_native_build import inspect_build

    provenance = inspect_build(Path(__file__).resolve().parents[1])
    if provenance["status"] != "match":
        raise RuntimeError(f"Native provenance check failed: {provenance}")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        FASTSCIPY_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    summaries = []
    for name in (args.case,) if args.case else CASES:
        reports = {}
        for mode in ("patched", "baseline") if args.reverse_order else ("baseline", "patched"):
            stem = root / f"{name}-{args.size}-{mode}"
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--case",
                    name,
                    "--mode",
                    mode,
                    "--repeat",
                    str(args.repeat),
                    "--size",
                    args.size,
                    "--output",
                    str(stem),
                ],
                env=environment,
                check=True,
                timeout=120,
            )
            reports[mode] = json.loads(stem.with_suffix(".json").read_text())
        baseline, patched = reports["baseline"], reports["patched"]
        assert baseline["counts"] == {}, "Baseline must have no patched calls"
        assert baseline["warnings"] == patched["warnings"]
        assert baseline["structure"] == patched["structure"]
        with (
            np.load(root / f"{name}-{args.size}-baseline.npz") as reference,
            np.load(root / f"{name}-{args.size}-patched.npz") as candidate,
        ):
            assert reference.files == candidate.files
            for key in reference.files:
                assert reference[key].tobytes() == candidate[key].tobytes(), (name, key)
        summaries.append(
            {
                "case": name,
                "size": args.size,
                "exact": True,
                "speedup": baseline["median_seconds"] / patched["median_seconds"],
                "baseline_seconds": baseline["median_seconds"],
                "patched_seconds": patched["median_seconds"],
                "seconds_saved": baseline["median_seconds"] - patched["median_seconds"],
                "input_array_bytes": patched["input_array_bytes"],
                "counts": patched["counts"],
            }
        )
    summary = {
        "provenance": provenance,
        "platform": platform.platform(),
        "python": sys.version,
        "versions": patched["versions"],
        "timing_scope": "workflow only; excludes imports, input construction, validation and warmup",
        "threads": {
            key: environment[key]
            for key in (
                "FASTSCIPY_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "count_scope": "one untimed execution after warmup; instrumented dispatch only; not time coverage",
        "cases": summaries,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
