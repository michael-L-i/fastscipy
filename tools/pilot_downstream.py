"""Bounded downstream application pilots with fresh-process SciPy baselines.

Optional dependencies: scikit-image, scikit-learn, statsmodels. Run with
PYTHONPATH=python python tools/pilot_downstream.py --output /tmp/downstream.
Imports and startup are excluded from timings; no library-wide claim is made.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings


CASES = ("segmentation", "gaussian_process", "seasonal")


def workload(name):
    import numpy as np

    rng = np.random.default_rng(20261025)
    if name == "segmentation":
        from skimage import filters, measure

        y, x = np.mgrid[:512, :512]
        image = rng.normal(0, 0.12, (512, 512))
        for cy, cx in ((100, 120), (290, 280), (380, 100)):
            image += np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 1600)

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

        x = rng.uniform(-3, 3, (128, 2))
        y = np.sin(x[:, 0]) + np.cos(x[:, 1])
        query = rng.uniform(-3, 3, (256, 2))

        def run():
            model = GaussianProcessRegressor(kernel=RBF(0.7), optimizer=None, alpha=1e-5)
            model.fit(x, y)
            mean, std = model.predict(query, return_std=True)
            return {"mean": mean, "std": std, "alpha": model.alpha_, "factor": model.L_}

        inputs = [x, y, query]
    else:
        from statsmodels.tsa.seasonal import seasonal_decompose

        t = np.arange(2400)
        data = 0.002 * t[:, None] + np.sin(2 * np.pi * t[:, None] / 12) + rng.normal(0, 0.2, (2400, 24))

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
    run, inputs = workload(args.case)
    before = [digest(value) for value in inputs]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run()
    warning_records = [(item.category.__name__, str(item.message)) for item in caught]
    warmup_end = time.perf_counter() + 0.1
    while time.perf_counter() < warmup_end:
        run()
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
        "mode": args.mode,
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
        "warnings": warning_records,
        "inputs_unchanged": True,
        "structure": {
            key: {"dtype": str(value.dtype), "shape": list(value.shape), "strides": list(value.strides)}
            for key, value in result.items()
        },
        "counts": {f"{key[0]}:{key[1]}": value for key, value in fastscipy.stats().items()},
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
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1"
    )
    summaries = []
    for name in (args.case,) if args.case else CASES:
        reports = {}
        for mode in ("patched", "baseline") if args.reverse_order else ("baseline", "patched"):
            stem = root / f"{name}-{mode}"
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
        with np.load(root / f"{name}-baseline.npz") as reference, np.load(root / f"{name}-patched.npz") as candidate:
            assert reference.files == candidate.files
            for key in reference.files:
                assert reference[key].tobytes() == candidate[key].tobytes(), (name, key)
        summaries.append(
            {
                "case": name,
                "exact": True,
                "speedup": baseline["median_seconds"] / patched["median_seconds"],
                "baseline_seconds": baseline["median_seconds"],
                "patched_seconds": patched["median_seconds"],
                "counts": patched["counts"],
            }
        )
    summary = {"provenance": provenance, "cases": summaries}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
