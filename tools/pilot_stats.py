"""Small sensor-analysis pilot for an installed fastscipy wheel; no source overlay."""
# ruff: noqa: E402 -- Set native thread budgets before importing NumPy/SciPy.

import hashlib
import json
import math
import os
import platform
import statistics
import time
import tracemalloc
import warnings

for key in ("FASTSCIPY_ANY_PLATFORM", "FASTSCIPY_ANY_VERSION"):
    os.environ.pop(key, None)
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["FASTSCIPY_NUM_THREADS"] = "2"

import numpy as np
import scipy
from scipy import stats
import fastscipy
from fastscipy.testing import check, fast_calls, identical


def gold_fit(x, y):
    """Independent compensated scalar reductions, avoiding BLAS alignment."""
    n = x.size
    xm, ym = math.fsum(map(float, x)) / n, math.fsum(map(float, y)) / n
    xx, yy = x - xm, y - ym
    vx = math.fsum(float(v) * float(v) for v in xx)
    vy = math.fsum(float(v) * float(v) for v in yy)
    xy = math.fsum(float(a) * float(b) for a, b in zip(xx, yy))
    slope = xy / vx
    return dict(r=xy / math.sqrt(vx * vy), slope=slope, intercept=ym - slope * xm, xm=xm, ym=ym)


def validator(kind, gold):
    def validate(a, b):
        assert type(a) is type(b)
        assert all(
            type(aa) is type(bb)
            and np.shape(aa) == np.shape(bb) == ()
            and np.asarray(aa).dtype == np.asarray(bb).dtype == np.dtype("float64")
            for aa, bb in zip(a, b)
        )
        numeric_metadata = {"correlation"} if kind == "pearsonr" else {"intercept_stderr"}
        assert vars(a).keys() == vars(b).keys()
        assert all(identical(v, vars(b)[k]) for k, v in vars(a).items() if k not in numeric_metadata)
        eps = 64 * np.finfo(np.float64).eps
        intercept_bound = eps * (1 + abs(gold["ym"]) + abs(gold["slope"] * gold["xm"]))
        floors = (
            [eps, eps]
            if kind == "pearsonr"
            else [eps * (1 + abs(gold["slope"])), intercept_bound, eps, eps, eps * (1 + abs(gold["slope"]))]
        )
        for aa, bb, floor in zip(a, b, floors):
            np.testing.assert_allclose(aa, bb, rtol=eps, atol=floor)
        if kind == "linregress":
            np.testing.assert_allclose(a.intercept_stderr, b.intercept_stderr, rtol=eps, atol=intercept_bound)
        for value in (a, b):
            r = value.statistic if kind == "pearsonr" else value.rvalue
            assert np.isfinite(r) and abs(r - gold["r"]) <= eps
            assert 0 <= value.pvalue < 1e-15
            if kind == "pearsonr":
                assert identical(value.correlation, value.statistic)
            else:
                assert abs(value.slope - gold["slope"]) <= eps * (1 + abs(gold["slope"]))
                bound = eps * (1 + abs(gold["ym"]) + abs(value.slope * gold["xm"]))
                assert abs(value.intercept - gold["intercept"]) <= bound
                assert value.stderr > 0 and value.intercept_stderr > 0
        return True

    return validate


def main():
    assert not any(k in os.environ for k in ("FASTSCIPY_ANY_PLATFORM", "FASTSCIPY_ANY_VERSION"))
    activation = fastscipy.patch(threads=2)
    assert fastscipy.status().get("stats") == "on", activation
    rng = np.random.default_rng(20261020)
    n, channels = 8192, 12
    t = np.arange(n) / 200.0
    reference = 22 + 0.001 * t + 0.2 * np.sin(2 * np.pi * t / 37) + rng.normal(0, 0.03, n)
    gains = np.linspace(0.8, 1.2, channels)
    offsets = np.linspace(-0.15, 0.15, channels)
    sensors = gains[:, None] * reference + offsets[:, None] + rng.normal(0, 0.01, (channels, n))
    for a in (reference, sensors):
        a.flags.writeable = False
    before_hash = hashlib.sha256(reference.tobytes() + sensors.tobytes()).hexdigest()
    gold = [gold_fit(reference, sensor) for sensor in sensors]
    q = np.linspace(0.001, 0.999, 2048)

    def workflow():
        return [
            (
                stats.sem(sensor),
                stats.skew(sensor),
                stats.kurtosis(sensor),
                stats.pearsonr(reference, sensor),
                stats.linregress(reference, sensor),
                stats.norm.ppf(q, loc=float(sensor.mean()), scale=float(sensor.std())),
            )
            for sensor in sensors
        ]

    calls_before = fast_calls()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for index, sensor in enumerate(sensors):
            for function in (stats.sem, stats.skew, stats.kurtosis):
                check(lambda function=function, sensor=sensor: function(sensor))
            check(
                lambda sensor=sensor: stats.pearsonr(reference, sensor),
                reference_validation=validator("pearsonr", gold[index]),
            )
            check(
                lambda sensor=sensor: stats.linregress(reference, sensor),
                reference_validation=validator("linregress", gold[index]),
            )
            check(lambda sensor=sensor: stats.norm.ppf(q, loc=float(sensor.mean()), scale=float(sensor.std())))

        contaminated = sensors[0].copy()
        contaminated[::31] = np.nan
        fallback_before = fast_calls()
        clean_sem = check(lambda: stats.sem(contaminated, nan_policy="omit"), fast=False)
        assert fast_calls() == fallback_before and np.isfinite(clean_sem)

        timings = {"scipy_seconds": [], "patched_seconds": []}
        for _ in range(3):
            with fastscipy.disabled():
                started = time.perf_counter()
                workflow()
                timings["scipy_seconds"].append(time.perf_counter() - started)
            started = time.perf_counter()
            outputs = workflow()
            timings["patched_seconds"].append(time.perf_counter() - started)
        for index, (sem, skew, kurtosis, corr, fit, quantiles) in enumerate(outputs):
            validator("pearsonr", gold[index])(corr, corr)
            validator("linregress", gold[index])(fit, fit)
            assert 0 < sem < 0.1 and abs(skew) < 1 and -2 < kurtosis < 5
            assert corr.statistic > 0.97 and abs(fit.slope - gains[index]) < 0.02
            assert np.isfinite(quantiles).all() and np.all(np.diff(quantiles) > 0)
            assert quantiles[0] < sensors[index].mean() < quantiles[-1]

        tracemalloc.start()
        for _ in range(16):
            workflow()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    after_hash = hashlib.sha256(reference.tobytes() + sensors.tobytes()).hexdigest()
    assert before_hash == after_hash
    print(
        json.dumps(
            dict(
                purpose="installed-wheel sensor-analysis pilot, not all-platform release certification",
                platform=platform.platform(),
                numpy=np.__version__,
                scipy=scipy.__version__,
                fastscipy=fastscipy.__version__,
                package_path=fastscipy.__file__,
                activation=activation,
                input_bytes=reference.nbytes + sensors.nbytes,
                channels=channels,
                samples_per_channel=n,
                default_gate_without_overrides=True,
                thread_budget=2,
                blas_threads=1,
                fast_calls=fast_calls() - calls_before,
                nan_omit_fallback_verified=True,
                input_unchanged=True,
                independent_ols_and_signal_properties_verified=True,
                sustained_repetitions=16,
                traced_python_bytes=current,
                traced_python_peak_bytes=peak,
                timing=timings,
                median_workflow_speedup=statistics.median(timings["scipy_seconds"])
                / statistics.median(timings["patched_seconds"]),
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
