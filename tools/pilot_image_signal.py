"""Small reproducible application pilots; run against the installed release wheel."""

# ruff: noqa: E402 -- Set thread budgets before importing NumPy/SciPy.
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["FASTSCIPY_NUM_THREADS"] = "2"

import argparse
import contextlib
import json
import platform
import sys
import time

import numpy as np
import scipy
from scipy import ndimage, signal
from scipy.spatial import distance

import fastscipy
from fastscipy.testing import assert_identical, fast_calls


def inputs():
    rng = np.random.default_rng(821)
    y, x = np.mgrid[:128, :128]
    centers = np.array([[24.0, 32.0], [86.0, 92.0]])
    image = sum(
        amplitude * np.exp(-((y - row) ** 2 + (x - col) ** 2) / (2 * radius**2))
        for (row, col), amplitude, radius in zip(centers, (0.8, 1.2), (8.0, 11.0))
    )
    image += rng.normal(0, 0.02, image.shape)
    pepper = rng.choice(image.size, 40, replace=False)
    image.ravel()[pepper] += rng.choice([-2.0, 2.0], pepper.size)
    t = np.arange(4096) / 256.0
    clean = np.sin(2 * np.pi * 5 * t + np.arange(4)[:, None] / 4)
    sensors = clean + 0.4 * np.sin(2 * np.pi * 70 * t) + rng.normal(0, 0.03, clean.shape)
    return image, centers, sensors, clean


def image_pipeline(image):
    cleaned = ndimage.gaussian_filter(ndimage.median_filter(image, size=3), sigma=1.0)
    mask = ndimage.binary_fill_holes(cleaned > 0.25)
    labels, count = ndimage.label(mask)
    ids = np.arange(1, count + 1)
    centers = np.asarray(ndimage.center_of_mass(cleaned, labels, ids))
    areas = np.asarray(ndimage.sum_labels(mask, labels, ids))
    return cleaned, labels, centers, areas, count


def sensor_pipeline(sensors):
    sos = signal.butter(4, 12.0, fs=256.0, output="sos")
    filtered = signal.sosfiltfilt(sos, sensors, axis=-1)
    frequency, power = signal.welch(filtered, fs=256.0, nperseg=256, axis=-1)
    peaks = [signal.find_peaks(row, height=0.6, prominence=0.8, distance=40)[0] for row in filtered]
    features = np.column_stack((filtered.mean(axis=1), filtered.std(axis=1), frequency[power.argmax(axis=1)]))
    return filtered, frequency, power, peaks, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    image, expected_centers, sensors, clean = inputs()
    input_bytes = (image.tobytes(), sensors.tobytes())
    status = fastscipy.patch(threads=2)
    assert all(value == "on" for value in status.values()), status
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "fastscipy": fastscipy.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": 2,
        "blas_threads": 1,
        "repeat": args.repeat,
        "input_array_bytes": image.nbytes + sensors.nbytes + clean.nbytes,
        "application_pilots_only": True,
        "pipelines": {},
    }
    for name, call in (
        ("image_cleanup_segmentation", lambda: image_pipeline(image)),
        ("sensor_filter_peak_features", lambda: sensor_pipeline(sensors)),
    ):
        runs = {}
        reference = None
        for patched in (False, True):
            timings = []
            for _ in range(args.repeat):
                before = fast_calls()
                with contextlib.nullcontext() if patched else fastscipy.disabled():
                    started = time.perf_counter()
                    output = call()
                    timings.append(time.perf_counter() - started)
                if patched:
                    assert fast_calls() > before, f"No accelerated calls in {name}"
                    assert_identical(reference, output)
                else:
                    if reference is not None:
                        assert_identical(reference, output)
                    reference = output
                if name.startswith("image"):
                    assert output[4] == 2
                    assert np.max(abs(output[2] - expected_centers)) < 0.5
                    assert np.all(output[3] > 100)
                else:
                    assert np.sqrt(np.mean((output[0][:, 100:-100] - clean[:, 100:-100]) ** 2)) < 0.05
                    assert np.all(output[4][:, 2] == 5.0)
                    assert all(78 <= len(peaks) <= 82 for peaks in output[3])
            runs["patched" if patched else "plain"] = timings
        report["pipelines"][name] = {"byte_identical": True, "independent_properties": "passed", "seconds": runs}
    # Custom feature similarity deliberately takes SciPy's callback fallback.
    features = output[4]
    callbacks = []

    def custom_metric(a, b):
        callbacks.append(1)
        return np.sum(abs(a - b))

    with fastscipy.disabled():
        expected = distance.cdist(features, features, metric=custom_metric)
    callbacks.clear()
    before = fast_calls()
    actual = distance.cdist(features, features, metric=custom_metric)
    assert fast_calls() == before and len(callbacks) == 16
    assert_identical(expected, actual)
    assert np.all(actual >= 0) and np.all(np.diag(actual) == 0)
    assert_identical(actual, actual.T.copy())
    assert input_bytes == (image.tobytes(), sensors.tobytes())
    report["fallback_control"] = {"custom_metric_callbacks": len(callbacks), "byte_identical": True, "fast_calls": 0}
    report["inputs_unchanged"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
