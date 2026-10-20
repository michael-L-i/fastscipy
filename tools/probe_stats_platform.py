"""Bounded diagnostics for Intel Mac stats reductions; never changes the wheel.

Run from the checked-out project with its test dependencies installed:
    python tools/probe_stats_platform.py --output artifacts/stats-diagnostics.json
"""

# Set numerical thread limits before importing NumPy or SciPy.
# ruff: noqa: E402

import argparse
import base64
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["FASTSCIPY_NUM_THREADS"] = "1"

import numpy as np
import scipy
import scipy.stats as stats

import fastscipy
from fastscipy.testing import fast_calls, identical


def scalar(value):
    a = np.asarray(value)
    return {"dtype": a.dtype.str, "shape": list(a.shape), "hex": a.tobytes().hex()}


def result(value):
    fields = list(value._fields)
    if hasattr(value, "intercept_stderr"):
        fields.append("intercept_stderr")
    return {name: scalar(getattr(value, name)) for name in fields}


def invoke(function, args, kwargs, *, plain):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fastscipy.disabled() if plain else contextlib.nullcontext():
            try:
                value = function(*args, **kwargs)
                record = {"result": result(value)}
            except Exception as error:
                value = None
                record = {"error": {"type": type(error).__name__, "message": str(error)}}
    record["warnings"] = [(w.category.__name__, str(w.message)) for w in caught]
    return value, record


def array_record(a):
    raw = a.tobytes()
    return {
        "dtype": a.dtype.str,
        "shape": list(a.shape),
        "strides": list(a.strides),
        "address_mod64": a.ctypes.data % 64,
        "writeable": a.flags.writeable,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def aligned_copy(values, offset):
    storage = np.empty(len(values) + 8, dtype=np.float64)
    start = ((offset - storage.ctypes.data) % 64) // 8
    view = storage[start : start + len(values)]
    view[:] = values
    assert view.ctypes.data % 64 == offset
    return view


def alignments():
    rng = np.random.default_rng(20261020)
    rows = []
    offsets = list(range(0, 64, 8))
    for n in (3, 7, 8, 9, 10, 17, 30, 64, 100, 257, 1000, 4096):
        x, y = rng.standard_normal((2, n))
        x -= x.mean()
        y -= y.mean()
        xmax, ymax = np.abs(x).max(), np.abs(y).max()
        xn = x / (xmax * np.linalg.norm(x / xmax, axis=-1, keepdims=True))
        yn = y / (ymax * np.linalg.norm(y / ymax, axis=-1, keepdims=True))
        reductions = {}
        for name, a, b in (("centered_xy", x, y), ("centered_xx", x, x), ("normalized_xy", xn, yn)):
            grid = []
            for left in offsets:
                aa = aligned_copy(a, left)
                line = []
                for right in offsets:
                    bb = aligned_copy(b, right)
                    line.append(np.asarray(np.vecdot(aa, bb, axis=-1)).tobytes().hex())
                grid.append(line)
            reductions[name] = {"hex_grid": grid, "distinct": sorted({v for line in grid for v in line})}
        rows.append({"n": n, "offsets_mod64": offsets, "reductions": reductions})
    return rows


def baseline_stability():
    rng = np.random.default_rng(20261021)
    rows = []
    for n in (7, 17, 30, 100, 257, 4096):
        source = rng.standard_normal((2, 2 * n))
        for layout in ("contiguous", "strided", "reversed"):
            if layout == "contiguous":
                x, y = source[:, :n].copy()
            elif layout == "strided":
                x, y = source[:, ::2]
            else:
                x, y = source[:, :n][:, ::-1]
            for name in ("pearsonr", "linregress"):
                for retain in (False, True):
                    kept, observed = [], []
                    for repeat in range(12):
                        # Change allocation history while keeping exact caller inputs.
                        scratch = [np.empty(n + j + repeat % 8) for j in range(repeat % 5)]
                        value, record = invoke(getattr(stats, name), (x, y), {}, plain=True)
                        observed.append(record)
                        if retain:
                            kept.append(value)
                        del value, scratch
                    encoded = [json.dumps(v, sort_keys=True) for v in observed]
                    rows.append(
                        {
                            "n": n,
                            "layout": layout,
                            "function": name,
                            "retain_results": retain,
                            "addresses_mod64": [x.ctypes.data % 64, y.ctypes.data % 64],
                            "distinct": [json.loads(v) for v in sorted(set(encoded))],
                            "sequence": [sorted(set(encoded)).index(v) for v in encoded],
                        }
                    )
    return rows


def randomized_replay():
    path = Path(__file__).resolve().parents[1] / "tests/test_stats.py"
    spec = importlib.util.spec_from_file_location("_stats_probe_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    # This follows the generators and failure-short-circuiting of the source
    # tests. Different allocation histories can move the first mismatch.
    for threads in (1, 4):
        # Preserve the source generator sequence, while respecting the
        # diagnostic's two-thread budget. These small stats paths are serial.
        fastscipy.set_num_threads(min(threads, 2))
        for name, generate in module.CASES.items():
            for iteration in range(150):
                args, kwargs = generate()
                if name not in ("pearsonr", "linregress"):
                    continue
                function = getattr(stats, name)
                expected, plain = invoke(function, args, kwargs, plain=True)
                before = fast_calls()
                got, patched = invoke(function, args, kwargs, plain=False)
                ran = fast_calls() > before
                matched = identical(expected, got) and plain == patched
                if not matched:
                    rows.append(
                        {
                            "function": name,
                            "source_test_threads": threads,
                            "threads": min(threads, 2),
                            "iteration": iteration,
                            "kwargs": kwargs,
                            "fast_path": ran,
                            "inputs": [array_record(a) for a in args],
                            "reference": plain,
                            "patched": patched,
                            "reference_repeats": [invoke(function, args, kwargs, plain=True)[1] for _ in range(8)],
                            "patched_repeats": [invoke(function, args, kwargs, plain=False)[1] for _ in range(8)],
                        }
                    )
                    break
            else:
                if name in ("pearsonr", "linregress"):
                    rows.append(
                        {
                            "function": name,
                            "source_test_threads": threads,
                            "threads": min(threads, 2),
                            "samples": 150,
                            "mismatch": False,
                        }
                    )
    fastscipy.set_num_threads(1)
    return {"seed": 12345, "test_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    report = {
        "purpose": "diagnostics only; no wheel/source overlay or production changes",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "fastscipy": fastscipy.__version__},
        "wheel_module": fastscipy.__file__,
    }
    if sys.platform != "darwin" or platform.machine().lower() != "x86_64":
        report["skipped"] = "requires native Intel Mac"
    else:
        fastscipy.patch(threads=1)
        report["status"] = fastscipy.status()
        report["alignment"] = alignments()
        report["baseline_stability"] = baseline_stability()
        report["randomized_replay"] = randomized_replay()
    report["seconds"] = time.monotonic() - started
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
