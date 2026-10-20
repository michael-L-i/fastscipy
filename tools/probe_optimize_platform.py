"""Small Intel Mac diagnostics for reference stability and NumPy reductions.

Uses an installed wheel; no builds and no tolerances or production overrides.
The copy-wrapper experiment is recorded separately from the installed path.
"""

from contextlib import nullcontext
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import _numdiff

import fastscipy
from fastscipy import _patch
from fastscipy.testing import identical


def address(a):
    return int(a.ctypes.data % 64)


def differences(a, b):
    return [key for key in a if not identical(a[key], b[key])]


def aligned(values, offset):
    storage = np.empty(values.size + 16)
    start = ((offset - storage.ctypes.data) % 64) // 8
    result = storage[start : start + values.size]
    result[:] = values
    assert address(result) == offset
    return result


def main():
    if sys.platform != "darwin" or platform.machine().lower() != "x86_64":
        print("Intel Mac optimization probe: not applicable", flush=True)
        return
    started = time.monotonic()
    fastscipy.patch(threads=2)
    report = {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "purpose": "diagnostic_only_installed_wheel",
        "least_squares": [],
        "dot_alignment": [],
        "bfgs": [],
    }
    rng = np.random.default_rng(852)
    retained = []
    for trial in range(8):
        a, b = rng.standard_normal((40, 10)), rng.standard_normal(40)
        for layout, matrix, rhs in (
            ("C", a, b),
            ("F", np.asfortranarray(a), b[::-1]),
            ("reverse", a[::-1], b[::-1]),
        ):
            for mode in ("reference", "patched"):
                with fastscipy.disabled() if mode == "reference" else nullcontext():
                    first = optimize.lsq_linear(matrix, rhs)
                    second = optimize.lsq_linear(matrix, rhs)
                retained.extend((first, second))
                report["least_squares"].append(
                    {
                        "trial": trial,
                        "layout": layout,
                        "mode": mode,
                        "changed_fields": differences(first, second),
                        "cost_hex": [float(r.cost).hex() for r in (first, second)],
                        "residual_mod64": [address(r.fun) for r in (first, second)],
                    }
                )
            reference, patched = retained[-4], retained[-2]
            report["least_squares"].append(
                {"trial": trial, "layout": layout, "pair_changed_fields": differences(reference, patched)}
            )

    for n in (10, 40, 100, 1000):
        values = np.random.default_rng(n).standard_normal(n)
        for offset in range(0, 64, 8):
            vector = aligned(values, offset)
            report["dot_alignment"].append(
                {
                    "n": n,
                    "mod64": address(vector),
                    "dot_hex": float(np.dot(vector, vector)).hex(),
                    "vecdot_hex": float(np.vecdot(vector, vector)).hex(),
                    "sum_hex": float(np.sum(vector * vector)).hex(),
                }
            )

    original_call = _numdiff._Fun_Wrapper.__call__

    def copy_call(self, x):
        if _patch.enabled and type(x) is np.ndarray and x.dtype == np.float64 and x is not self.x0:
            x = x.copy()
        return original_call(self, x)

    point = np.random.default_rng(0).random(1000)[:10]
    outputs = {}
    try:
        for mode in ("reference", "patched", "patched_with_copy_experiment"):
            _numdiff._Fun_Wrapper.__call__ = copy_call if mode.endswith("experiment") else original_call
            rows = []
            for _ in range(4):
                with fastscipy.disabled() if mode == "reference" else nullcontext():
                    result = optimize.minimize(optimize.rosen, point, method="BFGS")
                rows.append(result)
            outputs[mode] = rows
            report["bfgs"].append(
                {
                    "mode": mode,
                    "stable_x": all(identical(rows[0].x, r.x) for r in rows[1:]),
                    "reference_x_matches": [identical(outputs["reference"][0].x, r.x) for r in rows],
                    "iterations": [int(r.nit) for r in rows],
                    "x_hex": [[float(v).hex() for v in r.x] for r in rows],
                }
            )
    finally:
        _numdiff._Fun_Wrapper.__call__ = original_call
    report["seconds"] = time.monotonic() - started
    output = Path("artifacts/probe_optimize_platform.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
