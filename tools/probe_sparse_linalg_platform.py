"""Bounded Intel Mac solver repeatability and BLAS-alignment diagnostics."""

import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import psutil
import scipy
from scipy import sparse
from scipy.sparse import linalg as sl

import fastscipy


def digest(value):
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            h.update(str(len(item)).encode())
            for element in item:
                visit(element)
        else:
            array = np.asarray(item)
            h.update(array.dtype.str.encode())
            h.update(str(array.shape).encode())
            h.update(array.tobytes())

    visit(value)
    return h.hexdigest()


def alignment_probe():
    results = []
    for n in (13, 100):
        rng = np.random.default_rng(981 + n)
        x, y = rng.standard_normal(n), rng.standard_normal(n)
        rows = []
        for ix in range(8):
            xb = np.empty(n + 8)
            a = xb[ix : ix + n]
            a[:] = x
            for iy in range(8):
                yb = np.empty(n + 8)
                b = yb[iy : iy + n]
                b[:] = y
                rows.append(
                    {
                        "a_mod64": a.ctypes.data % 64,
                        "b_mod64": b.ctypes.data % 64,
                        "dot": float(np.dot(a, b)).hex(),
                        "norm": float(np.linalg.norm(a)).hex(),
                    }
                )
        results.append(
            {
                "n": n,
                "x_sha256": digest(x),
                "y_sha256": digest(y),
                "rows": rows,
                "unique_dot_results": len({r["dot"] for r in rows}),
                "unique_norm_results": len({r["norm"] for r in rows}),
            }
        )
    return results


def main():
    started = time.monotonic()
    output = Path("artifacts/sparse-linalg-probe")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "diagnostics_only": True,
        "cases": {},
        "budgets": {"seconds": 30, "rss_bytes": 256 * 1024**2, "threads": 1, "blas_threads": 1},
    }
    summary = output / "summary.json"
    process = psutil.Process()
    peak = 0

    def save():
        report["seconds"] = time.monotonic() - started
        report["sampled_peak_rss_bytes"] = peak
        summary.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def budget():
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        if time.monotonic() - started > 25 or peak > 240 * 1024**2:
            report["stopped"] = "time budget" if time.monotonic() - started > 25 else "memory budget"
            save()
            return False
        return True

    if sys.platform != "darwin" or platform.machine().lower() != "x86_64":
        report["skipped"] = "Intel Mac diagnostic only"
        save()
        print(report["skipped"], flush=True)
        return
    report["numpy_config"] = np.__config__.show(mode="dicts")
    provenance = Path("artifacts/provenance.json")
    report["workflow_provenance"] = json.loads(provenance.read_text()) if provenance.exists() else None
    fastscipy.unpatch()
    report["patch_status"] = fastscipy.patch(["sparse_linalg"], threads=1)
    report["alignment"] = alignment_probe()
    cases = []
    for solver in (sl.cg, sl.bicg, sl.lgmres, sl.qmr, sl.tfqmr, sl.gcrotmk):
        n = 100
        a = sparse.diags_array([np.full(n - 1, -0.5), np.arange(n) + 2.0, np.full(n - 1, -0.5)], offsets=[-1, 0, 1])
        b = np.random.default_rng(n).standard_normal(n)

        def solve(solver=solver, a=a, b=b):
            history = []
            result = solver(a, b, rtol=1e-9, callback=lambda x: history.append(np.array(x, copy=True)))
            return result, history

        cases.append((solver.__name__ + "-100", solve, a, b))
    rng = np.random.default_rng(82)
    dense, b = rng.standard_normal((13, 7)), rng.standard_normal(13)
    for cls in (sparse.csr_array, sparse.csc_matrix):
        a = cls(dense)
        cases.append(("lsqr-" + cls.__name__, lambda a=a, b=b: (sl.lsqr(a, b), []), a, b))
    for name, solve, a, b in cases:
        record = {
            "input_sha256": digest((a.data, b)),
            "format": a.format,
            "b_mod64": b.ctypes.data % 64,
            "runs": [],
            "distinct_results": {},
        }
        report["cases"][name] = record

        def retain(result, label):
            value, history = result
            fingerprint = digest(result)
            record["runs"].append(
                {
                    "phase": label,
                    "sha256": fingerprint,
                    "history_sha256": [digest(x) for x in history],
                    "history_length": len(history),
                }
            )
            if fingerprint not in record["distinct_results"]:
                path = output / f"{name}-{fingerprint[:12]}.npz"
                arrays = {f"result_{i}": np.asarray(v) for i, v in enumerate(value)}
                arrays["history"] = np.asarray(history)
                np.savez(path, **arrays)
                record["distinct_results"][fingerprint] = path.name

        for patched in (False, True):
            label = "patched" if patched else "baseline"
            for keep in (True, False):
                # Toggle enabled state without reinstalling wrappers, matching
                # the project's byte-comparison baseline context.
                held = []
                for _ in range(4):
                    if not budget():
                        return
                    if patched:
                        result = solve()
                    else:
                        with fastscipy.disabled():
                            result = solve()
                    if keep:
                        held.append(result)
                    else:
                        retain(result, label + "-released")
                    del result
                for result in held:
                    retain(result, label + "-retained")
                if held:
                    del result
                del held
        for _ in range(3):
            if not budget():
                return
            with fastscipy.disabled():
                plain = solve()
            patched = solve()
            retain(plain, "interleaved-baseline")
            retain(patched, "interleaved-patched")
            del plain, patched
        record["baseline_variants"] = len({r["sha256"] for r in record["runs"] if "baseline" in r["phase"]})
        record["patched_variants"] = len({r["sha256"] for r in record["runs"] if "patched" in r["phase"]})
        print(
            name,
            "baseline variants",
            record["baseline_variants"],
            "patched variants",
            record["patched_variants"],
            flush=True,
        )
        save()
    budget()
    report["fastscipy_counts"] = {str(key): value for key, value in fastscipy.stats().items()}
    save()
    print("Sparse linalg diagnostic:", summary, flush=True)


if __name__ == "__main__":
    main()
