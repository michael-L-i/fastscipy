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

    def solver_cases(scope, records, *, include_lsqr=False):
        cases = []
        solvers = (sl.cg, sl.bicg, sl.bicgstab, sl.cgs, sl.gmres, sl.lgmres, sl.minres, sl.qmr, sl.tfqmr, sl.gcrotmk)
        for solver in solvers:
            n = 100
            a = sparse.diags_array([np.full(n - 1, -0.5), np.arange(n) + 2.0, np.full(n - 1, -0.5)], offsets=[-1, 0, 1])
            b = np.random.default_rng(n).standard_normal(n)

            def solve(solver=solver, a=a, b=b):
                history = []
                kwargs = {"callback_type": "legacy"} if solver is sl.gmres else {}
                result = solver(a, b, rtol=1e-9, callback=lambda x: history.append(np.array(x, copy=True)), **kwargs)
                return result, history

            cases.append((solver.__name__ + "-100", solve, a, b))
        if include_lsqr:
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
            records[name] = record

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
                    path = output / f"{scope}-{name}-{fingerprint[:12]}.npz"
                    arrays = {f"result_{i}": np.asarray(v) for i, v in enumerate(value)}
                    arrays["history"] = np.asarray(history)
                    np.savez(path, **arrays)
                    record["distinct_results"][fingerprint] = path.name

            for patched in (False, True):
                label = "patched" if patched else "baseline"
                for mode in ("retained", "released", "churn"):
                    keep = mode == "retained"
                    # Toggle enabled state without reinstalling wrappers, matching
                    # the project's byte-comparison baseline context.
                    held = []
                    # Give the observed BiCGSTAB count boundary extra allocator
                    # histories without expanding the global time/RSS budget.
                    for _ in range(12 if name == "bicgstab-100" else 4):
                        if not budget():
                            return False
                        if mode == "churn":
                            churn = [np.empty(13 * i + 1) for i in range(1, 33)]
                            del churn
                        if patched:
                            result = solve()
                        else:
                            with fastscipy.disabled():
                                result = solve()
                        if keep:
                            held.append(result)
                        else:
                            retain(result, label + "-" + mode)
                        del result
                    for result in held:
                        retain(result, label + "-retained")
                    if held:
                        del result
                    del held
            for _ in range(8 if name == "bicgstab-100" else 3):
                if not budget():
                    return False
                with fastscipy.disabled():
                    plain = solve()
                patched = solve()
                retain(plain, "interleaved-baseline")
                retain(patched, "interleaved-patched")
                del plain, patched
            record["baseline_variants"] = len({r["sha256"] for r in record["runs"] if "baseline" in r["phase"]})
            record["patched_variants"] = len({r["sha256"] for r in record["runs"] if "patched" in r["phase"]})
            record["baseline_history_lengths"] = sorted(
                {r["history_length"] for r in record["runs"] if "baseline" in r["phase"]}
            )
            record["patched_history_lengths"] = sorted(
                {r["history_length"] for r in record["runs"] if "patched" in r["phase"]}
            )
            print(
                scope,
                name,
                "baseline variants",
                record["baseline_variants"],
                "patched variants",
                record["patched_variants"],
                flush=True,
            )
            save()
        return True

    if not solver_cases("sparse-linalg-only", report["cases"], include_lsqr=True):
        return

    # Repeat the exact rectangular test after installing every component. This
    # includes patched sparse construction, not just LinearOperator dispatch.
    report["all_components"] = {"patch_status": fastscipy.patch(threads=1), "matrices": {}, "cases": {}}
    report["all_components"]["iterative_cases"] = {}
    if not solver_cases("all-components", report["all_components"]["iterative_cases"]):
        return
    for dtype in (np.float64, np.complex128):
        rng = np.random.default_rng(82)
        dense = rng.standard_normal((13, 7)).astype(dtype)
        if np.dtype(dtype).kind == "c":
            dense += 1j * rng.standard_normal(dense.shape)
        rhs = rng.standard_normal(13).astype(dtype)
        for cls in (sparse.csr_array, sparse.csc_matrix):
            if not budget():
                return
            with fastscipy.disabled():
                reference_matrix = cls(dense)
            matrix = cls(dense)
            label = f"{np.dtype(dtype).name}-{cls.__name__}"
            buffers = lambda m: (m.data, m.indices, m.indptr)
            matrix_record = {
                "reference_buffers_sha256": digest(buffers(reference_matrix)),
                "patched_buffers_sha256": digest(buffers(matrix)),
                "reference_indices_dtype": str(reference_matrix.indices.dtype),
                "patched_indices_dtype": str(matrix.indices.dtype),
                "rhs_sha256": digest(rhs),
                "direct_checks": [],
            }
            matrix_record["buffers_identical"] = (
                matrix_record["reference_buffers_sha256"] == matrix_record["patched_buffers_sha256"]
            )
            report["all_components"]["matrices"][label] = matrix_record
            np.savez(
                output / f"all-inputs-{label}.npz",
                dense=dense,
                rhs=rhs,
                reference_data=reference_matrix.data,
                reference_indices=reference_matrix.indices,
                reference_indptr=reference_matrix.indptr,
                patched_data=matrix.data,
                patched_indices=matrix.indices,
                patched_indptr=matrix.indptr,
            )
            operator = sl.aslinearoperator(matrix)
            values = np.random.default_rng(482).standard_normal(7).astype(dtype)
            for offset in range(8):
                xb, yb = np.empty(15, dtype), np.empty(21, dtype)
                x, y = xb[offset : offset + 7], yb[offset : offset + 13]
                x[:], y[:] = values, rhs
                for operation, call in (
                    ("matrix", lambda: matrix @ x),
                    ("operator", lambda: operator.matvec(x)),
                    ("adjoint", lambda: operator.rmatvec(y)),
                ):
                    with fastscipy.disabled():
                        expected = call()
                    actual = call()
                    identical = digest(expected) == digest(actual)
                    matrix_record["direct_checks"].append(
                        {
                            "operation": operation,
                            "x_mod64": x.ctypes.data % 64,
                            "y_mod64": y.ctypes.data % 64,
                            "identical": identical,
                            "reference_sha256": digest(expected),
                            "patched_sha256": digest(actual),
                        }
                    )
                    if not identical:
                        np.savez(
                            output / f"direct-{label}-{operation}-{offset}.npz",
                            expected=expected,
                            actual=actual,
                            x=x,
                            y=y,
                        )
            for solver in (sl.lsqr, sl.lsmr):
                name = f"all-{solver.__name__}-{label}"
                record = {"runs": [], "distinct_results": {}, "pairs": []}
                report["all_components"]["cases"][name] = record
                before = digest((buffers(matrix), rhs))

                def solve(solver=solver, matrix=matrix, rhs=rhs):
                    return solver(matrix, rhs)

                def stash(result, phase):
                    fingerprint = digest(result)
                    record["runs"].append({"phase": phase, "sha256": fingerprint})
                    if fingerprint not in record["distinct_results"]:
                        path = output / f"{name}-{fingerprint[:12]}.npz"
                        np.savez(path, **{f"result_{i}": np.asarray(v) for i, v in enumerate(result)})
                        record["distinct_results"][fingerprint] = path.name
                    return fingerprint

                for patched in (False, True):
                    phase = "patched" if patched else "baseline"
                    for allocation in ("retained", "released", "churn"):
                        held = []
                        for _ in range(8):
                            if not budget():
                                return
                            junk = [np.empty(n, dtype) for n in (7, 13, 7, 13, 14, 26)] if allocation == "churn" else []
                            if patched:
                                result = solve()
                            else:
                                with fastscipy.disabled():
                                    result = solve()
                            del junk
                            if allocation == "retained":
                                held.append(result)
                            else:
                                stash(result, phase + "-" + allocation)
                            del result
                        for result in held:
                            stash(result, phase + "-retained")
                        if held:
                            del result
                        del held
                for _ in range(4):
                    if not budget():
                        return
                    with fastscipy.disabled():
                        expected = solve()
                    actual = solve()
                    plain_digest = stash(expected, "interleaved-baseline")
                    fast_digest = stash(actual, "interleaved-patched")
                    record["pairs"].append(
                        {"baseline": plain_digest, "patched": fast_digest, "identical": plain_digest == fast_digest}
                    )
                    del expected, actual
                record["baseline_variants"] = len({r["sha256"] for r in record["runs"] if "baseline" in r["phase"]})
                record["patched_variants"] = len({r["sha256"] for r in record["runs"] if "patched" in r["phase"]})
                record["input_unchanged"] = before == digest((buffers(matrix), rhs))
                print(
                    name,
                    "baseline variants",
                    record["baseline_variants"],
                    "patched variants",
                    record["patched_variants"],
                    "unequal pairs",
                    sum(not p["identical"] for p in record["pairs"]),
                    flush=True,
                )
                save()
    budget()
    report["fastscipy_counts"] = {str(key): value for key, value in fastscipy.stats().items()}
    save()
    print("Sparse linalg diagnostic:", summary, flush=True)


if __name__ == "__main__":
    main()
