"""Correctness diagnostics only: exact release CP312 Intel Pearson/QMR cases.

No source overlay, compilation, benchmarks, tolerances, or acceptance decisions.
--reference-only is useful when the failed release wheel was not uploaded.
"""
# Configure numerical thread limits before importing numerical libraries.
# ruff: noqa: E402

import argparse
import ast
import base64
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["FASTSCIPY_NUM_THREADS"] = "2"
for key in ("FASTSCIPY_ANY_PLATFORM", "FASTSCIPY_ANY_VERSION"):
    os.environ.pop(key, None)

import numpy as np
import scipy
from scipy import sparse, stats
from scipy.sparse import linalg as sl


def digest(data):
    return hashlib.sha256(data).hexdigest()


def array(a, *, values=True):
    a = np.asarray(a)
    result = dict(
        dtype=a.dtype.str,
        shape=list(a.shape),
        strides=list(a.strides),
        address_mod64=int(a.ctypes.data % 64),
        sha256=digest(a.tobytes()),
        writeable=bool(a.flags.writeable),
        hex=a.tobytes().hex(),
    )
    if values and a.size <= 100:
        result["repr"] = repr(a)
    return result


def encode(value):
    if isinstance(value, np.ndarray) or isinstance(value, np.generic):
        return array(value)
    if isinstance(value, (str, bool, int, type(None))):
        return value
    if isinstance(value, float):
        return dict(python_type="float", **array(np.asarray(value)))
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return dict(type=type(value).__name__, values=[encode(v) for v in value])
    return dict(type=f"{type(value).__module__}.{type(value).__qualname__}", repr=repr(value))


def canonical(value):
    # Buffer addresses are diagnostics, not part of the numeric contract.
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items() if k != "address_mod64"}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def variants(records):
    strings = [json.dumps(canonical(r), sort_keys=True) for r in records]
    unique = sorted(set(strings))
    return dict(
        count=len(unique), sequence=[unique.index(v) for v in strings], sha256=[digest(v.encode()) for v in unique]
    )


def pearson_record(x, y, kwargs):
    result = stats.pearsonr(x, y, **kwargs)
    return {
        "type": f"{type(result).__module__}.{type(result).__qualname__}",
        "statistic": encode(result.statistic),
        "pvalue": encode(result.pvalue),
        "metadata": encode(vars(result)),
        "domain_properties": {
            "input_has_infinity": bool(np.isinf(x).any() or np.isinf(y).any()),
            "input_has_nan": bool(np.isnan(x).any() or np.isnan(y).any()),
            "statistic_is_nan": bool(np.isnan(result.statistic)),
            "pvalue_is_nan": bool(np.isnan(result.pvalue)),
            "alias_bits_equal": bool(
                np.asarray(result.correlation).dtype == np.asarray(result.statistic).dtype
                and np.asarray(result.correlation).tobytes() == np.asarray(result.statistic).tobytes()
            ),
        },
    }


def aligned_copy(values, offset):
    storage = np.empty(values.size + 64 // values.dtype.itemsize, dtype=values.dtype)
    start = ((offset - storage.ctypes.data) % 64) // values.dtype.itemsize
    view = storage[start : start + values.size]
    view[:] = values
    assert view.ctypes.data % 64 == offset
    view.flags.writeable = values.flags.writeable
    return view


def source_cases(path):
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("sample", "pair", "opts"):
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("rng", "CASES", "ALTS", "XS", "QS") for t in node.targets
        ):
            selected.append(node)
    namespace = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    found = []
    for threads in (1, 4):
        for name, generate in namespace["CASES"].items():
            for iteration in range(150):
                args, kwargs = generate()
                if name != "pearsonr":
                    continue
                x, y = args
                # Match the exact input summaries in the failed tag's CP312 JUnit.
                match = (
                    threads == 1
                    and x.shape == y.shape == (17,)
                    and np.min(x) == 0
                    and np.max(x) == 4
                    and np.min(y) == 0.5443752653524361
                    and np.isposinf(y).any()
                )
                match |= (
                    threads == 4
                    and x.shape == y.shape == (30,)
                    and np.min(x) == -2.093968371476999
                    and np.isposinf(x).any()
                    and np.min(y) == 0
                    and np.max(y) == 4
                )
                if match:
                    found.append((threads, iteration, args, kwargs))
                    # The original test stops here, affecting subsequent RNG state.
                    break
    assert [r[0] for r in found] == [1, 4], "exact seeded failure inputs not recovered"
    return found


def invoke(function, *, plain, fastscipy, fast_calls):
    before = fast_calls() if fast_calls else 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fastscipy.disabled() if plain and fastscipy else contextlib.nullcontext():
            try:
                result = function()
                record = {"result": result}
            except Exception as exc:
                record = {"error": {"type": type(exc).__qualname__, "message": str(exc)}}
    record["warnings"] = [
        {"category": f"{w.category.__module__}.{w.category.__qualname__}", "message": str(w.message)} for w in caught
    ]
    numeric = canonical(record)
    record["numeric_contract_sha256"] = digest(json.dumps(numeric, sort_keys=True).encode())
    record["fast_calls"] = (fast_calls() - before) if fast_calls else None
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    assert sys.platform == "darwin" and platform.machine().lower() == "x86_64"
    assert sys.version_info[:2] == (3, 12)
    root = args.source_root.resolve()
    test_path = root / "tests/test_stats.py"
    source = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert source == "1cf67791b17ab2bb3c72d6af62bd2292e8034583", source
    fastscipy = fast_calls = None
    if not args.reference_only:
        import fastscipy
        from fastscipy.testing import fast_calls

        assert fastscipy.__version__ == "0.1.0a2"
        assert root not in Path(fastscipy.__file__).resolve().parents
        fastscipy.patch(threads=1)
    report = {
        "purpose": "partial correctness diagnostics only; no release validation or performance claim",
        "reference_only": args.reference_only,
        "source": source,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "test_source_sha256": digest(test_path.read_bytes()),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "thread_env": {
            k: os.environ.get(k)
            for k in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "FASTSCIPY_NUM_THREADS",
                "FASTSCIPY_ANY_PLATFORM",
                "FASTSCIPY_ANY_VERSION",
            )
        },
        "pearson": [],
        "solvers": [],
    }
    if fastscipy:
        package = Path(fastscipy.__file__).resolve().parent
        report["installed"] = {
            "path": str(package),
            "version": fastscipy.__version__,
            "files": {
                str(p.relative_to(package)): digest(p.read_bytes())
                for p in package.rglob("*")
                if p.suffix in (".py", ".so")
            },
        }
        report["status"] = fastscipy.status()
    for source_threads, iteration, values, kwargs in source_cases(test_path):
        x, y = values
        initial = [digest(v.tobytes()) for v in values]
        row = {
            "source_test_threads": source_threads,
            "native_threads": min(source_threads, 2),
            "seed": 12345,
            "iteration": iteration,
            "kwargs": kwargs,
            "inputs": [dict(array(v), base64=base64.b64encode(v.tobytes()).decode()) for v in values],
            "calls": [],
        }
        if fastscipy:
            fastscipy.set_num_threads(min(source_threads, 2))
        for plain in (True, False) if fastscipy else (True,):
            observations = []
            for repeat in range(16):
                scratch = [np.empty(8 * len(x) + j) for j in range(repeat % 4)]

                def call():
                    return pearson_record(x, y, kwargs)

                record = invoke(call, plain=plain, fastscipy=fastscipy, fast_calls=fast_calls)
                observations.append(record)
                del scratch
            row["calls"].append(
                {
                    "plain": plain,
                    "variants": variants([{k: v for k, v in r.items() if k != "fast_calls"} for r in observations]),
                    "repeats": observations,
                }
            )
        # These copies preserve caller values/dtypes/read-only flags while
        # deliberately varying pointer alignment. They are separate from the
        # unchanged original-layout repeated-call evidence above.
        grid = []
        for left in range(0, 64, x.dtype.itemsize):
            for right in range(0, 64, y.dtype.itemsize):
                xx, yy = aligned_copy(x, left), aligned_copy(y, right)
                plain = invoke(
                    lambda: pearson_record(xx, yy, kwargs), plain=True, fastscipy=fastscipy, fast_calls=fast_calls
                )
                grid.append({"offsets": [left, right], "inputs": [array(xx), array(yy)], "plain": plain})
        row["reference_alignment_grid"] = grid
        row["inputs_unchanged"] = initial == [digest(v.tobytes()) for v in values]
        assert row["inputs_unchanged"]
        report["pearson"].append(row)
    n = 100
    a = sparse.diags_array([np.full(n - 1, -0.5), np.arange(n) + 2.0, np.full(n - 1, -0.5)], offsets=[-1, 0, 1])
    b = np.random.default_rng(n).standard_normal(n)
    initial = (a.data.tobytes(), b.tobytes())
    for solver_name in ("qmr", "minres"):
        for threads in (1, 2):
            if fastscipy:
                fastscipy.set_num_threads(threads)
            row = {
                "solver": solver_name,
                "n": n,
                "rtol": 1e-9,
                "native_threads": threads,
                "b": array(b),
                "matrix_data": array(a.data),
                "calls": [],
            }
            for plain in (True, False) if fastscipy else (True,):
                records = []
                for repeat in range(12):
                    scratch = [np.empty(n + j) for j in range(repeat % 5)]

                    def call():
                        history = []
                        solution, info = getattr(sl, solver_name)(
                            a, b, rtol=1e-9, callback=lambda x: history.append(x.copy())
                        )
                        residual = []
                        for i in range(n):
                            terms = [float(b[i]), -float(i + 2) * float(solution[i])]
                            if i:
                                terms.append(0.5 * float(solution[i - 1]))
                            if i + 1 < n:
                                terms.append(0.5 * float(solution[i + 1]))
                            residual.append(math.fsum(terms))
                        return {
                            "solution": array(solution),
                            "info": encode(info),
                            "callback_count": len(history),
                            "history": [array(v, values=False) for v in history],
                            "independent_residual_norm": math.sqrt(math.fsum(v * v for v in residual)),
                            "residual_limit": 1e-9 * math.sqrt(math.fsum(float(v) * float(v) for v in b)),
                        }

                    records.append(invoke(call, plain=plain, fastscipy=fastscipy, fast_calls=fast_calls))
                    del scratch
                row["calls"].append(
                    {
                        "plain": plain,
                        "variants": variants([{k: v for k, v in r.items() if k != "fast_calls"} for r in records]),
                        "repeats": records,
                    }
                )
            row["inputs_unchanged"] = initial == (a.data.tobytes(), b.tobytes())
            assert row["inputs_unchanged"]
            report["solvers"].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source": source,
                "reference_only": args.reference_only,
                "pearson_cases": len(report["pearson"]),
                "solver_configs": len(report["solvers"]),
            }
        )
    )


if __name__ == "__main__":
    main()
