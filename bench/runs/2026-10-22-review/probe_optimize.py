import platform
import sys
import scipy
import timeit
import numpy as np
from scipy.optimize import _numdiff, least_squares
from scipy.sparse import eye_array
import fastscipy
from fastscipy import _optimize
from fastscipy.testing import assert_identical

import subprocess

print(
    {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
    flush=True,
)
source = subprocess.check_output(["git", "show", "f60dca0:python/fastscipy/_optimize.py"], text=True)
namespace = {"__name__": "fastscipy._optimize_previous", "__package__": "fastscipy"}
exec(compile(source, "previous_optimize.py", "exec"), namespace)
candidate_patch = _optimize.patch
rng = np.random.default_rng(820)
cases = []
for n in (1, 10, 1000, 10000):
    point = rng.uniform(0.5, 1.0, n)
    vector = rng.standard_normal(n)

    def residual(x):
        return np.sin(x) + x * x

    for method in ("2-point", "3-point"):
        cases.append(
            (
                f"matrix-free action n={n} {method}",
                lambda point=point, vector=vector, method=method: _numdiff.approx_derivative(
                    residual, point, method=method, as_linear_operator=True
                ).matvec(vector),
            )
        )
    if n > 10:
        pattern = eye_array(n, format="csc")
        groups = np.zeros(n, dtype=int)
        for bound in ((-np.inf, np.inf), (0.0, 2.0)):
            cases.append(
                (
                    f"sparse diagonal n={n} bounds={bound}",
                    lambda point=point, pattern=pattern, groups=groups, bound=bound: (
                        _numdiff.approx_derivative(residual, point, sparsity=(pattern, groups), bounds=bound).data
                    ),
                )
            )
    elif n == 10:
        cases.append(("dense jac n=10", lambda point=point: _numdiff.approx_derivative(residual, point)))
for n in (10, 1000, 10000):
    pattern = eye_array(n, format="csr")
    for initial in (0.5, 1.0):
        point = np.full(n, initial)
        cases.append(
            (
                f"least_squares analytic sparse n={n} start={initial}",
                lambda point=point, pattern=pattern: least_squares(lambda z: z - 1.0, point, jac=lambda z: pattern),
            )
        )
point = np.full(10, 0.5)
pattern = eye_array(10, format="csr")
array_bounds = (np.full(10, -1.0), np.full(10, 2.0))
cases.append(
    (
        "least_squares array bounds control n=10",
        lambda point=point, pattern=pattern: least_squares(
            lambda z: z - 1.0, point, jac=lambda z: pattern, bounds=array_bounds
        ),
    )
)
results = {}
for state in ("scipy", "previous", "candidate"):
    fastscipy.unpatch()
    if state == "previous":
        _optimize.patch = namespace["patch"]
        fastscipy.patch(["optimize"])
    elif state == "candidate":
        _optimize.patch = candidate_patch
        fastscipy.patch(["optimize"])
    assert (state == "scipy" and not fastscipy.status()) or fastscipy.status() == {"optimize": "on"}
    print(state, fastscipy.status(), flush=True)
    results[state] = {}
    for name, call in cases:
        value = call()
        if state == "scipy":
            results.setdefault("values", {})[name] = value
        else:
            assert_identical(results["values"][name], value)
        count = 10 if "10000" in name else 100
        us = min(timeit.repeat(call, number=count, repeat=5)) * 1e6 / count
        results[state][name] = us
for name, _ in cases:
    a, b, c = (results[state][name] for state in ("scipy", "previous", "candidate"))
    print(
        f"{name}: scipy={a:.3f} previous={b:.3f} candidate={c:.3f} us; true={a / c:.3f}x incremental={b / c:.3f}x",
        flush=True,
    )
