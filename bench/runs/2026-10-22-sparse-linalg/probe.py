"""Bounded candidate audit; run with the resource-monitor command in audit.json.

Reconstructs the old Python dispatch from the pinned commit. No native build.
"""

import json
import subprocess
import types
import statistics
import timeit
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sl
from scipy.sparse import _sparsetools
import fastscipy
from fastscipy import _sparse_linalg as fs, _patch
from fastscipy.testing import assert_identical

plain = sl.LinearOperator._shared_matvec
plain_inner = sl.LinearOperator._matvec
fastscipy.set_num_threads(1)
baseline = types.ModuleType("fastscipy._probe_baseline")
source = subprocess.run(
    ["git", "show", "f60dca0:python/fastscipy/_sparse_linalg.py"],
    check=True,
    capture_output=True,
    text=True,
).stdout
exec(compile(source, "f60dca0:_sparse_linalg.py", "exec"), baseline.__dict__)
baseline.patch()
old = sl.LinearOperator._shared_matvec
old_inner = sl.LinearOperator._matvec


def reshape(self, x, adjoint=False):
    if (
        not _patch.enabled
        or type(self) not in fs._OPERATORS
        or self._xp is not fs._NP
        or len(self.shape) != 2
        or type(x) is not np.ndarray
        or x.dtype.kind not in "fc"
        or type(adjoint) is not bool
    ):
        return plain(self, x, adjoint)
    m, n = self.shape
    inner, outer = (m, n) if adjoint else (n, m)
    if x.shape != (inner,):
        return plain(self, x, adjoint)
    _patch.counts["LinearOperator._shared_matvec", "fast"] += 1
    y = self._rmatvec(x) if adjoint else self._matvec(x)
    return y.reshape((outer,)) if type(y) is np.ndarray else fs._NP.reshape(y, (outer,))


def direct(self, x):
    a = getattr(self, "A", None)
    if (
        not _patch.enabled
        or type(self) not in fs._MATRIX_OPERATORS
        or self._xp is not fs._NP
        or type(a) not in fs._COMPRESSED
        or a.ndim != 2
        or not a.shape[0]
        or type(x) is not np.ndarray
        or x.ndim != 1
        or not x.size
        or x.shape[0] != a.shape[1]
        or a.dtype != x.dtype
        or a.dtype.char not in "fdFD"
        or not a.dtype.isnative
        or "_matmul_vector" in a.__dict__
    ):
        return old_inner(self, x)
    _patch.counts["MatrixLinearOperator sparse matvec", "fast"] += 1
    y = np.zeros(a.shape[0], dtype=x.dtype)
    fn = _sparsetools.csr_matvec if a.format == "csr" else _sparsetools.csc_matvec
    fn(*a.shape, a.indptr, a.indices, a.data, x.ravel(), y)
    return y


rng = np.random.default_rng(5)
cases = []
for n in (32, 256, 4096):
    a = sparse.diags_array([-np.ones(n - 1), np.full(n, 4.0), -np.ones(n - 1)], offsets=[-1, 0, 1], format="csr")
    b = rng.standard_normal(n)
    op = sl.aslinearoperator(a)
    custom = sl.LinearOperator(a.shape, matvec=lambda x: 2 * x, dtype=np.float64)
    comp = op + custom
    for name, call in [
        ("csr_matvec", lambda o=op, b=b: o.matvec(b)),
        ("custom_matvec", lambda o=custom, b=b: o.matvec(b)),
        ("composed_matvec", lambda o=comp, b=b: o.matvec(b)),
        ("cg", lambda a=a, b=b: sl.cg(a, b, rtol=1e-10)),
        ("gmres", lambda a=a, b=b: sl.gmres(a, b, rtol=1e-10)),
        ("cg_composed", lambda a=comp, b=b: sl.cg(a, b, rtol=1e-10)),
    ]:
        cases.append((f"{name} n={n}", call))
results = {name: {} for name, _ in cases}
for mode, shared, inner in [
    ("plain", plain, plain_inner),
    ("old", old, old_inner),
    ("reshape", reshape, old_inner),
    ("direct", reshape, direct),
]:
    sl.LinearOperator._shared_matvec = shared
    sl.LinearOperator._matvec = inner
    for name, call in cases:
        output = call()
        loops = 100 if "cg" in name or "gmres" in name else 2000
        value = statistics.median(timeit.repeat(call, number=loops, repeat=5)) / loops
        results[name][mode] = value
for v in results.values():
    v["reshape_vs_old"] = v["old"] / v["reshape"]
    v["direct_vs_old"] = v["old"] / v["direct"]
    v["direct_vs_plain"] = v["plain"] / v["direct"]
print(json.dumps(results, indent=2))

for n in (32, 256):
    a = sparse.diags_array([-np.ones(n - 1), np.full(n, 4.0), -np.ones(n - 1)], offsets=[-1, 0, 1], format="csr")
    b = rng.standard_normal(n)
    for solver in (sl.cg, sl.gmres):
        outcomes = []
        for shared, inner in [(plain, plain_inner), (old, old_inner), (reshape, old_inner), (reshape, direct)]:
            sl.LinearOperator._shared_matvec = shared
            sl.LinearOperator._matvec = inner
            history = []
            kw = {"callback_type": "legacy"} if solver is sl.gmres else {}
            result = solver(a, b, rtol=1e-10, callback=lambda x: history.append(np.array(x, copy=True)), **kw)
            outcomes.append((result, history))
            assert result[1] == 0
            assert np.linalg.norm(b - a @ result[0]) <= 1e-10 * np.linalg.norm(b)
        for result in outcomes[1:]:
            assert_identical(outcomes[0], result)
print("exact solve outputs, callback histories and independent residuals passed")
