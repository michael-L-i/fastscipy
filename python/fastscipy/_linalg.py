"""Batched stacks for the scipy.linalg functions that SciPy still loops over in
Python (``_apply_over_batch``): ``lu_factor``, ``lu_solve``, ``cho_solve``,
``solve_triangular``, ``eigh`` and ``eigvalsh``; plus ``null_space`` of tall matrices
(skips the unused (M, M) ``u``). Single NumPy triangular solves skip generic
array validation and use SciPy's own LAPACK dispatcher, including complex data.
Single factor solves bypass the batch adapter while retaining SciPy's core
validation and LAPACK calls.

Tier 1 (faithful port). The Rust kernels call the same LAPACK routines as SciPy's
f2py wrappers. They commit only if every matrix finishes with ``info == 0`` and
finite input; otherwise SciPy reruns the whole call and warns or raises itself.
"""

import functools
import math

import numpy as np

from . import _native, _patch
from ._threads import get_num_threads

_ND = np.ndarray
_F8 = np.dtype(np.float64)
_I4 = np.dtype(np.int32)
_MAX_BYTES = 1 << 29  # per array; SciPy handles anything bigger
_MIN_WORK_US = 150.0  # no threads for less work than this
_lwork_cache = {}
_triangular_core = None


def _ok(x, ndim, dtype=_F8):
    return x.__class__ is np.ndarray and x.dtype is dtype and x.ndim >= ndim


def _flatten(arrs):
    """``arrs``: [(array, core ndim)]. Returns (batch shape, B, [(B, *core) views]) or
    None when nothing is batched, the batch is empty or too big, or shapes do not broadcast."""
    shapes = [a.shape[: a.ndim - c] for a, c in arrs]
    if not any(shapes):
        return None
    try:
        batch = shapes[0] if all(s == shapes[0] for s in shapes[1:]) else np.broadcast_shapes(*shapes)
    except ValueError:
        return None
    B = math.prod(batch)
    if B == 0 or any(B * math.prod(a.shape[a.ndim - c :]) * 8 > _MAX_BYTES for a, c in arrs):
        return None
    flat = []
    for (a, c), shape in zip(arrs, shapes):
        core = a.shape[-c:]
        if shape != batch:
            a = np.broadcast_to(a, batch + core)
        flat.append(a.reshape((B,) + core))
    return batch, B, flat


def _threads(B, n, k=1, scale=1.0):
    est_us = B * (0.15 + scale * (0.004 * n**3 + 0.01 * n * n * k))
    return max(1, min(get_num_threads(), int(est_us / _MIN_WORK_US)))


def _fview(raw, batch, rows, cols):
    """The stack ``_apply_over_batch`` returns for per-matrix f2py results: each slice Fortran-ordered
    (C order for vectors), then reshaped to the batch shape."""
    if rows == 1 or cols == 1:
        return raw.reshape(batch + (rows, cols))
    return raw.reshape(-1, cols, rows).swapaxes(1, 2).reshape(batch + (rows, cols))


def _rhs(b):
    """(core ndim of b, (B, n, k) view builder input)."""
    return 2 if b.ndim >= 2 else 1


def _solve_shapes(a, b):
    n = a.shape[-1]
    bc = _rhs(b)
    if n == 0 or a.shape[-2] != n or b.shape[b.ndim - bc] != n or b.size == 0:
        return None
    fl = _flatten([(a, 2), (b, bc)])
    if fl is None:
        return None
    batch, B, (av, bv) = fl
    return n, bc, batch, B, av, (bv if bc == 2 else bv[:, :, None])


def _solution(raw, batch, n, bc, k):
    return _fview(raw, batch, n, k) if bc == 2 else raw.reshape(batch + (n,))


def _single_factor_inputs(factor, b):
    if type(factor) not in (tuple, list) or len(factor) != 2:
        return None
    a, option = factor
    if (
        type(a) is not _ND
        or a.ndim != 2
        or a.shape[0] != a.shape[1]
        or a.shape[0] != b.shape[0]
        or not b.size
        or a.dtype.char not in "fdFD"
        or b.dtype.char not in "fdFD"
    ):
        return None
    return a, option


def _cho_solver(original, core):
    @functools.wraps(original)
    def solve(c_and_lower, b, overwrite_b=False, check_finite=True):
        if _patch.enabled and type(b) is _ND and b.ndim in (1, 2):
            inputs = _single_factor_inputs(c_and_lower, b)
            if inputs is not None and overwrite_b is False and type(check_finite) is bool:
                a, lower = inputs
                if type(lower) in (bool, np.bool_) or (
                    type(lower) is _ND and lower.ndim == 0 and lower.dtype == np.bool_
                ):
                    _patch.counts["cho_solve", "fast"] += 1
                    return core(a, b, lower, False, check_finite)
        if _patch.enabled:
            try:
                result = _cho_solve(c_and_lower, b, overwrite_b, check_finite)
            except Exception:
                if _patch.STRICT:
                    raise
                _patch.counts["cho_solve", "error"] += 1
            else:
                if result is not NotImplemented:
                    _patch.counts["cho_solve", "fast"] += 1
                    return result
        _patch.counts["cho_solve", "scipy"] += 1
        return original(c_and_lower, b, overwrite_b=overwrite_b, check_finite=check_finite)

    return solve


def _lu_solver(original, core):
    @functools.wraps(original)
    def solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
        if _patch.enabled and type(b) is _ND and b.ndim in (1, 2):
            inputs = _single_factor_inputs(lu_and_piv, b)
            if (
                inputs is not None
                and overwrite_b is False
                and type(check_finite) is bool
                and type(trans) is int
                and trans in (0, 1, 2)
            ):
                a, piv = inputs
                if (
                    type(piv) is _ND
                    and piv.dtype == _I4
                    and piv.ndim == 1
                    and len(piv) == a.shape[0]
                    and not ((piv < 0) | (piv >= a.shape[0])).any()
                ):
                    _patch.counts["lu_solve", "fast"] += 1
                    # The original core checks only RHS finiteness. Run it
                    # outside the generic fallback: cast error handlers and
                    # exceptions must never cause the solve to run twice.
                    return core(a, piv, b, trans, False, check_finite)
        if _patch.enabled:
            try:
                result = _lu_solve(lu_and_piv, b, trans, overwrite_b, check_finite)
            except Exception:
                if _patch.STRICT:
                    raise
                _patch.counts["lu_solve", "error"] += 1
            else:
                if result is not NotImplemented:
                    _patch.counts["lu_solve", "fast"] += 1
                    return result
        _patch.counts["lu_solve", "scipy"] += 1
        return original(lu_and_piv, b, trans=trans, overwrite_b=overwrite_b, check_finite=check_finite)

    return solve


def _lu_factor(a, overwrite_a=False, check_finite=True):
    if (
        a.__class__ is not _ND
        or a.ndim < 3
        or not _ok(a, 3)
        or overwrite_a is not False
        or check_finite.__class__ is not bool
    ):
        return NotImplemented
    n = a.shape[-1]
    if a.shape[-2] != n or n == 0 or (fl := _flatten([(a, 2)])) is None:
        return NotImplemented
    batch, B, (av,) = fl
    r = _native.linalg.getrf(av, check_finite, _threads(B, n))
    if r is None:
        return NotImplemented
    return [_fview(r[0], batch, n, n), r[1].reshape(batch + (n,))]


def _lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    if lu_and_piv.__class__ not in (tuple, list) or len(lu_and_piv) != 2:
        return NotImplemented
    lu, piv = lu_and_piv
    if (
        lu.__class__ is not _ND
        or b.__class__ is not _ND
        or piv.__class__ is not _ND
        or not (lu.ndim > 2 or b.ndim > 2 or piv.ndim > 1)
        or not (_ok(lu, 2) and _ok(b, 1) and _ok(piv, 1, _I4))
    ):
        return NotImplemented
    if (
        trans.__class__ is not int
        or trans not in (0, 1, 2)
        or overwrite_b is not False
        or check_finite.__class__ is not bool
    ):
        return NotImplemented
    n = lu.shape[-1]
    bc = _rhs(b)
    if n == 0 or lu.shape[-2] != n or piv.shape[-1] != n or b.shape[b.ndim - bc] != n or b.size == 0:
        return NotImplemented
    fl = _flatten([(lu, 2), (piv, 1), (b, bc)])
    if fl is None:
        return NotImplemented
    batch, B, (luv, pv, bv) = fl
    k = bv.shape[2] if bc == 2 else 1
    x = _native.linalg.getrs(luv, pv, bv if bc == 2 else bv[:, :, None], "NTC"[trans], check_finite, _threads(B, n, k))
    return NotImplemented if x is None else _solution(x, batch, n, bc, k)


def _cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    if c_and_lower.__class__ not in (tuple, list) or len(c_and_lower) != 2:
        return NotImplemented
    c, lower = c_and_lower
    if (
        c.__class__ is not _ND
        or b.__class__ is not _ND
        or not (c.ndim > 2 or b.ndim > 2)
        or not (_ok(c, 2) and _ok(b, 1))
        or overwrite_b is not False
        or check_finite.__class__ is not bool
    ):
        return NotImplemented
    if lower.__class__ is not bool:  # cho_factor of a stack returns one flag per matrix
        if (
            lower.__class__ is not np.ndarray
            or lower.dtype != np.bool_
            or lower.size == 0
            or lower.any() != lower.all()
        ):
            return NotImplemented
        lower = bool(lower.flat[0])
    if (s := _solve_shapes(c, b)) is None:
        return NotImplemented
    n, bc, batch, B, cv, bv = s
    k = bv.shape[2]
    x = _native.linalg.potrs(cv, bv, "L" if lower else "U", check_finite, _threads(B, n, k))
    return NotImplemented if x is None else _solution(x, batch, n, bc, k)


def _single_triangular(a, b, trans, lower, unit_diagonal, overwrite_b, check_finite):
    """Validated NumPy matrices can go straight to SciPy's LAPACK dispatcher."""
    if (
        a.shape[0] != a.shape[1]
        or a.shape[0] != b.shape[0]
        or b.size == 0
        or overwrite_b is not False
        or type(lower) is not bool
        or type(unit_diagonal) is not bool
        or type(check_finite) is not bool
        or not all(
            (x.dtype.kind == "f" and x.dtype.itemsize in (4, 8))
            or (x.dtype.kind == "c" and x.dtype.itemsize in (8, 16))
            for x in (a, b)
        )
    ):
        return NotImplemented
    if type(trans) is str:
        if trans not in ("N", "T", "C"):
            return NotImplemented
    elif type(trans) is not int or trans not in (0, 1, 2):
        return NotImplemented
    if check_finite and not (np.isfinite(a).all() and np.isfinite(b).all()):
        return NotImplemented
    try:
        return _triangular_core(a, b, trans, lower, unit_diagonal, False)[0]
    except np.linalg.LinAlgError:
        # With overwrite disabled no inputs changed; let SciPy own singular
        # system errors, just as the batched kernels decline nonzero info.
        return NotImplemented


def _solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, check_finite=True):
    if a.__class__ is not _ND or b.__class__ is not _ND:
        return NotImplemented
    if a.ndim == 2 and b.ndim in (1, 2):
        return _single_triangular(a, b, trans, lower, unit_diagonal, overwrite_b, check_finite)
    if (
        not (a.ndim > 2 or b.ndim > 2)
        or not (_ok(a, 2) and _ok(b, 1))
        or overwrite_b is not False
        or check_finite.__class__ is not bool
        or lower.__class__ is not bool
        or unit_diagonal.__class__ is not bool
    ):
        return NotImplemented
    if trans.__class__ is str:
        trans = {"N": 0, "T": 1, "C": 2}.get(trans)
    if trans.__class__ is not int or trans not in (0, 1, 2) or (s := _solve_shapes(a, b)) is None:
        return NotImplemented
    n, bc, batch, B, av, bv = s
    # SciPy hands LAPACK the slice itself when it is F-contiguous (or trans == 2); otherwise it
    # solves the transposed system on the C-ordered memory with uplo and trans flipped.
    f_contig = n == 1 or (a.strides[-2] == 8 and a.strides[-1] == 8 * n)
    tin = not (f_contig or trans == 2)
    uplo = lower != tin
    tr = ("N" if trans else "T") if tin else "NTC"[trans]
    k = bv.shape[2]
    x = _native.linalg.trtrs(
        av, bv, "L" if uplo else "U", tr, "U" if unit_diagonal else "N", tin, check_finite, _threads(B, n, k)
    )
    return NotImplemented if x is None else _solution(x, batch, n, bc, k)


def _syevr_lwork(n, lower):
    if (key := (n, lower)) not in _lwork_cache:
        from scipy.linalg.lapack import _compute_lwork, get_lapack_funcs

        _, lw = get_lapack_funcs(("syevr", "syevr_lwork"), [np.zeros((1, 1))])
        _lwork_cache[key] = _compute_lwork(lw, n=n, lower=lower)
    return _lwork_cache[key]


def _eigh_fast(
    a, b, lower, eigvals_only, overwrite_a, overwrite_b, type, check_finite, subset_by_index, subset_by_value, driver
):
    if (
        a.__class__ is not _ND
        or a.ndim < 3
        or not _ok(a, 3)
        or b is not None
        or overwrite_a is not False
        or overwrite_b is not False
        or type != 1
        or subset_by_index is not None
        or subset_by_value is not None
        or driver not in (None, "evr")
        or lower.__class__ is not bool
        or eigvals_only.__class__ is not bool
        or check_finite.__class__ is not bool
    ):
        return NotImplemented
    n = a.shape[-1]
    if a.shape[-2] != n or n == 0 or (fl := _flatten([(a, 2)])) is None:
        return NotImplemented
    batch, B, (av,) = fl
    lwork, liwork = _syevr_lwork(n, lower)
    r = _native.linalg.syevr(
        av, "L" if lower else "U", not eigvals_only, lwork, liwork, check_finite, _threads(B, n, 1, 10.0)
    )
    if r is None:
        return NotImplemented
    w = r[0].reshape(batch + (n,))
    return w if eigvals_only else [w, _fview(r[1], batch, n, n)]


def _eigh(
    a,
    b=None,
    *,
    lower=True,
    eigvals_only=False,
    overwrite_a=False,
    overwrite_b=False,
    type=1,
    check_finite=True,
    subset_by_index=None,
    subset_by_value=None,
    driver=None,
):
    return _eigh_fast(
        a,
        b,
        lower,
        eigvals_only,
        overwrite_a,
        overwrite_b,
        type,
        check_finite,
        subset_by_index,
        subset_by_value,
        driver,
    )


def _eigvalsh(
    a,
    b=None,
    *,
    lower=True,
    overwrite_a=False,
    overwrite_b=False,
    type=1,
    check_finite=True,
    subset_by_index=None,
    subset_by_value=None,
    driver=None,
):
    return _eigh_fast(
        a, b, lower, True, overwrite_a, overwrite_b, type, check_finite, subset_by_index, subset_by_value, driver
    )


def _null_space(A, rcond=None, *, overwrite_a=False, check_finite=True, lapack_driver="gesdd"):
    """Tall input: ``u`` is never used, so skip the (M, M) one (pure Python, SciPy's own svd)."""
    if (
        A.__class__ is not np.ndarray
        or A.ndim != 2
        or A.dtype.kind not in "fc"
        or A.shape[0] <= A.shape[1]
        or A.shape[1] == 0
    ):
        return NotImplemented
    from scipy.linalg._decomp_svd import svd

    _, s, vh = svd(
        A, full_matrices=False, overwrite_a=overwrite_a, check_finite=check_finite, lapack_driver=lapack_driver
    )
    M, N = A.shape
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    num = np.sum(s > np.amax(s, initial=0.0) * rcond, dtype=int)
    return vh[num:, :].T.conj()


def patch():
    global _triangular_core
    _patch.require_validated("1.18.")
    import scipy.linalg._basic as basic
    import scipy.linalg._decomp as decomp
    import scipy.linalg._decomp_cholesky as cholesky
    import scipy.linalg._decomp_lu as lu
    import scipy.linalg._decomp_svd as svdmod

    _triangular_core = basic._solve_triangular
    _native.linalg.init()
    for mod, name, fast in (
        (lu, "lu_factor", _lu_factor),
        (lu, "lu_solve", _lu_solve),
        (cholesky, "cho_solve", _cho_solve),
        (basic, "solve_triangular", _solve_triangular),
        (decomp, "eigh", _eigh),
        (decomp, "eigvalsh", _eigvalsh),
        (svdmod, "null_space", _null_space),
    ):
        old = getattr(mod, name)
        if name == "cho_solve":
            wrapper = _cho_solver(old, cholesky._cho_solve.__wrapped__)
        elif name == "lu_solve":
            wrapper = _lu_solver(old, lu._lu_solve.__wrapped__)
        else:
            wrapper = _patch.wrap(old, fast, name)
        _patch.rebind(old, wrapper)


def smoke():
    import scipy.linalg as sl

    rng = np.random.default_rng(0)
    a = rng.random((4, 3, 3)) + 3 * np.eye(3)
    s = a + a.transpose(0, 2, 1)
    b = rng.random((4, 3, 2))
    lu = sl.lu_factor(a)
    sl.lu_solve(lu, b)
    sl.lu_solve((lu[0][0], lu[1][0]), b[0])
    sl.solve_triangular(a, b)
    sl.solve_triangular(a[0], b[0])
    sl.cho_solve(sl.cho_factor(s + 6 * np.eye(3)), b)
    sl.cho_solve(sl.cho_factor(s[0] + 6 * np.eye(3)), b[0])
    sl.eigh(s)
    sl.eigvalsh(s)
    sl.null_space(rng.random((6, 3)))
