import warnings
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg as sl

import fastscipy
from fastscipy.testing import assert_identical, check, fast_calls

SIZES = [1, 2, 3, 8, 32]
BATCHES = [(5,), (2, 3), (700,)]
THREADS = [1, 4]


@pytest.mark.parametrize(
    "routine",
    ["dgetrf", "dgetrs", "dpotrs", "dpbtrf", "dpbtrs", "dgtsv", "dgbsv", "dptsv", "dpbsv", "dtrtrs", "dsyevr"],
)
def test_lapack_capsule_abi_rejected_in_fresh_process(routine):
    script = r"""
import ctypes
import sys
import fastscipy
from fastscipy import _native
from scipy.linalg import cython_lapack

capi = cython_lapack.__pyx_capi__
name = sys.argv[1]
original = capi[name]
get_name = ctypes.pythonapi.PyCapsule_GetName
get_name.argtypes, get_name.restype = [ctypes.py_object], ctypes.c_char_p
get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
get_pointer.argtypes, get_pointer.restype = [ctypes.py_object, ctypes.c_char_p], ctypes.c_void_p
new_capsule = ctypes.pythonapi.PyCapsule_New
new_capsule.argtypes, new_capsule.restype = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p], ctypes.py_object
# The pointer is real; only its advertised ABI is changed. Keep both the
# original capsule and the replacement name bytes alive until process exit.
wrong_name = get_name(original).replace(b"int *", b"int64_t *", 1)
assert wrong_name != get_name(original)
capi[name] = new_capsule(get_pointer(original, get_name(original)), wrong_name, None)
try:
    _native.linalg.init()
except ValueError as error:
    assert "incorrect name" in str(error)
else:
    raise AssertionError("unsupported LAPACK ABI accepted")
assert fastscipy.patch(["linalg"])["linalg"].startswith("off:")
capi[name] = original
assert _native.linalg.init() is None
"""
    result = subprocess.run([sys.executable, "-c", script, routine], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(params=THREADS, autouse=True)
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield request.param
    fastscipy.set_num_threads(old)


def mats(rng, batch, n, spd=False):
    a = rng.standard_normal(batch + (n, n)) + 4 * np.eye(n)
    return a @ a.swapaxes(-1, -2) + n * np.eye(n) if spd else a


def declined(call):
    """``call`` gives SciPy's result (or raises SciPy's exception) without a fast path."""

    def run():
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                return call(), [str(x.message) for x in w]
        except Exception as e:
            return type(e), str(e)

    before = fast_calls()
    with fastscipy.disabled():
        expected = run()
    got = run()
    assert fast_calls() == before
    assert_identical(expected, got)


def rhs(rng, batch, n):
    """Stacked matrix, matrix broadcast over the stack, and a single vector."""
    return [rng.standard_normal(batch + (n, 3)), rng.standard_normal((n, 2)), rng.standard_normal(n)]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("trans", [0, 1, 2])
def test_lu(n, batch, trans):
    rng = np.random.default_rng(n)
    a = mats(rng, batch, n)
    lu = check(lambda: sl.lu_factor(a))
    bs = rhs(rng, batch, n)
    for b in bs:
        check(lambda: sl.lu_solve(lu, b, trans=trans))
    check(lambda: sl.lu_solve((lu[0], lu[1]), bs[0], check_finite=False))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("lower", [False, True])
def test_cho_solve(n, batch, lower):
    rng = np.random.default_rng(n)
    c = sl.cho_factor(mats(rng, batch, n, spd=True), lower=lower)
    for b in rhs(rng, batch, n):
        check(lambda: sl.cho_solve(c, b))
    check(lambda: sl.cho_solve((c[0], lower), b))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("batch", [(5,), (2, 3)])
@pytest.mark.parametrize("trans", [0, 1, 2, "N", "T", "C"])
@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("unit", [False, True])
@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_solve_triangular(n, batch, trans, lower, unit, order):
    rng = np.random.default_rng(n)
    a = (np.tril if lower else np.triu)(mats(rng, batch, n))
    if order == "F":  # each matrix Fortran-ordered inside the stack
        a = a.swapaxes(-1, -2).copy().swapaxes(-1, -2)
    elif order == "strided":
        big = np.zeros(batch + (2 * n, 2 * n))
        big[..., ::2, ::2] = a
        a = big[..., ::2, ::2]
    for b in rhs(rng, batch, n):
        check(lambda: sl.solve_triangular(a, b, trans=trans, lower=lower, unit_diagonal=unit))


def test_solve_triangular_broadcast_and_negative_strides():
    rng = np.random.default_rng(0)
    a = np.triu(mats(rng, (4,), 5))
    b, b2 = rng.standard_normal((4, 5, 2)), rng.standard_normal((5, 2))
    check(lambda: sl.solve_triangular(a[:, None], b[:, None]))
    check(lambda: sl.solve_triangular(a[::-1, ::-1, ::-1], b))
    check(lambda: sl.solve_triangular(a, b2))


@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("readonly", [False, True])
def test_small_stacks_preserve_inputs(broadcast, readonly):
    rng = np.random.default_rng(49)
    a = mats(rng, (2, 3), 3)
    b = rng.standard_normal((3, 3, 2) if broadcast else (2, 3, 3, 2))
    a, b = a[::-1], b[..., ::-1]
    if readonly:
        a.flags.writeable = b.flags.writeable = False
    before = a.tobytes(), b.tobytes()
    lu = check(lambda: sl.lu_factor(a))
    check(lambda: sl.lu_solve(lu, b))
    check(lambda: sl.solve_triangular(a, b))
    spd = a @ a.swapaxes(-1, -2) + 3 * np.eye(3)
    check(lambda: sl.cho_solve(sl.cho_factor(spd), b))
    check(lambda: sl.eigh(spd))
    assert (a.tobytes(), b.tobytes()) == before


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("lower", [True, False])
def test_eigh(n, batch, lower):
    rng = np.random.default_rng(n)
    a = mats(rng, batch, n)
    check(lambda: sl.eigh(a, lower=lower))
    check(lambda: sl.eigh(a, lower=lower, eigvals_only=True))
    check(lambda: sl.eigvalsh(a, lower=lower))
    check(lambda: sl.eigh(a, driver="evr", check_finite=False))
    check(lambda: sl.eigh(a.astype(np.float64)[..., ::-1, :]))


def test_threaded_large_batch():
    rng = np.random.default_rng(3)
    a = mats(rng, (20000,), 3)
    lu, b = check(lambda: sl.lu_factor(a)), rng.standard_normal((20000, 3, 2))
    check(lambda: sl.lu_solve(lu, b))
    check(lambda: sl.eigh(a))


def test_declined_cases_match_scipy():
    rng = np.random.default_rng(0)
    a = mats(rng, (6,), 4)
    b = rng.standard_normal((6, 4, 2))
    sing = a.copy()
    sing[2] = 0  # singular: SciPy warns or raises
    nan = a.copy()
    nan[1, 0, 0] = np.nan
    declined(lambda: sl.lu_factor(sing))
    lu = sl.lu_factor(a)
    declined(lambda: sl.lu_solve(lu, b, overwrite_b=True))
    declined(lambda: sl.lu_factor(a.astype(np.float32)))
    declined(lambda: sl.lu_factor(a, overwrite_a=True))
    declined(lambda: sl.lu_factor(a[0]))
    declined(lambda: sl.lu_factor(np.zeros((3, 0, 0))))
    declined(lambda: sl.eigh(a, subset_by_index=[0, 1]))
    declined(lambda: sl.eigh(a, a))
    declined(lambda: sl.eigh(a, driver="evd"))
    declined(lambda: sl.eigh(a.astype(complex)))
    declined(lambda: sl.solve_triangular(sing, b, overwrite_b=True))
    declined(lambda: sl.solve_triangular(np.triu(sing), b))
    check(lambda: sl.cho_solve((a, True), b))  # potrs does not check positive definiteness
    for f in (lambda: sl.lu_factor(nan), lambda: sl.eigh(nan), lambda: sl.solve_triangular(nan, b)):
        with fastscipy.disabled():
            with pytest.raises(ValueError) as e0:
                f()
        with pytest.raises(ValueError) as e1:
            f()
        assert str(e0.value) == str(e1.value)
    check(lambda: sl.lu_factor(nan, check_finite=False), fast=False)
    with pytest.raises(ValueError):
        sl.lu_solve(lu, np.ones((5, 4, 2)))  # incompatible batch: SciPy's error
    from fastscipy import _native  # SciPy itself would run LAPACK on this; only check the kernel declines

    assert _native.linalg.getrs(lu[0], np.full((6, 4), 9, np.int32), b, "N", True, 1) is None


def test_singular_warning_comes_from_scipy():
    a = np.stack([np.eye(3), np.zeros((3, 3))])
    with fastscipy.disabled(), pytest.warns(sl.LinAlgWarning) as w0:
        sl.lu_factor(a)
    with pytest.warns(sl.LinAlgWarning) as w1:
        sl.lu_factor(a)
    assert [str(x.message) for x in w0] == [str(x.message) for x in w1]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(50, 3), (200, 20), (30, 29)])
@pytest.mark.parametrize("driver", ["gesdd", "gesvd"])
def test_null_space_tall(dtype, shape, driver):
    rng = np.random.default_rng(0)
    A = rng.standard_normal(shape).astype(dtype)
    A[:, -2:] = A[:, :2]  # rank deficient
    check(lambda: sl.null_space(A, lapack_driver=driver))
    check(lambda: sl.null_space(A, rcond=1e-3, check_finite=False))


def test_null_space_declines():
    rng = np.random.default_rng(0)
    for A in (
        rng.standard_normal((4, 9)),
        rng.standard_normal((6, 6)),
        rng.standard_normal((3, 5, 2)),
        rng.standard_normal((8, 0)),
        rng.integers(0, 3, (8, 3)),
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
    ):
        declined(lambda: sl.null_space(A))


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("nrhs", [None, 1, 3])
def test_single_triangular_dtypes_and_layout(dtype, layout, nrhs):
    rng = np.random.default_rng(320)
    a = rng.standard_normal((5, 5)) + 5 * np.eye(5)
    b = rng.standard_normal((5,) if nrhs is None else (5, nrhs))
    if np.dtype(dtype).kind == "c":
        a = a + 1j * rng.standard_normal(a.shape)
        b = b + 1j * rng.standard_normal(b.shape)
    a, b = a.astype(dtype), b.astype(dtype)
    if layout == "F":
        a, b = np.asfortranarray(a), np.asfortranarray(b)
    elif layout == "strided":
        aa = np.zeros((10, 10), dtype=dtype)
        aa[::2, ::2] = a
        a, b = aa[::2, ::2], b[::-1]
    a.flags.writeable = b.flags.writeable = False
    before = a.tobytes(), b.tobytes()
    for lower in (False, True):
        for trans in (0, 1, 2, "N", "T", "C"):
            for unit in (False, True):
                for finite in (False, True):
                    check(
                        lambda: sl.solve_triangular(
                            a, b, trans=trans, lower=lower, unit_diagonal=unit, check_finite=finite
                        )
                    )
    assert (a.tobytes(), b.tobytes()) == before


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("dtypes", [(np.float32, np.float64), (np.float64, np.complex64), (np.complex128, np.float32)])
def test_single_triangular_mixed_dtypes(dtypes):
    rng = np.random.default_rng(321)
    a = (rng.standard_normal((5, 5)) + 5 * np.eye(5)).astype(dtypes[0])
    b = rng.standard_normal((5, 3)).astype(dtypes[1])
    check(lambda: sl.solve_triangular(a, b))


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
def test_single_triangular_declined_errors():
    a, b = np.eye(3), np.ones(3)
    for aa, bb, options in (
        (a.astype(int), b, {}),
        (a, b.astype(int), {}),
        (a[:2], b, {}),
        (a, b[:2], {}),
        (a, b, {"trans": "invalid"}),
        (a, b, {"overwrite_b": True}),
        (a, b, {"lower": np.bool_(True)}),
        (np.full((3, 3), np.nan), b, {}),
        (a, np.full(3, np.inf), {}),
    ):
        declined(lambda: sl.solve_triangular(aa, bb, **options))
    for aa, bb in ((np.eye(0), np.empty(0)), (a, np.empty((3, 0)))):
        check(lambda: sl.solve_triangular(aa, bb), fast=False)
    declined(lambda: sl.solve_triangular(a.astype(np.float16), b.astype(np.float16)))
    # The same SciPy dispatcher owns singular-system errors.
    check(lambda: sl.solve_triangular(np.zeros((3, 3)), b), fast=False)


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
def test_single_triangular_unchecked_nonfinite():
    a, b = np.eye(3), np.ones(3)
    a[0, 2] = np.nan
    check(lambda: sl.solve_triangular(a, b, check_finite=False))


@pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble])
@pytest.mark.parametrize("operand", ["a", "b", "both"])
def test_single_triangular_extended_precision_warning(dtype, operand):
    # On Apple Silicon longdouble has eight bytes but remains dtype 'g'.
    a, b = np.eye(3), np.ones(3)
    if operand in ("a", "both"):
        a = a.astype(dtype)
    if operand in ("b", "both"):
        b = b.astype(dtype)
    with fastscipy.disabled(), pytest.warns(DeprecationWarning) as expected_warnings:
        expected = sl.solve_triangular(a, b)
    before = fast_calls()
    with pytest.warns(DeprecationWarning) as actual_warnings:
        actual = sl.solve_triangular(a, b)
    assert fast_calls() == before
    assert_identical(expected, actual)
    assert [str(w.message) for w in actual_warnings] == [str(w.message) for w in expected_warnings]


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("nrhs", [None, 1, 3])
def test_single_factor_solves(dtype, layout, nrhs):
    rng = np.random.default_rng(352)
    a = rng.standard_normal((5, 5)) + 5 * np.eye(5)
    b = rng.standard_normal((5,) if nrhs is None else (5, nrhs))
    if np.dtype(dtype).kind == "c":
        a = a + 1j * rng.standard_normal(a.shape)
        b = b + 1j * rng.standard_normal(b.shape)
    a, b = a.astype(dtype), b.astype(dtype)
    factors = ((sl.lu_solve, sl.lu_factor(a)), (sl.cho_solve, sl.cho_factor(a @ a.conj().T)))
    for solver, (factor, option) in factors:
        if layout == "C":
            factor = factor.copy(order="C")
        elif layout == "strided":
            tmp = np.zeros((10, 10), dtype=dtype)
            tmp[::2, ::2] = factor
            factor = tmp[::2, ::2]
            b = b[::-1]
        factor.flags.writeable = b.flags.writeable = False
        before = factor.tobytes(), b.tobytes()
        for finite in (True, False):
            for trans in (0, 1, 2) if solver is sl.lu_solve else (0,):
                opts = {"trans": trans} if solver is sl.lu_solve else {}
                check(lambda: solver((factor, option), b, check_finite=finite, **opts))
        assert (factor.tobytes(), b.tobytes()) == before


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("lower", [False, True, np.bool_(False), np.array(True)])
def test_single_cholesky_flag_types(lower):
    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    factor = sl.cho_factor(a, lower=bool(lower))[0]
    check(lambda: sl.cho_solve((factor, lower), np.ones(2)))


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("solver", ["lu_solve", "cho_solve"])
@pytest.mark.parametrize("dtypes", [(np.float32, np.float64), (np.float64, np.complex64), (np.complex128, np.float32)])
def test_single_factor_solve_mixed_dtypes(solver, dtypes):
    solver = getattr(sl, solver)
    a = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=dtypes[0])
    factor = sl.lu_factor(a) if solver is sl.lu_solve else sl.cho_factor(a)
    check(lambda: solver(factor, np.ones(2, dtype=dtypes[1])))


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("solver", ["lu_solve", "cho_solve"])
def test_single_factor_solve_declined_errors_and_warnings(solver):
    solver = getattr(sl, solver)
    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    factor = sl.lu_factor(a) if solver is sl.lu_solve else sl.cho_factor(a)
    b = np.ones(2)
    for bad in ((), (factor[0],), (*factor, 1)):
        check(lambda: solver(bad, b), fast=False)
    check(lambda: solver(factor, b, overwrite_b=True), fast=False)
    check(lambda: solver(factor, np.ones(3)), fast=False)
    check(lambda: solver(factor, np.full(2, np.nan)), fast=False)
    check(lambda: solver(factor, np.empty((2, 0))), fast=False)
    declined(lambda: solver((factor[0].astype(np.float16), factor[1]), b.astype(np.float16)))


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
def test_single_lu_checks_only_rhs_finiteness():
    lu, piv = sl.lu_factor(np.array([[4.0, 1.0], [1.0, 3.0]]))
    lu[0, 0] = np.nan
    check(lambda: sl.lu_solve((lu, piv), np.ones(2), check_finite=True))


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("solver", ["lu_solve", "cho_solve"])
@pytest.mark.parametrize("mode", ["call", "log"])
def test_single_factor_cast_handler_is_not_retried(solver, mode, monkeypatch):
    from fastscipy import _patch

    monkeypatch.setattr(_patch, "STRICT", False)
    solver = getattr(sl, solver)
    factor = np.eye(3, dtype=np.float32)
    factor.view(np.uint32)[0, 0] = 0x7F800001  # Signaling NaN; promoted to float64 by LAPACK dispatch.
    option = np.arange(3, dtype=np.int32) if solver is sl.lu_solve else False
    calls = []

    def handler(*args):
        calls.append(args)
        raise RuntimeError("factor cast handler")

    class Logger:
        write = staticmethod(handler)

    old = np.seterrcall(handler if mode == "call" else Logger())
    try:
        with np.errstate(invalid=mode):
            with pytest.raises(RuntimeError, match="factor cast handler"):
                solver((factor, option), np.ones(3), check_finite=solver is sl.lu_solve)
    finally:
        np.seterrcall(old)
    assert len(calls) == 1


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
def test_single_lu_bad_pivots_decline_before_lapack():
    from fastscipy import _linalg

    calls = []

    def fallback(*args, **kwargs):
        calls.append("fallback")
        return "declined"

    def core(*args):
        raise AssertionError("invalid pivots must not reach LAPACK")

    solve = _linalg._lu_solver(fallback, core)
    for piv in (
        np.array([-1, 1, 2], np.int32),
        np.array([0, 1, 3], np.int32),
        np.array([0, 1], np.int32),
        np.arange(3, dtype=np.int64),
    ):
        assert solve((np.eye(3), piv), np.ones(3)) == "declined"
    assert calls == ["fallback"] * 4


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
def test_single_factor_public_keyword_arguments():
    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    lu, cho, b = sl.lu_factor(a), sl.cho_factor(a), np.ones(2)
    check(lambda: sl.lu_solve(lu_and_piv=lu, b=b, trans=1))
    check(lambda: sl.cho_solve(c_and_lower=cho, b=b))


def bands(rng, batch, rows, n, lower):
    # Strict diagonal dominance gives positive definite symmetric bands.
    a = rng.uniform(-0.2, 0.2, batch + (rows, n))
    a[..., 0 if lower else -1, :] = 2 * rows
    return a


@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("rows,n", [(1, 1), (1, 8), (2, 8), (4, 32), (10, 3)])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "broadcast"])
def test_banded_cholesky(lower, rows, n, layout):
    rng = np.random.default_rng(843)
    a = bands(rng, (2, 3), rows, n, lower)
    if layout == "F":
        a = np.asfortranarray(a)
    elif layout == "strided":
        a = a[::-1, ::-1, :, ::-1]
    elif layout == "broadcast":
        a = np.broadcast_to(a[0, 0], a.shape)
    before = (a.strides, a.tobytes())
    for finite in (False, True):
        c = check(lambda: sl.cholesky_banded(a, lower=lower, check_finite=finite))
        original_c = (c.strides, c.tobytes())
        for b in rhs(rng, (2, 3), n):
            original_b = (b.strides, b.tobytes())
            check(lambda: sl.cho_solve_banded((c, lower), b, check_finite=finite))
            assert (b.strides, b.tobytes()) == original_b
        assert (c.strides, c.tobytes()) == original_c
        b = rng.normal(size=(2, 3, n, 2))[::-1, :, ::-1, :]
        b.flags.writeable = False
        check(lambda: sl.cho_solve_banded((c[0, 0], lower), b, check_finite=finite))
    assert (a.strides, a.tobytes()) == before


@pytest.mark.parametrize("lower", [False, True])
def test_banded_cholesky_declines(lower):
    rng = np.random.default_rng(567)
    a = bands(rng, (3,), 2, 8, lower)
    for unsupported in (a.astype(np.float32), a.astype(complex), a[:, :, :0], a[0]):
        declined(lambda: sl.cholesky_banded(unsupported, lower=lower))
    bad = a.copy()
    bad[1, 0 if lower else -1, 0] = -1
    declined(lambda: sl.cholesky_banded(bad, lower=lower))
    bad = a.copy()
    # Finite validation must inspect unused band padding as well.
    bad[1, -1 if lower else 0, -1 if lower else 0] = np.nan
    declined(lambda: sl.cholesky_banded(bad, lower=lower))
    check(lambda: sl.cholesky_banded(bad, lower=lower, check_finite=False))
    c = sl.cholesky_banded(a, lower=lower)
    declined(lambda: sl.cho_solve_banded((c, lower), np.ones(7)))
    declined(lambda: sl.cho_solve_banded((c, lower), np.full(8, np.inf)))
    declined(lambda: sl.cho_solve_banded((bad, lower), np.ones(8)))
    declined(lambda: sl.cho_solve_banded((c, lower), np.ones((4, 8, 2))))


def test_banded_broadcast_and_overwrite_fallbacks():
    rng = np.random.default_rng(47)
    a = bands(rng, (2, 1), 3, 12, True)
    c = sl.cholesky_banded(a, lower=True)
    b = rng.normal(size=(1, 5, 12, 2))
    check(lambda: sl.cho_solve_banded((c, True), b))
    declined(lambda: sl.cholesky_banded(a.copy(), overwrite_ab=True, lower=True))
    declined(lambda: sl.cho_solve_banded((c, True), b.copy(), overwrite_b=True))
    declined(lambda: sl.cholesky_banded(a[:0], lower=True))
    declined(lambda: sl.cholesky_banded(a[..., :0, :], lower=True))


@pytest.mark.parametrize("lower,finite", [(1, True), (np.bool_(True), True), (True, 1)])
def test_banded_nonbool_options_decline(lower, finite):
    rng = np.random.default_rng(911)
    a = bands(rng, (3,), 2, 8, True)
    c = sl.cholesky_banded(a, lower=True)
    declined(lambda: sl.cholesky_banded(a, lower=lower, check_finite=finite))
    declined(lambda: sl.cho_solve_banded((c, lower), np.ones(8), check_finite=finite))


@pytest.mark.parametrize("operand", ["band", "factor", "rhs"])
def test_banded_unaligned_declines(operand):
    rng = np.random.default_rng(516)
    a = bands(rng, (3,), 2, 8, True)
    c = sl.cholesky_banded(a, lower=True)
    b = np.ones((3, 8, 2))
    original = {"band": a, "factor": c, "rhs": b}[operand]
    buffer = bytearray(original.nbytes + 1)
    unaligned = np.ndarray(original.shape, dtype=np.float64, buffer=buffer, offset=1)
    unaligned[...] = original
    assert not unaligned.flags.aligned
    if operand == "band":
        declined(lambda: sl.cholesky_banded(unaligned, lower=True))
    elif operand == "factor":
        declined(lambda: sl.cho_solve_banded((unaligned, True), b))
    else:
        declined(lambda: sl.cho_solve_banded((c, True), unaligned))


def test_batched_lu_unaligned_declines():
    rng = np.random.default_rng(923)
    a = mats(rng, (3,), 4)
    buffer = bytearray(a.nbytes + 1)
    unaligned = np.ndarray(a.shape, dtype=np.float64, buffer=buffer, offset=1)
    unaligned[...] = a
    declined(lambda: sl.lu_factor(unaligned))
    lu, piv = sl.lu_factor(a)
    unaligned[...] = lu
    declined(lambda: sl.lu_solve((unaligned, piv), np.ones(4)))
    buffer_b = bytearray(3 * 4 * 2 * 8 + 1)
    unaligned_b = np.ndarray((3, 4, 2), dtype=np.float64, buffer=buffer_b, offset=1)
    unaligned_b[...] = 1
    declined(lambda: sl.lu_solve((lu, piv), unaligned_b))


@pytest.mark.parametrize(
    "positive,kl,ku",
    [(False, 1, 1), (False, 2, 1), (False, 0, 2), (False, 0, 0), (True, 1, 0), (True, 3, 0), (True, 0, 0)],
)
@pytest.mark.parametrize("layout", ["C", "F", "strided", "broadcast"])
@pytest.mark.parametrize("n", [2, 12])
def test_direct_banded_solvers(positive, kl, ku, layout, n):
    rng = np.random.default_rng(932)
    rows = kl + ku + 1
    a = rng.uniform(-0.2, 0.2, (2, 3, rows, n))
    a[..., 0 if positive else ku, :] = 2 * rows
    if layout == "F":
        a = np.asfortranarray(a)
    elif layout == "strided":
        a = a[::-1, ::-1, :, ::-1]
    elif layout == "broadcast":
        a = np.broadcast_to(a[0, 0], a.shape)
    for lower in [True, False] if positive else [False]:
        if positive and not lower:
            a = a[..., ::-1, ::-1]
        original_a = a.tobytes()
        for finite in [True, False]:
            for b in rhs(rng, (2, 3), n):
                original_b = b.tobytes()
                if positive:
                    check(lambda: sl.solveh_banded(a, b, lower=lower, check_finite=finite))
                else:
                    check(lambda: sl.solve_banded((kl, ku), a, b, check_finite=finite))
                assert b.tobytes() == original_b
            b = rng.normal(size=(2, 3, n, 2))
            if positive:
                check(lambda: sl.solveh_banded(a[0, 0], b, lower=lower, check_finite=finite))
            else:
                check(lambda: sl.solve_banded((kl, ku), a[0, 0], b, check_finite=finite))
        assert a.tobytes() == original_a


@pytest.mark.parametrize("positive", [True, False])
def test_direct_banded_solver_fallbacks(positive):
    a = np.ones((4, 3, 8))
    a[:, 1, :] = 8
    b = np.ones((4, 8, 2))
    solve = (
        (lambda a, b, **kw: sl.solveh_banded(a, b, **kw))
        if positive
        else (lambda a, b, **kw: sl.solve_banded((1, 1), a, b, **kw))
    )
    for aa, bb in [
        (a.astype(np.float32), b.astype(np.float32)),
        (a.astype(complex), b),
        (a[:0], b[:0]),
        (a[..., :1], b[:, :1]),
        (a[0], b[0]),
        (a, b[:, :7]),
    ]:
        declined(lambda: solve(aa, bb))
    for kw in [dict(overwrite_ab=True), dict(overwrite_b=True), dict(check_finite=1)]:
        declined(lambda: solve(a.copy(), b.copy(), **kw))
    bad = a.copy()
    bad[0, 0, 0] = np.nan
    declined(lambda: solve(bad, b))
    bad = np.zeros_like(a)
    declined(lambda: solve(bad, b))


def test_banded_gtsv_pivoting():
    a = np.ones((4, 3, 8))
    a[:, 1, :] = 0
    b = np.arange(64.0).reshape(4, 8, 2)
    check(lambda: sl.solve_banded((1, 1), a, b))


@pytest.mark.parametrize("positive", [False, True])
def test_direct_banded_dense_residual_and_padding(positive):
    rng = np.random.default_rng(673)
    n = 9
    a = rng.uniform(-0.3, 0.3, (3, 3, n))
    a[:, 2 if positive else 1, :] = 9
    b = rng.normal(size=(3, n, 2))
    x = sl.solveh_banded(a, b) if positive else sl.solve_banded((1, 1), a, b)
    dense = np.zeros((3, n, n))
    for i in range(n):
        for j in range(n):
            if positive and 0 <= 2 + i - j < 3:
                dense[:, i, j] = a[:, 2 + i - j, j]
                dense[:, j, i] = dense[:, i, j]
            elif not positive and abs(i - j) <= 1:
                dense[:, i, j] = a[:, 1 + i - j, j]
    np.testing.assert_allclose(dense @ x, b, rtol=2e-14, atol=2e-14)
    # Finite validation includes unused padding, even for specialized drivers.
    a[:, 0, 0] = np.inf
    if positive:
        declined(lambda: sl.solveh_banded(a, b))
        check(lambda: sl.solveh_banded(a, b, check_finite=False))
    else:
        declined(lambda: sl.solve_banded((1, 1), a, b))
        check(lambda: sl.solve_banded((1, 1), a, b, check_finite=False))
