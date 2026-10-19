import warnings

import numpy as np
import pytest
import scipy.linalg as sl

import fastscipy
from fastscipy.testing import assert_identical, check, fast_calls

SIZES = [1, 2, 3, 8, 32]
BATCHES = [(5,), (2, 3), (700,)]
THREADS = [1, 4]


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
