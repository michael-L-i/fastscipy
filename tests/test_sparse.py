import numpy as np
import pytest
import scipy.sparse as sp

import fastscipy
from fastscipy import _sparse
from fastscipy.testing import assert_identical, check, fast_calls

VALUE_DTYPES = [np.float32, np.float64, np.int32, np.int64]
DECLINED_DTYPES = [np.bool_, np.int16, np.uint8, np.longdouble]
FORMATS = [sp.csr_array, sp.csc_array, sp.coo_array, sp.csr_matrix, sp.csc_matrix, sp.coo_matrix]


@pytest.fixture(params=["1 thread", "4 threads, no size thresholds"], autouse=True)
def threads(request, monkeypatch):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(1 if request.param[0] == "1" else 4)
    if request.param[0] == "4":
        monkeypatch.setattr(_sparse, "_PAR", dict.fromkeys(_sparse._PAR, 0))
    yield
    fastscipy.set_num_threads(old)


def dense(rng, shape, dtype, density=0.2, special=True):
    d = np.zeros(shape, dtype)
    mask = rng.random(shape) < density
    vals = rng.integers(-4, 5, shape) if np.dtype(dtype).kind in "iub" else rng.standard_normal(shape)
    d[mask] = vals[mask].astype(dtype)
    if special and d.dtype.kind == "f" and d.size > 8:
        flat = d.reshape(-1)
        flat[rng.integers(0, d.size, 3)] = -0.0
        flat[rng.integers(0, d.size, 2)] = np.nan
    return d


def unpatched(call):
    """``call()`` on plain SciPy, for the pure-Python replacements that ``disabled()`` does not switch off."""
    fastscipy.unpatch(["sparse"])
    try:
        return call()
    finally:
        fastscipy.patch(["sparse"])


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("shape", [(1, 1), (1, 40), (40, 1), (37, 53), (200, 150)])
@pytest.mark.parametrize("axis", [0, 1])
def test_argminmax(fmt, dtype, shape, axis):
    rng = np.random.default_rng(shape[0] * 7 + axis)
    for density in (0.0, 0.05, 0.5, 1.0):
        a = fmt(dense(rng, shape, dtype, density))
        for explicit in (False, True):
            ok = a.nnz > 0 or not explicit
            for f in (a.argmax, a.argmin):
                try:
                    with fastscipy.disabled():
                        f(axis=axis, explicit=explicit)
                except ValueError:
                    continue
                check(lambda: f(axis=axis, explicit=explicit), fast=a.nnz > 0 or ok)


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
def test_argminmax_duplicates_unsorted_and_implicit_zero_rules(dtype):
    rng = np.random.default_rng(1)
    for _ in range(30):
        n = 25
        ip = np.sort(rng.integers(0, 60, n + 1))
        ip[0] = 0
        nnz = ip[-1]
        ind = rng.integers(0, 12, nnz).astype(np.int32)  # unsorted, with duplicates
        data = rng.integers(-3, 4, nnz).astype(dtype)
        for fmt in (sp.csr_array, sp.csr_matrix):
            for f in ("argmax", "argmin"):
                check(lambda: getattr(fmt((data.copy(), ind.copy(), ip.astype(np.int32)), shape=(n, 12)), f)(axis=1))
                check(lambda: getattr(fmt((data.copy(), ind.copy(), ip.astype(np.int32)), shape=(n, 12)), f)(axis=0))


@pytest.mark.parametrize("dtype", DECLINED_DTYPES)
def test_argminmax_other_dtypes_run_in_scipy(dtype):
    rng = np.random.default_rng(2)
    a = sp.csr_array(dense(rng, (30, 20), np.int8 if dtype in (np.bool_, np.uint8) else np.float64).astype(dtype))
    before = fast_calls()
    check(lambda: a.argmax(axis=1), fast=False)
    assert fast_calls() == before


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("index", [np.int32, np.int64])
@pytest.mark.parametrize("fmt", [sp.coo_array, sp.coo_matrix])
def test_coo_sum_duplicates(dtype, index, fmt):
    rng = np.random.default_rng(3)
    for n, side in ((1, 5), (50, 8), (3000, 40), (20000, 6), (5000, 3000)):  # 20000 over 36 cells: segments > 128
        shape = (side, side + 3)
        row = rng.integers(0, side, n).astype(index)
        col = rng.integers(0, side + 3, n).astype(index)
        data = (
            (rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4, n)).astype(dtype)
            if np.dtype(dtype).kind == "f"
            else rng.integers(-(2**30), 2**30, n).astype(dtype)
        )
        if np.dtype(dtype).kind == "f":
            data[rng.integers(0, n, 2)] = [-0.0, np.nan]

        def run():
            m = fmt((data.copy(), (row.copy(), col.copy())), shape=shape)
            m.sum_duplicates()
            return m

        check(run)


def test_coo_sum_duplicates_declines():
    rng = np.random.default_rng(4)
    row, col = rng.integers(0, 9, 100), rng.integers(0, 9, 100)
    for data in (
        rng.random(100) + 1j * rng.random(100),
        rng.random(100) > 0.5,
        rng.integers(0, 9, 100).astype(np.int16),
    ):
        before = fast_calls()
        m = sp.coo_array((data, (row, col)), shape=(9, 9))
        with fastscipy.disabled():
            e = sp.coo_array((data, (row, col)), shape=(9, 9))
            e.sum_duplicates()
        m.sum_duplicates()
        assert_identical(e, m) and fast_calls() == before
    m = sp.coo_array((rng.random(5), (row[:5], col[:5])), shape=(10**9, 10))  # huge row count: declined
    check(lambda: (m.sum_duplicates(), m)[1], fast=False)
    m = sp.coo_array((np.ones(2), (np.array([0, 1], np.int32), np.array([1, 2], np.int64))), shape=(3, 3))
    check(lambda: (m.sum_duplicates(), m)[1], fast=False)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("shape", [(64, 64), (100, 70), (1, 5000), (5000, 1), (300, 200)])
@pytest.mark.parametrize("layout", ["C", "F", "transposed", "strided"])
def test_dense_to_sparse(fmt, dtype, shape, layout):
    rng = np.random.default_rng(shape[0])
    d = dense(rng, shape, dtype, density=0.15)
    if layout == "F":
        d = np.asfortranarray(d)
    elif layout == "transposed":
        d = np.ascontiguousarray(d.T).T
    elif layout == "strided":
        d = np.repeat(np.repeat(d, 2, 0), 2, 1)[::2, ::2]
    check(lambda: fmt(d))


@pytest.mark.parametrize("fmt", FORMATS)
def test_dense_declines(fmt):
    rng = np.random.default_rng(5)
    d = dense(rng, (80, 80), np.float64, 0.2)
    for call in (
        lambda: fmt(d[:3, :3]),
        lambda: fmt(d, shape=(80, 80)),
        lambda: fmt(d, dtype=np.float32),
        lambda: fmt(d.astype(np.complex128)),
        lambda: fmt(d > 0.5),
        lambda: fmt(d.astype(np.int16)),
        lambda: fmt(np.zeros((0, 90))),
    ):
        with fastscipy.disabled():
            e = call()
        assert_identical(e, call())  # (CSR/CSC may still use the COO fast path inside SciPy's own conversion)


@pytest.mark.parametrize("fmt", [sp.csr_array, sp.csc_array, sp.coo_array, sp.csr_matrix, sp.coo_matrix, sp.lil_array])
@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.complex128])
@pytest.mark.parametrize("index", [np.int32, np.int64])
def test_find(fmt, dtype, index):
    rng = np.random.default_rng(6)
    n, shape = 400, (30, 25)
    data = rng.integers(-2, 3, n).astype(dtype)  # explicit zeros, and sums that cancel
    coo = sp.coo_array(
        (data, (rng.integers(0, 30, n).astype(index), rng.integers(0, 25, n).astype(index))), shape=shape
    )
    a = fmt(coo.tocsr()) if fmt is sp.lil_array else fmt(coo)
    check(lambda: sp.find(a), fast=isinstance(a, sp.sparray) and a.format in ("csr", "coo"))
    # unsorted CSR with duplicate columns in a row
    ip = np.arange(0, n + 1, 10)[:31].astype(index)
    unsorted = sp.csr_array((data, rng.integers(0, 25, n).astype(index), ip), shape=shape)
    check(lambda: sp.find(unsorted))
    check(lambda: sp.find(sp.csc_array(coo)))


def test_small_calls_match_scipy():
    rng = np.random.default_rng(7)
    d, i, p = rng.random(9), np.array([0, 2, 1, 1, 0, 3, 2, 3, 3], np.int32), np.array([0, 3, 5, 9], np.int32)

    def run():
        out = []
        for f in (sp.csr_array, sp.csc_matrix, sp.csr_matrix):
            shape = (4, 3) if f is sp.csc_matrix else (3, 4)
            m = f((d, i, p), shape=shape)
            m2 = f((d, i.astype(np.int64), p.astype(np.int64)), shape=shape)
            z = f((5, 6))
            out += [m, m2, z, m.T, m + m, m @ m.T, m.tocsc(), m * 2, f((d[:3], (i[:3], i[3:6])))]
        out += [
            sp.eye_array(7, format="csr"),
            sp.diags_array([[1.0] * 9, [2.0] * 8], offsets=[0, 1]),
            sp.get_index_dtype(maxval=2**31 - 1),
            sp.get_index_dtype(maxval=2**31),
            sp.get_index_dtype((i,)),
            sp.get_index_dtype((i.astype(np.int64),), check_contents=True),
        ]
        return out

    assert_identical(unpatched(run), run())
    for bad in (
        lambda: sp.csr_array((-1, 2)),
        lambda: sp.csr_array((d, i, p[:2]), shape=(3, 4)),
        lambda: sp.csr_array((d, i, np.array([1, 3, 5, 9], np.int32)), shape=(3, 4)),
    ):

        def msg(f):
            try:
                f()
            except Exception as e:
                return type(e), str(e)

        assert msg(bad) == unpatched(lambda: msg(bad)) is not None
