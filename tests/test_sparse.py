import numpy as np
import pytest
import scipy.sparse as sp

import fastscipy
from fastscipy import _sparse
from fastscipy.testing import assert_identical, check, fast_calls

VALUE_DTYPES = [np.float32, np.float64, np.int32, np.int64]
DECLINED_DTYPES = [np.bool_, np.int16, np.uint8, np.longdouble]
FORMATS = [sp.csr_array, sp.csc_array, sp.coo_array, sp.csr_matrix, sp.csc_matrix, sp.coo_matrix]


@pytest.fixture(params=["1 thread", "2 threads, no size thresholds"], autouse=True)
def threads(request, monkeypatch):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(1 if request.param[0] == "1" else 2)
    if request.param[0] == "2":
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


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("index", [np.int32, np.int64])
@pytest.mark.parametrize("fmt", [sp.coo_array, sp.coo_matrix])
@pytest.mark.parametrize("order", ["sorted", "column descent", "row descent"])
def test_coo_sorted_duplicate_segments(dtype, index, fmt, order):
    # Exercise reduceat's singleton, short, eight-lane and recursive pairwise sums.
    lengths = np.array([1, 2, 7, 8, 9, 127, 128, 129, 257, 1025])
    row = np.repeat(np.arange(len(lengths)) // 4, lengths).astype(index)
    col = np.repeat(np.arange(len(lengths)) % 4, lengths).astype(index)
    rng = np.random.default_rng(243)
    data = rng.integers(-(2**30), 2**30, row.size).astype(dtype)
    if np.dtype(dtype).kind == "f":
        data *= np.resize(np.array([1e-10, 1, 1e10], dtype=dtype), data.size)
        data[0] = -0.0
        data[1:3] = -0.0
        data[3:10] = np.nan
        data[10:18] = np.resize(np.array([np.inf, -np.inf], dtype=dtype), 8)
    else:
        data[:3] = np.iinfo(dtype).max
    if order == "column descent":
        col[-1] = 0
    elif order == "row descent":
        row[-1] = 0

    def run():
        matrix = fmt((data.copy(), (row.copy(), col.copy())), shape=(4, 4))
        # Matrix constructors may narrow index types; keep both supported widths.
        matrix.coords = (row.copy(), col.copy())
        matrix.sum_duplicates()
        return matrix

    with np.errstate(invalid="ignore"):
        check(run)


@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("duplicates", [False, True])
def test_coo_sorted_unique_and_explicit_zero(dtype, duplicates):
    row = np.repeat(np.arange(50, dtype=np.int64), 4)
    col = np.tile(np.arange(4, dtype=np.int64), 50)
    if duplicates:
        col //= 2
    data = np.zeros(row.size, dtype=dtype)
    if np.dtype(dtype).kind == "f":
        data[::2] = -0.0

    def run():
        matrix = sp.coo_array((data.copy(), (row.copy(), col.copy())), shape=(50, 4))
        original = tuple(a.copy() for a in (*matrix.coords, matrix.data))
        inputs = (*matrix.coords, matrix.data)
        matrix.sum_duplicates()
        for before, source, result in zip(original, inputs, (*matrix.coords, matrix.data)):
            assert_identical(before, source)
            assert not np.shares_memory(source, result)
        return matrix

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
def test_dense_longlong_values(fmt):
    """The explicit long-long spelling also represents int64 on Windows."""
    values = np.zeros((64, 64), dtype="q")
    values[::3, ::5] = 2**40
    check(lambda: fmt(values))


@pytest.mark.parametrize("fmt", [sp.coo_array, sp.coo_matrix])
def test_coo_longlong_indices_and_values(fmt):
    rows = np.array([0, 1, 0, 1], dtype="q")
    cols = np.array([1, 2, 1, 2], dtype="q")
    values = np.array([2**40, -(2**40), 7, 11], dtype="q")

    def run():
        matrix = fmt((values.copy(), (rows, cols)), shape=(4, 4))
        # Small constructors can normalize coordinates to int32; explicitly
        # retain the valid long-long index spelling to cover its native path.
        matrix.coords = (rows.copy(), cols.copy())
        matrix.sum_duplicates()
        return matrix

    check(run)


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


def unusual_buffer(values, layout):
    """Valid NumPy arrays that cannot be read as native aligned element slices."""
    if layout == "byteswapped":
        return values.astype(values.dtype.newbyteorder("S"))
    strides = [values.itemsize + (layout == "byte_strides")]
    for length in reversed(values.shape[1:]):
        strides.insert(0, strides[0] * length)
    offset = int(layout == "unaligned")
    size = offset + values.itemsize + sum((n - 1) * s for n, s in zip(values.shape, strides))
    result = np.ndarray(values.shape, dtype=values.dtype, buffer=bytearray(size), offset=offset, strides=strides)
    result[...] = values
    return result


def forbidden_native(*args):
    pytest.fail("unsupported buffer reached a native sparse kernel")


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("layout", ["unaligned", "byteswapped", "byte_strides"])
def test_dense_buffer_layouts_decline(fmt, dtype, layout, monkeypatch):
    values = (np.arange(64 * 64).reshape(64, 64) % 11 - 5).astype(dtype)
    data = unusual_buffer(values, layout)
    before = data.tobytes()
    # Intercept the boundary so a missing guard fails without executing unsafe Rust.
    with monkeypatch.context() as guard:
        guard.setattr(_sparse._native.sparse, "dense_to_compressed", forbidden_native)
        assert _sparse._dense(data, None, None, False) is None
        assert _sparse._dense(data, None, None, True) is None
    check(lambda: fmt(data), fast=False)
    assert data.tobytes() == before


@pytest.mark.parametrize("fmt", [sp.coo_array, sp.coo_matrix])
@pytest.mark.parametrize("operand", ["row", "col", "data"])
@pytest.mark.parametrize("layout", ["unaligned", "byteswapped", "byte_strides"])
def test_coo_buffer_layouts_decline(fmt, operand, layout, monkeypatch):
    def make():
        matrix = fmt((np.array([1.5, -2.0, 4.0]), ([0, 1, 0], [1, 0, 1])), shape=(2, 2))
        row, col = matrix.coords
        if operand == "data":
            matrix.data = unusual_buffer(matrix.data, layout)
        else:
            matrix.coords = (
                unusual_buffer(row, layout) if operand == "row" else row,
                unusual_buffer(col, layout) if operand == "col" else col,
            )
        return matrix

    matrix = make()
    with monkeypatch.context() as guard:
        guard.setattr(_sparse._native.sparse, "sum_duplicates", forbidden_native)
        assert _sparse._sum_duplicates(matrix, matrix.coords, matrix.data) is NotImplemented

    def run():
        matrix = make()
        matrix.sum_duplicates()
        return matrix

    check(run, fast=False)


@pytest.mark.parametrize("fmt,axis", [(sp.csr_array, 1), (sp.csc_array, 0)])
@pytest.mark.parametrize("operand", ["indptr", "indices", "data"])
@pytest.mark.parametrize("layout", ["unaligned", "byteswapped", "byte_strides"])
def test_argminmax_buffer_layouts_decline(fmt, axis, operand, layout, monkeypatch):
    matrix = fmt(np.array([[0.0, 2.0, 1.0], [3.0, 0.0, 4.0]]))
    matrix.sum_duplicates()
    setattr(matrix, operand, unusual_buffer(getattr(matrix, operand), layout))
    for function, compare, method in ((np.argmax, np.greater, "argmax"), (np.argmin, np.less, "argmin")):
        with monkeypatch.context() as guard:
            guard.setattr(_sparse._native.sparse, "argminmax", forbidden_native)
            assert _sparse._argminmax_axis(matrix, axis, function, compare, False) is NotImplemented
        check(lambda: getattr(matrix, method)(axis=axis), fast=False)


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


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.bool_, np.int64])
@pytest.mark.parametrize("index", [np.int32, np.int64])
@pytest.mark.parametrize("canonical", [True, False])
def test_find_preserves_input_and_returns_detached_arrays(dtype, index, canonical):
    data = np.array([0, 2, -1, 3, 0, 4], dtype=dtype)
    if np.dtype(dtype).kind in "fc":
        data[2] = np.nan
    columns = np.array([0, 1, 2, 0, 1, 2] if canonical else [2, 0, 2, 1, 1, 0], dtype=index)
    a = sp.csr_array((data, columns, np.array([0, 3, 6], dtype=index)), shape=(2, 3))
    original = tuple(x.copy() for x in (a.data, a.indices, a.indptr))
    for arr in (a.data, a.indices, a.indptr):
        arr.flags.writeable = False
    result = check(lambda: sp.find(a))
    for before, after in zip(original, (a.data, a.indices, a.indptr)):
        assert_identical(before, after)
        assert not after.flags.writeable
    for output in result:
        assert output.flags.writeable
        assert output.flags.c_contiguous
        assert not any(np.shares_memory(output, arr) for arr in (a.data, a.indices, a.indptr))


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("kind", ["array", "matrix", "mixed"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128, np.int64, np.bool_])
def test_block_diag_compressed(fmt, kind, dtype):
    rng = np.random.default_rng(20)
    blocks = []
    for i, shape in enumerate(((3, 5), (2, 1), (0, 4), (4, 0), (6, 3))):
        cls = getattr(sp, fmt + ("_matrix" if kind == "matrix" or kind == "mixed" and i % 2 else "_array"))
        x = rng.integers(-2, 3, shape).astype(dtype)
        if np.dtype(dtype).kind in "fc" and x.size:
            x.flat[0] = np.nan
        block = cls(x)
        if block.nnz:
            block.data[-1] = -0.0 if np.dtype(dtype).kind in "fc" else 0
        if i == 4 and kind == "array":
            block.indices = block.indices.astype(np.int64)
            block.indptr = block.indptr.astype(np.int64)
        # Empty constructors don't populate their cached canonical property.
        assert block.has_canonical_format
        for array in (block.data, block.indices, block.indptr):
            array.flags.writeable = False
        blocks.append(block)
    before = [(b.data.copy(), b.indices.copy(), b.indptr.copy()) for b in blocks]
    flags = [(b._has_canonical_format, b._has_sorted_indices) for b in blocks]
    fast_before = fastscipy.stats().get(("block_diag", "fast"), 0)
    result = check(lambda: sp.block_diag(blocks, format=fmt))
    assert fastscipy.stats().get(("block_diag", "fast"), 0) > fast_before
    assert flags == [(b._has_canonical_format, b._has_sorted_indices) for b in blocks]
    for block, buffers in zip(blocks, before):
        for actual, expected in zip((block.data, block.indices, block.indptr), buffers):
            assert_identical(actual, expected)
            assert not actual.flags.writeable
            assert not any(np.shares_memory(actual, array) for array in (result.data, result.indices, result.indptr))
    empty = [getattr(sp, fmt + "_array")((0, 3)), getattr(sp, fmt + "_array")((4, 0))]
    for block in empty:
        assert block.has_canonical_format
    check(lambda: sp.block_diag(empty, format=fmt))


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_block_diag_large_offsets_with_mixed_index_widths(fmt):
    cls = getattr(sp, fmt + "_array")
    limit = np.iinfo(np.int32).max
    empty_shape = (1, limit - 3) if fmt == "csr" else (limit - 3, 1)
    tail_shape = (1, 3) if fmt == "csr" else (3, 1)
    empty = cls(empty_shape)
    empty.indices = empty.indices.astype(np.int64)
    empty.indptr = empty.indptr.astype(np.int64)
    tail = cls((np.array([2.0]), np.array([2], np.int32), np.array([0, 1], np.int32)), shape=tail_shape)
    result = check(lambda: sp.block_diag([empty, tail], format=fmt))
    assert result.indices.tolist() == [limit - 1]
    assert result.indices.dtype == np.int64


def test_block_diag_fallbacks():
    a = sp.csr_array([[1.0, 0], [0, 2.0]])
    for call in (
        lambda: sp.block_diag([a, a], format="coo"),
        lambda: sp.block_diag([a, a.tocsc()], format="csr"),
        lambda: sp.block_diag([a, a], format="csr", dtype=np.float32),
    ):
        check(call, fast=False)
    # Explicitly stale flags must not bypass SciPy's sorting and duplicate sum.
    invalid = sp.csr_array((np.array([2.0, 3.0]), np.array([1, 1]), np.array([0, 2, 2])), shape=(2, 2))
    invalid.has_canonical_format = True
    assert _sparse._block_diag([invalid, a], format="csr") is NotImplemented
    check(lambda: sp.block_diag([invalid, a], format="csr"))


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_block_diag_rejects_malformed_buffers_without_touching_them(fmt):
    cls = getattr(sp, fmt + "_array")
    mutations = (
        ("indptr", np.array([], dtype=np.int32)),
        ("indptr", np.array([0, 3, 2], dtype=np.int32)),
        ("indptr", np.array([0, -1, 2], dtype=np.int32)),
        ("indptr", np.array([0, 99, 2], dtype=np.int32)),
        ("indptr", np.array([[0, 1, 2]], dtype=np.int32)),
        ("indices", np.array([-1, 1], dtype=np.int32)),
        ("indices", np.array([0, 9], dtype=np.int32)),
        ("indices", np.array([0, 1], dtype=">i4")),
        ("data", np.array([1, 2], dtype=np.float16)),
        ("data", np.array([1, 2], dtype=">f8")),
    )
    for attribute, values in mutations:
        block = cls(np.eye(2))
        block.has_canonical_format = True
        setattr(block, attribute, values)
        before = [
            (array.tobytes(), array.shape, array.strides, array.dtype, array.flags.writeable)
            for array in (block.indptr, block.indices, block.data)
        ]
        flags = block._has_canonical_format, block._has_sorted_indices
        assert _sparse._block_diag([block], format=fmt) is NotImplemented
        after = [
            (array.tobytes(), array.shape, array.strides, array.dtype, array.flags.writeable)
            for array in (block.indptr, block.indices, block.data)
        ]
        assert before == after
        assert flags == (block._has_canonical_format, block._has_sorted_indices)


def test_block_diag_preserves_uncached_inputs_and_nan_payloads():
    data = np.array([0x7FF8000000000003, 0x8000000000000000], dtype=np.uint64).view(np.float64)
    block = sp.csr_array((data, np.array([0, 1], np.int32), np.array([0, 1, 2], np.int32)), shape=(2, 2))
    original_keys = set(vars(block))
    original_data = block.data.tobytes()
    result = _sparse._block_diag([block, block], format="csr")
    assert result is not NotImplemented
    assert set(vars(block)) == original_keys
    assert block.data.tobytes() == original_data
    with fastscipy.disabled():
        expected = sp.block_diag([block, block], format="csr")
    assert_identical(result, expected)


@pytest.mark.parametrize("fmt,axis", [("csr", 1), ("csc", 0)])
@pytest.mark.parametrize("kind", ["array", "matrix", "matrix_first", "array_first"])
@pytest.mark.parametrize("index", [np.int32, np.int64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128, np.int64, np.bool_])
def test_pair_stack_preserves_noncanonical_inputs(fmt, axis, kind, index, dtype):
    blocks = []
    for i, (width, columns, pointers, values) in enumerate(
        ((4, [2, 0, 2, 1, 3], [0, 3, 3, 5], [0, 1, 2, -1, 0]), (5, [4, 0, 2], [0, 2, 3, 3], [3, 0, 1]))
    ):
        matrix = kind == "matrix" or kind == "matrix_first" and i == 0 or kind == "array_first" and i == 1
        cls = getattr(sp, fmt + ("_matrix" if matrix else "_array"))
        shape = (3, width) if axis == 1 else (width, 3)
        data = np.array(values, dtype=dtype)
        if data.dtype.kind in "fc":
            data[0] = -0.0
            data[-1] = np.nan
        block = cls((data, np.array(columns, dtype=index), np.array(pointers, dtype=index)), shape=shape)
        block.indices, block.indptr = block.indices.astype(index), block.indptr.astype(index)
        block.has_canonical_format = True  # Deliberately stale; stacking must preserve duplicates.
        for a in (block.data, block.indices, block.indptr):
            a.flags.writeable = False
        blocks.append(block)
    before = [(b.data.tobytes(), b.indices.tobytes(), b.indptr.tobytes(), set(vars(b))) for b in blocks]
    name = "hstack" if axis == 1 else "vstack"
    fast_before = fastscipy.stats().get((name, "fast"), 0)
    result = check(lambda: getattr(sp, name)(blocks))
    assert fastscipy.stats().get((name, "fast"), 0) > fast_before
    assert before == [(b.data.tobytes(), b.indices.tobytes(), b.indptr.tobytes(), set(vars(b))) for b in blocks]
    for b in blocks:
        assert b._has_canonical_format is True and b._has_sorted_indices is True
        assert all(not a.flags.writeable for a in (b.data, b.indices, b.indptr))
        assert not any(
            np.shares_memory(x, y)
            for x in (b.data, b.indices, b.indptr)
            for y in (result.data, result.indices, result.indptr)
        )


@pytest.mark.parametrize("fmt,axis", [("csr", 1), ("csc", 0)])
def test_pair_stack_empty_and_fallbacks(fmt, axis):
    cls = getattr(sp, fmt + "_array")
    name = "hstack" if axis == 1 else "vstack"
    fast = _sparse._hstack if axis == 1 else _sparse._vstack
    for shapes in (((0, 2), (0, 3)), ((3, 0), (3, 0)), ((0, 0), (0, 0))):
        if axis == 0:
            shapes = tuple(s[::-1] for s in shapes)
        blocks = [cls(s) for s in shapes]
        check(lambda: getattr(sp, name)(blocks))
    block = cls(np.eye(3))
    for blocks, options in (
        ([block], {}),
        ([block] * 5, {}),
        ([block, block], {"dtype": np.float32}),
        ([block, block], {"format": "coo"}),
    ):
        assert fast(blocks, **options) is NotImplemented
        check(lambda: getattr(sp, name)(blocks, **options), fast=False)
    bad = block.copy()
    bad.indptr[1] = 99
    assert fast([bad, block]) is NotImplemented
    mismatch = cls(np.ones((2, 2)))
    assert fast([block, mismatch]) is NotImplemented
    check(lambda: getattr(sp, name)([block, mismatch]), fast=False)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("kind", ["array", "matrix", "mixed"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128, np.int64])
def test_block_diag_mixed_compressed_formats(fmt, kind, dtype):
    rng = np.random.default_rng(24)
    blocks = []
    for i in range(4):
        block_fmt = "csr" if i % 2 else "csc"
        cls = getattr(sp, block_fmt + ("_matrix" if kind == "matrix" or kind == "mixed" and i % 2 else "_array"))
        data = rng.integers(-3, 4, (3 + i, 2 + i)).astype(dtype)
        block = cls(data)
        if block.nnz and block.dtype.kind in "fc":
            block.data[0] = -0.0
            block.data[-1] = np.nan
        if kind == "array":
            block.indices, block.indptr = block.indices.astype(np.int64), block.indptr.astype(np.int64)
        assert block.has_canonical_format
        for a in (block.data, block.indices, block.indptr):
            a.flags.writeable = False
        blocks.append(block)
    before = [(b.data.tobytes(), b.indices.tobytes(), b.indptr.tobytes(), set(vars(b))) for b in blocks]
    fast_before = fastscipy.stats().get(("block_diag", "fast"), 0)
    check(lambda: sp.block_diag(blocks, format=fmt))
    assert fastscipy.stats().get(("block_diag", "fast"), 0) > fast_before
    assert before == [(b.data.tobytes(), b.indices.tobytes(), b.indptr.tobytes(), set(vars(b))) for b in blocks]


@pytest.mark.parametrize("fmt,axis", [("csr", 1), ("csc", 0)])
def test_pair_stack_mixed_value_dtypes(fmt, axis):
    cls = getattr(sp, fmt + "_array")
    a = cls(np.array([[0.0, -0.0], [1.0, 3.0]], dtype=np.float32))
    b = cls(np.array([[1j, 2.0], [3.0, np.nan]], dtype=np.complex128))
    name = "hstack" if axis == 1 else "vstack"
    check(lambda: getattr(sp, name)([a, b]))


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
