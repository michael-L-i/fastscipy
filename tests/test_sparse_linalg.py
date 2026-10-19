import warnings

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as sl

import fastscipy
from fastscipy.testing import assert_identical, check, fast_calls


@pytest.fixture(params=[1, 2], autouse=True)
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield
    fastscipy.set_num_threads(old)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("layout", ["contiguous", "reversed", "readonly"])
def test_operator_vector_and_adjoint(dtype, layout):
    r = np.random.default_rng(18)
    a = r.standard_normal((7, 5)).astype(dtype)
    if np.dtype(dtype).kind == "c":
        a += 1j * r.standard_normal(a.shape).astype(dtype)
    for matrix in (a, sparse.csr_array(a), sparse.csc_matrix(a)):
        operator = sl.aslinearoperator(matrix)
        for method, n in ((operator.matvec, 5), (operator.rmatvec, 7)):
            x = r.standard_normal(n).astype(dtype)
            if layout == "reversed":
                x = x[::-1]
            elif layout == "readonly":
                x.flags.writeable = False
            before = x.tobytes()
            check(lambda: method(x))
            assert x.tobytes() == before


@pytest.mark.parametrize(
    "solver", [sl.cg, sl.bicg, sl.bicgstab, sl.cgs, sl.gmres, sl.lgmres, sl.minres, sl.qmr, sl.tfqmr, sl.gcrotmk]
)
@pytest.mark.parametrize("n", [8, 100])
def test_solver_results_and_iteration_history(solver, n):
    a = sparse.diags_array([np.full(n - 1, -0.5), np.arange(n) + 2.0, np.full(n - 1, -0.5)], offsets=[-1, 0, 1])
    b = np.random.default_rng(n).standard_normal(n)
    a_before, b_before = a.data.tobytes(), b.tobytes()

    def run():
        history = []
        kwargs = {"callback_type": "legacy"} if solver is sl.gmres else {}
        result = solver(a, b, rtol=1e-9, callback=lambda x: history.append(np.array(x, copy=True)), **kwargs)
        return result, history

    check(run)
    assert a.data.tobytes() == a_before and b.tobytes() == b_before


@pytest.mark.parametrize("solver", [sl.lsqr, sl.lsmr])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_iterative_rectangular_least_squares(solver, dtype):
    r = np.random.default_rng(82)
    a = r.standard_normal((13, 7)).astype(dtype)
    if np.dtype(dtype).kind == "c":
        a += 1j * r.standard_normal(a.shape)
    b = r.standard_normal(13).astype(dtype)
    for cls in (sparse.csr_array, sparse.csc_matrix):
        matrix = cls(a)
        check(lambda: solver(matrix, b))


def test_custom_operator_subclass_uses_original_dispatch():
    class Custom(sl.LinearOperator):
        def __init__(self):
            super().__init__(np.float64, (3, 3))

        def _matvec(self, x):
            return x * 2

    operator = Custom()
    before = fast_calls()
    assert_identical(np.arange(3.0) * 2, operator.matvec(np.arange(3.0)))
    assert fast_calls() == before


def test_explicit_alternate_namespace_uses_original_dispatch():
    from scipy.sparse.linalg._interface import MatrixLinearOperator

    operator = MatrixLinearOperator(sparse.csr_array(np.eye(3)), xp=np)
    before = fast_calls()
    assert_identical(np.arange(3.0), operator.matvec(np.arange(3.0)))
    assert fast_calls() == before


def test_callback_is_called_once_on_error_and_bad_output():
    calls = []
    error = RuntimeError("callback failed")

    def failing(x):
        calls.append(x)
        raise error

    operator = sl.LinearOperator((3, 3), matvec=failing, dtype=np.float64)
    x = np.arange(3.0)
    with pytest.raises(RuntimeError) as raised:
        operator.matvec(x)
    assert raised.value is error and calls == [x]

    calls.clear()

    def bad_shape(x):
        calls.append(x)
        return np.zeros(4)

    operator = sl.LinearOperator((3, 3), matvec=bad_shape, dtype=np.float64)
    with pytest.raises(ValueError):
        operator.matvec(x)
    assert len(calls) == 1 and calls[0] is x


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (1, 1)])
def test_operator_empty_and_single_element_shapes(shape):
    for cls in (sparse.csr_array, sparse.csc_matrix):
        operator = sl.aslinearoperator(cls(shape, dtype=np.float64))
        check(lambda: operator.matvec(np.ones(shape[1])), fast=all(shape))


def test_overridden_sparse_kernel_exception_runs_once():
    a = sparse.csr_array(np.eye(3))
    operator = sl.aslinearoperator(a)
    calls = []

    def failing(x):
        calls.append(x)
        raise RuntimeError("sparse kernel failed")

    a._matmul_vector = failing
    with pytest.raises(RuntimeError, match="sparse kernel failed"):
        operator.matvec(np.ones(3))
    assert len(calls) == 1


def test_callback_mutation_and_non_array_outputs():
    for output in ("list", "scalar", "column"):

        def matvec(x):
            x[0] += 1
            if output == "list":
                return list(x)
            return x[0] if output == "scalar" else x[:, None]

        n = 1 if output == "scalar" else 3
        operator = sl.LinearOperator((n, n), matvec=matvec, dtype=np.float64)

        def run():
            x = np.arange(float(n))
            return operator.matvec(x), x

        check(run)


@pytest.mark.parametrize("kind", ["column", "batch", "list", "integer", "bad_shape"])
def test_declined_inputs_preserve_warnings_and_errors(kind):
    operator = sl.aslinearoperator(np.eye(3))
    x = {
        "column": np.ones((3, 1)),
        "batch": np.ones((2, 3)),
        "list": [1.0, 2.0, 3.0],
        "integer": np.arange(3),
        "bad_shape": np.ones(4),
    }[kind]

    def outcome():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                out = operator.matvec(x)
            except Exception as error:
                out = type(error), str(error)
        return out, [(w.category, str(w.message)) for w in caught]

    with fastscipy.disabled():
        expected = outcome()
    before = fast_calls()
    assert_identical(expected, outcome())
    assert fast_calls() == before


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("cls", [sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix])
def test_exact_adjoint_uses_sparse_vector_kernel(dtype, cls):
    r = np.random.default_rng(341)
    dense = r.standard_normal((13, 7)).astype(dtype)
    if np.dtype(dtype).kind == "c":
        dense += 1j * r.standard_normal(dense.shape).astype(dtype)
    matrix = cls(dense)
    operator = sl.aslinearoperator(matrix)
    x = r.standard_normal(13).astype(dtype)[::-1]
    x.flags.writeable = False
    before = matrix.data.tobytes(), x.tobytes()
    key = ("MatrixLinearOperator sparse matvec", "fast")
    calls = fastscipy.stats().get(key, 0)
    check(lambda: operator.rmatvec(x))
    assert fastscipy.stats().get(key, 0) == calls + 1
    check(lambda: operator.H.matvec(x))
    assert (matrix.data.tobytes(), x.tobytes()) == before


@pytest.mark.parametrize("solver", [sl.lsqr, sl.lsmr])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_adjoint_least_squares_results_and_inputs(solver, dtype):
    r = np.random.default_rng(342)
    dense = r.standard_normal((23, 11)).astype(dtype)
    if np.dtype(dtype).kind == "c":
        dense += 1j * r.standard_normal(dense.shape).astype(dtype)
    matrix = sparse.csc_array(dense)
    b = r.standard_normal(23).astype(dtype)
    before = matrix.data.tobytes(), b.tobytes()
    for damp in (0.0, 0.1):

        def run():
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                result = solver(matrix, b, damp=damp, atol=1e-9, btol=1e-9)
            return result, [(w.category, str(w.message)) for w in seen]

        check(run)
    assert (matrix.data.tobytes(), b.tobytes()) == before


def test_adjoint_sparse_callback_warning_and_error_once():
    operator = sl.aslinearoperator(sparse.csr_array(np.eye(3)))
    calls = []

    def fail(x):
        calls.append(x)
        warnings.warn("adjoint callback", RuntimeWarning, stacklevel=2)
        raise RuntimeError("adjoint sparse kernel")

    operator.H.A._matmul_vector = fail
    with pytest.warns(RuntimeWarning, match="adjoint callback") as seen:
        with pytest.raises(RuntimeError, match="adjoint sparse kernel"):
            operator.rmatvec(np.ones(3))
    assert len(calls) == len(seen) == 1


def test_adjoint_explicit_namespace_and_subclass_decline():
    from scipy.sparse.linalg._interface import _AdjointMatrixOperator

    class Custom(_AdjointMatrixOperator):
        pass

    matrix = sparse.csr_array(np.eye(3))
    for operator in (_AdjointMatrixOperator(matrix, xp=np), Custom(matrix)):
        before = fast_calls()
        assert_identical(operator.matvec(np.arange(3.0)), np.arange(3.0))
        assert fast_calls() == before
