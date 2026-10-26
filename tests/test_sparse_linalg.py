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


def _iteration_reference_validation(a, b, rtol, *, minres=False, scalar_history=False):
    """Validate complete histories for explicit numerical compatibility."""

    def validate(expected, actual):
        assert type(expected) is type(actual) is tuple and len(expected) == len(actual) == 2
        expected_result, expected_history = expected
        actual_result, actual_history = actual
        assert type(expected_result) is type(actual_result) is tuple
        assert len(expected_result) == len(actual_result) == 2
        expected_x, expected_info = expected_result
        actual_x, actual_info = actual_result
        assert type(expected_info) is type(actual_info) and expected_info == actual_info
        assert type(expected_history) is type(actual_history) is list
        assert len(expected_history) == len(actual_history)
        pairs = [(expected_x, actual_x, b.shape)]
        history_shape = () if scalar_history else b.shape
        pairs.extend((x, y, history_shape) for x, y in zip(expected_history, actual_history))
        for x, y, shape in pairs:
            assert type(x) is type(y) is np.ndarray
            assert x.shape == y.shape == shape and x.dtype == y.dtype == b.dtype
            assert x.strides == y.strides
            assert np.isfinite(x).all() and np.isfinite(y).all()
            np.testing.assert_allclose(y, x, rtol=2e-12, atol=2e-14, equal_nan=False)
        if actual_info == 0:
            # Recompute the original equation independently of solver-reported
            # residuals, for both the baseline and accelerated solution.
            with fastscipy.disabled():
                for solution in (expected_x, actual_x):
                    if minres:
                        # MINRES stops on ||r|| / (Anorm * ||x||), not ||r|| / ||b||.
                        # Use the independently computed Frobenius norm of A as a
                        # conservative bound for its projected Lanczos estimate.
                        limit = rtol * sparse.linalg.norm(a, ord="fro") * np.linalg.norm(solution)
                    else:
                        limit = rtol * np.linalg.norm(b)
                    assert np.linalg.norm(b - a @ solution) <= limit, "equation residual is too large"

    return validate


def _bicgstab_reference_validation(a, b, rtol):
    """Validate convergence without claiming matching solver trajectories."""

    def validate(expected, actual):
        solutions = []
        assert type(expected) is type(actual) is tuple
        assert len(expected) == len(actual) == 2
        for output in (expected, actual):
            result, history = output
            assert type(result) is tuple and len(result) == 2
            solution, info = result
            assert type(info) is int and info == 0
            assert type(history) is list and 0 < len(history) <= 10 * len(b)
            for vector in (solution, *history):
                assert type(vector) is np.ndarray
                assert vector.shape == b.shape and vector.dtype == b.dtype
                assert vector.strides == b.strides and np.isfinite(vector).all()
            solutions.append(solution)
        with fastscipy.disabled():
            # This exact fixture is symmetric and strictly diagonally dominant.
            # Gershgorin bounds its smallest eigenvalue, hence ||A^-1||_2.
            assert (a - a.T).count_nonzero() == 0, "fixture must be symmetric"
            magnitude = abs(a)
            lower = np.min(2 * a.diagonal() - np.asarray(magnitude.sum(axis=1)).ravel())
            assert lower >= 1.5
            residuals = [np.linalg.norm(b - a @ x) for x in solutions]
            assert max(residuals) <= rtol * np.linalg.norm(b), "equation residual is too large"
            # Include a standard gamma_n rounding bound for independent residual
            # evaluation, subtraction and norms, rather than an empirical atol.
            epsn = (len(b) + 2) * np.finfo(b.dtype).eps
            gamma = epsn / (1 - epsn)
            rounding = sum(gamma * np.linalg.norm(magnitude @ abs(x) + abs(b)) for x in solutions)
            limit = (sum(residuals) + rounding) / lower
            limit += gamma * sum(np.linalg.norm(x) for x in solutions)
            assert np.linalg.norm(solutions[0] - solutions[1]) <= limit

    return validate


def _least_squares_reference_validation(a, b):
    """Independently check the default undamped LSQR result."""

    def validate(expected, actual):
        assert type(expected) is type(actual) is tuple and len(expected) == len(actual) == 10
        for index in (1, 2):
            assert type(expected[index]) is type(actual[index]) and expected[index] == actual[index]
        assert actual[1] in (1, 2, 4, 5) and 0 <= actual[2] <= 2 * a.shape[1]
        for index, (x, y) in enumerate(zip(expected, actual)):
            if index in (1, 2):
                continue
            assert type(x) is type(y)
            x, y = np.asarray(x), np.asarray(y)
            assert x.shape == y.shape and x.dtype == y.dtype
            assert x.strides == y.strides
            assert x.shape == ((a.shape[1],) if index in (0, 9) else ())
            assert np.isfinite(x).all() and np.isfinite(y).all()
            np.testing.assert_allclose(y, x, rtol=2e-12, atol=2e-14, equal_nan=False)
        with fastscipy.disabled():
            for result in (expected, actual):
                solution = result[0]
                assert type(solution) is type(result[9]) is np.ndarray
                assert solution.dtype == result[9].dtype == np.float64
                residual = a @ solution - b
                normal = np.linalg.norm(a.T @ residual)
                # Componentwise rounding scale for A.T @ (A @ x - b), avoiding
                # a loose ||A||**2 * ||x|| bound on differently scaled columns.
                magnitude = abs(a)
                scale = np.linalg.norm(magnitude.T @ (magnitude @ abs(solution) + abs(b)))
                limit = 16 * np.finfo(np.float64).eps * max(a.shape) * scale
                assert normal <= limit, "least-squares normal residual is too large"
                residual_norm = np.linalg.norm(residual)
                for index in (3, 4):  # r1norm == r2norm when damp is zero.
                    np.testing.assert_allclose(result[index], residual_norm, rtol=2e-12, atol=2e-14)
                np.testing.assert_allclose(result[8], np.linalg.norm(solution), rtol=2e-12, atol=2e-14)
                assert abs(result[7] - normal) <= limit
                assert result[5] > 0 and result[6] >= 1
                assert np.all(result[9] == 0)  # calc_var=False in this exact call.

    return validate


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

    # Larger solver histories can vary with Intel Accelerate buffer alignment.
    # Explicit numerical compatibility is limited to that backend and records
    # accuracy validation without claiming observed reference variance or byte
    # identity. Small n=8 cases retain byte-only checks.
    validation = (
        _iteration_reference_validation(a, b, 1e-9, minres=solver is sl.minres, scalar_history=solver is sl.gmres)
        if n == 100
        else None
    )
    if n == 100 and solver is sl.bicgstab:
        # Native Intel plain references independently produce 40--42 callbacks,
        # with amplified trajectory drift. Do not claim byte/roundoff-identical
        # histories here; retain that coverage with the exact fixture below.
        validation = _bicgstab_reference_validation(a, b, 1e-9)
    check(run, numerical_validation=validation)
    assert a.data.tobytes() == a_before and b.tobytes() == b_before


def test_bicgstab_exact_complete_single_callback():
    dense = 2 * np.eye(100)
    dense[0, 1] = 1.0
    a = sparse.csr_array(dense)
    b = np.full(100, 2.0)
    b[0] = 0.0
    known = b / 2
    known[0] = -0.5

    def run():
        history = []
        result = sl.bicgstab(a, b, rtol=1e-9, callback=lambda x: history.append(x.copy()))
        assert result[1] == 0 and len(history) == 1
        assert_identical(known, result[0])
        assert_identical(known, history[0])
        return result, history

    check(run)


@pytest.mark.parametrize("solver", [sl.qmr, sl.minres])
def test_symmetric_solver_exact_complete_single_callback(solver):
    # A single nonzero RHS gives exact power-of-two dot products and norms;
    # neither allocator alignment nor reduction order can change this solve.
    a = sparse.eye_array(100, format="csr") * 2
    b = np.zeros(100)
    b[1] = 2.0
    known = b / 2
    before = a.data.tobytes(), b.tobytes()

    def run():
        history = []
        result = solver(a, b, rtol=1e-9, callback=lambda x: history.append(x.copy()))
        assert result[1] == 0 and len(history) == 1
        assert_identical(known, result[0])
        assert_identical(known, history[0])
        return result, history

    check(run)
    assert before == (a.data.tobytes(), b.tobytes())


def test_bicgstab_reference_validator_accepts_distinct_valid_histories():
    a = sparse.csr_array(np.eye(2) * 2)
    b = np.array([2.0, 6.0])
    x = b / 2
    expected = ((x, 0), [np.zeros(2)])
    actual = ((np.nextafter(x, np.inf), 0), [np.ones(2), x.copy()])
    _bicgstab_reference_validation(a, b, 1e-9)(expected, actual)


def test_bicgstab_reference_validator_handles_actual_dia_fixture():
    n = 100
    a = sparse.diags_array([np.full(n - 1, -0.5), np.arange(n) + 2.0, np.full(n - 1, -0.5)], offsets=[-1, 0, 1])
    b = np.random.default_rng(n).standard_normal(n)
    with fastscipy.disabled():
        # DIA retains stored zero entries in A - A.T. Storage nnz is therefore
        # not a symmetry predicate, even though the matrix is exactly symmetric.
        difference = a - a.T
        assert difference.nnz > 0 and difference.count_nonzero() == 0
        history = []
        result = sl.bicgstab(a, b, rtol=1e-9, callback=lambda x: history.append(x.copy()))
    output = (result, history)
    _bicgstab_reference_validation(a, b, 1e-9)(output, output)


def test_bicgstab_reference_validator_rejects_nonsymmetric_fixture():
    a = sparse.diags_array([[0.5], [2.0, 3.0]], offsets=[1, 0], shape=(2, 2))
    x = np.ones(2)
    with fastscipy.disabled():
        b = a @ x
    output = ((x, 0), [x.copy()])
    with pytest.raises(AssertionError, match="symmetric"):
        _bicgstab_reference_validation(a, b, 1e-9)(output, output)


@pytest.mark.parametrize("fault", ["solution", "nonfinite", "empty", "too_many", "shape", "dtype", "info"])
def test_bicgstab_reference_validator_rejects_invalid_outputs(fault):
    a = sparse.csr_array(np.eye(2) * 2)
    b = np.array([2.0, 6.0])
    x = b / 2
    expected = ((x, 0), [x.copy()])
    y, info, history = x.copy(), 0, [x.copy()]
    if fault == "solution":
        y[0] += 1e-5
    elif fault == "nonfinite":
        history[0][0] = np.nan
    elif fault == "empty":
        history = []
    elif fault == "too_many":
        history *= 21
    elif fault == "shape":
        history[0] = history[0][:, None]
    elif fault == "dtype":
        history[0] = history[0].astype(np.float32)
    elif fault == "info":
        info = 1
    with pytest.raises(AssertionError):
        _bicgstab_reference_validation(a, b, 1e-9)(expected, ((y, info), history))


def test_iterative_reference_validator_accepts_roundoff():
    a = sparse.csr_array(np.eye(2) * 2)
    b = np.array([2.0, 6.0])
    expected_x = b / 2
    actual_x = np.nextafter(expected_x, np.inf)
    expected = ((expected_x, 0), [np.zeros(2), expected_x.copy()])
    actual = ((actual_x, 0), [np.zeros(2), actual_x.copy()])
    _iteration_reference_validation(a, b, 1e-9)(expected, actual)


@pytest.mark.parametrize("fault", ["info", "history_count", "dtype", "shape", "strides", "nan", "inaccurate"])
def test_iterative_reference_validator_rejects_wrong_results(fault):
    a = sparse.csr_array(np.eye(2) * 2)
    b = np.array([2.0, 6.0])
    x = b / 2
    expected = ((x, 0), [np.zeros(2), x.copy()])
    y, info, history = x.copy(), 0, [np.zeros(2), x.copy()]
    if fault == "info":
        info = 1
    elif fault == "history_count":
        history.pop()
    elif fault == "dtype":
        y = y.astype(np.float32)
    elif fault == "shape":
        y = y[:, None]
    elif fault == "strides":
        y = np.repeat(y, 2)[::2]
    elif fault == "nan":
        history[0][0] = np.nan
    elif fault == "inaccurate":
        y[0] += 1e-6
    with pytest.raises(AssertionError):
        _iteration_reference_validation(a, b, 1e-9)(expected, ((y, info), history))


def test_iterative_reference_validator_checks_equation_despite_close_values():
    a = sparse.csr_array(np.diag([1e9, 1.0]))
    b = np.ones(2)
    x = np.array([1e-9, 1.0])
    y = x.copy()
    y[0] += 1e-14  # Within the agreement bound, but gives a residual of 1e-5.
    with pytest.raises(AssertionError):
        _iteration_reference_validation(a, b, 1e-9)(((x, 0), []), ((y, 0), []))


def test_iterative_reference_validator_accepts_scalar_gmres_history():
    a = sparse.csr_array(np.eye(2) * 2)
    b = np.array([2.0, 6.0])
    x = b / 2
    expected = ((x, 0), [np.array(0.5), np.array(1e-16)])
    actual = ((np.nextafter(x, np.inf), 0), [np.array(np.nextafter(0.5, np.inf)), np.array(1e-16)])
    validate = _iteration_reference_validation(a, b, 1e-9, scalar_history=True)
    validate(expected, actual)
    wrong_shape = ((x, 0), [np.array([0.5]), np.array([1e-16])])
    with pytest.raises(AssertionError):
        validate(expected, wrong_shape)


def test_minres_reference_validator_uses_backward_error():
    a = sparse.csr_array(np.diag([1e6, 1.0]))
    b = np.ones(2)
    x = np.array([1e-6, 1.0 + 1e-5])
    result = ((x, 0), [x.copy()])
    # This is a valid normwise backward error, but fails a norm(b) criterion.
    _iteration_reference_validation(a, b, 1e-9, minres=True)(result, result)
    with pytest.raises(AssertionError, match="equation residual"):
        _iteration_reference_validation(a, b, 1e-9)(result, result)
    wrong = np.array([1e-6, 1.1])
    wrong_result = ((wrong, 0), [wrong.copy()])
    with pytest.raises(AssertionError, match="equation residual"):
        _iteration_reference_validation(a, b, 1e-9, minres=True)(wrong_result, wrong_result)


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
        # Only real LSQR independently varied in the full-component native
        # probe. Complex LSQR and every LSMR case retain strict byte checks.
        validation = (
            _least_squares_reference_validation(matrix, b) if solver is sl.lsqr and dtype is np.float64 else None
        )
        check(lambda: solver(matrix, b), numerical_validation=validation)


def _least_squares_validator_example():
    a = sparse.csr_array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    b = np.array([1.0, 2.0, 3.0])
    x = np.ones(2)
    result = (x, 2, 2, 3.0, 3.0, np.sqrt(5.0), 3.0, 0.0, np.sqrt(2.0), np.zeros(2))
    return a, b, result


def test_least_squares_reference_validator_accepts_roundoff():
    a, b, expected = _least_squares_validator_example()
    actual = (np.nextafter(expected[0], np.inf), *expected[1:])
    _least_squares_reference_validation(a, b)(expected, actual)


@pytest.mark.parametrize("index,value", [(1, 1), (2, 3), (3, 4.0), (5, np.inf)])
def test_least_squares_reference_validator_rejects_wrong_report(index, value):
    a, b, expected = _least_squares_validator_example()
    actual = list(expected)
    actual[index] = value
    with pytest.raises(AssertionError):
        _least_squares_reference_validation(a, b)(expected, tuple(actual))


def test_least_squares_reference_validator_independently_checks_reported_norm():
    a, b, result = _least_squares_validator_example()
    wrong = (*result[:3], 4.0, *result[4:])
    with pytest.raises(AssertionError):
        _least_squares_reference_validation(a, b)(wrong, wrong)


def test_least_squares_reference_validator_close_but_nonstationary_solution():
    a = sparse.csr_array([[1e9, 0.0], [0.0, 1.0], [0.0, 0.0]])
    b = np.ones(3)
    x = np.array([1e-9, 1.0])
    expected = (x, 2, 2, 1.0, 1.0, 1e9, 1e9, 0.0, np.linalg.norm(x), np.zeros(2))
    y = x.copy()
    y[0] += 1e-14  # Close in coordinates, but violates A.T @ residual == 0.
    with pytest.raises(AssertionError, match="normal residual"):
        _least_squares_reference_validation(a, b)(expected, (y, *expected[1:]))


@pytest.mark.parametrize("method", ["iterative", "lsqr", "bicgstab"])
def test_reference_validators_reject_correlated_enabled_matvec_error(monkeypatch, method):
    from fastscipy import _patch

    if method == "lsqr":
        a, b, correct = _least_squares_validator_example()
        true_solution = correct[0]
        wrong = (np.zeros(2), *correct[1:8], 0.0, correct[9])
        validate = _least_squares_reference_validation(a, b)
    else:
        a = sparse.csr_array(np.eye(2) * 2)
        b = np.array([2.0, 6.0])
        true_solution = b / 2
        wrong = ((np.zeros(2), 0), [np.zeros(2)])
        factory = _bicgstab_reference_validation if method == "bicgstab" else _iteration_reference_validation
        validate = factory(a, b, 1e-9)

    original = sparse.csr_array._matmul_vector
    enabled_calls = []

    def correlated_matvec(self, x):
        enabled_calls.append(_patch.enabled)
        # A broken enabled kernel could conceal exactly this wrong solution.
        if _patch.enabled and self.shape == a.shape:
            return original(self, true_solution)
        return original(self, x)

    monkeypatch.setattr(sparse.csr_array, "_matmul_vector", correlated_matvec)
    with pytest.raises(AssertionError, match="residual"):
        validate(wrong, wrong)
    assert enabled_calls and not any(enabled_calls)


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


@pytest.mark.parametrize("output", ["vector", "column", "reversed", "readonly", "empty", "subclass"])
def test_operator_reshape_result_contract(output):
    class Array(np.ndarray):
        pass

    n = 0 if output == "empty" else 7
    values = np.arange(n, dtype=np.float64)
    if output == "column":
        values = values[:, None]
    elif output == "reversed":
        values = values[::-1]
    elif output == "readonly":
        values.flags.writeable = False
    elif output == "subclass":
        values = values.view(Array)
    operator = sl.LinearOperator((n, n), matvec=lambda x: values, dtype=np.float64)
    x = np.zeros(n)
    check(lambda: operator.matvec(x))
    with fastscipy.disabled():
        expected = operator.matvec(x)
    actual = operator.matvec(x)
    assert actual is not values
    assert actual.flags.writeable == expected.flags.writeable
    assert actual.flags.owndata == expected.flags.owndata
    if n:
        assert np.shares_memory(actual, values)


def test_operator_reshape_preserves_subclass_array_function():
    calls = []

    class Array(np.ndarray):
        def __array_function__(self, func, types, args, kwargs):
            if func is np.reshape:
                calls.append(func)
            return super().__array_function__(func, types, args, kwargs)

    values = np.arange(5.0).view(Array)
    operator = sl.LinearOperator((5, 5), matvec=lambda x: values, dtype=np.float64)
    for enabled in (False, True):
        calls.clear()
        if enabled:
            result = operator.matvec(np.ones(5))
        else:
            with fastscipy.disabled():
                result = operator.matvec(np.ones(5))
        assert type(result) is Array and calls == [np.reshape]


@pytest.mark.parametrize("solver", [sl.cg, sl.gmres])
@pytest.mark.parametrize("maxiter", [1, None])
def test_composed_preconditioned_solver_history_and_residual(solver, maxiter):
    n = 64
    a = sparse.diags_array([-np.ones(n - 1), np.linspace(3, 6, n), -np.ones(n - 1)], offsets=[-1, 0, 1], format="csr")
    diagonal = np.linspace(0.5, 1.5, n)
    op = sl.aslinearoperator(a) + sl.LinearOperator(a.shape, matvec=lambda x: diagonal * x, dtype=a.dtype)
    inverse = 1 / (a.diagonal() + diagonal)
    preconditioner = sl.LinearOperator(a.shape, matvec=lambda x: inverse * x, dtype=a.dtype)
    b = np.random.default_rng(784).standard_normal(n)
    before = a.data.tobytes(), b.tobytes(), diagonal.tobytes(), inverse.tobytes()

    def run():
        history = []
        kwargs = {"callback_type": "legacy"} if solver is sl.gmres else {}
        result = solver(
            op,
            b,
            M=preconditioner,
            maxiter=maxiter,
            rtol=1e-10,
            callback=lambda x: history.append(np.array(x, copy=True)),
            **kwargs,
        )
        assert history
        assert result[1] == (0 if maxiter is None else 1)
        if maxiter == 1:
            assert len(history) == 1
        if result[1] == 0:
            # Evaluate the original equation, avoiding the composed operator.
            with fastscipy.disabled():
                residual = b - (a @ result[0] + diagonal * result[0])
                assert np.linalg.norm(residual) <= 1e-10 * np.linalg.norm(b)
        return result, history

    # Use the existing Intel Accelerate numerical contract for this solver
    # trajectory, without claiming observed reference variance or byte identity.
    # Assemble the equation independently of the composed LinearOperator.
    equation = a + sparse.diags_array(diagonal, format="csr")
    validation = _iteration_reference_validation(equation, b, 1e-10, scalar_history=solver is sl.gmres)
    check(run, numerical_validation=validation)
    assert before == (a.data.tobytes(), b.tobytes(), diagonal.tobytes(), inverse.tobytes())


@pytest.mark.parametrize("kind", ["batched", "integer", "column"])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_declined_vector_calls_preserve_callback_contract(kind, adjoint, fails):
    # Early rejection must still let SciPy own shape preparation, warnings,
    # callback errors, and the returned array's shape/dtype.
    calls = []
    error = RuntimeError("declined callback failed")

    def callback(x):
        calls.append((x.shape, x.dtype, x.copy()))
        warnings.warn("declined callback warning", UserWarning, stacklevel=2)
        if fails:
            raise error
        return x * 2

    op = sl.LinearOperator((4, 4), matvec=callback, rmatvec=callback, dtype=np.float64)
    x = {"batched": np.arange(8.0).reshape(2, 4), "integer": np.arange(4), "column": np.arange(4.0)[:, None]}[kind]
    before = x.tobytes()
    call = op.rmatvec if adjoint else op.matvec

    def invoke():
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            if fails:
                with pytest.raises(RuntimeError) as raised:
                    call(x)
                assert raised.value is error
                out = None
            else:
                out = call(x)
        return out, [(type(w.message), str(w.message)) for w in emitted]

    with fastscipy.disabled():
        expected, expected_warnings = invoke()
    reference_calls = calls[:]
    calls.clear()
    before_fast = fast_calls()
    actual, actual_warnings = invoke()
    assert_identical(expected, actual)
    assert_identical(reference_calls, calls)
    assert len(calls) == 1
    assert actual_warnings == expected_warnings
    assert actual_warnings.count((UserWarning, "declined callback warning")) == 1
    assert x.tobytes() == before
    assert fast_calls() == before_fast


def test_nonboolean_adjoint_delegates_without_extra_truth_evaluation():
    evaluations = []

    class Adjoint:
        def __bool__(self):
            evaluations.append(True)
            return True

    op = sl.LinearOperator((4, 4), matvec=lambda x: x * 2, rmatvec=lambda x: x * 3, dtype=np.float64)
    x = np.arange(4.0)
    for adjoint in (np.bool_(True), 1, Adjoint()):
        evaluations.clear()
        with fastscipy.disabled():
            expected = op._shared_matvec(x, adjoint)
        reference_evaluations = evaluations[:]
        evaluations.clear()
        before = fast_calls()
        actual = op._shared_matvec(x, adjoint)
        assert_identical(expected, actual)
        assert evaluations == reference_evaluations
        assert fast_calls() == before
