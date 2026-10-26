"""NumPy vector calls through LinearOperator, including sparse iterative solvers.

The unbatched case avoids array conversion and broadcast-shape calculation.
SciPy still runs the operators, solvers, and callbacks. Floating-point
reproducibility depends on the installed numerical backend.
"""

import functools

import numpy as np
from scipy import sparse
from scipy._lib._array_api import array_namespace
from scipy.sparse.linalg import LinearOperator, _interface

from . import _patch

_NP = array_namespace(np.empty(0))
_OPERATORS = frozenset(
    cls for cls in vars(_interface).values() if isinstance(cls, type) and issubclass(cls, LinearOperator)
)
_COMPRESSED = (sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix)
_MATRIX_OPERATORS = (_interface.MatrixLinearOperator, _interface._AdjointMatrixOperator)


def patch():
    _patch.require_validated("1.18.")
    original = LinearOperator._shared_matvec
    original_matvec = LinearOperator._matvec

    # A callback must never run twice, even when it raises or returns a bad shape.
    @functools.wraps(original)
    def matvec(self, x, adjoint=False):
        if not _patch.enabled or type(x) is not np.ndarray:
            return original(self, x, adjoint)
        # Batched and column inputs delegate before inspecting operator metadata.
        shape = x.shape
        if (
            len(shape) != 1
            or x.dtype.kind not in "fc"
            or type(self) not in _OPERATORS
            or self._xp is not _NP
            or len(self.shape) != 2
            or (adjoint is not False and adjoint is not True)
        ):
            return original(self, x, adjoint)
        outer, inner = self.shape
        if adjoint:
            inner, outer = outer, inner
        if shape[0] != inner:
            return original(self, x, adjoint)
        _patch.counts["LinearOperator._shared_matvec", "fast"] += 1
        y = self._rmatvec(x) if adjoint else self._matvec(x)
        # NumPy's array-api wrapper ultimately calls this same method. Keep
        # subclass/array-like dispatch, and still return the reshaped view.
        if type(y) is np.ndarray:
            return y.reshape((outer,))
        return _NP.reshape(y, (outer,))

    _patch.replace(LinearOperator, "_shared_matvec", matvec)

    @functools.wraps(original_matvec)
    def matrix_matvec(self, x):
        if (
            not _patch.enabled
            or type(self) not in _MATRIX_OPERATORS
            or self._xp is not _NP
            or type(self.A) not in _COMPRESSED
            or self.A.ndim != 2
            or not self.A.shape[0]
            or type(x) is not np.ndarray
            or x.ndim != 1
            or not x.size
            or x.shape[0] != self.A.shape[1]
        ):
            return original_matvec(self, x)
        # The default inserts a column axis, dispatches A @ column, and removes
        # that axis again. Its CSR/CSC branch calls this exact kernel on ravel().
        _patch.counts["MatrixLinearOperator sparse matvec", "fast"] += 1
        return self.A._matmul_vector(x.ravel())

    _patch.replace(LinearOperator, "_matvec", matrix_matvec)


def smoke():
    from scipy.sparse import diags_array
    from scipy.sparse.linalg import aslinearoperator, cg

    a = diags_array([np.full(5, -1.0), np.full(6, 3.0), np.full(5, -1.0)], offsets=[-1, 0, 1], format="csr")
    cg(a, np.arange(6.0))
    aslinearoperator(a).rmatvec(np.arange(6.0))
