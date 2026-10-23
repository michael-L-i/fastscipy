import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sl


def cases():
    adj_rng = np.random.default_rng(343)
    for dtype in (np.float64, np.complex128):
        for n in (32, 128):
            adj_a = sparse.diags_array(
                [np.full(n - 1, -0.5), np.arange(n) + 2.0, np.full(n - 1, -0.5)],
                offsets=[-1, 0, 1],
                format="csr",
            ).astype(dtype)
            if np.dtype(dtype).kind == "c":
                adj_a.data += 1j * adj_rng.random(adj_a.data.shape) * 0.1
            adj_b = adj_rng.standard_normal(n).astype(dtype)
            adj_op = sl.aslinearoperator(adj_a)
            tag = f"{np.dtype(dtype).name} n={n}"
            yield (
                f"LinearOperator CSR rmatvec {tag} x100 calls",
                lambda o=adj_op, b=adj_b: [o.rmatvec(b) for _ in range(100)],
            )
            for solver in (sl.lsqr, sl.lsmr):
                yield f"{solver.__name__} tridiagonal {tag}", lambda f=solver, a=adj_a, b=adj_b: f(a, b)
    r = np.random.default_rng(19)
    for n in (32, 256):
        a = sparse.diags_array(
            [np.full(n - 1, -0.5), np.arange(n) + 2.0, np.full(n - 1, -0.5)], offsets=[-1, 0, 1], format="csr"
        )
        b = r.standard_normal(n)
        operator = sl.aslinearoperator(a)
        yield f"LinearOperator CSR matvec n={n} x100 calls", lambda o=operator, b=b: [o.matvec(b) for _ in range(100)]
        for solver in (sl.cg, sl.bicgstab, sl.gmres):
            yield f"{solver.__name__} tridiagonal n={n}", lambda f=solver, a=a, b=b: f(a, b, rtol=1e-9)

    # Matrix-free diagonal regularization and Jacobi preconditioning exercise
    # repeated operator dispatch inside a solve, as well as isolated overhead.
    for n in (32, 256, 4096):
        a = sparse.diags_array(
            [-np.ones(n - 1), np.linspace(3, 6, n), -np.ones(n - 1)], offsets=[-1, 0, 1], format="csr"
        )
        b = r.standard_normal(n)
        diagonal = np.linspace(0.5, 1.5, n)
        regularizer = sl.LinearOperator(a.shape, matvec=lambda x, d=diagonal: d * x, dtype=a.dtype)
        operator = sl.aslinearoperator(a) + regularizer
        inverse = 1 / (a.diagonal() + diagonal)
        preconditioner = sl.LinearOperator(a.shape, matvec=lambda x, d=inverse: d * x, dtype=a.dtype)
        yield (
            f"LinearOperator composed regularization n={n} x100 calls",
            lambda o=operator, b=b: [o.matvec(b) for _ in range(100)],
        )
        for solver in (sl.cg, sl.gmres):
            yield (
                f"{solver.__name__} regularized Jacobi n={n}",
                lambda f=solver, o=operator, p=preconditioner, b=b: f(o, b, M=p, rtol=1e-9),
            )
    custom = sl.LinearOperator((32, 32), matvec=lambda x: x * 2, dtype=np.float64)
    for label, x in (("integer", np.arange(32)), ("batched", np.ones((2, 32)))):
        yield (
            f"LinearOperator {label} fallback control x100 calls",
            lambda o=custom, x=x: [o.matvec(x) for _ in range(100)],
            False,
        )
