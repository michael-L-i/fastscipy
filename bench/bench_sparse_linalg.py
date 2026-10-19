import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sl


def cases():
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
