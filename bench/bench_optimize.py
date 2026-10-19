import numpy as np
from scipy import optimize

_x = np.random.default_rng(0).random(1000)


def cases():
    rng = np.random.default_rng(614)
    for m, n in ((12, 4), (40, 10), (100, 20)):
        A, b = rng.standard_normal((m, n)), rng.standard_normal(m)
        yield f"lsq_linear unbounded {m}x{n}", lambda A=A, b=b: optimize.lsq_linear(A, b)
    for n in (10, 100, 500):
        point = rng.standard_normal(n)
        yield f"rosen_hess n={n}", lambda point=point: optimize.rosen_hess(point)
    yield "approx_fprime rosen n=1000", lambda: optimize.approx_fprime(_x, optimize.rosen, 1e-8)
    yield "approx_fprime cheap sum-of-squares n=1000", lambda: optimize.approx_fprime(_x, lambda z: z @ z, 1e-8)
    x = _x[:10]
    yield "minimize BFGS rosen n=10 (numerical gradient)", lambda: optimize.minimize(optimize.rosen, x, method="BFGS").x
