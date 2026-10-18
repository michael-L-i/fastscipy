import numpy as np
from scipy import optimize

_x = np.random.default_rng(0).random(1000)


def cases():
    yield "approx_fprime rosen n=1000", lambda: optimize.approx_fprime(_x, optimize.rosen, 1e-8)
    yield "approx_fprime cheap sum-of-squares n=1000", lambda: optimize.approx_fprime(_x, lambda z: z @ z, 1e-8)
    x = _x[:10]
    yield "minimize BFGS rosen n=10 (numerical gradient)", lambda: optimize.minimize(optimize.rosen, x, method="BFGS").x
