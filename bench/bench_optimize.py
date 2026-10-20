import numpy as np
from scipy import optimize

_x = np.random.default_rng(0).random(1000)


def _bfgs_reference_validation(expected, actual):
    # Accelerate's alignment-sensitive dot products can change even plain
    # BFGS's path. Check this known minimizer independently, never its timing.
    for point in (expected, actual):
        assert type(point) is np.ndarray and point.dtype == np.float64 and point.shape == (10,)
        assert np.isfinite(point).all()
        assert np.max(np.abs(point - 1)) < 2e-4
        assert optimize.rosen(point) < 1e-8


def cases():
    derivative_rng = np.random.default_rng(984)
    for n in (10, 1000):
        point, vector = derivative_rng.standard_normal(n), derivative_rng.standard_normal(n)
        yield f"rosen_der n={n}", lambda point=point: optimize.rosen_der(point)
        yield f"rosen_hess_prod n={n}", lambda point=point, vector=vector: optimize.rosen_hess_prod(point, vector)
    initial = np.array([-1.2, 1.0, 0.8])
    yield (
        "minimize Newton-CG rosen analytic jac+hessp n=3",
        lambda: optimize.minimize(
            optimize.rosen, initial, method="Newton-CG", jac=optimize.rosen_der, hessp=optimize.rosen_hess_prod
        ),
    )
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
    yield (
        "minimize BFGS rosen n=10 (numerical gradient)",
        lambda: optimize.minimize(optimize.rosen, x, method="BFGS").x,
        True,
        _bfgs_reference_validation,
    )
