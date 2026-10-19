import numpy as np
from scipy import integrate

_y = np.random.default_rng(0).standard_normal(100)
_x = np.sort(np.random.default_rng(1).random(100))
_y2 = np.random.default_rng(2).standard_normal((1000, 30))


def cases():
    odd = np.random.default_rng(21).standard_normal(101)
    yield "simpson 101 uniform points", lambda: integrate.simpson(odd)
    odd2 = np.random.default_rng(22).standard_normal((101, 30))
    yield "simpson (101, 30) uniform axis=0", lambda: integrate.simpson(odd2, axis=0)
    yield "fixed_quad cosine 5 nodes", lambda: integrate.fixed_quad(np.cos, 0.0, 1.0)
    yield (
        "fixed_quad vector 16 nodes",
        lambda: integrate.fixed_quad(lambda y: np.stack((np.sin(y), y**2)), 0.0, 1.0, n=16),
    )
    yield "trapezoid 100 points with x, x100 calls", lambda: [integrate.trapezoid(_y, _x) for _ in range(100)]
    yield "trapezoid 100 points dx, x100 calls", lambda: [integrate.trapezoid(_y, dx=0.1) for _ in range(100)]
    yield "trapezoid (1000, 30) axis=0 x100 calls", lambda: [integrate.trapezoid(_y2, axis=0) for _ in range(100)]
    yield "cumulative_trapezoid 100 points dx", lambda: integrate.cumulative_trapezoid(_y, dx=0.1)
    yield "cumulative_trapezoid 100 points with x", lambda: integrate.cumulative_trapezoid(_y, _x)
    yield "cumulative_trapezoid 100 points initial=0", lambda: integrate.cumulative_trapezoid(_y, initial=0)
