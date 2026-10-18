import numpy as np
from scipy import integrate

_y = np.random.default_rng(0).standard_normal(100)
_x = np.sort(np.random.default_rng(1).random(100))
_y2 = np.random.default_rng(2).standard_normal((1000, 30))


def cases():
    yield "trapezoid 100 points with x, x100 calls", lambda: [integrate.trapezoid(_y, _x) for _ in range(100)]
    yield "trapezoid 100 points dx, x100 calls", lambda: [integrate.trapezoid(_y, dx=0.1) for _ in range(100)]
    yield "trapezoid (1000, 30) axis=0 x100 calls", lambda: [integrate.trapezoid(_y2, axis=0) for _ in range(100)]
