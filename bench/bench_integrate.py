import numpy as np
from scipy import integrate

_y = np.random.default_rng(0).standard_normal(100)
_x = np.sort(np.random.default_rng(1).random(100))
_y2 = np.random.default_rng(2).standard_normal((1000, 30))


def cases():
    simpson = np.random.default_rng(920).standard_normal(100)
    yield "cumulative_simpson 100 uniform points", lambda: integrate.cumulative_simpson(simpson)
    large_simpson = np.random.default_rng(921).standard_normal(100_000)
    yield "cumulative_simpson 100000 uniform points", lambda: integrate.cumulative_simpson(large_simpson)
    multi_simpson = np.random.default_rng(922).standard_normal((64, 1000))
    yield "cumulative_simpson (64, 1000) uniform", lambda: integrate.cumulative_simpson(multi_simpson)
    for n in (17, 257, 4097):
        romberg = np.random.default_rng(923).standard_normal(n)
        yield f"romb {n} points", lambda y=romberg: integrate.romb(y)
    f32 = _y.astype(np.float32)
    yield "cumulative_trapezoid float32 100 points", lambda: integrate.cumulative_trapezoid(f32)
    yield "cumulative_trapezoid (1000, 30) axis=0", lambda: integrate.cumulative_trapezoid(_y2, axis=0)
    yield "cumulative_trapezoid (16, 30) axis=0", lambda: integrate.cumulative_trapezoid(_y2[:16], axis=0)
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
