import numpy as np
from scipy import integrate

_y = np.random.default_rng(0).standard_normal(100)
_x = np.sort(np.random.default_rng(1).random(100))
_y2 = np.random.default_rng(2).standard_normal((1000, 30))


def cases():
    simpson_rng = np.random.default_rng(995)
    for n in (100, 101):
        irregular_y = simpson_rng.standard_normal(n)
        irregular_x = np.cumsum(simpson_rng.uniform(0.1, 1.0, n))
        yield f"simpson {n} irregular points", lambda y=irregular_y, x=irregular_x: integrate.simpson(y, x=x)
    for dtype in (np.float32, np.complex64, np.complex128):
        for n in (100, 101):
            typed_y = simpson_rng.standard_normal(n).astype(dtype)
            if typed_y.dtype.kind == "c":
                typed_y += 1j * simpson_rng.standard_normal(n).astype(dtype)
            yield f"simpson {n} uniform {np.dtype(dtype).name} points", lambda y=typed_y: integrate.simpson(y)
    wave_rng = np.random.default_rng(970)
    for n in (100, 100_001):
        irregular_y = wave_rng.standard_normal(n)
        irregular_x = np.cumsum(wave_rng.uniform(0.01, 1.0, n))
        yield (
            f"cumulative_simpson {n} irregular points",
            lambda y=irregular_y, x=irregular_x: integrate.cumulative_simpson(y, x=x),
        )
    irregular_channels = wave_rng.standard_normal((32, 1000))
    channel_x = np.cumsum(wave_rng.uniform(0.01, 1.0, 1000))
    yield (
        "cumulative_simpson (32, 1000) irregular points",
        lambda: integrate.cumulative_simpson(irregular_channels, x=channel_x),
    )
    odd_control = wave_rng.standard_normal(101)
    yield "cumulative_simpson 101 uniform points", lambda: integrate.cumulative_simpson(odd_control)
    even = wave_rng.standard_normal(100)
    yield "simpson 100 uniform even points", lambda: integrate.simpson(even)
    yield "simpson 2 uniform points", lambda: integrate.simpson(even[:2])
    for dtype in (np.float32, np.complex128):
        typed = wave_rng.standard_normal(257).astype(dtype)
        if dtype == np.complex128:
            typed += 1j * wave_rng.standard_normal(257)
        yield f"romb 257 {np.dtype(dtype).name} points", lambda y=typed: integrate.romb(y)
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
