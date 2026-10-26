"""scipy.interpolate lookups on random queries (and sorted ones, where we expect ~1x)."""

import numpy as np
import scipy.interpolate as si
import scipy.spatial as sp


def cases():
    r = np.random.default_rng(0)

    short_rng = np.random.default_rng(51)
    short_x = np.linspace(0, 1, 30)
    short_spline = si.make_interp_spline(short_x, short_rng.standard_normal(30))
    short_q = short_rng.random(1000)
    yield "BSpline 30 knots, 1000 random queries", lambda: short_spline(short_q), False

    piece_rng = np.random.default_rng(208)
    piece_x = np.linspace(0, 1, 100)
    for cls in (si.PPoly, si.BPoly):
        piece = cls(piece_rng.standard_normal((4, 99)), piece_x)
        for nq in (1, 10, 100):
            piece_q = piece_rng.random(nq)
            yield f"{cls.__name__} 100 knots, {nq} queries", lambda p=piece, q=piece_q: p(q)
        piece_q32 = piece_rng.random(10).astype(np.float32)
        piece_stride = piece_rng.random(20)[::2]
        yield f"{cls.__name__} 100 knots, 10 float32 queries", lambda p=piece, q=piece_q32: p(q)
        yield f"{cls.__name__} 100 knots, 10 strided queries", lambda p=piece, q=piece_stride: p(q)
        yield f"{cls.__name__} 100 knots, Python scalar query", lambda p=piece: p(0.25)
        yield f"{cls.__name__} 100 knots, float32 scalar query", lambda p=piece: p(np.float32(0.25))

    t = np.sort(r.random(1000))
    spl = si.make_interp_spline(t, r.standard_normal(1000), k=3)
    q = r.uniform(t[0], t[-1], 1_000_000)
    yield "BSpline 1000 knots, 1e6 random queries", lambda: spl(q)
    yield "BSpline 1000 knots, 1e6 sorted queries", lambda qs=np.sort(q): spl(qs), False
    yield "splev 1000 knots, 1e6 random queries", lambda: si.splev(q, spl.tck)
    yield "interp1d cubic, 1e6 random queries", lambda f=si.interp1d(t, spl.c, kind="cubic"): f(q)
    spl3 = si.make_interp_spline(t, r.standard_normal((1000, 3)), k=3)
    q3 = q[:200_000]
    yield "BSpline 3 values, 2e5 random queries", lambda: spl3(q3)
    yield "BSpline 2e5 queries, derivative", lambda: spl(q3, nu=1)

    pts = r.random((100_000, 2))
    tri = sp.Delaunay(pts)
    xq = r.random((100_000, 2))
    row = np.floor(xq[:, 1] * 300)
    xs = xq[np.lexsort((np.where(row % 2 == 0, xq[:, 0], -xq[:, 0]), row))]  # snake raster order
    yield "Delaunay.find_simplex 1e5 points, 1e5 random queries", lambda: tri.find_simplex(xq)
    yield "Delaunay.find_simplex 1e5 points, 1e5 raster-ordered queries", lambda: tri.find_simplex(xs), False
    lin = si.LinearNDInterpolator(tri, np.sin(5 * pts[:, 0]) + pts[:, 1])
    yield "LinearNDInterpolator 1e5 points, 1e5 random queries", lambda: lin(xq)
    yield "LinearNDInterpolator raster-ordered queries", lambda: lin(xs), False
    ct = si.CloughTocher2DInterpolator(tri, np.sin(5 * pts[:, 0]) + pts[:, 1])
    yield "CloughTocher2DInterpolator 1e5 points, 1e5 random queries", lambda: ct(xq)

    many_rng = np.random.default_rng(32)
    many_pts = many_rng.random((6000, 2))
    many_q = many_rng.random((24001, 2))
    many = si.CloughTocher2DInterpolator(many_pts, many_rng.standard_normal((6000, 128)))
    yield "CloughTocher2DInterpolator 128 outputs, 24001 queries", lambda: many(many_q)

    p2, v2 = pts[:20_000], np.sin(5 * pts[:20_000, 0])
    yield "griddata linear 2e4 points, 1e5 queries", lambda: si.griddata(p2, v2, xq, method="linear")
    yield "griddata cubic 2e4 points, 1e5 queries", lambda: si.griddata(p2, v2, xq, method="cubic")

    g = np.sort(r.random(512)), np.sort(r.random(512))
    rgi = si.RegularGridInterpolator(g, r.standard_normal((512, 512)), method="pchip", bounds_error=False)
    q2 = r.random((5000, 2))
    yield "RegularGridInterpolator pchip 512x512, 5000 queries", lambda: rgi(q2)
    g3 = [np.sort(r.random(64)) for _ in range(3)]
    rgi3 = si.RegularGridInterpolator(g3, r.standard_normal((64,) * 3), method="pchip", bounds_error=False)
    q3d = r.random((2000, 3))
    yield "RegularGridInterpolator pchip 64^3, 2000 queries", lambda: rgi3(q3d)

    for grid_shape, channel_shape in (((12,), (8,)), ((4096,), (8,)), ((64, 64), (3,)), ((12, 12, 12), (2, 3))):
        ndim = len(grid_shape)
        channel_grid = tuple(np.linspace(0, 1, n) for n in grid_shape)
        channel_values = r.standard_normal(grid_shape + channel_shape)
        field = si.RegularGridInterpolator(channel_grid, channel_values, method="pchip")
        queries = r.random((1000, ndim))
        yield (
            f"RegularGridInterpolator pchip grid {grid_shape} channels {channel_shape}, 1000 queries",
            lambda f=field, q=queries: f(q),
        )
