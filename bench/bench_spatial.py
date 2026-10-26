import numpy as np
import scipy.spatial as sp
import scipy.spatial.distance as dist

THREADED_ONLY = True  # large distance and tree cases; small pdist cases override this
rng = np.random.default_rng(0)


def cases():
    # Local neighborhoods: repeated small condensed distances in clustering,
    # point-cloud descriptors, and nearest-neighbor candidate reranking.
    local_rng = np.random.default_rng(842)
    for rows, dims in ((4, 3), (16, 3), (32, 16), (100, 16)):
        local = local_rng.random((rows, dims))
        for metric in ("euclidean", "cityblock", "sqeuclidean"):
            yield (
                f"small pdist {metric} {rows}x{dims}",
                lambda x=local, m=metric: dist.pdist(x, m),
                True,
            )
    A, B = rng.random((4000, 16)), rng.random((2000, 16))
    for metric in ("euclidean", "cityblock", "cosine"):
        yield f"cdist {metric} 4000x2000x16", lambda m=metric: dist.cdist(A, B, m)
    X = rng.random((3000, 16))
    for metric in ("euclidean", "cityblock"):
        yield f"pdist {metric} 3000x16", lambda m=metric: dist.pdist(X, m)
    af, bf, xf = np.asfortranarray(A), np.asfortranarray(B), np.asfortranarray(X)
    yield "cdist euclidean Fortran 4000x2000x16", lambda: dist.cdist(af, bf)
    yield "pdist euclidean Fortran 3000x16", lambda: dist.pdist(xf)
    yield "cdist euclidean strided 4000x2000x8", lambda: dist.cdist(A[:, ::2], B[:, ::2])
    extra = np.random.default_rng(817)
    sa, sb = extra.normal(size=(1500, 12)), extra.normal(size=(1000, 12))
    for metric in ("seuclidean", "mahalanobis"):
        yield f"cdist {metric} 1500x1000x12", lambda m=metric: dist.cdist(sa, sb, m)
        yield f"pdist {metric} 1500x12", lambda m=metric: dist.pdist(sa, m)
    sweep = np.random.default_rng(825)
    wa, wb, weights = sweep.random((1200, 16)), sweep.random((900, 16)), sweep.random(16)
    for metric, options in (("euclidean", {}), ("minkowski", {"p": 3.5})):
        yield (
            f"weighted cdist {metric} 1200x900x16",
            lambda m=metric, opts=options: dist.cdist(wa, wb, m, w=weights, **opts),
        )
        yield f"weighted pdist {metric} 1200x16", lambda m=metric, opts=options: dist.pdist(wa, m, w=weights, **opts)
    fa, fb = wa.astype(np.float32), wb.astype(np.float32)
    yield "float32 cdist euclidean 1200x900x16", lambda: dist.cdist(fa, fb)
    yield "float32 pdist euclidean 1200x16", lambda: dist.pdist(fa)
    # Low-dimensional clouds expose distance temporary memory and copy cost.
    cloud = np.random.default_rng(845).random((4000, 3))
    # With two threads these cheap pairs stay on SciPy; four can still split.
    yield "pdist euclidean 4000x3", lambda: dist.pdist(cloud), False
    for metric in ("sqeuclidean", "cityblock"):
        yield f"pdist {metric} 4000x3 (2-thread control)", lambda m=metric: dist.pdist(cloud, m), False
    pts = rng.random((1_000_000, 3))
    tree, q = sp.cKDTree(pts), rng.random((200_000, 3))
    for k in (1, 8):
        yield f"cKDTree.query 1M pts, 200k queries, k={k}", lambda k=k: tree.query(q, k)
    yield "cKDTree.query_ball_point 200k, r=0.01", lambda: tree.query_ball_point(q, 0.01, return_length=True)
    small, ssmall = rng.random((300, 16)), rng.random((100, 16))  # below the thresholds
    yield "cdist 300x100x16 (control)", lambda: dist.cdist(small, ssmall), False
    sfloat = ssmall.astype(np.float32)
    yield "pdist float32 100x16 (control)", lambda: dist.pdist(sfloat), False
    yield "cKDTree.query 100 queries (control)", lambda: tree.query(q[:100], 4), False
