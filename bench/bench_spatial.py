import numpy as np
import scipy.spatial as sp
import scipy.spatial.distance as dist

THREADED_ONLY = True  # the fast paths only thread; they decline on 1 thread
rng = np.random.default_rng(0)


def cases():
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
    pts = rng.random((1_000_000, 3))
    tree, q = sp.cKDTree(pts), rng.random((200_000, 3))
    for k in (1, 8):
        yield f"cKDTree.query 1M pts, 200k queries, k={k}", lambda k=k: tree.query(q, k)
    yield "cKDTree.query_ball_point 200k, r=0.01", lambda: tree.query_ball_point(q, 0.01, return_length=True)
    small, ssmall = rng.random((300, 16)), rng.random((100, 16))  # below the thresholds
    yield "cdist 300x100x16 (control)", lambda: dist.cdist(small, ssmall), False
    yield "pdist 100x16 (control)", lambda: dist.pdist(ssmall), False
    yield "cKDTree.query 100 queries (control)", lambda: tree.query(q[:100], 4), False
