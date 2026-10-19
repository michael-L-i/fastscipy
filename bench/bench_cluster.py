"""Vector quantization, feature normalization, and hierarchical workflows."""

import numpy as np
from scipy.cluster import hierarchy, vq


def cases():
    extra = np.random.default_rng(850)
    for shape, dtype in (
        ((100,), np.float64),
        ((200, 8), np.complex64),
        ((200, 8), np.complex128),
        ((30, 4, 3), np.float32),
    ):
        obs = extra.standard_normal(shape).astype(dtype)
        if np.iscomplexobj(obs):
            obs += 1j * extra.standard_normal(shape)
        yield f"whiten {np.dtype(dtype)} {shape}", lambda a=obs: vq.whiten(a)
    rng = np.random.default_rng(0)
    x = rng.random((200, 8))
    book = x[::20].copy()
    yield "vq 200x8, 10 centroids", lambda: vq.vq(x, book)
    yield "vq 200x8, finite check disabled", lambda: vq.vq(x, book, check_finite=False)
    yield "whiten 200x8", lambda: vq.whiten(x)
    yield "kmeans 200x8, explicit centroids", lambda: vq.kmeans(x, book, iter=1)
    for n in (20, 2000):
        z = np.empty((n, 4))
        z[:, 0], z[:, 1] = np.arange(1, n + 1), np.arange(n, 2 * n)
        z[0, :2] = 0, 1
        z[:, 2], z[:, 3] = np.arange(1, n + 1), np.arange(2, n + 2)
        yield f"is_valid_linkage {n} merges", lambda z=z: hierarchy.is_valid_linkage(z)
        yield f"leaves_list {n} merges", lambda z=z: hierarchy.leaves_list(z)
