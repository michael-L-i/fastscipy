import numpy as np
import scipy.sparse as sp


def cases():
    sweep_rng = np.random.default_rng(23)
    for fmt, name in (("csr", "hstack"), ("csc", "vstack")):
        cls = getattr(sp, fmt + "_array")
        for side in (8, 300):
            pair = [cls(sweep_rng.random((side, side)) * (sweep_rng.random((side, side)) < 0.1)) for _ in range(2)]
            yield (
                f"{name} pair of {fmt.upper()} arrays {side}x{side}",
                lambda pair=pair, name=name: getattr(sp, name)(pair),
            )
    mixed = [getattr(sp, ("csr" if i % 2 else "csc") + "_array")(sweep_rng.random((20, 20))) for i in range(10)]
    for fmt in ("csr", "csc"):
        yield f"block_diag mixed CSR/CSC to {fmt.upper()}", lambda fmt=fmt: sp.block_diag(mixed, format=fmt)
    construction_rng = np.random.default_rng(22)
    for fmt in ("csr", "csc"):
        cls = getattr(sp, fmt + "_array")
        blocks = [cls(construction_rng.standard_normal((4, 4))) for _ in range(50)]
        yield (
            f"block_diag 50 small {fmt.upper()} arrays",
            lambda blocks=blocks, fmt=fmt: sp.block_diag(blocks, format=fmt),
        )
        medium = [
            cls(construction_rng.random((100, 100)) * (construction_rng.random((100, 100)) < 0.1)) for _ in range(5)
        ]
        yield (
            f"block_diag 5 medium {fmt.upper()} arrays",
            lambda blocks=medium, fmt=fmt: sp.block_diag(blocks, format=fmt),
        )
    rng = np.random.default_rng(0)
    # many small matrices
    data, ind, ptr = (
        rng.random(30),
        rng.integers(0, 10, 30).astype(np.int32),
        np.array([0, 3, 9, 12, 12, 15, 18, 22, 25, 28, 30], np.int32),
    )
    r, c = rng.integers(0, 10, 10), rng.integers(0, 10, 10)
    yield (
        "2000 x csr_array((data, indices, indptr)) 10x10",
        lambda: [sp.csr_array((data, ind, ptr), shape=(10, 10)) for _ in range(2000)],
    )
    yield "2000 x csr_array((10, 10))", lambda: [sp.csr_array((10, 10)) for _ in range(2000)]
    yield (
        "2000 x coo->csr_array((v, (i, j)))",
        lambda: [sp.csr_array((data[:10], (r, c)), shape=(10, 10)) for _ in range(2000)],
    )
    small = sp.csr_array((data, ind, ptr), shape=(10, 10))
    yield "2000 x small A.T + A", lambda: [small.T + small for _ in range(2000)]

    canonical = sp.random_array((2000, 2000), density=0.025, format="csr", rng=9)
    yield "find canonical CSR 100k nnz", lambda: sp.find(canonical)

    # argmax / argmin on a 1e7-nnz CSR (50000 x 100000, 200 per row, no duplicates)
    cols = (np.arange(200) * 500 + rng.integers(0, 500, (50_000, 200))).astype(np.int32).ravel()
    big = sp.csr_array(
        (rng.standard_normal(10_000_000), cols, np.arange(0, 10_000_001, 200, dtype=np.int32)), shape=(50_000, 100_000)
    )
    del cols
    yield "csr 1e7 nnz argmax(axis=1)", lambda big=big: big.argmax(axis=1)
    yield "csr 1e7 nnz argmin(axis=1) explicit", lambda big=big: big.argmin(axis=1, explicit=True)
    yield "csc 1e7 nnz argmax(axis=0)", lambda big=big: big.tocsc().argmax(axis=0)
    del big

    # coo.sum_duplicates: 2e6 entries (SciPy needs 9 s at 1e7)
    n = 2_000_000
    row, col, val = rng.integers(0, 20_000, n), rng.integers(0, 20_000, n), rng.standard_normal(n)

    def sumdup():
        m = sp.coo_array((val, (row, col)), shape=(20_000, 20_000))
        m.sum_duplicates()
        return m

    yield "coo 2e6 entries sum_duplicates", sumdup
    dup = rng.integers(0, 300, 500_000)
    yield (
        "coo 5e5 entries on 300x300, heavy duplicates",
        lambda: (lambda m: (m.sum_duplicates(), m)[1])(
            sp.coo_array((val[:500_000], (dup, dup[::-1].copy())), shape=(300, 300))
        ),
    )

    # find
    a = sp.random_array((20_000, 20_000), density=0.0045, format="csr", rng=1)
    yield "find csr 20000^2, 1.8M nnz", lambda: sp.find(a)
    perm = rng.permutation(a.nnz)
    o = a.tocoo()
    shuffled = sp.coo_array(
        (
            np.concatenate([o.data, o.data[:1000]])[np.r_[perm, np.arange(a.nnz, a.nnz + 1000)]],
            (
                np.concatenate([o.row, o.row[:1000]])[np.r_[perm, np.arange(a.nnz, a.nnz + 1000)]],
                np.concatenate([o.col, o.col[:1000]])[np.r_[perm, np.arange(a.nnz, a.nnz + 1000)]],
            ),
        ),
        shape=a.shape,
    )
    yield "find coo (shuffled, duplicates) 20000^2, 1.8M nnz", lambda: sp.find(shuffled)

    # dense -> sparse
    d = rng.standard_normal((4000, 4000)) * (rng.random((4000, 4000)) < 0.16)
    for f in (sp.csr_array, sp.csc_array, sp.coo_array, sp.csr_matrix):
        yield f"{f.__name__}(dense 4000^2, 16%)", lambda f=f: f(d)
    yield "csr_array(dense 100x100)", lambda x=d[:100, :100].copy(): [sp.csr_array(x) for _ in range(200)]
