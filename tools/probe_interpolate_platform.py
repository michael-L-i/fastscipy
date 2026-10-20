"""Bounded arithmetic diagnostics for an installed SciPy/fastscipy wheel.

No builds or large arrays: separate basis-recurrence contraction from coefficient
reduction, then identify scalar/unrolled reductions that explain native misses.
These candidates diagnose compiler choices; they do not alter dispatch.
"""

import functools
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

import fastscipy
from fastscipy import _interpolate as interp


def basis(t, q, k, mask):
    n = len(t) - k - 1
    rows = []
    for x in q:
        ell = max(k, min(n - 1, int(np.searchsorted(t, x, side="right")) - 1))
        h = [1.0] + [0.0] * k
        for j in range(1, k + 1):
            hh = h[:j]
            h[0] = 0.0
            for r in range(1, j + 1):
                xb, xa = float(t[ell + r]), float(t[ell + r - j])
                if xb == xa:
                    h[r] = 0.0
                    continue
                w = hh[r - 1] / (xb - xa)
                h[r - 1] = math.fma(w, xb - x, h[r - 1]) if mask & (1 << (j - 1)) else w * (xb - x) + h[r - 1]
                h[r] = w * (x - xa)
        rows.append((ell - k, h))
    return rows


def serial(c, h, mask, reverse=False):
    value = 0.0
    for a in reversed(range(len(h))) if reverse else range(len(h)):
        value = math.fma(c[a], h[a], value) if mask & (1 << a) else c[a] * h[a] + value
    return value


def lanes(c, h, width, fused, reverse=False):
    sums = [0.0] * width
    for a in range(len(h)):
        lane = a % width
        sums[lane] = math.fma(c[a], h[a], sums[lane]) if fused else c[a] * h[a] + sums[lane]
    value = 0.0
    for x in reversed(sums) if reverse else sums:
        value += x
    return value


@functools.lru_cache(maxsize=None)
def trees(start, end):
    if end - start == 1:
        return (start,)
    return tuple((a, b) for mid in range(start + 1, end) for a in trees(start, mid) for b in trees(mid, end))


def tree_sum(products, tree):
    if isinstance(tree, int):
        return products[tree]
    return tree_sum(products, tree[0]) + tree_sum(products, tree[1])


def candidates(k):
    for reverse in (False, True):
        for mask in range(1 << (k + 1)):
            yield f"serial reverse={reverse} fma-mask={mask}", lambda c, h, m=mask, r=reverse: serial(c, h, m, r)
    for width in (2, 4):
        for fused in (False, True):
            for reverse in (False, True):
                yield (
                    f"lanes={width} fused={fused} reverse={reverse}",
                    lambda c, h, w=width, f=fused, r=reverse: lanes(c, h, w, f, r),
                )
    for tree in trees(0, k + 1):
        yield f"product-tree={tree}", lambda c, h, tr=tree: tree_sum([a * b for a, b in zip(c, h, strict=True)], tr)


def mismatch(a, b):
    return int(np.count_nonzero(np.ascontiguousarray(a).view(np.uint64) != np.ascontiguousarray(b).view(np.uint64)))


def reference(backend, t, c, k, q):
    if backend == "bspline":
        return interp._orig["numpy_bspline"](t, c, k)(q)
    return np.column_stack([interp._orig["splev"](q, (t, col, k)) for col in c.T])


def probe(backend, k, seed):
    rng = np.random.default_rng(seed)
    knots = np.sort(rng.random(32))
    t = np.r_[[knots[0]] * k, knots, [knots[-1]] * k]
    q = np.r_[rng.uniform(-0.2, 1.2, 48), t, np.nextafter(t, -np.inf), np.nextafter(t, np.inf)]
    # Python scalar arithmetic must not invoke NumPy's arithmetic implementations.
    q_list = [float(x) for x in q]
    n = len(t) - k - 1
    identity = np.eye(n)
    ref_basis = reference(backend, t, identity, k, q)
    # Retain tiny target-platform references for additional local arithmetic
    # experiments without rebuilding a wheel or rerunning the remote job.
    artifacts = Path("artifacts/interpolate-probe")
    artifacts.mkdir(parents=True, exist_ok=True)
    saved = {"t": t, "q": q, "basis": ref_basis}
    fits = []
    scores = []
    for mask in range(1 << k):
        rows = basis(t, q_list, k, mask)
        out = np.zeros_like(ref_basis)
        for i, (start, h) in enumerate(rows):
            out[i, start : start + k + 1] = h
        # Identity coefficients can change signed zero; compare nonzero basis values.
        differences = int(np.count_nonzero(out != ref_basis))
        scores.append((differences, mask))
        if differences == 0:
            fits.append((mask, rows))
    print(
        backend,
        "degree",
        k,
        "seed",
        seed,
        "basis masks",
        [mask for mask, _ in fits],
        "best",
        sorted(scores)[:4],
        flush=True,
    )
    # Even an unrecognized basis recurrence must not block diagnosing the
    # coefficient reduction independently. Mask -1 means target-derived basis.
    target_rows = []
    for i, x in enumerate(q):
        ell = max(k, min(n - 1, int(np.searchsorted(t, x, side="right")) - 1))
        target_rows.append((ell - k, [float(v) for v in ref_basis[i, ell - k : ell + 1]]))
    fits.append((-1, target_rows))
    widths = (1, 2, 3, 4, 5, 6, 7, 8) if k == 5 else (1, 3, 8)
    for width in widths if backend == "bspline" else (1,):
        c = rng.standard_normal((n, width))
        ref = reference(backend, t, c, k, q)
        saved[f"c{width}"] = c
        saved[f"ref{width}"] = ref
        np.savez(artifacts / f"{backend}-k{k}-seed{seed}.npz", **saved)
        mixed = np.empty_like(ref)
        mask = ((1 << (k + 1)) - 1) & (~15 if width == 1 else -1)
        for i, (start, h) in enumerate(basis(t, q_list, k, (1 << k) - 1)):
            for col in range(width):
                mixed[i, col] = serial([float(v) for v in c[start : start + k + 1, col]], h, mask)
        print("proposed mixed mode4", backend, k, seed, width, "mismatches", mismatch(ref, mixed), flush=True)
        native = [(mode, mismatch(ref, interp._k.bspline_eval(t, c, k, q, mode))) for mode in range(4)]
        winners = []
        best = []
        for mask, rows in fits:
            for label, reduce in candidates(k):
                out = np.empty_like(ref)
                for i, (start, h) in enumerate(rows):
                    for column in range(width):
                        values = [float(x) for x in c[start : start + k + 1, column]]
                        out[i, column] = reduce(values, h)
                differences = mismatch(ref, out)
                best.append((differences, mask, label))
                if differences == 0:
                    winners.append((mask, label))
        print(
            backend,
            "degree",
            k,
            "seed",
            seed,
            "width",
            width,
            "native mismatches",
            native,
            "reduction fits",
            winners[:12],
            "fit count",
            len(winners),
            "best",
            sorted(best)[:4],
            flush=True,
        )
        for mode, count in native:
            if count:
                out = interp._k.bspline_eval(t, c, k, q, mode)
                ix = np.flatnonzero(ref.view(np.uint64).ravel() != out.view(np.uint64).ravel())[:2]
                print(
                    "native examples",
                    mode,
                    [(int(i), float(ref.ravel()[i]).hex(), float(out.ravel()[i]).hex()) for i in ix],
                    flush=True,
                )


def main():
    started = time.monotonic()
    print(
        sys.version,
        platform.platform(),
        platform.machine(),
        "numpy",
        np.__version__,
        "scipy",
        scipy.__version__,
        flush=True,
    )
    if not hasattr(math, "fma"):
        raise RuntimeError("This arithmetic diagnostic requires Python 3.13 math.fma")
    fastscipy.patch(threads=2)
    for backend in ("bspline", "splev"):
        for k in (5, 4, 3):
            print("selected", backend, k, interp._spline_arithmetic(backend, k), flush=True)
            for seed in (720 + k, 820 + k):
                probe(backend, k, seed)
    print("seconds", time.monotonic() - started, flush=True)


if __name__ == "__main__":
    main()
