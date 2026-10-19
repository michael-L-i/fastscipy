import numpy as np
import scipy.linalg as sl


def cases():
    rng = np.random.default_rng(0)
    factor_rng = np.random.default_rng(354)
    for dtype in (np.float32, np.float64, np.complex64, np.complex128):
        factor_a = factor_rng.standard_normal((8, 8)) + 5 * np.eye(8)
        if np.dtype(dtype).kind == "c":
            factor_a = factor_a + 1j * factor_rng.standard_normal(factor_a.shape)
        factor_a = factor_a.astype(dtype)
        factor_b = factor_rng.standard_normal((8, 2)).astype(dtype)
        for solver, factor in (
            (sl.lu_solve, sl.lu_factor(factor_a)),
            (sl.cho_solve, sl.cho_factor(factor_a @ factor_a.conj().T)),
        ):
            for finite in (True, False):
                yield (
                    f"{solver.__name__} single 8x8 {np.dtype(dtype).name}, check_finite={finite}",
                    lambda f=solver, a=factor, b=factor_b, finite=finite: f(a, b, check_finite=finite),
                )
    single_rng = np.random.default_rng(322)
    for dtype, layout in ((np.float32, "C"), (np.float64, "F"), (np.complex64, "strided"), (np.complex128, "C")):
        single_a = single_rng.standard_normal((8, 8)) + 5 * np.eye(8)
        single_b = single_rng.standard_normal((8, 2))
        if np.dtype(dtype).kind == "c":
            single_a = single_a + 1j * single_rng.standard_normal(single_a.shape)
            single_b = single_b + 1j * single_rng.standard_normal(single_b.shape)
        single_a, single_b = single_a.astype(dtype), single_b.astype(dtype)
        if layout == "F":
            single_a, single_b = np.asfortranarray(single_a), np.asfortranarray(single_b)
        elif layout == "strided":
            single_big = np.zeros((16, 16), dtype=dtype)
            single_big[::2, ::2] = single_a
            single_a, single_b = single_big[::2, ::2], single_b[::-1]
        for finite in (True, False):
            tag = f"{np.dtype(dtype).name} {layout}, check_finite={finite}"
            yield (
                f"solve_triangular single 8x8 {tag}",
                lambda a=single_a, b=single_b, f=finite: sl.solve_triangular(a, b, check_finite=f),
            )
    small_rng = np.random.default_rng(45)
    for B in (4, 32):
        small_a = small_rng.standard_normal((B, 3, 3)) + 5 * np.eye(3)
        small_b = small_rng.standard_normal((B, 3, 2))
        small_lu = sl.lu_factor(small_a)
        yield f"lu_factor small stack {B}x3x3", lambda a=small_a: sl.lu_factor(a)
        yield f"lu_solve small stack {B}x3x3", lambda lu=small_lu, b=small_b: sl.lu_solve(lu, b)
        yield f"solve_triangular small stack {B}x3x3", lambda a=small_a, b=small_b: sl.solve_triangular(a, b)
    for n in (3, 8, 32):
        B = 10_000 if n < 32 else 5_000
        a = rng.standard_normal((B, n, n)) + 4 * np.eye(n)
        spd = a @ a.swapaxes(-1, -2) + n * np.eye(n)
        tri = np.triu(a)
        b = rng.standard_normal((B, n, 2))
        lu, c = sl.lu_factor(a), sl.cho_factor(spd)
        tag = f"{B}x{n}x{n}"
        yield f"lu_factor {tag}", lambda: sl.lu_factor(a)
        yield f"lu_solve {tag}", lambda: sl.lu_solve(lu, b)
        yield f"cho_solve {tag}", lambda: sl.cho_solve(c, b)
        yield f"solve_triangular {tag}", lambda: sl.solve_triangular(tri, b)
        yield f"eigh {tag}", lambda: sl.eigh(spd)
        yield f"eigvalsh {tag}", lambda: sl.eigvalsh(spd)
    tall = rng.standard_normal((2000, 500))
    yield "null_space 2000x500", lambda: sl.null_space(tall)
