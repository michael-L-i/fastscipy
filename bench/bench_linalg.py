import numpy as np
import scipy.linalg as sl


def cases():
    rng = np.random.default_rng(0)
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
