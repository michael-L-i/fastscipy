import numpy as np
import scipy.special as sc

THREADED_ONLY = True  # the fast path only threads; it declines on 1 thread
rng = np.random.default_rng(0)


def cases():
    fortran = np.asfortranarray(np.random.default_rng(840).uniform(0.1, 2, (512, 1024)))
    for name in ("erf", "gammaln", "airy"):
        yield f"{name} Fortran 512x1024 float64", lambda n=name: getattr(sc, n)(fortran)
    for n in (10**5, 10**6, 10**7):
        for dt in (np.float64, np.float32):
            x = rng.uniform(0.1, 5, n).astype(dt)
            tag = f"{n:.0e} {np.dtype(dt).name}"
            for name in ("erf", "gammaln", "i0", "expit"):
                if n == 10**7 and name in ("i0", "expit"):
                    continue
                yield f"{name} {tag}", lambda f=getattr(sc, name), x=x: f(x)
            yield f"jv {tag}", lambda x=x: sc.jv(1.5, x)
            yield f"betainc {tag}", lambda x=x: sc.betainc(2.0, 3.5, x / 6)
    x = rng.uniform(0.1, 5, 1000)  # below the threshold: must not be slower than SciPy
    yield "erf 1e3 (control)", lambda: sc.erf(x), False
    yield "jv 1e3 (control)", lambda: sc.jv(1.5, x), False
