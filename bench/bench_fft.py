"""Small image/tile transforms plus larger and declined controls (under 10 MiB)."""

import numpy as np
from scipy import fft, fftpack


def cases():
    rng = np.random.default_rng(618)
    for namespace in (fft, fftpack):
        label = namespace.__name__
        for shape in ((8, 8), (32, 32), (256, 256)):
            x = rng.standard_normal(shape)
            for name in ("fftn", "dctn", "idctn", "dstn"):
                yield f"{label}.{name} {shape}", lambda x=x, ns=namespace, name=name: getattr(ns, name)(x)
        x = rng.standard_normal((8, 8))
        yield f"{label}.fftn list control", lambda x=x.tolist(), ns=namespace: ns.fftn(x), False
        yield f"{label}.fftn resize control", lambda x=x, ns=namespace: ns.fftn(x, (16, 16)), False
        yield f"{label}.fft 1D unchanged control", lambda x=x[0], ns=namespace: ns.fft(x), False
    x = rng.standard_normal((8, 8))
    x.flags.writeable = False
    yield "scipy.fft.fft2 default axes", lambda: fft.fft2(x)
    yield "scipy.fft.fftn ortho", lambda: fft.fftn(x, norm="ortho")
    yield "scipy.fft.fftn transposed readonly", lambda: fft.fftn(x.T)
    yield "scipy.fft.dctn integer control", lambda x=x.astype(np.int64): fft.dctn(x), False
    yield "scipy.fft.rfft 1D unchanged control", lambda: fft.rfft(x[0]), False
    tiles = rng.standard_normal((64, 8, 8))
    yield "scipy.fft.dctn 64 image tiles batched", lambda: fft.dctn(tiles, axes=(-2, -1))
