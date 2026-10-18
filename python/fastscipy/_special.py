"""Threaded scipy.special ufuncs.

Each ufunc is replaced by a function that splits large calls along the longest
axis of the broadcast output into chunks of about 250 us, runs them on the
shared pool into one preallocated output, and otherwise calls the ufunc. The
chunks run SciPy's own loops, so results are identical.
"""
import warnings

import numpy as np
import scipy.special as sc
import scipy.special._ufuncs as _ufuncs

from . import _patch
from ._threads import chunks, get_num_threads, parallel_map

_OK = "?bBhHiIlLqQefdFD"          # no object, longdouble or datetime
_PROBE = 512                      # numpy releases the GIL only for inner loops above ~500 elements
_CHUNK_NS = 250_000
_SCALARS = (float, int, complex, np.generic)
_ATTRS = ("nin", "nout", "nargs", "ntypes", "types", "identity", "signature", "reduce",
          "outer", "accumulate", "reduceat", "at", "resolve_dtypes")


def _part(a, ax, nd, s):
    if type(a) is not np.ndarray:
        return a
    k = ax - (nd - a.ndim)
    return a if k < 0 or a.shape[k] == 1 else a[(slice(None),) * k + (s,)]


def _threaded(uf, st, args):
    import time
    if get_num_threads() <= 1:
        return NotImplemented
    shapes = []
    for a in args:
        if type(a) is np.ndarray:
            if a.dtype.char not in _OK or not a.flags.c_contiguous:
                return NotImplemented
            shapes.append(a.shape)
        elif not isinstance(a, _SCALARS):
            return NotImplemented
    shape = np.broadcast_shapes(*shapes)
    n, nd = int(np.prod(shape)), len(shape)
    if n < st[0]:
        return NotImplemented
    ax = shape.index(max(shape))
    rows, per = shape[ax], n // shape[ax]
    probe = -(-_PROBE // per)
    if rows < 2 * probe:
        return NotImplemented
    part = lambda s: [_part(a, ax, nd, s) for a in args]
    t = time.perf_counter_ns()
    first = uf(*part(slice(0, probe)))
    ns = (time.perf_counter_ns() - t) / (probe * per)
    st[0] = max(1024, int(200_000 / max(ns, 0.01)))
    if ns * n < 2 * _CHUNK_NS:
        return NotImplemented
    first = first if isinstance(first, tuple) else (first,)
    outs = tuple(np.empty(shape, f.dtype) for f in first)
    sel = lambda s: (slice(None),) * ax + (s,)
    for o, f in zip(outs, first):
        o[sel(slice(0, probe))] = f
    np_err, sc_err = np.geterr(), sc.geterr()

    def task(s):
        sc.seterr(**sc_err)
        np.seterr(**np_err)
        uf(*part(s), out=tuple(o[sel(s)] for o in outs))

    size = max(probe, -(-_CHUNK_NS // max(1, int(ns * per))))
    parallel_map(task, [slice(probe + c.start, probe + c.stop)
                        for c in chunks(rows - probe, size)])
    return outs if len(outs) > 1 else outs[0]


def _make(uf, module):
    st, nin, name = [1024], uf.nin, uf.__name__

    def fast(*args, **kw):
        if not kw and _patch.enabled and len(args) == nin:
            for a in args:
                if type(a) is np.ndarray and a.size >= st[0]:
                    try:
                        r = _threaded(uf, st, args)
                    except Exception:
                        if _patch.STRICT:
                            raise
                        _patch.counts[name, "error"] += 1
                    else:
                        if r is not NotImplemented:
                            _patch.counts[name, "fast"] += 1
                            return r
                    break
        return uf(*args, **kw)

    fast.__name__, fast.__qualname__, fast.__module__, fast.__doc__ = name, name, module, uf.__doc__
    fast.__wrapped__ = uf
    for attr in _ATTRS:
        setattr(fast, attr, getattr(uf, attr))
    return fast


def patch():
    _patch.require_scipy("1.18.")
    wrappers = {}
    for mod in (_ufuncs, sc):
        for name, uf in list(vars(mod).items()):
            if isinstance(uf, np.ufunc) and uf not in wrappers:
                wrappers[uf] = _make(uf, mod.__name__)
    for uf, fast in wrappers.items():
        _patch.rebind(uf, fast)


def smoke():
    x = np.linspace(0.1, 5, 4096)
    sc.erf(x), sc.gammaln(x), sc.jv(1.5, x)
