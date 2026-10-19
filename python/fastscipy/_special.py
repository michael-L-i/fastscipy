"""Threaded scipy.special ufuncs.

Each ufunc is replaced by a function that splits large calls along the longest
axis of the broadcast output into chunks of about 250 us, runs them on the
shared pool into one preallocated output, and otherwise calls the ufunc. The
chunks run SciPy's own loops, so results are identical. A call that raises a
floating-point event NumPy would report is rerun serially by SciPy, so warnings
and errors are SciPy's own; so is any call while ``sc.errstate`` reports errors.
"""

import math
import time

import numpy as np
import scipy.special as sc
import scipy.special._ufuncs as _ufuncs

from . import _patch
from ._threads import chunks, get_num_threads, parallel_map

_OK = "?bBhHiIlLqQefdFD"  # no object, longdouble or datetime
_PROBE = 512  # numpy releases the GIL only for inner loops above ~500 elements
_CHUNK_NS = 250_000
_SCALARS = (float, int, complex, np.generic)
_ATTRS = (
    "nin",
    "nout",
    "nargs",
    "ntypes",
    "types",
    "identity",
    "signature",
    "reduce",
    "outer",
    "accumulate",
    "reduceat",
    "at",
    "resolve_dtypes",
)


def _part(a, ax, nd, s):
    if type(a) is not np.ndarray:
        return a
    k = ax - (nd - a.ndim)
    return a if k < 0 or a.shape[k] == 1 else a[(slice(None),) * k + (s,)]


def _threaded(uf, st, args):
    if get_num_threads() <= 1 or any(v != "ignore" for k, v in sc.geterr().items() if k != "memory"):
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
    n, nd = math.prod(shape), len(shape)
    if n < st[0]:
        return NotImplemented
    ax = shape.index(max(shape))
    rows, per = shape[ax], n // shape[ax]
    probe = -(-_PROBE // per)
    if rows < 2 * probe:
        return NotImplemented
    part = lambda s: [_part(a, ax, nd, s) for a in args]
    events = []
    watch = {k: "ignore" if v == "ignore" else "call" for k, v in np.geterr().items()}

    def run(s, out=None):
        with np.errstate(call=lambda *e: events.append(e), **watch):
            return uf(*part(s)) if out is None else uf(*part(s), out=out)

    t = time.perf_counter_ns()
    first = run(slice(0, probe))
    ns = (time.perf_counter_ns() - t) / (probe * per)
    st[0] = max(1024, int(200_000 / max(ns, 0.01)))
    if ns * n < 2 * _CHUNK_NS:
        return NotImplemented
    first = first if isinstance(first, tuple) else (first,)
    outs = tuple(np.empty(shape, f.dtype) for f in first)
    sel = lambda s: (slice(None),) * ax + (s,)
    for o, f in zip(outs, first):
        o[sel(slice(0, probe))] = f
    size = max(probe, -(-_CHUNK_NS // max(1, int(ns * per))))
    parallel_map(
        lambda s: run(s, tuple(o[sel(s)] for o in outs)),
        [slice(probe + c.start, probe + c.stop) for c in chunks(rows - probe, size)],
    )
    if events:
        return NotImplemented
    return outs if len(outs) > 1 else outs[0]


def _make(uf, module):
    st, nin, name = [1024], uf.nin, uf.__name__

    def fast(*args, **kw):
        if kw:
            return uf(*args, **kw)
        if _patch.enabled and len(args) == nin:
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
        return uf(*args)

    fast.__name__, fast.__qualname__, fast.__module__, fast.__doc__ = name, name, module, uf.__doc__
    fast.__wrapped__ = uf
    for attr in _ATTRS:
        setattr(fast, attr, getattr(uf, attr))
    return fast


def patch():
    _patch.require_validated("1.18.")
    wrappers = {}
    for mod in (_ufuncs, sc):
        for uf in list(vars(mod).values()):
            if isinstance(uf, np.ufunc) and uf not in wrappers:
                wrappers[uf] = _make(uf, mod.__name__)
    for uf, fast in wrappers.items():
        _patch.rebind(uf, fast)


def smoke():
    x = np.linspace(0.1, 5, 4096)
    sc.erf(x), sc.gammaln(x), sc.jv(1.5, x)
