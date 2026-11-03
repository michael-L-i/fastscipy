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
import scipy
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


# SciPy <=1.12's Fortran CDFLIB inverse solvers use shared SAVE/ENTRY state
# (dinvr.f, dzror.f). Keep every CDFLIB-backed ufunc original on these versions.
_CDFLIB = frozenset(
    "bdtrik bdtrin btdtria btdtrib chdtriv chndtr chndtridf chndtrinc chndtrix "
    "fdtridfd gdtria gdtrib gdtrix nbdtrik nbdtrin ncfdtr ncfdtri ncfdtridfd "
    "ncfdtridfn ncfdtrinc nctdtr nctdtridf nctdtrinc nctdtrit nrdtrimn nrdtrisd "
    "pdtrik stdtr stdtridf stdtrit tklmbda".split()
)

# Before SciPy 1.14, Cephes jv's jnx workspace (u, ai/aip/bi/bip) is
# mutable static storage. Real AMOS jv/yv wrappers fall back to it after a
# NaN result; the Struve helpers also call those wrappers. Keep these ufuncs
# original until the Cephes C++ rewrite makes that workspace local.
_CEPHES_JV = frozenset({"jv", "yv", "struve", "_struve_bessel_series", "_struve_asymp_large_z"})


def _thread_safe(uf, version):
    return not (
        version.startswith(("1.10.", "1.11.", "1.12."))
        and uf.__name__ in _CDFLIB
        or version.startswith(("1.10.", "1.11.", "1.12.", "1.13."))
        and uf.__name__ in _CEPHES_JV
    )


def _part(a, ax, nd, s):
    if type(a) is not np.ndarray:
        return a
    k = ax - (nd - a.ndim)
    return a if k < 0 or a.shape[k] == 1 else a[(slice(None),) * k + (s,)]


def _threaded(uf, st, args):
    if get_num_threads() <= 1 or any(v != "ignore" for k, v in sc.geterr().items() if k != "memory"):
        return NotImplemented
    order = "C"
    for a in args:
        if type(a) is np.ndarray:
            if a.dtype.char not in _OK:
                return NotImplemented
            if not a.flags.c_contiguous:
                if len(args) != 1 or not a.flags.f_contiguous:
                    return NotImplemented
                order = "F"  # A unary ufunc preserves the input's Fortran layout.
        elif not isinstance(a, _SCALARS):
            return NotImplemented
    shape = np.broadcast(*args).shape
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
    outs = tuple(np.empty(shape, f.dtype, order=order) for f in first)
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
        if hasattr(uf, attr):
            setattr(fast, attr, getattr(uf, attr))
    return fast


def patch():
    _patch.require_validated(*(f"1.{minor}." for minor in range(10, 19)))
    wrappers = {}
    for mod in (_ufuncs, sc):
        for uf in list(vars(mod).values()):
            if isinstance(uf, np.ufunc) and uf not in wrappers and _thread_safe(uf, scipy.__version__):
                wrappers[uf] = _make(uf, mod.__name__)
    for uf, fast in wrappers.items():
        _patch.rebind(uf, fast)


def smoke():
    x = np.linspace(0.1, 5, 65536)
    sc.erf(x), sc.gammaln(x), sc.digamma(x)
