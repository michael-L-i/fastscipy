"""Fast path for built-in ``rv_continuous`` / ``rv_discrete`` methods and frozen-distribution creation.

For scalar, valid parameters and values strictly inside the support, SciPy's generic method
reduces to ``(x - loc) / scale`` and a call of the private ``_cdf`` (etc.) with broadcast
parameters, so that is all done here, with the parameter checks cached per (distribution,
parameters). NaN, inf, out-of-support values, array parameters, odd dtypes and user
subclasses decline to SciPy.
"""
import math
import types
import warnings
import weakref

import numpy as np
from scipy.stats import _distn_infrastructure as di

from . import _patch

_LIM = 2.0 ** 53
_INF = math.inf
_BIG = 1e100
_f8 = np.dtype("float64")
_i8 = np.dtype("int64")
try:
    _clip = np._core.umath.clip
except AttributeError:  # pragma: no cover
    _clip = np.clip

_OK_MODULES = {"scipy.stats._continuous_distns", "scipy.stats._discrete_distns"}
_full = np.full
_asarray = np.asarray
_minred = np.minimum.reduce
_maxred = np.maximum.reduce
_npfloat = np.float64
_ndarray = np.ndarray


# --------------------------------------------------------------------------
# per-class eligibility
# --------------------------------------------------------------------------
class _Cls:
    __slots__ = ("ok", "get_support_default", "support_mask_default", "discrete")


_CLS = {}


def _cls_info(cls):
    info = _CLS.get(cls)
    if info is None:
        info = _Cls()
        info.ok = (cls.__module__ in _OK_MODULES
                   and issubclass(cls, (di.rv_continuous, di.rv_discrete))
                   and not issubclass(cls, di.rv_sample))
        info.discrete = issubclass(cls, di.rv_discrete)
        info.get_support_default = cls._get_support in (
            di.rv_generic._get_support,
            getattr(di.rv_continuous, "_get_support", None),
            getattr(di.rv_discrete, "_get_support", None))
        info.support_mask_default = (cls._support_mask is di.rv_generic._support_mask
                                     and cls._open_support_mask
                                     is di.rv_generic._open_support_mask)
        # the discrete default _nonzero is floor(k) == k; if overridden, defer.
        if info.discrete and cls._nonzero is not di.rv_discrete._nonzero:
            info.ok = False
        _CLS[cls] = info
    return info


# --------------------------------------------------------------------------
# per (distribution instance, shape params) cache
# --------------------------------------------------------------------------
_CACHE = weakref.WeakKeyDictionary()


def _norm_param(p):
    t = type(p)
    if t is float:
        return p if p != 0 else None
    if t is int:
        return p if (p != 0 and -_LIM < p < _LIM) else None
    if t is _npfloat:
        return float(p) if p != 0 else None
    return None


def _entry(self, info, pa):
    """Return cached (args0d, a, b) for valid scalar params, else None."""
    key = None
    if len(pa) == 1:
        p = _norm_param(pa[0])
        if p is None:
            return None
        key = (p, type(p))
        pa2 = (p,)
    else:
        pa2 = tuple(map(_norm_param, pa))
        if None in pa2:
            return None
        key = pa2 + tuple(map(type, pa2))
    d = _CACHE.get(self)
    if d is None:
        d = _CACHE[self] = {}
    ent = d.get(key, NotImplemented)
    if ent is NotImplemented:
        try:
            args0d = tuple(_asarray(p) for p in pa2)
            # Parameter validation may warn (e.g. erlang non-integer shape);
            # such parameter sets are never cached so SciPy keeps warning.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ok = bool(self._argcheck(*args0d))
                if ok and not info.get_support_default:
                    a, b = self._get_support(*args0d)
                else:
                    a = b = None
            ent = (args0d, a, b) if (ok and not w) else None
        except Exception:
            ent = None
        if len(d) > 256:
            d.clear()
        d[key] = ent
    return ent


def _loc_scale_ok(v):
    t = type(v)
    if t is float:
        return v
    if t is int:
        return float(v) if -_LIM < v < _LIM else None
    if t is _npfloat:
        return float(v)
    return None


# --------------------------------------------------------------------------
# finishing
# --------------------------------------------------------------------------
def _finish(vals, shape, n):
    """Equivalent of: out = zeros(shape); place(out, all-true, vals); out[()]"""
    if type(vals) is _ndarray and vals.dtype is _f8 and vals.shape == (n,):
        if shape == ():
            return vals[0]
        return vals.reshape(shape)
    out = np.zeros(shape, _f8)
    np.place(out, np.ones(shape, bool), vals)
    if out.ndim == 0:
        return out[()]
    return out


# --------------------------------------------------------------------------
# continuous
# --------------------------------------------------------------------------
# kind: (private method, closed support?, post)
# post: 0 none, 1 /scale (pdf), 2 - log(scale) (logpdf)
def _cont_x(self, x, args, kwds, priv, closed, post):
    cls = type(self)
    info = _CLS.get(cls) or _cls_info(cls)
    if not info.ok or info.discrete:
        return NotImplemented
    try:
        pa, loc, scale = self._parse_args(*args, **kwds)
    except Exception:
        return NotImplemented
    loc = _loc_scale_ok(loc)
    scale = _loc_scale_ok(scale)
    if loc is None or scale is None or not (scale > 0):
        return NotImplemented
    ent = _entry(self, info, pa)
    if ent is None:
        return NotImplemented
    args0d, a, b = ent
    if a is None:
        a = self.a
        b = self.b
    tx = type(x)
    trivial = scale == 1.0 and loc == 0.0 and math.copysign(1.0, loc) > 0
    if tx is float or tx is int or tx is _npfloat:
        if tx is int and not (-_LIM < x < _LIM):
            return NotImplemented
        xs = float(x)
        if not trivial:
            xs = (xs - loc) / scale
        if not (-_INF < xs < _INF):
            return NotImplemented
        if not info.support_mask_default and closed:
            m = self._support_mask(_asarray(xs), *args0d)
            if not m.all():
                return NotImplemented
        elif closed:
            if not (a <= xs <= b):
                return NotImplemented
        else:
            if not (a < xs < b):
                return NotImplemented
        xf = np.array([xs])
        shape = ()
        n = 1
    elif tx is _ndarray and x.dtype is _f8 and x.size:
        # guard: no overflow / invalid in the arithmetic below, hence no
        # RuntimeWarning that SciPy would also emit (and we'd duplicate)
        xr = x.ravel()
        if not (-_BIG < _minred(xr) and _maxred(xr) < _BIG):
            return NotImplemented
        if trivial:
            xs = x.copy()
        else:
            if not (-_BIG < loc < _BIG and 1.0 / _BIG < scale < _BIG):
                return NotImplemented
            xs = (x - loc) / scale
        xf = xs.ravel()
        if not info.support_mask_default and closed:
            m = self._support_mask(xs, *args0d)
            if not m.all():
                return NotImplemented
        else:
            lo = _minred(xf)
            hi = _maxred(xf)
            if closed:
                if not (a <= lo and hi <= b):
                    return NotImplemented
            else:
                if not (a < lo and hi < b):
                    return NotImplemented
        shape = x.shape
        n = xf.size
    else:
        return NotImplemented
    pargs = [_full(n, p) for p in args0d]
    vals = getattr(self, priv)(xf, *pargs)
    if post:
        sc = _full(n, scale)
        if post == 1:
            vals = vals / sc
        else:
            vals = vals - np.log(sc)
    return _finish(vals, shape, n)


def _cont_q(self, q, args, kwds, priv):
    cls = type(self)
    info = _CLS.get(cls) or _cls_info(cls)
    if not info.ok or info.discrete:
        return NotImplemented
    try:
        pa, loc, scale = self._parse_args(*args, **kwds)
    except Exception:
        return NotImplemented
    loc = _loc_scale_ok(loc)
    scale = _loc_scale_ok(scale)
    if loc is None or scale is None or not (-_BIG < loc < _BIG) \
            or not (1.0 / _BIG < scale < _BIG):
        return NotImplemented
    ent = _entry(self, info, pa)
    if ent is None:
        return NotImplemented
    args0d = ent[0]
    tq = type(q)
    if tq is float or tq is _npfloat:
        if not (0.0 < q < 1.0):
            return NotImplemented
        qf = np.array([float(q)])
        shape = ()
        n = 1
    elif tq is _ndarray and q.dtype is _f8 and q.size:
        qf = q.ravel()
        if not (0.0 < _minred(qf) and _maxred(qf) < 1.0):
            return NotImplemented
        shape = q.shape
        n = qf.size
    else:
        return NotImplemented
    pargs = [_full(n, p) for p in args0d]
    try:
        vals = getattr(self, priv)(qf, *pargs) * _full(n, scale) + _full(n, loc)
    except OverflowError:       # raised by the kernel; SciPy's own call raises it with its context
        return NotImplemented
    if n == 1 and shape == () and type(vals) is _ndarray and vals.dtype is _f8 \
            and vals.shape == (1,):
        return vals[0]
    return _finish(vals, shape, n)


# --------------------------------------------------------------------------
# discrete
# --------------------------------------------------------------------------
# kind: 'pmf' (closed [a,b], clip), 'logpmf' (closed, no clip),
#       'cdf'/'sf' ([a,b), finite, clip), 'logcdf'/'logsf' ([a,b), no clip)
def _disc_k(self, k, args, kwds, priv, kind):
    cls = type(self)
    info = _CLS.get(cls) or _cls_info(cls)
    if not info.ok or not info.discrete:
        return NotImplemented
    try:
        pa, loc, _ = self._parse_args(*args, **kwds)
    except Exception:
        return NotImplemented
    tl = type(loc)
    if tl is int:
        if not (-_LIM < loc < _LIM):
            return NotImplemented
    elif tl is float or tl is _npfloat:
        loc = float(loc)
        if not (-_BIG < loc < _BIG):
            return NotImplemented
    else:
        return NotImplemented
    ent = _entry(self, info, pa)
    if ent is None:
        return NotImplemented
    args0d, a, b = ent
    if a is None:
        a = self.a
        b = self.b
    tk = type(k)
    if tk is int or tk is float or tk is _npfloat:
        if tk is int:
            if not (-_LIM < k < _LIM):
                return NotImplemented
        else:
            k = float(k)
        ks = k - loc
        if kind == "pmf" or kind == "logpmf":
            if not (a <= ks <= b):
                return NotImplemented
        else:
            if not (a <= ks < b):
                return NotImplemented
            if kind == "cdf" or kind == "sf":
                if ks == -math.inf:
                    return NotImplemented
        if type(ks) is float and -math.inf < ks < math.inf \
                and ks != math.floor(ks):
            return NotImplemented
        kf = np.array([ks])
        shape = ()
        n = 1
    elif tk is _ndarray and (k.dtype is _f8 or k.dtype is _i8) and k.size:
        if loc == 0 and (tl is int or math.copysign(1.0, loc) > 0):
            ksa = k.copy()
        else:
            ksa = k - loc
        kf = ksa.ravel()
        lo = _minred(kf)
        hi = _maxred(kf)
        if kind == "pmf" or kind == "logpmf":
            if not (a <= lo and hi <= b):
                return NotImplemented
        else:
            if not (a <= lo and hi < b):
                return NotImplemented
            if (kind == "cdf" or kind == "sf") and lo == -math.inf:
                return NotImplemented
        if kf.dtype is _f8:
            if not (np.floor(kf) == kf).all():
                return NotImplemented
        shape = k.shape
        n = kf.size
    else:
        return NotImplemented
    pargs = [_full(n, p) for p in args0d]
    vals = getattr(self, priv)(kf, *pargs)
    if kind == "pmf" or kind == "cdf" or kind == "sf":
        vals = _clip(vals, 0, 1)
    return _finish(vals, shape, n)


def _disc_q(self, q, args, kwds, priv):
    cls = type(self)
    info = _CLS.get(cls) or _cls_info(cls)
    if not info.ok or not info.discrete:
        return NotImplemented
    try:
        pa, loc, _ = self._parse_args(*args, **kwds)
    except Exception:
        return NotImplemented
    tl = type(loc)
    if tl is int:
        if not (-_LIM < loc < _LIM):
            return NotImplemented
    elif tl is float or tl is _npfloat:
        loc = float(loc)
        if not (-_BIG < loc < _BIG):
            return NotImplemented
    else:
        return NotImplemented
    ent = _entry(self, info, pa)
    if ent is None:
        return NotImplemented
    args0d = ent[0]
    tq = type(q)
    if tq is float or tq is _npfloat:
        if not (0.0 < q < 1.0):
            return NotImplemented
        qf = np.array([float(q)])
        shape = ()
        n = 1
    elif tq is _ndarray and q.dtype is _f8 and q.size:
        qf = q.ravel()
        if not (0.0 < _minred(qf) and _maxred(qf) < 1.0):
            return NotImplemented
        shape = q.shape
        n = qf.size
    else:
        return NotImplemented
    pargs = [_full(n, p) for p in args0d]
    vals = getattr(self, priv)(qf, *pargs) + _full(n, loc)
    return _finish(vals, shape, n)



# --------------------------------------------------------------------------
# cheap frozen distributions
# --------------------------------------------------------------------------
_TEMPLATE_CODE = {}
_KEYS = {}  # class -> frozenset of constructor-created attributes, or False (no clone)


def _attach_argparser_methods(self):
    tpl = self._parse_arg_template
    code = _TEMPLATE_CODE.get(tpl)
    if code is None:
        code = _TEMPLATE_CODE[tpl] = compile(tpl, "<string>", "exec")
    ns = {}
    exec(code, ns)
    for name in ("_parse_args", "_parse_args_stats", "_parse_args_rvs"):
        setattr(self, name, types.MethodType(ns[name], self))


def _clone(dist):
    """A new instance built the way unpickling does, not by re-running the constructor."""
    cls = type(dist)
    if not (_CLS.get(cls) or _cls_info(cls)).ok:
        return NotImplemented
    keys = _KEYS.get(cls)
    if keys is None:
        # First freeze: build it SciPy's way and learn which attributes the constructor creates
        # (module code may attach extras, such as ``_support``, to the shared instance later).
        fresh = cls(**dist._updated_ctor_param())
        odd = any(isinstance(v, types.MethodType) and v.__self__ is not fresh for v in fresh.__dict__.values())
        _KEYS[cls] = False if odd else frozenset(fresh.__dict__)   # poisson_binom binds another instance
        return fresh
    if keys is False:
        return NotImplemented
    new = object.__new__(cls)
    st = {k: v for k, v in dist.__getstate__().items() if k in keys}
    st["_ctor_param"] = dict(dist._updated_ctor_param())
    new.__setstate__(st)
    return new


def _frozen_init(self, dist, *args, **kwds):
    new = _clone(dist)
    if new is NotImplemented:
        return NotImplemented
    self.args, self.kwds, self.dist = args, kwds, new
    shapes, _, _ = new._parse_args(*args, **kwds)
    self.a, self.b = new._get_support(*shapes)


# --------------------------------------------------------------------------
# installation
# --------------------------------------------------------------------------
_CONT_X = {  # method: (private method, closed support?, post)
    "pdf": ("_pdf", True, 1), "logpdf": ("_logpdf", True, 2), "cdf": ("_cdf", False, 0),
    "logcdf": ("_logcdf", False, 0), "sf": ("_sf", False, 0), "logsf": ("_logsf", False, 0),
}
_DISC_K = ("pmf", "logpmf", "cdf", "logcdf", "sf", "logsf")


def _cont_x_method(priv, closed, post):
    return lambda self, x, *args, **kwds: _cont_x(self, x, args, kwds, priv, closed, post)


def _quantile_method(fast, priv):
    return lambda self, q, *args, **kwds: fast(self, q, args, kwds, priv)


def _disc_k_method(name):
    return lambda self, k, *args, **kwds: _disc_k(self, k, args, kwds, "_" + name, name)


def patch():
    def install(cls, name, fast):
        _patch.replace(cls, name, _patch.wrap(cls.__dict__[name], fast, f"{cls.__name__}.{name}"))

    for name, spec in _CONT_X.items():
        install(di.rv_continuous, name, _cont_x_method(*spec))
    for name in _DISC_K:
        install(di.rv_discrete, name, _disc_k_method(name))
    for name in ("ppf", "isf"):
        install(di.rv_continuous, name, _quantile_method(_cont_q, "_" + name))
        install(di.rv_discrete, name, _quantile_method(_disc_q, "_" + name))
    _patch.replace(di.rv_generic, "_attach_argparser_methods", _attach_argparser_methods)
    install(di.rv_frozen, "__init__", _frozen_init)
