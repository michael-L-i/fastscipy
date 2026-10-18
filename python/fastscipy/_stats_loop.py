"""Lean per-function fast paths for the most used scipy.stats functions.

Each applies only to exact float64, 1-D, NaN-free ndarrays with default options and
reproduces SciPy's arithmetic op for op (array ``**2`` squares, a numpy scalar's calls libm
``pow``). Everything runs under ``np.errstate(raise)``: any floating-point event, precision
loss, constant input or too-small sample declines, so SciPy decides every warning.
"""
import functools
import inspect
import math

import numpy as np
import scipy.special as special
import scipy.stats as stats
import scipy.stats._mannwhitneyu as _mw
import scipy.stats._stats_py as sp

from . import _patch

ND = np.ndarray
F64 = np.dtype('float64')
f64 = np.float64
EPSF = np.finfo(np.float64).eps              # np.float64, like xp.finfo(...).eps
EPS10 = f64(EPSF) * 10
ORD2, ORD3, ORD4 = (np.asarray(float(k)) for k in (2, 3, 4))
NAN = float('nan')
_ADD = np.add.reduce
_stdtr, _betainc, _betaincc, _entr, _ndtr = special.stdtr, special.betainc, special.betaincc, special.entr, special.ndtr
_ESTATE = dict(divide='raise', over='raise', invalid='raise')
_ALT = {'less': -1, 'two-sided': 0, 'greater': 1}
_ALTF = {k: f64(v) for k, v in _ALT.items()}   # what pack_TtestResult stores


class _Decline(Exception):
    pass


def _clean(a):
    """1-D float64 ndarray, no NaN -> True"""
    if type(a) is ND and a.dtype is F64 and a.ndim == 1 and a.shape[0] > 0:
        m = a.max()
        return m == m
    return False


def _demean_ok(d, mean):
    """True if SciPy's _demean would NOT warn (precision loss); d = a - mean, n > 1."""
    mx = max(d.max(), -d.min())
    if mean == 0:
        return True   # rel_diff is inf or nan -> never < eps
    return not (mx / abs(mean) < EPS10)


def _moment(a, mean, ORD):
    """_moment(a, order, axis=-1, center=mean) for 1-D with the precision-loss check; returns 0-d array like SciPy,
    raises _Decline when SciPy would warn"""
    d = a - mean
    if not _demean_ok(d, mean):
        raise _Decline
    res = np.mean(d ** ORD, axis=-1, keepdims=True)
    return res.reshape(())


# ---------------------------------------------------------------------------------------------- skew / kurtosis
def _skew(a, axis=0, bias=True, nan_policy='propagate'):
    if axis not in (0, -1) or nan_policy != 'propagate' or type(axis) is not int or not _clean(a):
        return NotImplemented
    if bias is not True and bias is not False:
        return NotImplemented
    n = a.shape[0]
    with np.errstate(**_ESTATE):
        mean = _ADD(a) / n
        mean_k = np.mean(a, keepdims=True)
        d = a - mean_k
        if not _demean_ok(d, mean):
            return NotImplemented
        m2 = np.mean(d ** ORD2, axis=-1, keepdims=True).reshape(())
        m3 = np.mean(d ** ORD3, axis=-1, keepdims=True).reshape(())
        zero = m2 <= (EPSF * mean) ** 2
        if zero:
            vals = f64(NAN)
        else:
            vals = m3 / np.power(m2, 1.5)
            if not bias and n > 2:
                vals = ((n - 1.0) * n) ** 0.5 / (n - 2.0) * m3 / np.power(m2, 1.5)
    return f64(vals)


def _kurtosis(a, axis=0, fisher=True, bias=True, nan_policy='propagate'):
    if axis not in (0, -1) or type(axis) is not int or nan_policy != 'propagate' or not _clean(a):
        return NotImplemented
    if (bias is not True and bias is not False) or (fisher is not True and fisher is not False):
        return NotImplemented
    n = a.shape[0]
    with np.errstate(**_ESTATE):
        mean = _ADD(a) / n
        mean_k = np.mean(a, keepdims=True)
        d = a - mean_k
        if not _demean_ok(d, mean):
            return NotImplemented
        m2 = np.mean(d ** ORD2, axis=-1, keepdims=True).reshape(())
        m4 = np.mean(d ** ORD4, axis=-1, keepdims=True).reshape(())
        zero = m2 <= (EPSF * mean) ** 2
        if zero:
            vals = f64(NAN)
        else:
            vals = m4 / np.square(m2)
            if not bias and n > 3:
                nval = 1.0 / (n - 2) / (n - 3) * ((n ** 2 - 1.0) * m4 / np.square(m2) - 3 * (n - 1) ** 2.0)
                vals = nval + 3.0
        if fisher:
            vals = vals - 3
    return f64(vals)


# ---------------------------------------------------------------------------------------------------------- sem
def _sem(a, axis=0, ddof=1, nan_policy='propagate'):
    if axis not in (0, -1) or type(axis) is not int or nan_policy != 'propagate' or not _clean(a):
        return NotImplemented
    if type(ddof) is not int or ddof < 0:
        return NotImplemented
    n = a.shape[0]
    if n <= ddof or n <= 1:        # too_small=1 in the decorator, correction >= n -> let SciPy warn
        return NotImplemented
    with np.errstate(**_ESTATE):
        # xp.std(a, axis=axis, correction=ddof) -> np.std(a, axis=axis, ddof=ddof)
        s = np.std(a, axis=axis, ddof=ddof) / n ** 0.5
    return s


# ---------------------------------------------------------------------------------------------------- zscore
def _zscore(a, axis=0, ddof=0, nan_policy='propagate'):
    if axis not in (0, -1) or type(axis) is not int or nan_policy != 'propagate' or not _clean(a):
        return NotImplemented
    if type(ddof) is not int or ddof < 0:
        return NotImplemented
    n = a.shape[0]
    if n <= ddof or n < 2:
        return NotImplemented
    with np.errstate(**_ESTATE):
        mn = np.mean(a, axis=axis, keepdims=True)
        # _xp_var(compare, axis, correction=ddof, keepdims=True)
        mean2 = np.mean(a, axis=axis, keepdims=True)
        xm = a - mean2
        if not _demean_ok(xm, mean2[0]):
            return NotImplemented
        var = np.mean(xm * xm, axis=axis, keepdims=True)
        if ddof != 0:
            nn = np.asarray(n, dtype=var.dtype)
            nc = nn - ddof
            var *= nn / nc
        std = var ** 0.5
        z = (a - mn) / std
        if std[0] <= abs(EPSF * mn[0]):
            return NotImplemented          # constant input -> SciPy warns/NaNs
    return z


# --------------------------------------------------------------------------------------------------- t tests
def _pack_ttest(t, prob, df, alt, se, est):
    return sp.TtestResult(t, prob, df=df, alternative=_ALTF[alt], standard_error=se, estimate=est)


def _ttest_1samp(a, popmean, axis=0, nan_policy='propagate', alternative='two-sided'):
    if (axis != 0 and axis != -1) or type(axis) is not int or nan_policy != 'propagate':
        return NotImplemented
    tp = type(popmean)
    if not (tp is float or tp is int or tp is f64):
        return NotImplemented
    if alternative not in _ALT or not _clean(a) or popmean != popmean:
        return NotImplemented
    n = a.shape[0]
    if n < 2:
        return NotImplemented
    with np.errstate(**_ESTATE):
        mean = np.mean(a, axis=-1)
        d = mean - popmean
        # _var(a, axis, ddof=1)
        mean_k = np.mean(a, keepdims=True)
        dd = a - mean_k
        if not _demean_ok(dd, mean_k[0]):
            return NotImplemented
        v = np.mean(dd ** ORD2, axis=-1, keepdims=True).reshape(())
        nn = np.asarray(n, dtype=a.dtype)
        v *= (nn / (n - 1))
        denom = np.sqrt(v / n)
        t = np.divide(d, denom)
        if t.ndim == 0:
            t = t[()]
        df = n - 1
        dfa = np.asarray(df, dtype=t.dtype)
        if alternative == 'two-sided':
            prob = 2 * _stdtr(dfa, -np.abs(t))
        elif alternative == 'less':
            prob = _stdtr(dfa, t)
        else:
            prob = _stdtr(dfa, -t)
        if prob.ndim == 0:
            prob = prob[()]
        dfo = np.broadcast_to(np.asarray(df), t.shape)
        dfo = dfo[()] if dfo.ndim == 0 else dfo
    return _pack_ttest(t, prob, dfo, alternative, denom, mean)


def _ttest_ind(a, b, *, axis=0, equal_var=True, nan_policy='propagate', alternative='two-sided', trim=0,
               method=None):
    if (axis != 0 and axis != -1) or type(axis) is not int or nan_policy != 'propagate' or trim != 0 \
            or method is not None or (equal_var is not True and equal_var is not False):
        return NotImplemented
    if type(trim) is not int or alternative not in _ALT or not _clean(a) or not _clean(b):
        return NotImplemented
    n1 = a.shape[0]
    n2 = b.shape[0]
    if n1 < 2 or n2 < 2:
        return NotImplemented
    with np.errstate(**_ESTATE):
        # _var(a, axis, ddof=1)
        mk = np.mean(a, keepdims=True)
        dd = a - mk
        if not _demean_ok(dd, mk[0]):
            return NotImplemented
        v1 = np.mean(dd ** ORD2, axis=-1, keepdims=True).reshape(())
        v1 *= (np.asarray(n1, dtype=a.dtype) / (n1 - 1))
        mk = np.mean(b, keepdims=True)
        dd = b - mk
        if not _demean_ok(dd, mk[0]):
            return NotImplemented
        v2 = np.mean(dd ** ORD2, axis=-1, keepdims=True).reshape(())
        v2 *= (np.asarray(n2, dtype=b.dtype) / (n2 - 1))
        m1 = np.mean(a, axis=-1)
        m2 = np.mean(b, axis=-1)
        if equal_var:
            # n1 == 1 / n2 == 1 excluded above
            df = n1 + n2 - 2.0
            svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
            denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
            df = np.asarray(df, dtype=denom.dtype)
        else:
            vn1 = v1 / n1
            vn2 = v2 / n2
            df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (n1 - 1) + vn2 ** 2 / (n2 - 1))
            if df != df:
                return NotImplemented
            denom = np.sqrt(vn1 + vn2)
        d = m1 - m2
        t = np.divide(d, denom)
        dfa = np.asarray(df, dtype=t.dtype)
        if alternative == 'two-sided':
            prob = 2 * _stdtr(dfa, -np.abs(t))
        elif alternative == 'less':
            prob = _stdtr(dfa, t)
        else:
            prob = _stdtr(dfa, -t)
        prob = prob[()] if prob.ndim == 0 else prob
        t = t[()] if t.ndim == 0 else t
        dfo = np.broadcast_to(df, t.shape)
        dfo = dfo[()] if dfo.ndim == 0 else dfo
        est = m1 - m2
    return _pack_ttest(t, prob, dfo, alternative, denom, est)


def _ttest_rel(a, b, axis=0, nan_policy='propagate', alternative='two-sided'):
    if (axis != 0 and axis != -1) or type(axis) is not int or nan_policy != 'propagate':
        return NotImplemented
    if alternative not in _ALT or not _clean(a) or not _clean(b) or a.shape != b.shape:
        return NotImplemented
    with np.errstate(**_ESTATE):
        diff = a - b
    if not _clean(diff):
        return NotImplemented
    return _ttest_1samp(diff, 0.0, axis=0, alternative=alternative)


# ------------------------------------------------------------------------------------------------ pearsonr
def _pearsonr(x, y, *, alternative='two-sided', method=None, axis=0):
    if method is not None or (axis != 0 and axis != -1) or type(axis) is not int or alternative not in _ALT:
        return NotImplemented
    if not _clean(x) or not _clean(y):
        return NotImplemented
    n = x.shape[0]
    if n != y.shape[0] or n < 3:
        return NotImplemented
    if (x == x[0]).all() or (y == y[0]).all():
        return NotImplemented
    with np.errstate(**_ESTATE):
        xmean = np.mean(x, axis=-1, keepdims=True)
        ymean = np.mean(y, axis=-1, keepdims=True)
        xm = x - xmean
        ym = y - ymean
        xmax = np.max(np.abs(xm), axis=-1, keepdims=True)
        ymax = np.max(np.abs(ym), axis=-1, keepdims=True)
        normxm = xmax * np.linalg.norm(xm / xmax, ord=2, axis=-1, keepdims=True)
        normym = ymax * np.linalg.norm(ym / ymax, ord=2, axis=-1, keepdims=True)
        thr = EPSF ** 0.75
        if (normxm < thr * np.abs(xmean)).any() or (normym < thr * np.abs(ymean)).any():
            return NotImplemented
        r = np.vecdot(xm / normxm, ym / normym, axis=-1)
        r = f64(min(max(r, -1.0), 1.0)) if r == r else r
        ab = np.asarray(n / 2 - 1, dtype=x.dtype)
        # _SimpleBeta(ab, ab, loc=-1, scale=2)
        xx = (r - (-1)) / 2
        if alternative == 'less':
            pvalue = _betainc(ab, ab, xx)
        elif alternative == 'greater':
            pvalue = _betaincc(ab, ab, xx)
        else:
            pvalue = 2 * _betaincc(ab, ab, (np.abs(r) - (-1)) / 2)
    nn = np.asarray(n, dtype=x.dtype)
    r = r[()] if r.ndim == 0 else r
    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
    return sp.PearsonRResult(statistic=r, pvalue=pvalue, n=nn, alternative=alternative, x=x[...], y=y[...], axis=-1)


# --------------------------------------------------------------------------------------------------- linregress
def _linregress(x, y, alternative='two-sided', *, axis=0):
    if (axis != 0 and axis != -1) or type(axis) is not int or alternative not in _ALT:
        return NotImplemented
    if not _clean(x) or not _clean(y):
        return NotImplemented
    n = x.shape[0]
    if n != y.shape[0] or n < 3:
        return NotImplemented
    TINY = 1e-20
    with np.errstate(**_ESTATE):
        xmean = np.mean(x, axis=-1, keepdims=True)
        ymean = np.mean(y, axis=-1, keepdims=True)
        x_ = x - xmean
        if not _demean_ok(x_, xmean[0]):
            return NotImplemented
        y_ = y - ymean
        xmean = xmean.reshape(())
        ymean = ymean.reshape(())
        ssxm = np.vecdot(x_, x_, axis=-1) / n
        ssym = np.vecdot(y_, y_, axis=-1) / n
        ssxym = np.vecdot(x_, y_, axis=-1) / n
        if ssxm == 0.0 or ssym == 0.0:
            return NotImplemented
        r = np.clip(ssxym / np.sqrt(ssxm * ssym), -1.0, 1.0)
        slope = ssxym / ssxm
        intercept = ymean - slope * xmean
        df = n - 2
        t = r * np.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))
        dfa = np.asarray(df, dtype=t.dtype)
        if alternative == 'less':
            prob = _stdtr(dfa, t)
        elif alternative == 'greater':
            prob = _stdtr(dfa, -t)
        else:
            prob = 2 * _stdtr(dfa, -np.abs(t))
        prob = prob[()] if prob.ndim == 0 else prob
        slope_stderr = np.sqrt((1 - np.square(r)) * ssym / ssxm / df)   # SciPy's r is a 0-d ndarray: r**2 == square, not scalar pow
        intercept_stderr = slope_stderr * np.sqrt(ssxm + xmean ** 2)
        outs = [o[()] if o.ndim == 0 else o for o in (slope, intercept, r, prob, slope_stderr, intercept_stderr)]
    slope, intercept, r, prob, slope_stderr, intercept_stderr = outs
    return sp.LinregressResult(slope, intercept, r, prob, slope_stderr, intercept_stderr=intercept_stderr)


# ----------------------------------------------------------------------------------------------------- entropy
def _entropy(pk, qk=None, base=None, axis=0):
    if qk is not None or (axis != 0 and axis != -1) or type(axis) is not int:
        return NotImplemented
    if base is not None and (type(base) is not float and type(base) is not int):
        return NotImplemented
    if base is not None and base <= 0:
        return NotImplemented
    if not _clean(pk):
        return NotImplemented
    with np.errstate(**_ESTATE):
        s = np.sum(pk, axis=axis, keepdims=True)
        pk = pk / s
        vec = _entr(pk)
        S = np.sum(vec, axis=axis)
        if base is not None:
            S /= math.log(base)
    return S


# ------------------------------------------------------------------------------------------------------ describe
def _describe(a, axis=0, ddof=1, bias=True, nan_policy='propagate'):
    if axis not in (0, -1) or type(axis) is not int or nan_policy != 'propagate' or not _clean(a):
        return NotImplemented
    if type(ddof) is not int or ddof < 0:
        return NotImplemented
    n = a.shape[0]
    if n <= ddof:
        return NotImplemented
    sk = _skew(a, 0, bias)
    ku = _kurtosis(a, 0, True, bias)
    if sk is NotImplemented or ku is NotImplemented:
        return NotImplemented
    with np.errstate(**_ESTATE):
        mm = (np.min(a, axis=axis), np.max(a, axis=axis))
        m = np.mean(a, axis=axis)
        mk = np.mean(a, keepdims=True)
        dd = a - mk
        if not _demean_ok(dd, mk[0]):
            return NotImplemented
        v = np.mean(dd ** ORD2, axis=-1, keepdims=True).reshape(())
        if ddof != 0:
            nn = np.asarray(n, dtype=a.dtype)
            v *= (nn / (nn - ddof))
        v = v[()] if v.ndim == 0 else v
    return sp.DescribeResult(np.int64(n), mm, m, v, sk, ku)



# -------------------------------------------------------------------------------------------- mannwhitneyu


def _mannwhitneyu(x, y, use_continuity=True, alternative='two-sided', axis=0, method='auto'):
    if (axis != 0 and axis != -1) or type(axis) is not int or (use_continuity is not True and use_continuity is not False):
        return NotImplemented
    if (method != 'auto' and method != 'asymptotic') or alternative not in _ALT:
        return NotImplemented
    if not _clean(x) or not _clean(y):
        return NotImplemented
    n1 = x.shape[0]
    n2 = y.shape[0]
    xy = np.concatenate((x, y), axis=-1)
    ranks, _, t = _mw._rankdata(xy, 'average', return_ties=True)
    if ranks.dtype is not F64 or t.dtype is not F64:
        return NotImplemented
    with np.errstate(**_ESTATE):
        R1 = np.sum(ranks[..., :n1], axis=-1)
        U1 = R1 - n1 * (n1 + 1) / 2
        if method == 'auto' and not (n1 > 8 and n2 > 8) and not np.any(t > 1):
            return NotImplemented            # exact method -> SciPy
        mu = n1 * n2 / 2
        n = n1 + n2
        tie_term = np.sum(t ** 3 - t, axis=-1)
        s = np.sqrt(n1 * n2 / 12 * (n + 1 - tie_term / (n * (n - 1))))
        numerator = U1 - mu
        if use_continuity:
            if alternative == 'greater':
                sign = 1
            elif alternative == 'less':
                sign = -1
            else:
                sign = np.sign(numerator)
            numerator -= 0.5 * sign
        z = numerator / s
        if alternative == 'less':
            p = _ndtr(z)
        elif alternative == 'greater':
            p = _ndtr(-z)
        else:
            p = 2 * _ndtr(-np.abs(z))
        if p != p:
            return NotImplemented
        p = f64(min(max(p, 0.0), 1.0))
    res = _mw.MannwhitneyuResult(U1, p)
    if method == 'asymptotic':
        res.zstatistic = z[()] if z.ndim == 0 else z
    return res


def _declining(lean):
    """Floating-point events and precision-loss cases decline instead of raising."""
    params = inspect.signature(lean).parameters
    names, npos = frozenset(params), sum(p.kind != p.KEYWORD_ONLY for p in params.values())

    @functools.wraps(lean)
    def fast(*args, **kwargs):
        # unknown keywords (keepdims, _no_deco, ...) and extra arguments are SciPy's to handle
        if len(args) > npos or (kwargs and not names.issuperset(kwargs)):
            return NotImplemented
        try:
            return lean(*args, **kwargs)
        except (FloatingPointError, _Decline):
            return NotImplemented
    return fast


_FUNCTIONS = (_skew, _kurtosis, _sem, _zscore, _ttest_1samp, _ttest_ind, _ttest_rel, _pearsonr,
              _linregress, _entropy, _describe, _mannwhitneyu)


def patch():
    for lean in _FUNCTIONS:
        old = getattr(stats, lean.__name__[1:])
        _patch.rebind(old, _patch.wrap(old, _declining(lean)))
