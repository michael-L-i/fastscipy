import warnings

import numpy as np
import pytest
import scipy.stats as st

import fastscipy
from fastscipy.testing import check, fast_calls, identical

ALTS = ["two-sided", "less", "greater"]


def outcome(call):
    """(result | exception, warnings) of ``call()``, with SciPy's own warnings recorded."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = call()
        except Exception as e:
            r = (type(e), str(e))
    return r, [(x.category, str(x.message)) for x in w]


def same(call):
    """Assert patched == plain SciPy (result bytes, exceptions, warnings); return whether a fast path ran."""
    with fastscipy.disabled():
        expected = outcome(call)
    before = fast_calls()
    got = outcome(call)
    assert identical(expected[0], got[0]) and expected[1] == got[1], f"{expected!r}\n{got!r}"
    return fast_calls() > before


@pytest.fixture(params=[1, 4], ids=["1t", "4t"])
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield request.param
    fastscipy.set_num_threads(old)


# ----------------------------------------------------------------- per-function fast paths (randomized)
rng = np.random.default_rng(12345)


def sample():
    n = int(rng.choice([2, 3, 4, 5, 8, 10, 17, 30, 64, 100, 257, 1000, 4096]))
    k = rng.choice([0, 1, 2, 3, 9, 12, 13]) if rng.random() < 0.65 else rng.integers(0, 14)
    if k == 0:
        x = rng.standard_normal(n)
    elif k == 1:
        x = rng.gamma(2.0, 3.0, n)
    elif k == 2:
        x = rng.standard_normal(n) * 10.0 ** rng.integers(-8, 8) + rng.standard_normal() * 10.0 ** rng.integers(-3, 8)
    elif k == 3:
        x = rng.integers(0, 5, n).astype(float)  # ties
    elif k == 4:
        x = np.full(n, rng.standard_normal())  # constant
    elif k == 5:
        x = 1e8 + rng.standard_normal(n) * 1e-7  # near constant
    elif k == 6:
        x = rng.standard_normal(n)
        x[rng.integers(0, n)] = np.nan
    elif k == 7:
        x = rng.standard_normal(n)
        x[rng.integers(0, n)] = np.inf
    elif k == 8:
        x = rng.standard_normal(n) * 1e200
    elif k == 9:
        x = rng.standard_normal(n * 2)[::2]  # strided
    elif k == 10:
        x = rng.standard_normal(n).astype(np.float32)
    elif k == 11:
        x = rng.integers(-5, 5, n)
    elif k == 12:
        x = rng.standard_normal(n)
        x.flags.writeable = False
    else:
        x = np.abs(rng.standard_normal(n)) + 1e-3
    return x


def pair():
    x, y = sample(), sample()
    if rng.random() < 0.93 and y.shape != x.shape:
        y = np.resize(y, x.shape).astype(y.dtype)
    if rng.random() < 0.15 and x.dtype.kind == "f":
        y = x + rng.standard_normal(x.shape) * 0.1
    return x, y


def opts(**choices):
    return {k: v[rng.integers(len(v))] for k, v in choices.items() if rng.random() < 0.4}


CASES = {
    "skew": lambda: ((sample(),), opts(bias=[True, False], axis=[0, -1, None, 1])),
    "kurtosis": lambda: ((sample(),), opts(bias=[True, False], fisher=[True, False], axis=[0, -1, None])),
    "sem": lambda: ((sample(),), opts(ddof=[0, 1, 2, 5], axis=[0, -1, None])),
    "zscore": lambda: ((sample(),), opts(ddof=[0, 1, 2], axis=[0, -1, None])),
    "describe": lambda: ((sample(),), opts(ddof=[0, 1, 2], bias=[True, False])),
    "entropy": lambda: (
        (np.abs(sample()) + (rng.random() < 0.8) * 0.01,),
        opts(base=[2, 2.0, 10, 0.5, -1, 1], axis=[0, -1]),
    ),
    "ttest_1samp": lambda: (
        (sample(), float(rng.choice([0.0, 0.1, -1.0, 3.0, 1e3]))),
        opts(alternative=ALTS, axis=[0, -1, None]),
    ),
    "ttest_ind": lambda: (pair(), opts(equal_var=[True, False], alternative=ALTS, axis=[0, -1, None], trim=[0, 0.1])),
    "ttest_rel": lambda: (pair(), opts(alternative=ALTS, axis=[0, -1])),
    "pearsonr": lambda: (pair(), opts(alternative=ALTS, axis=[0, -1, None])),
    "linregress": lambda: (pair(), opts(alternative=ALTS, axis=[0, -1])),
    "mannwhitneyu": lambda: (
        pair(),
        opts(alternative=ALTS, use_continuity=[True, False], method=["auto", "asymptotic"], axis=[0, -1]),
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_function_randomized(name, threads):
    f = getattr(st, name)
    fast = 0
    for _ in range(150):
        args, kw = CASES[name]()
        before = [a.copy() for a in args if isinstance(a, np.ndarray)]
        fast += same(lambda: f(*args, **kw))
        assert all(a.tobytes() == b.tobytes() for a, b in zip((a for a in args if isinstance(a, np.ndarray)), before))
        res = outcome(lambda: f(*args, **kw))[0]
        if hasattr(res, "confidence_interval"):
            assert same(lambda: res.confidence_interval(0.9)) is False
    assert fast > 30, f"{name}: fast path ran {fast}/150 times"


def test_functions_decline_odd_input():
    a = rng.standard_normal(20)
    for f, args in [
        (st.skew, ([1.0, 2.0, 4.0],)),
        (st.sem, (a.astype(np.float32),)),
        (st.zscore, (np.ones(5),)),
        (st.ttest_ind, (a, a[:1])),
        (st.pearsonr, (a, a[:2])),
        (st.entropy, (a, a)),
        (st.kurtosis, (np.ma.masked_array(a),)),
        (st.describe, (a.reshape(4, 5),)),
    ]:
        assert same(lambda: f(*args)) is False


# ----------------------------------------------------------------------------------------- rankdata
RANK_KINDS = {
    "float": lambda n: rng.standard_normal(n),
    "ties": lambda n: rng.integers(0, 6, n).astype(float),
    "special": lambda n: rng.choice([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, 5e-324], n),
    "int64": lambda n: rng.integers(-(2**62), 2**62, n),
    "uint64": lambda n: rng.integers(0, 2**64 - 1, n, dtype=np.uint64),
    "int8": lambda n: rng.integers(-5, 5, n).astype(np.int8),
}


@pytest.mark.parametrize("kind", RANK_KINDS)
@pytest.mark.parametrize("n", [1, 2, 7, 1500, 3000, 200_000])
@pytest.mark.parametrize("method", ["average", "min", "max", "dense", "ordinal"])
def test_rankdata(kind, n, method, threads):
    x = RANK_KINDS[kind](n)
    assert check(lambda: st.rankdata(x, method=method)) is not None
    if n < 5000:
        x2 = RANK_KINDS[kind](4 * n).reshape(2, 2, n)
        check(lambda: st.rankdata(x2, axis=-1, method=method))
        check(lambda: st.rankdata(x2, axis=0, method=method))
        check(lambda: st.rankdata(x2.transpose(2, 0, 1), axis=None, method=method))


def test_rankdata_nan_policies_and_declines():
    x = rng.standard_normal(50)
    x[3] = np.nan
    for policy in ("propagate", "omit"):
        check(lambda: st.rankdata(x, nan_policy=policy))
    check(lambda: st.rankdata(x.astype(np.float32)), fast=False)
    check(lambda: st.rankdata([3, 1, 2]), fast=False)
    check(lambda: st.rankdata(np.empty(0)), fast=False)


@pytest.mark.parametrize("n", [50, 3000, 150_000])
def test_rank_based_tests(n, threads):
    x, y = rng.integers(0, 40, n).astype(float), rng.standard_normal(n)
    check(lambda: st.spearmanr(x, y))
    check(lambda: st.mannwhitneyu(x, y, method="asymptotic"))
    check(lambda: st.wilcoxon(y - 0.01, method="approx"))
    check(lambda: st.kruskal(x, y, x[::-1] + 1))
    check(lambda: st.kendalltau(x[:2000], y[:2000]), fast=False)


# --------------------------------------------------------------------------------- distributions
CONT = [
    (st.norm, ()),
    (st.t, (7,)),
    (st.chi2, (3,)),
    (st.beta, (2, 3)),
    (st.gamma, (2.5,)),
    (st.expon, ()),
    (st.lognorm, (0.5,)),
    (st.uniform, ()),
    (st.f, (3, 5)),
    (st.weibull_min, (1.5,)),
    (st.erlang, (3,)),
]
DISC = [
    (st.poisson, (3.5,)),
    (st.binom, (10, 0.3)),
    (st.nbinom, (5, 0.4)),
    (st.zipf, (2.5,)),
    (st.geom, (0.2,)),
    (st.hypergeom, (50, 10, 20)),
    (st.randint, (2, 40)),
]
XS = [
    0.5,
    1,
    np.float64(2.5),
    0.0,
    -1.0,
    7.0,
    np.nan,
    np.inf,
    -np.inf,
    1e300,
    rng.uniform(0.01, 5, 30),
    rng.uniform(0.01, 5, (3, 4)),
    np.array([0.2, np.nan]),
    rng.uniform(0.01, 5, 30).astype(np.float32),
    [0.5, 1.5],
    np.empty(0),
    np.array(1.5),
]
QS = [
    0.3,
    np.float64(0.9),
    0.0,
    1.0,
    1.5,
    np.nan,
    rng.uniform(0.01, 0.99, 25),
    rng.uniform(0.01, 0.99, (2, 5)),
    np.array([0.5, 1.0]),
]
LOCSCALE = [
    {},
    {"loc": 1.5, "scale": 2.0},
    {"loc": -3},
    {"scale": 0.5},
    {"loc": np.array([1.0, 2.0])},
    {"scale": -1.0},
    {"loc": np.nan},
]


@pytest.mark.parametrize("dist,args", CONT, ids=lambda v: getattr(v, "name", ""))
def test_continuous_methods(dist, args):
    fast = 0
    for kw in LOCSCALE:
        for m in ("pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"):
            for x in XS:
                fast += same(lambda: getattr(dist, m)(x, *args, **kw))
        for m in ("ppf", "isf"):
            for q in QS:
                fast += same(lambda: getattr(dist, m)(q, *args, **kw))
    assert fast > 20


@pytest.mark.parametrize("dist,args", DISC, ids=lambda v: getattr(v, "name", ""))
def test_discrete_methods(dist, args):
    ks = [
        0,
        3,
        5.0,
        np.float64(2.0),
        2.5,
        -1,
        1e4,
        np.nan,
        np.inf,
        np.arange(12),
        np.arange(12.0),
        np.arange(12) % 5,
        rng.integers(0, 8, (3, 4)),
        [1, 2],
    ]
    fast = 0
    for kw in ({}, {"loc": 2}, {"loc": 0.5}):
        for m in ("pmf", "logpmf", "cdf", "logcdf", "sf", "logsf"):
            for k in ks:
                fast += same(lambda: getattr(dist, m)(k, *args, **kw))
        for m in ("ppf", "isf"):
            for q in QS:
                fast += same(lambda: getattr(dist, m)(q, *args, **kw))
    assert fast > 20


@pytest.mark.parametrize(
    "dist,args",
    [(st.poisson, (4.2,)), (st.binom, (30, 0.4)), (st.zipf, (2.0,)), (st.nbinom, (5, 0.3)), (st.geom, (0.1,))],
    ids=lambda v: getattr(v, "name", ""),
)
def test_discrete_tables(dist, args):
    n = 20_000
    k = rng.integers(0, 60, n)
    for m in ("pmf", "logpmf", "cdf", "logcdf", "sf", "logsf"):
        check(lambda: getattr(dist, m)(k, *args))
        check(lambda: getattr(dist, m)(k.astype(np.int32), *args))
        check(lambda: getattr(dist, m)(k.astype(float).reshape(100, -1), *args))
    q = rng.uniform(1e-3, 0.99, n)
    for m in ("ppf", "isf"):
        check(lambda: getattr(dist, m)(q, *args), fast=m == "ppf")
    cdf = dist.cdf(np.arange(30), *args)  # quantiles on exact cdf boundaries
    check(lambda: dist.ppf(np.repeat(cdf[(cdf > 0) & (cdf < 1)], 300), *args))
    check(lambda: dist.ppf(np.nextafter(np.repeat(cdf[(cdf > 0) & (cdf < 1)], 300), 1), *args))
    check(lambda: dist.cdf(k, *args, loc=3), fast=False)


def test_frozen():
    for dist, args in CONT + DISC:
        for kw in ({}, {"loc": 1, "scale": 2} if dist in dict(CONT) else {"loc": 1}):
            with fastscipy.disabled():
                ref = dist(*args, **kw)
            fz = dist(*args, **kw)
            assert fz.args == ref.args and fz.kwds == ref.kwds and (fz.a, fz.b) == (ref.a, ref.b)
            assert type(fz.dist) is type(ref.dist) and sorted(vars(fz.dist)) == sorted(vars(ref.dist))
            m = "pdf" if dist in dict(CONT) else "pmf"
            assert fz.dist is not dist and getattr(fz.dist, m).__self__ is fz.dist
            x = np.linspace(0.1, 3, 7)
            assert identical(getattr(fz, m)(x), getattr(ref, m)(x))
            assert identical(fz.cdf(x), ref.cdf(x)) and fz.stats() == ref.stats()
    check(lambda: st.norm(1, 2).cdf(0.3))


# ---------------------------------------------------------------------------------- mode, kde
@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16, np.uint8, np.int8])
def test_mode(dtype):
    a = rng.integers(0, 100, 5000).astype(dtype)
    check(lambda: st.mode(a))
    check(lambda: st.mode(a, axis=None))
    check(lambda: st.mode(a, keepdims=True), fast=False)
    check(lambda: st.mode(a.astype(float)), fast=False)
    check(lambda: st.mode((a.astype(np.int64) * 10**9)), fast=False)
    check(lambda: st.mode(a.reshape(50, 100), axis=1), fast=False)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_gaussian_kde(d, threads):
    data = rng.standard_normal((d, 1500))
    pts = rng.standard_normal((d, 2000))
    for weights in (None, rng.uniform(1, 2, 1500)):
        kde = st.gaussian_kde(data, weights=weights)
        check(lambda: kde(pts))
        check(lambda: kde.evaluate(pts))
        check(lambda: kde.pdf(pts))
    check(lambda: kde(pts[:, :1]), fast=False)
    check(lambda: kde(pts.astype(np.float32)))
