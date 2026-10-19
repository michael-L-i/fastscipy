import pickle
import warnings

import numpy as np
import pytest
import scipy.special as sc
import scipy.special._ufuncs as _ufuncs

import fastscipy
from fastscipy.testing import check, fast_calls

N = 300_000
RNG = np.random.default_rng(0)
# Hang or crash in SciPy's own serial ufunc on NaN/inf input.
SKIP = {"hyperu", "lpmv", "pbdv", "pbvv", "_stirling2_inexact"}


@pytest.fixture(params=[1, 4], ids=["1t", "4t"], autouse=True)
def threads(request):
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(request.param)
    yield request.param
    fastscipy.set_num_threads(old)


def x64(n=N):
    return RNG.uniform(0.1, 5, n)


def go(call, threads):
    check(call, fast=threads > 1)


@pytest.mark.parametrize("dt", [np.float64, np.float32])
@pytest.mark.parametrize("name", ["erf", "gammaln", "i0", "expit", "digamma", "ndtri_exp"])
def test_unary(name, dt, threads):
    x = x64().astype(dt)
    go(lambda: getattr(sc, name)(x), threads)


def test_binary_ternary_scalar(threads):
    x = x64()
    go(lambda: sc.jv(1.5, x), threads)
    go(lambda: sc.jv(x, 2.0), threads)
    go(lambda: sc.betainc(2.0, 3.5, x / 6), threads)
    go(lambda: sc.betainc(x, x + 1, x / 6), threads)
    go(lambda: sc.jv(x.astype(np.float32), 1.5), threads)


def test_integer_and_complex_inputs(threads):
    ints = RNG.integers(1, 50, N)
    go(lambda: sc.gammaln(ints), threads)
    go(lambda: sc.gammaln(np.arange(1, N, dtype=np.int32)), threads)
    z = x64() + 1j * x64()
    go(lambda: sc.jv(0.5, z), threads)


def test_multi_output(threads):
    x = x64()
    go(lambda: sc.sici(x), threads)
    go(lambda: sc.airy(x), threads)
    go(lambda: sc.ellipj(x, 0.3), threads)


@pytest.mark.parametrize("shape", [(500, 600), (3, 100_000), (200_000, 2), (2, 3, 50_000)])
def test_shapes(shape, threads):
    x = x64(int(np.prod(shape))).reshape(shape)
    go(lambda: sc.jv(1.5, x), threads)


def test_broadcast(threads):
    a, b = x64(1000)[:, None], x64(500)[None, :]
    c = x64(300_000).reshape(300, 1000)
    go(lambda: sc.betainc(a[:300] + 1, c, c / 6), threads)
    row = x64(1000)
    go(lambda: sc.jv(row, c), threads)
    check(lambda: sc.jv(a, b), fast=False)  # large output from small inputs: declined


def test_special_values(threads):
    x = x64()
    x[::7], x[1::11], x[2::13] = np.nan, np.inf, -0.0
    go(lambda: sc.gammaln(x), threads)
    go(lambda: sc.erfinv(x - 2.5), threads)


def test_declined_still_correct():
    x = x64()
    for call in (
        lambda: sc.erf(x[::2]),  # strided
        lambda: sc.erf(np.asfortranarray(x.reshape(500, 600))),
        lambda: sc.erf(list(x[:5000])),
        lambda: sc.erf(x.astype(np.longdouble)),
        lambda: sc.erf(x.astype(object)[:100]),
        lambda: sc.erf(x, out=np.empty_like(x)),
        lambda: sc.erf(x, out=np.zeros_like(x), where=x > 1),
        lambda: sc.erf(x, dtype=np.float32),
        lambda: sc.erf(x[:100]),
        lambda: sc.erf(np.float64(1.0)),
        lambda: sc.erf(2.0),
        lambda: sc.erf(np.empty(0)),
    ):
        try:
            check(call, fast=False)
        except TypeError:
            pass  # object arrays raise in SciPy too


def test_out_buffer_written():
    x, out = x64(), np.empty(N)
    r = sc.erf(x, out=out)
    assert r is out and np.array_equal(out, sc.erf.__wrapped__(x))


def test_errstate_propagates(threads):
    x = x64()
    x[::5000] = 0.0
    with sc.errstate(all="raise"):
        for call in (lambda: sc.digamma(-x), lambda: sc.gammaln(x - 5)):
            with pytest.raises(sc.SpecialFunctionError):
                call()
    with sc.errstate(all="warn"), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sc.digamma(-np.round(x))
    assert w and w[0].category is sc.SpecialFunctionWarning
    with sc.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("error")
        sc.digamma(-np.round(x))
    assert sc.geterr()["singular"] == "ignore"


def test_numpy_errstate_propagates(threads):
    x = x64()
    x[0] = 1e308
    with np.errstate(all="raise"):
        check(lambda: sc.xlogy(x, x), fast=False)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("error")
        sc.xlogy(x, x)


def test_ufunc_attributes():
    f, g = sc.jv, sc.jv.__wrapped__
    assert (f.nin, f.nout, f.types, f.__name__, f.__doc__) == (g.nin, g.nout, g.types, g.__name__, g.__doc__)
    assert f.identity is g.identity and callable(f.outer) and callable(f.reduce)
    a, b = x64(30), x64(40)
    assert np.array_equal(f.outer(a, b), g.outer(a, b))
    assert _ufuncs.jv is f is sc.jv


def test_pickle():
    assert pickle.loads(pickle.dumps(sc.erf)) is sc.erf
    assert pickle.loads(pickle.dumps(sc.digamma)) is sc.digamma


def test_unpatch_restores():
    f = sc.erf
    fastscipy.unpatch(["special"])
    assert isinstance(sc.erf, np.ufunc) and sc.erf is f.__wrapped__
    fastscipy.patch(["special"])
    assert sc.erf is not f and not isinstance(sc.erf, np.ufunc)


def test_rebound_inside_scipy():
    import scipy.stats._continuous_distns as cd
    import scipy.spatial.distance as sd

    assert sd.__dict__.get("xlogy", sc.xlogy) is sc.xlogy
    assert cd.sc.gammaln is sc.gammaln


def test_disabled_runs_scipy():
    x = x64()
    before = fast_calls()
    with fastscipy.disabled():
        sc.erf(x)
    assert fast_calls() == before


def test_all_ufuncs_identical(threads):
    """Every ufunc on float64 arguments with NaN/inf/0 mixed in, bytes against SciPy."""
    n, fast = 120_000, 0
    for name, fast_uf in vars(_ufuncs).items():
        uf = getattr(fast_uf, "__wrapped__", None)
        if not isinstance(uf, np.ufunc) or name in SKIP or name.startswith(("obl_", "pro_", "mathieu_")):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            args = []
            for _ in range(uf.nin):
                a = RNG.uniform(0.1, 5, n)
                a[::97], a[1::101], a[2::103] = np.nan, np.inf, 0.0
                args.append(a)
            if not any(t.startswith("d" * uf.nin) for t in uf.types):
                continue
            args = [a.astype(np.float64) for a in args]
            with fastscipy.disabled():
                try:
                    expected = uf(*args)
                except Exception:
                    continue
            before = fast_calls()
            got = getattr(_ufuncs, name)(*args)
            fast += fast_calls() > before
        exp = expected if isinstance(expected, tuple) else (expected,)
        res = got if isinstance(got, tuple) else (got,)
        assert all(e.dtype == r.dtype and e.tobytes() == r.tobytes() for e, r in zip(exp, res)), name
    assert fast > (50 if threads > 1 else -1) and (threads > 1 or fast == 0)


def test_error_warnings_match_scipy(threads):
    x = np.full(N, -1.0)  # gamma is singular at every element

    def warned():
        with sc.errstate(all="warn"), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sc.gamma(x)
        return [(i.category, str(i.message)) for i in w]

    with fastscipy.disabled():
        expected = warned()
    assert expected and warned() == expected
