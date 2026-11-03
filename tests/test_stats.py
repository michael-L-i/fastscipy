import warnings

import numpy as np
import pytest
import scipy.stats as st
from scipy.stats._stats_py import LinregressResult, PearsonRResult

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


def same(call, *, numerical_validation=None):
    """Check SciPy compatibility, using bytes unless explicitly opting into bounded numeric validation."""
    if numerical_validation is not None:

        def validated(expected, got):
            accepted = numerical_validation(expected, got)
            assert accepted is True or accepted is None, _stats_validation_rejection(
                expected, got, numerical_validation.input_summary()
            )
            return accepted

        before = fast_calls()
        check(lambda: outcome(call), fast=False, numerical_validation=validated)
        return fast_calls() > before
    with fastscipy.disabled():
        expected = outcome(call)
    before = fast_calls()
    got = outcome(call)
    assert identical(expected[0], got[0]) and expected[1] == got[1], f"{expected!r}\n{got!r}"
    return fast_calls() > before


def _stats_validation_rejection(expected, got, inputs=None):
    a, aw = expected
    b, bw = got
    metadata = (
        {key: identical(value, vars(b).get(key)) for key, value in vars(a).items()}
        if hasattr(a, "__dict__") and hasattr(b, "__dict__")
        else None
    )
    fields = (
        {
            name: {
                "expected_dtype": np.asarray(aa).dtype.str,
                "actual_dtype": np.asarray(bb).dtype.str,
                "expected_shape": np.shape(aa),
                "actual_shape": np.shape(bb),
                "expected": repr(aa),
                "actual": repr(bb),
                "expected_hex": np.asarray(aa).tobytes().hex(),
                "actual_hex": np.asarray(bb).tobytes().hex(),
            }
            for name, aa, bb in zip(getattr(a, "_fields", ()), a, b)
        }
        if isinstance(a, tuple) and isinstance(b, tuple)
        else None
    )
    return (
        f"stats numerical validator rejected: warnings_equal={identical(aw, bw)}; "
        f"result_types={type(a)!r}/{type(b)!r}; metadata_equal={metadata!r}; fields={fields!r}; inputs={inputs!r}"
    )


def _stats_numerical_validation(x, y):
    """Validate explicit, bounded Intel Accelerate numerical compatibility.

    ``check`` owns platform/backend gating and compatibility reporting.
    This does not claim byte identity or require a repeated-reference witness.
    The validator preserves result metadata and warning contracts.
    Unitful regression fields get absolute floors in their input data units.
    """
    eps64 = 64 * np.finfo(np.float64).eps

    def close(a, b, atol):
        if identical(a, b):
            return True
        if type(a) is not type(b):
            return False
        aa, bb = np.asarray(a), np.asarray(b)
        if aa.dtype != bb.dtype or aa.shape != bb.shape or aa.ndim != 0 or aa.dtype.kind != "f":
            return False
        if not np.isfinite(aa) or not np.isfinite(bb):
            return False  # NaN payloads and infinity signs stay exact.
        if aa == 0 and bb == 0:
            return False  # Preserve signed zero when both values are zero.
        return bool(np.isclose(aa, bb, rtol=64 * np.finfo(aa.dtype).eps, atol=atol))

    def validate(expected, got):
        a, aw = expected
        b, bw = got
        if not identical(aw, bw) or type(a) is not type(b) or type(a) not in (PearsonRResult, LinregressResult):
            return False
        if vars(a).keys() != vars(b).keys():
            return False
        for aa, bb in zip(a, b):
            if type(aa) is not type(bb) or np.asarray(aa).dtype != np.asarray(bb).dtype:
                return False
            if np.ndim(aa) != 0 or np.ndim(bb) != 0:
                return False
        if type(a) is PearsonRResult:
            if not all(identical(result.correlation, result.statistic) for result in (a, b)):
                return False
            if not all(identical(v, vars(b)[k]) for k, v in vars(a).items() if k != "correlation"):
                return False
            xx, yy = np.asarray(x), np.asarray(y)
            undefined_infinite_input = (
                xx.ndim == yy.ndim == 1
                and xx.size == yy.size
                and xx.size >= 3
                and xx.dtype.kind in "fiu"
                and yy.dtype.kind in "fiu"
                and not (np.isnan(xx).any() or np.isnan(yy).any())
                and (np.isinf(xx).any() or np.isinf(yy).any())
            )
            if undefined_infinite_input and all(np.isnan(value) for result in (a, b) for value in result):
                # These inputs have no finite centered covariance. Official
                # Intel CP312 SciPy alternates the statistic's NaN sign on the
                # exact seeded release failures, including repeated plain calls:
                # https://github.com/michael-L-i/fastscipy/actions/runs/39679272811
                # NaN sign/payload are not a numerical value. Keep both fields
                # NaN, scalar types/dtypes, every warning, metadata and each
                # result's own correlation alias; do not relax other NaN cases.
                return True
            if not (0 <= b.pvalue <= 1 or identical(a.pvalue, b.pvalue)):
                return False
            if not (-1 <= b.statistic <= 1 or identical(a.statistic, b.statistic)):
                return False
            statistic_floor = 64 * np.finfo(np.asarray(a.statistic).dtype).eps
            pvalue_floor = 64 * np.finfo(np.asarray(a.pvalue).dtype).eps
            return close(a.statistic, b.statistic, statistic_floor) and close(a.pvalue, b.pvalue, pvalue_floor)

        if not (0 <= b.pvalue <= 1 or identical(a.pvalue, b.pvalue)):
            return False
        if not (-1 <= b.rvalue <= 1 or identical(a.rvalue, b.rvalue)):
            return False
        if b.stderr < 0 or b.intercept_stderr < 0:
            return False
        if not all(identical(v, vars(b)[k]) for k, v in vars(a).items() if k != "intercept_stderr"):
            return False
        with np.errstate(all="ignore"):
            xx, yy = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
            yscale = np.max(np.abs(yy))
            xspread = np.max(np.abs(xx - np.mean(xx)))
            slope_scale = yscale / xspread
            xscale = np.max(np.abs(xx))
            intercept_scale = yscale + abs(a.slope) * xscale
        if not np.isfinite(slope_scale) or not np.isfinite(intercept_scale):
            return False
        tiny = np.finfo(np.float64).tiny
        floors = (slope_scale, intercept_scale, 1.0, 1.0, slope_scale)
        intercept_ok = close(a.intercept, b.intercept, eps64 * max(intercept_scale, tiny))
        if xspread <= eps64 * xscale:
            # Nearly constant x can make the slope nonunique at floating-
            # point precision. Check each intercept's defining identity,
            # rather than pretending that the two slopes must be identical.
            # Longdouble means are independent of Accelerate dot reductions;
            # the allowance covers mean, multiply, and subtraction rounding
            # in the original float64 calculation, in the input's own units.
            with fastscipy.disabled(), np.errstate(all="ignore"):
                xl, yl = xx.astype(np.longdouble), yy.astype(np.longdouble)
                xm, ym = np.mean(xl), np.mean(yl)
                mx, my = np.mean(np.abs(xl)), np.mean(np.abs(yl))
                intercept_ok = True
                for result in (a, b):
                    slope, intercept = np.longdouble(result.slope), np.longdouble(result.intercept)
                    predicted = ym - slope * xm
                    allowance = eps64 * (my + abs(slope) * mx + abs(predicted) + abs(intercept))
                    if not np.isfinite(allowance) or abs(intercept - predicted) > allowance:
                        intercept_ok = False
        return (
            intercept_ok
            and all(
                close(aa, bb, eps64 * max(scale, tiny))
                for index, (aa, bb, scale) in enumerate(zip(a, b, floors))
                if index != 1
            )
            and close(a.intercept_stderr, b.intercept_stderr, eps64 * max(intercept_scale, tiny))
        )

    def input_summary():
        with fastscipy.disabled(), np.errstate(all="ignore"):
            return {
                name: {
                    "n": int(np.size(values)),
                    "dtype": str(values.dtype),
                    "min": repr(np.min(values)),
                    "max": repr(np.max(values)),
                    "mean": repr(np.mean(values)),
                    "std": repr(np.std(values)),
                    "centered_max": repr(np.max(np.abs(values - np.mean(values)))),
                }
                for name, values in (("x", np.asarray(x)), ("y", np.asarray(y)))
            }

    validate.input_summary = input_summary
    return validate


@pytest.mark.parametrize("kind", ["pearsonr", "linregress"])
def test_stats_reference_validator_contract(kind):
    x, y = np.arange(8.0), np.arange(8.0) / 3
    validate = _stats_numerical_validation(x, y)

    def make(**changes):
        if kind == "pearsonr":
            values = {"statistic": np.float64(0.25), "pvalue": np.float64(0.4)}
            values.update(changes)
            return PearsonRResult(**values, alternative="two-sided", n=np.asarray(8.0), x=x, y=y, axis=-1)
        values = {
            name: np.float64(value)
            for name, value in (
                ("slope", 0.1),
                ("intercept", 0.2),
                ("rvalue", 0.25),
                ("pvalue", 0.4),
                ("stderr", 0.03),
                ("intercept_stderr", 0.04),
            )
        }
        values.update(changes)
        return LinregressResult(**values)

    expected = make()
    coefficient = "statistic" if kind == "pearsonr" else "rvalue"
    assert validate((expected, []), (make(**{coefficient: np.nextafter(np.float64(0.25), 1.0)}), []))
    assert not validate((expected, []), (make(**{coefficient: np.float64(0.26)}), []))
    assert not validate((expected, []), (make(pvalue=np.float64(0.41)), []))
    assert not validate((expected, []), (make(pvalue=np.float64(-1e-16)), []))
    assert not validate((expected, []), (make(pvalue=np.float32(0.4)), []))
    assert not validate((expected, []), (make(pvalue=np.array([0.4, 0.4])), []))
    assert not validate((expected, []), (make(pvalue=np.float64(np.nan)), []))
    assert not validate((expected, []), (make(pvalue=np.float64(np.inf)), []))
    assert not validate((expected, []), (expected, [(RuntimeWarning, "different warning")]))
    if kind == "pearsonr":
        other = make()
        other._alternative = "less"
        assert not validate((expected, []), (other, []))
    else:
        assert not validate((expected, []), (make(slope=np.float64(0.11)), []))
        assert not validate((expected, []), (make(intercept_stderr=np.float64(-0.04)), []))


def test_stats_reference_validator_preserves_tiny_regression_units():
    x, y = np.arange(8.0), np.arange(8.0) * 1e-100
    validate = _stats_numerical_validation(x, y)
    values = dict(
        slope=np.float64(1e-100),
        intercept=np.float64(0),
        rvalue=np.float64(1),
        pvalue=np.float64(0),
        stderr=np.float64(0),
        intercept_stderr=np.float64(0),
    )
    expected = LinregressResult(**values)
    wrong = LinregressResult(**dict(values, slope=np.float64(1e-15)))
    assert not validate((expected, []), (wrong, []))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [7, 30])
def test_stats_reference_validator_real_pearson_returns_literal_true(dtype, n):
    local = np.random.default_rng(718 + n)
    x, y = local.standard_normal((2, n)).astype(dtype)
    with fastscipy.disabled():
        expected = st.pearsonr(x, y)
    validate = _stats_numerical_validation(x, y)
    assert validate((expected, []), (expected, [])) is True

    def other(statistic=expected.statistic, pvalue=expected.pvalue):
        return PearsonRResult(
            statistic,
            pvalue,
            alternative=expected._alternative,
            n=expected._n,
            x=expected._x,
            y=expected._y,
            axis=expected._axis,
        )

    assert validate((expected, []), (other(statistic=np.nextafter(expected.statistic, dtype(1))), [])) is True
    assert validate((expected, []), (other(pvalue=np.nextafter(expected.pvalue, dtype(1))), [])) is True
    assert validate((expected, []), (other(statistic=dtype(expected.statistic + 1e-3)), [])) is False
    assert validate((expected, []), (other(pvalue=dtype(expected.pvalue + 1e-3)), [])) is False
    assert validate((expected, []), (expected, [(RuntimeWarning, "changed warning")])) is False
    changed_metadata = other()
    changed_metadata._alternative = "less"
    assert validate((expected, []), (changed_metadata, [])) is False


@pytest.mark.parametrize("inf_index", [0, -1])
def test_pearson_undefined_infinity_nan_value_contract(inf_index):
    # Real SciPy outputs with strided/read-only caller inputs. The native Intel
    # proof reproduces both release cases' +/-NaN signs with unchanged warnings.
    x = np.arange(34.0)[::2]
    y = np.linspace(0.25, 2.25, 17)
    y[inf_index] = np.inf
    x.flags.writeable = y.flags.writeable = False
    before = (x.tobytes(), y.tobytes())
    with fastscipy.disabled():
        expected, observed_warnings = outcome(lambda: st.pearsonr(x, y))
    assert np.isnan(expected.statistic) and np.isnan(expected.pvalue)
    assert observed_warnings == [(RuntimeWarning, "invalid value encountered in subtract")]
    validate = _stats_numerical_validation(x, y)

    def other(statistic_bits=0xFFF8000000000001, pvalue_bits=0x7FF8000000000002, **changes):
        values = {
            "statistic": np.asarray(statistic_bits, dtype=np.uint64).view(np.float64)[()],
            "pvalue": np.asarray(pvalue_bits, dtype=np.uint64).view(np.float64)[()],
            "alternative": expected._alternative,
            "n": expected._n,
            "x": expected._x,
            "y": expected._y,
            "axis": expected._axis,
        }
        values.update(changes)
        return PearsonRResult(**values)

    assert validate((expected, observed_warnings), (other(), observed_warnings)) is True
    for wrong in (
        other(statistic=np.float64(0)),
        other(pvalue=np.float64(0.5)),
        other(pvalue=np.float64(np.inf)),
        other(pvalue=np.float32(np.nan)),
        other(pvalue=np.array([np.nan])),
        other(alternative="greater"),
        other(axis=123),
    ):
        assert validate((expected, observed_warnings), (wrong, observed_warnings)) is False
    inconsistent_alias = other()
    inconsistent_alias.correlation = np.float64(0)
    assert validate((expected, observed_warnings), (inconsistent_alias, observed_warnings)) is False
    assert validate((inconsistent_alias, observed_warnings), (other(), observed_warnings)) is False
    assert validate((expected, observed_warnings), (other(), [])) is False
    # The observed infinity domain does not authorize changing NaN bits for
    # finite inputs, NaN-containing inputs, or SciPy's special two-point case.
    for xx, yy in ((np.arange(17.0), np.arange(17.0)), (x, np.full(17, np.nan)), (x[:2], y[:2])):
        reject = _stats_numerical_validation(xx, yy)
        assert reject((expected, observed_warnings), (other(), observed_warnings)) is False
    assert before == (x.tobytes(), y.tobytes())


def test_stats_reference_validator_ill_conditioned_intercept_identity():
    # Exact seed12345/4t/linregress iteration69 from the Intel rejection.
    # x is truly constant, but np.mean leaves a one-ULP centered residual.
    # SciPy warns: no uniquely accurate slope is asserted for this input.
    value = 0.05016188054896638
    x = np.full(30, value)
    y = np.array(
        [
            0.4708395161929273,
            1.5922804525973209,
            3.2527440791749798,
            0.6741764876659379,
            0.8306254005235675,
            1.9816696902556956,
            0.5878055799310182,
            1.25537498638792,
            0.3490925474552964,
            1.1692190818499577,
            0.41807620936171314,
            0.9532265933038834,
            0.4151773016093703,
            0.6272154891298194,
            0.44691347284617833,
            0.7397013404947995,
            1.040745627028402,
            0.38564974778307687,
            0.39310202220407364,
            0.139466468535131,
            0.948403778094678,
            0.8570980571890897,
            1.3529817447713361,
            1.1351274804849003,
            0.11939894820968862,
            0.046006762023237786,
            2.0166775645072645,
            0.91256621988236,
            0.4604600672527679,
            0.045124281167207264,
        ]
    )
    with fastscipy.disabled():
        expected, observed_warnings = outcome(lambda: st.linregress(x, y))
    assert any("catastrophic cancellation" in message for _, message in observed_warnings)
    assert np.isfinite(expected.slope)
    for shift in (0.0, value):
        shifted_x = x + shift
        validate = _stats_numerical_validation(shifted_x, y)
        fields = dict(zip(expected._fields, expected))
        fields["intercept"] = expected.intercept - expected.slope * shift
        fields["intercept_stderr"] = expected.intercept_stderr * (1 + shift / value)
        reference = LinregressResult(**fields)

        def other(slope=expected.slope + 0.5, **changes):
            mean_x = np.mean(shifted_x.astype(np.longdouble))
            mean_y = np.mean(y.astype(np.longdouble))
            values = dict(fields, slope=np.float64(slope), intercept=np.float64(mean_y - slope * mean_x))
            values.update(changes)
            return LinregressResult(**values)

        def accepted(candidate, warns=observed_warnings):
            return validate((reference, observed_warnings), (candidate, warns))

        assert accepted(other()) is True
        assert accepted(other(intercept=other().intercept + 0.01)) is False
        assert accepted(other(slope=1e9)) is False
        assert accepted(other(rvalue=np.float64(0.25))) is False
        assert accepted(other(pvalue=np.float64(0.5))) is False
        assert accepted(other(), []) is False


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_sem_nd_and_dtypes(dtype, layout, threads):
    local = np.random.default_rng(836)
    a = local.standard_normal((9, 7, 5)).astype(dtype)
    if np.iscomplexobj(a):
        a += 1j * local.standard_normal(a.shape)
    if layout == "F":
        a = np.asfortranarray(a)
    elif layout == "strided":
        a = a[::-1, ::2, ::-1]
    a.flags.writeable = False
    before = a.tobytes()
    for axis in (0, 1, 2, -1, None):
        for ddof in (0, 1, 2):
            check(lambda: st.sem(a, axis=axis, ddof=ddof))
    check(lambda: st.sem(a.ravel()))
    assert a.tobytes() == before


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_sem_nd_warning_and_error_events(dtype, monkeypatch):
    from fastscipy import _patch

    monkeypatch.setattr(_patch, "STRICT", False)
    for data in (
        np.empty((0, 7), dtype=dtype),
        np.ones((1, 7), dtype=dtype),
        np.full((9, 7), np.inf, dtype=dtype),
        np.full((9, 7), np.nan, dtype=dtype),
        np.full((9, 7), 1e30, dtype=dtype),
        np.zeros((9, 7), dtype=dtype),
    ):
        for opts in ({}, {"axis": None}, {"ddof": 20}, {"axis": 5}, {"keepdims": True}, {"nan_policy": "omit"}):
            for err in ({}, {"all": "warn"}, {"all": "raise"}):
                with np.errstate(**err):
                    same(lambda: st.sem(data, **opts))


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
        validation = _stats_numerical_validation(*args) if name in ("pearsonr", "linregress") else None
        fast += same(lambda: f(*args, **kw), numerical_validation=validation)
        assert all(a.tobytes() == b.tobytes() for a, b in zip((a for a in args if isinstance(a, np.ndarray)), before))
        res = outcome(lambda: f(*args, **kw))[0]
        if hasattr(res, "confidence_interval"):
            assert same(lambda: res.confidence_interval(0.9)) is False
    assert fast > 30, f"{name}: fast path ran {fast}/150 times"


def test_functions_decline_odd_input():
    a = rng.standard_normal(20)
    for f, args in [
        (st.skew, ([1.0, 2.0, 4.0],)),
        (st.sem, (a.astype(np.int32),)),
        (st.zscore, (np.ones(5),)),
        (st.ttest_ind, (a, a[:1])),
        (st.pearsonr, (a, a[:2])),
        (st.entropy, (a, a)),
        (st.kurtosis, (np.ma.masked_array(a),)),
        (st.describe, (a.reshape(4, 5),)),
    ]:
        assert same(lambda: f(*args)) is False


@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("n", [2, 3, 4, 17, 257, 1000])
@pytest.mark.parametrize("bias", [True, False])
def test_describe_shared_moments(n, bias, num_threads):
    local_rng = np.random.default_rng(510 + n)
    values = local_rng.standard_normal(2 * n)
    arrays = [values[:n], values[::2], values[:n][::-1], np.round(values[:n]), np.zeros(n)]
    readonly = values[:n].copy()
    readonly.flags.writeable = False
    arrays.append(readonly)
    old = fastscipy.get_num_threads()
    fastscipy.set_num_threads(num_threads)
    try:
        for a in arrays:
            before = a.tobytes()
            for axis in (0, -1):
                for ddof in (0, 1):
                    assert same(lambda: st.describe(a, axis=axis, ddof=ddof, bias=bias))
            assert a.tobytes() == before
    finally:
        fastscipy.set_num_threads(old)


@pytest.mark.parametrize(
    "a,kw",
    [
        (np.empty(0), {}),
        (np.ones(10), {}),
        (1e8 + np.arange(10) * 1e-8, {}),
        (np.array([1.0, np.nan, 3.0]), {}),
        (np.array([1.0, np.inf, 3.0]), {}),
        (np.array([1e200, -1e200, 0.0]), {}),
        (np.arange(10, dtype=np.float32), {}),
        (np.arange(10, dtype=np.int64), {}),
        (np.arange(10.0).reshape(2, 5), {}),
        (np.arange(10.0), {"ddof": 10}),
        (np.arange(10.0), {"ddof": -1}),
        (np.arange(10.0), {"ddof": 1.0}),
        (np.arange(10.0), {"bias": 1}),
        (np.arange(10.0), {"axis": None}),
        (np.arange(10.0), {"nan_policy": "omit"}),
    ],
)
def test_describe_shared_moments_declines(a, kw):
    from fastscipy._stats_loop import _declining, _describe

    assert _declining(_describe)(a, **kw) is NotImplemented
    # The original describe may still reach patched skew/kurtosis helpers.
    same(lambda: st.describe(a, **kw))


# ----------------------------------------------------------------------------------------- rankdata
@pytest.mark.parametrize("name", ["skew", "kurtosis", "zscore", "ttest_1samp", "ttest_ind"])
@pytest.mark.parametrize("n", [2, 17, 257])
def test_reused_means_and_centered_data(name, n):
    r = np.random.default_rng(n + 891)
    values = r.standard_normal(2 * n)
    f = getattr(st, name)
    for a in (values[:n], values[::2], values[:n][::-1], np.zeros(n)):
        a.flags.writeable = False
        before = a.tobytes()
        args = (a, 0.1) if name == "ttest_1samp" else (a, values[1::2]) if name == "ttest_ind" else (a,)
        for axis in (0, -1):
            same(lambda: f(*args, axis=axis))
        assert a.tobytes() == before


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


@pytest.mark.parametrize("threads", [1, 2], indirect=True)
@pytest.mark.parametrize("n", [33, 1025, 4097])
@pytest.mark.parametrize("kind", ["float64", "int64", "uint64"])
@pytest.mark.parametrize("method", ["average", "min", "max", "dense", "ordinal"])
def test_rankdata_batched_row_independence(n, kind, method, threads):
    from scipy.stats import _stats_py

    # Adjacent rows exercise different histograms, constant radix digits, and
    # ties. Enough rows cross the parallel threshold for both radix sizes.
    rng = np.random.default_rng(219)
    x = rng.integers(0, 32, size=(65, n)).astype(kind)
    x[::3] = 0
    x[1::3] = np.arange(n, dtype=kind)[::-1]
    if kind == "float64":
        x[2::3, :7] = [-np.inf, -0.0, 0.0, np.inf, np.nan, 5e-324, -5e-324]
    elif kind == "int64":
        x[2::3, :3] = [np.iinfo(np.int64).min, -1, np.iinfo(np.int64).max]
    else:
        x[2::3, :2] = [2**63, np.iinfo(np.uint64).max]
    before = x.tobytes()
    check(lambda: _stats_py._rankdata(x, method, return_sorted=True, return_ties=True))
    assert x.tobytes() == before


@pytest.mark.parametrize("method", ["average", "min", "max", "dense", "ordinal"])
def test_rankdata_float32(method, threads):
    from scipy.stats import _stats_py

    # Include adjacent floats, subnormals, signed zero, ties and readonly strides.
    x = np.random.default_rng(371).integers(-8, 8, (7, 2, 1031)).astype(np.float32)
    x[0, 0, :7] = [0.0, -0.0, 1.0, np.nextafter(np.float32(1), np.float32(2)), 1e-45, -1e-45, 0.0]
    x = x[:, :, ::-2]
    x.flags.writeable = False
    before = x.tobytes()
    for axis in (0, 1, -1, None):
        check(lambda: st.rankdata(x, axis=axis, method=method))
    check(lambda: _stats_py._rankdata(x, method, return_sorted=True, return_ties=True))
    assert x.tobytes() == before


def test_rankdata_float32_nonfinite_declines():
    from scipy.stats import _stats_py

    for bits in (0x7F800000, 0xFF800000, 0x7FC00123, 0x7F800123):
        x = np.array([0x40000000, bits, 0x80000000, 0], dtype=np.uint32).view(np.float32)
        for err in ("warn", "raise"):
            with np.errstate(invalid=err):
                assert not same(lambda: _stats_py._rankdata(x, "average", return_sorted=True, return_ties=True))


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
    check(lambda: st.mode(a, keepdims=True))
    check(lambda: st.mode(a.astype(float)), fast=False)
    check(lambda: st.mode((a.astype(np.int64) * 10**9)), fast=False)
    check(lambda: st.mode(a.reshape(50, 100), axis=1), fast=False)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16, np.uint8, np.int8])
@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_mode_batched(dtype, axis, threads):
    a = np.random.default_rng(187).integers(0, 16, (256, 512)).astype(dtype)
    original = a.copy()
    for values in (a, a.T, a[::-1, ::-1]):
        for keepdims in (False, True):
            for policy in ("propagate", "omit", "raise"):
                check(lambda: st.mode(values, axis=axis, keepdims=keepdims, nan_policy=policy))
    np.testing.assert_array_equal(a, original)
    # Every slice ties; the smallest value wins, including near integer limits.
    for low in (np.iinfo(dtype).min, np.iinfo(dtype).max - 1):
        tied = np.tile(np.array([low + 1, low], dtype=dtype), (3, 256))
        check(lambda: st.mode(tied, axis=1))
        result = st.mode(tied, axis=1)
        assert np.all(result.mode == low) and np.all(result.count == 256)


def test_mode_batched_chunks():
    a = np.random.default_rng(188).integers(-1024, 1024, (5, 70, 4096), dtype=np.int64)
    check(lambda: st.mode(a, axis=-1, keepdims=True))
    check(lambda: st.mode(a.transpose(2, 0, 1), axis=0))


def test_mode_batched_fallbacks():
    for a in (
        np.empty((2, 0), dtype=int),
        np.zeros((2, 512)),
        np.full((2, 512), np.nan),
        np.arange(1024).reshape(2, 512) * 10**9,
    ):
        assert not same(lambda: st.mode(a, axis=1))
    a = np.zeros((2, 512), dtype=np.int32)
    for kw in ({"axis": 3}, {"axis": (0, 1)}, {"nan_policy": "invalid"}):
        same(lambda: st.mode(a, **kw))


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


@pytest.mark.parametrize("statistic", ["min", "max", np.min, np.max])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_binned_extrema(statistic, ndim):
    rng = np.random.default_rng(635)
    sample = rng.uniform(-2, 2, (6000, ndim))
    values = rng.integers(1, 40, (2, 6000)).astype(float)
    before = sample.copy(), values.copy()
    for expanded in (False, True):
        assert same(
            lambda: st.binned_statistic_dd(
                sample, values, statistic, bins=8, range=[(-1, 1)] * ndim, expand_binnumbers=expanded
            )
        )
    with fastscipy.disabled():
        bins = st.binned_statistic_dd(sample, values, "count", bins=8)
    assert not same(lambda: st.binned_statistic_dd(sample, values[:, ::-1], statistic, binned_statistic_result=bins))
    assert identical(sample, before[0]) and identical(values, before[1])


@pytest.mark.parametrize("kind", ["zero", "nan", "inf", "complex", "small", "empty"])
def test_binned_extrema_declines(kind):
    values = np.ones(6000)
    if kind in ("zero", "nan", "inf"):
        values[::2] = {"zero": -0.0, "nan": np.nan, "inf": np.inf}[kind]
    elif kind == "complex":
        values = values.astype(complex)
    elif kind == "small":
        values = values[:12]
    else:
        values = values[:0]
    for statistic in ("min", "max"):
        assert not same(lambda: st.binned_statistic(np.arange(values.size), values, statistic))


def test_binned_extrema_errors_and_callback():
    sample = np.arange(6000, dtype=float)
    values = np.ones(6000)
    for kwargs in ({"bins": -1}, {"bins": [0, 0]}, {"range": [(2, 1)]}):
        same(lambda: st.binned_statistic_dd(sample[:, None], values, "min", **kwargs))
    same(lambda: st.binned_statistic(sample, values[:-1], "max"))
    sample[0] = np.nan
    same(lambda: st.binned_statistic(sample, values, "min"))
    calls = []

    def callback(a):
        calls.append(a.copy() if isinstance(a, np.ndarray) else list(a))
        return len(a)

    sample[0] = 0
    with fastscipy.disabled():
        expected = st.binned_statistic(sample, values, callback)
    expected_calls = calls.copy()
    calls.clear()
    assert identical(st.binned_statistic(sample, values, callback), expected)
    assert identical(calls, expected_calls)


@pytest.mark.parametrize("disabled", [False, True])
def test_binned_extrema_reduction_fallback(monkeypatch, disabled):
    from scipy.stats import _binned_statistic as sp

    original = sp._calc_binned_statistic.__wrapped__

    def fallback(*args):
        if disabled:
            with fastscipy.disabled():
                return original(*args)
        return original(*args)

    monkeypatch.setattr(sp, "_calc_binned_statistic", fallback)
    sample = np.zeros(6000)
    values = np.arange(1, 6001, dtype=float)[::-1]
    for statistic in ("min", "max"):
        assert not same(lambda: st.binned_statistic(sample, values, statistic, bins=8, range=(-1, 1)))


@pytest.mark.parametrize("statistic", ["min", "max"])
@pytest.mark.parametrize("mutation", ["nan", "zero"])
@pytest.mark.parametrize("disable", [False, True])
def test_binned_extrema_bin_conversion_mutates_values(statistic, mutation, disable):
    from fastscipy import _patch

    values = np.ones(4096)
    sample = np.arange(4096, dtype=float)[:, None]

    class Bins:
        def __index__(self):
            values[0] = np.nan if mutation == "nan" else -0.0
            if mutation == "zero":
                values[1::2] = 0.0
            if disable:
                _patch.enabled = False
            return 10

    def call():
        values.fill(1)
        return st.binned_statistic_dd(sample, values, statistic, bins=Bins())

    try:
        assert not same(call)
    finally:
        _patch.enabled = True


def test_binned_extrema_conversion_error_not_retried():
    values = np.ones(4096)
    calls = []

    class Bins:
        def __index__(self):
            calls.append("index")
            values[0] = np.nan
            raise ValueError("bin conversion failed")

    assert not same(lambda: st.binned_statistic(np.arange(4096), values, "max", bins=Bins()))
    assert calls == ["index", "index"]  # Once per reference/candidate call.
