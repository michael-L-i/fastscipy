import numpy as np
import scipy.stats as st


def cases():
    sem_rng = np.random.default_rng(837)
    for shape, dtype in (
        ((10,), np.float32),
        ((30, 8), np.float64),
        ((30, 8), np.float32),
        ((30, 8), np.complex128),
        ((1000, 30), np.float64),
        ((10000, 100), np.float64),
    ):
        sem_data = sem_rng.normal(size=shape).astype(dtype)
        if np.iscomplexobj(sem_data):
            sem_data += 1j * sem_rng.normal(size=shape)
        yield f"sem {np.dtype(dtype)} {shape}", lambda a=sem_data: st.sem(a)
    binned_rng = np.random.default_rng(635)
    for n, dimensions, rows in ((4096, 1, 1), (100_000, 1, 1), (100_000, 2, 3), (1_000_000, 3, 1)):
        sample = binned_rng.uniform(-1, 1, (n, dimensions))
        values = binned_rng.normal(size=(rows, n))
        for statistic in ("min", "max"):
            yield (
                f"binned {statistic} n={n} d={dimensions} rows={rows}",
                lambda s=sample, v=values, op=statistic: st.binned_statistic_dd(s, v, op, bins=10),
            )
    sample = np.arange(100_000)
    for label, values in (("zeros fallback", np.zeros(100_000)), ("small fallback", np.ones(100))):
        yield (
            f"binned min {label}",
            lambda s=sample[: values.size], v=values: st.binned_statistic(s, v, "min"),
            False,
        )
    rng = np.random.default_rng(0)
    for n in (10, 1000):
        a, b = rng.standard_normal(n), rng.standard_normal(n) + 0.1
        yield f"ttest_1samp n={n}", lambda a=a: st.ttest_1samp(a, 0.1)
        yield f"ttest_ind n={n}", lambda a=a, b=b: st.ttest_ind(a, b)
        yield f"ttest_rel n={n}", lambda a=a, b=b: st.ttest_rel(a, b)
        yield f"pearsonr n={n}", lambda a=a, b=b: st.pearsonr(a, b)
        yield f"linregress n={n}", lambda a=a, b=b: st.linregress(a, b)
        yield f"zscore n={n}", lambda a=a: st.zscore(a)
        yield f"skew n={n}", lambda a=a: st.skew(a)
        yield f"kurtosis n={n}", lambda a=a: st.kurtosis(a)
        yield f"sem n={n}", lambda a=a: st.sem(a)
        yield f"entropy n={n}", lambda a=a: st.entropy(np.abs(a))
        yield f"describe n={n}", lambda a=a: st.describe(a)
        yield f"mannwhitneyu n={n}", lambda a=a, b=b: st.mannwhitneyu(a, b)

    for n in (100_000, 1_000_000):
        a = np.random.default_rng(74).standard_normal(n)
        yield f"zscore n={n}", lambda a=a: st.zscore(a)
        yield f"describe n={n}", lambda a=a: st.describe(a)
        yield f"describe n={n} bias=False", lambda a=a: st.describe(a, bias=False)

    pairs = [(rng.standard_normal(20), rng.standard_normal(20)) for _ in range(1000)]
    yield "loop: 1000 ttest_ind + pearsonr n=20", lambda: [(st.ttest_ind(x, y), st.pearsonr(x, y)) for x, y in pairs]

    yield "norm.cdf scalar", lambda: st.norm.cdf(0.5)
    yield "norm.ppf scalar", lambda: st.norm.ppf(0.3, loc=1, scale=2)
    yield "t.sf scalar", lambda: st.t.sf(1.2, 7)
    yield "poisson.pmf scalar", lambda: st.poisson.pmf(3, 2.5)
    yield "frozen norm(1, 2) + cdf", lambda: st.norm(1, 2).cdf(0.5)
    yield "loop: 1000 frozen t(5).cdf", lambda: [st.t(5).cdf(0.1 * i) for i in range(1000)]
    x = rng.standard_normal(1_000_000)
    u = rng.uniform(0.001, 0.999, 1_000_000)
    yield "norm.cdf 1e6", lambda: st.norm.cdf(x)
    yield "norm.ppf 1e6", lambda: st.norm.ppf(u)
    yield "t.cdf 1e6 (df=7)", lambda: st.t.cdf(x, 7)
    yield "poisson.ppf 3e5", lambda: st.poisson.ppf(u[:300_000], 40.0)
    ks = rng.integers(0, 40, 1_000_000)
    yield "poisson.pmf 1e6 ints", lambda: st.poisson.pmf(ks, 12.0)
    yield "zipf.cdf 1e5 ints", lambda: st.zipf.cdf(np.arange(1, 100_001) % 150 + 1, 2.0)

    ties = rng.integers(0, 1000, 1_000_000).astype(float)
    y = rng.standard_normal(1_000_000)
    yield "rankdata 1e6", lambda: st.rankdata(x)
    yield "rankdata 1e6 ties", lambda: st.rankdata(ties)
    yield "rankdata 3000x300 axis=1", lambda: st.rankdata(x[:900_000].reshape(3000, 300), axis=1)
    # Separate RNG keeps existing benchmark fixtures unchanged.
    rank_rng = np.random.default_rng(752)
    for kind in ("random", "ties"):
        batch = (
            rank_rng.standard_normal((512, 2048))
            if kind == "random"
            else rank_rng.integers(0, 16, (512, 2048)).astype(float)
        )
        yield f"rankdata 512x2048 {kind} axis=1", lambda a=batch: st.rankdata(a, axis=1)
    yield "spearmanr 1e6", lambda: st.spearmanr(x, y)
    yield "mannwhitneyu 1e6", lambda: st.mannwhitneyu(x[:500_000], y[:500_000], method="asymptotic")
    yield "wilcoxon 1e6", lambda: st.wilcoxon(x, method="approx")
    yield "kruskal 3 x 1e5", lambda: st.kruskal(x[:100_000], y[:100_000], ties[:100_000])
    yield "mode 1e7 ints", lambda: st.mode(_modes)
    kde = st.gaussian_kde(rng.standard_normal((2, 2000)))
    pts = rng.standard_normal((2, 20_000))
    yield "gaussian_kde 2-D 2000 x 20000", lambda: kde(pts)


_modes = np.random.default_rng(1).integers(0, 1000, 10_000_000, dtype=np.int32)
