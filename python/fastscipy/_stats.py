"""scipy.stats: lean per-function fast paths, cheap distribution methods and frozen
distributions, Rust ranking, and a few algorithmic shortcuts.

Order matters: distribution methods first (the discrete shortcuts wrap them), ranking and
shortcuts next, per-function fast paths last (``mannwhitneyu`` ranks through the Rust ``_rankdata``).
"""
import os

import numpy as np
import scipy.stats as stats

from . import _patch, _stats_algos, _stats_dist, _stats_loop, _stats_rank


def patch():
    _patch.require_scipy("1.18.")
    if os.environ.get("SCIPY_ARRAY_API", "0") not in ("", "0"):
        raise RuntimeError("SCIPY_ARRAY_API is set")
    _stats_dist.patch()
    _stats_rank.patch()
    _stats_algos.patch()
    _stats_loop.patch()


def smoke():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=20), rng.normal(size=20)
    for f in (stats.ttest_1samp, stats.ttest_rel):
        f(a, 0.0) if f is stats.ttest_1samp else f(a, b)
    stats.ttest_ind(a, b), stats.pearsonr(a, b), stats.linregress(a, b), stats.mannwhitneyu(a, b)
    stats.zscore(a), stats.skew(a), stats.kurtosis(a), stats.sem(a), stats.entropy(np.abs(a)), stats.describe(a)
    stats.rankdata(np.round(a, 1)), stats.spearmanr(a, b)
    stats.norm.cdf(0.5), stats.norm(1, 2).ppf(0.3), stats.poisson.pmf(3, 2.0)
    big = rng.integers(0, 50, 5000)
    stats.poisson.cdf(big, 7.0), stats.mode(big)
    stats.gaussian_kde(rng.normal(size=(2, 2000)))(rng.normal(size=(2, 2000)))
