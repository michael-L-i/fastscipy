# Downstream application pilots

These are three bounded applications of existing patches, not certification of
whole downstream libraries. No runtime dependency or production patch is added.
Patch **before importing the downstream library**: scikit-learn's Gaussian
process module captures `scipy.linalg.cho_solve` and `solve_triangular` aliases.
Each baseline runs in a fresh process that never calls `fastscipy.patch()`.

## Reproduction

Install the optional pilot dependencies in a separate environment:

```sh
uv pip install numpy==2.5.4 scipy==1.18.1 scikit-image==0.26.0 scikit-learn==1.9.1 statsmodels==0.15.0
PYTHONPATH=python python tools/pilot_downstream.py --output /tmp/downstream --repeat 99 --reverse-order
```

Build the native extension first. The driver checks the embedded native source
fingerprint against the checkout before launching children. Imports and process
startup are excluded from timings. Each child performs one correctness/warmup
call, at least 100 ms of warmup, then nine complete application calls by default.
The retained final runs use 99 calls and reverse the child order. Output arrays
are compared byte for byte, including dtype, shape and strides; warning
category/message and unchanged input bytes are checked. Counts include all warmup
and timed calls; the time-based warmup means totals can differ between runs.
Timings cover fit/predict or the full image/time-series workflow, not isolated
kernels.

The driver fixes fastscipy to two threads and BLAS/OpenMP to one. These settings
are independent of scikit-learn's `n_jobs`; `threadpoolctl` does not control
fastscipy's pools. When using process workers, initialize patches before downstream
imports in each worker. See [scikit-learn parallelism documentation](https://scikit-learn.org/stable/computing/parallelism.html).

## Source-level boundaries

- **scikit-image**: the 512×512 float image passes through a 7×7 median, Gaussian
  smoothing, Otsu threshold, boolean connected components, and region measurements.
  [Gaussian wraps ndimage](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html#skimage.filters.gaussian).
  In the installed 0.26.0 source, the boolean path of
  [`measure.label`](https://scikit-image.org/docs/0.26.x/api/skimage.measure.html#skimage.measure.label) uses `ndimage.label`; integer labeling uses scikit-image's own Cython code.
  `filters.rank` and other independent scikit-image kernels are not patched.
- **scikit-learn**: GaussianProcessRegressor fits 128 two-dimensional training
  samples and predicts mean/std at 256 points, with hyperparameter optimization
  disabled. [Its source](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/gaussian_process/_gpr.py)
  calls SciPy factor/triangular solves. Kernel construction also reaches the
  accelerated `pdist` path; `cdist` falls back for this input. Cholesky and the
  vendor BLAS/LAPACK arithmetic remain unchanged. This small workload does not represent
  cross-validation, hyperparameter search, or large dense GP training.
- **statsmodels**: seasonal decomposition of 24 series with 2,400 observations
  and a 12-sample period returns observed, trend, seasonal, and residual arrays.
  [The filtering implementation](https://www.statsmodels.org/stable/_modules/statsmodels/tsa/filters/filtertools.html)
  calls `scipy.signal.convolve`. Other estimator code may primarily use NumPy or
  its own compiled kernels and does not automatically gain from these seams.

Only the returned values listed above are checked, not the complete estimator
object graph, pandas metadata, sparse arrays, callbacks, or every supported dtype.
Fresh-process pairs reduce alias contamination; these short local timings do not
establish a platform-wide guarantee or a confidence interval.

## Local measurements

Python 3.13.6 on macOS ARM64, NumPy 2.5.4, SciPy 1.18.1,
scikit-image 0.26.0, scikit-learn 1.9.1, statsmodels 0.15.0. The native artifact
matches source fingerprint `5f9095417fd0f5fa4bfdd5ef4364c8ee926d6379342c5b9edfd7aa630023fdc4`.
All three comparisons passed byte/structure/warning/input checks. The initial
process-tree monitor recorded 217 MiB peak under a 1 GiB limit; final runs peaked
at 214 MiB. Timings ran in a reserved interval without other project test/build
jobs.

| Application | SciPy median | Patched median | Interpretation |
|---|---:|---:|---|
| Image segmentation, reversed order | 109.099 ms | 3.658 ms | **29.82×**; median, Gaussian, label, find_objects fast paths |
| GP fit/predict, reversed order | 0.4651 ms | 0.4313 ms | 1.078×; pdist and factor/triangular solves reached |
| GP fit/predict, forward order | 0.4626 ms | 0.4332 ms | 1.068×; modest local benefit only |
| Seasonal, reversed order | 0.5988 ms | 0.5834 ms | 1.026×, zero patched calls |
| Seasonal, forward order | 0.5626 ms | 0.5710 ms | 0.985×, zero patched calls |

The substantive opportunity is float-image cleanup and segmentation through
scikit-image's existing SciPy seams. The GP workload shows a small local gain,
not evidence for broad scikit-learn acceleration. Seasonal decomposition reaches
no patched path in this configuration, so there is no attributable speedup.

The initial nine-repeat run used only one warmup call and reported 31.29×,
1.37×, and an apparent 1.79× for segmentation, GP, and seasonal respectively.
After adding time-based warmup and reversing process order, the short-workload
ratios settled near the values above. This zero-hit control illustrates why raw
ratios are insufficient. The initial seasonal call also used the deprecated
`extrapolate_trend='freq'`; its FutureWarning category and message matched exactly.
The final script uses the supported `period` spelling; final warning lists are empty.

Raw samples, versions, returned-array structure, and per-function hit counts are
in [initial results](results/), [reversed results](reverse/),
[forward GP](forward-gp/) and [forward seasonal](forward-seasonal/). Output arrays
are regenerated by the driver and excluded from version control. The adjacent
`*.log.monitor.json` files record bounded execution. See
[pilot source](../../../tools/pilot_downstream.py). No production code changed,
so validation consists of running these application comparisons and Ruff on the
new driver, rather than rerunning unrelated kernels.
