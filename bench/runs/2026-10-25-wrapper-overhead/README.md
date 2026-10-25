# Lower call-accounting overhead

Every instrumented call increments a mapping keyed by `(function, path)`.
The implementation used `collections.Counter`, although consumers only use
lookup, increment, iteration, clearing, and conversion to a plain dictionary.
The candidate uses a small dictionary subclass whose `__missing__` returns
zero. Missing reads still do not insert entries; `fastscipy.stats()` still
returns a detached plain-dictionary snapshot. Pytest worker aggregation and
all fast/fallback/error counting remain enabled.

No numerical kernels, input guards, callback retry rules, patch registration,
or public APIs change. Counter-specific multiset operations on the private
`_patch.counts` object are intentionally not supported; no repository consumer
uses them.

## Measurement

`probe.py` alternates the old Counter and candidate dictionary in the same
patched process, keeping calls and inputs identical. Each of nine paired
batches contains 10,000 calls after 100 warmup calls; the table reports median
nanoseconds per call. Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1, macOS ARM64;
one numerical thread and BLAS/OpenMP limited to one. The final probe followed
the project suite; a transiently loaded probe was repeated before retaining
these results. Raw paired samples and
environment metadata are in `results.json`.

| Call | Counter ns | Dictionary ns | Before / after |
|---|---:|---:|---:|
| Wrapper accepting a trivial call | 164.3 | 132.1 | 1.24× |
| Wrapper declining a trivial call | 185.4 | 152.2 | 1.22× |
| `pdist`, 16×3 float64 | 912.4 | 881.6 | 1.03× |
| `fftn`, 8×8 float64 | 1826.7 | 1789.8 | 1.02× |
| `dctn`, 8×8 float64 | 2017.0 | 1977.6 | 1.02× |
| Integer `dctn` fallback | 3574.1 | 3611.0 | 0.99× |
| Batched `LinearOperator` fallback, control | 3192.8 | 3195.7 | 1.00× |

The synthetic wrapper cases isolate accounting overhead; they are not
application speedups. The absolute saving is about 30–40 ns per increment in
these local probes. Small numerical calls benefit modestly; differences at
1–3% are noise-sensitive. The batched operator fallback does not increment a
counter, and its difference is measurement variation. This change does not
resolve all fallback overhead or establish cross-platform performance.
These are comparisons against the previous fastscipy accounting, not speedups
against unpatched SciPy.

A tuple-key caching experiment was rejected: its measured benefit was only a
few nanoseconds and did not justify changes throughout the wrappers.

## Validation

- Full project suite: **7,997 passed**, 58 warnings, 110.96 seconds. The
  resource monitor measured **605 MiB** peak process-tree RSS under a 1 GiB
  limit, with no timeout or memory stop.

- New tests cover missing-read behavior, snapshot isolation, pytest worker
  aggregation, wrapper metadata and accepted/declined/disabled/error/strict
  paths.
- Paired upstream `scipy.fft.tests.test_real_transforms`: **3,056 tests**, zero
  changed outcomes, zero fast-path errors.
- Ruff check and formatting pass for changed files.

The final suite used a fresh CPython 3.13 dev-release native build of unchanged
main `8ea95b24f88256f0f062b1a6a47305ea31c083a2`, SHA256
`9c1b963ebc4d20c08baf91cbd839ca6c96f45a21d82cbb3515fe888a9dd798cd`.
An initial focused run against the older local native extension passed 755
tests but failed the all-component spline smoke test. The same failure
reproduced in untouched main; rebuilding the baseline extension resolved it.
No tests were skipped or relaxed to obtain the full-suite result.

## Reproduction

```sh
export PYTHONPATH=python FASTSCIPY_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python bench/runs/2026-10-25-wrapper-overhead/probe.py
python -m pytest tests/test_patch_counts.py tests/test_core.py tests/test_fft.py tests/test_spatial_dispatch.py tests/test_sparse_linalg.py -q
python tools/scipy_suite.py --pyargs scipy.fft.tests.test_real_transforms
```
