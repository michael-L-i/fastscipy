# Small `pdist` dispatch

Strict exact NumPy float64 arrays with a canonical metric, no options, and
`rows * rows * dimensions < 2_000_000` can call the same SciPy registered
`pdist` compiled kernel directly. This removes redundant NumPy conversion,
array-namespace dispatch, and metric lookup. It does not change pairwise
arithmetic, approximate results, cache outputs, or add workers.

The direct path covers 14 compiled metrics. Statistics metrics, cosine,
correlation, Jensen-Shannon, callbacks, aliases, other dtypes, array subclasses,
and calls with `out`, weights, or other options retain existing behavior.
Large distances retain the existing threaded/fallback path. `cdist` and
`special` have no implementation changes in this candidate.

## Measurement

`compare.py` times true unpatched SciPy, the previous spatial wrapper from
`f60dca0`, and the candidate. It uses seven cyclic-interleaved batches and
reports median microseconds per call plus all samples in `comparison.json`.
Every case is checked for exact identity. Small cases include repeated local
neighborhood distances, as used in point-cloud processing and clustering.

Mac ARM64, Python 3.13.12, NumPy 2.5.4, SciPy 1.18.1; one BLAS thread. Native
code is the unchanged published a4 extension, with this source checkout on
`PYTHONPATH`. This is local evidence, not a cross-platform performance claim.

| 1-thread Euclidean input | True SciPy us | Previous us | Candidate us | SciPy/candidate | Previous/candidate |
|---|---:|---:|---:|---:|---:|
| 4 x 3 | 3.620 | 4.128 | 0.794 | 4.56x | 5.20x |
| 16 x 3 | 3.772 | 4.115 | 0.914 | 4.13x | 4.50x |
| 32 x 16 | 4.979 | 5.263 | 2.009 | 2.48x | 2.62x |
| 100 x 16 | 15.592 | 16.046 | 12.666 | 1.23x | 1.27x |
| 1500 x 16 control | 2700.615 | 2693.698 | 2699.052 | 1.00x | 1.00x |

The 2-thread setting gives 4.73x, 4.01x, 2.41x, and 1.22x versus true SciPy
for the four small Euclidean cases. This improvement comes from dispatch,
not parallelism. The large 2-thread control gives 1.16x versus SciPy and
1.00x versus the previous wrapper.

Tiny declined calls still pay wrapper costs. The extra candidate gate is
measurable; this change does **not** claim zero regression for every fallback.

| 1-thread 16 x 3 control | True SciPy us | Previous us | Candidate us | Previous/candidate |
|---|---:|---:|---:|---:|
| float32 | 3.873 | 4.208 | 4.354 | 0.97x |
| list | 5.326 | 5.621 | 5.672 | 0.99x |
| weights | 4.336 | 4.716 | 4.791 | 0.98x |
| out | 4.036 | 4.508 | 4.483 | 1.01x |
| cosine | 4.202 | 4.569 | 4.882 | 0.94x |

The largest measured incremental fallback cost was 0.313 us (6.9%) for
1-thread cosine. The 2-thread fallback controls range from 0.97x to 1.02x
relative to the previous wrapper. All raw true-SciPy ratios are retained;
pre-existing wrapper overhead is not attributed to this candidate.

The normal benchmark harness uses a disabled wrapper as its baseline, so its
speedups in `benchmark/spatial.json` are validation results rather than the
true-SciPy headline measurements above.

## Validation

- 561 project tests passed, including 288 new dispatch cases on 1/2 threads.
  These cover every supported metric, zero rows/dimensions, read-only,
  strided/Fortran/unaligned arrays, nonfinite values and signaling NaNs,
  warning identity, independent geometry and condensed ordering, callbacks,
  options/output fallbacks, threshold boundaries, and patch restoration.
- 127 existing cases were deselected by `not 4t and not query and not tree`:
  121 four-thread parametrizations and six one-thread tree/query cases.
- All 21 `pdist` benchmark cases passed identity, expected fast paths, and the
  harness's one-thread regression gate.
- Paired full `scipy.spatial.tests.test_distance`: 470 common tests, zero
  changed outcomes. Each side had 464 passes and six upstream "not applicable"
  skips. The patched side recorded 110 fast-path calls and zero fast-path
  errors. The paired quarantine was inactive and selected no tests.
- Final serial validation took 25.2 seconds and peaked at 338 MiB with a
  512 MiB process-tree watchdog and `SCIPY_AVAILABLE_MEM=256Mib`.
- Ruff check/format and `git diff --check` passed.

The initial **full spatial benchmark did not complete**: the 512 MiB watchdog
stopped it at a sampled 524 MiB during the existing large KDTree query cases.
That is not a passing full-area run. The failed resource record is retained
in `resources/full-benchmark-memory-stop.json`; final validation was narrowed
to changed distance paths, with the complete upstream distance module.

The upstream manifest records the pre-commit base revision because validation
ran on the uncommitted candidate. `source-sha256.txt` identifies the exact
validated implementation/tests/benchmark source. Its installed distribution
metadata says a2 because the reused environment contains that metadata; actual
execution used this source checkout plus the unchanged published a4 native
extension identified in the resource records.

## Reproduce

From the repository root with its `python` directory on `PYTHONPATH`, set
`FASTSCIPY_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 SCIPY_AVAILABLE_MEM=256Mib`, then run:

```sh
python -m pytest -q tests/test_spatial_dispatch.py tests/test_spatial.py -k 'not 4t and not query and not tree'
python bench/runs/2026-10-22-pdist-dispatch/compare.py
python bench/run.py spatial --check --budget 0.01 --only pdist --output-dir /tmp/pdist-benchmark-new
python tools/scipy_suite.py --output-dir /tmp/pdist-upstream-new --pyargs scipy.spatial.tests.test_distance
```

Use an empty output directory for the paired upstream run. The comparison
requires the previous git revision to exist locally. Numerical work was run
serially under the shared resource watchdog; no native code was rebuilt.
