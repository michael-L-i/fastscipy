# Bounded temporary memory for threaded pdist

The threaded `pdist` path used one rectangular `cdist` result per worker
block, followed by a boolean mask and an advanced-indexing copy. Its temporary
storage grew with the whole distance problem, even though the final condensed
output was already allocated.

The candidate subdivides each worker block into tiles of at most 131,072
distance entries (1 MiB), or one full remaining row when that row is larger.
It releases each distance tile before computing the next. The mask and selected
values remain temporary allocations, but now scale with the tile, not the
worker block. The unavoidable condensed output still requires O(n²) memory.
This preserves SciPy's metric kernels, full-input statistical parameters,
weighted metrics, output ordering, and the existing shared Python pool.
No new native code or numerical arithmetic is introduced.

## Fresh-process measurements

Baseline: `8ea95b2`. macOS 26.6.2 ARM64, Python 3.13.6, NumPy 2.5.4,
SciPy 1.18.1. Two fastscipy threads; BLAS/OpenMP limited to one. Each row below
uses separate baseline/candidate processes, one warmup and 15 timed calls.
RSS is the process high-water mark through the timed calls, before creating
the SciPy reference. All 12 workload pairs were byte-identical to genuinely
unpatched SciPy, compared through uint64 views without byte-string clones.

| Shape / metric | Median ms, before → after | Peak RSS MiB, before → after |
|---|---:|---:|
| 3000×16 Euclidean | 9.17 → 8.83 | 133.0 → 110.6 |
| 3000×16 cityblock | 7.68 → 7.39 | 133.4 → 106.0 |
| 3000×16 cosine | 15.14 → 14.03 | 132.9 → 102.3 |
| 4000×3 Euclidean | 6.79 → 6.89 | 191.3 → 131.1 |
| 4000×3 cityblock | 7.48 → 6.46 | 195.3 → 127.9 |
| 4000×3 cosine | 10.06 → 10.05 | 191.4 → 133.0 |

Memory is the primary improvement: about 31–35% lower measured process RSS
for the 4000-row cases. Timing differences are modest and workload dependent;
Euclidean at 4000×3 was 1.4% slower in this sample. Smaller 1500-row cases
changed little. These are local development measurements, not cross-platform
or published-wheel claims. The low-dimensional threaded path can still be
slower than single-threaded SciPy; this change does not retune its dispatch.
See `results.json` for every measurement, including the 1500×128 cases.

An initial exploratory script compared large outputs by creating byte-string
copies, inflating its memory high-water mark above the intended 512 MiB
budget. Those measurements are excluded here. The retained probe uses array
views, smaller cases, and fresh processes; retained timed peaks were below
196 MiB.

## Validation

- Spatial and spatial-dispatch project tests: **748 passed**.
- Additional assertion that statistical parameters are shared across tiles:
  **36 forced-tile cases passed** after that test-only addition.
- Paired upstream `scipy.spatial.tests.test_distance`: **470 tests** in both
  modes, **zero changed outcomes** (464 passed, 6 skipped in each).
- Ruff check and formatting passed for changed Python files.
- Spatial benchmark `--check`: **37 cases passed** with byte-identical results
  and required fast paths; raw output is in `benchmark.json`. This existing
  runner uses a disabled-wrapper reference, unlike the true unpatched identity
  reference in the separate probe.

Forced-tile tests cover every supported threaded metric, both one-row and
multirow tiles, exact condensed placement, and a bound on the actual `cdist`
allocation shape. Existing tests cover weights, statistics, noncontiguous
inputs, float32 conversion, NaNs, invalid options and fallback behavior.

## Reproduction

From the repository root with its development environment active:

```sh
export PYTHONPATH=python FASTSCIPY_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
git show 8ea95b2:python/fastscipy/_spatial.py > /tmp/fastscipy-pdist-baseline.py
python bench/runs/2026-10-25-pdist-memory/probe.py --source /tmp/fastscipy-pdist-baseline.py --rows 4000 --dims 3
python bench/runs/2026-10-25-pdist-memory/probe.py --rows 4000 --dims 3
python -m pytest tests/test_spatial.py tests/test_spatial_dispatch.py -q
python tools/scipy_suite.py --pyargs scipy.spatial.tests.test_distance
python bench/run.py spatial --budget 0.03 --check --output-dir /tmp/pdist-benchmark
```

Repeat the two probe commands with `--metric cityblock` or `--metric cosine`,
and with `(rows, dims)` equal to `(1500, 3)`, `(3000, 16)`, or `(1500, 128)`
for the other retained cases. Always use fresh processes for RSS comparisons.
