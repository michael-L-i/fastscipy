# Binned extrema candidate

Python-only candidate on base `d90251ee2dda4f7a2e9c7e531bdfc034205ac835`,
macOS arm64, CPython 3.13, NumPy 2.5, SciPy 1.18.1. Normal activation,
no platform/version overrides or host tuning. Native sources are unchanged;
`native.log` verifies the existing extension against this tree.

SciPy retains validation, bin assignment, outlier removal, and result metadata.
For float64 ndarray values with at least 4096 elements, one or two dimensions,
finite nonzero entries, and no reused bin metadata, the callable reduction seam
uses NumPy `fmin.at`/`fmax.at` instead of sorting. Signed zeros, NaNs, infinities,
complex values, other dtypes, smaller inputs, and reused metadata retain SciPy.
Eligibility is checked at the reduction seam after metadata conversion, which
can run user code. Declined or disabled marker reductions use SciPy's exact
global argsort assignment, preserving NaNs and signed-zero tie behavior even
if conversion mutates values.
This is a portable linear scan; LAPACK and numerical tolerance policies do not apply.

`bench/run.py stats --only binned --check --budget 0.1` uses truly unpatched
SciPy references and validates exact bytes. Local speedups, including eligibility
scans and bin assignment, were 1.5–1.6x at 4096 samples, 3.6–3.8x at 100000
samples (one dimension/one row), 4.1–4.5x at 100000 samples (two dimensions/three
rows), and 2.0–2.1x at one million samples (three dimensions/one row).
Zero and small-input fallback controls were approximately 1.0x. Both one and two
thread settings were measured; the scan itself is serial. These measurements do
not establish performance or validation on other platforms.

Validation: 623 project stats tests passed; paired upstream
`scipy.stats.tests.test_binned_statistic` ran 64 tests on each side with zero
changed outcomes. Tests cover multidimensional bins, empty bins, metadata,
callbacks, input mutation, errors, signed zeros, nonfinite values, and forced
reduction fallback. Ruff passed. Rust was not changed or rebuilt.

Commands ran serially through `tools.platform_suite.run` with two numerical
threads and BLAS/OpenMP/Accelerate threads limited to one. Project/upstream tests
had 512 MiB process-tree limits (sampled peaks 215/152 MiB); benchmarks had a
2048 MiB limit (peak 414 MiB). Raw logs, monitor records, JUnit, manifests, and
benchmark JSON are retained here. No process was stopped by the monitor.

## Metadata mutation correction

`review-fix/` records validation after moving eligibility checks to the reduction
seam and adding the exact global-sort fallback. The original pre-review records
above remain historical. A custom bin converter can mutate values or disable
acceleration, so public dispatch checks alone were insufficient. New regressions
cover NaNs, signed zeros, disabling during conversion, and a converter exception
that must occur once per call. Full stats: 631 passed; a subsequent focused run
with the final exception regression: 30 passed. Paired upstream: 64 per side,
zero changed outcomes. The benchmark check still passes with approximately
1.5–4.5x local speedups and 1.0x fallback controls. Sampled peaks were 213 MiB
for full stats, 144 MiB upstream, 414 MiB benchmarks, and 126 MiB focused tests;
limits and thread budgets were unchanged.
