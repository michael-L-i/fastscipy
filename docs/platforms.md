# Platform validation

The published `0.1.0a1` release supports macOS 14+ on Apple Silicon, SciPy 1.18,
and Python 3.12–3.14. Broader support is being evaluated for `0.1.0a2`.
The candidate release has not completed validation or been published.

| Platform | Published `0.1.0a1` | `0.1.0a2` candidate |
|---|---|---|
| macOS arm64 | Experimental support | Revalidation pending |
| macOS x86_64 | Unsupported | Validation pending |
| Linux aarch64 | Unsupported | Validation pending |
| Linux x86_64 | Unsupported | Validation pending |
| Windows arm64 | Unsupported | Validation pending |
| Windows x86_64 | Unsupported | Validation pending |

Linux wheel builds also distinguish glibc and musl. A result from a glibc
runner does not validate a musl wheel, another Python version, or a custom
SciPy build. The candidate release matrix targets eight platform variants
and Python 3.12–3.15, for 32 planned wheels; that matrix is broader than the
Python 3.13 development checks below.

## Evidence

The [wheel preflight](https://github.com/michael-L-i/fastscipy/actions/runs/39377562130)
checks repaired, installed Python 3.13 wheels on all eight target variants.
The [Intel Mac and Windows x86 development run](https://github.com/michael-L-i/fastscipy/actions/runs/39376214465)
checks the latest native arithmetic fixes. Completed jobs retain raw logs,
JUnit results, per-case timings, and resource reports. Inspect each source
revision and artifact; a workflow still running is not a passing result.

The following are development checkpoints at different source revisions,
not final release validation:

- macOS arm64: 7,048 project tests passed on each of
  [Python 3.13](../bench/runs/2026-10-20-portability/mac-arm64/tests-py313.json)
  and [Python 3.15](../bench/runs/2026-10-20-portability/mac-arm64/tests-py315.json).
  A separate [paired upstream comparison](../bench/runs/2026-10-20-portability/mac-arm64/upstream-corrected-comparison.json)
  found no changed outcomes across 95,772 common test executions. Three
  failures occurred in both the unpatched and patched runs; the report
  retains those failures, expected collection differences, and the corrected
  comparison's source revision.
- macOS arm64, newer revision `b471`: all 7,116 project tests passed on
  each of [Python 3.13](../bench/runs/2026-10-20-portability/mac-arm64/tests-b471-py313.json)
  and [Python 3.15](../bench/runs/2026-10-20-portability/mac-arm64/tests-b471-py315.json).
  All 407 benchmark cases across 13 areas matched SciPy and met their expected
  fast-path checks in the [benchmark report](../bench/runs/2026-10-20-portability/mac-arm64/b471/summary.json).
  The [paired upstream comparison](../bench/runs/2026-10-20-portability/mac-arm64/b471/upstream-comparison.json)
  found no changed outcomes across 95,772 common executions, including the
  same three baseline failures. Its manifest retains the later harness
  checkout separately from the `b471` benchmark revision.
- Linux aarch64 (glibc), revision `80dec52`: the official repaired
  `manylinux_2_28` CP313 wheel passed installed-wheel checks with default
  activation, all 7,221 project tests, and all 407 benchmark cases across
  13 areas. All 19 paired upstream partitions completed: 95,772 common
  executions, no changed outcomes, and no failures. The
  [checkpoint archive](../bench/runs/2026-10-20-portability/linux-arm64/release80/README.md)
  retains raw reports, resource scope, and wheel provenance from the
  [eight-wheel preflight](https://github.com/michael-L-i/fastscipy/actions/runs/39377562130).
  This validates that wheel checkpoint, not later source changes or the full
  32-wheel release. Earlier development reports, including the
  [override-enabled `021c329` run](../bench/runs/2026-10-20-portability/linux-arm64/021/README.md),
  remain archived separately.
- Linux x86_64 (glibc), revision `675bd52`: all 7,131 project tests and
  407 benchmark cases across 13 areas passed. All 19 paired upstream
  partitions completed: 95,772 common executions, no changed outcomes,
  no unexpected collection differences, and no failures. The
  [checkpoint archive](../bench/runs/2026-10-20-portability/linux-x86_64/675/README.md)
  retains unchanged reports and provenance from
  [run 39374986938](https://github.com/michael-L-i/fastscipy/actions/runs/39374986938),
  plus a compact comparison with all 38 raw XML hashes. ndimage achieved
  8.59× / 14.91× geometric-mean speedups on one/two threads; hosted-VM
  differences prevent a controlled incremental comparison with the older
  [4b checkpoint](../bench/runs/2026-10-20-portability/linux-x86_64/4b/README.md),
  whose four initialization failures are now fixed. The tested source includes
  candidate platform-gate change `277fc71`, but the workflow still set
  `FASTSCIPY_ANY_PLATFORM=1`; default activation and final release-wheel
  validation remain separate pending checks.
- Windows arm64, revision `b4710e6`: 7,112 project tests passed and four
  were skipped; all 407 benchmark cases matched SciPy and met their expected
  fast-path checks. The [checkpoint archive](../bench/runs/2026-10-20-portability/windows-arm64/b471/README.md)
  retains unchanged reports and provenance from
  [run 39372506350](https://github.com/michael-L-i/fastscipy/actions/runs/39372506350).
  Upstream comparison completed 12,324 common executions across ten package
  partitions without changed outcomes, then unpatched NumPy crashed inside
  `linalg.solve` during SciPy's `TestSolve.test_structure_detection` reference
  calculation. Patched linalg and the remaining eight partitions were not
  run. The archive records this baseline crash and partial coverage; a full
  upstream comparison is still pending.

The Windows ARM [bounded reproduction](../bench/runs/2026-10-20-portability/windows-arm64/b471/upstream-crash-reproduction/sources.json)
independently reproduces eight NumPy reference-solve crashes in baseline and
patched children; NumPy also crashes without importing SciPy. The upstream
harness visibly skips only those eight `TestSolve.test_structure_detection`
parameters with banded complex128 matrices and `transposed=False`, on native
Windows ARM64 CPython 3.13.16 with NumPy 2.5.4, SciPy 1.18.1, and the exact
recorded upstream test-source SHA256. The other 408 method cases remain
covered. Both sides record the reason, scope, and selected IDs in per-run
JSON and retain skips in JUnit. This is an explicit coverage exception, not
a passing result for those tests; other crashes still fail the comparison.
The [native postcheck](../bench/runs/2026-10-20-portability/windows-arm64/b471/quarantine-validation-b90/sources.json)
confirmed 416 cases on each side: 408 passed, eight visibly skipped, and
zero changed outcomes, using a reused wheel rather than a new release build.

These checkpoints do not certify the final `0.1.0a2` wheel. The latest source
must complete its own project, benchmark, and paired upstream checks.
Reports retain their original revisions and metadata; a passing earlier
checkpoint does not validate subsequent native changes or other targets.

The repository also retains earlier, hardware-specific Mac measurements:
[published raw cases](../bench/results/) and
[additional workload reports](../bench/runs/2026-10-19-sweep/summary.json).
Those archives are separate from the platform run and do not establish
Linux or Windows performance.

## Measured Linux ARM performance

Geometric-mean speedups from the repaired `manylinux_2_28` CP313 wheel
checkpoint above, on a GitHub-hosted ARM runner with NumPy 2.5.4 and
SciPy 1.18.1. Each comparison uses the same machine and inputs; BLAS is
limited to one thread. Counts exclude nine declined controls from the
407-case suite. These measurements do not describe other platforms or
all calls in each area.

| Area | Accelerated cases | 1 thread | 2 threads |
|---|---:|---:|---:|
| linalg | 49 | 3.94× | 4.03× |
| ndimage | 43 | 7.72× | 13.15× |
| interpolate | 27 | 3.86× | 4.64× |
| csgraph | 16 | 3.75× | 4.42× |
| signal | 41 | 3.61× | 4.24× |
| sparse | 27 | 3.76× | 4.43× |
| sparse linalg | 20 | 1.31× | 1.31× |
| stats | 58 | 6.14× | 6.50× |
| special | 35 | 1.00× | 1.85× |
| spatial | 21 | 1.00× | 1.67× |
| integrate | 35 | 1.75× | 1.76× |
| optimize | 14 | 1.81× | 1.81× |
| cluster | 12 | 1.73× | 1.74× |

Special and spatial gain through threading; one-thread results are about
1×. [Raw cases and machine metadata](../bench/runs/2026-10-20-portability/linux-arm64/release80/summary.json)
retain individual timings, controls, identity checks, and expected paths.

## Why each build is checked

SciPy wheels can differ in compiler contraction, floating-point libraries,
and BLAS arithmetic. Fastscipy reads the installed signal kernels' contraction
behavior and checks compatible matrix-vector paths against NumPy. Native
BLAS and LAPACK capsule calls require the expected C signatures; incompatible
integer ABIs are declined before calling a function pointer.

These checks preserve acceleration for compatible paths. They do not replace
comparison against the installed SciPy wheel. A platform label alone cannot
establish byte identity for a custom build or new dependency version.

## Reproduce a platform report

Use a complete checkout and an environment containing the built package,
SciPy, NumPy, pytest, Hypothesis, and psutil. The development matrix uses
Python 3.13, NumPy 2.5.4, and SciPy 1.18.1. Keep the build profile and
dependency versions with the resulting report.

```sh
export FASTSCIPY_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python tools/platform_suite.py --output artifacts/platform-local --max-rss-mb 4096 --upstream
```

The output directory must be empty. The runner executes project tests
serially and benchmarks each of the 13 areas in its own process. Optional
upstream comparisons run unpatched and patched SciPy in module partitions.
Process-tree RSS and elapsed-time budgets stop runs that exceed their
configured limits. RSS is sampled and summed across processes, so shared
pages can be counted more than once.

Linux release builds run inside Docker containers with hard limits of 6 GiB
and two CPUs. Their outer `build.json` measures the host cibuildwheel process
tree, which excludes compiler processes owned by the Docker daemon. Its
`rss_scope` records this boundary; container peak memory remains unknown.
The hard limit is not a measured peak. Reports collected inside the container
for wheel tests measure those test process trees separately.

For upstream tests, `SCIPY_AVAILABLE_MEM` defaults to half the RSS budget
(2 GiB with the default limit). SciPy skips its own tests that require more
memory; both runs use the same budget, and the raw JUnit files retain skips.

`summary.json` records the package version, benchmark revision, dependencies,
NumPy configuration, thread budgets, child return codes, and sampled peak
RSS. Each area's report lists numerical mismatches and missing expected fast
paths. It reports geometric-mean speedups both across all cases and with
control cases excluded; the per-case JSON retains individual timings.

Speedups compare unpatched and patched calls on the same machine and inputs,
at one thread and the configured thread budget. They describe those
workloads, not every call in a SciPy module. Compare reports only with their
hardware, dependencies, build profile, and thread counts in view.
