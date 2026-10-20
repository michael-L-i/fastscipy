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

The [six-platform Actions run](https://github.com/michael-L-i/fastscipy/actions/runs/39362357649)
records the ongoing development checks. Its completed jobs upload raw logs,
JUnit results, per-case benchmark JSON, and a `summary.json` report under
`platform-<target>-py313` artifacts. Inspect the job revision and artifacts
before treating a result as evidence for a release.

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
- Linux aarch64 (glibc), revision `66f20ae`: 7,048 project tests passed;
  all 407 benchmark cases across 13 areas matched SciPy and met their
  expected fast-path checks. The archive contains the
  [summary](../bench/runs/2026-10-20-portability/linux-arm64/summary.json),
  [13 per-area case reports](../bench/runs/2026-10-20-portability/linux-arm64/cases/),
  and [build resource report](../bench/runs/2026-10-20-portability/linux-arm64/build.json).
- Linux x86_64: all 407 benchmark cases matched SciPy and met their
  expected fast-path checks. Four project tests exposed NaN-bit differences
  in uniform filtering; the fix awaits fresh native CI. The initial
  job also contained a stale Windows-platform simulation test, since fixed.
- Windows arm64: all 407 benchmark outputs matched SciPy, but five missed
  their expected fast paths. Project tests had 120 failures in arithmetic
  calibration and fast-path checks. Fixes await the
  [fresh native run](https://github.com/michael-L-i/fastscipy/actions/runs/39369748578);
  a [diagnostic run](https://github.com/michael-L-i/fastscipy/actions/runs/39369369748)
  confirmed the proposed correlation arithmetic against the installed SciPy
  wheel in 85 of 85 software probe cases.

These checkpoints do not certify the final `0.1.0a2` wheel. The latest source
must complete its own project, benchmark, and paired upstream checks.
Reports retain their original revisions and metadata; a passing earlier
checkpoint does not validate subsequent native changes or other targets.

The repository also retains earlier, hardware-specific Mac measurements:
[published raw cases](../bench/results/) and
[additional workload reports](../bench/runs/2026-10-19-sweep/summary.json).
Those archives are separate from the platform run and do not establish
Linux or Windows performance.

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
