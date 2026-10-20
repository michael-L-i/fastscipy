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

The [eight-wheel CP313 preflight](https://github.com/michael-L-i/fastscipy/actions/runs/39377562130)
finished at source `80dec52`: seven wheel jobs passed and Mac Intel failed
numerical and fast-path validation. Complete-set verification and both
publishers were skipped. This is a failed preflight, not release approval.
The [compact overview](../bench/runs/2026-10-20-portability/preflight80.json)
retains every target's outcome and archive path.

Each job built and repaired its wheel, installed it in an isolated environment,
and passed installed-package/native smoke checks with default activation and
no version/platform overrides. Project and benchmark outcomes were:

| Official CP313 wheel checkpoint | Project tests | Benchmark identity |
|---|---|---|
| [manylinux aarch64](../bench/runs/2026-10-20-portability/linux-arm64/release80/README.md) | 7,221 passed | 407/407 |
| [manylinux x86_64](../bench/runs/2026-10-20-portability/linux-x86_64/release80/README.md) | 7,221 passed | 407/407 |
| [musllinux aarch64](../bench/runs/2026-10-20-portability/musllinux-aarch64/release80/README.md) | 7,221 passed | 407/407 |
| [musllinux x86_64](../bench/runs/2026-10-20-portability/musllinux-x86_64/release80/README.md) | 7,221 passed | 407/407 |
| [macOS arm64](../bench/runs/2026-10-20-portability/mac-arm64/release80/README.md) | 7,221 passed | 407/407 |
| [macOS x86_64 — failed](../bench/runs/2026-10-20-portability/mac-x86_64/release80/README.md) | 7,189 passed; 32 failed | 406/407 |
| [Windows AMD64](../bench/runs/2026-10-20-portability/windows-x86_64/release80/README.md) | 7,217 passed; 4 skipped | 407/407 |
| [Windows ARM64](../bench/runs/2026-10-20-portability/windows-arm64/release80/README.md) | 7,217 passed; 4 skipped | 407/407 |

All expected benchmark fast paths ran. Mac Intel's project failures comprised
18 ndimage interpolation fast-path misses, two dense `lsq_linear` identity
failures, eight sparse iterative-solver identity failures, and four
`pearsonr`/`linregress` identity failures. Its BFGS numerical-gradient benchmark
also differed from SciPy. The [failure record](../bench/runs/2026-10-20-portability/mac-x86_64/release80/project-failures.json)
retains identifiers, excerpts, and full-trace hashes. Later candidate fixes,
including the Mac selector change, require their own validation.

All eight upstream comparisons completed 19 partition pairs with 95,772
common executions, no changed outcomes, no incomplete pairs, and no failures.
Windows ARM visibly skipped the same eight known NumPy-reference crash cases
on both sides; those cases remain coverage exceptions. Every checkpoint
retains original project/resource/case JSON, upstream manifests/completion
records, and a derived comparison with all 38 raw XML hashes. Passing upstream
comparisons do not override Mac Intel's failed project/benchmark results.

These source-specific CP313 wheel checks do not validate later source changes,
other Python versions, or the final 32-wheel publication gate. The
[archive index](../bench/runs/2026-10-20-portability/README.md) separately retains
earlier development checkpoints, including override-enabled runs and the
[failed native Intel build](../bench/runs/2026-10-20-portability/mac-x86_64/daf9/README.md)
from [run 39376214465](https://github.com/michael-L-i/fastscipy/actions/runs/39376214465).

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
