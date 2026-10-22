# Native Intel optimization fix check

[Focused run 39737974936](https://github.com/michael-L-i/fastscipy/actions/runs/39737974936)
passed on native macOS 15.7.9 x86_64, CPython 3.13.15, NumPy 2.5.4 and
SciPy 1.18.1. **Three serial runs each passed all 98 optimization tests**, with
zero failures, errors or skips (294 executions total).

This is a focused correctness check, not a full release, upstream-suite run,
benchmark, or fresh wheel build. It does not change the archived failed a3
release result. The current test source was
`c29cc5a7be8b4c29055097d360689520682eb6a5`; the wheel remained the original
`0.1.0a3` debug artifact from [release run 39680660563](https://github.com/michael-L-i/fastscipy/actions/runs/39680660563),
source `cf06af752abee2aebffa3b687398f1f76ef94eb8`.

| Repeat | Passed | Process-tree seconds | Peak RSS MiB | Numerical compatibility records |
|---|---:|---:|---:|---:|
| 1 | 98 | 8.419 | 116.461 | 14 |
| 2 | 98 | 25.476 | 106.078 | 12 |
| 3 | 98 | 11.141 | 105.605 | 16 |

Each child had a 512 MiB / 900 second limit. All three returned zero with
complete sampled readable-process-tree RSS, no inaccessible processes,
cleanup errors or resource stops. Native threads were capped at two and
BLAS at one; platform/version overrides were absent. These are correctness
execution/resource observations, not performance measurements.

All 42 exception records explicitly identify
`intel-accelerate-numerical-compatibility` and
`tests.test_optimize._validate_lsq_cost`; none claims observed per-call
reference variation or byte identity. The revised tests permit only the
least-squares dot-product cost to differ, check it independently against a
longdouble residual sum, and preserve every other result field exactly.

The runner checked the wheel RECORD, unchanged installed Python/native bytes,
and byte equality of the installed testing helper with the current source
helper, before and after testing. It did not overlay source modules or fake
package versions. Wheel SHA256:
`728044480a929d6c387ad36660dbb65fdf4cb17c3a727117493e8533b539bbec`.
Native extension SHA256:
`12568446427debe6d92819fc0d3cd303a139b4f9df9d159fffd993729a9b9402`.
The raw summary retains all installed/source file hashes.

[Raw summary](summary.json), [run provenance](run.json),
[original wheel-run provenance](wheel-run.json), all three original logs,
JUnit XML files, resource-monitor sidecars and policy JSONL files are preserved
unchanged. The [hash manifest](raw-file-manifest.json) records their original
bytes and SHA256 values; JUnit hashes were independently checked against the
summary while archiving. No tests, builds or benchmarks ran during this audit.
