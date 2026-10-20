# MacIntel development checkpoint `daf9acd`: failed project validation

[Actions run 39376214465](https://github.com/michael-L-i/fastscipy/actions/runs/39376214465)
built source `daf9acd697e89612d1bede34d569522ea35fd7d7` on macOS x86_64
with CPython 3.13, NumPy 2.5.4, and SciPy 1.18.1. This checkpoint failed
validation and does not certify subsequent fixes or release wheels.

- [Summary](summary.json): 7,166 project tests passed and 36 failed.
  The compact [failure list](project-failures.json) retains all identifiers
  and the raw project XML hash: 18 ndimage interpolation fast-path misses,
  14 sparse iterative-solver result/history mismatches, and four stats
  `pearsonr`/`linregress` result mismatches.
- [Raw benchmark reports](cases/): 406 of 407 cases matched SciPy;
  `minimize BFGS rosen n=10 (numerical gradient)` was nonidentical. No
  expected fast paths were missing. All 43 ndimage cases matched, with
  geometric-mean speedups of 8.27× / 12.70× on one/two numerical threads
  and one BLAS thread.
- [Build resources](build.json) and [provenance](provenance.json): a fresh
  release-profile wheel with eight codegen units and thin LTO. Compilation
  took 94 minutes 55 seconds and peaked at 4.17 GiB RSS. The largest measured
  runtime RSS was 1.58 GiB in signal benchmarks.
- [Upstream manifest](upstream-manifest.json) and
  [compact comparison](upstream-comparison.json): all 19 partition pairs
  completed, with 95,772 common executions, zero changed outcomes, zero
  unexpected collection differences, zero incomplete pairs, and no failures
  on either side. The 684 baseline-only special-ufunc parametrizations and
  two patched-only skipped empty parametrizations are expected. Upstream
  comparison took 51 minutes 51 seconds and peaked at 1.10 GiB RSS.

All 17 original JSON reports were copied byte for byte and verified against
the downloaded artifact. Compact comparisons were derived without rerunning
tests and retain filenames, sizes, and SHA256 hashes of all 38 upstream XML
files and the project XML. Bulk XML, logs, and the wheel remain outside this
repository. The workflow used `FASTSCIPY_ANY_PLATFORM=1`.
