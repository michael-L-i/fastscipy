# Linux x86_64 development checkpoint `675bd52`

[Actions run 39374986938](https://github.com/michael-L-i/fastscipy/actions/runs/39374986938)
built and tested source `675bd521f890f022768ad1d7beae91d89f3a1b89`, using
CPython 3.13.16, NumPy 2.5.4, and SciPy 1.18.1 on glibc x86_64.
This is a development checkpoint, not final release-wheel validation.
The source includes candidate platform-gate change `277fc71`, but the
workflow still set `FASTSCIPY_ANY_PLATFORM=1`; this run does not independently
verify activation without that override.

- [Summary](summary.json): all 7,131 project tests passed. All 407 benchmark
  cases across 13 areas passed identity and expected fast-path checks.
- [Raw case reports](cases/): all 13 unchanged per-area JSON files.
- [Build report](build.json) and [wheel/source provenance](provenance.json):
  the development profile used four codegen units with LTO disabled.
  Compilation took 38 minutes 41 seconds and peaked at 4.39 GiB RSS.
- [Upstream manifest](upstream-manifest.json) and
  [compact comparison](upstream-comparison.json): all 19 partition pairs
  completed, with 95,772 common executions, zero changed outcomes, zero
  unexpected collection differences, zero incomplete pairs, and no failures.
  The 684 baseline-only special-ufunc parametrizations and two patched-only
  skipped empty parametrizations are expected collection differences.
  Upstream validation took 15 minutes 54 seconds and peaked at 1.14 GiB RSS.

The 43 ndimage cases achieved geometric-mean speedups of 8.59× on one
thread and 14.91× on two threads, with BLAS limited to one thread.
Representative float-filter speedups against SciPy were:

| Case | One thread | Two threads |
|---|---:|---:|
| Gaussian f64, 2048², sigma 2 | 6.87× | 13.10× |
| Gaussian f32, 2048², sigma 2 | 6.79× | 12.67× |
| Gaussian f32, 256³, sigma 1 | 7.26× | 13.25× |
| Uniform f64, 2048², size 5 | 10.04× | 16.20× |
| Sobel f32, 2048² | 12.59× | 23.55× |

These results confirm acceleration in this hosted run. They are not a
controlled incremental comparison with the older [4b checkpoint](../4b/README.md):
the Azure VM and SciPy reference timings differ between runs. That older
checkpoint retains its four initialization failures; all initialization
regressions pass here.

The build, summary, case reports, provenance, and upstream manifest were copied
byte for byte from the downloaded artifact and verified against their originals.
The comparison was derived from all 38 completed raw JUnit files without
rerunning tests; it retains their filenames, sizes, and SHA256 hashes. Bulk
logs, XML, and the development wheel remain outside this repository.
