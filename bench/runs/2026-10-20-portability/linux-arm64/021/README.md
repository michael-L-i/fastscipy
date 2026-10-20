# Linux arm64 development checkpoint `021c329`

[Actions run 39373079547](https://github.com/michael-L-i/fastscipy/actions/runs/39373079547)
built and tested source `021c3293c314c154bf6824c2402eeb0a80b6e5ed`, using
Python 3.13, NumPy 2.5.4, and SciPy 1.18.1 on glibc aarch64.
This is a development checkpoint, not final release validation.
`FASTSCIPY_ANY_PLATFORM=1` enabled the then-unreleased platform family;
the later default platform gate was not exercised.

- [Summary](summary.json): 7,116 passing project tests and all 407 benchmark
  cases across 13 areas passing identity and expected fast-path checks.
- [Raw case reports](cases/): all 13 original per-area JSON files.
- [Build report](build.json) and [wheel/source provenance](provenance.json):
  the development profile used four codegen units with LTO disabled.
- [Upstream manifest](upstream-manifest.json) and
  [compact comparison](upstream-comparison.json): all 19 partition pairs
  completed, with 95,772 common executions, zero changed outcomes, zero
  incomplete pairs, and no failures. The 684 baseline-only collections and
  two patched-only skipped empty parametrizations are expected differences.

The build, summary, case reports, provenance, and manifest were copied byte
for byte from the downloaded artifact and verified by SHA256. The comparison
was derived from all 38 completed raw JUnit files without rerunning tests;
it retains their filenames, sizes, and SHA256 hashes. Bulk logs, XML, and
the wheel remain outside this repository. Older `66f20ae` reports are retained
in the parent directory.
