# Windows AMD64 checkpoint `daf9acd`

[Actions run 39376214465](https://github.com/michael-L-i/fastscipy/actions/runs/39376214465)
built source `daf9acd697e89612d1bede34d569522ea35fd7d7` on Windows Server
2025 AMD64 with CPython 3.13.16, NumPy 2.5.4, and SciPy 1.18.1.

- [Project and benchmark summary](summary.json): 7,198 project tests passed,
  with four fork-dependent tests skipped. All 407 benchmark cases across
  13 areas matched SciPy and met expected fast-path checks.
- [Raw case reports](cases/): ndimage's 43 cases achieved geometric-mean
  speedups of 7.23× / 11.58× with one/two numerical threads and one BLAS thread.
- [Build resources](build.json) and [provenance](provenance.json): a fresh
  release-profile wheel, eight codegen units and thin LTO. Compilation took
  91 minutes 26 seconds and peaked at 3.84 GiB RSS.
- [Upstream manifest](upstream-manifest.json) and
  [compact comparison](upstream-comparison.json): all 19 partition pairs
  completed, with 95,772 common executions, no changed outcomes, no unexpected
  collection differences, no incomplete pairs, and no failures. The 684
  baseline-only special-ufunc parametrizations and two patched-only skipped
  empty parametrizations are expected. Upstream validation took 19 minutes
  18 seconds and peaked at 1.78 GiB RSS.

All 17 original JSON reports were copied byte for byte and verified against
the downloaded artifact. The compact comparison was derived without rerunning
tests and records filenames, sizes, and SHA256 hashes of all 38 completed raw
JUnit files. Bulk XML, logs, and the wheel remain outside this repository.

This checkpoint covers one Python version and one dependency pair. Its
release-profile build does not certify every target in the release matrix;
the workflow also used `FASTSCIPY_ANY_PLATFORM=1` rather than independently
testing default platform activation.
