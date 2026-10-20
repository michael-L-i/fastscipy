# Windows ARM64 checkpoint: `b4710e6`

[Run 39372506350](https://github.com/michael-L-i/fastscipy/actions/runs/39372506350)
built revision `b4710e6cadd9fa6b4701ec3b8363c521f30f61aa` on native Windows
ARM64 with CPython 3.13.16, NumPy 2.5.4, and SciPy 1.18.1. The unchanged
[provenance](provenance.json), [build report](build.json), and
[platform summary](summary.json) retain the original run metadata.

Project tests reported **7,112 passed and four skipped**, with no failures.
All **407 benchmark cases across 13 areas** matched SciPy and met their
expected fast-path checks. All 346 interpolation test instances passed;
interpolation's 31 benchmark cases also passed. The unchanged
[per-area reports](cases/) retain individual timings and controls.

The paired upstream comparison is **incomplete**. The first ten package
partitions completed 12,324 common executions with no changed outcomes,
unexpected missing tests, or extra tests. The next unpatched partition,
`scipy.linalg`, terminated with Windows access violation `0xC0000005` in
NumPy's `linalg.solve`, called by SciPy's
`TestSolve.test_structure_detection` reference calculation. The exact
parameter ID was not printed by the original quiet runner. Patched linalg
and the subsequent eight package partitions were not run.

The [partial comparison](upstream-partial-comparison.json) includes all 20
completed JUnit filenames, sizes, and SHA256 hashes, plus hashes of the raw
manifest and upstream log. The [crash record](upstream-crash.json) retains
the reported frames, exit code, raw-log hash, and resource report. The
[unchanged upstream manifest](upstream-manifest.json) records versions,
revision, memory budget, and all planned partitions. Bulk XML and logs
remain in the Actions artifact rather than this compact archive. The actual
installed Windows test source was not uploaded, so no source-code hash is
inferred from another platform.

The upstream process tree peaked at 1,570,549,760 bytes and was not stopped
by the memory or time watchdog. No test quarantine was applied. This
checkpoint confirms project and benchmark results; it does not certify a
complete upstream comparison or the final release matrix.
