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

## Independent upstream crash reproduction

[Diagnostic run 39376117965](https://github.com/michael-L-i/fastscipy/actions/runs/39376117965)
reused this `b471` wheel with diagnostic source `daf9acd`. It collected the
actual eight `TestSolve.test_structure_detection` cases with `banded`,
`complex128`, and `transposed=False`, covering both matrix layouts,
overwrite settings, and RHS shapes. All eight crashed with `0xC0000005`
in fresh baseline and patched interpreters. A fresh NumPy-only process also
crashed on the same deterministic 20-by-20 matrix, with SciPy never imported.
The probe completed in 23.23 seconds without a watchdog stop.

The [reproduction archive](upstream-crash-reproduction/) retains unchanged
summary/provenance, actual collected IDs, all small child logs, and the exact
probe source. Its [source index](upstream-crash-reproduction/sources.json)
hashes each raw input and records the installed upstream test source hash.
The historical platform report remains unchanged and incomplete.

The upstream harness now applies visible paired skips only for those eight
proven cases on native Windows ARM64 CPython 3.13.16, NumPy 2.5.4, SciPy
1.18.1, and that exact upstream test source hash. The other 408 cases in the
method remain covered. Both interpreters retain selected IDs and guards in
JSON reports; JUnit records skipped tests. A new upstream run must establish
the results with this policy; it does not complete the historical comparison.
