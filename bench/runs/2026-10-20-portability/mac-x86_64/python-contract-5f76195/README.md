# Intel tests-only contract diagnostic: failed

[Run 39462230112](https://github.com/michael-L-i/fastscipy/actions/runs/39462230112)
paired Python source `5f76195` with the unchanged reused `daf9acd` native wheel.
The five selected test files completed: **828 passed, one failed, zero skipped**.
This is selected-test overlay evidence, not a complete project rerun or a
rebuilt release-wheel validation. No benchmarks or upstream comparisons ran.

The remaining failure was `test_function_randomized[4t-linregress]`: the
validator rejected slope -19.733333333333334 versus -19.2 and intercept
1.84375934276339 versus 1.817006339803941. The reported correlation was near
-2e-16 and stderr was 1.8357904985830864e16 on both sides. These diagnostics
are preserved as a failure, without inferring acceptance or a production fix.
Both previously failing BiCGSTAB parameters and both Pearson parameters
passed in this run. [project-failures.json](project-failures.json) retains
the complete failure message/trace and raw XML hash; the original test log
is archived unchanged.

The inner monitor recorded 85.27 seconds and 182.31 MiB peak RSS; the outer
monitor recorded 87.32 seconds and 217.22 MiB. Neither hit a resource limit.
Both returned 1 because the selected tests failed. Original source/import
provenance, summaries, monitors, reference-variation records, and logs are
retained unchanged. Source hashes match the recorded checkout, its native
source diff against the reused wheel revision is empty, and actual overlay
imports are recorded. The runtime record shows both platform/version
overrides absent, strict mode enabled, and two native/one BLAS threads.
[audit.json](audit.json) records hashes and the limited experiment scope.
