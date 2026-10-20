# Intel Python-source diagnostic: failed and partial

[Run 39448533018](https://github.com/michael-L-i/fastscipy/actions/runs/39448533018)
paired current Python source `1f48cd6` with the unchanged installed `daf9acd`
native wheel. This is an overlay experiment, not release-wheel certification.

The full project suite finished: **7,273 passed, two failed, zero skipped**.
Both failures were `test_solver_results_and_iteration_history` BiCGSTAB
parameters `[1-100-bicgstab]` and `[2-100-bicgstab]`: reference history length
41 versus actual 42. [project-failures.json](project-failures.json) retains
exact IDs, messages, and original XML/log hashes.

The outer monitor then killed the diagnostic at 1,200.287 seconds, with peak
process-tree RSS 1,626.3 MiB. There is **no completed inner monitor or final
summary**. Eleven complete area JSON files contain 312 cases: 311 byte-identical
and one BFGS case explicitly marked `reference_variance`; all expected fast
paths ran. The original incremental summary covers only 288 cases in ten areas;
its completed spatial run has another 24 cases retained separately. Special
has only a partial text log and stats never started: 95 cases remain incomplete.
This is not a passing 407-case result.

All 32 Python package file hashes match the recorded checkout, whose Python
source was clean. Its `src`, `Cargo.toml`, and `Cargo.lock` diff against the
reused wheel's source is empty. Native and wheel hashes independently match
the original downloaded artifact. The import record verifies temporary overlay
Python and native paths. The executed committed probe removes both platform
and version overrides before spawning children; runtime environment absence
was not separately dumped. Recorded budgets are two native threads and one
BLAS thread. Default source guards were used through that probe path; this
experiment does not validate a newly built or repaired release wheel.

Original source/import/workflow provenance, outer diagnostic results, incremental
summary, completed case JSON, suite log, and reference-variation events are
retained unchanged. [audit.json](audit.json) records hashes and evidence limits.
Bulk project XML and partial logs remain in the external Actions artifact.
