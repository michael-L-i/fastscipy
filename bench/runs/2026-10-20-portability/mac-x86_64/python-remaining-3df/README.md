# Intel remaining checks: failed targeted diagnostic

[Run 39454435225](https://github.com/michael-L-i/fastscipy/actions/runs/39454435225)
paired Python source `3df5f54` with the reused, unchanged `daf9acd` native
wheel. It is a targeted overlay experiment, not a rebuilt wheel, complete
project rerun, or release certification.

The selected 821 tests finished: **817 passed, four failed, zero skipped**.
Two BiCGSTAB tests failed because their symmetry validator treated DIA
stored-zero entries as nonzero values through `.nnz`. Two Pearson tests
failed because their validator returned `numpy.bool_`, while the strict
helper requires `None` or Python `True`. These validator failures remain
recorded as failures; subsequent fixes need independent validation.
[project-failures.json](project-failures.json) retains exact IDs and XML/log
hashes. The five selected test files are recorded in the original summary.

All **95 targeted benchmark cases completed**: special 37 and stats 58,
all byte-identical with every expected fast path. These counts cover only
the subset missing from the earlier interrupted diagnostic, not a new
passing 407-case run. The inner monitor completed without a resource stop
in 580.43 seconds; the outer monitor recorded 583.45 seconds and a peak
of 775.42 MiB. Overall return code was 1 from the test failures.

Original source hashes, overlay imports/runtime environment, workflow
provenance, inner/outer monitors, summaries, case JSON, and small logs are
retained unchanged. The runtime environment records both platform/version
overrides absent and strict mode enabled, with two native threads and one
BLAS thread. Source hashes match the recorded checkout, and its native
source diff against the reused wheel revision is empty. Python source
and reused native provenance remain separate in both summaries.
[audit.json](audit.json) records hashes and the experiment boundary.
