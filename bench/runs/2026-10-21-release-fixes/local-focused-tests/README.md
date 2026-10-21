# Focused a3 fix checks

866 tests passed in 8.77 seconds, with 198 MiB sampled process-tree peak RSS
under a 512 MiB budget. No benchmarks were run.

This check used the installed CP313 Mac ARM native extension from source
`1f48`, with the current `fastscipy.testing` module loaded from the fix
checkout. It exercised monitor, numerical compatibility, reference variation,
sparse linear algebra, statistics, benchmark-policy, and optimize tests,
including stats fix `53902b5`. Rust kernels were unchanged. This is focused
regression evidence, not certification of a newly built a3 wheel or Intel
behavior. The fresh release workflow must validate each installed wheel.

Source version guards, Ruff checks and formatting, workflow YAML parsing,
and whitespace checks also passed before tagging.
