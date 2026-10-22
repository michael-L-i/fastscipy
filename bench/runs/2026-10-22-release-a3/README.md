# Failed a3 final release checkpoint

[Release run 39680660563](https://github.com/michael-L-i/fastscipy/actions/runs/39680660563),
tag `v0.1.0a3`, source `cf06af752abee2aebffa3b687398f1f76ef94eb8`,
finished **failed: 31 of 32 wheel jobs passed**. Only Mac Intel CP313 failed.
The source-distribution guard passed; complete-set verification, TestPyPI,
and PyPI publishers were skipped. **This run published nothing.**
[Job conclusions](final-tag-attempt/jobs.json) and the byte-preserved compressed
API response retain the full run provenance. Earlier archives are unchanged.

Mac Intel CP313 project tests reported **7,346 passed, one failed, zero skipped**.
`test_lsq_linear_dense_unbounded[1-shape4]` failed byte comparison: the printed
costs were 19.49568140154564 and 19.495681401545635. The complete original
message/trace is retained in [the failure record](intel-cp313/project-failure.json),
with compressed original test XML/logs. This remains a failed test; no later
fix or numerical-compatibility acceptance is inferred.

All 407 benchmark cases completed with expected fast paths: 406 were
byte-identical, and the BFGS numerical-gradient case retained its explicit
reference-variation exception. The summary separately records five observed
reference-variation records and 57 numerical-compatibility records; numerical
compatibility is not relabeled as observed reference variation or byte identity.
All 13 original case JSON files and the original summary are retained.

The paired upstream run completed all 19 pairs and 38 sides, with **95,772
common executions and zero changed outcomes under the existing classifier**.
There were no failures, incomplete pairs, or unexpected collection differences.
Special's 684 baseline-only ufunc cases and two patched `[NOTSET]` skips were
the existing expected differences. No Windows ARM quarantine was active.

One **raw** outcome did differ: SciPy optimize
`tests.test_linprog.TestLinprogIPSparse::test_bug_6690` passed in baseline and
xfail'ed when patched, with reason `Fails with ATLAS, see gh-7877`. The existing
harness explicitly counts expected failures as passed; this explains upstream
exit zero and the one-count skip/xfail difference. This is not a claim that
all raw outcomes were identical. [The comparison](intel-cp313/upstream-comparison.json)
retains this transition, per-partition raw counts, and all 38 XML hashes.
No additional blocker was found under that unchanged upstream policy.

The outer cibuildwheel process-tree report recorded 12667.144 seconds
and 4368.039 MiB sampled peak RSS, with complete readable
RSS coverage, no inaccessible PIDs, no resource stop, and return code 1. This
is the whole build/test command's sampled peak, not a separate compiler peak.
The project test child peaked at 376.406 MiB; upstream
peaked at 1117.602 MiB. Raw resource reports and
monitor sidecars are retained. The [raw-file manifest](raw-file-manifest.json)
records original and archived hashes; gzip records decompress byte for byte.
No tests, builds, or benchmarks were executed to prepare this archive.
