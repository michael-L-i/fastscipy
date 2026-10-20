# Portability development checkpoints

These reports describe intermediate `0.1.0a2` builds, not a validated final
release. Each JSON file was copied byte for byte from its original report;
revisions, dependency versions, resource measurements, and results were not
rewritten to match a newer checkout.

- `linux-arm64/`: glibc aarch64 revision `66f20ae`; build resource report,
  platform summary, and all 13 raw per-area benchmark reports. The summary
  records 7,048 passing project tests and 407 benchmark cases with no
  numerical mismatches or missing expected fast paths.
- `mac-arm64/tests-py313.json` and `tests-py315.json`: project-test summaries
  for the ABI-guard builds, with 7,048 passing tests on each Python version.
  These summaries contain no benchmark results.
- `mac-arm64/tests-b471-py313.json` and `tests-b471-py315.json`: newer
  revision `b4710e6` project-test summaries, with 7,116 passing tests on each
  Python version. These summaries contain no benchmark or upstream results.
- `mac-arm64/upstream-corrected-comparison.json`: comparison of 38 completed
  raw JUnit records across 19 upstream module partitions, re-evaluated with
  the corrected special-ufunc classifier. All 95,772 common executions have
  unchanged outcomes, including three baseline failures. The report retains
  both the tested revision and comparator revision, plus expected collection
  differences. Bulk JUnit XML and logs are not included here.

See [platform status](../../../docs/platforms.md) for the pending checks and
Actions links. Earlier reports do not establish the behavior of later builds.
