# Portability development checkpoints

These reports describe intermediate `0.1.0a2` builds, not a validated final
release. Each JSON file was copied byte for byte from its original report;
revisions, dependency versions, resource measurements, and results were not
rewritten to match a newer checkout.

- `linux-arm64/`: glibc aarch64 revision `66f20ae`; build resource report,
  platform summary, and all 13 raw per-area benchmark reports. The summary
  records 7,048 passing project tests and 407 benchmark cases with no
  numerical mismatches or missing expected fast paths.
- `linux-arm64/021/`: newer glibc aarch64 revision `021c329`; 7,116 passing
  project tests, all 407 benchmark cases passing, and all 19 upstream pairs
  complete with no changed outcomes across 95,772 common executions and no
  failures. It retains original build/source provenance, case reports, and a
  compact upstream comparison with all 38 raw XML hashes. This development
  checkpoint used the platform-gate override; later default-gate and release
  wheel validation remain separate.
- `linux-x86_64/4b/`: glibc x86_64 revision `4b1342d`; unchanged build and
  provenance reports, platform summary, and 13 raw per-area reports. Project
  tests reported 7,060 passed and four NaN initialization failures; all 407
  benchmark cases passed identity and expected fast-path checks. Its README
  compares float-filter timings with archived raw revision `5914b77` results
  and retains that intermediate result separately from newer validation.
- `linux-x86_64/675/`: newer glibc x86_64 revision `675bd52`; all 7,131
  project tests and 407 benchmark cases passed. All 19 upstream pairs
  completed with no changed outcomes across 95,772 common executions and no
  failures. Original reports and provenance are retained, along with a
  compact comparison containing all 38 raw XML hashes. ndimage speedups
  were 8.59× / 14.91× on one/two threads. The source includes the candidate
  platform gate, but the workflow used its override; default activation and
  final release-wheel validation remain separate.
- `mac-arm64/tests-py313.json` and `tests-py315.json`: project-test summaries
  for the ABI-guard builds, with 7,048 passing tests on each Python version.
  These summaries contain no benchmark results.
- `mac-arm64/tests-b471-py313.json` and `tests-b471-py315.json`: newer
  revision `b4710e6` project-test summaries, with 7,116 passing tests on each
  Python version. These summaries contain no benchmark or upstream results.
- `mac-arm64/b471/`: unchanged platform summary, 13 raw benchmark case
  reports, and upstream manifest for the `b4710e6` wheel. All 407 benchmark
  cases pass identity and expected fast-path checks. The compact upstream
  comparison was derived from all 38 completed JUnit files: 95,772 common
  executions, no changed outcomes, and the same three baseline failures.
  It includes raw XML filenames, sizes, and SHA256 hashes without archiving
  bulk XML. The original manifest records harness checkout `2d55def`, while
  the platform summary records benchmark revision `b4710e6`; both are retained.
- `mac-arm64/upstream-corrected-comparison.json`: comparison of 38 completed
  raw JUnit records across 19 upstream module partitions, re-evaluated with
  the corrected special-ufunc classifier. All 95,772 common executions have
  unchanged outcomes, including three baseline failures. The report retains
  both the tested revision and comparator revision, plus expected collection
  differences. Bulk JUnit XML and logs are not included here.
- `windows-arm64/b471/`: unchanged build/provenance reports, platform summary,
  upstream manifest, and 13 raw per-area case reports. Project tests reported
  7,112 passed and four skipped; all 407 benchmark cases passed. The compact
  partial upstream comparison records 12,324 unchanged common executions
  across ten completed partitions. Unpatched NumPy crashed during the next
  linalg partition; the crash record retains its raw-log hash and resource
  report. This is not a complete upstream comparison.

See [platform status](../../../docs/platforms.md) for the pending checks and
Actions links. Earlier reports do not establish the behavior of later builds.
