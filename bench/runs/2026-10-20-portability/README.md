# Portability development checkpoints

These reports describe intermediate `0.1.0a2` builds, not a validated final
release. Original JSON reports were copied byte for byte; revisions,
dependency versions, resource measurements, and results were not rewritten
to match a newer checkout. Derived comparisons and provenance records are
explicitly labeled and retain hashes of their raw evidence.

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
- `linux-arm64/release80/`: official repaired `manylinux_2_28` CP313 wheel
  from source `80dec52`, checked with default activation and no overrides.
  All 7,221 project tests, 407 benchmark cases, and 19 upstream partition
  pairs pass; 95,772 common upstream executions have no changed outcomes or
  failures. It retains original case, project, resource, manifest, and
  completion reports plus derived wheel provenance and raw XML hashes.
  Container peak build memory is unknown; the outer RSS metric excludes
  Docker compiler processes. This checkpoint does not complete the full
  32-wheel release gate.
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
- `windows-x86_64/daf9/`: native AMD64 revision `daf9acd`, built with the
  release profile. Project tests reported 7,198 passed and four fork-dependent
  skips; all 407 benchmark cases passed. All 19 paired upstream partitions
  completed with no changed outcomes across 95,772 common executions and no
  failures. Original reports and a compact comparison containing all 38 raw
  XML hashes are retained; full release-matrix validation remains separate.
- `mac-x86_64/daf9/`: native Intel revision `daf9acd`, built with the release
  profile. Project validation failed: 7,166 passed and 36 failed; benchmark
  identity passed 406 of 407 cases, with one BFGS result mismatch. All 19
  upstream pairs nevertheless completed with 95,772 unchanged common outcomes
  and no failures. Raw reports, compact upstream comparison with 38 XML
  hashes, and project failure identifiers are retained. Later fixes require
  their own validation.

## Official CP313 preflight at `80dec52`: failed overall

[Run 39377562130](https://github.com/michael-L-i/fastscipy/actions/runs/39377562130)
finished with seven passing wheel jobs and one failed Mac Intel job. The
[compact overview](preflight80.json) records all eight outcomes. Complete-set
verification and both publishers were skipped; nothing was published by this
run. Default activation and installed native smoke checks passed on each
wheel before the project/benchmark checks.

- `linux-x86_64/release80/`, `musllinux-aarch64/release80/`,
  `musllinux-x86_64/release80/`, and `mac-arm64/release80/`: each passed
  7,221 project tests and all 407 benchmark identity/expected-path checks.
  The previously archived `linux-arm64/release80/` checkpoint covers the
  manylinux aarch64 wheel with the same passing counts.
- `windows-x86_64/release80/` and `windows-arm64/release80/`: each passed
  7,217 project tests with four fork-dependent skips; all 407 benchmark cases
  passed. Windows ARM's upstream records retain eight explicit, identical
  NumPy-reference quarantine skips per side; these are not passing cases.
- `mac-x86_64/release80/`: **failed**, with 7,189 project tests passing and
  32 failing. Benchmark identity passed 406/407 cases; numerical-gradient
  BFGS differed. The project failure record retains all identifiers, bounded
  message excerpts, complete-message/trace hashes, and raw project-file hashes.

Each of the eight upstream comparisons completed all 19 pairs with 95,772
common executions, no changed outcomes, no incomplete pairs, and no failures.
All expected benchmark fast paths ran. Each new folder retains 17 original
JSON reports unchanged, plus derived comparison/wheel provenance and log/XML
hashes. Those successful upstream comparisons do not make the failed Mac
Intel wheel or overall preflight successful. Full release validation remains
pending for the final source and all 32 wheels.

The [Intel Mac reference-repeatability diagnostics](mac-x86_64/reference-stability-956/README.md)
reuse the unchanged `daf9acd` wheel with probe checkout `956fc49`. Raw stats,
optimization, and sparse probes demonstrate varying repeated unpatched
results and alignment-dependent NumPy reductions. The archive retains exact
reports, provenance, source/wheel hashes, and hashes for external sparse
arrays. It does not establish a numerical exception or resolve independent
LSQR discrepancies.

The [second Intel Mac sparse-reference diagnostics](mac-x86_64/reference-stability-621/README.md)
reuse the same native `daf9acd` wheel with diagnostic checkout `62147de` and
all 13 components enabled. All 96 direct sparse byte checks passed; float64
LSQR produced two exact variants on both reference and patched sides, with
unchanged input buffers. Unchanged reports and an external-array hash manifest
retain this narrower evidence separately from the earlier diagnostics.

The [local Mac ARM reference-policy checkpoint](mac-arm64/reference-policy-1f48/README.md)
tested installed Python source `1f48cd6` with the unchanged `ec2d724` Rust
tree: 7,275 project tests and all 407 benchmark identity/expected-path checks
passed, with zero reference-variance exceptions. It includes no new upstream
run. Later oracle/report/layout safety changes have separate targeted results;
the full result is not a claim about those later revisions.

See [platform status](../../../docs/platforms.md) for the pending checks and
Actions links. Earlier reports do not establish the behavior of later builds.

The [Intel Mac cast-selector experiment](mac-x86_64/daf9/cast-selector-621/README.md)
reuses the unchanged `daf9acd` native wheel with probe checkout `62147de`.
Enabling only `_SCIPY_INT_CAST` passed all 64 existing selected tests, including
all 18 previous fast-path misses, with the selector restored afterwards.
Original focused JUnit, worker, monitor, log, provenance, and verified hashes
are retained. This does not certify a rebuilt release wheel or resolve the
independent numerical failures.

The [Intel current-Python overlay diagnostic](mac-x86_64/python-source-1f48/README.md)
**failed and remained partial**: 7,273 project tests passed, two BiCGSTAB
callback-history tests failed, and the outer 1,200-second timeout interrupted
benchmarking. Eleven raw area reports retain 312 completed cases (311
byte-identical, one explicitly marked reference variation), with no missing
expected fast paths; the remaining 95 cases did not complete. Current Python
source `1f48cd6` and unchanged `daf9acd` native provenance are kept distinct.
This is not a passing full-suite or release-wheel result.

The [Intel BiCGSTAB repeatability diagnostic](mac-x86_64/reference-stability-b785/README.md)
records unpatched callback counts 40/41/42 in both patch scopes and patched
counts spanning 40–43. Common prefixes differ; all 65 saved final BiCGSTAB
solutions satisfy the requested residual bound, while the other nine solver
counts remain stable. Original reports, exact probe source, external NPZ
hashes, and derived per-scope analysis are retained. This is reference
stability evidence, not a production fix or release certification.

The [remaining Intel overlay checks](mac-x86_64/python-remaining-3df/README.md)
**failed** at source `3df5f54`: 817 selected tests passed and four test-validator
checks failed (DIA stored-zero symmetry counting and Pearson validator return
type). The targeted special/stats benchmarks completed all 95 cases with
byte identity and expected acceleration. Original reports and distinct
Python-source/reused-native provenance are retained; this is not a complete
project, 407-case, or newly built release-wheel pass.

The [Intel tests-only contract diagnostic](mac-x86_64/python-contract-5f76195/README.md)
**failed** at Python source `5f76195`, with 828 selected tests passing and
one rejected linregress comparison. Both earlier BiCGSTAB and Pearson
parameters passed; exact failure diagnostics remain archived. This reused
the unchanged `daf9acd` native wheel and ran no benchmarks or upstream
comparisons. It is not a complete-project or release-wheel pass.

The failed contract archive also retains an [exact linregress input replay](mac-x86_64/python-contract-5f76195/reproduction/README.md):
seed 12345, iteration 69, constant x with a float64 centering residue and
SciPy's catastrophic-cancellation warning. This was a bounded local ARM replay;
reported resource observations and the later `e290714` test-validator candidate
are explicitly separated from raw output and pending native Intel validation.

The successor [Intel selected-test contract diagnostic](mac-x86_64/python-contract-e290714/README.md)
**passed all 830 selected tests**, with no failures/skips, at Python source
`e290714` using the unchanged `daf9acd` native wheel. It retains 108 explicit
reference-variation records and verified source/import/environment provenance.
It ran no benchmarks or upstream comparisons and does not certify a complete
project rerun or the final 32 release wheels. Earlier failed runs stay unchanged.
