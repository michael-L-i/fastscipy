# Local Mac ARM reference-policy checkpoint `1f48cd6`

This installed-wheel checkpoint tested Python source
`1f48cd63a544379363234081cc8178a574fea73f` on macOS 26.6.2 arm64,
CPython 3.13.12, NumPy 2.5.4, and SciPy 1.18.1. The Rust source tree was
unchanged from `ec2d724`; all 32 packaged Python files were verified byte for
byte against `1f48cd6`. The [provenance](provenance.json) retains the exact
revisions, Rust tree ID, final wheel SHA256, source hashes, and external-log
hashes. No wheel or bulk logs are stored here.

- The unchanged [platform summary](summary.json) and compact
  [project XML summary/hash](tests-summary.json) record 7,275 passing project
  tests, no failures/errors/skips, and all 407 benchmark cases across 13 areas
  passing byte identity and expected fast-path checks.
- All [13 raw case reports](cases/) are preserved unchanged. There were zero
  reference-variance exceptions. Numerical work used two threads and BLAS one.
- **No upstream suite was run** in this checkpoint. Earlier upstream evidence
  remains separate and does not establish upstream coverage for this source.

The unchanged [validation resource records](validation.json) retain every
setup, build, packaging, test, and later targeted-check record. The successful
native build took 211.69 seconds and peaked at 5,052 MiB RSS. Final packaging
reused that native artifact, taking 0.44 seconds and 18 MiB RSS. The complete
project/benchmark process took 331.08 seconds and peaked at 1,715 MiB RSS.
The isolated environment initially lacked `pip`; installation then succeeded
with `uv`. That setup return code is retained and is not a source failure.

The separate [uncached-build record](uncached-validation.json) preserves an
earlier intentionally cancelled build that used the wrong target cache. Its
return code −9 is not the outcome of the successful native build above.

Later safety changes received targeted validation only: `bfd6e5a` passed
205 oracle-safety tests, `d2c9d76` passed 13 report-isolation tests, and
`d4fe822` passed 194 layout-safety tests. Their resource records are retained,
but the 7,275-test/407-case full run applies to `1f48cd6`, **not** those later
revisions or current HEAD.

All 16 original JSON reports were copied byte for byte and hash-verified.
The project XML stays outside Git; its filename, size, SHA256, and derived
test totals are retained. No tests or builds were rerun to create this archive.
