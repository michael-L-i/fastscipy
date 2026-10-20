# Official macos-x86_64 CP313 checkpoint `80dec52`

[Preflight run 39377562130](https://github.com/michael-L-i/fastscipy/actions/runs/39377562130), job `118479156969`, built source
`80dec520bdc38f0601a746b676e5bdab907eb78a` with Maturin 1.15.0 and the
production release profile. The repaired wheel installed in an isolated
environment with NumPy 2.5.4 and SciPy 1.18.1. Installed-wheel checks passed
with default platform activation, no version/platform overrides, installed
native imports, and all 13 component smoke paths exercised.

**This wheel failed validation.** It built, repaired, and installed, but its
project and benchmark results below prevent release. No distribution artifact
was uploaded. Later fixes require their own checks.

- [Project and benchmark summary](summary.json): 7,189 passed, 32 failed, 0 skipped.
  Benchmark identity passed 406 of 407 cases; numerical-gradient BFGS was
  nonidentical. All expected benchmark fast paths ran. The compact
  [project failure record](project-failures.json) retains all 32 identifiers,
  bounded message excerpts, full trace hashes, and raw project-file hashes.
- [Raw case reports](cases/): all 13 original per-area JSON files.
- [Build/validation resources](build.json): total time includes build, repair,
  installed checks, project tests, benchmarks, and upstream comparisons.
  The sampled host process-tree RSS includes build and validation children.
- [Upstream manifest](upstream-manifest.json),
  [per-side completion records](upstream-partition-results.json), and
  [compact comparison](upstream-comparison.json): all 38 sides of 19 pairs
  completed with exit zero; 95,772 common executions had no changed outcomes,
  no incomplete pairs, and no failures. The 684 baseline-only collections and
  two patched-only skipped empty parametrizations are expected differences.
- [Derived wheel provenance](wheel-provenance.json): source, job conclusion,
  wheel identity/hash when available, installed-check scope, and raw log hashes.

All 17 original JSON reports were copied byte for byte and verified by
SHA256. The comparison was derived from all 38 completed raw XML records
without rerunning tests and retains their filenames, sizes, and hashes.
Bulk logs, XML, and wheel binaries remain outside git. Original metadata was
not rewritten to describe later source changes.

Passing upstream comparisons do not override the failed project tests or
benchmark mismatch. See the [failed preflight overview](../../preflight80.json)
for the complete eight-target outcome.
