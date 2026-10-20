# Official manylinux-x86_64 CP313 checkpoint `80dec52`

[Preflight run 39377562130](https://github.com/michael-L-i/fastscipy/actions/runs/39377562130), job `118479157007`, built source
`80dec520bdc38f0601a746b676e5bdab907eb78a` with Maturin 1.15.0 and the
production release profile. The repaired wheel installed in an isolated
environment with NumPy 2.5.4 and SciPy 1.18.1. Installed-wheel checks passed
with default platform activation, no version/platform overrides, installed
native imports, and all 13 component smoke paths exercised.

This is a passing source-specific wheel checkpoint, not a passing result for
the entire preflight or the final 32-wheel publication gate. Mac Intel failed
in the same run; complete-set verification and both publishers were skipped.

- [Project and benchmark summary](summary.json): 7,221 passed, 0 failed, 0 skipped.
  All 407 benchmark cases across 13 areas passed identity and expected
  fast-path checks.
- [Raw case reports](cases/): all 13 original per-area JSON files.
- [Build/validation resources](build.json): total time includes build, repair,
  installed checks, project tests, benchmarks, and upstream comparisons.
  Outer host process-tree RSS excludes Docker compiler processes. The
  container had hard limits of 6 GiB and two CPUs; actual container peak is
  unknown. Test process trees were measured separately inside the container.
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
