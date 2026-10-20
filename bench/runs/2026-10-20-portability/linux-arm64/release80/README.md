# Official manylinux aarch64 CP313 wheel checkpoint `80dec52`

[Preflight run 39377562130](https://github.com/michael-L-i/fastscipy/actions/runs/39377562130),
job `118479156993`, built source `80dec520bdc38f0601a746b676e5bdab907eb78a`
with Rust 1.97.1, Maturin 1.15.0, and the production release profile.
Auditwheel repaired the result as
`fastscipy-0.1.0a2-cp313-cp313-manylinux_2_28_aarch64.whl`.
An isolated environment installed that repaired wheel with NumPy 2.5.4 and
SciPy 1.18.1. Installed-wheel checks rejected platform/version overrides,
confirmed site-packages imports, and exercised all 13 component smoke paths.

- [Summary](summary.json): all 7,221 project tests and 407 benchmark cases
  across 13 areas passed, with identity and expected fast paths checked.
- [Raw case reports](cases/): all 13 original per-area JSON files.
- [Outer build report](build.json): 2,969 seconds for build, repair, and
  wheel validation together. Its sampled RSS covers the host cibuildwheel
  process tree, excluding Docker compiler processes. The container had a
  6 GiB hard memory limit and two CPUs; actual container peak is unknown.
  Project and upstream process-tree peaks were measured inside the container
  and remain in the summary.
- [Upstream manifest](upstream-manifest.json),
  [per-side completion records](upstream-partition-results.json), and
  [compact comparison](upstream-comparison.json): all 38 sides of 19 pairs
  completed with exit zero; 95,772 common executions have zero changed
  outcomes and no failures. The 684 baseline-only collections and two
  patched-only skipped empty parametrizations are expected differences.
- [Derived wheel provenance](wheel-provenance.json): wheel filename/hash,
  source revision, installed-check scope, and raw log hashes.

The original summary, cases, build report, manifest, and completion records
were copied byte for byte and verified by SHA256. The compact comparison
was derived from all 38 completed raw XML files and retains their hashes.
Bulk logs, XML, and the wheel remain outside git. This source-specific
preflight checkpoint does not validate later changes or complete the
32-wheel publication gate; older development reports remain in the parent
directory.
