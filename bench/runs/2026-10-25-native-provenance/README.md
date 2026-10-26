# Native build provenance

An extension with the right package version can still be stale relative to the
Python checkout. This change embeds a versioned SHA-256 of native build inputs
in `_native._build_info()` and adds `tools/check_native_build.py` to compare the
loaded extension with a requested source tree. `platform_suite.py` records the
result and refuses to start numerical validation unless it is a match.

The fingerprint includes relative file names and bytes for `src/`, `Cargo.toml`,
`Cargo.lock`, and `build.rs`. It excludes Git state, timestamps, absolute checkout
paths, Python files and documentation. The build script watches the source
directory as well as individual inputs so additions and removals trigger it.
The metadata also records Rust compiler version, target, Cargo's profile
classification and optimization level. Cargo reports custom `dev-release` as
`release`; this field is not presented as the exact requested profile name.

Normal package import and numerical calls do not hash files. Source identity
does not establish identical compiler flags, ABI compatibility, reproducible
machine code or correctness on an untested platform. The standalone report
also includes the actual loaded binary's SHA-256 for artifact identification.

## Validation

- **39 focused tests passed**: 20 provenance cases and 19 existing platform
  monitor cases. Tests cover source changes/additions/removals/renames, relocated
  source trees without Git, ignored Python/docs changes, stale/missing/invalid
  metadata, missing binaries, unavailable wheel sources, source/binary records,
  real Rust/Python fingerprint agreement, and the platform preflight stopping
  before any numerical subprocess. [Log](tests.log)
- A real CPython 3.13 development wheel was built with one Cargo job. The final
  build took 139 seconds and peaked at 2,179 MiB sampled process-tree RSS under
  a 6 GiB cap. [Resources](build-resources.json)
- Both the extracted wheel and the development extension report a matching
  source fingerprint. [Wheel](wheel-check.json), [development](source-check.json)
- A real Maturin source archive includes `build.rs`, `Cargo.lock` and the checker;
  its extracted tree has no `.git` and matches the checkout fingerprint.
  [Archive check](sdist-check.json)

Archive validation initially caught Maturin adding the implicit `readme` field
to `Cargo.toml`. Making that existing metadata explicit preserved identical
native input bytes across checkout and source archive. The recorded wheel was
rebuilt after that fix; the initial differing archive is not reported as a pass.

The final native source fingerprint is
`5f9095417fd0f5fa4bfdd5ef4364c8ee926d6379342c5b9edfd7aa630023fdc4`.
The recorded wheel's native binary SHA-256 is
`954aa1f8513058831b05f055b74d62ae9374e0392f64e2b8921c3c3755d59011`.
These are local macOS ARM64 / Python 3.13.6 development artifacts, not a release
or cross-platform validation claim. No numerical algorithm changed.

## Reproduction

With the development environment active:

```sh
CARGO_BUILD_JOBS=1 maturin develop --profile dev-release
python tools/check_native_build.py
python -m pytest tests/test_native_build.py tests/test_platform_monitor.py -q
maturin sdist --out /tmp/fastscipy-source
```

Extract the generated source archive and pass its root with `--source-root` to
the checker. An installed wheel can be checked the same way using its Python
environment. Without complete native sources the checker reports
`source_unavailable` and exits nonzero; legacy extensions report
`metadata_missing`, never a successful match.
