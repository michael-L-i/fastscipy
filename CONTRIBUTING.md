# Contributing

Contributions are welcome: new fast paths, ports to other platforms, bug
reports with a reproducer, and benchmark results from other machines.

## Setup

For candidate development, use Rust (stable) and Python 3.12–3.15.

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin "scipy==1.18.1" pytest hypothesis ruff psutil pooch
CARGO_BUILD_JOBS=1 maturin develop --profile dev-release  # rebuild after Rust changes
python tools/check_native_build.py                     # verify the imported extension
```

## Checks

```sh
export FASTSCIPY_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
pytest tests                                          # identity tests, serial
python tools/scipy_suite.py --pyargs scipy.ndimage      # SciPy's suite, with vs without
python bench/run.py ndimage --check                    # benchmarks
ruff check python tests bench tools examples && ruff format --check python tests bench tools examples
cargo fmt --check && cargo clippy --profile dev-release --all-targets -- -D warnings
```

The published `0.1.0a4` wheels support all eight Linux, Windows, and Mac
platform variants on Python 3.12–3.15. See [platform support and measured
limitations](docs/platforms.md). The Release workflow validates distribution
wheels; the Platforms workflow collects development reports.

Use `python tools/platform_suite.py --output artifacts/platform-local` for a
serial project-suite and all-area benchmark report. The output directory must
be empty. `--upstream` also runs paired upstream SciPy comparisons, partitioned
by module. Choose explicit thread and memory budgets before larger runs.

The native build check compares the imported extension's embedded SHA-256 with
the current `src/` files, `Cargo.toml`, `Cargo.lock`, and `build.rs`. It works
without Git and ignores Python-only or documentation changes. A stale extension
or one built before provenance metadata was added fails with rebuild guidance.
`platform_suite.py` performs this check before running tests or benchmarks and
saves `native-build.json` beside its report. To compare an installed wheel with
a particular source tree, run `tools/check_native_build.py` with
`--source-root /path/to/source` using the wheel's interpreter. Unavailable sources are reported
as unverified, never as a match.

This verifies native source identity, not reproducible machine code or numerical
compatibility on an untested platform. The report also records the extension's
binary SHA-256 and build-time compiler, target, optimization level, and Cargo
profile classification (`release` can include custom release-derived profiles).
No source hashing or build checks happen during normal package import or calls.

## Adding a fast path

Each SciPy area is one component, `python/fastscipy/_<area>.py`, with
Rust kernels in `src/<area>/` exposed as `fastscipy._native.<area>`.
A component defines:

- `patch()`, which starts with `_patch.require_validated("1.18.")` and
  installs fast paths with `_patch.wrap`, `_patch.replace` and
  `_patch.rebind` (never by assigning to SciPy directly, or `unpatch()`
  cannot restore it);
- `smoke()`, a few small calls that reach every fast path, used by the fork
  test.

A fast path takes the same arguments as the function it replaces, and either
returns SciPy's exact result or `NotImplemented` to hand the call to SciPy.
The rules that keep results compatible:

1. **Decline anything you have not matched.** Check dtype, layout, options
   and sizes up front, and leave every warning and error to SciPy. Never
   write into a caller's `out=` array and then decline.
2. **Reproduce SciPy's arithmetic, not just its algorithm.** Keep the
   summation order, and use `mul_add` where SciPy's compiled code fuses a
   multiply and add (see [docs/design.md](docs/design.md)).
3. **Parallel work goes through the shared pools**:
   `crate::pool::pool().install(...)` in Rust, inside `py.detach(...)`, and
   `fastscipy._threads.parallel_map` in Python. Never create another pool.
   Only go parallel above a measured size.
4. **Compare with SciPy and check independent properties.** Add randomized
   cases to `tests/test_<area>.py` with `fastscipy.testing.check`, on one
   and several threads. Byte comparison is the default. Explicit numerical
   contracts are limited to documented Intel Accelerate cases and must
   retain result-contract, warning, input-mutation, and fast-path checks.
   Include declined cases and edge cases (empty, NaN, inf, ties, strided).
5. **Show it is faster.** Add cases to `bench/bench_<area>.py`, and make
   sure `bench/run.py <area> --check` passes.

Before opening a pull request, run SciPy's suite for the area with
`tools/scipy_suite.py`; no test may change outcome.

## Portable performance

Prefer algorithmic improvements and reusable kernels that benefit the supported
Linux, Windows and macOS targets on x86_64 and ARM64. Keep portable fallbacks
for architecture-specific SIMD; do not compile distribution wheels for the
developer's host CPU or assume its BLAS, floating-point contraction, alignment
or byte order applies elsewhere.

A benchmark on one machine establishes a local result, not a portable speedup.
Treat new thresholds and architecture-specific paths as provisional until the
platform matrix has checked correctness, fallback behavior and representative
performance. Require development platform checks to pass before merging new
fast paths. Do not tune general dispatch to one host's timing noise. Run normal
platform validation without `FASTSCIPY_ANY_PLATFORM` or `FASTSCIPY_ANY_VERSION`;
those overrides are only for explicit porting diagnostics. Published support
requires the complete release validation matrix, including Linux musl targets.

## License

By contributing you agree that your contributions are licensed under the
BSD-3-Clause license of this project.

## Releases

The `0.1.0a4` release passed all 32 wheel jobs, source-distribution and
complete artifact gates, PyPI publication, independent published-file hash
verification, and eight clean installed CP313 application pilots.
See [release evidence](docs/platforms.md#evidence). Every wheel runs installed
package and project checks; CP313 also runs benchmarks and paired upstream
comparisons. Publish only the complete validated set.

Builds use one Cargo job and bounded numerical thread and memory budgets.
The preflight runs two wheel jobs concurrently on separate GitHub runner VMs;
the full release runs eight. Each VM retains its own 6 GiB build limit,
4 GiB test limit, and two numerical threads; this adds no local RAM load.
Publish a target only after its wheel validation finishes successfully.

For a build-only preflight, dispatch the source branch to validate:

```sh
gh workflow run release.yml --ref main -f publish=none -f validation_only=true
```

This builds and validates eight Python 3.13 wheels plus one source archive,
with publishing disabled. It does not replace the full 32-wheel publication
gate or establish published platform support.

To rehearse, run the workflow manually with `publish: none`. After all jobs
succeed, download the 32 `release-dist-<target>-<interpreter>` wheel artifacts
and the `release-dist-sdist` source archive. Review the corresponding
`release-report-<target>-<interpreter>` validation logs. `publish: testpypi`
exercises an upload to TestPyPI. Publishing to PyPI requires a matching `v<version>` tag;
version declarations in Python, Cargo and the lockfile must agree.

Before the first upload, sign into PyPI and configure a pending Trusted Publisher:

| Field | Value |
|---|---|
| Project | `fastscipy` |
| Owner | `michael-L-i` |
| Repository | `fastscipy` |
| Workflow | `release.yml` |
| Environment | `pypi` |

Use a separate `testpypi` environment and publisher on TestPyPI for rehearsals.
No API token is needed. The workflow publishes only after the wheel checks pass.
A pending publisher does not reserve the package name until the first upload.

Update the README's installation instructions in the release commit before
creating and pushing the version tag: PyPI renders the README from that archive.
Push the tag only when ready to publish.

After publication, independently verify published file hashes against approved
artifacts, then run `Installed release pilots` at the matching source/version
for clean binary-only PyPI application checks on eight platform variants.
Preserve install manifests, process reports, and numerical compatibility
records. Application children use a 512 MiB process-tree limit; installation
and Linux containers use 1 GiB. Minimal Alpine images need runtime
`libstdc++`, without a compiler.
