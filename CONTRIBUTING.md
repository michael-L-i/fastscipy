# Contributing

Contributions are welcome: new fast paths, ports to other platforms, bug
reports with a reproducer, and benchmark results from other machines.

## Setup

You need Rust (stable) and Python 3.12 to 3.14.

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin "scipy==1.18.1" pytest pytest-xdist hypothesis ruff
maturin develop --profile dev-release     # rebuild after every Rust change
```

## Checks

```sh
pytest tests -n auto                                   # identity tests
python tools/scipy_suite.py --pyargs scipy.ndimage -n auto   # SciPy's suite, with vs without
python bench/run.py ndimage --check                    # benchmarks
ruff check python tests bench tools && ruff format --check python tests bench tools
cargo fmt --check && cargo clippy --profile dev-release --all-targets -- -D warnings
```

CI runs all of these on macOS arm64. A separate workflow runs the identity
tests on Linux to track that port.

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
The rules that keep results identical:

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
4. **Test against SciPy, not against expected values.** Add randomized
   cases to `tests/test_<area>.py` with `fastscipy.testing.check`, which
   compares bytes and asserts the fast path ran, on 1 and several threads.
   Include declined cases and edge cases (empty, NaN, inf, ties, strided).
5. **Show it is faster.** Add cases to `bench/bench_<area>.py`, and make
   sure `bench/run.py <area> --check` passes.

Before opening a pull request, run SciPy's suite for the area with
`tools/scipy_suite.py`; no test may change outcome.

## License

By contributing you agree that your contributions are licensed under the
BSD-3-Clause license of this project.
