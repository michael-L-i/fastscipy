# How fastscipy works

## Patching

`fastscipy.patch()` changes SciPy in the running process only. Nothing on
disk changes, and `unpatch()` restores every binding. Each component uses
the narrowest seam that works:

1. **Private kernel swap.** Replace the compiled function a public Python
   wrapper calls, and keep the wrapper. ndimage is done this way: SciPy's
   `_nd_image` C module is replaced by a stand-in whose functions try a Rust
   kernel first. SciPy's own argument checking stays in place.
2. **Method patch.** Replace a method or private helper, for example
   `rv_continuous.cdf`, `Delaunay.find_simplex` or `BSpline.__call__`.
3. **Public rebind.** Replace a public function wherever SciPy's modules
   bind it, including `from ._x import f` copies, and give the wrapper the
   original's entries in SciPy's identity-keyed registries
   (`xp_capabilities_table`). Wrappers keep the original's name, docstring
   and signature.

All replacements go through `fastscipy._patch.replace`, which records the
old value, so `unpatch()` can restore it.

Code that ran `from scipy.special import expit` before `patch()` keeps the
original function, so call `patch()` early.

## Exactness

Every fast path takes the same arguments as the SciPy function it replaces,
and returns its result or declines by returning `NotImplemented`. A declined
call runs SciPy on the untouched arguments. General wrappers also fall back
on internal errors. Paths that invoke user callbacks or accept computations
that can raise instead use wrappers that propagate those exceptions without
retrying the call. This preserves callback side effects and error-handler
counts. The fast path covers a common, well-defined case and leaves the rest
to SciPy.

Byte-identical output takes more than the same algorithm:

- Interpolation, signal widths, and correlation kernels probe the installed
  SciPy binary to select matching fused or separate multiply-add arithmetic.
  C++ BSpline, Fortran FITPACK, and Cython interpolation are checked separately;
  an unmatched kernel leaves the call to SciPy.
- ndimage spline transforms and EDT follow the compiler expression trees
  verified in official SciPy binaries, including operations that must remain
  separate. Matching one kernel does not establish another kernel's arithmetic.
- Summation order matches SciPy's loops (and NumPy's pairwise summation, or
  the installed BLAS's dot order, where SciPy relies on them). The `dlsim`
  matrix-vector paths are checked against NumPy for each matrix shape.
- Correlation output conversions are calibrated separately from arithmetic.
  Hybrid correlation and spline paths can compute in Rust and let SciPy apply
  the final compiler-specific integer cast. There is no universal rule for
  out-of-range C casts.
- Scalar powers, array squares, signed zeros, and NaN propagation can differ
  across builds. Fast paths preserve SciPy's operations and dtypes.
- Native BLAS and LAPACK loaders check complete C API capsule signatures
  before loading function pointers. Incompatible integer ABIs leave those
  native paths off.

These details depend on how SciPy was built. [Platform status](platforms.md)
distinguishes published support from candidate validation. Identity results
apply to the tested binaries and dependencies.

For porting tests, `FASTSCIPY_ANY_PLATFORM=1` bypasses the platform guard and
`FASTSCIPY_ANY_VERSION=1` bypasses the version guard. Neither override makes
an unvalidated configuration byte-identical; run the identity tests when
developing a port.

## Threads

There is one thread budget (`set_num_threads`, `FASTSCIPY_NUM_THREADS`,
default all cores) and two pools that share it:

- A Rust (rayon) pool for the kernels. It is created lazily and rebuilt in
  a forked child, whose copy of the parent's worker threads does not exist.
  The child replaces the pool's mutex slot before locking it, so it cannot
  wait on a lock held by a vanished parent thread.
- A Python thread pool for running SciPy's own GIL-releasing code on chunks
  (special functions, `cdist`, multichannel filters). Only one call uses it
  at a time. A second caller, such as another user thread or a nested call,
  runs serially instead of oversubscribing the machine.

Repeating `set_num_threads()` with the current budget keeps the Python pool
warm. Changing the budget is serialized with task submission; already submitted
chunks finish on their original executor, and the next parallel call creates
a pool with the new budget. Workers can change the budget without waiting for
themselves to finish.

Work is split only above measured size thresholds, so small calls never pay
for threads. Thread-local state that SciPy depends on, such as
`scipy.special` error handling, is either copied into workers or makes the
call go to SciPy.

## Testing

- `tests/` compares every fast path with plain SciPy on randomized inputs
  (dtypes, layouts, shapes, options, NaN and inf, ties) on 1 and several
  threads, using `fastscipy.testing.check`. It also checks that each call
  took the fast path, and that declined calls give SciPy's result.
- `tests/test_core.py` forks a child after every pool has run and checks it
  finishes, and checks that `unpatch()` restores every binding.
- `fastscipy.pytest_plugin` runs SciPy's own test suite with the patch on.
  `tools/scipy_suite.py` runs it patched and unpatched and compares the
  outcome of every test.
