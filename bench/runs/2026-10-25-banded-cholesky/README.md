# Batched banded Cholesky

SciPy 1.18.1 already implements dense `cholesky`, `inv`, `solve`, QR, SVD,
and general eigenvalue batches in its native `_batched_linalg` implementation.
Removing a supposed Python loop from those functions would duplicate upstream
work. Its `cholesky_banded` and `cho_solve_banded` still use `_apply_over_batch`
with one f2py call per matrix. These are the selected new coverage.

The implementation calls SciPy's own DPBTRF/DPBTRS capsules, retains compact
band storage, and uses the existing shared Rayon batch scheduler. It copies
unused padding too, matching both returned factor bytes and finite validation.
Unsupported types/options and unsuccessful batches return to SciPy with the
original inputs. The two new capsules receive the same explicit ABI signature
checks as the existing dense LAPACK kernels.

The workloads represent independent banded positive-definite systems, such as
ensembles of one-dimensional discretizations or banded covariance problems.
The benchmark includes factorization, repeated-factor solves, and complete
factor-and-solve calls, with tiny batches, narrow and wider bands, plus genuine
SciPy single-matrix and complex-dtype fallback controls. Timing excludes fixture
construction and uses independently generated reference/candidate fixtures.

Other audited opportunities remain future work: `eigh_tridiagonal` still loops
in Python but has considerably more LAPACK driver and selection semantics;
Floyd–Warshall row parallelism needs a barrier for each pivot and a separate
crossover/exactness study. No speedup is claimed for these unimplemented paths.

API references:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky_banded.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_solve_banded.html

## Validation and measurements

Measured on Apple Silicon, CPython 3.13.6, NumPy 2.5.4, SciPy 1.18.1,
with BLAS/OpenMP limited to one thread and fastscipy limited to one or two.
`native-build.json` confirms the imported artifact matches the native sources.
The native build used one Cargo job and peaked at 5122 MiB under a 6 GiB cap.

| End-to-end factor + solve | SciPy ms | 1 thread ms | 2 threads ms | Speedup 1T / 2T |
| --- | ---: | ---: | ---: | ---: |
| 4 systems, 2 band rows, size 8 | 0.033 | 0.007 | 0.007 | 4.9x / 4.9x |
| 1000 systems, 2 band rows, size 32 | 5.226 | 1.020 | 0.623 | 5.1x / 8.4x |
| 1000 systems, 5 band rows, size 128 | 9.329 | 4.977 | 2.709 | 1.9x / 3.4x |
| 128 systems, 9 band rows, size 1024 | 6.384 | 5.676 | 3.030 | 1.1x / 2.1x |

Reusing the factor for 1000 size-32 systems improves solves 6.5x on one
thread and 11.3x on two. Larger bandwidths are increasingly dominated by
LAPACK rather than dispatch, so the gains narrow; no blanket speedup is claimed.
Single-matrix and complex-input fallback controls remain approximately 1.0x.
All comparisons include returned array strides and exact bytes. The baseline
is genuinely unpatched SciPy, which is also the merged-main implementation for
these two functions: `baseline-bindings.json` confirms that patching commit
95c4304 leaves both function objects unchanged.

Validation:
- 2034 project linalg tests pass, including upper/lower layouts, unused padding,
  NaNs, negative strides, readonly/broadcast inputs, failures, overwrite/options,
  and exact non-mutation checks, at the existing 1/4-thread test settings.
- 11605 upstream tests ran in both modes with zero changed outcomes. Both modes
  retain SciPy's existing `TestEig::test_singular` failure. No partition was
  incomplete and no quarantine applied on this platform.
- The initial upstream run exceeded its 1 GiB RSS cap (1167 MiB); its monitor
  report is retained. An authorized 2 GiB rerun completed at 1259 MiB. No tests
  were removed to fit the budget.
- Full linalg benchmark `--check` passes; peak RSS 908 MiB under a 2 GiB cap.
  An earlier timing run overlapped another task, was canceled, and is excluded.
- Ruff, formatting, Cargo formatting and all-target Clippy with warnings denied
  pass. The shared ndarray eligibility guard now declines unaligned buffers,
  protecting both new and existing batched LAPACK views; offset-one buffer
  regressions cover banded factor/solve and LU factor/solve.

Reproduce using a freshly built extension whose provenance check passes:

```sh
export FASTSCIPY_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python tools/check_native_build.py
python -m pytest tests/test_linalg.py -q
SCIPY_AVAILABLE_MEM=256Mib python tools/scipy_suite.py --pyargs scipy.linalg -q
python bench/run.py linalg --check
```
