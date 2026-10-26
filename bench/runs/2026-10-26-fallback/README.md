# LinearOperator fallback dispatch · 2026-10-26

Batched and column inputs cannot use the unbatched `LinearOperator` fast path.
Previously they reached several operator metadata checks before falling back to
SciPy. The candidate reads an exact ndarray's shape once, rejects nonvectors
first, and reuses that shape for the remaining size check. It preserves every
eligibility condition, including exact boolean `adjoint` values. Unpacking the
operator dimensions and swapping only for the adjoint avoids creating an
intermediate dimension tuple on each accepted vector call.

SciPy still owns declined-call shape handling, warnings, callbacks and errors.
Numerical kernels, public signatures, counters and callback retry behavior are
unchanged. In particular, callbacks are not retried after exceptions.

## Measurements

[probe.py](probe.py) rotates three states in one process: genuinely unpatched
SciPy, previous fastscipy (`a5961a5`), and the candidate. It retains the same
operator/input objects; restoring class methods actually restores these
operators' dispatch. Each of nine rotations times 10,000 calls after 100 warmups.
Every case is byte-identical to the genuinely unpatched result. Values below
are median nanoseconds per public call, including dispatch.

| Call | SciPy | Previous fastscipy | Candidate | Previous / candidate |
|---|---:|---:|---:|---:|
| Accepted vector | 1492.8 | 658.8 | 664.4 | 0.99× |
| Integer vector fallback | 1453.5 | 1588.4 | 1570.3 | 1.01× |
| Batched fallback | 1747.0 | 1950.2 | 1836.9 | 1.06× |
| Column fallback | 2496.1 | 2771.6 | 2652.5 | 1.04× |
| Integer DCT fallback, unchanged control | 3274.6 | 3514.9 | 3521.0 | 1.00× |
| Floating DCT, unchanged control | 3390.2 | 1928.9 | 1934.8 | 1.00× |

The useful gain is about 110–120 ns on the batched/column fallback calls.
The accepted-vector difference is about 6 ns, similar to the unchanged FFT
controls; small differences are noise-sensitive. These are local macOS ARM64,
CPython 3.13.6, NumPy 2.5.4, SciPy 1.18.1 measurements with one numerical thread
and BLAS/OpenMP limited to one. They are not solver or cross-platform speedup
claims. Raw samples are in [results.json](results.json). Column-vector timings
use the normal default warning filter after warmup; warning behavior itself is
covered by the tests.

An extra `ndim` check without reusing shape/dimension work regressed accepted
vectors by roughly 4% and was rejected. Explicit-input FFT wrappers and skipping
empty keyword expansion did not establish an end-to-end improvement. **No FFT
production code changes are included.**

## Genuine-baseline benchmark gate

Using draft PR #7's runner (`2bf6417`) against this candidate, all **67 cases**
(36 FFT, 31 sparse linear algebra) passed byte identity, expected-fast-path, and
the unchanged 0.9× one-thread regression threshold. The budget was 0.1 seconds
per timing series, with one and two numerical threads and BLAS/OpenMP one.
[Full log](true-baseline.log), [FFT rows](true-baseline/fft.json),
[sparse rows](true-baseline/sparse_linalg.json).

The previously problematic controls were repeated three more times with the
same runner and budget. All nine checks passed:

| Control | Genuine SciPy / candidate range |
|---|---:|
| Integer DCT | 0.924–0.931× |
| Integer LinearOperator, 100 calls | 0.910–0.926× |
| Batched LinearOperator, 100 calls | 0.934–0.986× |

[Repeated raw rows](control-repeats.json). These comparisons include the shared
accounting improvement already on main; the FFT improvement relative to the
old draft's measurements is not attributable to this sparse-only change.
The checks retain their original threshold. The integer-vector control has a
smaller margin, so passing local checks does not establish every platform's
performance. PR #7 readiness requires this production change and current main,
not merely its original, older head.

## Correctness and provenance

- **297 project tests passed** (sparse linear algebra and core), including
  unsupported vector shapes, integer inputs, both adjoint directions, callback
  warnings/errors exactly once, unchanged input bytes, and no extra truth
  evaluation for nonboolean adjoint arguments. [Log](project-tests.log)
- **1,587 paired upstream cases**, zero changed outcomes and zero fast-path
  errors, covering `scipy.sparse.linalg.tests.test_interface` and
  `scipy.sparse.linalg._isolve`. [Log](upstream.log)
- Ruff checks and formatting passed for changed Python files.
- Numerical validation ran serially under a 512 MiB process-tree monitor.
  Project, upstream and benchmark sampled peaks were 209, 232 and 120 MiB.
  [Resource records](resources.json)
- No native source changed. Validation reused the verified baseline native
  build whose Rust source matches `a5961a5`, SHA-256
  `9c1b963ebc4d20c08baf91cbd839ca6c96f45a21d82cbb3515fe888a9dd798cd`.

## Reproduction

From the candidate checkout with a matching native build and test dependencies:

```sh
export PYTHONPATH=python FASTSCIPY_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python bench/runs/2026-10-26-fallback/probe.py
python -m pytest tests/test_sparse_linalg.py tests/test_core.py -q
python tools/scipy_suite.py --pyargs scipy.sparse.linalg.tests.test_interface scipy.sparse.linalg._isolve

git show 2bf6417:bench/run.py > bench/run_reference.py
python bench/run_reference.py fft sparse_linalg --budget 0.1 --check --output-dir /tmp/fallback-reference
```

For the repeated controls, run the reference runner three times for `fft
--only 'dctn integer control'` and `sparse_linalg --only 'fallback control'`,
with the same budget and separate output directories. `bench/run_reference.py`
is a temporary copy of the reviewed runner and is not part of the production
change; remove that temporary file afterwards.
