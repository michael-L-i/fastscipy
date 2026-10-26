# Small cdist dispatch audit

**No production dispatch change is included.** SciPy's `cdist` wrapper already
performs little Python work: array conversion, shape checks and metric lookup,
then the same compiled metric used by the proposed direct path. Unlike the
previous small-`pdist` improvement, bypassing this wrapper did not establish a
speedup over genuine SciPy once fastscipy's guards and accounting were included.

The PR adds 286 compatibility cases for small rectangular calls: canonical
metrics, empty dimensions, C/Fortran/reversed/strided/unaligned read-only inputs,
NaN payloads and warning behavior, input nonmutation, callback count and ordering,
foreign inputs, unsupported options, caller-provided outputs, and patch lifecycle.
Independent Euclidean distances and symmetry check more than two implementations
against each other. These cases retain the existing small-call fallback behavior.

## Rejected experiments

[initial-prototype.patch](initial-prototype.patch) adds a narrow float64 ndarray
path at the beginning of `_cdist_fast`, accepting only canonical pybind metrics
without options. It keeps SciPy's exact compiled arithmetic. It improves on the
existing patched dispatch at two threads, but remains slower than unpatched
SciPy and adds overhead to several declined calls. Adding complexity for that
tradeoff was rejected.

A second [fused-wrapper diagnostic](fused_probe.py) removes the generic
wrapper-to-fast-function call by combining guards and counting in a wrapper with
an explicit signature. It deliberately omits threaded dispatch and the `.fast`
testing hook, so it is not a complete implementation. Even this lower-overhead
diagnostic did not beat SciPy on the tiny input tested. It is retained as a
measurement experiment, not supported package code.

The existing tiny-call overhead remains an improvement opportunity. This audit
does not claim to solve it, and it does not add knowingly failing cases to the
ordinary benchmark runner or weaken its regression threshold.

## Measurements

The reference is a function captured from genuinely unpatched SciPy. The
previous-fastscipy and rejected-candidate functions use the same accounting
implementation and original compiled kernels. The previous source is `a5961a5`.
Each workload alternates the three implementations over seven batches of 1,000
calls after warmup. Both one and two thread settings are recorded in
[results.json](results.json). Inputs are identical and outputs are byte-identical
for all measured cases. Unsupported controls include float32, lists, aliases,
weights, `out`, and cosine.

Two-thread setting; times are microseconds per call:

| Call | SciPy | Previous fastscipy | Rejected candidate |
|---|---:|---:|---:|
| euclidean 4x5x3 | 0.763 | 1.650 | 1.020 |
| euclidean 16x17x3 | 1.011 | 1.891 | 1.314 |
| euclidean 32x33x16 | 3.219 | 4.157 | 3.507 |
| euclidean 100x101x16 | 24.785 | 25.746 | 24.938 |
| float32 control | 1.022 | 1.938 | 2.130 |
| alias control | 0.756 | 1.132 | 1.441 |
| cosine control | 1.210 | 2.107 | 2.440 |

The fused diagnostic uses nine alternating batches of 10,000 calls on 4×4
Euclidean distances with three dimensions. It measured **702.0 ns** for SciPy versus **798.3 ns** for the diagnostic;
the guarded wrapper remained slower. See [samples](fused-results.json).

These are local Python 3.13.6 / NumPy 2.5.4 / SciPy 1.18.1 measurements on macOS
ARM64, with BLAS/OpenMP limited to one thread. They do not establish platform-wide
performance. Complete samples, including larger shapes and controls, are retained.
An initial run may have overlapped another agent's numerical work; the retained
final timing files were rerun in an explicitly reserved quiet interval.

## Validation

Production code was restored before final validation:

- **1,034 project tests passed**, including 286 new small-call cases; sampled
  process-tree peak **238 MiB**. [Log](project-tests.log)
- **470 paired upstream distance cases**, zero changed outcomes (464 passed,
  six skipped in each mode); peak **154 MiB**. [Log](upstream.log)
- Ruff checks and formatting passed for changed Python files.
- Numerical jobs used a 512 MiB process-tree cap. [Resource records](resources.json)

## Reproduction

From the repository root with its development environment active:

```sh
export PYTHONPATH=python FASTSCIPY_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python bench/runs/2026-10-25-small-cdist/probe.py
python bench/runs/2026-10-25-small-cdist/fused_probe.py
python -m pytest tests/test_cdist_dispatch.py tests/test_spatial.py tests/test_spatial_dispatch.py -q
python tools/scipy_suite.py --pyargs scipy.spatial.tests.test_distance
```

`probe.py` reconstructs the rejected patch in a temporary directory using Git;
it does not modify the checkout. The fused diagnostic uses only its own local
wrapper. Each script writes its JSON results alongside itself. Run timing jobs
serially on an otherwise quiet machine.
