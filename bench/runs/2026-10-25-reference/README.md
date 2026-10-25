# True SciPy benchmark references

The runner now reconstructs seeded reference and candidate fixtures in separate
module namespaces, advances them in their respective patch states, and measures
fully unpatched SciPy. Captured functions and `cKDTree` subclasses therefore do
not survive into the baseline. A counter check rejects contaminated references.
Timing uses median calibrated batches, with at least five measured batches.
Controls remain checked and reported, but do not enter headline means.

Validation used macOS 26.6.2 ARM64, Python 3.13.6, NumPy 2.5.4 and SciPy
1.18.1, with the existing native build and Python-only runner changes, two
native threads, and BLAS/OpenMP limited to one. No kernels were changed. The two
area sweeps used `--budget 0.01`; selected controls were then measured three more
times with `--budget 0.1`. Historical disabled-wrapper/minimum-time reports are
not directly comparable to these results.

- Ten focused tests pass, including captured-function/object isolation,
  independent module-level random generators, restoration after exceptions,
  contaminated-reference rejection, timing calibration/median, and control
  exclusion without dropping regression checks.
- FFT: 36/36 byte-identical with expected fast-path coverage. `--check` fails
  the integer `dctn` control: 3.412 µs unpatched versus 3.986 µs patched at one
  thread (0.856×).
- Sparse linear algebra: 31/31 byte-identical with expected fast-path coverage.
  `--check` fails the batched operator control: 0.183 ms unpatched versus
  0.204 ms patched for 100 calls (0.898×).
- Repeats confirm overhead but straddle the existing 0.9× regression threshold.
  Integer `dctn` adds 0.272–0.459 µs at one thread (0.879–0.927×); the batched
  operator adds 0.013–0.022 ms per 100 calls (0.892–0.937×). One of three repeats
  fails for each control. Neither threshold nor kernels were changed.

These checks establish runner behavior on the selected cases, not a performance
improvement or cross-platform certification. The stricter baseline exposes
existing fallback overhead and timing variation. Keep the PR in draft until
these downstream check failures have been addressed or explicitly accepted.

Reproduce a sweep from the repository root:

```sh
export FASTSCIPY_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m pytest tests/test_bench_reference.py -q
python bench/run.py fft --budget 0.01 --check --output-dir /tmp/true-reference-fft
python bench/run.py sparse_linalg --budget 0.01 --check --output-dir /tmp/true-reference-sparse
```

Use `--only 'dctn integer control'` or `--only 'batched fallback control'` and
`--budget 0.1` to repeat the affected controls. Only current fixture pairs are
retained, but each side owns its inputs: run one area per process to bound memory.
Raw area rows and repeated-control rows are stored beside this note.
