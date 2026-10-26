# Genuine-baseline refresh after fallback improvements

PR #7 now incorporates current main and depends on the focused LinearOperator
fallback change in PR #13. Its runner still measures genuinely unpatched SciPy
with independent fixtures and median batch timings. No regression threshold,
identity requirement, or expected-fast-path check was relaxed.

## Results

All **475 cases across 14 areas** passed byte identity, expected-fast-path checks,
and the unchanged **0.9x one-thread regression threshold** in the completed runs.
The final rows are in [final-cases](final-cases/); [manifest.json](manifest.json)
records the tested source and native artifact. This is local macOS ARM64,
Python 3.13.6 / NumPy 2.5.4 / SciPy 1.18.1 validation, not a new platform or
published-wheel performance claim. Native source was unchanged from the verified
baseline build. Each area ran in a separate process with two numerical threads
and BLAS/OpenMP limited to one.

The initial pass used a 0.03-second sampling budget and a 1 GiB process-tree cap.
Ten areas completed successfully. FFT retained identical results and paths but
its FFTPACK resize control measured 0.881x. A separate full FFT run with a
0.1-second budget passed all 36 cases; that control measured 0.924x. Both runs
are retained. Tiny fallback ratios remain noise-sensitive, so a local pass does
not establish a performance margin on every machine.

Three areas (ndimage, signal, sparse) were stopped by the initial memory cap.
Independent reference/candidate fixtures increase benchmark memory. They were
rerun successfully under a bounded 2 GiB cap, with sampled peaks of 1,165,
1,709 and 1,406 MiB respectively. These are validation-process peaks, not the
memory consumption of a single patched call. No incomplete run is counted as
passing. Initial logs/resources and available case JSON are in [initial](initial/);
rerun logs/resources are in [rechecks](rechecks/).

The earlier integer-DCT and LinearOperator controls also passed nine longer
repeated checks on the dependency branch; see the [fallback report](../2026-10-26-fallback/README.md).
The sparse dispatch improvement is scoped to PR #13; no FFT production change
is attributed to this benchmark PR. Historical results using disabled wrappers
and minimum-call timings remain historical and should not be compared directly
to the new method.

## Reproduction

Run one area per process with a matching native extension:

```sh
export PYTHONPATH=python FASTSCIPY_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python bench/run.py AREA --budget 0.03 --check --output-dir /tmp/reference-AREA
python bench/run.py fft --budget 0.1 --check --output-dir /tmp/reference-fft
```

The sampling budget is not a timeout: each timing series collects at least five
batches, including for slow calls. Allow up to 2 GiB for the large fixture areas
and retain process-tree monitoring when reproducing the full sweep.
