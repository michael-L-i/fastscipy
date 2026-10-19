# Benchmarks

```sh
python bench/run.py                     # every area
python bench/run.py ndimage stats       # some areas
python bench/run.py --only median       # cases whose name contains "median"
python bench/run.py --check             # fail on any difference, missing fast path or slowdown
python bench/plot.py                    # redraw the charts in docs/ from bench/results/*.json
```

The published numbers use a release build (`maturin develop --release`).

## Method

Each `bench_<area>.py` yields cases: a name and a function making one SciPy
call at a realistic size. `run.py` times every case three ways in one
process:

1. plain SciPy, inside `fastscipy.disabled()`;
2. fastscipy on 1 thread;
3. fastscipy on all threads (`FASTSCIPY_NUM_THREADS`, default all cores).

A time is the best of repeated calls within a budget of 0.3 s (at least 3
calls), after one warm-up call. Every case checks that the patched result is
byte-identical to SciPy's, and that the fast path actually ran. Cases marked
`False` (controls: small calls or already-ordered queries the fast path
should hand back to SciPy) only have to be no slower. Areas whose fast paths
only add threads (`special`, `spatial`) only need a fast path with threads.

Per-area figures are geometric means over the non-control cases. Speedups
are SciPy's time divided by fastscipy's.

## Published results

[`RESULTS.md`](RESULTS.md) and `results/*.json` come from one run on
2026-10-18:

- Apple M5 Pro (6 Super + 12 Performance cores), 48 GB, macOS 26
- Python 3.13, NumPy 2.5.4, SciPy 1.18.1 (Accelerate BLAS/LAPACK)
- release build (thin LTO), 18 threads
- one job at a time on an otherwise idle machine

Timings on other machines will differ; on fewer cores the multithreaded
column shrinks roughly in proportion, while single-thread speedups mostly
carry over. Memory-bound kernels scale less than compute-bound ones.
