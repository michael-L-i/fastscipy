# Benchmarks

All speedups compare fastscipy with unpatched SciPy on the same input. Timings
are hardware-dependent; byte identity and expected fast-path use are checked
alongside performance.

## Run the suite

Use a release build (`maturin develop --release`):

```sh
python bench/run.py                     # every area
python bench/run.py ndimage stats       # selected areas
python bench/run.py --only median       # matching case names
python bench/run.py --check             # enforce correctness and performance checks
FASTSCIPY_NUM_THREADS=2 python bench/run.py signal --check
```

Each `bench_<area>.py` yields a name and a callable. The runner times:

1. Plain SciPy inside `fastscipy.disabled()`.
2. fastscipy on one thread.
3. fastscipy on the configured thread budget (all cores by default).

A time is the best of repeated calls within a 0.3-second budget, with at least
three calls after warmup. Results must be byte-identical, and accelerated cases
must use a fast path. Declined controls have a separate performance check;
thread-only areas (`special`, `spatial`) require fast-path use with threads.
Published per-area averages are geometric means over non-control cases.

Run one benchmark process at a time. For a smaller resource budget, limit native
threads and set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`VECLIB_MAXIMUM_THREADS=1`.

## Published broad run · 2026-10-18

[Full table](RESULTS.md) · [Raw per-area data](results/)

- Apple M5 Pro, 18 cores (6 Super + 12 Performance), 48 GB, macOS 26
- Python 3.13, NumPy 2.5.4, SciPy 1.18.1, Accelerate BLAS/LAPACK
- Release build with thin LTO; 1 and 18 threads
- One benchmark at a time

The root README summarizes this run. More cores do not imply proportional gains:
threading overhead, memory bandwidth, and work delegated to SciPy vary by case.

## Expansion sweep · 2026-10-19

[Summary and validation](runs/2026-10-19-sweep/summary.json) ·
[All final measurements](runs/2026-10-19-sweep/final-checks.json) ·
[Candidate audit](runs/2026-10-19-sweep/candidate-audit.json)

These **117 cases** exercise new variants and existing controls across 13 areas.
They are a separate sample from the broad run, with **1 and 2 native threads**,
BLAS limited to one thread, serial measurement, and a 0.1-second timing budget.
The geometric means below include all retained cases, including controls.

| Area | Cases | 1 thread | 2 threads |
|---|---:|---:|---:|
| linalg | 24 | 1.70× | 1.70× |
| ndimage | 2 | 3.18× | 3.19× |
| interpolate | 14 | 1.34× | 1.35× |
| csgraph | 4 | 1.24× | 1.25× |
| signal | 15 | 2.44× | 2.45× |
| sparse | 6 | 1.77× | 1.79× |
| sparse linear algebra | 12 | 1.29× | 1.30× |
| stats | 6 | 7.04× | 7.04× |
| special | 3 | 1.00× | 1.85× |
| spatial | 6 | 1.00× | 1.59× |
| integrate | 16 | 2.12× | 2.13× |
| optimize | 5 | 1.42× | 1.43× |
| cluster | 4 | 1.19× | 1.20× |

Every retained case produced byte-identical output and used the expected fast
path. Peak benchmark RSS was 265 MB; the full project test run passed 6,929 tests
with 57 warnings and used 525 MB peak RSS with zero swaps.

These are ratios versus plain SciPy, not incremental ratios versus the previous
fastscipy revision. Individual area artifacts retain incremental comparisons
where measured, rejected prototypes, and known costs. Three older float64 signal
controls measured 3–8% slower after safety and dtype guards; absolute overhead
was 0.06–0.72 microseconds. The audit separates measured rejections from ideas
that still need profiling; neither function counts nor these cases measure a
percentage of all SciPy workloads accelerated.

## Runnable visual demo

[Image-cleanup example](../examples/median_filter.py) applies a 3×3 median filter
to eight synthetic noisy images. It needs no data download and defaults to one
thread. Both implementations are warmed on the same shape, then run a sequential
batch. Timers include filtering and output allocation, and exclude input
generation, validation, and GIF rendering.

```sh
python examples/median_filter.py                    # print timings and validate bytes
python examples/median_filter.py --threads 2        # choose a thread budget
pip install "pillow>=10.1"                          # optional rendering dependency
python examples/median_filter.py --gif docs/median-filter-demo.gif
```

Run these commands from the repository root with fastscipy installed. The final
command saves both the GIF and a JSON record of per-image completion times,
versions, and replay speed. The animation uses a shared clock, slowed equally
for both implementations; it is a replay of measurements, not a screen capture.
The committed example uses about 192 MB peak RSS including GIF rendering.
