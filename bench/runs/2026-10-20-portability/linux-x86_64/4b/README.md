# Linux x86_64 checkpoint: `4b1342d`

[Run 39368833274](https://github.com/michael-L-i/fastscipy/actions/runs/39368833274)
built revision `4b1342d3694cd166ceb0097c5a169ced181f0847`, fastscipy `0.1.0a2`,
on glibc Linux x86_64 with CPython 3.13.16, NumPy 2.5.4, and SciPy 1.18.1.
The numerical thread budget was two, with one BLAS thread.

The unchanged [summary](summary.json) records 7,060 passing project tests and
four failures in the new uniform-filter NaN initialization checks: axis 0,
seven or nine columns, on one and four threads. All 407 benchmark cases
across 13 areas matched SciPy and met their expected fast-path checks. The
[case reports](cases/), [build resource report](build.json), and
[provenance](provenance.json) retain the original measurements. Compilation
took 54 minutes 35 seconds and peaked at 4.39 GiB RSS.

The ordered running-sum addition restored float32/float64 line-cast
calibration, removing extra SciPy conversion passes. The later `b4710e6`
change also orders initialization additions; validation of the latest
`675bd52` source, including paired upstream tests, is pending in
[run 39374986938](https://github.com/michael-L-i/fastscipy/actions/runs/39374986938).

## Incremental float measurements

The earlier [run 39362357649](https://github.com/michael-L-i/fastscipy/actions/runs/39362357649)
used revision `5914b773d9e644b6caf700a70cc88dcb6abdfc39` and fastscipy
`0.1.0a1`. Its unchanged [ndimage cases](baseline-5914/ndimage.json) and
[summary](baseline-5914/summary.json) are retained alongside the
[new ndimage cases](cases/ndimage.json). The older summary predates revision
metadata; the source revision comes from that run's checkout log.
These are separate hosted-run measurements; the raw SciPy baselines are
included to show run-to-run variation. Paired values below are one/two threads.

| Case | `5914` SciPy ms | `5914` fast ms | `4b` SciPy ms | `4b` fast ms | Speedup: `5914` → `4b` |
|---|---:|---:|---:|---:|---:|
| Gaussian f64, 2048², sigma 2 | 126.804 | 121.481 / 112.994 | 111.605 | 23.227 / 12.758 | 1.04 / 1.12× → 4.81 / 8.75× |
| Gaussian f32, 2048², sigma 2 | 87.214 | 129.765 / 94.910 | 87.564 | 18.717 / 10.104 | 0.67 / 0.92× → 4.68 / 8.67× |
| Gaussian f32, 256³, sigma 1 | 362.540 | 370.850 / 344.242 | 457.572 | 85.466 / 43.321 | 0.98 / 1.05× → 5.35 / 10.56× |
| Uniform f64, 2048², size 5 | 109.403 | 115.161 / 106.573 | 84.608 | 14.984 / 8.216 | 0.95 / 1.03× → 5.65 / 10.30× |
| Sobel f32, 2048² | 59.875 | 113.489 / 96.038 | 65.944 | 8.052 / 4.601 | 0.53 / 0.62× → 8.19 / 14.33× |

Across all 43 ndimage cases, geometric-mean speedups changed from
6.57 / 10.50× to 7.42 / 12.87×. This checkpoint retains its four project-test
failures and does not certify the later source or a final release wheel.
