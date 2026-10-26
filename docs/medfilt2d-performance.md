# Integer signal median filtering

`signal.medfilt2d` previously parallelized SciPy's selection kernel over row
bands. For uint8 images it now reuses the existing ndimage rank kernels:
selection networks for small windows and sliding histograms where profitable.
The median integer value is independent of tie selection order. Float inputs
retain signal's own selection to preserve NaN and signed-zero behavior.

The public adapter accepts exact uint8 ndarrays with at least 4096 pixels and
32 pixels per axis, and positive odd integer window dimensions up to 31.
The 1x1 window is declined. Output is a fresh C-contiguous array; noncontiguous
and read-only inputs are supported without mutation. Padding is constant zero,
as in SciPy signal. Unsupported calls retain the previous fallback paths.
Using the signal component alone is sufficient; ndimage need not be patched.

## Measured public calls

Measured on Apple M5 Pro, macOS arm64, Python 3.13.6, NumPy 2.5.4,
SciPy 1.18.1. Numerical thread budget: two, with BLAS/OpenMP limited to one.
Native source SHA-256:
`5f9095417fd0f5fa4bfdd5ef4364c8ee926d6379342c5b9edfd7aa630023fdc4`.
The Rust binary is unchanged from baseline main `95c4304`.

The benchmark runner constructs independent fixtures in genuinely unpatched
SciPy and patched contexts, checks byte identity, and reports medians of
calibrated timing batches. The focused run used a 2 GiB process-tree limit,
peaked at 412 MiB, and completed in 15.8 seconds. These are local measurements,
not performance guarantees for other processors or image distributions.

| Image/window | SciPy ms | 1 thread ms | 2 threads ms | 2-thread speedup |
|---|---:|---:|---:|---:|
| 512x512 / 3x3 | 12.065 | 0.035 | 0.035 | 348x |
| 512x512 / 5x5 | 44.193 | 0.141 | 0.140 | 315x |
| 512x512 / 7x7 | 81.146 | 0.364 | 0.364 | 223x |
| 512x512 / 15x15 | 328.346 | 4.665 | 2.463 | 133x |
| 512x512 / 3x5 | 27.229 | 0.298 | 0.297 | 92x |
| Strided 256x256 / 5x5 | 11.000 | 0.056 | 0.056 | 196x |

All six outputs were byte-identical at one and two threads. Raw results are
in [the focused benchmark report](../bench/results/signal-medfilt2d.json).
Reproduce with `python bench/run.py signal --only medfilt2d --check --budget 0.08`.
The initial exploration also compared with the previous fastscipy threading
path: for 512x512 images, 3/5/7/15 square windows took approximately
9/41/44/193 ms, respectively.

## Boundaries and rejected scope

A boundary sweep used 4096-pixel images (64x64, 128x32, 32x128), square windows
3/5/7/9/15/31, and rectangular windows 1x31, 31x1, 3x31, 31x3 and 7x15.
Accepted cases improved approximately 6–90x. An earlier broader prototype
regressed on 4096x1 columns: interpreted rank planning/padding dominated short
rows, with some windows tens of times slower. Images narrower than 32 pixels
on either axis now decline, including oversized-window cases. The 1x1 window
also declines because it has too little work to justify this dispatch.

Tests compare results, warnings, exceptions and input mutation; cover C,
Fortran, reversed, strided and read-only layouts; and verify unsigned, huge,
ragged and invalid kernels decline. A foreign kernel entry is declined before
conversion, avoiding additional user callbacks. A separate subprocess checks
signal-only patching, fast-path use and unpatching on one and two threads.

Further candidates were inspected but not implemented: caching custom rank
selection networks (`NdKernel::nets` in `src/ndimage/rank.rs`) could reduce
repeated setup, but needs bounded concurrent cache accounting; avoiding output
concatenation in other signal threading paths is more limited and must preserve
SciPy's output strides. Neither justified additional complexity in this round.

Final validation: 722 project signal tests passed (35.4 seconds, 520 MiB peak
under a 1 GiB cap). The complete paired SciPy signal suite shared 15,907 test
outcomes with zero changes, missing tests or incomplete pairs (26.4 seconds,
429 MiB peak); the patched run recorded 127,407 fast-path calls and zero
fast-path errors. Repository-wide Ruff lint and formatting checks passed.
The first broad project run exceeded a 512 MiB cap at 515 MiB and was stopped;
the explicitly larger 1 GiB rerun completed successfully.
