# fastscipy

**Faster SciPy calls. The same results, byte for byte.**

fastscipy accelerates supported SciPy calls with Rust kernels, less Python
overhead, and shared thread pools. One call enables it for the current process;
unsupported inputs and options use SciPy's original implementation.

```python
import fastscipy
fastscipy.patch()

from scipy import ndimage
clean = ndimage.median_filter(image, size=3)
```

**Experimental · macOS Apple Silicon · SciPy 1.18 · Python 3.12–3.14**

[Install](#install) · [Demo](#see-it-run) · [Performance](#performance) ·
[Coverage](docs/coverage.md) · [Design](docs/design.md) ·
[Contributing](CONTRIBUTING.md)

## Install

Not on PyPI yet. With a [Rust toolchain](https://rustup.rs) installed:

```sh
pip install git+https://github.com/michael-L-i/fastscipy
```

Call `patch()` before importing libraries that bind SciPy functions. Check what
is enabled, choose a thread budget, or temporarily run plain SciPy:

```python
fastscipy.patch(threads=4)           # optional; default is all cores
print(fastscipy.status())           # each component reports "on" or a reason

with fastscipy.disabled():
    reference = ndimage.median_filter(image, size=3)

fastscipy.unpatch()                 # restore SciPy
```

Use `patch(components=["ndimage"])` to enable a single area, or set
`FASTSCIPY_NUM_THREADS` to control the default thread budget. Libraries that call
patched SciPy functions can benefit too; existing `from scipy.x import f`
bindings made before patching retain their original function.

## See it run

A common task: remove salt-and-pepper noise from **eight 1024 × 1024 images**.
Both implementations run the same `ndimage.median_filter(image, size=3)` call.

![Measured image-cleanup demo: fastscipy finishes the batch while SciPy is still processing, with identical output](docs/median-filter-demo.gif)

| Image-cleanup batch | SciPy | fastscipy, 1 thread | Speedup |
|---|---:|---:|---:|
| Eight images, 3 × 3 median filter | 213.39 ms | 1.57 ms | **135.7×** |

This is a measured example on an Apple M5 Pro, not a suite average. Both paths
are warmed first; image generation, comparison, and rendering are excluded.
All eight outputs are byte-identical. The GIF replays measured completion times
**16.9× slower** on a shared clock so the difference is visible.
[Raw timings](docs/median-filter-demo.json) · [Runnable source](examples/median_filter.py)

Run it locally; add Pillow only if you want to generate the GIF:

```sh
python examples/median_filter.py
pip install "pillow>=10.1"
python examples/median_filter.py --gif docs/median-filter-demo.gif
```

## Performance

The published broad benchmark run covers these areas. Values are geometric-mean
speedups over unpatched SciPy; the single-thread column includes algorithm and
dispatch improvements, while the 18-thread column also includes parallelism.

| Area | Cases | 1 thread | 18 threads |
|---|---:|---:|---:|
| linalg — stacks of small matrices | 19 | 10.0× | 75.9× |
| ndimage | 41 | 13.3× | 68.2× |
| interpolate | 12 | 15.6× | 44.3× |
| csgraph | 9 | 5.8× | 15.5× |
| signal | 22 | 4.7× | 12.4× |
| sparse | 16 | 4.7× | 11.9× |
| stats | 46 | 7.0× | 9.3× |
| special | 32 | 1.0× | 7.4× |
| spatial | 8 | 1.0× | 6.7× |
| integrate | 3 | 1.6× | 1.6× |
| optimize | 3 | 1.1× | 1.1× |

Measured on 2026-10-18: Apple M5 Pro, 18 cores, SciPy 1.18.1, Python 3.13,
release build, one benchmark at a time. Each result is checked byte for byte.
Controls are excluded from these averages. [Full published results](bench/RESULTS.md).

A **2026-10-19 expansion sweep** checked another targeted set of **117 cases
across 13 areas**, including controls, at 1 and 2 threads. Highlights include
15.1× for a small multidimensional `sem`, 6.7× for float32 `sosfiltfilt`, and
4.2× for irregular Simpson integration. These are separate workloads and thread
counts, so they are not merged into the table above.
[Latest tables and reproduction](bench/README.md) ·
[Measured sweep results](bench/runs/2026-10-19-sweep/summary.json)

Performance depends on input size, dtype, layout, and hardware. Some paths only
gain from threads; small calls can be dominated by dispatch. The expansion also
measured a 3–8% cost (0.06–0.72 µs) on three older float64 signal controls after
adding dtype and safety guards. The benchmark artifacts retain those tradeoffs.

## Coverage and validation

Fast paths cover image processing, interpolation, small-matrix batches,
statistics, signal processing, sparse operations, graphs, distances, clustering,
and selected integration and optimization helpers. Large special-function calls
use SciPy's own loops on several threads.

[Coverage by function and input](docs/coverage.md) describes the supported cases
and fallbacks. This is not comprehensive SciPy coverage: FFT, large dense
BLAS/LAPACK operations, scientific IO, and arbitrary user callbacks mostly
remain with SciPy.

| Check | Latest recorded result |
|---|---|
| Project test suite | **6,929 passed**, 57 warnings |
| Expansion benchmarks | **117/117 byte-identical**, expected fast paths used |
| Test memory | **525 MB peak**, zero swaps |
| Static checks | Ruff lint and formatting passed |

The expansion includes targeted paired upstream SciPy comparisons. A separate,
earlier full upstream run recorded matching outcomes for the tests collected in
both modes, across a suite of roughly 88,000 tests. That historical run is not a
fresh full-upstream result for every later change. Use
[tools/scipy_suite.py](tools/scipy_suite.py) to reproduce comparisons.

## Compatibility

Validated for **macOS arm64 and SciPy 1.18**. Byte identity depends on the SciPy
wheel's arithmetic, including fused operations and BLAS summation order.
Unsupported platforms or SciPy versions stay unpatched and report why in
`status()`. CI also tracks Linux portability; test-only overrides are described
in [the design notes](docs/design.md).

Two observable API differences while patched:

- Special-function wrappers forward ufunc attributes, but are not `np.ufunc`
  instances. Tests that select functions by that type may not be collected.
- `scipy.spatial.cKDTree` uses a subclass; `isinstance`, `issubclass`, and
  pickling are supported.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, correctness rules, and checks.
Python wrappers live in `python/fastscipy/`, Rust kernels in `src/`, identity tests
in `tests/`, and reproducible workloads in `bench/` and `examples/`.

BSD-3-Clause. Ports of SciPy code retain
[SciPy's license](LICENSES/scipy-LICENSE.txt). This project is independent of SciPy.
