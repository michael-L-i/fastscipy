# fastscipy

**Faster SciPy calls. SciPy-compatible results.**

Portable Rust kernels and shared thread pools accelerate supported calls.
Everything else stays with SciPy.

```sh
pip install --pre fastscipy
```

```python
import fastscipy

fastscipy.patch()

from scipy import ndimage

clean = ndimage.median_filter(image, size=3)
```

![Image cleanup with SciPy and fastscipy](https://raw.githubusercontent.com/michael-L-i/fastscipy/main/docs/median-filter-demo.gif)

*Demo recorded on a MacBook Pro with M5 Pro.*

[Explore eight more examples →](https://github.com/michael-L-i/fastscipy/blob/main/examples/README.md)

## Install and use

Prebuilt wheels for **Python 3.12–3.15 · SciPy 1.18**; no Rust required.
Supports **x86_64 and ARM64** on Linux (glibc 2.28+ or musl 1.2+), macOS 14+,
and Windows. [Platform details and measurements →](https://github.com/michael-L-i/fastscipy/blob/main/docs/platforms.md)

Call `patch()` before importing libraries that use SciPy.
Use `patch(threads=4)` to choose a thread budget, `status()` to see what's enabled,
and `unpatch()` to restore SciPy.

## Performance across platforms

Geometric-mean speedups at **two threads**, with BLAS limited to one.
These portability checkpoints use Python 3.13, NumPy 2.5.4, and SciPy 1.18.1;
each comparison uses the same machine and inputs.

| Platform | ndimage | stats | linalg |
|---|---:|---:|---:|
| Linux x86_64 · glibc | 15.1× | 5.9× | 3.9× |
| Linux ARM64 · glibc | 13.2× | 6.5× | 4.0× |
| Linux x86_64 · musl | 12.3× | 6.0× | 5.1× |
| Linux ARM64 · musl | 11.8× | 6.4× | 4.0× |
| macOS Apple Silicon | 16.9× | 5.9× | 4.8× |
| macOS Intel | 13.0× | 5.7× | 6.5× |
| Windows x86_64 | 11.7× | 5.9× | 4.7× |
| Windows ARM64 | 11.7× | 6.4× | 4.2× |

The suite covers **407 cases across 13 areas**; a case is one function and
input combination. The three columns summarize 43 image-processing,
58 statistics, and 49 linear-algebra cases. Results vary by workload,
hardware, and SciPy build.
[Platform measurements and limitations](https://github.com/michael-L-i/fastscipy/blob/main/docs/platforms.md) · [Original M5 Pro benchmarks](https://github.com/michael-L-i/fastscipy/blob/main/bench/RESULTS.md)

## More

[Coverage](https://github.com/michael-L-i/fastscipy/blob/main/docs/coverage.md) · [How it works](https://github.com/michael-L-i/fastscipy/blob/main/docs/design.md) ·
[Demo source](https://github.com/michael-L-i/fastscipy/blob/main/examples/median_filter.py) · [Contributing](https://github.com/michael-L-i/fastscipy/blob/main/CONTRIBUTING.md)

BSD-3-Clause. SciPy ports retain [SciPy's license](https://github.com/michael-L-i/fastscipy/blob/main/LICENSES/scipy-LICENSE.txt).
Independent of the SciPy project.
