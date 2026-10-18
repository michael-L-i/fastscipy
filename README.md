# fastscipy

Drop-in acceleration for SciPy. Same results, byte for byte. Faster on one core
and on all of them.

```python
import fastscipy
fastscipy.patch()          # SciPy calls made after this are faster

import scipy.ndimage       # ...including calls made by scikit-image and others
```

`patch()` swaps SciPy internals for Rust kernels and multithreaded wrappers.
Every fast path returns exactly what SciPy returns. Any call outside what a
fast path covers (an unusual dtype, option or layout) goes to SciPy's own code
unchanged.

| | |
|---|---|
| `fastscipy.patch(components=None, threads=None)` | patch all components, or some; returns `status()` |
| `fastscipy.unpatch()` | restore SciPy |
| `fastscipy.set_num_threads(n)` | thread budget (default: all cores, or `FASTSCIPY_NUM_THREADS`) |
| `fastscipy.disabled()` | context manager that runs SciPy's own code |
| `fastscipy.stats()` | calls per function that took the fast path or fell back |

Call `patch()` before importing libraries that bind SciPy functions at import
time (`from scipy.special import expit`), or they keep the original.

## Results

219 benchmark cases, every one byte-identical to SciPy 1.18.1. Geometric-mean
speedup per area on an 18-core Apple M5 Pro (full table: `bench/RESULTS.md`):

| Area | 1 thread | 18 threads | How | Highlights (1T / 18T) |
|---|---:|---:|---|---|
| ndimage | 13.3× | 68× | Rust kernels (fastndimage), labelling, morphology, measurements | median 3×3 u8 496× / 1240×, `generic_filter(np.mean)` 244× / 677× |
| linalg | 10.0× | 76× | batched small-matrix LAPACK calls without Python loops | `solve_triangular` 10k×3×3 147× / 586× |
| interpolate | 9.1× | 23× | query reordering, Rust spline and pchip kernels | RGI pchip 659× / 732×, `find_simplex` 69× / 160× |
| stats | 7.0× | 9.3× | lean small-call paths, Rust `rankdata`, discrete tables | `ttest_ind` n=10 9×, `zipf.cdf` 546× |
| csgraph | 5.8× | 15.5× | Rust shortest paths and components | `bellman_ford` 311× / 2435× |
| signal | 4.7× | 12.4× | small-call paths, channel threading, Rust kernels | direct 2-D convolve 49× / 260×, `dlsim` 105× |
| sparse | 4.7× | 11.9× | Rust argmax, `sum_duplicates`, dense→sparse, `find` | `sum_duplicates` 2e6 25× / 73× |
| special | 1.0× | 6.6× | threads over SciPy's own ufunc loops (large arrays) | `jv`, `betainc` 1e7 13–16× on 18 threads |
| spatial | 1.0× | 4.0× | threaded `cdist`/`pdist`, KD-tree queries | |
| integrate, optimize | 1.6×, 1.1× | same | less Python overhead (`trapezoid`, numerical gradients) | |

Across all cases: 5.2× on one thread and 17.5× on 18 (geometric mean); 102
cases are at least 5× faster on one thread and 126 at least 10× on 18. Small
calls a fast path does not cover are not slower.

SciPy's full test suite (about 88,000 tests) with every component patched in
at once gives the same outcome, test by test, as plain SciPy: the same 3
failures (Accelerate-related, also unpatched), no new ones, and 1.86 million
fast-path calls with 0 fast-path errors. The only differences are the 684
tests described under "Known differences" and SciPy's own flaky
`test_sepfir2d_strided_3` (an out-of-bounds read in code we don't touch).

**Not accelerated:** dense linear algebra, FFT, `linprog`, iterative sparse
solvers, `cluster` and `io` (already near the hardware or
I/O-bound), and solvers whose time is spent in a Python callback (`minimize`,
`solve_ivp`, `quad`). Only macOS arm64 with SciPy 1.18 has been validated;
every component refuses other SciPy versions.

## Known differences

- `isinstance(scipy.special.erf, np.ufunc)` is False while patched (the
  wrappers forward `nin`, `types`, `reduce`, ...). 684 SciPy tests that check
  this are not collected under the patch.
- `scipy.spatial.cKDTree` is a subclass while patched; `isinstance` and
  pickling behave as before.

## Development

```sh
uv venv -p 3.13 .venv && uv pip install --python .venv/bin/python numpy scipy maturin pytest pytest-xdist
.venv/bin/maturin develop --profile dev-release      # fast rebuild; `--release` for real numbers
.venv/bin/python -m pytest tests                     # identity tests
.venv/bin/python bench/run.py                        # SciPy vs fastscipy, 1 thread and all threads
.venv/bin/python -m pytest -p fastscipy.pytest_plugin --pyargs scipy.ndimage -n 2   # SciPy's own suite, patched
```

## Layout

- `python/fastscipy/_<area>.py`: one component per SciPy area. `patch()` installs it; `smoke()` exercises every fast path.
- `src/<area>/`: its Rust kernels, exposed as `fastscipy._native.<area>`. `src/pool.rs` is the one thread pool.
- `tests/test_<area>.py`: byte-identity tests against SciPy. `bench/bench_<area>.py`: benchmark cases.

## License

BSD-3-Clause. Parts are ports of SciPy code; see `LICENSES/scipy-LICENSE.txt`.
