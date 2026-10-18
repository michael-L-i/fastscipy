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
