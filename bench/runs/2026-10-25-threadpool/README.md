# Preserve the Python pool across unchanged thread settings

Calling `set_num_threads(2)` used to shut down the Python executor even when
its budget was already two. The next threaded call then paid worker startup
again. Repeating a setting now preserves the warm executor.

A separate lifecycle lock serializes executor creation and task submission
with budget changes. It is released before waiting for results, so a worker
can change the budget and make a nested call without waiting for itself.
Already submitted work finishes on its original executor; a later parallel
call creates the replacement. Invalid settings leave the executor and budget
intact. The lifecycle lock is reset after fork alongside the existing pool
and busy lock.

## Measurements

The probe isolates coordination overhead: eight integer `abs` tasks per call,
with and without repeating the same thread setting. It does not measure a
numerical kernel and does not establish an application-level speedup.
Each process uses two workers, ten warmup calls, and seven batches of 100
calls. Every result is checked. Raw batch measurements are retained in
`baseline.json` and `candidate.json`.

Baseline Python source is commit `8ea95b2`. Both processes use the same freshly built native
extension from that commit, Python 3.13.6, and macOS 26.6.2 ARM64. Local measurements are not cross-platform
or published-wheel claims.

| Operation | Before, median µs | After, median µs |
|---|---:|---:|
| Warm `parallel_map` | 32.33 | 31.97 |
| Repeat `set_num_threads(2)` then `parallel_map` | 111.70 | 32.74 |

Repeating the setting plus a small batch was about 3.4× faster in this local
probe because it avoided worker startup. The warm-map control changed little;
small timing differences there should be treated as measurement variation.

## Validation

- Fresh-baseline native build: **39 core and lifecycle tests passed**, including
  the all-component fork smoke test.
- Shared-pool consumers: **792 special-function and spatial tests passed**.
- Ruff check, Ruff formatting, and `git diff --check` passed.


Deterministic concurrency tests force a budget change while task submission
is in progress. Further tests cover unchanged-budget reuse, invalid settings,
a worker changing its budget while making nested calls, and fork with the
lifecycle lock held. The worker deadlock regression runs in a subprocess with
a timeout so a failure cannot hang fixture teardown. Existing core tests
exercise exception draining and forked calls across all components.

## Reproduction

From the repository root with a current native development build:

```sh
export PYTHONPATH=python FASTSCIPY_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
git show 8ea95b2:python/fastscipy/_threads.py > /tmp/fastscipy-threads-baseline.py
python bench/runs/2026-10-25-threadpool/probe.py --source /tmp/fastscipy-threads-baseline.py
python bench/runs/2026-10-25-threadpool/probe.py
python -m pytest tests/test_threadpool_lifecycle.py tests/test_core.py tests/test_special.py tests/test_spatial.py tests/test_spatial_dispatch.py -q
```
