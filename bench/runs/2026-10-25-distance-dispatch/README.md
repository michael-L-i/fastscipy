# Conservative two-thread dispatch for low-dimensional pdist

For large, unweighted Euclidean, squared-Euclidean and city-block `pdist`
problems with one to three coordinates, the two-thread tiled path was slower
than SciPy in the local measurements. The candidate declines those calls before
conversion or tile allocation. SciPy computes the result on the original input.

This is a dispatch correction, not a faster distance kernel. The existing small
float64 direct path remains enabled. Four-thread calls, weighted metrics, other
metrics, inputs with four or more coordinates, and `cdist` retain their existing
dispatch. No native code, distance arithmetic, statistical preparation or public
API changes.

## Evidence and scope

The initial [240-case sweep](discovery.json) compared untouched SciPy with the
existing implementation across 500/1500/3000 rows, 1/3/8/16/32 coordinates, four
metrics, both distance APIs, and two/four threads. It identified two distinct
costs: cheap low-dimensional `pdist` pays for rectangular distance tiles and
condensed-output copies, while small threaded problems can lose to handoff
costs. `cdist` and expensive Minkowski distances generally benefited from the
existing split. The discovery sweep uses three repeated calls per implementation
and is directional evidence, not a precision comparison.

The implemented policy addresses only the consistently unprofitable two-thread,
one-to-three-coordinate cheap metrics. A general size threshold would sacrifice
measured four-thread gains and depend more strongly on hardware. Some untouched
workloads still lose to SciPy, including the four-coordinate Euclidean control;
this PR does not claim to solve every distance-dispatch crossover.

[Paired measurements](paired-results.json) use the normal counted wrappers,
untouched SciPy callables, nine repeated calls, and rotating implementation order
on identical inputs. Mac ARM64, CPython 3.13.6, NumPy 2.5.4, SciPy 1.18.1;
BLAS/OpenMP limited to one. Source and native hashes are retained in the JSON.

| Input, two threads | Old fastscipy ms | Candidate ms | Old / candidate |
|---|---:|---:|---:|
| Euclidean 1500×1 | 1.032 | 0.518 | 1.99× |
| Squared Euclidean 1500×1 | 1.009 | 0.443 | 2.28× |
| City-block 1500×3 | 1.234 | 0.795 | 1.55× |
| Euclidean 4000×3 | 6.093 | 5.374 | 1.13× |
| City-block 4000×3 | 6.314 | 5.698 | 1.11× |

Across the 24 affected contiguous/strided/float32 cases, candidate speed was
1.11–2.28× the previous implementation. Ratios against genuine SciPy were
0.94–1.02× (mostly 0.98–1.02×), consistent with delegating to SciPy plus wrapper
cost and timing variation. All 31 comparison cases were byte-identical. Four-
thread 3000×3 controls retained 1.22–1.27× SciPy speed; expensive Minkowski
controls retained 1.62–1.67×. These are local measurements, not cross-platform
performance guarantees. The paired probe peaked at 213 MiB sampled process-tree
RSS under a 512 MiB cap.

## Validation

- **787 spatial project tests passed**, including 39 new dispatch tests; sampled
  process-tree peak RSS 235 MiB. New tests check no fast hit for declined calls,
  original exception delivery exactly once, unchanged inputs, no preparation
  before decline, NaN/inf, float32 and strided inputs, zero-coordinate inputs,
  and preservation of small/four-thread/weighted/expensive/four-coordinate paths.
- Existing `pdist euclidean 4000x3` keeps its name and input but is now explicitly
  a control. Squared-Euclidean and city-block controls use that same fixture.
  These are controls for the standard two-thread report; four-thread use may
  still take the fast path, separately enforced by the project tests.
- **470 paired upstream distance tests**, zero changed outcomes (464 passed,
  six skipped in each mode); 154 MiB peak. [Log](upstream.log)
- **39 spatial benchmark checks passed**, including byte identity, expected
  fast paths and the unchanged one-thread regression threshold; 537 MiB peak
  under a 768 MiB cap. [Log](benchmark.log) / [JSON](benchmark/spatial.json).
  This existing runner uses disabled wrappers for its reference; genuine-SciPy
  timing claims above come from the separate paired probe.
- Ruff checks, formatting and `git diff --check` passed. Monitor JSON files
  beside each log retain timing and resource-limit results.

## Reproduction

```sh
export PYTHONPATH=python FASTSCIPY_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
git show a5961a5:python/fastscipy/_spatial.py > /tmp/distance-baseline.py
python bench/runs/2026-10-25-distance-dispatch/probe.py --baseline-source /tmp/distance-baseline.py --output /tmp/paired-distance.json
python -m pytest tests/test_spatial.py tests/test_spatial_dispatch.py tests/test_spatial_thread_dispatch.py -q
python tools/scipy_suite.py --pyargs scipy.spatial.tests.test_distance
python bench/run.py spatial --budget 0.03 --check --output-dir /tmp/distance-bench
```

The broad discovery script measures whichever implementation is checked out;
run it on baseline `a5961a5` to reproduce the discovery dataset. Keep timing
processes serial and numerical thread limits explicit.
