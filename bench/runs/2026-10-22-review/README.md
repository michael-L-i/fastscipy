# Optimization review · 2026-10-22

Five Astra investigations produced four implementation PRs and one audit-only PR. The production changes were validated with **7,934 project tests** and **472 benchmark checks across 14 areas**. Every benchmark output is byte-identical and uses its expected fast path. These are local source measurements, not published-wheel or cross-platform results. PyPI 0.1.0a4 is unchanged.

## What improved

| Change | Versus genuinely unpatched SciPy | Versus previous fastscipy | PR |
|---|---:|---:|---|
| Small Euclidean `pdist`, 4×3 / 16×3 / 32×16 | 4.56× / 4.13× / 2.48× | 5.20× / 4.50× / 2.62× | [#4](https://github.com/michael-L-i/fastscipy/pull/4) |
| Matrix-free Jacobian + action, 10,000 parameters | 7.36× | 7.37× | [#5](https://github.com/michael-L-i/fastscipy/pull/5) |
| Colored sparse Jacobian, 10,000 parameters | 2.30× | 2.29× | [#5](https://github.com/michael-L-i/fastscipy/pull/5) |
| Public sparse `least_squares`, iterative, 10,000 parameters | 1.64× | 1.64× | [#5](https://github.com/michael-L-i/fastscipy/pull/5) |
| Exact-ndarray linear-operator calls | 2.0–2.24× | 1.22–1.46× | [#1](https://github.com/michael-L-i/fastscipy/pull/1) |
| Small 8×8 modern ND FFT transforms | 1.45–1.69× | Similar; this area was unaccelerated | [#3](https://github.com/michael-L-i/fastscipy/pull/3) |

These selected one-thread probes compare true unpatched, previous and candidate states. Exact workloads, raw timings and limitations are in the individual PRs and sibling audit directories. Matrix-free Jacobian examples use SciPy's private `approx_derivative`; the public solver result is listed separately. [Differentiation audit #2](https://github.com/michael-L-i/fastscipy/pull/2) adds compatibility coverage and rejects 1.03–1.10× prototypes. Integration temporary-reuse prototypes were also rejected.

## Expanded category table

Geometric means versus SciPy in the common benchmark runner, which uses `fastscipy.disabled()` for its reference. Disabled wrappers retain dispatch/counter overhead; use the true-unpatched probes above for microsecond-scale dispatch claims. Case counts below include controls; means exclude the 19 controls. All 453 averaged cases are byte-identical.

| Area | Cases | 1 thread | 2 threads |
|---|---:|---:|---:|
| linalg | 49 | 4.18× | 5.16× |
| ndimage | 43 | 12.45× | 20.83× |
| interpolate | 31 | 3.90× | 4.68× |
| csgraph | 16 | 3.64× | 4.29× |
| signal | 41 | 3.30× | 3.92× |
| sparse | 27 | 3.86× | 4.49× |
| sparse linear algebra | 31 | 1.44× | 1.45× |
| stats | 58 | 6.15× | 6.28× |
| special | 37 | 1.00× | 1.84× |
| spatial | 36 | 1.49× | 2.08× |
| integrate | 35 | 1.73× | 1.73× |
| optimize | 20 | 1.88× | 1.88× |
| cluster | 12 | 1.69× | 1.68× |
| FFT / FFTPACK | 36 | 1.34× | 1.32× |

## Matched before / after

The original **398 non-control cases** are retained as an unchanged sample. New cases are excluded from this comparison. Overall geometric mean: **3.21× → 3.23×** at one thread and **3.96× → 3.99×** at two threads. Most old workloads are unchanged; improvements concentrate on small distances, repeated operator dispatch and newly covered bound-preparation workloads. Small differences in unchanged areas are measurement variation.

| Area | Matched cases | Before → after, 1 thread | Before → after, 2 threads |
|---|---:|---:|---:|
| linalg | 49 | 4.17× → 4.18× | 5.16× → 5.16× |
| ndimage | 43 | 12.64× → 12.45× | 20.97× → 20.83× |
| interpolate | 27 | 3.86× → 3.90× | 4.63× → 4.68× |
| csgraph | 16 | 3.68× → 3.64× | 4.32× → 4.29× |
| signal | 41 | 3.28× → 3.30× | 3.88× → 3.92× |
| sparse | 27 | 3.88× → 3.86× | 4.52× → 4.49× |
| sparse linear algebra | 20 | 1.29× → 1.44× | 1.29× → 1.45× |
| stats | 58 | 6.12× → 6.15× | 6.26× → 6.28× |
| special | 35 | 1.00× → 1.00× | 1.82× → 1.84× |
| spatial | 21 | 1.01× → 1.00× | 1.69× → 1.70× |
| integrate | 35 | 1.71× → 1.73× | 1.72× → 1.73× |
| optimize | 14 | 1.63× → 1.64× | 1.64× → 1.64× |
| cluster | 12 | 1.69× → 1.69× | 1.70× → 1.68× |

The new spatial fixtures use a separate RNG, preserving all original spatial inputs. The former float64 `pdist` control is replaced by a float32 fallback control and is excluded from the matched sample. FFT has no original benchmark cohort.

## Limits and validation

- FFT dispatch saves Python overhead; large transforms are approximately unchanged. Eight declined FFT controls averaged 0.966× true SciPy, with roughly 0.2–0.4 µs added overhead on some tiny fallbacks.
- The worst measured tiny spatial fallback added 0.313 µs (6.9%). Large distance threading is unchanged.
- A tiny public least-squares solve measured 0.96×; array-bound fallback measured 0.99×. Scalar filling helps large parameter vectors, not every solve.
- Full combined project suite before the test-only portability repairs below: **7,934 passed**, 58 warnings, 104.5 seconds, **586 MiB** sampled process-tree peak RSS. Existing thread-contract tests retain their explicit thread settings.
- Extended-precision fallback portability repair: Linux CI found four new FFT tests comparing independently allocated `longdouble` output storage. Those unsupported inputs already delegate to SciPy; padding bytes are a likely cause, not proven by printed values alone. The replacement tests call the actual original SciPy function and assert returned-object identity, unchanged input/forwarded options/warnings, and no fast-path use. Supported-dtype byte checks and numerical tolerances are unchanged. **190 focused FFT tests passed** after this test-only change, using **114 MiB** peak RSS; raw failure and repair logs are retained. No production change or benchmark rerun was needed.
- Composed-solver portability repair: Intel Mac CI reported four byte-comparison failures in the new CG/GMRES history test; its 7,374 other tests and all 13 benchmark areas passed. The repair reuses the existing Intel-only numerical contract, preserving exact result structure, status, history length, dtype and strides, bounded value comparisons, and independently evaluated equation residuals. Other backends retain byte comparisons. One-iteration solves now also require exactly one callback. No production arithmetic, helper tolerances or global policy changed; reference variation is not claimed as proven. **240 sparse-linear-algebra tests passed** locally after this repair, including validator rejection controls, using **125 MiB** peak RSS. Intel CI remains a separate gate; raw failure and local repair logs are retained.
- Final combined-main Windows ARM validation exposed an unsafe pre-existing test fixture: the disabled SciPy reference received a length-7 Mahalanobis `VI` where its native loop reads 49 doubles. The supplied-`VI` validator only converts dtype, so these inputs permit an out-of-bounds read; see [pinned source evidence](spatial-statistics-source-evidence.txt) and [crash log](spatial-statistics-windows-arm-failure.log). The test-only repair uses correctly shaped list/float32 matrices for real SciPy fallback comparisons and retains malformed-shape guard/delegation checks through forbidden-kernel and identity spies. No production code, tolerances, skips or quarantines changed. **26 focused tests passed**, with **157 MiB** peak RSS; the revised combined main requires its own platform checks.
- Individual paired upstream comparisons: sparse linear algebra 1,533; differentiation 202; FFT 6,363; spatial 470; optimization 178. Each reported zero changed outcomes. See individual scopes/exclusions and raw logs; these counts are not a full SciPy coverage percentage.
- Full candidate benchmark: **472/472 checks passed**, one area per process, native 1/2 threads, BLAS/OpenMP 1, 0.03-second timing budget with at least three calls after warmup. This is a short comparison sweep, not a precision performance guarantee.
- Numerical jobs ran sequentially. Focused agent jobs used a 512 MiB cap; the combined project suite used 1 GiB; broad benchmark processes used 2 GiB. The first all-in-one baseline hit its 2 GiB cap and was terminated safely. Completed areas were retained and the remaining areas passed in fresh processes. Failed/incomplete attempts are retained in resource records rather than presented as passes.

Raw [comparison](comparison.json), [source/environment manifest](manifest.json), [baseline results](baseline-bench/), [candidate results](candidate-bench/), [resources](resources/), [project log](project-tests.log) and compressed JUnit are included. The optimization reproduction script can be run from the repository root with the test environment and native/BLAS threads limited to one:

```sh
PYTHONPATH=python FASTSCIPY_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/runs/2026-10-22-review/probe_optimize.py
```

Run broader benchmarks one area at a time with `FASTSCIPY_NUM_THREADS=2 python bench/run.py AREA --budget 0.03 --check`; the original broad runner retains its existing disabled-wrapper reference method. Native source is unchanged; no local Rust rebuild or new package release was performed.
