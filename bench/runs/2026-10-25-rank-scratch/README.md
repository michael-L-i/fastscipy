# Batched ranking audit · 2026-10-25

**No production optimization is included.** Reusing sort buffers reduced
allocation frequency, but the measured tradeoff did not justify changing the
ranking kernel. This audit adds two public benchmark cases and 90 compatibility
cases for independent rows, including signed zeros, NaNs, infinities, integer
extremes, constant rows and ties, all five methods, and one/two threads.

## Experiments and decision

All three prototypes reused per-call scratch across serial rows and within Rayon
tasks. They retained the sorting algorithm and rank arithmetic. Patches below
apply independently to `8ea95b2`; they are evidence, not enabled code.

1. [Dynamic histogram](rejected-dynamic-histogram.patch): a reusable dynamically
   sized histogram lost substantial performance on tied radix rows. Its timings
   are in [rejected-dynamic-histogram.json](rejected-dynamic-histogram.json), paired
   with [baseline.json](baseline.json).
2. [Fixed histogram](rejected-fixed-histogram.patch): retaining the six histogram
   planes in the type recovered that loss. Batched tied cases improved around
   4–10%, but single-row radix cases were consistently slower. See
   [fixed-histogram.json](fixed-histogram.json) and the adjacent rerun pair
   [baseline-repeat.json](baseline-repeat.json) /
   [fixed-histogram-repeat.json](fixed-histogram-repeat.json).
3. [First-use specialization](rejected-first-use.patch): retained the original
   initial pair collection and zeroed histogram allocation, reusing storage on
   later rows. Single-row regressions remained while batch improvements were
   generally small. The final comparison is
   [final-baseline.json](final-baseline.json) /
   [final-candidate.json](final-candidate.json).

Selected final public `scipy.stats.rankdata(..., axis=-1)` medians, in milliseconds:

| Input | Threads | Existing fastscipy | Rejected candidate | Existing / candidate |
|---|---:|---:|---:|---:|
| 1 × 4096 random | 1 | 0.041 | 0.043 | 0.95× |
| 1 × 200000 random | 1 | 1.565 | 1.682 | 0.93× |
| 1 × 200000 tied | 1 | 1.214 | 1.296 | 0.94× |
| 512 × 2048 tied | 1 | 7.228 | 7.087 | 1.02× |
| 512 × 2048 tied | 2 | 3.928 | 3.736 | 1.05× |
| 64 × 16384 tied | 2 | 3.328 | 3.486 | 0.95× |
| 512 × 2048 mostly constant | 2 | 2.340 | 2.226 | 1.05× |

Ratios below one mean a slowdown. All shapes and samples, including random and
single-row controls, are retained. These are local Apple Silicon measurements,
not cross-platform or published-wheel results. The suspected compiler effect of
histogram shape is an inference from the experiments, not an assembly-level
proof. Allocation reuse alone was insufficient to establish a general win.

## Measurement and reproduction

Both source builds used CPython 3.13.6, NumPy 2.5.4, SciPy 1.18.1, the same
`dev-release` Cargo profile, one Cargo build job, and separate native artifacts.
[manifest.json](manifest.json) records source, patch and native artifact identity.
The reference here is the **previous fastscipy implementation**, not plain SciPy.
The same public calls run through identical Python wrappers in each build.

[probe.py](probe.py) records medians of seven timed batches after warmup. The final
pair uses batches targeting 0.05 seconds, capped at 10,000 calls; earlier probes
capped at 100 calls. Compare pairs within the same experiment. Final JSON includes
the batch counts. Inputs use a fixed seed. The mostly-constant fixture places its
random rows together to expose task-granularity imbalance.

Build each revision with `maturin build --profile dev-release`, place its package
and matching native extension in separate directories, then run serially:

```sh
PYTHONPATH=/path/to/baseline-package FASTSCIPY_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/runs/2026-10-25-rank-scratch/probe.py baseline.json
PYTHONPATH=/path/to/candidate-package FASTSCIPY_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python bench/runs/2026-10-25-rank-scratch/probe.py candidate.json
```

## Final audit branch validation

The production Rust source and native artifact were restored to the verified
baseline before these checks:

- **602 project stats tests passed**, including the 90 new cases; 216 MiB sampled
  process-tree peak RSS. [Log](project-tests.log)
- **Five rankdata benchmark checks passed**, requiring byte identity and expected
  fast paths at one and two threads; 350 MiB peak. [Log](benchmark.log) /
  [JSON](benchmark.json). The existing runner uses disabled wrappers for its
  SciPy reference; its speedup columns are not the before/after evidence above.
- **177 paired upstream cases**, zero changed outcomes, selected with
  `rankdata or spearman or mannwhitney or wilcoxon or kruskal` from SciPy's
  `test_rank`, `test_stats`, and `test_morestats` modules. [Log](upstream.log)
- Ruff checks and formatting passed for changed Python files. Rust prototypes
  also passed formatting and Clippy; the final branch changes no Rust code.

Numerical jobs ran serially with a 512 MiB process-tree cap. Existing project tests
retain their explicit thread settings. [resources.json](resources.json) includes
measurements for validation and each timing run. Initial plain-Cargo build
attempts failed on macOS Python-extension linker flags; all measured binaries
were subsequently built successfully with Maturin.
