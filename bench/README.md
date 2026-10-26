# Benchmarks

The runner compares fully unpatched SciPy with fastscipy on deterministic,
equivalent inputs. Each side loads an independent benchmark module and advances
its case generator in the appropriate patch state. This matters for functions
captured in closures and objects such as `cKDTree`: disabling dispatch or
unpatching their module bindings does not undo captured implementations.

Fixtures must use explicit random seeds and be repeatable. Two sets of fixtures
are live at once; run one area per process when memory is limited. The runner
keeps only the current pair of cases and their generators, rather than collecting
all cases or outputs. Fixture construction and patch transitions are outside
timing. Do not run the runner alongside application threads using SciPy.

Timings are medians of at least five calibrated batches. Tiny calls share a
batch to reduce timer overhead; calibration is excluded. `--budget` is a sampling
target per implementation, not a timeout. JSON retains the existing timing and
validation fields and adds reference/timing methodology metadata. Historical
reports below used disabled-wrapper references and minimum single-call timings;
their speedups are not directly comparable to new measurements.

Controls remain in the output and regression checks, but are excluded from
headline geometric means, as are nonidentical results.
Measured on a MacBook Pro with M5 Pro; results vary by workload and hardware.

| Results | Thread budget | Details |
|---|---|---|
| Latest source improvements (unreleased) | 1 and 2 | [Review and before/after tables](runs/2026-10-22-review/README.md) |
| Original broad run | 1 and 18 | [Full table](RESULTS.md) |
| Additional workloads | 1 and 2 | [Results](runs/2026-10-19-sweep/summary.json) |

[Eight visual examples](../examples/README.md) · [Raw benchmark data](results/)

Run locally with a release build:

```sh
FASTSCIPY_NUM_THREADS=2 python bench/run.py
```

Byte identity remains the default correctness check. The Intel Accelerate
BFGS case can instead record proven variation in repeated plain SciPy calls
and pass independent checks against the known minimizer. It stays marked
nonidentical and is excluded from headline geometric means.
[Reference evidence and policy](../docs/platforms.md)
