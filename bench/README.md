# Benchmarks

Speedups compare fastscipy with unpatched SciPy on the same inputs.
Measured on a MacBook Pro with M5 Pro; results vary by workload and hardware.

| Results | Thread budget | Details |
|---|---|---|
| Published benchmarks | 1 and 18 | [Full table](RESULTS.md) |
| Additional workloads | 1 and 2 | [Results](runs/2026-10-19-sweep/summary.json) |

[Nine visual examples](../examples/README.md) · [Raw benchmark data](results/)

Run locally with a release build:

```sh
FASTSCIPY_NUM_THREADS=2 python bench/run.py
```
