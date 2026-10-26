# Final combined validation

Combined PRs #15, #16 and #17 on merged main `95c4304`; exact source commits
are recorded in `manifest.json`. No production changes were needed to combine
them. The native source fingerprint and binary hash are in `native-build.json`.

- Full project suite: **8766 passed**, 58 warnings, 112.7 seconds, 603 MiB
  sampled process-tree peak under a 1 GiB cap.
- Six public uint8 median benchmarks: all byte-identical and all performance
  checks passed with the new banded-LAPACK native artifact. Two-thread gains
  were **88.9–338.8x** versus genuinely unpatched SciPy; 412 MiB peak.
- All three downstream applications passed exact output bytes, structure,
  warnings, and input-preservation checks with the combined implementation;
  220 MiB peak under a 1 GiB cap.
- Repository-wide Ruff check and format check passed (178 files).

The downstream rerun used nine warmed calls in reversed process order as a
compatibility confirmation. Segmentation remained about 30x faster; the small
Gaussian-process pilot measured 1.03x, reinforcing that its benefit is modest
and variable. Seasonal decomposition still recorded zero fast-path calls.
The longer 99-call controls and original workload definitions remain in
`../2026-10-26-downstream/`; no broad library-wide speedup is claimed.

Numerical work ran serially for the final timing measurements. Numerical
threads were limited to two, BLAS/OpenMP to one; existing project tests retained
their explicit thread parameterizations. Platform CI was still queued or
running when these local checks completed and is not represented as passed.

Reproduce after combining the listed source commits and building a matching
native extension:

```sh
export FASTSCIPY_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python tools/check_native_build.py
python -m pytest tests -q
python bench/run.py signal --only medfilt2d --check --budget 0.08
python tools/pilot_downstream.py --output /tmp/downstream-final --repeat 9 --reverse-order
```

The downstream driver requires the separately documented optional dependencies.
Raw output arrays are regenerable and excluded; JSON reports and bounded-run
logs are retained here.
