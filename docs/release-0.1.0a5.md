# 0.1.0a5

- Broaden Python metadata from 3.12–3.15 to 3.9–3.15, SciPy from 1.18 to
  1.10–1.18, and NumPy to >=1.19.5,<2.8, subject to SciPy's requirements.
  Stable NumPy <2.6 is required for acceleration; newer or prerelease NumPy
  leaves acceleration off with a status reason.
- Configure 53 wheels across eight 64-bit Linux, macOS and Windows variants.
  Python 3.9 on musl arm64 and Python 3.9–3.10 on Windows arm64 are excluded
  because the selected SciPy dependency wheels are unavailable.
- Qualify older-Python wheels for `ndimage`, `special` and `spatial` with
  installed ABI checks, isolated fast-path smokes, selected project suites
  and paired upstream suites. SciPy 1.18 retains the all-component gate.
  Installation compatibility does not promise acceleration of every call.
- Include small ND FFT dispatch, small Euclidean `pdist` paths, sparse
  linear-operator dispatch and scalar optimization-bound improvements.
  The [optimization review](../bench/runs/2026-10-22-review/README.md) retains
  historical workload-specific measurements; no new release speedup is inferred.
- Keep interpolation and Euclidean distance transforms with SciPy before
  1.18. Keep SciPy 1.10–1.12 CDFLIB ufuncs unthreaded because their upstream
  saved state can hang under concurrent calls. Missing or unloadable native
  extensions leave native components off with a recorded reason.

The full [53-wheel rehearsal](https://github.com/michael-L-i/fastscipy/actions/runs/42139402656)
passed on 2026-11-06 UTC at source
`074db9aa2dfd0db97264dc6bf82aa1a3c3e8ce00`, including the source archive and
complete artifact gate, with publishing disabled. This does not qualify
a different source or establish publication. Release-source validation,
publication, independent file-hash verification and clean installed pilots
are separate steps.

The [platform record](platforms.md#compatibility-candidates) lists exact
dependency pins and upstream baseline limitations, including nonfinite legacy
special-function orders, SciPy 1.16/1.17 fragmented masked `erf` calls,
Intel Accelerate numerical contracts and Windows ARM upstream crash exclusions.
Those exceptions are retained visibly; they are not passing coverage of the
excluded cases or production fixes to SciPy.
