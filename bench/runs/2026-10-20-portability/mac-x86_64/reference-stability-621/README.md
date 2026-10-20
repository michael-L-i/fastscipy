# Intel Mac all-components sparse reference diagnostics `62147de`

[Run 39446755156](https://github.com/michael-L-i/fastscipy/actions/runs/39446755156)
used diagnostic checkout `62147ded75a5c930aaa945b876cddd5fa8771a62` and reused
the unchanged native `daf9acd697e89612d1bede34d569522ea35fd7d7` wheel from
[run 39376214465](https://github.com/michael-L-i/fastscipy/actions/runs/39376214465).
This probe enabled all 13 fastscipy components. It ran on macOS 15.7.9 x86_64,
CPython 3.13.15, NumPy 2.5.4 with Accelerate BLAS, and SciPy 1.18.1.
No native rebuild or installed-source overlay was performed by this sparse
probe. Other probes in the same diagnostics run had separate purposes.

The unchanged [sparse summary](sparse-summary.json) and compact
[derived counts](derived-summary.json) record:

- All 96 direct sparse matrix/LinearOperator byte comparisons passed, across
  float64/complex128 CSR-array and CSC-matrix inputs and alignment offsets.
  Reference and patched sparse buffers were identical for all four matrices.
- With all components enabled, float64 LSQR produced two exact result variants
  in **both unpatched and patched** repetitions for each sparse format. The
  observed result-hash sets matched. All four interleaved pairs per format
  happened to match exactly; this does not make every repeated result stable.
- The other six all-components solver cases (float64 LSMR and complex128
  LSQR/LSMR, each in two formats) had one variant per side. All eight solver
  cases recorded unchanged input matrix buffers and right-hand sides.

These observations extend the earlier [956 diagnostics](../reference-stability-956/README.md),
where the two isolated LSQR probes were stable. They establish reference
variability for these particular all-components float64 LSQR calls, without
certifying every LSQR discrepancy, a newer implementation, or a release wheel.

[Provenance](provenance.json), [resource measurements](diagnostics.json), and
the [hash manifest](hash-manifest.json) retain the exact source/wheel relation.
The sparse probe completed in 3.79 seconds with 92.1 MiB monitored peak RSS;
its numerical and BLAS budgets were one thread each.

All three raw JSON reports were copied byte for byte and hash-verified.
The manifest records the probe-source hash at `62147de`, the original native
wheel SHA256, and names/sizes/SHA256 hashes for 45 external sparse NPZ files.
NPZ arrays, the wheel, and bulk logs remain outside Git. Earlier reference
and cast archives are preserved unchanged.
