# Intel BiCGSTAB reference repeatability

[Diagnostic run 39452770758](https://github.com/michael-L-i/fastscipy/actions/runs/39452770758)
used probe checkout `b785d7c` and the reused `daf9acd` native wheel, on Intel
macOS with NumPy 2.5.4/SciPy 1.18.1 and Accelerate BLAS. Original summary,
workflow provenance, and outer monitor are retained unchanged. The recorded
probe source is archived separately; this was not a rebuilt release wheel.

Repeated **unpatched** BiCGSTAB runs produced callback counts **40, 41, and 42**
in both sparse-linalg-only and all-component patch scopes. Patched counts
were **40–43** in the former and **40–42** in the latter. Other nine iterative
methods retained stable callback counts on both sides in both scopes.
Common trajectory prefixes were not byte-identical: maximum absolute
differences across saved variants reached approximately 3.57e-8 and 7.60e-8
in the respective scopes. This is trajectory variation, not a claim of
roundoff-identical callbacks or a production fix.

All 65 distinct saved BiCGSTAB solutions returned `info=0`; every final residual
was within the requested `1e-9 * ||b||` bound of 9.715211908718822e-9. The
maximum was 9.55624847591387e-9. These residuals were derived offline from the
saved small arrays and the source-defined tridiagonal system, whose input
hash matches both raw scopes. They do not establish callback byte identity.

[analysis.json](analysis.json) gives per-phase counts, common-prefix
comparisons, final residuals, and scope limits. [external-arrays.json](external-arrays.json)
retains all original NPZ sizes/hashes; arrays remain in the external Actions
artifact. No solver or project native code was executed for this archive.
