# Intel Mac reference repeatability diagnostics `956fc49`

[Diagnostic run 39445302011](https://github.com/michael-L-i/fastscipy/actions/runs/39445302011)
used probe checkout `956fc499e367ecf65969bb8e5dcb46138874ef0a` with the unchanged
release-profile wheel from [native run 39376214465](https://github.com/michael-L-i/fastscipy/actions/runs/39376214465),
source `daf9acd697e89612d1bede34d569522ea35fd7d7`. No native code was rebuilt
and no candidate source was overlaid onto the installed wheel. The diagnostic
runner used macOS 15.7.9 x86_64, CPython 3.13.15, NumPy 2.5.4, and SciPy 1.18.1;
the original wheel validation used CPython 3.13.16.

The [provenance](provenance.json), [resource report](diagnostics.json), and
[hash manifest](hash-manifest.json) retain the checkout/wheel relationship.
The original downloaded wheel's SHA256 is
`1fdf1de788fcac733fecb318e6178623dfa211640d83ffa52743bfee35e36657`.
The manifest also records the three probe-source hashes at the diagnostic
checkout and the installed-wheel source-file hashes. The probes used at most
two numerical threads and one BLAS thread. Stats, optimization, and sparse
probes completed in 4.24 / 3.74 / 1.62 seconds and peaked at
95.1 / 86.8 / 50.2 MiB RSS, respectively.

## Observed baseline variability

- [Stats raw JSON log](probe_stats_platform.log): 28 of 72 groups of repeated
  **unpatched** Pearson/linear-regression calls produced two or three distinct
  exact results for fixed caller inputs. Changing only vector addresses within
  a 64-byte alignment block changed `np.vecdot` result bits for nine of twelve
  tested lengths. For normalized length-nine vectors, left offset zero and
  right offsets zero/eight/24 produced `0x1.2ba4d75bf2b0ep-5`,
  `0x1.2ba4d75bf2b10p-5`, and `0x1.2ba4d75bf2b13p-5`.
  Seeded replay captured four mismatches with exact input bytes. One occurred
  with no fast path at all: both calls executed original SciPy. Three other
  captured inputs produced multiple results in both reference and patched
  repeats. Allocation history differs from the earlier project-test process;
  replay records its own seed and first mismatch positions.
- [Optimization raw JSON](probe_optimize_platform.json): one of 24 repeated
  unpatched `lsq_linear` pairs differed in `cost`; all four repeated unpatched
  BFGS runs did not share the same `x`. Patched runs and the separately labeled
  copy-wrapper experiment also varied. The report preserves complete changed
  field lists and the input-alignment reduction measurements.
- [Sparse raw JSON](sparse-summary.json): unpatched CG, BiCG, LGMRES, QMR,
  TFQMR, and GCROT(m,k) each produced two to ten distinct results for fixed
  inputs. Their recorded iteration histories also retain byte hashes. The
  two LSQR cases observed here were stable on both sides. This does **not**
  resolve independent LSQR discrepancies or establish that every sparse
  mismatch is caused by baseline variability.

[Derived counts](derived-summary.json) summarize these measurements. They
establish that this native reference environment does not provide a unique
bitwise result for every observed computation. They do not approve a numerical
tolerance, excuse unrelated mismatches, certify a new implementation, or
change the project's production paths or identity checks.

All seven raw reports/logs were copied byte for byte and hash-verified.
`probe_stats_platform.log` is itself valid JSON and retains its original bytes.
The wheel and 42 sparse NPZ array files remain outside Git; their names, sizes,
and SHA256 hashes are in the manifest. Other diagnostics from the same run are
outside this focused archive.
