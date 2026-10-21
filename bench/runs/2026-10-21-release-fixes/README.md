# Failed `v0.1.0a2` release attempt and independent Intel reference proof

This archive records [release run 39464216282](https://github.com/michael-L-i/fastscipy/actions/runs/39464216282)
from tag `v0.1.0a2`, source `1cf67791b17ab2bb3c72d6af62bd2292e8034583`.
**28 of 32 wheel jobs passed; the overall run failed.** Complete-set verification,
TestPyPI and PyPI publication were skipped. This is not a completed portable release
or evidence that any fix below has passed the final wheel gate.

[All job conclusions](final-a2-tag-attempt/jobs.json) retain the source, IDs, links
and timestamps for all 32 wheel jobs and the source/verification/publication jobs.
The complete API response is retained as [raw-run.json.gz](final-a2-tag-attempt/raw-run.json.gz).
All 16 Linux wheel jobs and all eight Windows wheel jobs passed. Four Mac jobs
failed:

- Intel CP312: QMR `test_solver_results_and_iteration_history[1-100-qmr]` and
  Pearson `test_function_randomized[1t-pearsonr]` / `[4t-pearsonr]` failed.
- Intel CP313: MINRES `test_solver_results_and_iteration_history[1-100-minres]` failed.
- ARM64 and Intel CP315: the build supervisor raised `psutil.AccessDenied` while
  reading a child process's RSS during interpreter setup. Installed-wheel
  activation and project tests were not reached in these jobs.

[Failure identities and JUnit totals](final-a2-tag-attempt/failures.json), original
compressed test logs/XML, build records, and both CP315 job logs preserve the
actual failures. QMR and MINRES passed their existing tight numerical, history,
count and independent residual validators. Their failures were the subsequent
byte comparison: eight repeated plain calls did not expose reference variation
in that invocation. They were not failures of the solver accuracy checks.
No broader trajectory tolerance was introduced to describe those logs.

[Native CP312 reference-only diagnostic 39679272811](https://github.com/michael-L-i/fastscipy/actions/runs/39679272811)
used official NumPy 2.5.4 / SciPy 1.18.1 on Intel macOS with CPython 3.12.10.
Its [original provenance](intel-cp312-reference-proof/provenance.json),
[resources](intel-cp312-reference-proof/resources.json),
[raw proof](intel-cp312-reference-proof/numeric-proof.json.gz) and
[derived comparison](intel-cp312-reference-proof/comparison.json) are archived.
The diagnostic source is `94355b17b36bf4b8082de6de048514c7e3dde371`, targeting the
same release source; it ran no fastscipy wheel or patched comparison.

- Twelve plain calls per solver/thread configuration produced QMR byte variants
  8/4 at one/two threads, always 53 callbacks, and MINRES variants 6/3, always 47
  callbacks. Inputs were unchanged and all calls returned `info=0` without warnings.
- QMR true residuals were approximately `7.296581e-9`, below its
  `rtol * ||b|| = 9.715212e-9` criterion.
- MINRES true residuals were approximately `2.273531e-7`. MINRES uses a normwise
  backward-error stopping test, not QMR's norm-of-RHS criterion. They satisfy
  the independent validator's `rtol * ||A||_F * ||x|| ≈ 4.444969e-7` bound.
  The raw diagnostic's generic `residual_limit` field is therefore not a
  MINRES pass criterion; the derived comparison makes the distinction explicit.
- Both exact Pearson failure inputs were recovered. Sixteen same-layout plain
  repeats split 8/8 between positive and negative NaN statistic bits; each
  64-position alignment grid split 32/32. P-values remained positive NaN,
  warning category/message matched, and statistic/correlation alias bits were
  consistent. This establishes plain-reference NaN-sign instability for these
  specific nonfinite inputs; it does not certify patched outputs or a new release.

The reference proof completed in 2.805 seconds with 94,695,424 bytes sampled
process-tree peak RSS. [Raw-file manifest](raw-file-manifest.json) lists original
byte sizes and SHA256 hashes plus archived-file hashes. Gzip files decompress
byte for byte to the original records; JSON observations were not rewritten to
claim later source changes or successful publication.

The [official Mac3.15 installer monitor smoke](mac315-installer/README.md)
passed on ARM and Intel at source `e057c24`, using the exact cibuildwheel4.3.1
package/choices command. Both installed native ordinary-ABI Python3.15.0.
Privileged installer RSS remained inaccessible: reported36.453/38.129MiB
readable-tree peaks are lower bounds, not total memory measurements or proof
of a1GiB total cap. This ran no Rust build, project suite, benchmarks, or wheel
release validation. Original reports and source/package hashes are retained.
