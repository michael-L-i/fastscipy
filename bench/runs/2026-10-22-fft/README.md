# ND FFT/DCT/DST dispatch audit

This candidate adds the `fft` component. It accelerates unresized ND transforms
on exact, aligned NumPy arrays with native float32/float64/complex64/complex128
dtypes. Real DCT/DST paths accept real arrays only. Supported transforms are
`fftn`, `ifftn`, `dctn`, `idctn`, `dstn`, and `idstn` in both `scipy.fft` and
`scipy.fftpack`; the FFT2 aliases also reach these paths. Resizing, conversion,
overwrite requests, unusual option types, and unsupported inputs fall back.

The implementation removes repeated array conversion, shape parsing, and
unchanged-shape slicing before calling the **same SciPy DUCC kernel** with the
same axes, normalization, and worker resolution. The modern public uarray
entry points remain untouched, so alternate and skipped backends still work.
No worker defaults are overridden, no pools are created, and no result is
cached. `set_workers` contexts and explicit worker arguments retain SciPy's
meaning, independently of fastscipy's thread count.

## Measurements

Mac arm64, Python 3.13.12, NumPy 2.5.4, SciPy 1.18.1, unchanged published
fastscipy 0.1.0a4 native extension plus source Python. These are local results,
not evidence of performance on every supported wheel platform.

The primary results below use [true-baseline.json](true-baseline.json), where
plain SciPy is actually unpatched. The previous-fastscipy comparison enables
the unchanged components from base `f60dca0`, with the new `fft` component
absent. Inputs and thread settings match. For each input, phases run in order
SciPy / previous / candidate / candidate / previous / SciPy; each phase takes
the best of three batches of 100 calls. Every result matches in type, dtype,
shape, strides, and bytes.

| Workload | Plain SciPy µs | Previous fastscipy µs | Candidate µs | vs SciPy | vs previous |
|---|---:|---:|---:|---:|---:|
| fft.fftn, 8×8 | 2.783 | 2.794 | 1.916 | 1.45× | 1.46× |
| fft.dctn, 8×8 | 3.163 | 3.423 | 1.969 | 1.61× | 1.74× |
| fft.idctn, 8×8 | 3.257 | 3.270 | 1.924 | 1.69× | 1.70× |
| fft.fftn, 32×32 | 4.262 | 4.419 | 3.388 | 1.26× | 1.30× |
| fft.dctn, 32×32 | 5.043 | 5.192 | 3.794 | 1.33× | 1.37× |
| fftpack.fftn, 8×8 | 2.425 | 2.418 | 1.659 | 1.46× | 1.46× |
| fftpack.dctn, 8×8 | 2.485 | 2.500 | 1.865 | 1.33× | 1.34× |
| fft.fftn, 256×256 | 124.272 | 123.695 | 123.145 | 1.01× | 1.00× |
| fft.dctn, 64 batched 8×8 tiles | 16.986 | 17.142 | 15.751 | 1.08× | 1.09× |
| fft.dctn, integer control | 3.032 | 3.123 | 3.315 | 0.91× | 0.94× |
| fftpack.fftn, list control | 3.776 | 3.775 | 4.053 | 0.93× | 0.93× |
| fftpack.fftn, resize control | 3.887 | 3.845 | 4.280 | 0.91× | 0.90× |

Across all 36 cases, the geometric mean is **1.193× vs SciPy / 1.192× vs
previous fastscipy**. The 28 eligible cases average 1.267× / 1.273×; eight
controls average 0.966× / 0.946×. The eight 8×8 cases average 1.512× / 1.526×,
32×32 cases 1.265× / 1.276×, and 256×256 cases 1.032× / 1.029×.

The gains belong to small repeated transforms. There is no demonstrated 3×
speedup, and larger or already batched transforms benefit little. Declined
cases still pay roughly 0.2–0.4 µs dispatch overhead in the examples above;
the fftpack resize control is 0.898× against the previous-version timing.
Even unchanged 1D controls vary slightly between timing phases. These costs
and noise are retained in the table and aggregate, not filtered out.

The regular [36-case benchmark](benchmark.json) passes `--check` on one and
two fastscipy threads. Its disabled-wrapper baseline is useful for the normal
project gate but includes wrapper overhead; it must not replace the true
baseline above when reporting gains or fallback costs.

## Validation and resource bounds

- 158 project tests passed, including dtype/layout/read-only/mutation checks,
  warning/error equality, malformed options, endian/alignment conversions,
  NaN/inf/signed zero, singleton/empty inputs, alternate/skipped backends,
  worker contexts and explicit workers, and wrapper disabled/STRICT/counters.
- Independent direct Fourier and DCT-II formulas, inverse reconstruction,
  orthonormal energy preservation, and changed-input checks supplement byte
  comparison. No numerical tolerance exception or test-helper change is used.
- Paired upstream `scipy.fft` and `scipy.fftpack` runs executed **6,363 tests
  each with zero outcome changes** (6,362 passed and one expected failure).
  The patched run reached 1,905 fast paths with zero fast-path errors.
- `-m 'not slow'` deselected two marked tests: the multiprocessing/thread
  stress test and fftpack's package-wide source-import scan. The regular
  upstream worker=-1 and 16-thread contract tests ran, under a coordinator
  exception to the usual two-thread budget. No other new skip was added.
- Upstream results precede the final cheap input rejection wrapper; the
  arithmetic and eligibility remain the same. The full 158-test project
  matrix and all 36 benchmark cases were rerun after that wrapper change.
- All numerical runs used a global `fcntl.flock` slot and
  `tools.platform_suite.run`, 512 MiB process-tree RSS cap, explicit timeouts,
  BLAS=1, and native budget 1/2 except the explicit upstream contract tests.
  Upstream plus the initial audit took 18.4 s, peak 409 MiB. Final tests,
  interleaved comparison, and benchmarks took 34.7 s, peak 138 MiB. Both
  monitors recorded complete readable trees, no kills, and no cleanup errors.

The retained logs and resource manifests give commands and detailed results.
No native rebuild, dependency installation, or shared-environment change was
needed. The platform benchmark area registry includes the new FFT area.

## Rejected or limited approaches

1. Blanket FFT worker overrides were rejected: they change default/context
   behavior and bypass the project's thread policy. This candidate preserves
   SciPy worker selection.
2. Replacing the modern public FFT entry points was rejected because it could
   bypass alternate uarray backends. Only the SciPy backend is wrapped.
3. Replacing DUCC arithmetic, caching outputs, or adding a separate parallel
   FFT implementation was unnecessary and would weaken compatibility.
4. A generic wrapper first passed the standard benchmark gate but the
   [initial true-baseline audit](initial-true-baseline.json) exposed integer
   DCT fallback at 0.84× previous fastscipy. The final counted wrapper rejects
   non-array/unsupported-dtype inputs before entering the candidate's shape
   helpers; that control is now 0.94× previous. Its enabled, STRICT, and
   counter behavior is tested. Some fallback overhead remains as shown above.
5. Large transforms do not show a material gain. Their same-kernel path is
   retained for simplicity, with explicit large/batched controls rather than
   a platform-specific size threshold. One-dimensional FFT and real-packed
   FFT algorithms remain unchanged.

## Reproduction

Use the source checkout with its matching native extension and the ordinary
project resource monitor. Example child commands, run **serially inside a
512 MiB process-tree watchdog** with `FASTSCIPY_NUM_THREADS=2`, BLAS=1 and
`SCIPY_AVAILABLE_MEM=256Mib`:

```sh
python -m pytest tests/test_fft.py -q
python tools/scipy_suite.py --output-dir artifacts/fft-upstream --pyargs scipy.fft scipy.fftpack -m 'not slow'
python bench/audit_fft.py --output artifacts/fft-true-baseline.json
python bench/run.py fft --budget 0.03 --check --output-dir artifacts/fft-bench
```

The upstream command intentionally tests explicit worker/thread counts beyond
two; authorize that bounded contract coverage separately when enforcing a
strict native-thread budget. The project suite and benchmark need no such
exception.
