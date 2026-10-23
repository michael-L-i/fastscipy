# Differentiation overhead audit

No differentiation fast path is proposed. Three portable Python dispatch
experiments did not demonstrate a material public-workload improvement worth
the additional compatibility surface. SciPy's derivative/Jacobian/Hessian
implementation remains in use, and this audit does not change the component
registry, numerical comparison policy, or supported-area count.

## Scope and source

The audit targets SciPy 1.18.1 `scipy.differentiate._differentiate` and
fastscipy `0.1.0a4`, starting from repository commit `f60dca0`. The inspected
SciPy source has SHA-256
`0ea78f4e88f000a330b5af615a401239b23fb63c7e027c414a665b7cdea58c97`.
The demo and testing environments contain the same source bytes.

The public derivative retains its initial function evaluation, adaptive
stencil, nested-point reuse, stopping criteria, and result packing. Jacobian
and Hessian continue to call that implementation. The existing optimize
`_Fun_Wrapper` acceleration is unrelated to these functions and was not edited.

Experiments:

- **Indexed updates:** replace only this module's `xpx` binding with a facade
  that forwards writable exact NumPy numeric-array updates to the same final
  indexed assignment and in-place operator used by array-api-extra. Other
  inputs retain its dispatch. This leaves the shared iterative-method module
  and other SciPy modules untouched.
- **Private-state dictionary alias:** private `_RichResult` work dictionaries
  also serve as their instance attribute dictionaries. Public/callback results
  are constructed by the unchanged iterative-method module. This creates
  reference cycles and therefore adds a memory-lifetime concern.
- **Private-state subclass:** directly look up attributes in the private work
  dictionary, with normal attribute fallback. This avoids reference cycles but
  also failed to produce a material gain.

`tools/probe_differentiate.py` preserves these experiments as explicit opt-in
measurement code. They are not imported by the package or registered as a
component. Every context restores its replacements, including on failure.

## Initial measurements

Seconds per call are the minimum of five batches; the first experiment used
10 calls per batch, and the private-state experiment used 20. All five output
objects matched plain SciPy byte-for-byte, including dtype, shape, strides,
status, iteration count, evaluation count, and other result fields. These are
small local screening measurements, not release performance claims.

| Workload | Plain SciPy, ms | Existing a4, ms | Indexed updates, ms | Ratio vs SciPy |
|---|---:|---:|---:|---:|
| Scalar sine derivative | 0.177 | 0.177 | 0.165 | 1.07x |
| 32-point sine derivative | 0.189 | 0.187 | 0.173 | 1.10x |
| 4096-point sine derivative | 0.866 | 0.852 | 0.838 | 1.03x |
| 8-variable Jacobian | 0.234 | 0.233 | 0.216 | 1.09x |
| 3-variable Hessian | 3.256 | 3.282 | 3.168 | 1.03x |

The private-state alias alone measured 0.98–1.04x versus SciPy; the subclass
measured approximately 0.97–1.02x. Combined state/update changes remained
approximately 1.02–1.09x. Timing variability is significant at this scale.
There is no evidence here for a 3x improvement.

`indexed-updates.log` and `private-state.log` retain the initial outputs.
Their resource records report 2.2 s / 106 MiB and 3.9 s / 110 MiB respectively.
Each numerical process held the shared `heavy.lock`, ran through
`tools.platform_suite.run` with a 512 MiB process-tree limit and 45 s timeout,
and used one native thread and one BLAS/OMP/Accelerate thread. The existing
published native module was linked unchanged into this worktree; it was not
rebuilt or changed. The resource JSON records the exact interpreter and native
source paths.

## Reproduction and coverage

Run `tools/probe_differentiate.py --output <path>` under the same bounded runner.
The tool emits versions, platform, source hash, thread environment, all five
batch timings, and byte checks of both results and function-argument traces.
It compares plain SciPy, existing fastscipy, and each rejected candidate.
The 4096-point case is a control for workloads where array processing dominates
Python dispatch overhead. No changes to numerical expressions or call counts
are used to obtain a timing improvement.

`tests/test_differentiate.py` adds public compatibility controls, explicitly
without requiring a fast path. They cover float32/float64, randomized points,
orders and step directions, strided/read-only inputs, argument/keyword
broadcasting, preserved shapes, empty/nonfinite values, warning/error modes,
callback termination and traces, user-function exceptions, and result metadata.
Analytic sine derivatives, a polynomial derivative, a Jacobian, and a quadratic
Hessian provide checks independent of the SciPy oracle.

This is source- and workload-based coverage, not a percentage estimate of
SciPy's API. No GPU, alternative array backend, Windows, Linux, Intel Mac, or
additional Python-version validation is claimed. Any future acceleration must
retain the array API fallback and separately demonstrate callback, warning,
mutation, metadata, and numerical compatibility on its accepted domain.

## Completed validation

The final bounded run took 6.9 s with 167 MiB sampled process-tree peak RSS
under a 512 MiB cap and 90 s timeout. It completed:

- 35 new compatibility/analytic tests, including native-thread settings 1 and 2;
- the six-mode reproduction with exact results and function-argument traces;
- the full paired SciPy differentiation suite: 200 passed and 2 expected
  failures in each mode, 202 shared tests, zero changed outcomes, and no
  incomplete pairs.

`reproduction.json`, `validation.log`, `validation.resources.json`, and the
`upstream/` JUnit/manifest files retain the evidence. The upstream runs both
emit the same `PytestUnknownMarkWarning` about `array_api_backends`.
Ruff checks and formatting checks passed for the two added Python files.

The broadcast controls also record SciPy 1.18.1's error for multidimensional
broadcast args/kwargs combined with `preserve_shape=True`; valid broadcasting
and preserved-shape calls are checked separately against analytic derivatives.
