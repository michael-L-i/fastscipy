# Compatibility candidate evidence

These are development results, not a new published release qualification.
The published `0.1.0a4` support record remains unchanged.

`local/` retains installed-wheel canaries, monitored process reports,
dependency snapshots, and exact wheel hashes. The CP39 wheel's packaged Python
files match the runtime sources in this candidate; the first CP311 wheel
predates the legacy ndimage adapters and only establishes the narrower initial
ABI, special, and spatial result.

| Interpreter | NumPy | SciPy | Completed installed-wheel checks |
|---|---|---|---|
| CPython 3.11.14, macOS arm64 | 1.23.5 | 1.10.1 | Native ABI; special/spatial isolated fast-path smokes |
| CPython 3.9, macOS arm64 | 1.21.6 | 1.10.1 | Native ABI and source provenance; ndimage/special/spatial isolated fast-path smokes; dependency preservation |

The native ABI canary writes a NumPy array through Rust and compares exact
SciPy bytes. Component smokes run with one component patched at a time and
require actual fast-call increments. These small canaries do not replace
the numerical suites or paired upstream comparisons.

Builds used one Cargo job and a 6 GiB process-tree RSS limit; installed canaries
used a 1 GiB limit. Numerical threads were two and BLAS threads one. Neither
build used host CPU tuning. The CP39 wheel uses the existing macOS 14 floor;
the initial local CP311 wheel's macOS 11 tag is not an expanded OS support claim.

The pinned cibuildwheel 4.3.1 identifier probe confirms that the configured
older interpreters have build identifiers. It does not establish successful
cross-platform builds. The proposed 53-wheel release matrix has not been run
or published by this work.
