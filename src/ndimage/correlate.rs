//! Kernels for the correlate family: `_nd_image.correlate1d`,
//! `uniform_filter1d` and `correlate` (N-D), which power `gaussian_filter`,
//! `uniform_filter`, `sobel`, `prewitt`, `laplace`, `convolve`, ...
//!
//! Results are bit-identical to SciPy: every output is computed in double
//! with the same sequence of operations as SciPy's C loop (SIMD lanes hold
//! different outputs, never parts of one sum), and converted back with the
//! same casts.
//!
//! * `ext`: boundary handling (`NI_ExtendLine`, `NI_InitFilterOffsets`).
//! * `cast`: dtype conversions matching SciPy's compiled C casts, task runner.
//! * `simd`: two-lane double vectors (NEON / SSE2 / portable).
//! * `kern`: compute kernels; also documents how SciPy's fused multiply-adds
//!   are reproduced (the scheme is chosen at run time by the Python shim,
//!   which checks it against the installed SciPy).
//! * `line`: drivers for the 1-D filters (axis passes, in place, threading).
//! * `nd`: driver for the N-D correlation.

use pyo3::prelude::*;

mod cast;
mod ext;
mod kern;
mod line;
mod nd;
mod simd;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(line::corr_correlate1d, m)?)?;
    m.add_function(wrap_pyfunction!(line::corr_uniform_filter1d, m)?)?;
    m.add_function(wrap_pyfunction!(nd::corr_correlate, m)?)?;
    Ok(())
}
