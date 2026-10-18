//! `scipy.ndimage` kernels, one per function of SciPy's private `_nd_image`
//! module. Each returns `Ok(true)` when it handled the call and `Ok(false)`
//! when the arguments are outside what it reproduces exactly (the Python
//! stand-in then calls SciPy's C code).

use pyo3::prelude::*;

pub mod common;
mod correlate;
mod interp;
mod rank;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    correlate::register(m)?;
    interp::register(m)?;
    rank::register(m)
}
