//! `scipy.ndimage` kernels, one per function of SciPy's private `_nd_image`
//! module (plus a few helpers behind SciPy's Python wrappers). Each returns
//! `Ok(true)` or a result when it handled the call and `Ok(false)` or `None`
//! when the arguments are outside what it reproduces exactly (the Python side
//! then calls SciPy).

use pyo3::prelude::*;

mod binmorph;
pub mod common;
mod correlate;
mod edt;
mod fill;
mod genfilt;
mod interp;
mod label;
mod measure;
mod objects;
mod rank;
mod watershed;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    correlate::register(m)?;
    interp::register(m)?;
    rank::register(m)?;
    binmorph::register(m)?;
    measure::register(m)?;
    edt::register(m)?;
    genfilt::register(m)?;
    label::register(m)?;
    fill::register(m)?;
    watershed::register(m)?;
    objects::register(m)
}
