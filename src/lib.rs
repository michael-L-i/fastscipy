//! Rust kernels behind `fastscipy`, one submodule per SciPy area
//! (`fastscipy._native.<area>`). All parallel work runs in `pool::pool()`.

use pyo3::prelude::*;

pub mod pool;

mod csgraph;
mod interp;
mod linalg;
mod ndimage;
mod signal;
mod sparse;
mod stats;

type Register = fn(&Bound<'_, PyModule>) -> PyResult<()>;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pool::register(m)?;
    let areas: [(&str, Register); 7] = [
        ("csgraph", csgraph::register),
        ("interp", interp::register),
        ("linalg", linalg::register),
        ("ndimage", ndimage::register),
        ("signal", signal::register),
        ("sparse", sparse::register),
        ("stats", stats::register),
    ];
    for (name, register) in areas {
        let sub = PyModule::new(m.py(), name)?;
        register(&sub)?;
        m.add_submodule(&sub)?;
    }
    Ok(())
}
