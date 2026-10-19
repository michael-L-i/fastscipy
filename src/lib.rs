//! Rust kernels behind `fastscipy`, one submodule per SciPy area
//! (`fastscipy._native.<area>`). All parallel work runs in `pool::pool()`.

// Kernels mirror SciPy's C loops and arithmetic, so some lints don't fit:
#![allow(
    clippy::needless_range_loop,       // index loops follow SciPy's loop structure
    clippy::neg_cmp_op_on_partial_ord, // `!(a < b)` keeps SciPy's NaN behaviour
    clippy::excessive_precision,       // constants are copied verbatim from SciPy
    clippy::redundant_locals,          // `let p = p;` makes closures capture Send wrappers whole
    clippy::too_many_arguments,
    clippy::type_complexity
)]

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
