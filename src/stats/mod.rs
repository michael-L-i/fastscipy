//! Rust kernels for `scipy.stats`.

use pyo3::prelude::*;

mod rank;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rank::rankdata, m)?)
}
