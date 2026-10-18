//! `scipy.signal` kernels: peak finding, `dlsim` and direct 2-D correlation.

use pyo3::prelude::*;

mod correlate;
mod dlsim;
mod peaks;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(peaks::peak_prominences, m)?)?;
    m.add_function(wrap_pyfunction!(peaks::peak_widths, m)?)?;
    m.add_function(wrap_pyfunction!(dlsim::gemv_matches, m)?)?;
    m.add_function(wrap_pyfunction!(dlsim::dlsim_loop, m)?)?;
    m.add_function(wrap_pyfunction!(correlate::correlate2d, m)?)
}
