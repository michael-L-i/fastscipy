//! Rust kernels for `scipy.interpolate` and for the query reordering around
//! `scipy.spatial.Delaunay.find_simplex`.

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::pool;

mod bspline;
mod pchip;
mod simplex;

const ROWS_PER_TASK: usize = 4096;

/// Match the contraction chosen by the compiler of the installed SciPy binary.
#[inline(always)]
fn madd<const FUSED: bool>(a: f64, b: f64, c: f64) -> f64 {
    if FUSED {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Fill `out` (rows of `width` values) with `f(first_row, rows)`, on the shared
/// pool when the thread budget allows and there are at least `parallel_from` rows.
fn fill<T: Send>(
    out: &mut [T],
    width: usize,
    parallel_from: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let rows = out.len() / width.max(1);
    if pool::num_threads() > 1 && rows >= parallel_from {
        pool::pool().install(|| {
            out.par_chunks_mut(ROWS_PER_TASK * width)
                .enumerate()
                .for_each(|(i, c)| f(i * ROWS_PER_TASK, c))
        });
    } else {
        f(0, out);
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bspline::bspline_eval, m)?)?;
    m.add_function(wrap_pyfunction!(pchip::pchip_eval, m)?)?;
    m.add_function(wrap_pyfunction!(simplex::cell_order, m)?)?;
    m.add_function(wrap_pyfunction!(simplex::tie_flags, m)?)?;
    m.add_function(wrap_pyfunction!(simplex::replay_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(simplex::linear_eval, m)?)?;
    Ok(())
}
