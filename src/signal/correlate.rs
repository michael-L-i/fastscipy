//! 2-D float64 core of `scipy.signal._sigtools._correlateND` (direct `correlate`/`convolve`).
//!
//! SciPy's C loop walks the kernel in C order and accumulates `acc += x * k`
//! sequentially from +0 (clang fuses it), skipping nothing but reading zeros
//! outside the input; skipping those taps changes nothing (`acc + 0 * k == acc`
//! for finite `k`; Python declines non-finite kernels). Output `(i, j)` is
//! `sum_{p, q} in[i - off_i + p, j - off_j + q] * k[p, q]`.

use numpy::{PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::pool;

fn row(a: &[f64], w: usize, k: &[f64], kw: usize, (off_i, off_j): (usize, usize), i: usize, out: &mut [f64]) {
    out.fill(0.0);
    let h = a.len() / w;
    for (p, krow) in k.chunks_exact(kw).enumerate() {
        let Some(src) = (i + p).checked_sub(off_i).filter(|&r| r < h) else { continue };
        let src = &a[src * w..][..w];
        for (q, &kv) in krow.iter().enumerate() {
            // output columns j with 0 <= j - off_j + q < w
            let (j0, j1) = (off_j.saturating_sub(q), out.len().min((w + off_j).saturating_sub(q)));
            if j0 >= j1 {
                continue;
            }
            for (o, &s) in out[j0..j1].iter_mut().zip(&src[j0 + q - off_j..]) {
                *o = s.mul_add(kv, *o);
            }
        }
    }
}

#[pyfunction]
pub fn correlate2d(
    py: Python<'_>, a: PyReadonlyArray2<f64>, k: PyReadonlyArray2<f64>, mut out: PyReadwriteArray2<f64>,
    off_i: usize, off_j: usize,
) -> PyResult<()> {
    let (w, kw, ow) = (a.shape()[1], k.shape()[1], out.shape()[1]);
    if w == 0 || kw == 0 || ow == 0 {
        return Err(PyValueError::new_err("empty array"));
    }
    let (a, k, o) = (a.as_slice()?, k.as_slice()?, out.as_slice_mut()?);
    let one = |i: usize, o: &mut [f64]| row(a, w, k, kw, (off_i, off_j), i, o);
    py.detach(|| {
        if pool::num_threads() > 1 && o.len() * k.len() >= 1 << 18 {
            pool::pool().install(|| o.par_chunks_mut(ow).enumerate().for_each(|(i, o)| one(i, o)));
        } else {
            o.chunks_mut(ow).enumerate().for_each(|(i, o)| one(i, o));
        }
    });
    Ok(())
}
