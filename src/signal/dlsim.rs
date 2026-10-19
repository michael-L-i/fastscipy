//! The `dlsim` state loop with `np.dot(M, v)` (Accelerate `dgemv`) reproduced bit for bit.
//!
//! Measured on Accelerate: for a row of `c` products the sum runs in two lane chains
//! (even and odd columns). A chain of `m` products starts with the first product, adds
//! products 1 to m-2 with a fused multiply-add, and adds the last one plainly; the two
//! chains are added, then a leftover odd column is added plainly. That holds for some
//! shapes only, so Python checks each shape against `np.dot` before using this
//! (`gemv_matches`), and any other BLAS fails that check.

use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn dot(row: &[f64], x: &[f64]) -> f64 {
    let c = row.len();
    if c == 1 {
        return row[0] * x[0];
    }
    let (even, m) = (c / 2 * 2, c / 2);
    let mut lanes = [0.0; 2];
    for (lane, sum) in lanes.iter_mut().enumerate() {
        *sum = row[lane] * x[lane];
        if m >= 2 {
            for k in 1..m - 1 {
                *sum = row[lane + 2 * k].mul_add(x[lane + 2 * k], *sum);
            }
            let j = lane + 2 * (m - 1);
            *sum += row[j] * x[j];
        }
    }
    let mut total = lanes[0] + lanes[1];
    for j in even..c {
        total += row[j] * x[j];
    }
    total
}

fn gemv(m: &[f64], cols: usize, v: &[f64], out: &mut [f64]) {
    for (row, o) in m.chunks_exact(cols).zip(out) {
        *o = dot(row, v);
    }
}

/// True if the emulation equals the reference products `refs[t] = np.dot(ms[t], vs[t])` in every trial.
#[pyfunction]
pub fn gemv_matches(
    ms: PyReadonlyArray3<f64>,
    vs: PyReadonlyArray2<f64>,
    refs: PyReadonlyArray2<f64>,
) -> PyResult<bool> {
    let (shape, ms, vs, refs) = (
        ms.shape().to_vec(),
        ms.as_slice()?,
        vs.as_slice()?,
        refs.as_slice()?,
    );
    let (rows, cols) = (shape[1], shape[2]);
    let mut out = vec![0.0; rows];
    for t in 0..shape[0] {
        gemv(
            &ms[t * rows * cols..][..rows * cols],
            cols,
            &vs[t * cols..][..cols],
            &mut out,
        );
        if out
            .iter()
            .zip(&refs[t * rows..][..rows])
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Fills `xout[1..]` and `yout` from `xout[0]` and the inputs `u`; all C-contiguous.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn dlsim_loop(
    py: Python<'_>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
    c: PyReadonlyArray2<f64>,
    d: PyReadonlyArray2<f64>,
    u: PyReadonlyArray2<f64>,
    mut xout: PyReadwriteArray2<f64>,
    mut yout: PyReadwriteArray2<f64>,
) -> PyResult<()> {
    let (n, ns, ni, no) = (xout.shape()[0], a.shape()[0], b.shape()[1], c.shape()[0]);
    let fits = a.shape() == [ns, ns]
        && b.shape() == [ns, ni]
        && c.shape() == [no, ns]
        && d.shape() == [no, ni]
        && u.shape() == [n, ni]
        && xout.shape() == [n, ns]
        && yout.shape() == [n, no]
        && n > 0;
    if !fits {
        return Err(PyValueError::new_err("inconsistent state-space shapes"));
    }
    let (a, b, c, d, u) = (
        a.as_slice()?,
        b.as_slice()?,
        c.as_slice()?,
        d.as_slice()?,
        u.as_slice()?,
    );
    let (x, y) = (xout.as_slice_mut()?, yout.as_slice_mut()?);
    py.detach(|| {
        let (mut ax, mut bu, mut cx, mut du) =
            (vec![0.0; ns], vec![0.0; ns], vec![0.0; no], vec![0.0; no]);
        for i in 0..n {
            let ui = &u[i * ni..][..ni];
            if i + 1 < n {
                let (head, tail) = x.split_at_mut((i + 1) * ns);
                gemv(a, ns, &head[i * ns..], &mut ax);
                gemv(b, ni, ui, &mut bu);
                for (o, (p, q)) in tail[..ns].iter_mut().zip(ax.iter().zip(&bu)) {
                    *o = p + q;
                }
            }
            gemv(c, ns, &x[i * ns..][..ns], &mut cx);
            gemv(d, ni, ui, &mut du);
            for (o, (p, q)) in y[i * no..][..no].iter_mut().zip(cx.iter().zip(&du)) {
                *o = p + q;
            }
        }
    });
    Ok(())
}
