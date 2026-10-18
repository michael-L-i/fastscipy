//! De Boor evaluation of a B-spline with binary-search interval lookup.
//!
//! SciPy finds the interval by walking from the previous query's interval,
//! which is slow for random queries; the interval itself depends only on the
//! query and the knots, so the result is the same. clang fuses the two
//! multiply-adds below on arm64, hence `mul_add`.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

/// The largest `l` in `[k, n)` with `t[l] <= x` (the last interval for `x >= t[n]`).
fn interval(t: &[f64], k: usize, n: usize, x: f64) -> usize {
    if x < t[k] {
        return k;
    }
    if x >= t[n] {
        return n - 1;
    }
    let (mut lo, mut hi) = (k, n);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x < t[mid] { hi = mid } else { lo = mid }
    }
    lo
}

fn eval(t: &[f64], c: &[f64], k: usize, n: usize, x: f64) -> f64 {
    let l = interval(t, k, n, x);
    let (mut h, mut hh) = ([0.0f64; 6], [0.0f64; 6]);
    h[0] = 1.0;
    for j in 1..=k {
        hh[..j].copy_from_slice(&h[..j]);
        h[0] = 0.0;
        for m in 1..=j {
            let (xb, xa) = (t[l + m], t[l + m - j]);
            if xb == xa {
                h[m] = 0.0;
                continue;
            }
            let w = hh[m - 1] / (xb - xa);
            h[m - 1] = w.mul_add(xb - x, h[m - 1]);
            h[m] = w * (x - xa);
        }
    }
    (0..=k).fold(0.0, |acc, a| c[l + a - k].mul_add(h[a], acc))
}

/// Values of the spline `(t, c, k)` at `x` with `extrapolate=True`; `None` if the
/// knots (not sorted, degenerate end intervals), `k`, `c` or `x` (not finite) are unusual.
#[pyfunction]
pub fn bspline_eval<'py>(
    py: Python<'py>, t: PyReadonlyArray1<'py, f64>, c: PyReadonlyArray1<'py, f64>, k: usize, x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let (t, c, x) = (t.as_slice()?, c.as_slice()?, x.as_slice()?);
    if !(1..=5).contains(&k) || t.len() < 2 * k + 2 {
        return Ok(None);
    }
    let n = t.len() - k - 1;
    let ok = c.len() >= n
        && t.windows(2).all(|w| w[0] <= w[1])
        && t[k] < t[k + 1]
        && t[n - 1] < t[n]
        && t[0].is_finite()
        && t[t.len() - 1].is_finite()
        && x.iter().all(|v| v.is_finite());
    if !ok {
        return Ok(None);
    }
    let out = PyArray1::<f64>::zeros(py, x.len(), false);
    // SAFETY: freshly created, contiguous, and only written inside this call.
    let o = unsafe { out.as_slice_mut()? };
    py.detach(|| super::fill(o, 1, 1 << 15, |i, o| {
        for (o, &x) in o.iter_mut().zip(&x[i..]) {
            *o = eval(t, c, k, n, x);
        }
    }));
    Ok(Some(out))
}
