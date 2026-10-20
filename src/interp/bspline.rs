//! De Boor evaluation of a B-spline with binary-search interval lookup.
//!
//! SciPy finds the interval by walking from the previous query's interval,
//! which is slow for random queries; the interval itself depends only on the
//! query and the knots, so the result is the same. Basis and coefficient
//! multiply-add contraction are selected independently against SciPy's binary.

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
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

/// Write the spline value of every column of `c` (n, m) at `x` into `out`.
fn eval<const BASIS: bool, const SUM: bool>(
    t: &[f64],
    c: &[f64],
    m: usize,
    k: usize,
    n: usize,
    x: f64,
    out: &mut [f64],
) {
    let l = interval(t, k, n, x);
    let (mut h, mut hh) = ([0.0f64; 6], [0.0f64; 6]);
    h[0] = 1.0;
    for j in 1..=k {
        hh[..j].copy_from_slice(&h[..j]);
        h[0] = 0.0;
        for r in 1..=j {
            let (xb, xa) = (t[l + r], t[l + r - j]);
            if xb == xa {
                h[r] = 0.0;
                continue;
            }
            let w = hh[r - 1] / (xb - xa);
            h[r - 1] = super::madd::<BASIS>(w, xb - x, h[r - 1]);
            h[r] = w * (x - xa);
        }
    }
    out.fill(0.0);
    for a in 0..=k {
        for (o, &c) in out.iter_mut().zip(&c[(l + a - k) * m..][..m]) {
            *o = super::madd::<SUM>(c, h[a], *o);
        }
    }
}

fn evaluate<const BASIS: bool, const SUM: bool>(
    t: &[f64],
    c: &[f64],
    m: usize,
    k: usize,
    n: usize,
    x: &[f64],
    out: &mut [f64],
) {
    super::fill(out, m, 1 << 15, |i, out| {
        for (out, &x) in out.chunks_mut(m).zip(&x[i..]) {
            eval::<BASIS, SUM>(t, c, m, k, n, x, out);
        }
    });
}

/// Values of the spline `(t, c, k)` at `x` (shape (len(x), m)) with `extrapolate=True`; `None`
/// if the knots (not sorted, degenerate end intervals), `k`, `c` or `x` (not finite) are unusual.
#[pyfunction]
#[pyo3(signature = (t, c, k, x, arithmetic=3))]
pub fn bspline_eval<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    c: PyReadonlyArray2<'py, f64>,
    k: usize,
    x: PyReadonlyArray1<'py, f64>,
    arithmetic: u8,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    let m = c.shape()[1];
    let (t, c, x) = (t.as_slice()?, c.as_slice()?, x.as_slice()?);
    if !(1..=5).contains(&k) || t.len() < 2 * k + 2 || m == 0 || arithmetic > 3 {
        return Ok(None);
    }
    let n = t.len() - k - 1;
    let ok = c.len() >= n * m
        && t.windows(2).all(|w| w[0] <= w[1])
        && t[k] < t[k + 1]
        && t[n - 1] < t[n]
        && t[0].is_finite()
        && t[t.len() - 1].is_finite()
        && x.iter().all(|v| v.is_finite());
    if !ok {
        return Ok(None);
    }
    let out = PyArray2::<f64>::zeros(py, (x.len(), m), false);
    // SAFETY: freshly created, contiguous, and only written inside this call.
    let o = unsafe { out.as_slice_mut()? };
    py.detach(|| match arithmetic {
        0 => evaluate::<false, false>(t, c, m, k, n, x, o),
        1 => evaluate::<true, false>(t, c, m, k, n, x, o),
        2 => evaluate::<false, true>(t, c, m, k, n, x, o),
        _ => evaluate::<true, true>(t, c, m, k, n, x, o),
    });
    Ok(Some(out))
}
