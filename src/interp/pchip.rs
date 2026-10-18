//! `RegularGridInterpolator(method="pchip")` on a local window.
//!
//! PCHIP is local: only the (up to) 4 nodes around the query cell matter per
//! axis. Each axis is reduced last to first with the slope, Hermite-coefficient
//! and polynomial arithmetic of `PchipInterpolator` (plain mul/add, no FMA).

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn sign(v: f64) -> i8 {
    (v > 0.0) as i8 - (v < 0.0) as i8
}

fn edge(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if sign(d) != sign(m0) {
        0.0
    } else if sign(m0) != sign(m1) && d.abs() > 3.0 * m0.abs() {
        3.0 * m0
    } else {
        d
    }
}

/// Slope at grid node `node` from the window `y`, whose first node is `lo`.
fn slope(x: &[f64], y: &[f64], lo: usize, node: usize) -> f64 {
    let (n, j) = (x.len(), node - lo);
    if node == 0 {
        let (h0, h1) = (x[1] - x[0], x[2] - x[1]);
        edge(h0, h1, (y[1] - y[0]) / h0, (y[2] - y[1]) / h1)
    } else if node == n - 1 {
        let (h0, h1) = (x[n - 1] - x[n - 2], x[n - 2] - x[n - 3]);
        edge(h0, h1, (y[j] - y[j - 1]) / h0, (y[j - 1] - y[j - 2]) / h1)
    } else {
        let (h0, h1) = (x[node] - x[node - 1], x[node + 1] - x[node]);
        let (m0, m1) = ((y[j] - y[j - 1]) / h0, (y[j + 1] - y[j]) / h1);
        if sign(m0) != sign(m1) || m1 == 0.0 || m0 == 0.0 {
            return 0.0;
        }
        let (w1, w2) = (2.0 * h1 + h0, h1 + 2.0 * h0);
        1.0 / ((w1 / m0 + w2 / m1) / (w1 + w2))
    }
}

/// Cubic Hermite value in cell `i` of the 1-D grid `x`, from the window `y` starting at node `lo`.
fn hermite(x: &[f64], i: usize, lo: usize, y: &[f64], q: f64) -> f64 {
    let (d0, d1) = (slope(x, y, lo, i), slope(x, y, lo, i + 1));
    let dx = x[i + 1] - x[i];
    let (y0, y1) = (y[i - lo], y[i + 1 - lo]);
    let sl = (y1 - y0) / dx;
    let t = (d0 + d1 - 2.0 * sl) / dx;
    let (c0, c1) = (t / dx, (sl - d0) / dx - t);
    let s = q - x[i];
    let mut res = 0.0 + y0 * 1.0;
    let mut z = s;
    res += d0 * z;
    z *= s;
    res += c1 * z;
    z *= s;
    res + c0 * z
}

fn cell(x: &[f64], q: f64) -> usize {
    let n = x.len();
    if q < x[0] {
        return 0;
    }
    if q >= x[n - 1] {
        return n - 2;
    }
    let (mut lo, mut hi) = (0, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if q < x[mid] { hi = mid } else { lo = mid }
    }
    lo
}

/// `v` is the C-contiguous values array with `grids.len()` (1 to 3) axes, `q` is (m, ndim).
/// Returns `None` unless every grid has 4 or more nodes and all values and queries are finite,
/// and no intermediate result is not (SciPy refits `PchipInterpolator` on those and may raise).
#[pyfunction]
pub fn pchip_eval<'py>(
    py: Python<'py>, grids: Vec<PyReadonlyArray1<'py, f64>>, v: PyReadonlyArray1<'py, f64>, q: PyReadonlyArray2<'py, f64>,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let g: Vec<&[f64]> = grids.iter().map(|a| a.as_slice()).collect::<Result<_, _>>()?;
    let (v, q) = (v.as_slice()?, q.as_slice()?);
    let nd = g.len();
    let m = q.len() / nd.max(1);
    let size: usize = g.iter().map(|x| x.len()).product();
    if !(1..=3).contains(&nd) || g.iter().any(|x| x.len() < 4) || v.len() != size
        || !v.iter().all(|v| v.is_finite()) || !q.iter().all(|v| v.is_finite())
    {
        return Ok(None);
    }
    let mut stride = [1usize; 3];
    for d in (0..nd.saturating_sub(1)).rev() {
        stride[d] = stride[d + 1] * g[d + 1].len();
    }
    let out = PyArray1::<f64>::zeros(py, m, false);
    // SAFETY: freshly created, contiguous, and only written inside this call.
    let o = unsafe { out.as_slice_mut()? };
    py.detach(|| super::fill(o, 1, 64, |i0, o| {
        for (r, o) in o.iter_mut().enumerate() {
            let q = &q[(i0 + r) * nd..][..nd];
            let (mut cl, mut lo, mut w) = ([0; 3], [0; 3], [0; 3]);
            for d in 0..nd {
                let n = g[d].len();
                cl[d] = cell(g[d], q[d]);
                lo[d] = cl[d].saturating_sub(1);
                w[d] = (cl[d] + 2).min(n - 1) - lo[d] + 1;
            }
            // Gather the window (row-major in the window extents), then fold the last axis away repeatedly.
            let mut buf = [0.0f64; 64];
            let mut len = 0;
            for a in 0..w[0] {
                for b in 0..if nd > 1 { w[1] } else { 1 } {
                    for c in 0..if nd > 2 { w[2] } else { 1 } {
                        let idx = (lo[0] + a) * stride[0] + if nd > 1 { (lo[1] + b) * stride[1] } else { 0 }
                            + if nd > 2 { lo[2] + c } else { 0 };
                        buf[len] = v[idx];
                        len += 1;
                    }
                }
            }
            for d in (0..nd).rev() {
                for p in 0..len / w[d] {
                    buf[p] = hermite(g[d], cl[d], lo[d], &buf[p * w[d]..(p + 1) * w[d]], q[d]);
                }
                len /= w[d];
                if d > 0 && !buf[..len].iter().all(|v| v.is_finite()) {
                    break;
                }
            }
            *o = if len == 1 { buf[0] } else { f64::NAN };
        }
    }));
    Ok((!o.iter().any(|v| v.is_nan())).then_some(out))
}
