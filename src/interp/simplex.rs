//! Helpers for reordering `Delaunay.find_simplex` queries and for evaluating
//! `LinearNDInterpolator` once the simplex of every query is known.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

/// Stable counting sort of the rows of `x` (n, ndim) by grid cell, boustrophedon
/// order (the last axis reverses on odd cells) so neighbouring cells follow
/// each other. Non-finite coordinates go to the lower bound.
#[pyfunction]
pub fn cell_order<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    lo: PyReadonlyArray1<'py, f64>,
    span: PyReadonlyArray1<'py, f64>,
    g: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let nd = x.shape()[1];
    let (x, lo, span) = (x.as_slice()?, lo.as_slice()?, span.as_slice()?);
    let n = x.len() / nd.max(1);
    let out = PyArray1::<i64>::zeros(py, n, false);
    // SAFETY: freshly created, contiguous, and only written inside this call.
    let order = unsafe { out.as_slice_mut()? };
    py.detach(|| {
        let key: Vec<usize> = x
            .chunks(nd)
            .map(|p| {
                let (mut k, mut flip) = (0, 0);
                for d in 0..nd {
                    let v = if p[d].is_finite() { p[d] } else { lo[d] };
                    let mut c =
                        (((v - lo[d]) / span[d] * g as f64) as i64).clamp(0, g as i64 - 1) as usize;
                    if d == nd - 1 && flip == 1 {
                        c = g - 1 - c;
                    }
                    flip ^= c & 1;
                    k = k * g + c;
                }
                k
            })
            .collect();
        let mut start = vec![0usize; g.pow(nd as u32) + 1];
        for &k in &key {
            start[k + 1] += 1;
        }
        for c in 1..start.len() {
            start[c] += start[c - 1];
        }
        for (i, &k) in key.iter().enumerate() {
            order[start[k]] = i as i64;
            start[k] += 1;
        }
    });
    Ok(out)
}

/// Set `flag[i]` where query `i` lies within `band` of a face of its simplex
/// `res[i]` (or the barycentric coordinates are not finite), so that the answer
/// of SciPy's start-dependent walk may differ from ours. The error term widens
/// the band for ill-conditioned simplices. Queries outside the hull are not judged here.
#[pyfunction]
pub fn tie_flags<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray2<'py, f64>,
    res: PyReadonlyArray1<'py, i32>,
    transform: PyReadonlyArray3<'py, f64>,
    band: f64,
    mut flag: PyReadwriteArray1<'py, bool>,
) -> PyResult<()> {
    let nd = xs.shape()[1];
    let (xs, res, tr, flag) = (
        xs.as_slice()?,
        res.as_slice()?,
        transform.as_slice()?,
        flag.as_slice_mut()?,
    );
    py.detach(|| {
        super::fill(flag, 1, 1 << 15, |i0, flag| {
            for (r, f) in flag.iter_mut().enumerate() {
                let s = res[i0 + r];
                if s < 0 {
                    continue;
                }
                let x = &xs[(i0 + r) * nd..][..nd];
                let t = &tr[s as usize * (nd + 1) * nd..][..(nd + 1) * nd];
                let (mut last, mut lerr, mut bad) = (1.0, 0.0, false);
                for a in 0..nd {
                    let (mut acc, mut err) = (0.0, 0.0);
                    for b in 0..nd {
                        let tt = t[a * nd + b];
                        acc = tt.mul_add(x[b] - t[nd * nd + b], acc);
                        err += tt.abs() * (x[b].abs() + t[nd * nd + b].abs());
                    }
                    err *= 1e-12;
                    last -= acc;
                    lerr += err;
                    bad |= !(acc - err >= band && acc + err <= 1.0 - band);
                }
                *f = bad || !(last - lerr >= band && last + lerr <= 1.0 - band);
            }
        })
    });
    Ok(())
}

/// Inclusive index ranges `(a, b)` of the original query order that must be re-run
/// sequentially: each flagged query is preceded by the nearest unflagged query
/// found inside the hull (whose answer does not depend on where the walk
/// started), and overlapping ranges are merged. `None` if the ranges would
/// cover more than `max_total` queries or number more than `max_ranges`.
#[pyfunction]
pub fn replay_intervals(
    res: PyReadonlyArray1<'_, i32>,
    flagged: PyReadonlyArray1<'_, bool>,
    max_total: usize,
    max_ranges: usize,
) -> PyResult<Option<Vec<(usize, usize)>>> {
    let (res, flagged) = (res.as_slice()?, flagged.as_slice()?);
    let (mut out, mut anchor, mut total): (Vec<(usize, usize)>, usize, usize) = (vec![], 0, 0);
    for (f, &fl) in flagged.iter().enumerate() {
        if !fl {
            if res[f] >= 0 {
                anchor = f;
            }
            continue;
        }
        match out.last_mut() {
            Some(last) if anchor <= last.1 => {
                total += f - last.1;
                last.1 = f;
            }
            _ => {
                total += f - anchor + 1;
                out.push((anchor, f));
            }
        }
        if total > max_total || out.len() > max_ranges {
            return Ok(None);
        }
    }
    Ok(Some(out))
}

/// `LinearNDInterpolator` arithmetic (barycentric weights, then the weighted
/// sum of the vertex values) given the simplex of each query; `fill` where it is -1.
/// clang fuses the multiply-adds on arm64, hence `mul_add`.
#[pyfunction]
pub fn linear_eval<'py>(
    py: Python<'py>,
    simplex: PyReadonlyArray1<'py, i32>,
    transform: PyReadonlyArray3<'py, f64>,
    simplices: PyReadonlyArray2<'py, i32>,
    values: PyReadonlyArray2<'py, f64>,
    q: PyReadonlyArray2<'py, f64>,
    fill: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (nd, nv) = (q.shape()[1], values.shape()[1]);
    let (sx, tr, sp, vals, q) = (
        simplex.as_slice()?,
        transform.as_slice()?,
        simplices.as_slice()?,
        values.as_slice()?,
        q.as_slice()?,
    );
    let out = PyArray2::<f64>::zeros(py, (sx.len(), nv), false);
    // SAFETY: freshly created, contiguous, and only written inside this call.
    let o = unsafe { out.as_slice_mut()? };
    py.detach(|| {
        super::fill(o, nv, 1 << 12, |i0, o| {
            let mut c = [0.0f64; 8];
            for (r, row) in o.chunks_mut(nv).enumerate() {
                let s = sx[i0 + r];
                if s < 0 {
                    row.fill(fill);
                    continue;
                }
                let (x, t) = (
                    &q[(i0 + r) * nd..][..nd],
                    &tr[s as usize * (nd + 1) * nd..][..(nd + 1) * nd],
                );
                c[nd] = 1.0;
                for a in 0..nd {
                    c[a] = (0..nd).fold(0.0, |acc, b| {
                        t[a * nd + b].mul_add(x[b] - t[nd * nd + b], acc)
                    });
                    c[nd] -= c[a];
                }
                row.fill(0.0);
                for j in 0..=nd {
                    let m = sp[s as usize * (nd + 1) + j] as usize;
                    for (o, &v) in row.iter_mut().zip(&vals[m * nv..][..nv]) {
                        *o = c[j].mul_add(v, *o);
                    }
                }
            }
        })
    });
    Ok(out)
}
