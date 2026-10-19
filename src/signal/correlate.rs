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

fn row(
    a: &[f64],
    w: usize,
    k: &[f64],
    kw: usize,
    (off_i, off_j): (usize, usize),
    i: usize,
    out: &mut [f64],
) {
    out.fill(0.0);
    let h = a.len() / w;
    for (p, krow) in k.chunks_exact(kw).enumerate() {
        let Some(src) = (i + p).checked_sub(off_i).filter(|&r| r < h) else {
            continue;
        };
        let src = &a[src * w..][..w];
        for (q, &kv) in krow.iter().enumerate() {
            // output columns j with 0 <= j - off_j + q < w
            let (j0, j1) = (
                off_j.saturating_sub(q),
                out.len().min((w + off_j).saturating_sub(q)),
            );
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
    py: Python<'_>,
    a: PyReadonlyArray2<f64>,
    k: PyReadonlyArray2<f64>,
    mut out: PyReadwriteArray2<f64>,
    off_i: usize,
    off_j: usize,
) -> PyResult<()> {
    let (w, kw, ow) = (a.shape()[1], k.shape()[1], out.shape()[1]);
    if w == 0 || kw == 0 || ow == 0 {
        return Err(PyValueError::new_err("empty array"));
    }
    let (a, k, o) = (a.as_slice()?, k.as_slice()?, out.as_slice_mut()?);
    let one = |i: usize, o: &mut [f64]| row(a, w, k, kw, (off_i, off_j), i, o);
    let parallel = pool::num_threads() > 1 && o.len() * k.len() >= 1 << 18;
    py.detach(|| {
        if parallel {
            pool::pool().install(|| {
                o.par_chunks_mut(ow)
                    .enumerate()
                    .for_each(|(i, o)| one(i, o))
            });
        } else {
            o.chunks_mut(ow).enumerate().for_each(|(i, o)| one(i, o));
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// `_sigtools._convolve2d` (`convolve2d` / `correlate2d`), float64.
//
// SciPy's C sums each kernel row in groups of four, `sum += ((t0*x0 + t1*x1) + t2*x2) + t3*x3`,
// which clang compiles to `fma(t3, x3, fma(t2, x2, fma(t0, x0, t1 * x1)))` followed by a plain add;
// the remaining taps are fused into the sum one by one. Out-of-range samples are real taps (the fill
// value, reflected or wrapped samples), so only an all-fill row with fill 0 and a finite kernel row,
// which adds exact zeros, is skipped.

#[derive(Clone, Copy)]
struct Conv<'a> {
    a: &'a [f64],
    k: &'a [f64],
    shape: (i64, i64),
    kshape: (i64, i64),
    flip: bool,
    mode: u8,
    boundary: u8,
    fill: f64,
}

fn reflect(j: i64, m: i64) -> i64 {
    let k = if j >= 0 {
        j % (2 * m)
    } else {
        (j + 1).abs() % (2 * m)
    };
    if k >= m { 2 * m - k - 1 } else { k }
}

fn index(j: i64, m: i64, boundary: u8) -> Option<usize> {
    match boundary {
        _ if (0..m).contains(&j) => Some(j as usize),
        4 => Some(reflect(j, m) as usize),
        8 => Some(if j >= 0 { j % m } else { (j % m + m) % m } as usize),
        _ => None,
    }
}

/// Image position of output 0 along one axis, for kernel length `k`.
fn origin(mode: u8, flip: bool, k: i64) -> i64 {
    match (mode, flip) {
        (2, true) | (0, false) => 0,
        (2, false) => 1 - k,
        (1, true) => (k - 1) >> 1,
        (1, false) => -((k - 1) >> 1),
        _ => k - 1,
    }
}

impl Conv<'_> {
    fn row(&self, m: usize, out: &mut [f64], ext: &mut Vec<f64>) {
        let ((h, w), (kh, kw)) = (self.shape, self.kshape);
        let (m0, n0) = (
            origin(self.mode, self.flip, kh) + m as i64,
            origin(self.mode, self.flip, kw),
        );
        let (ow, kwu) = (out.len(), kw as usize);
        let cmin = if self.flip { n0 - (kw - 1) } else { n0 };
        out.fill(0.0);
        for j in 0..kh {
            let src = index(if self.flip { m0 - j } else { m0 + j }, h, self.boundary);
            let t = &self.k[(j * kw) as usize..][..kwu];
            if src.is_none() && self.fill == 0.0 && t.iter().all(|v| v.is_finite()) {
                continue;
            }
            ext.clear();
            ext.extend((0..(ow + kwu - 1) as i64).map(|e| {
                match src.zip(index(cmin + e, w, self.boundary)) {
                    Some((r, c)) => self.a[r * w as usize + c],
                    None => self.fill,
                }
            }));
            let tap = |q: usize| &ext[if self.flip { kwu - 1 - q } else { q }..][..ow];
            let mut q = 0;
            while q + 4 <= kwu {
                let (p0, p1, p2, p3) = (tap(q), tap(q + 1), tap(q + 2), tap(q + 3));
                for n in 0..ow {
                    out[n] += t[q + 3].mul_add(
                        p3[n],
                        t[q + 2].mul_add(p2[n], t[q].mul_add(p0[n], t[q + 1] * p1[n])),
                    );
                }
                q += 4;
            }
            for q in q..kwu {
                for (o, &p) in out.iter_mut().zip(tap(q)) {
                    *o = t[q].mul_add(p, *o);
                }
            }
        }
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn convolve2d(
    py: Python<'_>,
    a: PyReadonlyArray2<f64>,
    k: PyReadonlyArray2<f64>,
    mut out: PyReadwriteArray2<f64>,
    flip: bool,
    mode: u8,
    boundary: u8,
    fill: f64,
) -> PyResult<()> {
    let (shape, kshape, oshape) = (a.shape().to_vec(), k.shape().to_vec(), out.shape().to_vec());
    let want = |n: usize, kn: usize| match mode {
        0 => n + 1 - kn.min(n + 1),
        1 => n,
        _ => n + kn - 1,
    };
    if shape.contains(&0)
        || kshape.contains(&0)
        || mode > 2
        || !matches!(boundary, 0 | 4 | 8)
        || oshape != [want(shape[0], kshape[0]), want(shape[1], kshape[1])]
        || oshape.contains(&0)
    {
        return Err(PyValueError::new_err("unsupported arguments"));
    }
    let c = Conv {
        a: a.as_slice()?,
        k: k.as_slice()?,
        shape: (shape[0] as i64, shape[1] as i64),
        kshape: (kshape[0] as i64, kshape[1] as i64),
        flip,
        mode,
        boundary,
        fill,
    };
    let o = out.as_slice_mut()?;
    let parallel = pool::num_threads() > 1 && o.len() * c.k.len() >= 1 << 18;
    py.detach(|| {
        if parallel {
            pool::pool().install(|| {
                o.par_chunks_mut(oshape[1])
                    .enumerate()
                    .for_each_init(Vec::new, |ext, (m, o)| c.row(m, o, ext))
            });
        } else {
            let mut ext = Vec::new();
            o.chunks_mut(oshape[1])
                .enumerate()
                .for_each(|(m, o)| c.row(m, o, &mut ext));
        }
    });
    Ok(())
}
