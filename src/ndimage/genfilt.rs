//! Native versions of `generic_filter(input, np.<reducer>)`.
//!
//! The C code gathers each window into a double buffer (footprint order) and
//! calls the Python function on it. Here the same buffer is reduced with numpy's
//! own algorithms: pairwise summation (blocks of 8 accumulators, 128-element
//! leaves) for sum/mean/var/std, ordered folds for max/min.

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ndimage::common::{pool, Elem};

/// numpy's DOUBLE_pairwise_sum (loops_utils.h.src)
fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// np.add.reduce on a 1-D buffer: identity +0.0 + pairwise sum
fn np_sum(a: &[f64]) -> f64 {
    0.0 + pairwise(a)
}

fn np_max(a: &[f64]) -> f64 {
    let mut acc = a[0];
    for &x in &a[1..] {
        // numpy maximum: (in1 >= in2 || isnan(in1)) ? in1 : in2
        // reduction semantics (SIMD fmax-like): NaN propagates, +0.0 beats -0.0
        if acc.is_nan() { break; }
        if x.is_nan() || x > acc || (x == acc && acc.is_sign_negative()) { acc = x; }
    }
    acc
}
fn np_min(a: &[f64]) -> f64 {
    let mut acc = a[0];
    for &x in &a[1..] {
        if acc.is_nan() { break; }
        if x.is_nan() || x < acc || (x == acc && x.is_sign_negative()) { acc = x; }
    }
    acc
}

fn np_var(a: &[f64], tmp: &mut Vec<f64>) -> f64 {
    let n = a.len() as f64;
    let mean = np_sum(a) / n;
    tmp.clear();
    for &x in a {
        let d = x - mean;
        tmp.push(d * d);
    }
    np_sum(tmp) / n
}

fn np_median(a: &mut [f64]) -> f64 {
    if a.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let n = a.len();
    let h = n / 2;
    if n % 2 == 1 {
        let (_, m, _) = a.select_nth_unstable_by(h, |x, y| x.partial_cmp(y).unwrap());
        np_sum(&[*m]) / 1.0
    } else {
        a.select_nth_unstable_by(h, |x, y| x.partial_cmp(y).unwrap());
        let hi = a[h];
        let lo = a[..h].iter().copied().fold(f64::NEG_INFINITY, |m, x| if x > m { x } else { m });
        // mean of the two middle values: sum then divide by 2
        np_sum(&[lo, hi]) / 2.0
    }
}

fn run<T: Elem>(
    padded: &[T],
    pshape: &[usize],
    fshape: &[usize],
    fdata: &[u8],
    out: &mut [f64],
    oshape: &[usize],
    which: u8,
) {
    let n = pshape.len();
    let mut pstride = vec![1usize; n];
    for k in (0..n - 1).rev() {
        pstride[k] = pstride[k + 1] * pshape[k + 1];
    }
    // footprint offsets in padded coords relative to window origin, C order
    let mut offs = Vec::new();
    let total: usize = fshape.iter().product();
    let mut idx = vec![0usize; n];
    for f in 0..total {
        if fdata[f] != 0 {
            offs.push((0..n).map(|k| idx[k] * pstride[k]).sum::<usize>());
        }
        for k in (0..n).rev() {
            idx[k] += 1;
            if idx[k] < fshape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    let len = oshape[n - 1];
    let lead = &oshape[..n - 1];
    let lines_per_chunk = (16384 / len.max(1)).max(1);
    pool().install(|| {
        out.par_chunks_mut(lines_per_chunk * len).enumerate().for_each(|(ci, chunk)| {
            let mut buf = vec![0f64; offs.len()];
            let mut tmp = Vec::with_capacity(offs.len());
            let mut coord = vec![0usize; n - 1];
            for li in 0..chunk.len() / len {
                let mut r = ci * lines_per_chunk + li;
                for k in (0..n - 1).rev() {
                    coord[k] = r % lead[k];
                    r /= lead[k];
                }
                let base: usize = (0..n - 1).map(|k| coord[k] * pstride[k]).sum();
                for x in 0..len {
                    let p = base + x;
                    for (b, &o) in buf.iter_mut().zip(&offs) {
                        *b = padded[p + o].to_f64();
                    }
                    chunk[li * len + x] = match which {
                        0 => np_sum(&buf) / buf.len() as f64,
                        1 => np_sum(&buf),
                        2 => np_max(&buf),
                        3 => np_min(&buf),
                        4 => np_median(&mut buf),
                        5 => np_var(&buf, &mut tmp).sqrt(),
                        6 => np_var(&buf, &mut tmp),
                        _ => np_max(&buf) - np_min(&buf),
                    };
                }
            }
        });
    });
}

#[pyfunction]
fn generic_reduce<'py>(
    py: Python<'py>,
    padded: &Bound<'py, PyUntypedArray>,
    footprint: &Bound<'py, PyArrayDyn<u8>>,
    output: &Bound<'py, PyArrayDyn<f64>>,
    which: u8,
) -> PyResult<bool> {
    if !padded.is_c_contiguous() {
        return Ok(false);
    }
    let pshape = padded.shape().to_vec();
    let fshape = footprint.shape().to_vec();
    let oshape = output.shape().to_vec();
    let fdata = unsafe { footprint.as_slice()? };
    let out = unsafe { output.as_slice_mut()? };
    crate::dispatch_elem!(py, padded.dtype(), T => {
        let a = padded.cast::<PyArrayDyn<T>>()?;
        let ps = unsafe { a.as_slice()? };
        py.detach(|| run::<T>(ps, &pshape, &fshape, fdata, out, &oshape, which));
        Ok(true)
    } else Ok(false))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generic_reduce, m)?)?;
    Ok(())
}
