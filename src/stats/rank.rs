//! `scipy.stats._stats_py._rankdata` along the last axis of a 2-D array.
//!
//! Values become order-preserving u64 keys (NaN above everything, -0.0 and 0.0
//! equal) paired with their index. Sorting the pairs is stable by construction
//! and gives SciPy's `argsort(stable=True)` order; ranks, tie counts and the
//! sorted copy are then read off the sorted run. NaNs never tie with each other.

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

use crate::pool;

const SIGN: u64 = 1 << 63;
const NAN_KEY: u64 = u64::MAX;
const SMALL: usize = 1024; // pairs sort faster than radix passes below this
const PARALLEL_ROW: usize = 1 << 17; // one row: sort on all threads from here
const PARALLEL_ELEMS: usize = 1 << 16; // many rows: rows on all threads from here

type Pair = (u64, u32);

fn key(bits: u64, kind: u8) -> u64 {
    match kind {
        0 => match bits & !SIGN {
            m if m > 0x7FF0_0000_0000_0000 => NAN_KEY,
            0 => SIGN,
            _ if bits & SIGN != 0 => !bits,
            _ => bits | SIGN,
        },
        1 => bits ^ SIGN,
        _ => bits,
    }
}

/// Stable LSD radix sort by the key (11-bit digits, constant digits skipped).
fn radix(a: &mut Vec<Pair>) {
    let n = a.len();
    let mut hist = vec![[0u32; 2048]; 6];
    for &(k, _) in a.iter() {
        for (p, h) in hist.iter_mut().enumerate() {
            h[((k >> (11 * p)) & 2047) as usize] += 1;
        }
    }
    let mut b = vec![(0u64, 0u32); n];
    for (p, h) in hist.iter_mut().enumerate() {
        if h.iter().any(|&c| c as usize == n) {
            continue;
        }
        let mut s = 0;
        for c in h.iter_mut() {
            (*c, s) = (s, s + *c);
        }
        for &(k, i) in a.iter() {
            let d = &mut h[((k >> (11 * p)) & 2047) as usize];
            b[*d as usize] = (k, i);
            *d += 1;
        }
        std::mem::swap(a, &mut b);
    }
}

struct Opts {
    kind: u8,
    method: u8, // average, min, max, dense, ordinal
}

fn rank_row(
    bits: &[u64], o: &Opts, parallel: bool, ranks: &mut [f64],
    mut ties: Option<&mut [f64]>, sorted: Option<&mut [u64]>,
) {
    let n = bits.len();
    let mut a: Vec<Pair> = bits.iter().enumerate().map(|(i, &b)| (key(b, o.kind), i as u32)).collect();
    if parallel {
        a.par_sort_unstable();
    } else if n <= SMALL {
        a.sort_unstable();
    } else {
        radix(&mut a);
    }
    if let Some(s) = sorted {
        for (s, &(_, i)) in s.iter_mut().zip(&a) {
            *s = bits[i as usize];
        }
    }
    let nan = o.kind == 0;
    let (mut q, mut group) = (0, 0.0);
    while q < n {
        let mut e = q + 1;
        while e < n && a[e].0 == a[q].0 && !(nan && a[e].0 == NAN_KEY) {
            e += 1;
        }
        let c = (e - q) as f64;
        group += 1.0;
        let first = (q + 1) as f64;
        let r = match o.method {
            0 => first + (c - 1.0) / 2.0,
            1 => first,
            2 => first + (c - 1.0),
            _ => group,
        };
        for t in q..e {
            ranks[a[t].1 as usize] = if o.method == 4 { (t + 1) as f64 } else { r };
        }
        if let Some(t) = ties.as_deref_mut() {
            t[q] = c;
        }
        q = e;
    }
}

type Out<'py> = (Bound<'py, PyArray2<f64>>, Option<Bound<'py, PyArray2<u64>>>, Option<Bound<'py, PyArray2<f64>>>);

/// `x` is a C-contiguous (rows, n) view of float64 (`kind` 0), int64 (1) or
/// uint64 (2) bits. Returns (ranks, sorted bits, tie counts).
#[pyfunction]
pub fn rankdata<'py>(
    py: Python<'py>, x: PyReadonlyArray2<'py, u64>, kind: u8, method: u8, want_sorted: bool, want_ties: bool,
) -> PyResult<Out<'py>> {
    let (rows, n) = (x.shape()[0], x.shape()[1]);
    let xs = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    if n == 0 || n >= u32::MAX as usize || kind > 2 || method > 4 {
        return Err(PyValueError::new_err("unsupported rank problem"));
    }
    let ranks = PyArray2::<f64>::zeros(py, (rows, n), false);
    let ties = want_ties.then(|| PyArray2::<f64>::zeros(py, (rows, n), false));
    let sorted = want_sorted.then(|| PyArray2::<u64>::zeros(py, (rows, n), false));
    // SAFETY: freshly created, contiguous, and only written inside this call.
    let (rs, ts, ss) = unsafe {
        (ranks.as_slice_mut()?, ties.as_ref().map(|t| t.as_slice_mut()).transpose()?,
         sorted.as_ref().map(|s| s.as_slice_mut()).transpose()?)
    };
    let o = Opts { kind, method };
    let threads = pool::num_threads();
    py.detach(|| {
        let mut ts: Vec<_> = match ts { Some(t) => t.chunks_mut(n).map(Some).collect(), None => (0..rows).map(|_| None).collect() };
        let mut ss: Vec<_> = match ss { Some(s) => s.chunks_mut(n).map(Some).collect(), None => (0..rows).map(|_| None).collect() };
        let jobs: Vec<_> = xs.chunks(n).zip(rs.chunks_mut(n)).zip(ts.drain(..)).zip(ss.drain(..)).collect();
        if threads > 1 && rows > 1 && rows * n >= PARALLEL_ELEMS {
            pool::pool().install(|| {
                jobs.into_par_iter().for_each(|(((b, r), t), s)| rank_row(b, &o, false, r, t, s));
            });
        } else {
            let big = threads > 1 && n >= PARALLEL_ROW;
            let run = || jobs.into_iter().for_each(|(((b, r), t), s)| rank_row(b, &o, big, r, t, s));
            if big { pool::pool().install(run) } else { run() }
        }
    });
    Ok((ranks, sorted, ties))
}
