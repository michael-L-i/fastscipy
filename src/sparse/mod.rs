//! scipy.sparse kernels: `argmax`/`argmin` along an axis, COO `sum_duplicates`,
//! and dense to CSR/CSC conversion. Index and value arrays arrive contiguous with
//! a dtype the Python layer has already checked; dense input may have any strides.

use numpy::{Element, PyArray1, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::pool::pool;

type Arr<'a, 'py> = &'a Bound<'py, PyUntypedArray>;

trait Idx: Copy + Send + Sync + Ord + Default + Element {
    fn get(self) -> i64;
    fn new(v: usize) -> Self;
}
macro_rules! idx {
    ($($t:ty),*) => {$(impl Idx for $t {
        fn get(self) -> i64 { self as i64 }
        fn new(v: usize) -> Self { v as $t }
    })*};
}
idx!(i32, i64);

trait Val: Copy + Send + Sync + PartialOrd + Default + Element {
    fn is_nan(self) -> bool;
    /// What `np.add.reduceat` returns for one segment.
    fn seg_sum(s: &[Self]) -> Self;
}

/// numpy's pairwise summation (`@TYPE@_pairwise_sum`), which `np.add.reduce` uses for contiguous floats.
macro_rules! float_val {
    ($($t:ty => $pw:ident),*) => {$(
        fn $pw(a: &[$t]) -> $t {
            let n = a.len();
            if n < 8 {
                return a.iter().fold(-0.0, |r, &x| r + x);
            }
            if n <= 128 {
                let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
                let mut i = 8;
                while i < n - n % 8 {
                    for j in 0..8 {
                        r[j] += a[i + j];
                    }
                    i += 8;
                }
                let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for &x in &a[i..] {
                    res += x;
                }
                return res;
            }
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            $pw(&a[..n2]) + $pw(&a[n2..])
        }
        impl Val for $t {
            fn is_nan(self) -> bool { self != self }
            fn seg_sum(s: &[Self]) -> Self { if s.len() == 1 { s[0] } else { s[0] + $pw(&s[1..]) } }
        }
    )*};
}
float_val!(f32 => pairwise_f32, f64 => pairwise_f64);

macro_rules! int_val {
    ($($t:ty),*) => {$(impl Val for $t {
        fn is_nan(self) -> bool { false }
        fn seg_sum(s: &[Self]) -> Self { s.iter().fold(0, |a, &b| a.wrapping_add(b)) }
    })*};
}
int_val!(i32, i64);

/// The array's elements as a slice. The caller guarantees a contiguous array of dtype `T`.
unsafe fn slice<'a, T>(a: Arr) -> &'a [T] {
    unsafe { std::slice::from_raw_parts(ptr(a) as *const T, a.len()) }
}

fn ptr(a: Arr) -> *const u8 {
    unsafe { (*a.as_array_ptr()).fields.data as *const u8 }
}

fn code(a: Arr) -> (u8, usize) {
    let d = a.dtype();
    (d.kind(), d.itemsize())
}

/// Call `$f::<I, T>(args)` for the index dtype `$ic` (i32 or i64) and value dtype `$vc` (f32, f64, i32 or i64).
macro_rules! by_types {
    ($ic:expr, $vc:expr, $f:ident($($a:expr),*)) => {
        match ($ic, $vc) {
            ((b'i', 4), (b'f', 4)) => $f::<i32, f32>($($a),*),
            ((b'i', 4), (b'f', 8)) => $f::<i32, f64>($($a),*),
            ((b'i', 4), (b'i', 4)) => $f::<i32, i32>($($a),*),
            ((b'i', 4), (b'i', 8)) => $f::<i32, i64>($($a),*),
            ((b'i', 8), (b'f', 4)) => $f::<i64, f32>($($a),*),
            ((b'i', 8), (b'f', 8)) => $f::<i64, f64>($($a),*),
            ((b'i', 8), (b'i', 4)) => $f::<i64, i32>($($a),*),
            ((b'i', 8), (b'i', 8)) => $f::<i64, i64>($($a),*),
            _ => None,
        }
    };
}

/// A raw output buffer that workers write disjoint parts of.
#[derive(Clone, Copy)]
struct Out<T>(*mut T);
unsafe impl<T> Send for Out<T> {}
unsafe impl<T> Sync for Out<T> {}
impl<T> Out<T> {
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, off: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

/// Run `f` on each range, on the pool when `threads > 1`.
fn for_ranges(
    py: Python<'_>,
    ranges: &[(usize, usize)],
    threads: usize,
    f: impl Fn(usize, usize) + Sync,
) {
    py.detach(|| {
        if threads <= 1 {
            ranges.iter().for_each(|&(lo, hi)| f(lo, hi));
        } else {
            pool().install(|| ranges.par_iter().for_each(|&(lo, hi)| f(lo, hi)));
        }
    });
}

/// Split `0..n` into chunks of at least `min` items, up to `8 * threads` of them.
fn split(n: usize, min: usize, threads: usize) -> Vec<(usize, usize)> {
    let k = (n / min.max(1)).clamp(1, if threads > 1 { 8 * threads } else { 1 });
    (0..k).map(|i| (n * i / k, n * (i + 1) / k)).collect()
}

// ---- argmax / argmin along the compressed axis -------------------------------------------------

/// SciPy's `_argminmax_axis` per line, including its implicit-zero rules and the quirk
/// `min(extreme_index, zero_index)` where `extreme_index` is a position within the line.
fn argminmax_k<I: Idx, T: Val>(
    py: Python<'_>,
    ip: Arr,
    ij: Arr,
    x: Arr,
    line: i64,
    is_max: bool,
    explicit: bool,
    threads: usize,
) -> Option<Vec<i64>> {
    let (ip, ij, x) = unsafe { (slice::<I>(ip), slice::<I>(ij), slice::<T>(x)) };
    let rows = ip.len() - 1;
    let zero = T::default();
    let row = |r: usize| -> i64 {
        let (p, q) = (ip[r].get() as usize, ip[r + 1].get() as usize);
        if q == p {
            return 0;
        }
        let (mut ei, mut ev) = (0, x[p]);
        if !ev.is_nan() {
            for k in 1..q - p {
                let v = x[p + k];
                if v.is_nan() {
                    (ei, ev) = (k, v);
                    break;
                }
                if (is_max && v > ev) || (!is_max && v < ev) {
                    (ei, ev) = (k, v);
                }
            }
        }
        let len = q - p;
        if explicit || (if is_max { ev > zero } else { ev < zero }) || len as i64 == line {
            return ij[p + ei].get();
        }
        let k = (0..len)
            .find(|&k| ij[p + k].get() != k as i64)
            .unwrap_or(len);
        let zi = if k < len || (k as i64) < line {
            k as i64
        } else {
            -1
        };
        if ev == zero { (ei as i64).min(zi) } else { zi }
    };
    let mut ret = vec![0i64; rows];
    let out = Out(ret.as_mut_ptr());
    let ranges = split(rows, 4096, threads);
    for_ranges(py, &ranges, threads, |lo, hi| {
        let o = unsafe { out.slice(lo, hi - lo) };
        o.iter_mut().zip(lo..hi).for_each(|(o, r)| *o = row(r));
    });
    Some(ret)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn argminmax<'py>(
    py: Python<'py>,
    ip: Arr,
    ij: Arr,
    x: Arr,
    line: i64,
    is_max: bool,
    explicit: bool,
    threads: usize,
) -> Option<Bound<'py, PyArray1<i64>>> {
    let r = by_types!(
        code(ip),
        code(x),
        argminmax_k(py, ip, ij, x, line, is_max, explicit, threads)
    )?;
    Some(PyArray1::from_vec(py, r))
}

// ---- COO sum_duplicates ------------------------------------------------------------------------

type Triple<'py, I, T> = (
    Bound<'py, PyArray1<I>>,
    Bound<'py, PyArray1<I>>,
    Bound<'py, PyArray1<T>>,
);

/// Stable sort by (row, col), then add up equal coordinates in order, like
/// `np.lexsort` + `np.add.reduceat`. `None` if a row index is outside `0..m`.
fn sumdup_k<'py, I: Idx, T: Val>(
    py: Python<'py>,
    r: Arr,
    c: Arr,
    x: Arr,
    m: usize,
    threads: usize,
) -> Option<Triple<'py, I, T>> {
    let (r, c, x) = unsafe { (slice::<I>(r), slice::<I>(c), slice::<T>(x)) };
    let n = x.len();
    let mut start = vec![0usize; m + 1];
    for &ri in r {
        let ri = ri.get();
        if ri < 0 || ri as usize >= m {
            return None;
        }
        start[ri as usize + 1] += 1;
    }
    for i in 0..m {
        start[i + 1] += start[i];
    }
    let mut pos = start.clone();
    let mut buf = vec![(I::default(), T::default()); n];
    for k in 0..n {
        let p = &mut pos[r[k].get() as usize];
        buf[*p] = (c[k], x[k]);
        *p += 1;
    }
    // Chunks of whole rows with about equal numbers of entries.
    let mut ranges = Vec::new();
    let (mut lo, parts) = (0, if threads > 1 { 8 * threads } else { 1 });
    for i in 1..=parts {
        let hi = if i == parts {
            m
        } else {
            start.partition_point(|&s| s < n * i / parts).min(m)
        };
        if hi > lo || i == parts && lo < m {
            ranges.push((lo, hi));
            lo = hi;
        }
    }
    let bufp = Out(buf.as_mut_ptr());
    let mut uniq = vec![0usize; m + 1];
    let up = Out(uniq.as_mut_ptr());
    for_ranges(py, &ranges, threads, |lo, hi| {
        for i in lo..hi {
            let row = unsafe { bufp.slice(start[i], start[i + 1] - start[i]) };
            row.sort_by_key(|e| e.0);
            let u = row
                .iter()
                .enumerate()
                .filter(|&(k, e)| k == 0 || e.0 != row[k - 1].0)
                .count();
            unsafe { *up.slice(i + 1, 1).as_mut_ptr() = u };
        }
    });
    for i in 0..m {
        uniq[i + 1] += uniq[i];
    }
    let total = uniq[m];
    let (mut orow, mut ocol, mut oval) = (
        vec![I::default(); total],
        vec![I::default(); total],
        vec![T::default(); total],
    );
    let (rp, cp, vp) = (
        Out(orow.as_mut_ptr()),
        Out(ocol.as_mut_ptr()),
        Out(oval.as_mut_ptr()),
    );
    for_ranges(py, &ranges, threads, |lo, hi| {
        for i in lo..hi {
            let row = &buf[start[i]..start[i + 1]];
            let (u, mut k) = (uniq[i], 0);
            let (rr, cc, vv) = unsafe {
                (
                    rp.slice(u, uniq[i + 1] - u),
                    cp.slice(u, uniq[i + 1] - u),
                    vp.slice(u, uniq[i + 1] - u),
                )
            };
            let vals: Vec<T> = row.iter().map(|e| e.1).collect();
            let mut s = 0;
            while s < row.len() {
                let e = s + row[s..].iter().take_while(|e| e.0 == row[s].0).count();
                (rr[k], cc[k], vv[k]) = (I::new(i), row[s].0, T::seg_sum(&vals[s..e]));
                (k, s) = (k + 1, e);
            }
        }
    });
    Some((
        PyArray1::from_vec(py, orow),
        PyArray1::from_vec(py, ocol),
        PyArray1::from_vec(py, oval),
    ))
}

#[pyfunction]
fn sum_duplicates<'py>(
    py: Python<'py>,
    row: Arr,
    col: Arr,
    data: Arr,
    m: usize,
    threads: usize,
) -> Option<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    macro_rules! go {
        ($i:ty, $t:ty) => {
            sumdup_k::<$i, $t>(py, row, col, data, m, threads)
                .map(|(a, b, c)| (a.into_any(), b.into_any(), c.into_any()))
        };
    }
    match (code(row), code(data)) {
        ((b'i', 4), (b'f', 4)) => go!(i32, f32),
        ((b'i', 4), (b'f', 8)) => go!(i32, f64),
        ((b'i', 4), (b'i', 4)) => go!(i32, i32),
        ((b'i', 4), (b'i', 8)) => go!(i32, i64),
        ((b'i', 8), (b'f', 4)) => go!(i64, f32),
        ((b'i', 8), (b'f', 8)) => go!(i64, f64),
        ((b'i', 8), (b'i', 4)) => go!(i64, i32),
        ((b'i', 8), (b'i', 8)) => go!(i64, i64),
        _ => None,
    }
}

// ---- dense to CSR / CSC ------------------------------------------------------------------------

type Csr<'py, T> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<T>>,
);

/// Nonzeros of a 2-D array, row-major (CSR) or column-major (CSC), as 32-bit `(indptr, indices, data)`.
/// `None` if the counts do not fit in 32 bits.
fn dense_k<'py, T: Val>(py: Python<'py>, a: Arr, csc: bool, threads: usize) -> Option<Csr<'py, T>> {
    let (sh, st) = (a.shape(), a.strides());
    let item = std::mem::size_of::<T>() as isize;
    let (major, minor, sa, sb) = if csc {
        (sh[1], sh[0], st[1] / item, st[0] / item)
    } else {
        (sh[0], sh[1], st[0] / item, st[1] / item)
    };
    if a.len() > i32::MAX as usize {
        return None;
    }
    let base = ptr(a) as usize;
    let at = move |i: usize, j: usize| unsafe {
        *(base as *const T).offset(i as isize * sa + j as isize * sb)
    };
    let zero = T::default();
    if csc && threads <= 1 && sb.abs() >= sa.abs() {
        return Some(csc_row_major(py, minor, major, |i, j| at(j, i)));
    }
    let mut indptr = vec![0i32; major + 1];
    let ranges = split(major, (1 << 16) / minor.max(1) + 1, threads);
    let ipp = Out(indptr.as_mut_ptr());
    let count = |lo: usize, hi: usize| {
        for i in lo..hi {
            let k = (0..minor).filter(|&j| at(i, j) != zero).count();
            unsafe { *ipp.slice(i + 1, 1).as_mut_ptr() = k as i32 };
        }
    };
    for_ranges(py, &ranges, threads, count);
    let mut total = 0usize;
    for i in 0..major {
        total += indptr[i + 1] as usize;
        if total > i32::MAX as usize {
            return None;
        }
        indptr[i + 1] = total as i32;
    }
    let (mut indices, mut data) = (vec![0i32; total], vec![zero; total]);
    let (xp, dp) = (Out(indices.as_mut_ptr()), Out(data.as_mut_ptr()));
    let ip = &indptr;
    for_ranges(py, &ranges, threads, |lo, hi| {
        for i in lo..hi {
            let (o, e) = (ip[i] as usize, ip[i + 1] as usize);
            let (xs, ds) = unsafe { (xp.slice(o, e - o), dp.slice(o, e - o)) };
            let mut k = 0;
            for j in 0..minor {
                let v = at(i, j);
                if v != zero {
                    (xs[k], ds[k]) = (j as i32, v);
                    k += 1;
                }
            }
        }
    });
    Some((
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, data),
    ))
}

/// CSC of a row-major array on one thread: scan rows (cache friendly) and append to per-column buckets.
fn csc_row_major<'py, T: Val>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    at: impl Fn(usize, usize) -> T,
) -> Csr<'py, T> {
    let zero = T::default();
    let mut indptr = vec![0i32; cols + 1];
    for i in 0..rows {
        for j in 0..cols {
            indptr[j + 1] += (at(i, j) != zero) as i32;
        }
    }
    for j in 0..cols {
        indptr[j + 1] += indptr[j];
    }
    let total = indptr[cols] as usize;
    let (mut indices, mut data, mut pos) = (vec![0i32; total], vec![zero; total], indptr.clone());
    for i in 0..rows {
        for j in 0..cols {
            let v = at(i, j);
            if v != zero {
                let p = &mut pos[j];
                (indices[*p as usize], data[*p as usize]) = (i as i32, v);
                *p += 1;
            }
        }
    }
    (
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, data),
    )
}

#[pyfunction]
fn dense_to_compressed<'py>(
    py: Python<'py>,
    a: Arr,
    csc: bool,
    threads: usize,
) -> Option<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    macro_rules! go {
        ($t:ty) => {
            dense_k::<$t>(py, a, csc, threads)
                .map(|(p, i, d)| (p.into_any(), i.into_any(), d.into_any()))
        };
    }
    match code(a) {
        (b'f', 4) => go!(f32),
        (b'f', 8) => go!(f64),
        (b'i', 4) => go!(i32),
        (b'i', 8) => go!(i64),
        _ => None,
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(argminmax, m)?)?;
    m.add_function(wrap_pyfunction!(sum_duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(dense_to_compressed, m)?)?;
    Ok(())
}
