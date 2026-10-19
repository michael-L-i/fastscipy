//! One-pass labelled min/max/positions and center-of-mass sums.
//!
//! SciPy's `_select` argsorts the whole image. Values and tie-free positions
//! are order independent, so a single pass reproduces them exactly. Positions
//! of tied extrema depend on numpy's unstable argsort, so the kernel reports
//! whether any tie occurred and the shim then declines.

use std::collections::HashMap;

use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ndimage::common::{Elem, pool};

trait Val: Elem {
    fn isnan(self) -> bool;
    fn nanval() -> Self;
    fn bits(self) -> u64;
}
macro_rules! val_int { ($($t:ty),*) => {$(
    impl Val for $t {
        #[inline(always)] fn isnan(self) -> bool { false }
        fn nanval() -> Self { 0 }
        #[inline(always)] fn bits(self) -> u64 { self as u64 }
    }
)*}; }
val_int!(u8, i8, u16, i16, u32, i32, u64, i64);
impl Val for f32 {
    #[inline(always)]
    fn isnan(self) -> bool {
        self.is_nan()
    }
    fn nanval() -> Self {
        f32::NAN
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
}
impl Val for f64 {
    #[inline(always)]
    fn isnan(self) -> bool {
        self.is_nan()
    }
    fn nanval() -> Self {
        f64::NAN
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

enum Mapper {
    Table(Vec<u32>),
    Hash(HashMap<i64, u32>),
}
const NONE: u32 = u32::MAX;
impl Mapper {
    fn new(index: &[i64], lmin: i64, lmax: i64) -> (Mapper, usize) {
        // slots are assigned per distinct requested label; returns slot count
        if lmin >= 0 && lmax < (1 << 28) {
            let mut t = vec![NONE; lmax as usize + 1];
            let mut n = 0u32;
            for &i in index {
                if i >= 0 && i <= lmax && t[i as usize] == NONE {
                    t[i as usize] = n;
                    n += 1;
                }
            }
            (Mapper::Table(t), n as usize)
        } else {
            let mut h = HashMap::new();
            let mut n = 0u32;
            for &i in index {
                h.entry(i).or_insert_with(|| {
                    n += 1;
                    n - 1
                });
            }
            (Mapper::Hash(h), n as usize)
        }
    }
    #[inline(always)]
    fn slot(&self, l: i64) -> u32 {
        match self {
            Mapper::Table(t) => {
                if (l as u64) < t.len() as u64 {
                    t[l as usize]
                } else {
                    NONE
                }
            }
            Mapper::Hash(h) => h.get(&l).copied().unwrap_or(NONE),
        }
    }
    fn lookup(&self, l: i64) -> u32 {
        self.slot(l)
    }
}

#[derive(Clone)]
struct St<T> {
    has_min: bool,
    minv: T,
    minpos: u64,
    mintie: bool,
    has_max: bool,
    maxv: T,
    maxpos: u64,
    maxtie: bool,
    nan_cnt: u64,
    nanpos: u64,
    zero_issue: bool,
}
impl<T: Val> St<T> {
    fn new() -> Self {
        St {
            has_min: false,
            minv: T::nanval(),
            minpos: 0,
            mintie: false,
            has_max: false,
            maxv: T::nanval(),
            maxpos: 0,
            maxtie: false,
            nan_cnt: 0,
            nanpos: 0,
            zero_issue: false,
        }
    }
    #[inline(always)]
    fn add(&mut self, v: T, pos: u64) {
        if v.isnan() {
            if self.nan_cnt == 0 {
                self.nanpos = pos;
            }
            self.nan_cnt += 1;
            return;
        }
        if !self.has_min || v < self.minv {
            self.has_min = true;
            self.minv = v;
            self.minpos = pos;
            self.mintie = false;
        } else if v == self.minv {
            self.mintie = true;
            if v.bits() != self.minv.bits() {
                self.zero_issue = true;
            }
        }
        if !self.has_max || v > self.maxv {
            self.has_max = true;
            self.maxv = v;
            self.maxpos = pos;
            self.maxtie = false;
        } else if v == self.maxv {
            self.maxtie = true;
            if v.bits() != self.maxv.bits() {
                self.zero_issue = true;
            }
        }
    }
    fn merge(&mut self, b: &St<T>) {
        self.zero_issue |= b.zero_issue;
        if b.has_min {
            if !self.has_min || b.minv < self.minv {
                self.has_min = true;
                self.minv = b.minv;
                self.minpos = b.minpos;
                self.mintie = b.mintie;
            } else if b.minv == self.minv {
                self.mintie = true;
                if b.minv.bits() != self.minv.bits() {
                    self.zero_issue = true;
                }
            }
        }
        if b.has_max {
            if !self.has_max || b.maxv > self.maxv {
                self.has_max = true;
                self.maxv = b.maxv;
                self.maxpos = b.maxpos;
                self.maxtie = b.maxtie;
            } else if b.maxv == self.maxv {
                self.maxtie = true;
                if b.maxv.bits() != self.maxv.bits() {
                    self.zero_issue = true;
                }
            }
        }
        if b.nan_cnt > 0 && self.nan_cnt == 0 {
            self.nanpos = b.nanpos;
        }
        self.nan_cnt += b.nan_cnt;
    }
}

fn select_impl<T: Val>(
    values: &[T],
    labels: &[i64],
    index: &[i64],
    lmin: i64,
    lmax: i64,
) -> (Vec<T>, Vec<i64>, Vec<T>, Vec<i64>, bool, bool, bool) {
    let (map, nslots) = Mapper::new(index, lmin, lmax);
    let n = values.len();
    let nthreads = pool().current_num_threads();
    let chunk = n.div_ceil(nthreads * 4).max(1 << 16);
    let parts: Vec<Vec<St<T>>> = pool().install(|| {
        (0..n.div_ceil(chunk))
            .into_par_iter()
            .map(|c| {
                let lo = c * chunk;
                let hi = (lo + chunk).min(n);
                let mut st = vec![St::<T>::new(); nslots];
                for i in lo..hi {
                    let s = map.slot(labels[i]);
                    if s != NONE {
                        st[s as usize].add(values[i], i as u64);
                    }
                }
                st
            })
            .collect()
    });
    let mut it = parts.into_iter();
    let mut tot = it.next().unwrap();
    for p in it {
        for (a, b) in tot.iter_mut().zip(&p) {
            a.merge(b);
        }
    }
    let mut minv = Vec::with_capacity(index.len());
    let mut minp = Vec::with_capacity(index.len());
    let mut maxv = Vec::with_capacity(index.len());
    let mut maxp = Vec::with_capacity(index.len());
    let (mut mintie, mut maxtie, mut zi) = (false, false, false);
    let mut seen = vec![false; nslots];
    for &l in index {
        let s = map.lookup(l);
        if s == NONE {
            // label absent from the image: SciPy returns zeros
            minv.push(T::from_f64(0.0));
            minp.push(0);
            maxv.push(T::from_f64(0.0));
            maxp.push(0);
            continue;
        }
        let a = &tot[s as usize];
        if !seen[s as usize] {
            seen[s as usize] = true;
            zi |= a.zero_issue;
            if a.has_min && a.mintie {
                mintie = true;
            }
            if a.has_max && a.maxtie {
                maxtie = true;
            }
            if a.nan_cnt >= 2 {
                if !a.has_min {
                    mintie = true;
                }
                maxtie = true;
            }
        }
        if a.nan_cnt == 0 && !a.has_min {
            // label present in index but with no pixels (cannot happen: slot only filled by pixels)
            minv.push(T::from_f64(0.0));
            minp.push(0);
            maxv.push(T::from_f64(0.0));
            maxp.push(0);
            continue;
        }
        if a.has_min {
            minv.push(a.minv);
            minp.push(a.minpos as i64);
        } else {
            minv.push(T::nanval());
            minp.push(a.nanpos as i64);
        }
        if a.nan_cnt > 0 {
            maxv.push(T::nanval());
            maxp.push(a.nanpos as i64);
        } else {
            maxv.push(a.maxv);
            maxp.push(a.maxpos as i64);
        }
    }
    (minv, minp, maxv, maxp, mintie, maxtie, zi)
}

#[pyfunction]
fn label_select<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyUntypedArray>,
    labels: &Bound<'py, PyArrayDyn<i64>>,
    index: &Bound<'py, PyArrayDyn<i64>>,
    lmin: i64,
    lmax: i64,
) -> PyResult<Option<Py<PyAny>>> {
    let lab = unsafe { labels.as_slice()? };
    let idx = unsafe { index.as_slice()? };
    if !values.is_c_contiguous() || values.len() != lab.len() || lab.is_empty() {
        return Ok(None);
    }
    crate::dispatch_elem!(py, values.dtype(), T => {
        let v = values.cast::<PyArrayDyn<T>>()?;
        let vs = unsafe { v.as_slice()? };
        let (a, b, c, d, t1, t2, zi) = py.detach(|| select_impl::<T>(vs, lab, idx, lmin, lmax));
        let out = (
            PyArray1::from_vec(py, a).into_any().unbind(),
            PyArray1::from_vec(py, b).into_any().unbind(),
            PyArray1::from_vec(py, c).into_any().unbind(),
            PyArray1::from_vec(py, d).into_any().unbind(),
            t1, t2, zi,
        );
        Ok(Some(out.into_pyobject(py)?.into_any().unbind()))
    } else Ok(None))
}

fn com_impl<T: Elem>(
    values: &[T],
    labels: &[i64],
    index: &[i64],
    lmin: i64,
    lmax: i64,
    shape: &[usize],
) -> Vec<Vec<f64>> {
    let (map, nslots) = Mapper::new(index, lmin, lmax);
    let nd = shape.len();
    let mut sums = vec![vec![0.0f64; nslots]; nd + 1];
    let mut coord = vec![0usize; nd];
    for i in 0..values.len() {
        let s = map.slot(labels[i]);
        if s != NONE {
            let w = values[i].to_f64();
            let s = s as usize;
            sums[0][s] += w;
            for d in 0..nd {
                sums[d + 1][s] += w * coord[d] as f64;
            }
        }
        for d in (0..nd).rev() {
            coord[d] += 1;
            if coord[d] < shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }
    // expand to the requested index order
    let mut out = vec![Vec::with_capacity(index.len()); nd + 1];
    for &l in index {
        let s = map.lookup(l);
        for d in 0..=nd {
            out[d].push(if s == NONE { 0.0 } else { sums[d][s as usize] });
        }
    }
    out
}

#[pyfunction]
fn label_com_sums<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyUntypedArray>,
    labels: &Bound<'py, PyArrayDyn<i64>>,
    index: &Bound<'py, PyArrayDyn<i64>>,
    lmin: i64,
    lmax: i64,
) -> PyResult<Option<Vec<Bound<'py, PyArray1<f64>>>>> {
    let lab = unsafe { labels.as_slice()? };
    let idx = unsafe { index.as_slice()? };
    if !values.is_c_contiguous() || values.len() != lab.len() || lab.is_empty() {
        return Ok(None);
    }
    let shape = values.shape().to_vec();
    crate::dispatch_elem!(py, values.dtype(), T => {
        let v = values.cast::<PyArrayDyn<T>>()?;
        let vs = unsafe { v.as_slice()? };
        let r = py.detach(|| com_impl::<T>(vs, lab, idx, lmin, lmax, &shape));
        Ok(Some(r.into_iter().map(|x| PyArray1::from_vec(py, x)).collect()))
    } else Ok(None))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(label_select, m)?)?;
    m.add_function(wrap_pyfunction!(label_com_sums, m)?)?;
    Ok(())
}
