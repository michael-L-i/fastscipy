//! Batched small-matrix LAPACK loops for `lu_factor`, `lu_solve`, `cho_solve`,
//! `solve_triangular`, `eigh` and `eigvalsh`.
//!
//! Each matrix goes to the Accelerate routine SciPy's f2py wrappers call (the
//! pointers come from `scipy.linalg.cython_lapack`), with the same flags and
//! workspace sizes, so results are bit-identical. Every kernel returns `None`
//! unless all matrices finished with `info == 0` and finite input (when asked);
//! the Python layer then lets SciPy redo the call and raise or warn itself.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use numpy::{PyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::pool::pool;

type I = c_int;
type Getrf = unsafe extern "C" fn(*const I, *const I, *mut f64, *const I, *mut I, *mut I);
type Getrs = unsafe extern "C" fn(
    *const c_char,
    *const I,
    *const I,
    *const f64,
    *const I,
    *const I,
    *mut f64,
    *const I,
    *mut I,
);
type Potrs = unsafe extern "C" fn(
    *const c_char,
    *const I,
    *const I,
    *const f64,
    *const I,
    *mut f64,
    *const I,
    *mut I,
);
type Trtrs = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    *const I,
    *const I,
    *const f64,
    *const I,
    *mut f64,
    *const I,
    *mut I,
);
type Syevr = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    *const I,
    *mut f64,
    *const I,
    *const f64,
    *const f64,
    *const I,
    *const I,
    *const f64,
    *mut I,
    *mut f64,
    *mut f64,
    *const I,
    *mut I,
    *mut f64,
    *const I,
    *mut I,
    *const I,
    *mut I,
);

struct Lapack {
    getrf: Getrf,
    getrs: Getrs,
    potrs: Potrs,
    trtrs: Trtrs,
    syevr: Syevr,
}

static LAPACK: OnceLock<Lapack> = OnceLock::new();

fn lapack() -> &'static Lapack {
    LAPACK.get().expect("linalg.init() not called")
}

/// Look up LAPACK routines in `scipy.linalg.cython_lapack.__pyx_capi__`.
#[pyfunction]
fn init(py: Python<'_>) -> PyResult<()> {
    if LAPACK.get().is_some() {
        return Ok(());
    }
    let capi = py
        .import("scipy.linalg.cython_lapack")?
        .getattr("__pyx_capi__")?;
    let capi = capi.cast::<PyDict>()?;
    let get = |name: &str, signature: &CStr| -> PyResult<*mut c_void> {
        let cap = capi
            .get_item(name)?
            .ok_or_else(|| PyRuntimeError::new_err(format!("no {name}")))?;
        unsafe {
            // Passing our expected name, rather than the capsule's own name,
            // checks the complete C ABI before any function-pointer cast.
            let p = pyo3::ffi::PyCapsule_GetPointer(cap.as_ptr(), signature.as_ptr());
            if p.is_null() {
                Err(PyErr::fetch(py))
            } else {
                Ok(p)
            }
        }
    };
    unsafe {
        let l = Lapack {
            getrf: std::mem::transmute::<*mut c_void, Getrf>(get("dgetrf", c"void (int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *, int *)")?),
            getrs: std::mem::transmute::<*mut c_void, Getrs>(get("dgetrs", c"void (char *, int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *)")?),
            potrs: std::mem::transmute::<*mut c_void, Potrs>(get("dpotrs", c"void (char *, int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *)")?),
            trtrs: std::mem::transmute::<*mut c_void, Trtrs>(get("dtrtrs", c"void (char *, char *, char *, int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *)")?),
            syevr: std::mem::transmute::<*mut c_void, Syevr>(get("dsyevr", c"void (char *, char *, char *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *, __pyx_t_5scipy_6linalg_13cython_lapack_d *, int *, int *, int *, int *)")?),
        };
        let _ = LAPACK.set(l);
    }
    Ok(())
}

/// A strided read-only view of a (B, r, c) array, strides in elements.
#[derive(Clone, Copy)]
struct View<T> {
    p: *const T,
    s: [isize; 3],
}
unsafe impl<T> Send for View<T> {}
unsafe impl<T> Sync for View<T> {}

impl View<f64> {
    fn new(a: &PyReadonlyArray3<f64>) -> Self {
        let v = a.as_array();
        let s = v.strides();
        View {
            p: v.as_ptr(),
            s: [s[0], s[1], s[2]],
        }
    }

    /// Copy matrix `i` (or its transpose) into column-major `dst`; false if it holds a non-finite value.
    unsafe fn load(
        &self,
        i: usize,
        rows: usize,
        cols: usize,
        dst: &mut [f64],
        tr: bool,
        chk: bool,
    ) -> bool {
        let (sr, sc) = if tr {
            (self.s[2], self.s[1])
        } else {
            (self.s[1], self.s[2])
        };
        let base = unsafe { self.p.offset(i as isize * self.s[0]) };
        let mut ok = true;
        for c in 0..cols {
            for r in 0..rows {
                let x = unsafe { *base.offset(r as isize * sr + c as isize * sc) };
                dst[c * rows + r] = x;
                ok &= x.is_finite() || !chk;
            }
        }
        ok
    }
}

/// A raw output buffer that workers write disjoint slices of.
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

/// Run `f(i, f64 scratch, i32 scratch)` for every matrix `i < b`; true if every call returned true.
fn run<F>(py: Python<'_>, b: usize, threads: usize, nf: usize, ni: usize, f: F) -> bool
where
    F: Fn(usize, &mut [f64], &mut [I]) -> bool + Sync,
{
    let bad = AtomicBool::new(false);
    let work = |lo: usize, hi: usize, sf: &mut [f64], si: &mut [I]| {
        for i in lo..hi {
            if bad.load(Relaxed) || !f(i, sf, si) {
                bad.store(true, Relaxed);
                return;
            }
        }
    };
    py.detach(|| {
        if threads <= 1 || b < 2 {
            work(0, b, &mut vec![0.0; nf], &mut vec![0; ni]);
        } else {
            let chunk = (b / (threads * 16)).clamp(1, 1024);
            let ranges: Vec<_> = (0..b)
                .step_by(chunk)
                .map(|lo| (lo, (lo + chunk).min(b)))
                .collect();
            pool().install(|| {
                ranges.into_par_iter().for_each_init(
                    || (vec![0.0; nf], vec![0; ni]),
                    |(sf, si), (lo, hi)| work(lo, hi, sf, si),
                )
            });
        }
    });
    !bad.load(Relaxed)
}

fn ch(c: char) -> c_char {
    c as u8 as c_char
}

#[pyfunction]
fn getrf<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f64>,
    chk: bool,
    threads: usize,
) -> Option<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<I>>)> {
    let (b, n) = (a.shape()[0], a.shape()[1]);
    let (av, mut out, mut piv) = (View::new(&a), vec![0.0; b * n * n], vec![0 as I; b * n]);
    let (o, p) = (Out(out.as_mut_ptr()), Out(piv.as_mut_ptr()));
    let ni = n as I;
    let ok = run(py, b, threads, 0, 0, |i, _, _| unsafe {
        let m = o.slice(i * n * n, n * n);
        let ip = p.slice(i * n, n);
        let mut info = 0;
        if !av.load(i, n, n, m, false, chk) {
            return false;
        }
        (lapack().getrf)(&ni, &ni, m.as_mut_ptr(), &ni, ip.as_mut_ptr(), &mut info);
        ip.iter_mut().for_each(|x| *x -= 1);
        info == 0
    });
    ok.then(|| (PyArray1::from_vec(py, out), PyArray1::from_vec(py, piv)))
}

#[pyfunction]
fn getrs<'py>(
    py: Python<'py>,
    lu: PyReadonlyArray3<'py, f64>,
    piv: PyReadonlyArray2<'py, I>,
    rhs: PyReadonlyArray3<'py, f64>,
    trans: char,
    chk: bool,
    threads: usize,
) -> Option<Bound<'py, PyArray1<f64>>> {
    let (b, n, k) = (lu.shape()[0], lu.shape()[1], rhs.shape()[2]);
    let (lv, bv, mut out) = (View::new(&lu), View::new(&rhs), vec![0.0; b * n * k]);
    let pv = piv.as_array();
    let (pp, ps) = (pv.as_ptr() as usize, [pv.strides()[0], pv.strides()[1]]);
    let o = Out(out.as_mut_ptr());
    let (ni, ki, tr) = (n as I, k as I, ch(trans));
    let ok = run(py, b, threads, n * n, n, |i, sf, si| unsafe {
        let x = o.slice(i * n * k, n * k);
        let mut info = 0;
        lv.load(i, n, n, sf, false, false);
        if !bv.load(i, n, k, x, false, chk) {
            return false;
        }
        for q in 0..n {
            let v = *(pp as *const I).offset(i as isize * ps[0] + q as isize * ps[1]);
            if v < 0 || v >= ni {
                return false;
            }
            si[q] = v + 1;
        }
        (lapack().getrs)(
            &tr,
            &ni,
            &ki,
            sf.as_ptr(),
            &ni,
            si.as_ptr(),
            x.as_mut_ptr(),
            &ni,
            &mut info,
        );
        info == 0
    });
    ok.then(|| PyArray1::from_vec(py, out))
}

#[pyfunction]
fn potrs<'py>(
    py: Python<'py>,
    c: PyReadonlyArray3<'py, f64>,
    rhs: PyReadonlyArray3<'py, f64>,
    uplo: char,
    chk: bool,
    threads: usize,
) -> Option<Bound<'py, PyArray1<f64>>> {
    let (b, n, k) = (c.shape()[0], c.shape()[1], rhs.shape()[2]);
    let (cv, bv, mut out) = (View::new(&c), View::new(&rhs), vec![0.0; b * n * k]);
    let o = Out(out.as_mut_ptr());
    let (ni, ki, up) = (n as I, k as I, ch(uplo));
    let ok = run(py, b, threads, n * n, 0, |i, sf, _| unsafe {
        let x = o.slice(i * n * k, n * k);
        let mut info = 0;
        if !(cv.load(i, n, n, sf, false, chk) & bv.load(i, n, k, x, false, chk)) {
            return false;
        }
        (lapack().potrs)(
            &up,
            &ni,
            &ki,
            sf.as_ptr(),
            &ni,
            x.as_mut_ptr(),
            &ni,
            &mut info,
        );
        info == 0
    });
    ok.then(|| PyArray1::from_vec(py, out))
}

/// `tin`: load `a` transposed (SciPy solves the transposed system when `a` is C-ordered).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn trtrs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f64>,
    rhs: PyReadonlyArray3<'py, f64>,
    uplo: char,
    trans: char,
    diag: char,
    tin: bool,
    chk: bool,
    threads: usize,
) -> Option<Bound<'py, PyArray1<f64>>> {
    let (b, n, k) = (a.shape()[0], a.shape()[1], rhs.shape()[2]);
    let (av, bv, mut out) = (View::new(&a), View::new(&rhs), vec![0.0; b * n * k]);
    let o = Out(out.as_mut_ptr());
    let (ni, ki, up, tr, dg) = (n as I, k as I, ch(uplo), ch(trans), ch(diag));
    let ok = run(py, b, threads, n * n, 0, |i, sf, _| unsafe {
        let x = o.slice(i * n * k, n * k);
        let mut info = 0;
        if !(av.load(i, n, n, sf, tin, chk) & bv.load(i, n, k, x, false, chk)) {
            return false;
        }
        (lapack().trtrs)(
            &up,
            &tr,
            &dg,
            &ni,
            &ki,
            sf.as_ptr(),
            &ni,
            x.as_mut_ptr(),
            &ni,
            &mut info,
        );
        info == 0
    });
    ok.then(|| PyArray1::from_vec(py, out))
}

type Eig<'py> = (Bound<'py, PyArray1<f64>>, Option<Bound<'py, PyArray1<f64>>>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn syevr<'py>(
    py: Python<'py>,
    a: PyReadonlyArray3<'py, f64>,
    uplo: char,
    wantv: bool,
    lwork: usize,
    liwork: usize,
    chk: bool,
    threads: usize,
) -> Option<Eig<'py>> {
    let (b, n) = (a.shape()[0], a.shape()[1]);
    let zn = if wantv { n * n } else { 1 };
    let (av, mut w, mut z) = (View::new(&a), vec![0.0; b * n], vec![0.0; b * zn]);
    let (wo, zo) = (Out(w.as_mut_ptr()), Out(z.as_mut_ptr()));
    let (ni, lw, liw, up) = (n as I, lwork as I, liwork as I, ch(uplo));
    let (jobz, range) = (ch(if wantv { 'V' } else { 'N' }), ch('A'));
    let ldz = if wantv { ni.max(1) } else { 1 };
    let ok = run(
        py,
        b,
        threads,
        n * n + lwork,
        2 * n + liwork,
        |i, sf, si| unsafe {
            let (a2, work) = sf.split_at_mut(n * n);
            let (isuppz, iwork) = si.split_at_mut(2 * n);
            let (wi, zi) = (wo.slice(i * n, n), zo.slice(i * zn, zn));
            let (mut info, mut m) = (0, 0);
            if !av.load(i, n, n, a2, false, chk) {
                return false;
            }
            (lapack().syevr)(
                &jobz,
                &range,
                &up,
                &ni,
                a2.as_mut_ptr(),
                &ni,
                &0.0,
                &1.0,
                &1,
                &ni,
                &0.0,
                &mut m,
                wi.as_mut_ptr(),
                zi.as_mut_ptr(),
                &ldz,
                isuppz.as_mut_ptr(),
                work.as_mut_ptr(),
                &lw,
                iwork.as_mut_ptr(),
                &liw,
                &mut info,
            );
            info == 0
        },
    );
    ok.then(|| {
        (
            PyArray1::from_vec(py, w),
            wantv.then(|| PyArray1::from_vec(py, z)),
        )
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(getrf, m)?)?;
    m.add_function(wrap_pyfunction!(getrs, m)?)?;
    m.add_function(wrap_pyfunction!(potrs, m)?)?;
    m.add_function(wrap_pyfunction!(trtrs, m)?)?;
    m.add_function(wrap_pyfunction!(syevr, m)?)?;
    Ok(())
}
