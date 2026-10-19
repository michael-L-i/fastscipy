//! Shared pieces: boundary-mode codes, element types, dtype dispatch.

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;

pub use crate::pool::{num_threads, pool};

// ---------------------------------------------------------------------------
// Boundary modes, using the integer codes of scipy.ndimage._ni_support._extend_mode_to_code

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Nearest = 0,
    Wrap = 1,
    Reflect = 2,
    Mirror = 3,
    Constant = 4,
    GridWrap = 5,
    GridConstant = 6,
}

impl Mode {
    pub fn from_code(code: i64) -> Option<Mode> {
        Some(match code {
            0 => Mode::Nearest,
            1 => Mode::Wrap,
            2 => Mode::Reflect,
            3 => Mode::Mirror,
            4 => Mode::Constant,
            5 => Mode::GridWrap,
            6 => Mode::GridConstant,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// Element types

/// Real element types handled natively. Others (bool, float16, complex,
/// non-native byte order) fall back to SciPy.
pub trait Elem: numpy::Element + Copy + Send + Sync + PartialOrd + 'static {
    fn to_f64(self) -> f64;
    /// Conversion from double with Rust `as` semantics: truncation toward zero,
    /// saturating out-of-range values (matching aarch64 C casts), NaN to 0 for ints.
    fn from_f64(v: f64) -> Self;
    const IS_FLOAT: bool;
}

macro_rules! impl_elem {
    ($float:expr; $($t:ty),*) => {$(
        impl Elem for $t {
            #[inline(always)]
            fn to_f64(self) -> f64 { self as f64 }
            #[inline(always)]
            fn from_f64(v: f64) -> Self { v as $t }
            const IS_FLOAT: bool = $float;
        }
    )*};
}
impl_elem!(false; u8, i8, u16, i16, u32, i32, u64, i64);
impl_elem!(true; f32, f64);

/// Run `$body` with `$T` bound to the Rust type of `$descr`, or evaluate
/// `$otherwise` when the dtype is not one of the native real types.
#[macro_export]
macro_rules! dispatch_elem {
    ($py:expr, $descr:expr, $T:ident => $body:block else $otherwise:expr) => {{
        #[allow(unused_imports)]
        use numpy::PyArrayDescrMethods as _;
        let __d = $descr;
        if __d.is_equiv_to(&numpy::dtype::<u8>($py)) {
            type $T = u8;
            $body
        } else if __d.is_equiv_to(&numpy::dtype::<i8>($py)) {
            type $T = i8;
            $body
        } else if __d.is_equiv_to(&numpy::dtype::<u16>($py)) {
            type $T = u16;
            $body
        } else if __d.is_equiv_to(&numpy::dtype::<i16>($py)) {
            type $T = i16;
            $body
        } else if __d.is_equiv_to(&numpy::dtype::<u32>($py)) {
            type $T = u32;
            $body
        } else if __d.is_equiv_to(&numpy::dtype::<i32>($py)) {
            type $T = i32;
            $body
        } else if __d.is_equiv_to(&numpy::dtype::<u64>($py)) {
            type $T = u64;
            $body
        } else if __d.is_equiv_to(&numpy::dtype::<i64>($py)) {
            type $T = i64;
            $body
        } else if __d.is_equiv_to(&numpy::dtype::<f32>($py)) {
            type $T = f32;
            $body
        } else if __d.is_equiv_to(&numpy::dtype::<f64>($py)) {
            type $T = f64;
            $body
        } else {
            $otherwise
        }
    }};
}

/// True when the array is C-contiguous, aligned and writeable-agnostic usable.
pub fn usable(a: &Bound<'_, PyUntypedArray>) -> bool {
    a.is_c_contiguous()
}

/// View an untyped array as a typed one (caller has checked the dtype).
pub fn typed<'py, T: Elem>(a: &Bound<'py, PyUntypedArray>) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    Ok(a.cast::<PyArrayDyn<T>>()?.clone())
}

/// Read-only slice of a C-contiguous typed array.
///
/// # Safety
/// The caller must not hold a mutable view of the same array.
pub unsafe fn slice<'a, T: Elem>(a: &'a Bound<'_, PyArrayDyn<T>>) -> &'a [T] {
    unsafe { a.as_slice().expect("array must be C-contiguous") }
}

/// Mutable slice of a C-contiguous typed array.
///
/// # Safety
/// The caller must ensure no other view of the array is alive, and that the
/// array does not overlap the input (the Python shim guarantees this).
#[allow(clippy::mut_from_ref)] // exclusivity is the caller's obligation, as above
pub unsafe fn slice_mut<'a, T: Elem>(a: &'a Bound<'_, PyArrayDyn<T>>) -> &'a mut [T] {
    unsafe { a.as_slice_mut().expect("array must be C-contiguous") }
}
