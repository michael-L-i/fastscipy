//! Array access and element conversion for the correlate family.
//!
//! Inputs are converted to double exactly as SciPy's `NI_ArrayToLineBuffer`
//! does (`(double)x`). Outputs are converted back with one of two flavors,
//! each reproducing what SciPy's compiled C casts do:
//!
//! * `line`: `NI_LineBufferToArray` (`correlate1d`, `uniform_filter1d`),
//!   a plain `(type)value` C cast.
//! * `safe`: `NI_Correlate` (`CASE_FILTER_OUT_SAFE`), which for unsigned
//!   outputs computes `v > -1 ? (type)v : -(type)(-v)`.
//!
//! Out-of-range double -> integer casts are undefined behaviour in C, so the
//! result depends on the instructions the compiler picked. On aarch64, clang
//! (SciPy's macOS arm64 wheels) converts to 8/16-bit types through a
//! saturating `fcvtzs` to int32 followed by truncation, to int32/int64 with
//! saturating `fcvtzs`, and to uint32/uint64 with saturating `fcvtzu`; this
//! is Rust's `as` with an intermediate i32 for the narrow types. On x86-64,
//! `cvttsd2si` returns the "integer indefinite" value (MIN) when out of range.
//! The Python shim verifies these against SciPy at first use and declines
//! integer outputs whose casts do not match.

use numpy::npyffi;
use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }
}

/// The element type of `a`, if it is a native-byte-order real type we handle.
pub fn dtype_of(py: Python<'_>, a: &Bound<'_, PyUntypedArray>) -> Option<DType> {
    let d = a.dtype();
    macro_rules! check {
        ($($t:ty => $v:ident),*) => {$(
            if d.is_equiv_to(&numpy::dtype::<$t>(py)) {
                return Some(DType::$v);
            }
        )*};
    }
    check!(f64 => F64, f32 => F32, u8 => U8, i8 => I8, u16 => U16, i16 => I16,
           u32 => U32, i32 => I32, u64 => U64, i64 => I64);
    None
}

/// Raw view of a C-contiguous, aligned array.
#[derive(Clone, Copy)]
pub struct Raw {
    pub ptr: *mut u8,
    pub dt: DType,
    pub nbytes: usize,
}

unsafe impl Send for Raw {}
unsafe impl Sync for Raw {}

impl Raw {
    /// Byte pointer to element `idx` (row-major linear index).
    #[inline(always)]
    pub fn at(&self, idx: usize) -> *mut u8 {
        unsafe { self.ptr.add(idx * self.dt.size()) }
    }

    pub fn overlaps(&self, other: &Raw) -> bool {
        let (a0, a1) = (self.ptr as usize, self.ptr as usize + self.nbytes);
        let (b0, b1) = (other.ptr as usize, other.ptr as usize + other.nbytes);
        a0 < b1 && b0 < a1
    }
}

/// A C-contiguous, aligned array of a supported dtype (writeable if `write`).
pub fn raw(py: Python<'_>, a: &Bound<'_, PyUntypedArray>, write: bool) -> Option<Raw> {
    let dt = dtype_of(py, a)?;
    if !a.is_c_contiguous() {
        return None;
    }
    let p = a.as_array_ptr();
    let flags = unsafe { (*p).fields.flags };
    if flags & npyffi::NPY_ARRAY_ALIGNED == 0 {
        return None;
    }
    if write && flags & npyffi::NPY_ARRAY_WRITEABLE == 0 {
        return None;
    }
    let n: usize = a.shape().iter().product();
    Some(Raw {
        ptr: unsafe { (*p).fields.data as *mut u8 },
        dt,
        nbytes: n * dt.size(),
    })
}

// ---------------------------------------------------------------------------
// Double -> integer conversions with the target's C cast behaviour.

#[cfg(not(target_arch = "x86_64"))]
mod conv {
    // aarch64 fcvtzs / fcvtzu: truncate, saturate, NaN -> 0 (Rust `as`).
    #[inline(always)]
    pub fn i32_(v: f64) -> i32 {
        v as i32
    }
    #[inline(always)]
    pub fn u32_(v: f64) -> u32 {
        v as u32
    }
    #[inline(always)]
    pub fn i64_(v: f64) -> i64 {
        v as i64
    }
    #[inline(always)]
    pub fn u64_(v: f64) -> u64 {
        v as u64
    }
}

#[cfg(target_arch = "x86_64")]
mod conv {
    // cvttsd2si: truncate; NaN or out of range -> MIN ("integer indefinite").
    #[inline(always)]
    pub fn i32_(v: f64) -> i32 {
        if v > -2147483649.0 && v < 2147483648.0 {
            v as i32
        } else {
            i32::MIN
        }
    }
    #[inline(always)]
    pub fn i64_(v: f64) -> i64 {
        if (-9223372036854775808.0..9223372036854775808.0).contains(&v) {
            v as i64
        } else {
            i64::MIN
        }
    }
    // double -> unsigned int goes through a 64-bit cvttsd2si.
    #[inline(always)]
    pub fn u32_(v: f64) -> u32 {
        i64_(v) as u32
    }
    #[inline(always)]
    pub fn u64_(v: f64) -> u64 {
        if v >= 9223372036854775808.0 {
            (i64_(v - 9223372036854775808.0) as u64) ^ (1u64 << 63)
        } else {
            i64_(v) as u64
        }
    }
}

/// Element types with SciPy's conversions to and from double.
pub trait Num: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
    /// `(type)v` as compiled in `NI_LineBufferToArray`.
    fn line(v: f64) -> Self;
    /// `CASE_FILTER_OUT[_SAFE]` as compiled in `NI_Correlate`.
    fn safe(v: f64) -> Self;
}

macro_rules! num_narrow {
    ($t:ty, unsigned) => {
        impl Num for $t {
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn line(v: f64) -> Self {
                conv::i32_(v) as $t
            }
            #[inline(always)]
            fn safe(v: f64) -> Self {
                if v > -1.0 {
                    conv::i32_(v) as $t
                } else {
                    (conv::i32_(-v) as $t).wrapping_neg()
                }
            }
        }
    };
    ($t:ty, signed) => {
        impl Num for $t {
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn line(v: f64) -> Self {
                conv::i32_(v) as $t
            }
            #[inline(always)]
            fn safe(v: f64) -> Self {
                conv::i32_(v) as $t
            }
        }
    };
}
num_narrow!(u8, unsigned);
num_narrow!(u16, unsigned);
num_narrow!(i8, signed);
num_narrow!(i16, signed);

impl Num for i32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn line(v: f64) -> Self {
        conv::i32_(v)
    }
    #[inline(always)]
    fn safe(v: f64) -> Self {
        conv::i32_(v)
    }
}
impl Num for i64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn line(v: f64) -> Self {
        conv::i64_(v)
    }
    #[inline(always)]
    fn safe(v: f64) -> Self {
        conv::i64_(v)
    }
}
impl Num for u32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn line(v: f64) -> Self {
        conv::u32_(v)
    }
    #[inline(always)]
    fn safe(v: f64) -> Self {
        if v > -1.0 {
            conv::u32_(v)
        } else {
            conv::u32_(-v).wrapping_neg()
        }
    }
}
impl Num for u64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn line(v: f64) -> Self {
        conv::u64_(v)
    }
    #[inline(always)]
    fn safe(v: f64) -> Self {
        if v > -1.0 {
            conv::u64_(v)
        } else {
            conv::u64_(-v).wrapping_neg()
        }
    }
}
impl Num for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn line(v: f64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn safe(v: f64) -> Self {
        v as f32
    }
}
impl Num for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn line(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn safe(v: f64) -> Self {
        v
    }
}

// ---------------------------------------------------------------------------
// Contiguous run conversions, selected once per call by dtype.

/// Convert `n` elements at `src` to double into `dst`.
pub type LoadFn = unsafe fn(src: *const u8, n: usize, dst: *mut f64);
/// Convert `n` doubles at `src` into `n` elements at `dst`.
pub type StoreFn = unsafe fn(src: *const f64, n: usize, dst: *mut u8);

// Slices (rather than raw pointers) tell the compiler the buffers do not
// alias, so these loops vectorize without runtime checks.

unsafe fn load_t<T: Num>(src: *const u8, n: usize, dst: *mut f64) {
    let (s, d) = unsafe {
        (
            std::slice::from_raw_parts(src as *const T, n),
            std::slice::from_raw_parts_mut(dst, n),
        )
    };
    for (d, &s) in d.iter_mut().zip(s) {
        *d = s.to_f64();
    }
}

unsafe fn store_line_t<T: Num>(src: *const f64, n: usize, dst: *mut u8) {
    let (s, d) = unsafe {
        (
            std::slice::from_raw_parts(src, n),
            std::slice::from_raw_parts_mut(dst as *mut T, n),
        )
    };
    for (d, &s) in d.iter_mut().zip(s) {
        *d = T::line(s);
    }
}

unsafe fn store_safe_t<T: Num>(src: *const f64, n: usize, dst: *mut u8) {
    let (s, d) = unsafe {
        (
            std::slice::from_raw_parts(src, n),
            std::slice::from_raw_parts_mut(dst as *mut T, n),
        )
    };
    for (d, &s) in d.iter_mut().zip(s) {
        *d = T::safe(s);
    }
}

macro_rules! by_dtype {
    ($dt:expr, $f:ident) => {
        match $dt {
            DType::U8 => $f::<u8> as _,
            DType::I8 => $f::<i8> as _,
            DType::U16 => $f::<u16> as _,
            DType::I16 => $f::<i16> as _,
            DType::U32 => $f::<u32> as _,
            DType::I32 => $f::<i32> as _,
            DType::U64 => $f::<u64> as _,
            DType::I64 => $f::<i64> as _,
            DType::F32 => $f::<f32> as _,
            DType::F64 => $f::<f64> as _,
        }
    };
}

pub fn loader(dt: DType) -> LoadFn {
    by_dtype!(dt, load_t)
}

pub fn store_line(dt: DType) -> StoreFn {
    by_dtype!(dt, store_line_t)
}

pub fn store_safe(dt: DType) -> StoreFn {
    by_dtype!(dt, store_safe_t)
}

/// Work (roughly, element operations) worth one more thread: waking pool
/// threads costs tens of microseconds, so small calls stay on one thread.
const PAR_GRAIN: usize = 1 << 20;

/// Threads worth using for `work` element operations.
pub fn threads_for(work: usize) -> usize {
    let t = crate::ndimage::common::num_threads();
    if t == 1 {
        1
    } else {
        (work / PAR_GRAIN).clamp(1, t)
    }
}

/// Run `f(state, task)` for every task in `0..n` on up to `threads` threads
/// (the calling thread plus pool threads), each with one `init()` state;
/// tasks are handed out in order from a shared counter. Only as many pool
/// threads as requested are woken.
pub fn for_each_task<S, I, F>(n: usize, threads: usize, init: I, f: F)
where
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) + Sync + Send,
{
    use std::sync::atomic::{AtomicUsize, Ordering};
    let threads = threads.min(n);
    if threads <= 1 {
        if n > 0 {
            let mut s = init();
            for t in 0..n {
                f(&mut s, t);
            }
        }
        return;
    }
    let next = AtomicUsize::new(0);
    let work = || {
        let mut s = None;
        loop {
            let t = next.fetch_add(1, Ordering::Relaxed);
            if t >= n {
                break;
            }
            f(s.get_or_insert_with(&init), t);
        }
    };
    crate::ndimage::common::pool().in_place_scope(|scope| {
        for _ in 1..threads {
            scope.spawn(|_| work());
        }
        work();
    });
}

/// Hint that `bytes` bytes from `p` will be read soon (into L2).
#[inline(always)]
pub fn prefetch_read(p: *const u8, bytes: usize) {
    let mut o = 0;
    while o < bytes {
        let a = p.wrapping_add(o);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("prfm pldl2keep, [{0}]", in(reg) a, options(nostack, readonly, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T1 }>(a as *const i8);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = a;
        o += 64;
    }
}

/// Hint that `bytes` bytes from `p` will be written soon (into L2).
#[inline(always)]
pub fn prefetch_write(p: *mut u8, bytes: usize) {
    let mut o = 0;
    while o < bytes {
        let a = p.wrapping_add(o);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("prfm pstl2keep, [{0}]", in(reg) a, options(nostack, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T1 }>(a as *const i8);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = a;
        o += 64;
    }
}

/// `Send`/`Sync` wrapper for raw pointers handed to worker threads.
pub struct Ptr<T>(pub *mut T);
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}
impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Ptr<T> {}
