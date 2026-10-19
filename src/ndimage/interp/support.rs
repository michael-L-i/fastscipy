//! Exact ports of the scalar helpers in SciPy's `ni_interpolation.c` and
//! `ni_splines.c` (boundary mapping, spline weights, filter poles) plus the
//! output conversions, shared by the interpolation kernels.

use crate::ndimage::common::Mode;

// ---------------------------------------------------------------------------
// Floating-point contraction
//
// SciPy's macOS arm64 wheel is built by clang with its default
// `-ffp-contract=on`, which fuses `a * b + c` written within one C expression
// into a single `fmadd` (never across statements). Disassembling
// `_nd_image.cpython-313-darwin.so` (SciPy 1.18.1) shows fused multiply-adds
// in `get_spline_interpolation_weights`, in every spline-filter recursion and
// initialisation, and in the affine coordinate loop of NI_GeometricTransform,
// while the interpolation sum (`coeff *= w; t += coeff`) is separate fmul and
// fadd. The randomized tests confirm that fusing exactly those expressions
// reproduces SciPy bit for bit; computing them unfused does not. x86-64
// baseline wheels have no FMA instructions, so there the product is rounded
// before the addition. (Linux aarch64 wheels are built with GCC, whose
// contraction may differ; only macOS arm64 was verified.)

#[cfg(target_arch = "aarch64")]
pub const FUSED: bool = true;
#[cfg(not(target_arch = "aarch64"))]
pub const FUSED: bool = false;

/// `a * b + c` as SciPy computes a contracted C expression.
#[inline(always)]
pub fn madd(a: f64, b: f64, c: f64) -> f64 {
    if FUSED { a.mul_add(b, c) } else { a * b + c }
}

/// `c - a * b` as SciPy computes a contracted C expression.
#[inline(always)]
pub fn nmadd(a: f64, b: f64, c: f64) -> f64 {
    if FUSED { (-a).mul_add(b, c) } else { c - a * b }
}

/// libm `pow`, as called by SciPy (the exponent is an integer converted to
/// double; `black_box` stops LLVM from rewriting the call as `powi`).
#[inline(never)]
pub fn libm_pow(x: f64, n: i64) -> f64 {
    unsafe extern "C" {
        fn pow(x: f64, y: f64) -> f64;
    }
    let y = std::hint::black_box(n as f64);
    unsafe { pow(x, y) }
}

// ---------------------------------------------------------------------------
// Boundary mapping: map_coordinate() of ni_interpolation.c.
//
// Integer conversions use Rust `as` (truncation, saturation, NaN -> 0), which
// matches fcvtzs on aarch64; integer arithmetic wraps like the compiled C.

#[inline]
pub fn map_coordinate(mut x: f64, len: i64, mode: Mode) -> f64 {
    if x < 0.0 {
        match mode {
            Mode::Mirror => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz2 = 2 * len - 2;
                    x = sz2.wrapping_mul((-x / sz2 as f64) as i64) as f64 + x;
                    x = if x <= (1 - len) as f64 {
                        x + sz2 as f64
                    } else {
                        -x
                    };
                }
            }
            Mode::Reflect => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz2 = 2 * len;
                    if x < -(sz2 as f64) {
                        x = sz2.wrapping_mul((-x / sz2 as f64) as i64) as f64 + x;
                    }
                    x = if x < -(len as f64) {
                        x + sz2 as f64
                    } else {
                        (if x > -1e-15 { 1e-15 } else { -x }) - 1.0
                    };
                }
            }
            Mode::Wrap => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz = len - 1;
                    x += sz.wrapping_mul(((-x / sz as f64) as i64).wrapping_add(1)) as f64;
                }
            }
            Mode::GridWrap => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    x +=
                        len.wrapping_mul((((-1.0 - x) / len as f64) as i64).wrapping_add(1)) as f64;
                }
            }
            Mode::Nearest => x = 0.0,
            Mode::Constant => x = -1.0,
            Mode::GridConstant => {}
        }
    } else if x > (len - 1) as f64 {
        match mode {
            Mode::Mirror => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz2 = 2 * len - 2;
                    x -= sz2.wrapping_mul((x / sz2 as f64) as i64) as f64;
                    if x >= len as f64 {
                        x = sz2 as f64 - x;
                    }
                }
            }
            Mode::Reflect => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz2 = 2 * len;
                    x -= sz2.wrapping_mul((x / sz2 as f64) as i64) as f64;
                    if x >= len as f64 {
                        x = sz2 as f64 - x - 1.0;
                    }
                }
            }
            Mode::Wrap => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    let sz = len - 1;
                    x -= sz.wrapping_mul((x / sz as f64) as i64) as f64;
                }
            }
            Mode::GridWrap => {
                if len <= 1 {
                    x = 0.0;
                } else {
                    x -= len.wrapping_mul((x / len as f64) as i64) as f64;
                }
            }
            Mode::Nearest => x = (len - 1) as f64,
            Mode::Constant => x = -1.0,
            Mode::GridConstant => {}
        }
    }
    x
}

/// `_get_spline_boundary_mode`: modes without an analytic prefilter or
/// explicit prepadding use mirror extension for the spline taps.
#[inline]
pub fn spline_mode(mode: Mode) -> Mode {
    match mode {
        Mode::Constant | Mode::Wrap => Mode::Mirror,
        m => m,
    }
}

/// First tap of the spline footprint around coordinate `cc`.
#[inline(always)]
pub fn spline_start(cc: f64, order: usize) -> i64 {
    let f = if order & 1 == 1 {
        cc.floor()
    } else {
        (cc + 0.5).floor()
    };
    (f as i64).wrapping_sub((order / 2) as i64)
}

// ---------------------------------------------------------------------------
// get_spline_interpolation_weights(), with the contractions clang applies.

#[inline(always)]
pub fn spline_weights(cc: f64, order: usize, w: &mut [f64; 6]) {
    let x = cc
        - if order & 1 == 1 {
            cc.floor()
        } else {
            (cc + 0.5).floor()
        };
    let z = 1.0 - x;
    match order {
        1 => {
            w[0] = z;
        }
        2 => {
            w[1] = nmadd(x, x, 0.75);
            let y = 0.5 - x;
            w[0] = 0.5 * y * y;
        }
        3 => {
            let y = x;
            w[1] = madd(y * y * (y - 2.0), 3.0, 4.0) / 6.0;
            w[2] = madd(z * z * (z - 2.0), 3.0, 4.0) / 6.0;
            w[0] = z * z * z / 6.0;
        }
        4 => {
            let t = x * x;
            w[2] = madd(t, madd(t, 0.25, -0.625), 115.0 / 192.0);
            let y = 1.0 + x;
            w[1] = madd(
                y,
                madd(y, y * (5.0 - y) / 6.0 - 1.25, 5.0 / 24.0),
                55.0 / 96.0,
            );
            w[3] = madd(
                z,
                madd(z, z * (5.0 - z) / 6.0 - 1.25, 5.0 / 24.0),
                55.0 / 96.0,
            );
            let y = 0.5 - x;
            let t = y * y;
            w[0] = t * t / 24.0;
        }
        5 => {
            let y = x;
            let t = y * y;
            w[2] = madd(t, madd(t, 0.25 - y / 12.0, -0.5), 0.55);
            let t = z * z;
            w[3] = madd(t, madd(t, 0.25 - z / 12.0, -0.5), 0.55);
            let y1 = y + 1.0;
            w[1] = madd(
                y1,
                madd(
                    y1,
                    madd(y1, madd(y1, y1 / 24.0 - 0.375, 1.25), -1.75),
                    0.625,
                ),
                0.425,
            );
            let z1 = z + 1.0;
            w[4] = madd(
                z1,
                madd(
                    z1,
                    madd(z1, madd(z1, z1 / 24.0 - 0.375, 1.25), -1.75),
                    0.625,
                ),
                0.425,
            );
            let y = 1.0 - x;
            let t = y * y;
            w[0] = y * t * t / 120.0;
        }
        _ => return,
    }
    let mut last = 1.0;
    for &wi in &w[..order] {
        last -= wi;
    }
    w[order] = last;
}

// ---------------------------------------------------------------------------
// Spline filter poles: get_filter_poles().

pub fn filter_poles(order: usize) -> Option<&'static [f64]> {
    const P2: [f64; 1] = [-0.171572875253809902396622551580603843];
    const P3: [f64; 1] = [-0.267949192431122706472553658494127633];
    const P4: [f64; 2] = [
        -0.361341225900220177092212841325675255,
        -0.013725429297339121360331226939128204,
    ];
    const P5: [f64; 2] = [
        -0.430575347099973791851434783493520110,
        -0.043096288203264653822712376822550182,
    ];
    match order {
        2 => Some(&P2),
        3 => Some(&P3),
        4 => Some(&P4),
        5 => Some(&P5),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Output conversions

/// Element types the interpolation kernels write.
pub trait OutElem: Copy + Send + Sync + 'static {
    /// CASE_INTERP_OUT*: integers round half away from zero and clamp.
    fn from_interp(t: f64) -> Self;
    /// CASE_COPY_LINE_TO_DATA: a plain C cast, as clang compiles it for
    /// aarch64 (8- and 16-bit types go through a saturating 32-bit signed
    /// conversion and keep the low bits).
    fn from_line(d: f64) -> Self;
    const IS_INT: bool;
    const IS_SIGNED: bool;
    const IS_U64: bool;
}

/// Truncating conversion of a value already clamped to the type's range.
/// 8- to 32-bit types use an unchecked cast (it vectorizes; the clamps make
/// it exact, and NaN has been mapped to 0), 64-bit types keep Rust's
/// saturating cast because their maximum rounds up to 2^63 / 2^64.
macro_rules! clamped_cast {
    ($t:ty, $v:expr) => {{
        let v: f64 = $v;
        if std::mem::size_of::<$t>() < 8 {
            // SAFETY: v is finite and within [MIN, MAX + 1) of the type.
            unsafe { v.to_int_unchecked::<$t>() }
        } else {
            v as $t
        }
    }};
}

macro_rules! out_uint {
    ($t:ty, $line:expr, $u64:expr) => {
        impl OutElem for $t {
            #[inline(always)]
            fn from_interp(t: f64) -> Self {
                let t = if t > 0.0 { t + 0.5 } else { 0.0 };
                let t = if t > <$t>::MAX as f64 {
                    <$t>::MAX as f64
                } else {
                    t
                };
                let t = if t < 0.0 { 0.0 } else { t };
                clamped_cast!($t, t)
            }
            #[inline(always)]
            fn from_line(d: f64) -> Self {
                $line(d)
            }
            const IS_INT: bool = true;
            const IS_SIGNED: bool = false;
            const IS_U64: bool = $u64;
        }
    };
}

macro_rules! out_sint {
    ($t:ty, $line:expr) => {
        impl OutElem for $t {
            #[inline(always)]
            fn from_interp(t: f64) -> Self {
                let t = if t > 0.0 { t + 0.5 } else { t - 0.5 };
                let t = if t > <$t>::MAX as f64 {
                    <$t>::MAX as f64
                } else {
                    t
                };
                let t = if t < <$t>::MIN as f64 {
                    <$t>::MIN as f64
                } else {
                    t
                };
                // NaN survives the clamps; aarch64 converts it to 0.
                let t = if t != t { 0.0 } else { t };
                clamped_cast!($t, t)
            }
            #[inline(always)]
            fn from_line(d: f64) -> Self {
                $line(d)
            }
            const IS_INT: bool = true;
            const IS_SIGNED: bool = true;
            const IS_U64: bool = false;
        }
    };
}

out_uint!(u8, |d: f64| (d as i32) as u8, false);
out_uint!(u16, |d: f64| (d as i32) as u16, false);
out_uint!(u32, |d: f64| d as u32, false);
out_uint!(u64, |d: f64| d as u64, true);
out_sint!(i8, |d: f64| (d as i32) as i8);
out_sint!(i16, |d: f64| (d as i32) as i16);
out_sint!(i32, |d: f64| d as i32);
out_sint!(i64, |d: f64| d as i64);

impl OutElem for f32 {
    #[inline(always)]
    fn from_interp(t: f64) -> Self {
        t as f32
    }
    #[inline(always)]
    fn from_line(d: f64) -> Self {
        d as f32
    }
    const IS_INT: bool = false;
    const IS_SIGNED: bool = false;
    const IS_U64: bool = false;
}

impl OutElem for f64 {
    #[inline(always)]
    fn from_interp(t: f64) -> Self {
        t
    }
    #[inline(always)]
    fn from_line(d: f64) -> Self {
        d
    }
    const IS_INT: bool = false;
    const IS_SIGNED: bool = false;
    const IS_U64: bool = false;
}

/// Convert a block of interpolated values into the output.
#[inline]
pub fn store_interp<O: OutElem>(dst: &mut [O], src: &[f64]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = O::from_interp(s);
    }
}

/// Whether an integer output conversion could reach behaviour that is
/// undefined in C (NaN to a signed integer, u64 saturation at 2^64). Only
/// SciPy's aarch64 build was verified, so other targets decline such calls:
/// u64 output always, signed output when NaN could occur (float input or
/// non-finite parameters). Unsigned clamps map NaN to 0 explicitly.
#[inline]
pub fn int_output_risky<O: OutElem>(input_is_float: bool, params_finite: bool) -> bool {
    if cfg!(target_arch = "aarch64") {
        return false;
    }
    O::IS_U64 || (O::IS_SIGNED && (input_is_float || !params_finite))
}

// ---------------------------------------------------------------------------
// Raw pointers that may cross into the thread pool.

/// (Closures must call `get()` so they capture the wrapper, not the field.)
pub struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline(always)]
    pub fn get(self) -> *mut T {
        self.0
    }
}

pub struct SendConstPtr<T>(pub *const T);
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}
impl<T> Clone for SendConstPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendConstPtr<T> {}
impl<T> SendConstPtr<T> {
    #[inline(always)]
    pub fn get(self) -> *const T {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Serial or parallel loops. Small calls run on the calling thread: handing
// them to the pool costs more (tens of microseconds) than the work itself.

/// `f(i)` for `i` in `0..n`, in the current rayon pool when `par`.
pub fn for_each_index(n: usize, par: bool, f: impl Fn(usize) + Sync + Send) {
    use rayon::prelude::*;
    if par {
        (0..n).into_par_iter().for_each(f);
    } else {
        (0..n).for_each(f);
    }
}

/// Like `for_each_index`, with per-worker state made by `init`.
pub fn for_each_index_init<T>(
    n: usize,
    par: bool,
    init: impl Fn() -> T + Sync + Send,
    f: impl Fn(&mut T, usize) + Sync + Send,
) {
    use rayon::prelude::*;
    if par {
        (0..n).into_par_iter().for_each_init(init, f);
    } else {
        let mut state = init();
        for i in 0..n {
            f(&mut state, i);
        }
    }
}

/// `f(index, chunk)` for the `size`-element chunks of `data`.
pub fn for_each_chunk<T: Send>(
    data: &mut [T],
    size: usize,
    par: bool,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    use rayon::prelude::*;
    if par {
        data.par_chunks_mut(size)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
    } else {
        data.chunks_mut(size).enumerate().for_each(|(i, c)| f(i, c));
    }
}

/// Whether a call doing about `work` elementary operations is worth
/// running in the thread pool.
pub fn worth_parallel(work: usize) -> bool {
    work >= 1 << 17 && crate::ndimage::common::num_threads() > 1
}
