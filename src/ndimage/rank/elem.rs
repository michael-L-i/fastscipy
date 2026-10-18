//! Element traits for the rank family: fast min/max, detection of values whose
//! selection result depends on SciPy's comparison order, and the exact C
//! conversions SciPy applies between doubles and array element types.

use crate::ndimage::common::Elem;

/// Element type handled by the rank kernels.
pub trait RElem: Elem + Default + std::fmt::Debug {
    /// Minimum / maximum used by the fast paths. Only called on values that
    /// are not `special`, where every correct selection gives identical bits.
    fn mn(a: Self, b: Self) -> Self;
    fn mx(a: Self, b: Self) -> Self;
    /// NaN or negative zero: values for which two "equal" elements may differ
    /// in bits, or comparisons are not a total order. Always false for ints.
    fn is_special(self) -> bool;
    /// True when every value converts to f64 exactly (all but 64-bit ints).
    const EXACT_F64: bool;
    /// `(T)v` in C, or None when that conversion is undefined behaviour.
    fn c_cast(v: f64) -> Option<Self>;
    /// Lowest and highest value.
    const LOWEST: Self;
    const HIGHEST: Self;
    /// Total order for values that are not special.
    fn cmp_fast(a: &Self, b: &Self) -> std::cmp::Ordering;
    fn is_float() -> bool {
        <Self as Elem>::IS_FLOAT
    }
    /// 256 for 8-bit types, whose values map order-preservingly to histogram
    /// bins with `to_bin` / `from_bin`; 0 otherwise.
    const BINS: usize = 0;
    #[inline(always)]
    fn to_bin(self) -> usize {
        unreachable!()
    }
    #[inline(always)]
    fn from_bin(_b: usize) -> Self {
        unreachable!()
    }
}

/// Output conversions from double, with SciPy's two flavours.
pub trait OutElem: Elem {
    /// `CASE_FILTER_OUT_SAFE` for unsigned types, `CASE_FILTER_OUT` otherwise
    /// (nd filters). None when C would invoke undefined behaviour.
    fn conv_filter(v: f64) -> Option<Self>;
    /// `CASE_COPY_LINE_TO_DATA` (plain cast, 1-D line buffers).
    fn conv_line(v: f64) -> Option<Self>;
}

macro_rules! impl_unsigned {
    ($($t:ty $(=> $flip:expr)?),*) => {$(
        impl RElem for $t {
            #[inline(always)] fn mn(a: Self, b: Self) -> Self { a.min(b) }
            #[inline(always)] fn mx(a: Self, b: Self) -> Self { a.max(b) }
            #[inline(always)] fn is_special(self) -> bool { false }
            const EXACT_F64: bool = <$t>::BITS <= 32;
            #[inline] fn c_cast(v: f64) -> Option<Self> { <$t as OutElem>::conv_line(v) }
            const LOWEST: Self = <$t>::MIN;
            const HIGHEST: Self = <$t>::MAX;
            #[inline(always)] fn cmp_fast(a: &Self, b: &Self) -> std::cmp::Ordering { a.cmp(b) }
            $(
                const BINS: usize = 256;
                #[inline(always)] fn to_bin(self) -> usize { (self as u8 ^ $flip) as usize }
                #[inline(always)] fn from_bin(b: usize) -> Self { (b as u8 ^ $flip) as $t }
            )?
        }
        impl OutElem for $t {
            #[inline]
            fn conv_filter(v: f64) -> Option<Self> {
                // C: v > -1. ? (T)v : -(T)(-v)
                let lim = 2f64.powi(<$t>::BITS as i32);
                if v > -1.0 {
                    if v < lim { Some(v as $t) } else { None }
                } else {
                    let w = -v;
                    if w < lim { Some((w as $t).wrapping_neg()) } else { None }
                }
            }
            #[inline]
            fn conv_line(v: f64) -> Option<Self> {
                let lim = 2f64.powi(<$t>::BITS as i32);
                if v > -1.0 && v < lim { Some(v as $t) } else { None }
            }
        }
    )*};
}

macro_rules! impl_signed {
    ($($t:ty $(=> $flip:expr)?),*) => {$(
        impl RElem for $t {
            #[inline(always)] fn mn(a: Self, b: Self) -> Self { a.min(b) }
            #[inline(always)] fn mx(a: Self, b: Self) -> Self { a.max(b) }
            #[inline(always)] fn is_special(self) -> bool { false }
            const EXACT_F64: bool = <$t>::BITS <= 32;
            #[inline] fn c_cast(v: f64) -> Option<Self> { <$t as OutElem>::conv_line(v) }
            const LOWEST: Self = <$t>::MIN;
            const HIGHEST: Self = <$t>::MAX;
            #[inline(always)] fn cmp_fast(a: &Self, b: &Self) -> std::cmp::Ordering { a.cmp(b) }
            $(
                const BINS: usize = 256;
                #[inline(always)] fn to_bin(self) -> usize { (self as u8 ^ $flip) as usize }
                #[inline(always)] fn from_bin(b: usize) -> Self { (b as u8 ^ $flip) as $t }
            )?
        }
        impl OutElem for $t {
            #[inline]
            fn conv_filter(v: f64) -> Option<Self> { Self::conv_line(v) }
            #[inline]
            fn conv_line(v: f64) -> Option<Self> {
                let hi = 2f64.powi(<$t>::BITS as i32 - 1); // exclusive
                let lo = -hi; // inclusive (exact power of two)
                // truncation toward zero lands in range iff lo - 1 < v < hi
                let ok = if <$t>::BITS >= 53 { v >= lo && v < hi } else { v > lo - 1.0 && v < hi };
                if ok { Some(v as $t) } else { None }
            }
        }
    )*};
}

impl_unsigned!(u8 => 0u8, u16, u32, u64);
impl_signed!(i8 => 0x80u8, i16, i32, i64);

macro_rules! impl_float {
    ($($t:ty: $bits:ty),*) => {$(
        impl RElem for $t {
            #[inline(always)] fn mn(a: Self, b: Self) -> Self { if a < b { a } else { b } }
            #[inline(always)] fn mx(a: Self, b: Self) -> Self { if a > b { a } else { b } }
            #[inline(always)] fn is_special(self) -> bool {
                // NaN: magnitude bits above infinity; -0.0: only the sign bit
                let b = self.to_bits();
                let sign: $bits = 1 << (<$bits>::BITS - 1);
                ((b & !sign) > <$t>::INFINITY.to_bits()) | (b == sign)
            }
            const EXACT_F64: bool = true;
            #[inline] fn c_cast(v: f64) -> Option<Self> { Some(v as $t) }
            const LOWEST: Self = <$t>::NEG_INFINITY;
            const HIGHEST: Self = <$t>::INFINITY;
            #[inline(always)] fn cmp_fast(a: &Self, b: &Self) -> std::cmp::Ordering {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
        impl OutElem for $t {
            #[inline] fn conv_filter(v: f64) -> Option<Self> { Some(v as $t) }
            #[inline] fn conv_line(v: f64) -> Option<Self> { Some(v as $t) }
        }
    )*};
}
impl_float!(f32: u32, f64: u64);

/// Summary of the values in an array, used to decide between the fast path,
/// the exact (SciPy-order) path, and declining.
#[derive(Clone, Copy, Debug)]
pub struct Scan {
    /// Contains NaN or negative zero.
    pub special: bool,
    /// Smallest and largest non-NaN values, as f64 (None for an empty array
    /// or all-NaN).
    pub lo: f64,
    pub hi: f64,
}

/// Parallel scan of `data` for special values and range.
pub fn scan<T: RElem>(data: &[T], want_range: bool) -> Scan {
    let chunk = 1 << 16;
    let (special, lo, hi) = super::par::map_reduce(
        data,
        chunk,
        |c| {
            if want_range {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                let mut sp = false;
                // min/max in T domain, then convert
                let mut tlo = T::HIGHEST;
                let mut thi = T::LOWEST;
                let mut any = false;
                for &v in c {
                    if v.is_special() {
                        sp = true;
                        if v.partial_cmp(&v).is_none() {
                            continue;
                        }
                    }
                    any = true;
                    if v < tlo {
                        tlo = v;
                    }
                    if v > thi {
                        thi = v;
                    }
                }
                if any {
                    lo = tlo.to_f64();
                    hi = thi.to_f64();
                }
                (sp, lo, hi)
            } else if T::is_float() {
                let mut sp = false;
                // branch-free OR over the chunk; vectorizes
                for &v in c {
                    sp |= v.is_special();
                }
                (sp, f64::INFINITY, f64::NEG_INFINITY)
            } else {
                (false, f64::INFINITY, f64::NEG_INFINITY)
            }
        },
        || (false, f64::INFINITY, f64::NEG_INFINITY),
        |a, b| (a.0 | b.0, a.1.min(b.1), a.2.max(b.2)),
    );
    Scan { special, lo, hi }
}
