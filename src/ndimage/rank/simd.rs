//! Minimal SIMD vectors for lane-wise min/max: NEON on aarch64, plain arrays
//! (left to the autovectorizer) elsewhere. Only used on values without NaN or
//! negative zero, where every min/max flavour gives identical bits.

use super::elem::RElem;

/// A vector of `LANES` elements of `T`.
pub trait Vx<T>: Copy + Send + Sync {
    const LANES: usize;
    /// # Safety
    /// `p` must be valid for reading `LANES` elements (any alignment).
    unsafe fn load(p: *const T) -> Self;
    /// # Safety
    /// `p` must be valid for writing `LANES` elements (any alignment).
    unsafe fn store(self, p: *mut T);
    fn mn(self, o: Self) -> Self;
    fn mx(self, o: Self) -> Self;
}

/// Portable vector: an array, min/max element by element.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct Arr<T, const N: usize>(pub [T; N]);

impl<T: RElem, const N: usize> Vx<T> for Arr<T, N> {
    const LANES: usize = N;
    #[inline(always)]
    unsafe fn load(p: *const T) -> Self {
        Arr(unsafe { std::ptr::read_unaligned(p as *const [T; N]) })
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut T) {
        unsafe { std::ptr::write_unaligned(p as *mut [T; N], self.0) }
    }
    #[inline(always)]
    fn mn(self, o: Self) -> Self {
        let mut r = self.0;
        for i in 0..N {
            r[i] = T::mn(self.0[i], o.0[i]);
        }
        Arr(r)
    }
    #[inline(always)]
    fn mx(self, o: Self) -> Self {
        let mut r = self.0;
        for i in 0..N {
            r[i] = T::mx(self.0[i], o.0[i]);
        }
        Arr(r)
    }
}

/// Two vectors processed together (more independent work per instruction
/// stream).
#[derive(Clone, Copy)]
pub struct X2<V>(pub V, pub V);

impl<T, V: Vx<T>> Vx<T> for X2<V> {
    const LANES: usize = 2 * V::LANES;
    #[inline(always)]
    unsafe fn load(p: *const T) -> Self {
        unsafe { X2(V::load(p), V::load(p.add(V::LANES))) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut T) {
        unsafe {
            self.0.store(p);
            self.1.store(p.add(V::LANES));
        }
    }
    #[inline(always)]
    fn mn(self, o: Self) -> Self {
        X2(self.0.mn(o.0), self.1.mn(o.1))
    }
    #[inline(always)]
    fn mx(self, o: Self) -> Self {
        X2(self.0.mx(o.0), self.1.mx(o.1))
    }
}

/// The native 128-bit vector type of each element type.
pub trait HasVec: RElem {
    type V: Vx<Self>;
    /// Use the generated networks (limited to common types to bound code size).
    const GEN: bool;
}

#[cfg(target_arch = "aarch64")]
mod native {
    use super::{HasVec, Vx};
    use std::arch::aarch64::*;

    macro_rules! neon {
        ($name:ident, $t:ty, $reg:ty, $n:expr, $ld:ident, $st:ident, $mn:ident, $mx:ident, $gen:expr) => {
            #[derive(Clone, Copy)]
            pub struct $name($reg);
            impl Vx<$t> for $name {
                const LANES: usize = $n;
                #[inline(always)]
                unsafe fn load(p: *const $t) -> Self {
                    unsafe { $name($ld(p)) }
                }
                #[inline(always)]
                unsafe fn store(self, p: *mut $t) {
                    unsafe { $st(p, self.0) }
                }
                #[inline(always)]
                fn mn(self, o: Self) -> Self {
                    unsafe { $name($mn(self.0, o.0)) }
                }
                #[inline(always)]
                fn mx(self, o: Self) -> Self {
                    unsafe { $name($mx(self.0, o.0)) }
                }
            }
            impl HasVec for $t {
                type V = $name;
                const GEN: bool = $gen;
            }
        };
    }
    neon!(
        U8x16, u8, uint8x16_t, 16, vld1q_u8, vst1q_u8, vminq_u8, vmaxq_u8, true
    );
    neon!(
        I8x16, i8, int8x16_t, 16, vld1q_s8, vst1q_s8, vminq_s8, vmaxq_s8, false
    );
    neon!(
        U16x8, u16, uint16x8_t, 8, vld1q_u16, vst1q_u16, vminq_u16, vmaxq_u16, true
    );
    neon!(
        I16x8, i16, int16x8_t, 8, vld1q_s16, vst1q_s16, vminq_s16, vmaxq_s16, true
    );
    neon!(
        U32x4, u32, uint32x4_t, 4, vld1q_u32, vst1q_u32, vminq_u32, vmaxq_u32, false
    );
    neon!(
        I32x4, i32, int32x4_t, 4, vld1q_s32, vst1q_s32, vminq_s32, vmaxq_s32, false
    );
    neon!(
        F32x4,
        f32,
        float32x4_t,
        4,
        vld1q_f32,
        vst1q_f32,
        vminq_f32,
        vmaxq_f32,
        true
    );
    neon!(
        F64x2,
        f64,
        float64x2_t,
        2,
        vld1q_f64,
        vst1q_f64,
        vminq_f64,
        vmaxq_f64,
        true
    );

    // 64-bit integers: compare and select
    #[inline(always)]
    unsafe fn min_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vbslq_u64(vcgtq_u64(a, b), b, a) }
    }
    #[inline(always)]
    unsafe fn max_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vbslq_u64(vcgtq_u64(a, b), a, b) }
    }
    #[inline(always)]
    unsafe fn min_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vbslq_s64(vcgtq_s64(a, b), b, a) }
    }
    #[inline(always)]
    unsafe fn max_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
        unsafe { vbslq_s64(vcgtq_s64(a, b), a, b) }
    }
    neon!(
        U64x2, u64, uint64x2_t, 2, vld1q_u64, vst1q_u64, min_u64, max_u64, false
    );
    neon!(
        I64x2, i64, int64x2_t, 2, vld1q_s64, vst1q_s64, min_s64, max_s64, false
    );
}

#[cfg(not(target_arch = "aarch64"))]
mod native {
    use super::{Arr, HasVec};
    macro_rules! portable {
        ($($t:ty => $n:expr, $gen:expr);*) => {$(
            impl HasVec for $t {
                type V = Arr<$t, $n>;
                const GEN: bool = $gen;
            }
        )*};
    }
    portable!(u8 => 16, true; i8 => 16, false; u16 => 8, true; i16 => 8, true; u32 => 4, false; i32 => 4, false;
              u64 => 2, false; i64 => 2, false; f32 => 4, true; f64 => 2, true);
}
