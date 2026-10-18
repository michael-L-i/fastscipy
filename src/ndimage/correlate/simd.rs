//! A minimal lane abstraction: `V` holds two doubles in a SIMD register
//! (NEON on aarch64, SSE2 on x86-64, an array elsewhere); `f64` is the
//! one-lane version used for tails. Kernels are written once over `Lane`.
//!
//! All operations are IEEE double operations applied lane by lane, so a
//! kernel's result does not depend on the lane width.

pub trait Lane: Copy {
    const W: usize;
    unsafe fn load(p: *const f64) -> Self;
    unsafe fn store(self, p: *mut f64);
    fn splat(x: f64) -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn div(self, o: Self) -> Self;
    /// `self + a * b` with a single rounding.
    fn fma(self, a: Self, b: Self) -> Self;
}

impl Lane for f64 {
    const W: usize = 1;
    #[inline(always)]
    unsafe fn load(p: *const f64) -> f64 {
        unsafe { *p }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        unsafe { *p = self }
    }
    #[inline(always)]
    fn splat(x: f64) -> f64 {
        x
    }
    #[inline(always)]
    fn add(self, o: f64) -> f64 {
        self + o
    }
    #[inline(always)]
    fn sub(self, o: f64) -> f64 {
        self - o
    }
    #[inline(always)]
    fn mul(self, o: f64) -> f64 {
        self * o
    }
    #[inline(always)]
    fn div(self, o: f64) -> f64 {
        self / o
    }
    #[inline(always)]
    fn fma(self, a: f64, b: f64) -> f64 {
        a.mul_add(b, self)
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use super::Lane;
    use std::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct V(float64x2_t);

    #[inline(always)]
    pub fn zip(a: V, b: V) -> (V, V) {
        unsafe { (V(vzip1q_f64(a.0, b.0)), V(vzip2q_f64(a.0, b.0))) }
    }

    impl Lane for V {
        const W: usize = 2;
        #[inline(always)]
        unsafe fn load(p: *const f64) -> V {
            unsafe { V(vld1q_f64(p)) }
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f64) {
            unsafe { vst1q_f64(p, self.0) }
        }
        #[inline(always)]
        fn splat(x: f64) -> V {
            unsafe { V(vdupq_n_f64(x)) }
        }
        #[inline(always)]
        fn add(self, o: V) -> V {
            unsafe { V(vaddq_f64(self.0, o.0)) }
        }
        #[inline(always)]
        fn sub(self, o: V) -> V {
            unsafe { V(vsubq_f64(self.0, o.0)) }
        }
        #[inline(always)]
        fn mul(self, o: V) -> V {
            unsafe { V(vmulq_f64(self.0, o.0)) }
        }
        #[inline(always)]
        fn div(self, o: V) -> V {
            unsafe { V(vdivq_f64(self.0, o.0)) }
        }
        #[inline(always)]
        fn fma(self, a: V, b: V) -> V {
            unsafe { V(vfmaq_f64(self.0, a.0, b.0)) }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use super::Lane;
    use std::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct V(__m128d);

    #[inline(always)]
    pub fn zip(a: V, b: V) -> (V, V) {
        unsafe { (V(_mm_unpacklo_pd(a.0, b.0)), V(_mm_unpackhi_pd(a.0, b.0))) }
    }

    impl Lane for V {
        const W: usize = 2;
        #[inline(always)]
        unsafe fn load(p: *const f64) -> V {
            unsafe { V(_mm_loadu_pd(p)) }
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f64) {
            unsafe { _mm_storeu_pd(p, self.0) }
        }
        #[inline(always)]
        fn splat(x: f64) -> V {
            unsafe { V(_mm_set1_pd(x)) }
        }
        #[inline(always)]
        fn add(self, o: V) -> V {
            unsafe { V(_mm_add_pd(self.0, o.0)) }
        }
        #[inline(always)]
        fn sub(self, o: V) -> V {
            unsafe { V(_mm_sub_pd(self.0, o.0)) }
        }
        #[inline(always)]
        fn mul(self, o: V) -> V {
            unsafe { V(_mm_mul_pd(self.0, o.0)) }
        }
        #[inline(always)]
        fn div(self, o: V) -> V {
            unsafe { V(_mm_div_pd(self.0, o.0)) }
        }
        #[inline(always)]
        fn fma(self, a: V, b: V) -> V {
            // Baseline x86-64 has no FMA instruction; f64::mul_add is exact.
            let (mut s, mut x, mut y) = ([0.0f64; 2], [0.0f64; 2], [0.0f64; 2]);
            unsafe {
                _mm_storeu_pd(s.as_mut_ptr(), self.0);
                _mm_storeu_pd(x.as_mut_ptr(), a.0);
                _mm_storeu_pd(y.as_mut_ptr(), b.0);
                V(_mm_set_pd(x[1].mul_add(y[1], s[1]), x[0].mul_add(y[0], s[0])))
            }
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
mod imp {
    use super::Lane;

    #[derive(Clone, Copy)]
    pub struct V([f64; 2]);

    #[inline(always)]
    pub fn zip(a: V, b: V) -> (V, V) {
        (V([a.0[0], b.0[0]]), V([a.0[1], b.0[1]]))
    }

    impl Lane for V {
        const W: usize = 2;
        #[inline(always)]
        unsafe fn load(p: *const f64) -> V {
            unsafe { V([*p, *p.add(1)]) }
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f64) {
            unsafe {
                *p = self.0[0];
                *p.add(1) = self.0[1];
            }
        }
        #[inline(always)]
        fn splat(x: f64) -> V {
            V([x, x])
        }
        #[inline(always)]
        fn add(self, o: V) -> V {
            V([self.0[0] + o.0[0], self.0[1] + o.0[1]])
        }
        #[inline(always)]
        fn sub(self, o: V) -> V {
            V([self.0[0] - o.0[0], self.0[1] - o.0[1]])
        }
        #[inline(always)]
        fn mul(self, o: V) -> V {
            V([self.0[0] * o.0[0], self.0[1] * o.0[1]])
        }
        #[inline(always)]
        fn div(self, o: V) -> V {
            V([self.0[0] / o.0[0], self.0[1] / o.0[1]])
        }
        #[inline(always)]
        fn fma(self, a: V, b: V) -> V {
            V([a.0[0].mul_add(b.0[0], self.0[0]), a.0[1].mul_add(b.0[1], self.0[1])])
        }
    }
}

pub use imp::V;

/// `[a0, b0]` and `[a1, b1]` from `a = [a0, a1]`, `b = [b0, b1]` (a 2x2 transpose).
#[inline(always)]
pub fn zip(a: V, b: V) -> (V, V) {
    imp::zip(a, b)
}

/// `K` lane vectors stored contiguously (`K * L::W` doubles).
#[inline(always)]
pub unsafe fn load_k<L: Lane, const K: usize>(p: *const f64) -> [L; K] {
    let mut r = [L::splat(0.0); K];
    for k in 0..K {
        r[k] = unsafe { L::load(p.add(k * L::W)) };
    }
    r
}

#[inline(always)]
pub unsafe fn store_k<L: Lane, const K: usize>(p: *mut f64, v: [L; K]) {
    for k in 0..K {
        unsafe { v[k].store(p.add(k * L::W)) };
    }
}
