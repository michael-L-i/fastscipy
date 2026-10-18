//! Spline prefilter along an axis that is not the last one: lines are
//! filtered a *band* at a time (`bw` adjacent lines, i.e. a block of columns
//! of every row along the filtered axis). Each step of SciPy's recursions is
//! then a row sweep, vectorized across the band's columns, and each line
//! still sees exactly SciPy's sequence of operations.
//!
//! When the output is float64 the band is filtered in place in the output
//! (after converting the input into it), otherwise in a float64 buffer.

use std::any::TypeId;

use super::spline::{Init, Lines};
use super::support::{
    OutElem, SendConstPtr, SendPtr, for_each_index_init, libm_pow, madd, nmadd,
};
use crate::ndimage::common::Elem;

/// Widest band (columns).
const MAX_BW: usize = 512;
/// Target band size in bytes (kept cache-resident between passes).
const BAND_BYTES: usize = 1 << 20;

// Row kernels. Separate slice arguments tell LLVM the rows do not alias, so
// the loops vectorize without runtime checks.

#[inline(never)]
fn causal_row(z: f64, prev: &[f64], cur: &mut [f64]) {
    for (c, &p) in cur.iter_mut().zip(prev) {
        *c = madd(z, p, *c);
    }
}

#[inline(never)]
fn anticausal_row(z: f64, next: &[f64], cur: &mut [f64]) {
    for (c, &x) in cur.iter_mut().zip(next) {
        *c = z * (x - *c);
    }
}

/// `acc = madd(zi, madd(zn, b, a), acc)`
#[inline(never)]
fn init_sym_row(zi: f64, zn: f64, a: &[f64], b: &[f64], acc: &mut [f64]) {
    for ((s, &x), &y) in acc.iter_mut().zip(a).zip(b) {
        *s = madd(zi, madd(zn, y, x), *s);
    }
}

/// `acc = madd(zi, a, acc)`
#[inline(never)]
fn init_row(zi: f64, a: &[f64], acc: &mut [f64]) {
    for (s, &x) in acc.iter_mut().zip(a) {
        *s = madd(zi, x, *s);
    }
}

/// `dst = src * gain`; returns whether every value is tame.
#[inline(never)]
fn load_row<I: Elem>(src: &[I], dst: &mut [f64], gain: f64) -> bool {
    let mut bad = 0u64;
    for (d, s) in dst.iter_mut().zip(src) {
        let v = s.to_f64() * gain;
        *d = v;
        bad |= !(v.abs() <= 1e300) as u64;
    }
    bad == 0
}

/// In place: `row *= gain`; returns whether every value is tame.
#[inline(never)]
fn scale_row(row: &mut [f64], gain: f64) -> bool {
    let mut bad = 0u64;
    for d in row.iter_mut() {
        let v = *d * gain;
        *d = v;
        bad |= !(v.abs() <= 1e300) as u64;
    }
    bad == 0
}

/// `bw` columns of `n` rows: element `(i, c)` at `ptr + i * stride + c`.
#[derive(Clone, Copy)]
struct Band {
    ptr: *mut f64,
    stride: usize,
    n: usize,
    bw: usize,
}

impl Band {
    /// # Safety
    /// Rows must be valid and not otherwise borrowed.
    #[inline(always)]
    unsafe fn row<'a>(self, i: usize) -> &'a mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(i * self.stride), self.bw) }
    }

    /// Whether every value is finite with magnitude <= 1e300 (see `tame` in
    /// spline.rs: then the initial-value sums may stop once z^i is zero).
    unsafe fn tame(self) -> bool {
        let mut ok = true;
        for i in 0..self.n {
            let r = unsafe { self.row(i) };
            let mut m = 0u64;
            for &v in r.iter() {
                m |= !(v.abs() <= 1e300) as u64;
            }
            ok &= m == 0;
        }
        ok
    }

    unsafe fn causal_mirror(self, z: f64, tame: bool, acc: &mut [f64]) {
        let (n, bw) = (self.n, self.bw);
        let zn1 = libm_pow(z, (n - 1) as i64);
        let acc = &mut acc[..bw];
        unsafe {
            let (r0, rl) = (self.row(0), self.row(n - 1));
            for c in 0..bw {
                acc[c] = madd(zn1, rl[c], r0[c]);
            }
            let mut zi = z;
            let mut i = 1;
            let mut early = false;
            while i + 1 < n {
                if zi == 0.0 && !early {
                    if tame && acc.iter().all(|&v| v != 0.0) {
                        break;
                    }
                    early = true;
                }
                init_sym_row(zi, zn1, self.row(i), self.row(n - 1 - i), acc);
                zi *= z;
                i += 1;
            }
            let d = nmadd(zn1, zn1, 1.0);
            let r0 = self.row(0);
            for c in 0..bw {
                r0[c] = acc[c] / d;
            }
        }
    }

    unsafe fn anticausal_mirror(self, z: f64) {
        let n = self.n;
        let d = madd(z, z, -1.0);
        unsafe {
            let (a, b) = (self.row(n - 2), self.row(n - 1));
            for c in 0..self.bw {
                b[c] = madd(z, a[c], b[c]) * z / d;
            }
        }
    }

    unsafe fn causal_wrap(self, z: f64, tame: bool, acc: &mut [f64]) {
        let (n, bw) = (self.n, self.bw);
        let acc = &mut acc[..bw];
        unsafe {
            acc.copy_from_slice(self.row(0));
            let mut zi = z;
            let mut i = 1;
            let mut early = false;
            while i < n {
                if zi == 0.0 && !early {
                    if tame && acc.iter().all(|&v| v != 0.0) {
                        // z^n underflows to a signed zero too; only 1 - zi is used.
                        zi = 0.0;
                        break;
                    }
                    early = true;
                }
                init_row(zi, self.row(n - i), acc);
                zi *= z;
                i += 1;
            }
            let d = 1.0 - zi;
            let r0 = self.row(0);
            for c in 0..bw {
                r0[c] = acc[c] / d;
            }
        }
    }

    unsafe fn anticausal_wrap(self, z: f64, tame: bool, acc: &mut [f64]) {
        let (n, bw) = (self.n, self.bw);
        let acc = &mut acc[..bw];
        unsafe {
            acc.copy_from_slice(self.row(n - 1));
            let mut zi = z;
            let mut i = 0;
            let mut early = false;
            while i + 1 < n {
                if zi == 0.0 && !early {
                    if tame && acc.iter().all(|&v| v != 0.0) {
                        zi = 0.0;
                        break;
                    }
                    early = true;
                }
                init_row(zi, self.row(i), acc);
                zi *= z;
                i += 1;
            }
            let f = z / (zi - 1.0);
            let rl = self.row(n - 1);
            for c in 0..bw {
                rl[c] = f * acc[c];
            }
        }
    }

    unsafe fn causal_reflect(self, z: f64, tame: bool, acc: &mut [f64]) {
        let (n, bw) = (self.n, self.bw);
        let zn = libm_pow(z, n as i64);
        let acc = &mut acc[..bw];
        unsafe {
            let (r0, rl) = (self.row(0), self.row(n - 1));
            for c in 0..bw {
                acc[c] = madd(zn, rl[c], r0[c]);
            }
            let mut zi = z;
            let mut i = 1;
            let mut early = false;
            while i < n {
                if zi == 0.0 && !early {
                    if tame && acc.iter().all(|&v| v != 0.0) {
                        break;
                    }
                    early = true;
                }
                init_sym_row(zi, zn, self.row(i), self.row(n - 1 - i), acc);
                zi *= z;
                i += 1;
            }
            let d = nmadd(zn, zn, 1.0);
            let r0 = self.row(0);
            for c in 0..bw {
                r0[c] += acc[c] * z / d;
            }
        }
    }

    unsafe fn anticausal_reflect(self, z: f64) {
        let f = z / (z - 1.0);
        let rl = unsafe { self.row(self.n - 1) };
        for v in rl.iter_mut() {
            *v *= f;
        }
    }

    /// `apply_filter` minus the gain, for `n >= 2`. `tame` describes the
    /// band's values before the first pole.
    unsafe fn filter(self, poles: &[f64], init: Init, mut tame: bool, acc: &mut [f64]) {
        let n = self.n;
        for (p, &z) in poles.iter().enumerate() {
            if p > 0 {
                tame = unsafe { self.tame() };
            }
            unsafe {
                match init {
                    Init::Mirror => self.causal_mirror(z, tame, acc),
                    Init::Wrap => self.causal_wrap(z, tame, acc),
                    Init::Reflect => self.causal_reflect(z, tame, acc),
                }
                for i in 1..n {
                    causal_row(z, self.row(i - 1), self.row(i));
                }
                match init {
                    Init::Mirror => self.anticausal_mirror(z),
                    Init::Wrap => self.anticausal_wrap(z, tame, acc),
                    Init::Reflect => self.anticausal_reflect(z),
                }
                for i in (0..n - 1).rev() {
                    anticausal_row(z, self.row(i + 1), self.row(i));
                }
            }
        }
    }
}

/// Filter lines along an axis with `inner >= 16` (`n >= 2`).
pub fn run_bands<I: Elem, O: OutElem>(
    src: SendConstPtr<I>,
    dst: SendPtr<O>,
    lines: Lines,
    gain: f64,
    poles: &[f64],
    init: Init,
    par: bool,
) {
    let Lines { outer, n, inner } = lines;
    let threads = if par { rayon::current_num_threads() } else { 1 };
    let mut bw = (BAND_BYTES / (n * 8)).clamp(16, MAX_BW) / 16 * 16;
    while bw > 16 && outer * inner.div_ceil(bw) < 4 * threads {
        bw = (bw / 2).max(16);
    }
    let bw = bw.min(inner);
    let per = inner.div_ceil(bw);
    let direct = TypeId::of::<O>() == TypeId::of::<f64>();
    for_each_index_init(
        outer * per,
        par,
        || (vec![0.0f64; MAX_BW], if direct { Vec::new() } else { vec![0.0f64; n * bw] }),
        |(acc, buf), item| {
            let (o, k0) = (item / per, (item % per) * bw);
            let w = bw.min(inner - k0);
            let base = o * n * inner + k0;
            let band = if direct {
                Band { ptr: dst.get().cast::<f64>().wrapping_add(base), stride: inner, n, bw: w }
            } else {
                Band { ptr: buf.as_mut_ptr(), stride: w, n, bw: w }
            };
            // Load: convert to double and apply the gain (as
            // NI_ArrayToLineBuffer and _apply_filter_gain do).
            let in_place = direct && src.get() as usize == dst.get() as usize;
            let mut tame = true;
            for i in 0..n {
                // SAFETY: rows are in range and distinct; in place, the row is
                // only accessed through `band`.
                let d = unsafe { band.row(i) };
                tame &= if in_place {
                    scale_row(d, gain)
                } else {
                    let s = unsafe { std::slice::from_raw_parts(src.get().add(base + i * inner), w) };
                    load_row(s, d, gain)
                };
            }
            unsafe { band.filter(poles, init, tame, acc) };
            if !direct {
                for i in 0..n {
                    let d = dst.get().wrapping_add(base + i * inner);
                    let r = unsafe { band.row(i) };
                    for (c, &v) in r.iter().enumerate() {
                        unsafe { *d.add(c) = O::from_line(v) };
                    }
                }
            }
        },
    );
}
