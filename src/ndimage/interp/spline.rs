//! `NI_SplineFilter1D`: the B-spline prefilter, a pair of recursive (IIR)
//! filters per pole along every line of one axis.
//!
//! Lines are filtered in blocks of `W` interleaved lanes (`buf[i][lane]`), so
//! every arithmetic step is a SIMD operation across `W` independent lines and
//! each line sees exactly SciPy's sequence of operations.

use super::support::{
    OutElem, SendConstPtr, SendPtr, for_each_index, for_each_index_init, libm_pow, madd, nmadd,
};
use crate::ndimage::common::{Elem, Mode};

/// Lanes per block.
const W: usize = 16;
/// Largest block buffer (bytes) before falling back to one line at a time.
const MAX_BLOCK_BYTES: usize = 8 << 20;

type Lanes<const N: usize> = [f64; N];

/// True when every value is finite with magnitude <= 1e300. Then, once z^i
/// underflows to zero, the remaining terms of the initial-value sums are
/// signed zeros, which leave a nonzero sum unchanged.
fn tame<const N: usize>(c: &[Lanes<N>]) -> bool {
    let mut bad = [0u64; N];
    for row in c {
        for l in 0..N {
            bad[l] |= !(row[l].abs() <= 1e300) as u64;
        }
    }
    bad.iter().all(|&m| m == 0)
}

#[inline(always)]
fn nonzero<const N: usize>(v: &Lanes<N>) -> bool {
    v.iter().all(|&x| x != 0.0)
}

/// `_init_causal_mirror`
fn init_causal_mirror<const N: usize>(c: &mut [Lanes<N>], z: f64, tame: bool) {
    let n = c.len();
    let zn1 = libm_pow(z, (n - 1) as i64);
    let mut c0 = [0.0; N];
    for l in 0..N {
        c0[l] = madd(zn1, c[n - 1][l], c[0][l]);
    }
    let mut zi = z;
    let mut i = 1;
    let step = |i: usize, zi: f64, c0: &mut Lanes<N>, c: &[Lanes<N>]| {
        let (a, b) = (&c[i], &c[n - 1 - i]);
        for l in 0..N {
            c0[l] = madd(zi, madd(zn1, b[l], a[l]), c0[l]);
        }
    };
    while i + 1 < n && zi != 0.0 {
        step(i, zi, &mut c0, c);
        zi *= z;
        i += 1;
    }
    if !(tame && nonzero(&c0)) {
        while i + 1 < n {
            step(i, zi, &mut c0, c);
            zi *= z;
            i += 1;
        }
    }
    let d = nmadd(zn1, zn1, 1.0);
    for l in 0..N {
        c[0][l] = c0[l] / d;
    }
}

/// `_init_anticausal_mirror`
fn init_anticausal_mirror<const N: usize>(c: &mut [Lanes<N>], z: f64) {
    let n = c.len();
    let d = madd(z, z, -1.0);
    for l in 0..N {
        c[n - 1][l] = madd(z, c[n - 2][l], c[n - 1][l]) * z / d;
    }
}

/// `_init_causal_wrap`
fn init_causal_wrap<const N: usize>(c: &mut [Lanes<N>], z: f64, tame: bool) {
    let n = c.len();
    let mut c0 = c[0];
    let mut zi = z;
    let mut i = 1;
    let step = |i: usize, zi: f64, c0: &mut Lanes<N>, c: &[Lanes<N>]| {
        let a = &c[n - i];
        for l in 0..N {
            c0[l] = madd(zi, a[l], c0[l]);
        }
    };
    while i < n && zi != 0.0 {
        step(i, zi, &mut c0, c);
        zi *= z;
        i += 1;
    }
    if i < n {
        if tame && nonzero(&c0) {
            // z^n underflows to a signed zero as well; only 1 - zi is used.
            zi = 0.0;
        } else {
            while i < n {
                step(i, zi, &mut c0, c);
                zi *= z;
                i += 1;
            }
        }
    }
    let d = 1.0 - zi;
    for l in 0..N {
        c[0][l] = c0[l] / d;
    }
}

/// `_init_anticausal_wrap`
fn init_anticausal_wrap<const N: usize>(c: &mut [Lanes<N>], z: f64, tame: bool) {
    let n = c.len();
    let mut cl = c[n - 1];
    let mut zi = z;
    let mut i = 0;
    let step = |i: usize, zi: f64, cl: &mut Lanes<N>, c: &[Lanes<N>]| {
        let a = &c[i];
        for l in 0..N {
            cl[l] = madd(zi, a[l], cl[l]);
        }
    };
    while i + 1 < n && zi != 0.0 {
        step(i, zi, &mut cl, c);
        zi *= z;
        i += 1;
    }
    if i + 1 < n {
        if tame && nonzero(&cl) {
            zi = 0.0;
        } else {
            while i + 1 < n {
                step(i, zi, &mut cl, c);
                zi *= z;
                i += 1;
            }
        }
    }
    let f = z / (zi - 1.0);
    for l in 0..N {
        c[n - 1][l] = f * cl[l];
    }
}

/// `_init_causal_reflect`
fn init_causal_reflect<const N: usize>(c: &mut [Lanes<N>], z: f64, tame: bool) {
    let n = c.len();
    let zn = libm_pow(z, n as i64);
    let mut sum = [0.0; N];
    for l in 0..N {
        sum[l] = madd(zn, c[n - 1][l], c[0][l]);
    }
    let mut zi = z;
    let mut i = 1;
    let step = |i: usize, zi: f64, sum: &mut Lanes<N>, c: &[Lanes<N>]| {
        let (a, b) = (&c[i], &c[n - 1 - i]);
        for l in 0..N {
            sum[l] = madd(zi, madd(zn, b[l], a[l]), sum[l]);
        }
    };
    while i < n && zi != 0.0 {
        step(i, zi, &mut sum, c);
        zi *= z;
        i += 1;
    }
    if !(tame && nonzero(&sum)) {
        while i < n {
            step(i, zi, &mut sum, c);
            zi *= z;
            i += 1;
        }
    }
    let d = nmadd(zn, zn, 1.0);
    for l in 0..N {
        c[0][l] += sum[l] * z / d;
    }
}

/// `_init_anticausal_reflect`
fn init_anticausal_reflect<const N: usize>(c: &mut [Lanes<N>], z: f64) {
    let n = c.len();
    let f = z / (z - 1.0);
    for l in 0..N {
        c[n - 1][l] *= f;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Init {
    Mirror,
    Wrap,
    Reflect,
}

fn init_kind(mode: Mode) -> Init {
    match mode {
        Mode::GridConstant | Mode::Constant | Mode::Mirror | Mode::Wrap => Init::Mirror,
        Mode::GridWrap => Init::Wrap,
        Mode::Nearest | Mode::Reflect => Init::Reflect,
    }
}

/// `apply_filter` minus the gain (applied while loading), for `n >= 2`.
/// `first_tame` is `tame(c)` before the first pole.
fn filter_lanes<const N: usize>(c: &mut [Lanes<N>], poles: &[f64], init: Init, first_tame: bool) {
    let n = c.len();
    for (p, &z) in poles.iter().enumerate() {
        let tame = if p == 0 { first_tame } else { tame(c) };
        match init {
            Init::Mirror => init_causal_mirror(c, z, tame),
            Init::Wrap => init_causal_wrap(c, z, tame),
            Init::Reflect => init_causal_reflect(c, z, tame),
        }
        let mut prev = c[0];
        for row in c[1..].iter_mut() {
            for l in 0..N {
                row[l] = madd(z, prev[l], row[l]);
            }
            prev = *row;
        }
        match init {
            Init::Mirror => init_anticausal_mirror(c, z),
            Init::Wrap => init_anticausal_wrap(c, z, tame),
            Init::Reflect => init_anticausal_reflect(c, z),
        }
        let mut next = c[n - 1];
        for row in c[..n - 1].iter_mut().rev() {
            for l in 0..N {
                row[l] = z * (next[l] - row[l]);
            }
            next = *row;
        }
    }
}

/// Layout of the lines: the array is viewed as `(outer, n, inner)`, C order.
#[derive(Clone, Copy)]
pub struct Lines {
    pub outer: usize,
    pub n: usize,
    pub inner: usize,
}

/// Filter every line. `src` and `dst` are C-contiguous with the same shape;
/// they may be the same buffer (in-place filtering) since each line is read
/// completely before it is written.
///
/// # Safety
/// `src` and `dst` must be valid for `outer * n * inner` elements, and no
/// other code may access `dst` while this runs.
pub unsafe fn spline_filter1d<I: Elem, O: OutElem>(
    src: *const I,
    dst: *mut O,
    lines: Lines,
    poles: &[f64],
    mode: Mode,
    par: bool,
) {
    let Lines { outer, n, inner } = lines;
    let total = outer * n * inner;
    if total == 0 {
        return;
    }
    let src = SendConstPtr(src);
    let dst = SendPtr(dst);
    if n == 1 {
        // A line of one sample is copied without the gain.
        const CHUNK: usize = 1 << 15;
        for_each_index(total.div_ceil(CHUNK), par, |c| {
            let (src, dst) = (src, dst);
            let lo = c * CHUNK;
            let hi = (lo + CHUNK).min(total);
            for k in lo..hi {
                unsafe { *dst.get().add(k) = O::from_line((*src.get().add(k)).to_f64()) };
            }
        });
        return;
    }
    let mut gain = 1.0;
    for &z in poles {
        gain *= (1.0 - z) * (1.0 - 1.0 / z);
    }
    let init = init_kind(mode);
    let nlines = outer * inner;
    if inner >= 16 {
        super::band::run_bands::<I, O>(src, dst, lines, gain, poles, init, par);
    } else if nlines >= W && n * W * 8 <= MAX_BLOCK_BYTES {
        run_blocks::<I, O, W>(src, dst, lines, gain, poles, init, par);
    } else {
        run_blocks::<I, O, 1>(src, dst, lines, gain, poles, init, par);
    }
}

/// Filter lines along an axis with `inner < 16` in blocks of `N` lines
/// (consecutive line indices; the last block repeats its last line in the
/// unused lanes, whose results are discarded).
fn run_blocks<I: Elem, O: OutElem, const N: usize>(
    src: SendConstPtr<I>,
    dst: SendPtr<O>,
    lines: Lines,
    gain: f64,
    poles: &[f64],
    init: Init,
    par: bool,
) {
    let Lines { outer, n, inner } = lines;
    let nlines = outer * inner;
    let line_base = move |line: usize| (line / inner) * n * inner + line % inner;
    let nblocks = nlines.div_ceil(N);
    for_each_index_init(
        nblocks,
        par,
        || vec![[0.0f64; N]; n],
        |buf, b| {
            let first = b * N;
            let count = N.min(nlines - first);
            let bases: [usize; N] = std::array::from_fn(|l| line_base(first + l.min(count - 1)));
            // SAFETY: every line has n elements at stride `inner`; blocks
            // read and write disjoint lines.
            let tame = unsafe { load_lines(src.get(), &bases, inner, gain, buf) };
            filter_lanes(buf, poles, init, tame);
            unsafe { store_lines(dst.get(), &bases, count, inner, buf) };
        },
    );
}

/// `buf[i][l] = src[bases[l] + i * inner] * gain` (converted to double, as
/// NI_ArrayToLineBuffer and _apply_filter_gain do); returns whether every
/// value is tame.
#[inline(always)]
unsafe fn load_lines<I: Elem, const N: usize>(
    src: *const I,
    bases: &[usize; N],
    inner: usize,
    gain: f64,
    buf: &mut [[f64; N]],
) -> bool {
    let n = buf.len();
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
    let mut start = 0;
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
    let mut ok = true;
    #[cfg(target_arch = "aarch64")]
    if N.is_multiple_of(2) {
        use std::arch::aarch64::*;
        // 2 x 2 transposes: lanes (l, l + 1) by columns (i, i + 1).
        let full = n / 2 * 2;
        unsafe {
            let lim = vdupq_n_f64(1e300);
            let mut okv = vdupq_n_u64(u64::MAX);
            for i in (0..full).step_by(2) {
                let (r0, r1) = buf.split_at_mut(i + 1);
                let (r0, r1) = (r0[i].as_mut_ptr(), r1[0].as_mut_ptr());
                for l in (0..N).step_by(2) {
                    let pa = src.add(bases[l] + i * inner);
                    let pb = src.add(bases[l + 1] + i * inner);
                    let a = [(*pa).to_f64() * gain, (*pa.add(inner)).to_f64() * gain];
                    let b = [(*pb).to_f64() * gain, (*pb.add(inner)).to_f64() * gain];
                    let (va, vb) = (vld1q_f64(a.as_ptr()), vld1q_f64(b.as_ptr()));
                    okv = vandq_u64(okv, vcleq_f64(vabsq_f64(va), lim));
                    okv = vandq_u64(okv, vcleq_f64(vabsq_f64(vb), lim));
                    vst1q_f64(r0.add(l), vzip1q_f64(va, vb));
                    vst1q_f64(r1.add(l), vzip2q_f64(va, vb));
                }
            }
            ok = vminvq_u32(vreinterpretq_u32_u64(okv)) == u32::MAX;
        }
        start = full;
    }
    let mut bad = [0u64; N];
    for i in start..n {
        for l in 0..N {
            let v = unsafe { (*src.add(bases[l] + i * inner)).to_f64() } * gain;
            buf[i][l] = v;
            bad[l] |= !(v.abs() <= 1e300) as u64;
        }
    }
    ok && bad.iter().all(|&m| m == 0)
}

/// `dst[bases[l] + i * inner] = buf[i][l]` (plain C cast) for the first
/// `count` lanes.
#[inline(always)]
unsafe fn store_lines<O: OutElem, const N: usize>(
    dst: *mut O,
    bases: &[usize; N],
    count: usize,
    inner: usize,
    buf: &[[f64; N]],
) {
    let n = buf.len();
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
    let mut start = 0;
    #[cfg(target_arch = "aarch64")]
    if N.is_multiple_of(2) {
        use std::arch::aarch64::*;
        let full = n / 2 * 2;
        let pairs = count / 2 * 2;
        unsafe {
            for i in (0..full).step_by(2) {
                let (r0, r1) = (buf[i].as_ptr(), buf[i + 1].as_ptr());
                for l in (0..pairs).step_by(2) {
                    let v0 = vld1q_f64(r0.add(l));
                    let v1 = vld1q_f64(r1.add(l));
                    let a = vzip1q_f64(v0, v1); // line l, columns i, i + 1
                    let b = vzip2q_f64(v0, v1); // line l + 1
                    let pa = dst.add(bases[l] + i * inner);
                    let pb = dst.add(bases[l + 1] + i * inner);
                    *pa = O::from_line(vgetq_lane_f64::<0>(a));
                    *pa.add(inner) = O::from_line(vgetq_lane_f64::<1>(a));
                    *pb = O::from_line(vgetq_lane_f64::<0>(b));
                    *pb.add(inner) = O::from_line(vgetq_lane_f64::<1>(b));
                }
                for l in pairs..count {
                    *dst.add(bases[l] + i * inner) = O::from_line(buf[i][l]);
                    *dst.add(bases[l] + (i + 1) * inner) = O::from_line(buf[i + 1][l]);
                }
            }
        }
        start = full;
    }
    for i in start..n {
        for l in 0..count {
            unsafe { *dst.add(bases[l] + i * inner) = O::from_line(buf[i][l]) };
        }
    }
}
