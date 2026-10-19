//! `NI_GeometricTransform` without a Python mapping: affine transforms
//! (`matrix` + `offset`) and `map_coordinates` (`coordinates`).
//!
//! Output points are processed `LANES` at a time. When every tap of every
//! lane lies inside the input (no boundary mapping involved), the lanes run
//! a vectorized fast path; other lanes use an exact scalar port of SciPy's
//! per-point loop. Both perform SciPy's operations in SciPy's order.

use rayon::prelude::*;

use super::support::{
    OutElem, for_each_chunk, madd, map_coordinate, spline_mode, spline_start, spline_weights,
    store_interp,
};
use crate::ndimage::common::{Elem, Mode};

const LANES: usize = 8;
/// Output points per work item.
const BLOCK: usize = 2048;
/// Highest input rank supported (NumPy allows 64).
pub const MAX_RANK: usize = 32;
/// Coordinates beyond this overflow SciPy's integer arithmetic; such calls
/// are declined.
pub const COORD_LIMIT: f64 = 1e15;

pub struct Params {
    pub order: usize,
    pub mode: Mode,
    pub cval: f64,
    pub npad: i64,
}

struct Input<'a, I> {
    data: &'a [I],
    rank: usize,
    dims: [i64; MAX_RANK],
    strides: [usize; MAX_RANK],
}

/// Exact port of the per-point body of NI_GeometricTransform, for inputs of
/// rank at most `MR`.
fn point_exact<I: Elem, const MR: usize>(inp: &Input<I>, icoor: &[f64], p: &Params) -> f64 {
    let rank = inp.rank;
    let order = p.order;
    let mode = p.mode;
    let smode = spline_mode(mode);
    let mapped = mode != Mode::GridConstant && mode != Mode::Nearest;
    let mut idx = [[0usize; 6]; MR];
    let mut cvf = [[false; 6]; MR];
    let mut w = [[0.0f64; 6]; MR];
    for d in 0..rank {
        let len = inp.dims[d];
        let mut cc = icoor[d] + p.npad as f64;
        if mapped {
            cc = map_coordinate(cc, len, mode);
        }
        if !(cc > -1.0 || !mapped) {
            return p.cval;
        }
        let start = spline_start(cc, order);
        if mode == Mode::GridConstant {
            for j in 0..=order {
                let i = start.wrapping_add(j as i64);
                if i < 0 || i >= len {
                    cvf[d][j] = true;
                } else {
                    idx[d][j] = i as usize;
                }
            }
        } else if start < 0 || start.wrapping_add(order as i64) >= len {
            for j in 0..=order {
                let i = map_coordinate(start.wrapping_add(j as i64) as f64, len, smode) as i64;
                // Always in range for coordinates within COORD_LIMIT.
                idx[d][j] = i.clamp(0, len - 1) as usize;
            }
        } else {
            for j in 0..=order {
                idx[d][j] = (start + j as i64) as usize;
            }
        }
        if order != 0 {
            spline_weights(cc, order, &mut w[d]);
        }
    }
    let mut t = 0.0;
    let mut f = [0usize; MR];
    loop {
        let is_cval = mode == Mode::GridConstant && (0..rank).any(|d| cvf[d][f[d]]);
        let mut coeff = if is_cval {
            p.cval
        } else {
            let mut off = 0;
            for d in 0..rank {
                off += idx[d][f[d]] * inp.strides[d];
            }
            inp.data[off].to_f64()
        };
        if order > 0 {
            for d in 0..rank {
                coeff *= w[d][f[d]];
            }
        }
        t += coeff;
        let mut d = rank;
        loop {
            if d == 0 {
                return t;
            }
            d -= 1;
            f[d] += 1;
            if f[d] <= order {
                break;
            }
            f[d] = 0;
        }
    }
}

/// Interpolation of a batch of points.
trait Kernel<I>: Sync {
    /// Points whose coordinates are `icoor[d][i]`, `i < n`, into `out`.
    fn run(&self, inp: &Input<I>, p: &Params, icoor: &[&[f64]], n: usize, out: &mut [f64]);

    /// Points `x0..x0 + out.len()` of an affine row: coordinate `d` of point
    /// `x` is `madd(x, m[d], b[d])`, the last step of SciPy's accumulation.
    fn affine_row(
        &self,
        inp: &Input<I>,
        p: &Params,
        b: &[f64],
        m: &[f64],
        x0: usize,
        out: &mut [f64],
    ) {
        affine_generic(self, inp, p, b, m, x0, out);
    }
}

fn affine_generic<I, K: Kernel<I> + ?Sized>(
    k: &K,
    inp: &Input<I>,
    p: &Params,
    b: &[f64],
    m: &[f64],
    x0: usize,
    out: &mut [f64],
) {
    let n = out.len();
    if n == 0 {
        return;
    }
    let rank = b.len();
    let mut icoor = vec![0.0f64; n * rank];
    for (d, ic) in icoor.chunks_exact_mut(n).enumerate() {
        for (i, c) in ic.iter_mut().enumerate() {
            *c = madd((x0 + i) as f64, m[d], b[d]);
        }
    }
    let views: Vec<&[f64]> = icoor.chunks_exact(n).collect();
    k.run(inp, p, &views, n, out);
}

/// Generic path: every point through the exact scalar port.
struct Slow;

impl<I: Elem> Kernel<I> for Slow {
    fn run(&self, inp: &Input<I>, p: &Params, icoor: &[&[f64]], n: usize, out: &mut [f64]) {
        let mut c = [0.0; MAX_RANK];
        for i in 0..n {
            for d in 0..inp.rank {
                c[d] = icoor[d][i];
            }
            out[i] = if inp.rank <= 8 {
                point_exact::<I, 8>(inp, &c[..inp.rank], p)
            } else {
                point_exact::<I, MAX_RANK>(inp, &c[..inp.rank], p)
            };
        }
    }
}

/// Weights per axis, tap and lane.
type LaneWeights<const R: usize> = [[[f64; LANES]; 6]; R];

/// SciPy's interpolation sum for `LANES` points whose first taps are at
/// offsets `b`: `t = 0; t += ((v * w_0) * w_1) ...` over the taps in C order.
///
/// # Safety
/// Every tap of every lane must be inside `data`.
#[inline(always)]
unsafe fn taps_sum<I: Elem, const ORDER: usize, const R: usize>(
    data: &[I],
    st: &[usize; R],
    b: &[usize; LANES],
    w: impl Fn(usize, usize, usize) -> f64,
) -> [f64; LANES] {
    let get = |i: usize| unsafe { data.get_unchecked(i) }.to_f64();
    let mut t = [0.0f64; LANES];
    match R {
        1 => {
            for j0 in 0..=ORDER {
                for l in 0..LANES {
                    let mut v = get(b[l] + j0);
                    if ORDER > 0 {
                        v *= w(0, j0, l);
                    }
                    t[l] += v;
                }
            }
        }
        2 => {
            for j0 in 0..=ORDER {
                for j1 in 0..=ORDER {
                    let off = j0 * st[0] + j1;
                    for l in 0..LANES {
                        let mut v = get(b[l] + off);
                        if ORDER > 0 {
                            v *= w(0, j0, l);
                            v *= w(1 % R, j1, l);
                        }
                        t[l] += v;
                    }
                }
            }
        }
        _ => {
            for j0 in 0..=ORDER {
                for j1 in 0..=ORDER {
                    for j2 in 0..=ORDER {
                        let off = j0 * st[0] + j1 * st[1 % R] + j2;
                        for l in 0..LANES {
                            let mut v = get(b[l] + off);
                            if ORDER > 0 {
                                v *= w(0, j0, l);
                                v *= w(1 % R, j1, l);
                                v *= w(2 % R, j2, l);
                            }
                            t[l] += v;
                        }
                    }
                }
            }
        }
    }
    t
}

/// Points per pass of the classifying fast path.
const SUB: usize = 256;

/// Per-thread scratch of the classifying fast path (structure of arrays,
/// for input ranks up to 3).
#[derive(Default)]
struct Scratch {
    /// Coordinates `x[d * SUB + i]` (including the prepadding).
    x: Vec<f64>,
    /// Offset of the first tap (0 for points not on the fast path).
    base: Vec<usize>,
    /// 0: fast path, 1: cval (constant mode, outside), 2: exact path.
    cls: Vec<u64>,
    slow: Vec<u64>,
    outside: Vec<u64>,
    /// Weights `w[(d * 6 + j) * SUB + i]`.
    w: Vec<f64>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// Fast path for a fixed order and input rank (1 to 3).
struct Fast<const ORDER: usize, const R: usize>;

impl<const ORDER: usize, const R: usize> Fast<ORDER, R> {
    /// Phase 1 of `run`: coordinates, classification, first-tap offsets and
    /// weights of points `s0..s0 + m` (`m <= SUB`, padded to `mm`, a multiple
    /// of LANES). Returns whether any point is fast.
    #[inline(always)]
    fn classify<I>(
        inp: &Input<I>,
        p: &Params,
        icoor: &[&[f64]],
        s0: usize,
        m: usize,
        mm: usize,
        sc: &mut Scratch,
    ) -> bool {
        let mapped = p.mode != Mode::GridConstant && p.mode != Mode::Nearest;
        let constant = p.mode == Mode::Constant;
        let npad = p.npad as f64;
        let (slow, outside, base) = (
            &mut sc.slow[..mm],
            &mut sc.outside[..mm],
            &mut sc.base[..mm],
        );
        slow.fill(0);
        outside.fill(0);
        base.fill(0);
        for d in 0..R {
            let len = inp.dims[d];
            let hi = (len - 1) as f64;
            let stride = inp.strides[d];
            let src = &icoor[d][s0..s0 + m];
            let xs = &mut sc.x[d * SUB..d * SUB + mm];
            for i in 0..m {
                xs[i] = src[i] + npad;
            }
            for i in m..mm {
                xs[i] = xs[0];
            }
            for i in 0..mm {
                let x = xs[i];
                let s = spline_start(x, ORDER);
                let in_range = (x >= 0.0) & (x <= hi);
                // Fast: boundary mapping leaves the coordinate unchanged and
                // no tap needs border handling.
                let ok = (!mapped | in_range) & (s >= 0) & (s.wrapping_add(ORDER as i64) < len);
                slow[i] |= !ok as u64;
                outside[i] |= !in_range as u64;
                let sc = if ok { s } else { 0 };
                base[i] = base[i].wrapping_add(sc as usize * stride);
            }
        }
        let cmask = constant as u64;
        let mut nfast = 0u64;
        let cls = &mut sc.cls[..mm];
        for i in 0..mm {
            // Constant mode: a coordinate outside [0, len - 1] (or NaN) maps
            // to -1, which makes the whole point cval.
            let c = if cmask & outside[i] != 0 {
                1
            } else {
                2 * slow[i]
            };
            let c = if i < m { c } else { 2 };
            cls[i] = c;
            if c != 0 {
                base[i] = 0;
            }
            nfast += (c == 0) as u64;
        }
        if nfast > 0 && ORDER > 0 {
            for d in 0..R {
                let xs = &sc.x[d * SUB..d * SUB + mm];
                let w = &mut sc.w[d * 6 * SUB..(d + 1) * 6 * SUB];
                for i in 0..mm {
                    let mut ws = [0.0; 6];
                    spline_weights(xs[i], ORDER, &mut ws);
                    for j in 0..=ORDER {
                        w[j * SUB + i] = ws[j];
                    }
                }
            }
        }
        nfast > 0
    }

    /// Whether every point of `s0..s0 + m` is fast: the fast coordinates of
    /// each axis form an interval, so the extremes decide (NaN never is).
    #[inline(always)]
    fn block_fast<I>(inp: &Input<I>, p: &Params, icoor: &[&[f64]], s0: usize, m: usize) -> bool {
        let mapped = p.mode != Mode::GridConstant && p.mode != Mode::Nearest;
        let npad = p.npad as f64;
        for d in 0..R {
            let (mut lo, mut hi, mut nan) = (f64::INFINITY, f64::NEG_INFINITY, 0u64);
            for &x in &icoor[d][s0..s0 + m] {
                let v = x + npad;
                lo = lo.min(v);
                hi = hi.max(v);
                nan |= v.is_nan() as u64;
            }
            let len = inp.dims[d];
            let top = (len - 1) as f64;
            let lower = (!mapped || lo >= 0.0) && spline_start(lo, ORDER) >= 0;
            let upper = (!mapped || hi <= top) && spline_start(hi, ORDER) + (ORDER as i64) < len;
            if nan != 0 || !lower || !upper {
                return false;
            }
        }
        true
    }

    /// Points `s0..` (all fast) of a coordinate list.
    #[inline(always)]
    fn fast_block<I: Elem>(
        inp: &Input<I>,
        npad: f64,
        icoor: &[&[f64]],
        s0: usize,
        out: &mut [f64],
    ) {
        let st: [usize; R] = std::array::from_fn(|d| inp.strides[d]);
        let n = out.len();
        let chunk = |cc: &[[f64; LANES]; R]| -> [f64; LANES] {
            let mut base = [0usize; LANES];
            let mut w: LaneWeights<R> = [[[0.0; LANES]; 6]; R];
            for d in 0..R {
                for l in 0..LANES {
                    base[l] += spline_start(cc[d][l], ORDER) as usize * st[d];
                }
                if ORDER > 0 {
                    for l in 0..LANES {
                        let mut ws = [0.0; 6];
                        spline_weights(cc[d][l], ORDER, &mut ws);
                        for j in 0..=ORDER {
                            w[d][j][l] = ws[j];
                        }
                    }
                }
            }
            // SAFETY: every point was checked to be fast.
            unsafe { taps_sum::<I, ORDER, R>(inp.data, &st, &base, |d, j, l| w[d][j][l]) }
        };
        let full = n / LANES * LANES;
        for c in (0..full).step_by(LANES) {
            let cc: [[f64; LANES]; R] = std::array::from_fn(|d| {
                let src: &[f64; LANES] = icoor[d][s0 + c..s0 + c + LANES].try_into().unwrap();
                std::array::from_fn(|l| src[l] + npad)
            });
            let t = chunk(&cc);
            out[c..c + LANES].copy_from_slice(&t);
        }
        if full < n {
            let k = n - full;
            let cc: [[f64; LANES]; R] = std::array::from_fn(|d| {
                std::array::from_fn(|l| icoor[d][s0 + full + l.min(k - 1)] + npad)
            });
            let t = chunk(&cc);
            out[full..].copy_from_slice(&t[..k]);
        }
    }

    /// Fast points of an affine row (no classification needed).
    #[inline(always)]
    fn interior<I: Elem>(
        inp: &Input<I>,
        npad: f64,
        b: &[f64],
        m: &[f64],
        x0: usize,
        out: &mut [f64],
    ) {
        let st: [usize; R] = std::array::from_fn(|d| inp.strides[d]);
        let mb: [(f64, f64); R] = std::array::from_fn(|d| (m[d], b[d]));
        let n = out.len();
        let full = n / LANES * LANES;
        for c in (0..full).step_by(LANES) {
            let xs: [f64; LANES] = std::array::from_fn(|l| (x0 + c + l) as f64);
            // SAFETY: the caller checked that these points are fast.
            let t = unsafe { Self::interior_chunk(inp.data, &st, &mb, npad, &xs) };
            out[c..c + LANES].copy_from_slice(&t);
        }
        if full < n {
            let k = n - full;
            let xs: [f64; LANES] = std::array::from_fn(|l| (x0 + full + l.min(k - 1)) as f64);
            let t = unsafe { Self::interior_chunk(inp.data, &st, &mb, npad, &xs) };
            out[full..].copy_from_slice(&t[..k]);
        }
    }

    /// # Safety
    /// Every point must be fast.
    #[inline(always)]
    unsafe fn interior_chunk<I: Elem>(
        data: &[I],
        st: &[usize; R],
        mb: &[(f64, f64); R],
        npad: f64,
        xs: &[f64; LANES],
    ) -> [f64; LANES] {
        let mut cc = [[0.0f64; LANES]; R];
        for d in 0..R {
            for l in 0..LANES {
                cc[d][l] = madd(xs[l], mb[d].0, mb[d].1) + npad;
            }
        }
        let mut base = [0usize; LANES];
        let mut w: LaneWeights<R> = [[[0.0; LANES]; 6]; R];
        for d in 0..R {
            for l in 0..LANES {
                base[l] += spline_start(cc[d][l], ORDER) as usize * st[d];
            }
            if ORDER > 0 {
                for l in 0..LANES {
                    let mut ws = [0.0; 6];
                    spline_weights(cc[d][l], ORDER, &mut ws);
                    for j in 0..=ORDER {
                        w[d][j][l] = ws[j];
                    }
                }
            }
        }
        unsafe { taps_sum::<I, ORDER, R>(data, st, &base, |d, j, l| w[d][j][l]) }
    }
}

/// The points `x` of `lo..hi` where `lower(x) && upper(x)`, given that along
/// x `lower` switches from false to true at most once and `upper` from true
/// to false at most once (the other way round when `rising` is false).
fn interval(
    lo: usize,
    hi: usize,
    rising: bool,
    lower: impl Fn(usize) -> bool,
    upper: impl Fn(usize) -> bool,
) -> (usize, usize) {
    // First x in lo..hi where `f` holds (hi if none); `f` is monotone false->true.
    let first = |f: &dyn Fn(usize) -> bool| {
        let (mut a, mut b) = (lo, hi);
        while a < b {
            let mid = a + (b - a) / 2;
            if f(mid) {
                b = mid;
            } else {
                a = mid + 1;
            }
        }
        a
    };
    if rising {
        (first(&lower), first(&|x| !upper(x)))
    } else {
        (first(&upper), first(&|x| !lower(x)))
    }
}

impl<I: Elem, const ORDER: usize, const R: usize> Kernel<I> for Fast<ORDER, R> {
    fn run(&self, inp: &Input<I>, p: &Params, icoor: &[&[f64]], n: usize, out: &mut [f64]) {
        let st: [usize; R] = std::array::from_fn(|d| inp.strides[d]);
        SCRATCH.with(|cell| {
            let sc = &mut *cell.borrow_mut();
            if sc.x.len() < 3 * SUB {
                sc.x = vec![0.0; 3 * SUB];
                sc.base = vec![0; SUB];
                sc.cls = vec![0; SUB];
                sc.slow = vec![0; SUB];
                sc.outside = vec![0; SUB];
                sc.w = vec![0.0; 3 * 6 * SUB];
            }
            for s0 in (0..n).step_by(SUB) {
                let m = SUB.min(n - s0);
                let res = &mut out[s0..s0 + m];
                if Self::block_fast(inp, p, icoor, s0, m) {
                    Self::fast_block(inp, p.npad as f64, icoor, s0, res);
                    continue;
                }
                let mm = m.next_multiple_of(LANES);
                let any = Self::classify(inp, p, icoor, s0, m, mm, sc);
                if any {
                    // Phase 2. SAFETY: fast points have every tap in range;
                    // the others read the first taps of the array, which exist
                    // because some point is fast, so len >= ORDER + 1 along
                    // every axis.
                    for c in (0..mm).step_by(LANES) {
                        let b: [usize; LANES] = std::array::from_fn(|l| sc.base[c + l]);
                        let ws = &sc.w[..];
                        // SAFETY: d < R <= 3, j <= ORDER < 6, c + l < SUB.
                        let w = |d: usize, j: usize, l: usize| unsafe {
                            *ws.get_unchecked((d * 6 + j) * SUB + c + l)
                        };
                        let t = unsafe { taps_sum::<I, ORDER, R>(inp.data, &st, &b, w) };
                        if c + LANES <= m {
                            res[c..c + LANES].copy_from_slice(&t);
                        } else {
                            for (r, v) in res[c..].iter_mut().zip(t) {
                                *r = v;
                            }
                        }
                    }
                }
                // Phase 3: cval and exact points.
                let mut cc = [0.0; R];
                for i in 0..m {
                    match sc.cls[i] {
                        0 => {}
                        1 => res[i] = p.cval,
                        _ => {
                            for d in 0..R {
                                cc[d] = icoor[d][s0 + i];
                            }
                            res[i] = point_exact::<I, R>(inp, &cc, p);
                        }
                    }
                }
            }
        })
    }

    fn affine_row(
        &self,
        inp: &Input<I>,
        p: &Params,
        b: &[f64],
        m: &[f64],
        x0: usize,
        out: &mut [f64],
    ) {
        let n = out.len();
        let end = x0 + n;
        let mapped = p.mode != Mode::GridConstant && p.mode != Mode::Nearest;
        let constant = p.mode == Mode::Constant;
        let npad = p.npad as f64;
        // Each coordinate is monotone in x, and both "inside [0, len - 1]"
        // and "no border handling needed" are intervals of coordinate values,
        // so both sets of points are intervals of x; find their exact ends.
        let (mut fa, mut fb) = (x0, end); // fast points
        let (mut ra, mut rb) = (x0, end); // points inside the input
        for d in 0..R {
            let len = inp.dims[d];
            let hi = (len - 1) as f64;
            let c = |x: usize| madd(x as f64, m[d], b[d]) + npad;
            let rising = m[d] >= 0.0;
            let (a, z) = interval(
                x0,
                end,
                rising,
                |x| {
                    let v = c(x);
                    (!mapped || v >= 0.0) && spline_start(v, ORDER) >= 0
                },
                |x| {
                    let v = c(x);
                    (!mapped || v <= hi) && spline_start(v, ORDER).wrapping_add(ORDER as i64) < len
                },
            );
            (fa, fb) = (fa.max(a), fb.min(z));
            if constant {
                let (a, z) = interval(x0, end, rising, |x| c(x) >= 0.0, |x| c(x) <= hi);
                (ra, rb) = (ra.max(a), rb.min(z));
            }
        }
        if !constant {
            (ra, rb) = (x0, end);
        }
        if ra >= rb {
            // Constant mode, the whole segment is outside the input.
            out.fill(p.cval);
            return;
        }
        if fa >= fb {
            (fa, fb) = (ra, ra);
        }
        let (fa, fb) = (fa.clamp(ra, rb), fb.clamp(ra, rb));
        out[..ra - x0].fill(p.cval);
        out[rb - x0..].fill(p.cval);
        affine_generic(self, inp, p, b, m, ra, &mut out[ra - x0..fa - x0]);
        Self::interior(inp, npad, b, m, fa, &mut out[fa - x0..fb - x0]);
        affine_generic(self, inp, p, b, m, fb, &mut out[fb - x0..rb - x0]);
    }
}

fn make_kernel<I: Elem>(order: usize, rank: usize) -> Box<dyn Kernel<I>> {
    macro_rules! fast {
        ($o:literal) => {
            match rank {
                1 => Box::new(Fast::<$o, 1>),
                2 => Box::new(Fast::<$o, 2>),
                3 => Box::new(Fast::<$o, 3>),
                _ => Box::new(Slow),
            }
        };
    }
    match order {
        0 => fast!(0),
        1 => fast!(1),
        2 => fast!(2),
        3 => fast!(3),
        4 => fast!(4),
        _ => fast!(5),
    }
}

fn make_input<'a, I>(data: &'a [I], shape: &[usize]) -> Input<'a, I> {
    let rank = shape.len();
    let mut inp = Input {
        data,
        rank,
        dims: [0; MAX_RANK],
        strides: [0; MAX_RANK],
    };
    let mut s = 1usize;
    for d in (0..rank).rev() {
        inp.dims[d] = shape[d] as i64;
        inp.strides[d] = s;
        s *= shape[d];
    }
    inp
}

thread_local! {
    static TBUF: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` with a reused per-thread buffer of `n` doubles (contents
/// unspecified; `f` overwrites them).
fn with_tbuf(n: usize, f: impl FnOnce(&mut [f64])) {
    TBUF.with(|cell| {
        let mut v = cell.borrow_mut();
        if v.len() < n {
            v.resize(n, 0.0);
        }
        f(&mut v[..n]);
    })
}

fn supported(in_shape: &[usize], out_len: usize, order: usize) -> bool {
    !in_shape.is_empty()
        && in_shape.len() <= MAX_RANK
        && in_shape.iter().all(|&n| n > 0)
        && out_len > 0
        && order <= 5
}

/// Affine transform: input coordinate `d` of output point `o` is
/// `offset[d] + sum_k matrix[d, k] * o[k]`, accumulated as SciPy does.
/// Returns false (without writing) when the call is declined.
pub fn affine<I: Elem, O: OutElem>(
    src: &[I],
    in_shape: &[usize],
    dst: &mut [O],
    out_shape: &[usize],
    matrix: &[f64],
    offset: &[f64],
    p: &Params,
    par: bool,
) -> bool {
    let irank = in_shape.len();
    let orank = out_shape.len();
    if !supported(in_shape, dst.len(), p.order)
        || orank == 0
        || orank > 16
        || matrix.len() != irank * orank
        || offset.len() != irank
    {
        return false;
    }
    // The coordinates are affine in the output index, so their extremes are
    // at the corners of the output box.
    for corner in 0..(1u64 << orank) {
        let mut o = [0.0f64; 16];
        for k in 0..orank {
            if corner >> k & 1 == 1 {
                o[k] = (out_shape[k] - 1) as f64;
            }
        }
        for d in 0..irank {
            let mut c = offset[d];
            for k in 0..orank {
                c = madd(o[k], matrix[d * orank + k], c);
            }
            if c.abs() > COORD_LIMIT {
                return false;
            }
        }
    }
    let inp = make_input(src, in_shape);
    let kernel = make_kernel::<I>(p.order, irank);
    let out_last = out_shape[orank - 1];
    for_each_chunk(dst, out_last, par, |row, out_row| {
        // Leading output coordinates of this row and the partial sums.
        let mut oc = [0usize; 16];
        let mut rem = row;
        for k in (0..orank - 1).rev() {
            oc[k] = rem % out_shape[k];
            rem /= out_shape[k];
        }
        let mut base = [0.0f64; MAX_RANK];
        for d in 0..irank {
            let mut c = offset[d];
            for k in 0..orank - 1 {
                c = madd(oc[k] as f64, matrix[d * orank + k], c);
            }
            base[d] = c;
        }
        let mlast: Vec<f64> = (0..irank).map(|d| matrix[d * orank + orank - 1]).collect();
        for_each_chunk(out_row, BLOCK, par, |b, out_blk| {
            with_tbuf(out_blk.len(), |t| {
                kernel.affine_row(&inp, p, &base[..irank], &mlast, b * BLOCK, t);
                store_interp(out_blk, t);
            });
        });
    });
    true
}

/// map_coordinates: `coords` is C-contiguous with shape `(rank, npoints)`.
/// Returns false (without writing) when the call is declined.
pub fn map_coords<I: Elem, O: OutElem>(
    src: &[I],
    in_shape: &[usize],
    dst: &mut [O],
    coords: &[f64],
    p: &Params,
    par: bool,
) -> bool {
    let rank = in_shape.len();
    let n = dst.len();
    if !supported(in_shape, n, p.order) || coords.len() != rank * n {
        return false;
    }
    let scan = |c: &[f64]| {
        let mut bad = 0u64;
        for &x in c {
            bad |= (x.abs() > COORD_LIMIT) as u64;
        }
        bad != 0
    };
    let too_big = if par {
        coords.par_chunks(1 << 16).any(scan)
    } else {
        scan(coords)
    };
    if too_big {
        return false;
    }
    let inp = make_input(src, in_shape);
    let kernel = make_kernel::<I>(p.order, rank);
    for_each_chunk(dst, BLOCK, par, |b, out_blk| {
        let p0 = b * BLOCK;
        let m = out_blk.len();
        let icoor: Vec<&[f64]> = (0..rank)
            .map(|d| &coords[d * n + p0..d * n + p0 + m])
            .collect();
        with_tbuf(m, |t| {
            kernel.run(&inp, p, &icoor, m, t);
            store_interp(out_blk, t);
        });
    });
    true
}

/// Whether every coordinate is finite (used to vet integer outputs on
/// targets whose NaN casts were not verified).
pub fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}
