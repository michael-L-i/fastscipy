//! Double-precision compute kernels.
//!
//! Every kernel evaluates many outputs at once, one per SIMD lane, and gives
//! each output exactly the sequence of roundings SciPy's C loop performs for
//! it; nothing is reassociated within an output.
//!
//! The 1-D kernels see their input as rows of an "extended" array (see
//! `Rows`): row `t` holds extended-line element `t` of every output
//! lane. For a pass along a non-last axis the rows are array rows (one output
//! per column); along the last axis, `stride == 1` and row `t` is the
//! extended line shifted by `t` (one output per position).

use super::simd::{Lane, V, load_k, store_k};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Sym,
    Anti,
    Gen,
}

/// How the multiply-adds of SciPy's `NI_Correlate1D` were compiled.
///
/// `oline[ll] += (iline[jj] + iline[-jj]) * fw[jj]` may be contracted into a
/// fused multiply-add. x86-64 baseline builds have no FMA (all separate);
/// GCC on aarch64 contracts everything; clang on aarch64 (SciPy's macOS arm64
/// wheels) contracts the scalar loops but, for the symmetric and
/// antisymmetric cases with `size1 >= 10`, vectorizes the first
/// `size1 & !7` taps with an in-order reduction of separately rounded
/// products. The shim checks which scheme reproduces the installed SciPy.
pub const SCHEME_UNFUSED: u32 = 0;
pub const SCHEME_FUSED: u32 = 1;
pub const SCHEME_CLANG_ARM64: u32 = 2;

/// A 1-D correlation as `NI_Correlate1D` evaluates it.
pub struct Prog {
    pub kind: Kind,
    pub fs: usize,
    pub size1: usize,
    pub w: Vec<f64>,
    /// Leading loop iterations computed as a rounded product plus an add;
    /// the rest use a fused multiply-add.
    pub n_unfused: usize,
}

impl Prog {
    pub fn new(w: &[f64], scheme: u32) -> Option<Prog> {
        let fs = w.len();
        if fs == 0 {
            return None;
        }
        let size1 = fs / 2;
        let eps = f64::EPSILON;
        let mut kind = Kind::Gen;
        if fs & 1 == 1 {
            // `fabs(a - b) > DBL_EPSILON` is false for NaN: such pairs count as equal.
            if (1..=size1).all(|ii| !((w[ii + size1] - w[size1 - ii]).abs() > eps)) {
                kind = Kind::Sym;
            } else if (1..=size1).all(|ii| !((w[size1 + ii] + w[size1 - ii]).abs() > eps)) {
                kind = Kind::Anti;
            }
        }
        let iters = if kind == Kind::Gen { fs - 1 } else { size1 };
        let n_unfused = match scheme {
            SCHEME_UNFUSED => iters,
            SCHEME_FUSED => 0,
            SCHEME_CLANG_ARM64 => {
                if kind != Kind::Gen && size1 >= 10 {
                    size1 & !7
                } else {
                    0
                }
            }
            _ => return None,
        };
        Some(Prog {
            kind,
            fs,
            size1,
            w: w.to_vec(),
            n_unfused,
        })
    }
}

/// `a + b` (symmetric) or `a - b` (antisymmetric), lane-wise.
#[inline(always)]
fn comb<L: Lane, const K: usize, const ANTI: bool>(a: &[L; K], b: &[L; K]) -> [L; K] {
    let mut s = *a;
    for k in 0..K {
        s[k] = if ANTI { a[k].sub(b[k]) } else { a[k].add(b[k]) };
    }
    s
}

/// `acc += s * w`, fused or as a rounded product plus an add.
#[inline(always)]
fn madd<L: Lane, const K: usize, const FUSED: bool>(acc: &mut [L; K], s: &[L; K], w: L) {
    for k in 0..K {
        acc[k] = if FUSED {
            acc[k].fma(s[k], w)
        } else {
            acc[k].add(s[k].mul(w))
        };
    }
}

#[inline(always)]
fn scale<L: Lane, const K: usize>(a: &[L; K], w: L) -> [L; K] {
    let mut r = *a;
    for k in 0..K {
        r[k] = a[k].mul(w);
    }
    r
}

/// Where the extended rows are.
pub trait Rows: Copy {
    /// Start of extended row `t`.
    unsafe fn row(self, t: usize) -> *const f64;
}

/// Row `t` at `base + t * stride`.
#[derive(Clone, Copy)]
pub struct Strided {
    pub base: *const f64,
    pub stride: usize,
}

impl Rows for Strided {
    #[inline(always)]
    unsafe fn row(self, t: usize) -> *const f64 {
        unsafe { self.base.add(t * self.stride) }
    }
}

/// Row `t` at `ptrs[t]`.
#[derive(Clone, Copy)]
pub struct Ptrs(pub *const *const f64);

impl Rows for Ptrs {
    #[inline(always)]
    unsafe fn row(self, t: usize) -> *const f64 {
        unsafe { *self.0.add(t) }
    }
}

// In the blocks below, extended row `t` of the block's first output lane `j`
// is at `rows.row(t) + j`; the second output of the two-row variants is one
// row further down, so the two share all but two rows and each loaded row
// serves both.

/// Symmetric/antisymmetric, one output row:
/// `iline[0]*fw[0]`, then `+= (iline[jj] +- iline[-jj]) * fw[jj]` for jj = -size1..-1.
#[inline(always)]
unsafe fn pair1<R: Rows, L: Lane, const K: usize, const ANTI: bool>(
    w: *const f64,
    s1: usize,
    nu: usize,
    rows: R,
    j: usize,
    out: *mut f64,
) {
    unsafe {
        let fs = 2 * s1 + 1;
        let row = |t: usize| rows.row(t).add(j);
        let mut acc = scale::<L, K>(&load_k(row(s1)), L::splat(*w.add(s1)));
        for it in 0..nu {
            let s = comb::<L, K, ANTI>(&load_k(row(it)), &load_k(row(fs - 1 - it)));
            madd::<L, K, false>(&mut acc, &s, L::splat(*w.add(it)));
        }
        for it in nu..s1 {
            let s = comb::<L, K, ANTI>(&load_k(row(it)), &load_k(row(fs - 1 - it)));
            madd::<L, K, true>(&mut acc, &s, L::splat(*w.add(it)));
        }
        store_k(out.add(j), acc);
    }
}

/// Symmetric/antisymmetric, two consecutive output rows (rows 0..=fs).
#[inline(always)]
#[allow(unused_assignments)]
unsafe fn pair2<R: Rows, L: Lane, const K: usize, const ANTI: bool>(
    w: *const f64,
    s1: usize,
    nu: usize,
    rows: R,
    j: usize,
    out0: *mut f64,
    out1: *mut f64,
) {
    unsafe {
        let fs = 2 * s1 + 1;
        let row = |t: usize| rows.row(t).add(j);
        let w0 = L::splat(*w.add(s1));
        let mut acc0 = scale::<L, K>(&load_k(row(s1)), w0);
        let mut acc1 = scale::<L, K>(&load_k(row(s1 + 1)), w0);
        // Pair k = size1 - it: output 0 uses rows (it, fs-1-it), output 1
        // uses (it+1, fs-it); `a` and `bp` carry the shared rows. Steps are
        // unrolled by two so the carried rows need no register moves.
        let mut a: [L; K] = load_k(row(0));
        let mut bp: [L; K] = load_k(row(fs));
        macro_rules! steps {
            ($fused:expr, $from:expr, $to:expr) => {{
                let (mut it, to) = ($from, $to);
                while it + 2 <= to {
                    let a1: [L; K] = load_k(row(it + 1));
                    let b1: [L; K] = load_k(row(fs - 1 - it));
                    let wk = L::splat(*w.add(it));
                    madd::<L, K, $fused>(&mut acc0, &comb::<L, K, ANTI>(&a, &b1), wk);
                    madd::<L, K, $fused>(&mut acc1, &comb::<L, K, ANTI>(&a1, &bp), wk);
                    let a2: [L; K] = load_k(row(it + 2));
                    let b2: [L; K] = load_k(row(fs - 2 - it));
                    let wk = L::splat(*w.add(it + 1));
                    madd::<L, K, $fused>(&mut acc0, &comb::<L, K, ANTI>(&a1, &b2), wk);
                    madd::<L, K, $fused>(&mut acc1, &comb::<L, K, ANTI>(&a2, &b1), wk);
                    a = a2;
                    bp = b2;
                    it += 2;
                }
                if it < to {
                    let a1: [L; K] = load_k(row(it + 1));
                    let b1: [L; K] = load_k(row(fs - 1 - it));
                    let wk = L::splat(*w.add(it));
                    madd::<L, K, $fused>(&mut acc0, &comb::<L, K, ANTI>(&a, &b1), wk);
                    madd::<L, K, $fused>(&mut acc1, &comb::<L, K, ANTI>(&a1, &bp), wk);
                    a = a1;
                    bp = b1;
                }
            }};
        }
        steps!(false, 0, nu);
        steps!(true, nu, s1);
        store_k(out0.add(j), acc0);
        store_k(out1.add(j), acc1);
    }
}

/// General weights, one output row:
/// `iline[size2]*fw[size2]`, then `+= iline[jj]*fw[jj]` for jj = -size1..size2-1.
#[inline(always)]
unsafe fn gen1<R: Rows, L: Lane, const K: usize>(p: &Prog, rows: R, j: usize, out: *mut f64) {
    unsafe {
        let (fs, w) = (p.fs, p.w.as_ptr());
        let row = |t: usize| rows.row(t).add(j);
        let mut acc = scale::<L, K>(&load_k(row(fs - 1)), L::splat(*w.add(fs - 1)));
        for it in 0..p.n_unfused {
            madd::<L, K, false>(&mut acc, &load_k(row(it)), L::splat(*w.add(it)));
        }
        for it in p.n_unfused..fs - 1 {
            madd::<L, K, true>(&mut acc, &load_k(row(it)), L::splat(*w.add(it)));
        }
        store_k(out.add(j), acc);
    }
}

/// General weights, two consecutive output rows (rows 0..=fs).
#[inline(always)]
#[allow(unused_assignments)]
unsafe fn gen2<R: Rows, L: Lane, const K: usize>(
    p: &Prog,
    rows: R,
    j: usize,
    out0: *mut f64,
    out1: *mut f64,
) {
    unsafe {
        let (fs, w) = (p.fs, p.w.as_ptr());
        let row = |t: usize| rows.row(t).add(j);
        let wl = L::splat(*w.add(fs - 1));
        let mut acc0 = scale::<L, K>(&load_k(row(fs - 1)), wl);
        let mut acc1 = scale::<L, K>(&load_k(row(fs)), wl);
        let mut cur: [L; K] = load_k(row(0));
        macro_rules! steps {
            ($fused:expr, $from:expr, $to:expr) => {{
                let (mut it, to) = ($from, $to);
                while it + 2 <= to {
                    let n1: [L; K] = load_k(row(it + 1));
                    let wk = L::splat(*w.add(it));
                    madd::<L, K, $fused>(&mut acc0, &cur, wk);
                    madd::<L, K, $fused>(&mut acc1, &n1, wk);
                    let n2: [L; K] = load_k(row(it + 2));
                    let wk = L::splat(*w.add(it + 1));
                    madd::<L, K, $fused>(&mut acc0, &n1, wk);
                    madd::<L, K, $fused>(&mut acc1, &n2, wk);
                    cur = n2;
                    it += 2;
                }
                if it < to {
                    let n1: [L; K] = load_k(row(it + 1));
                    let wk = L::splat(*w.add(it));
                    madd::<L, K, $fused>(&mut acc0, &cur, wk);
                    madd::<L, K, $fused>(&mut acc1, &n1, wk);
                    cur = n1;
                }
            }};
        }
        steps!(false, 0, p.n_unfused);
        steps!(true, p.n_unfused, fs - 1);
        store_k(out0.add(j), acc0);
        store_k(out1.add(j), acc1);
    }
}

/// Run `$body` for lane offsets `$j` covering `0..$lanes`: blocks of 4
/// vectors (`$L = V, $K = 4`), then single vectors, then scalars.
macro_rules! lanes {
    ($lanes:expr, |$j:ident, $L:ident, $K:ident| $body:expr) => {
        lanes!([4, 1]; $lanes, |$j, $L, $K| $body)
    };
    ([$($k:literal),*]; $lanes:expr, |$j:ident, $L:ident, $K:ident| $body:expr) => {{
        let lanes = $lanes;
        let mut $j = 0usize;
        $({
            type $L = V;
            const $K: usize = $k;
            while $j + $k * <V as Lane>::W <= lanes {
                $body;
                $j += $k * <V as Lane>::W;
            }
        })*
        {
            type $L = f64;
            const $K: usize = 1;
            while $j < lanes {
                $body;
                $j += 1;
            }
        }
    }};
}

/// Which `n_unfused` a specialized kernel was built for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Nu {
    Zero,
    All,
    Clang,
}

/// The specialized kernel family covering `p.n_unfused` on this target (see
/// `spec_match`; when size1 is a multiple of 8, clang's scheme and "all
/// separate" coincide).
fn nu_class(p: &Prog) -> Option<Nu> {
    let (s1, nu) = (p.size1, p.n_unfused);
    if nu == 0 {
        Some(Nu::Zero)
    } else if cfg!(target_arch = "aarch64") {
        (nu == s1 & !7).then_some(Nu::Clang)
    } else if nu == s1 {
        Some(Nu::All)
    } else {
        (nu == s1 & !7).then_some(Nu::Clang)
    }
}

/// Symmetric/antisymmetric kernels with `size1` and `n_unfused` known at
/// compile time: the tap loops unroll completely, so the per-step overhead
/// disappears and, for a line (`Strided` with stride 1), loads of the same
/// vector by different taps are merged.
unsafe fn corr1_spec<R: Rows, const S1: usize, const NU: usize, const ANTI: bool>(
    w: *const f64,
    rows: R,
    lanes: usize,
    out: *mut f64,
) {
    unsafe {
        lanes!(lanes, |j, L, K| pair1::<R, L, K, ANTI>(
            w, S1, NU, rows, j, out
        ))
    }
}

unsafe fn corr2_spec<R: Rows, const S1: usize, const NU: usize, const ANTI: bool>(
    w: *const f64,
    rows: R,
    lanes: usize,
    out0: *mut f64,
    out1: *mut f64,
) {
    unsafe {
        lanes!(lanes, |j, L, K| pair2::<R, L, K, ANTI>(
            w, S1, NU, rows, j, out0, out1
        ))
    }
}

/// Call `$f::<_, S1, NU, ANTI>($args)` for the specialized (size1, scheme)
/// combinations this target's SciPy builds use (separate multiply-adds on
/// x86-64; fused or clang's mixed scheme on aarch64), returning true; false
/// when there is no specialization.
macro_rules! spec_match {
    ($f:ident, $p:expr, $args:tt, [$($s:literal)*]) => {{
        let p: &Prog = $p;
        let anti = match p.kind {
            Kind::Sym => false,
            Kind::Anti => true,
            Kind::Gen => return false,
        };
        let Some(nu) = nu_class(p) else { return false };
        match (p.size1, nu, anti) {
            $(
                #[cfg(not(target_arch = "x86_64"))]
                ($s, Nu::Zero, false) => { unsafe { $f::<_, $s, 0, false>$args }; true }
                #[cfg(not(target_arch = "x86_64"))]
                ($s, Nu::Zero, true) => { unsafe { $f::<_, $s, 0, true>$args }; true }
                #[cfg(not(target_arch = "aarch64"))]
                ($s, Nu::All, false) => { unsafe { $f::<_, $s, $s, false>$args }; true }
                #[cfg(not(target_arch = "aarch64"))]
                ($s, Nu::All, true) => { unsafe { $f::<_, $s, $s, true>$args }; true }
                #[cfg(not(target_arch = "x86_64"))]
                ($s, Nu::Clang, false) => { unsafe { $f::<_, $s, { $s & !7 }, false>$args }; true }
                #[cfg(not(target_arch = "x86_64"))]
                ($s, Nu::Clang, true) => { unsafe { $f::<_, $s, { $s & !7 }, true>$args }; true }
            )*
            _ => false,
        }
    }};
}

/// One output row through a specialized kernel; false if there is none.
pub unsafe fn correlate_spec<R: Rows>(p: &Prog, rows: R, lanes: usize, out: *mut f64) -> bool {
    let w = p.w.as_ptr();
    spec_match!(corr1_spec, p, (w, rows, lanes, out), [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16])
}

/// Two output rows through a specialized kernel; false if there is none.
pub unsafe fn correlate2_spec<R: Rows>(
    p: &Prog,
    rows: R,
    lanes: usize,
    out0: *mut f64,
    out1: *mut f64,
) -> bool {
    let w = p.w.as_ptr();
    spec_match!(corr2_spec, p, (w, rows, lanes, out0, out1), [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16])
}

/// `out[j]` for `j < lanes`: the correlation of extended rows `0..fs`.
///
/// # Safety
/// Rows `0..fs` are valid for `lanes` reads; `out` for `lanes` writes.
pub unsafe fn correlate<R: Rows>(p: &Prog, rows: R, lanes: usize, out: *mut f64) {
    unsafe {
        match p.kind {
            Kind::Sym => lanes!(lanes, |j, L, K| pair1::<R, L, K, false>(
                p.w.as_ptr(),
                p.size1,
                p.n_unfused,
                rows,
                j,
                out
            )),
            Kind::Anti => lanes!(lanes, |j, L, K| pair1::<R, L, K, true>(
                p.w.as_ptr(),
                p.size1,
                p.n_unfused,
                rows,
                j,
                out
            )),
            Kind::Gen => lanes!(lanes, |j, L, K| gen1::<R, L, K>(p, rows, j, out)),
        }
    }
}

/// Two consecutive output rows: `out0` from rows `0..fs` and `out1` from
/// rows `1..=fs`.
///
/// # Safety
/// Rows `0..=fs` are valid for `lanes` reads; outputs for `lanes` writes.
pub unsafe fn correlate2<R: Rows>(p: &Prog, rows: R, lanes: usize, out0: *mut f64, out1: *mut f64) {
    unsafe {
        match p.kind {
            Kind::Sym => lanes!(lanes, |j, L, K| pair2::<R, L, K, false>(
                p.w.as_ptr(),
                p.size1,
                p.n_unfused,
                rows,
                j,
                out0,
                out1
            )),
            Kind::Anti => lanes!(lanes, |j, L, K| pair2::<R, L, K, true>(
                p.w.as_ptr(),
                p.size1,
                p.n_unfused,
                rows,
                j,
                out0,
                out1
            )),
            Kind::Gen => lanes!(lanes, |j, L, K| gen2::<R, L, K>(p, rows, j, out0, out1)),
        }
    }
}

// ---------------------------------------------------------------------------
// Uniform filter: running sum, as NI_UniformFilter1D.

/// `tmp[j] = 0.0 + row0[j] + ... + row(n-1)[j]` (in order).
pub unsafe fn uniform_init<R: Rows>(rows: R, n: usize, lanes: usize, tmp: &mut [f64]) {
    let tmp = &mut tmp[..lanes];
    tmp.fill(0.0);
    for t in 0..n {
        let r = unsafe { std::slice::from_raw_parts(rows.row(t), lanes) };
        for (x, &v) in tmp.iter_mut().zip(r) {
            *x += v;
        }
    }
}

/// `out[j] = tmp[j] / size` for `j < tmp.len()`.
pub unsafe fn uniform_out(tmp: &[f64], size: f64, out: *mut f64) {
    for (j, &t) in tmp.iter().enumerate() {
        unsafe { *out.add(j) = t / size };
    }
}

#[inline(always)]
unsafe fn uni_block<L: Lane, const K: usize>(
    enter: *const f64,
    leave: *const f64,
    tmp: *mut f64,
    size: L,
    out: *mut f64,
    j: usize,
) {
    unsafe {
        let e: [L; K] = load_k(enter.add(j));
        let l: [L; K] = load_k(leave.add(j));
        let mut t: [L; K] = load_k(tmp.add(j));
        let mut o = t;
        for k in 0..K {
            t[k] = t[k].add(e[k].sub(l[k]));
            o[k] = t[k].div(size);
        }
        store_k(tmp.add(j), t);
        store_k(out.add(j), o);
    }
}

/// `tmp[j] += enter[j] - leave[j]; out[j] = tmp[j] / size` for `j < lanes`.
pub unsafe fn uniform_step(
    enter: *const f64,
    leave: *const f64,
    lanes: usize,
    tmp: &mut [f64],
    size: f64,
    out: *mut f64,
) {
    assert!(tmp.len() >= lanes);
    let tmp = tmp.as_mut_ptr();
    unsafe {
        lanes!(lanes, |j, L, K| uni_block::<L, K>(
            enter,
            leave,
            tmp,
            L::splat(size),
            out,
            j
        ))
    }
}

// ---------------------------------------------------------------------------
// N-D correlation (NI_Correlate): out = 0.0 (+ w * v)* over the taps in order.

/// One term of an N-D correlation: weight `w` times `rows[row][j + off]`.
#[derive(Clone, Copy)]
pub struct Tap {
    pub row: usize,
    pub off: usize,
    pub w: f64,
}

#[inline(always)]
unsafe fn nd_block<L: Lane, const K: usize, const FUSED: bool>(
    taps: &[Tap],
    rows: &[*const f64],
    j: usize,
    out: *mut f64,
) {
    unsafe {
        let mut acc = [L::splat(0.0); K];
        for t in taps {
            let v: [L; K] = load_k(rows.get_unchecked(t.row).add(j + t.off));
            let w = L::splat(t.w);
            for k in 0..K {
                // `_res += _weights[_ii] * value`
                acc[k] = if FUSED {
                    acc[k].fma(w, v[k])
                } else {
                    acc[k].add(w.mul(v[k]))
                };
            }
        }
        store_k(out.add(j), acc);
    }
}

/// `out[j] = sum over taps of w * rows[row][j + off]`, accumulated in tap
/// order from 0.0, for `j < lanes`.
pub unsafe fn nd_correlate(
    taps: &[Tap],
    rows: &[*const f64],
    lanes: usize,
    fused: bool,
    out: *mut f64,
) {
    unsafe {
        if fused {
            lanes!([16, 4, 1]; lanes, |j, L, K| nd_block::<L, K, true>(taps, rows, j, out))
        } else {
            lanes!([16, 4, 1]; lanes, |j, L, K| nd_block::<L, K, false>(taps, rows, j, out))
        }
    }
}

// ---------------------------------------------------------------------------
// 2-D correlation with dense kernel rows, two output rows at a time.
//
// Output row i applies kernel row a to input row R_a (a < k0); output row
// i + 1 applies it to R_{a+1}. Step g loads R_g once and serves kernel row g
// of output i and kernel row g - 1 of output i + 1. Within a row, tap t of a
// block of K vectors reads the vectors at offsets t + 2k: even and odd
// offsets form two sets, each loaded once, shared by all taps of both rows.

#[inline(always)]
unsafe fn nd_pair_block<const M: usize, const K: usize, const FUSED: bool>(
    w: *const f64,
    k0: usize,
    rows: *const *const f64,
    j: usize,
    out0: *mut f64,
    out1: *mut f64,
) {
    const SET: usize = 16;
    unsafe {
        let mut acc0 = [V::splat(0.0); K];
        let mut acc1 = [V::splat(0.0); K];
        let n_even = K + (M - 1) / 2;
        let n_odd = if M > 1 { K + (M - 2) / 2 } else { 0 };
        for g in 0..=k0 {
            let r = (*rows.add(g)).add(j);
            let e: [V; SET] = std::array::from_fn(|i| {
                if i < n_even {
                    V::load(r.add(2 * i))
                } else {
                    V::splat(0.0)
                }
            });
            let o: [V; SET] = std::array::from_fn(|i| {
                if i < n_odd {
                    V::load(r.add(1 + 2 * i))
                } else {
                    V::splat(0.0)
                }
            });
            let apply = |acc: &mut [V; K], wr: *const f64| {
                for t in 0..M {
                    let wt = V::splat(*wr.add(t));
                    for k in 0..K {
                        let v = if t % 2 == 0 {
                            e[t / 2 + k]
                        } else {
                            o[t / 2 + k]
                        };
                        // `_res += _weights[_ii] * value`
                        acc[k] = if FUSED {
                            acc[k].fma(wt, v)
                        } else {
                            acc[k].add(wt.mul(v))
                        };
                    }
                }
            };
            if g < k0 {
                apply(&mut acc0, w.add(g * M));
            }
            if g >= 1 {
                apply(&mut acc1, w.add((g - 1) * M));
            }
        }
        for k in 0..K {
            acc0[k].store(out0.add(j + 2 * k));
            acc1[k].store(out1.add(j + 2 * k));
        }
    }
}

unsafe fn nd_pair_m<const M: usize, const FUSED: bool>(
    w: *const f64,
    k0: usize,
    rows: *const *const f64,
    lanes: usize,
    out0: *mut f64,
    out1: *mut f64,
) -> usize {
    const K: usize = 6;
    let mut j = 0;
    while j + K * <V as Lane>::W <= lanes {
        unsafe { nd_pair_block::<M, K, FUSED>(w, k0, rows, j, out0, out1) };
        j += K * <V as Lane>::W;
    }
    j
}

/// Widest kernel row `nd_pair` handles.
pub const ND_PAIR_MAX_M: usize = 7;

/// Two consecutive output rows of a 2-D correlation whose `k0 x m` kernel
/// has every weight above DBL_EPSILON (`w` row-major, `taps` the same terms):
/// `rows[g]` for `g <= k0` are the extended input rows, output 0 uses
/// `rows[0..k0]` and output 1 `rows[1..=k0]`.
///
/// # Safety
/// Rows valid for `lanes + m - 1` reads; outputs for `lanes` writes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn nd_pair(
    w: &[f64],
    m: usize,
    taps: &[Tap],
    rows: &[*const f64],
    lanes: usize,
    fused: bool,
    out0: *mut f64,
    out1: *mut f64,
) {
    let k0 = rows.len() - 1;
    debug_assert!(w.len() == k0 * m && taps.len() == k0 * m && m <= ND_PAIR_MAX_M);
    let (wp, rp) = (w.as_ptr(), rows.as_ptr());
    macro_rules! go {
        ($($m:literal)*) => {
            match (m, fused) {
                $(
                    ($m, true) => unsafe { nd_pair_m::<$m, true>(wp, k0, rp, lanes, out0, out1) },
                    ($m, false) => unsafe { nd_pair_m::<$m, false>(wp, k0, rp, lanes, out0, out1) },
                )*
                _ => 0,
            }
        };
    }
    let done = go!(1 2 3 4 5 6 7);
    if done < lanes {
        let mut r0 = [std::ptr::null::<f64>(); 64];
        let mut r1 = [std::ptr::null::<f64>(); 64];
        let k0 = k0.min(64);
        for g in 0..k0 {
            r0[g] = unsafe { rows[g].add(done) };
            r1[g] = unsafe { rows[g + 1].add(done) };
        }
        unsafe {
            nd_correlate(taps, &r0[..k0], lanes - done, fused, out0.add(done));
            nd_correlate(taps, &r1[..k0], lanes - done, fused, out1.add(done));
        }
    }
}
