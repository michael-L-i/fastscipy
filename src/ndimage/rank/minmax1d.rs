//! 1-D minimum / maximum filter along one axis (`NI_MinOrMaxFilter1D`).
//!
//! Fast path: sliding min/max by doubling (span 1, 2, 4, ... then one final
//! combination of two overlapping spans), vectorized across adjacent
//! elements. min/max are idempotent, so overlapping spans give the exact
//! window extreme.
//!
//! Exact path, for lines holding NaN or -0.0 (or a NaN / -0.0 constant):
//! Richard Harter's ascending-minima algorithm as in SciPy, in double
//! precision, always over whole lines (with NaN its result depends on values
//! before the window, so lines cannot be split).
//!
//! Input and output may be the same array: every task copies what it reads
//! into a private buffer before writing, and tasks write disjoint regions
//! that no other task reads in that case.

use std::sync::atomic::{AtomicBool, Ordering};

use super::boundary::axis_table;
use super::elem::RElem;
use super::engine::SendPtr;
use super::par::{self, Arena, Arenas};
use super::simd::{HasVec, Vx};
use crate::ndimage::common::Mode;

/// Column tile width in bytes for in-place filtering along a non-contiguous
/// axis (whole columns are buffered).
const TILE_BYTES: usize = 256;
/// Column tile width in bytes when streaming rows along a non-contiguous axis
/// (wide, so reads are long contiguous runs), and the target number of
/// output rows per task.
const STREAM_BYTES: usize = 16384;
const CHUNK_ROWS: usize = 48;
/// Windows up to this size are reduced directly; larger ones by van Herk /
/// Gil-Werman (3 operations per element whatever the size).
const DIRECT_MAX: usize = 5;

#[derive(Clone, Copy)]
pub struct Params {
    pub size: usize,
    /// size / 2 + origin: number of elements of extension before the line
    pub before: usize,
    pub mode: Mode,
    pub minimum: bool,
}

/// The constant value: as an element for the fast path, as SciPy's double for
/// the exact path, and whether it forces the exact path.
#[derive(Clone, Copy)]
pub struct Cval<T> {
    pub t: T,
    pub raw: f64,
    pub special: bool,
}

/// dst[i] = op(a[i], b[i]) for i < n.
#[inline(always)]
unsafe fn combine<T: HasVec>(minimum: bool, dst: *mut T, a: *const T, b: *const T, n: usize) {
    let l = <T::V as Vx<T>>::LANES;
    let mut i = 0;
    unsafe {
        if minimum {
            while i + 2 * l <= n {
                let x0 = T::V::load(a.add(i)).mn(T::V::load(b.add(i)));
                let x1 = T::V::load(a.add(i + l)).mn(T::V::load(b.add(i + l)));
                x0.store(dst.add(i));
                x1.store(dst.add(i + l));
                i += 2 * l;
            }
            while i < n {
                *dst.add(i) = T::mn(*a.add(i), *b.add(i));
                i += 1;
            }
        } else {
            while i + 2 * l <= n {
                let x0 = T::V::load(a.add(i)).mx(T::V::load(b.add(i)));
                let x1 = T::V::load(a.add(i + l)).mx(T::V::load(b.add(i + l)));
                x0.store(dst.add(i));
                x1.store(dst.add(i + l));
                i += 2 * l;
            }
            while i < n {
                *dst.add(i) = T::mx(*a.add(i), *b.add(i));
                i += 1;
            }
        }
    }
}

/// Sliding extreme over `rows` units of width `w` (flat, unit = w elements):
/// out unit y = op over ext units y .. y + k. `ext` holds rows + k - 1 units
/// and is clobbered; `tmp` is scratch of the same size.
fn sliding<T: HasVec>(
    minimum: bool,
    ext: &mut [T],
    tmp: &mut [T],
    rows: usize,
    w: usize,
    k: usize,
    out: &mut [T],
) {
    assert!(ext.len() >= (rows + k - 1) * w && tmp.len() >= ext.len() && out.len() >= rows * w);
    if k == 1 {
        out[..rows * w].copy_from_slice(&ext[..rows * w]);
        return;
    }
    let mut len = rows + k - 1; // valid units in cur
    let mut span = 1;
    let (mut cur, mut nxt) = (ext.as_mut_ptr(), tmp.as_mut_ptr());
    unsafe {
        while span * 2 <= k {
            let n = len - span;
            if span * 4 > k && k - span * 2 == 0 {
                // last level lands exactly on k: write the output directly
                combine(minimum, out.as_mut_ptr(), cur, cur.add(span * w), rows * w);
                return;
            }
            combine(minimum, nxt, cur, cur.add(span * w), n * w);
            std::mem::swap(&mut cur, &mut nxt);
            len = n;
            span *= 2;
        }
        let sh = k - span;
        combine(minimum, out.as_mut_ptr(), cur, cur.add(sh * w), rows * w);
    }
}

#[inline]
fn has_special<T: RElem>(s: &[T]) -> bool {
    if !T::is_float() {
        return false;
    }
    let mut sp = false;
    for &v in s {
        sp |= v.is_special();
    }
    sp
}

/// Fill `ext` with the boundary-extended line(s) and report special values.
/// `row(q)` gives the source pointer of extended unit q (w elements), or None
/// for the constant value.
#[inline]
fn fill_ext<T: RElem>(
    ext: &mut [T],
    units: usize,
    w: usize,
    cval: T,
    mut row: impl FnMut(usize) -> Option<*const T>,
) -> bool {
    let mut special = false;
    for q in 0..units {
        let dst = &mut ext[q * w..(q + 1) * w];
        match row(q) {
            None => dst.fill(cval),
            Some(p) => {
                let s = unsafe { std::slice::from_raw_parts(p, w) };
                dst.copy_from_slice(s);
                if !special {
                    special = has_special(s);
                }
            }
        }
    }
    special
}

/// Copy `src` to `dst`, reporting whether it holds NaN or -0.0.
#[inline]
fn copy_check<T: RElem>(dst: &mut [T], src: &[T]) -> bool {
    if !T::is_float() {
        dst.copy_from_slice(src);
        return false;
    }
    let mut sp = false;
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = v;
        sp |= v.is_special();
    }
    sp
}

/// Runs of the extension table: (start q, source index or -1, length).
fn segments(tbl: &[i64]) -> Vec<(usize, i64, usize)> {
    let mut segs = Vec::new();
    let mut q = 0;
    while q < tbl.len() {
        let s = tbl[q];
        let mut e = q + 1;
        if s < 0 {
            while e < tbl.len() && tbl[e] < 0 {
                e += 1;
            }
        } else {
            while e < tbl.len() && tbl[e] == s + (e - q) as i64 {
                e += 1;
            }
        }
        segs.push((q, s, e - q));
        q = e;
    }
    segs
}

/// dst = op over the `K` rows `r` (w elements), accumulated in registers.
#[inline(always)]
unsafe fn reduce_rows<T: HasVec, const K: usize>(
    minimum: bool,
    dst: *mut T,
    r: [*const T; K],
    w: usize,
) {
    let l = <T::V as Vx<T>>::LANES;
    let mut i = 0;
    unsafe {
        macro_rules! body {
            ($op:ident, $sop:ident) => {{
                while i + 2 * l <= w {
                    let mut a0 = T::V::load(r[0].add(i));
                    let mut a1 = T::V::load(r[0].add(i + l));
                    for j in 1..K {
                        a0 = a0.$op(T::V::load(r[j].add(i)));
                        a1 = a1.$op(T::V::load(r[j].add(i + l)));
                    }
                    a0.store(dst.add(i));
                    a1.store(dst.add(i + l));
                    i += 2 * l;
                }
                while i < w {
                    let mut a = *r[0].add(i);
                    for j in 1..K {
                        a = T::$sop(a, *r[j].add(i));
                    }
                    *dst.add(i) = a;
                    i += 1;
                }
            }};
        }
        if minimum {
            body!(mn, mn)
        } else {
            body!(mx, mx)
        }
    }
}

/// Reduce rows `q0 .. q0 + k` (k <= DIRECT_MAX) of a row source into `dst`.
#[inline(always)]
unsafe fn direct<T: HasVec>(
    minimum: bool,
    dst: *mut T,
    row: &impl Fn(usize) -> *const T,
    q0: usize,
    k: usize,
    w: usize,
) {
    unsafe {
        match k {
            1 => std::ptr::copy_nonoverlapping(row(q0), dst, w),
            2 => reduce_rows::<T, 2>(minimum, dst, [row(q0), row(q0 + 1)], w),
            3 => reduce_rows::<T, 3>(minimum, dst, [row(q0), row(q0 + 1), row(q0 + 2)], w),
            4 => reduce_rows::<T, 4>(
                minimum,
                dst,
                [row(q0), row(q0 + 1), row(q0 + 2), row(q0 + 3)],
                w,
            ),
            5 => reduce_rows::<T, 5>(
                minimum,
                dst,
                [row(q0), row(q0 + 1), row(q0 + 2), row(q0 + 3), row(q0 + 4)],
                w,
            ),
            _ => unreachable!(),
        }
    }
}

/// Filter `shape` along `axis` from `src` to `dst` (T to T; may alias when
/// `inplace`).
///
/// # Safety
/// `src` and `dst` must point to `shape.iter().product()` elements and may
/// overlap only if they are equal and `inplace` is set.
pub unsafe fn filter<T: HasVec>(
    src: *const T,
    dst: *mut T,
    shape: &[usize],
    axis: usize,
    p: Params,
    cv: Cval<T>,
    inplace: bool,
) {
    let n = shape[axis];
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let k = p.size;
    let after = k - 1 - p.before;
    let tbl = axis_table(p.mode, n, p.before, after);
    let ext_len = n + k - 1;
    let srcp = SendPtr(src as *mut T);
    let dstp = SendPtr(dst);
    let all_exact = T::is_float() && cv.special && tbl.iter().any(|&v| v < 0);
    let levels = (usize::BITS - k.leading_zeros()) as usize + 1;
    if inner == 1 {
        // contiguous lines: copy each extended line (in-place safe), then
        // reduce from the copy straight into the destination
        let segs = segments(&tbl);
        par::for_each(
            outer,
            par::items_per_chunk(par::work(n, levels, std::mem::size_of::<T>())),
            |line, ar| {
                let Arenas { a, b, c, d } = ar;
                let (src, dst) = (srcp, dstp);
                let ext = a.take::<T>(ext_len);
                let s = unsafe { std::slice::from_raw_parts(src.0.add(line * n), n) };
                let mut special = false;
                for &(q, t, len) in &segs {
                    if t < 0 {
                        ext[q..q + len].fill(cv.t);
                    } else {
                        special |=
                            copy_check(&mut ext[q..q + len], &s[t as usize..t as usize + len]);
                    }
                }
                let o = unsafe { std::slice::from_raw_parts_mut(dst.0.add(line * n), n) };
                if special || all_exact {
                    Harter::new(c, d, n, k).run_line(ext, &tbl, cv.raw, p.minimum, o);
                } else if k <= DIRECT_MAX {
                    let e = ext.as_ptr();
                    unsafe { direct(p.minimum, o.as_mut_ptr(), &|q| e.add(q), 0, k, n) };
                } else {
                    sliding(p.minimum, ext, b.take::<T>(ext_len), n, 1, k, o);
                }
            },
        );
        return;
    }
    if inplace || all_exact {
        // whole columns per task, buffered before writing
        let tw = (TILE_BYTES / std::mem::size_of::<T>()).min(inner);
        let ntiles = inner.div_ceil(tw);
        par::for_each(
            outer * ntiles,
            par::items_per_chunk(par::work(n * tw, levels, std::mem::size_of::<T>())),
            |task, ar| {
                let Arenas { a, b, c, d } = ar;
                let (src, dst) = (srcp, dstp);
                let slab = (task / ntiles) * n * inner;
                let c0 = (task % ntiles) * tw;
                let w = tw.min(inner - c0);
                let (ext, rest) = a
                    .take::<T>(2 * ext_len * w + n * w)
                    .split_at_mut(ext_len * w);
                let (tmp, o) = rest.split_at_mut(ext_len * w);
                let special = fill_ext(ext, ext_len, w, cv.t, |q| {
                    let t = tbl[q];
                    (t >= 0)
                        .then(|| unsafe { src.0.add(slab + t as usize * inner + c0) as *const T })
                });
                if special || all_exact {
                    Harter::new(c, d, n, k).run_columns(ext, w, &tbl, cv.raw, p.minimum, o);
                } else {
                    sliding(p.minimum, ext, tmp, n, w, k, o);
                }
                let _ = b;
                for y in 0..n {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            o.as_ptr().add(y * w),
                            dst.0.add(slab + y * inner + c0),
                            w,
                        )
                    };
                }
            },
        );
        return;
    }
    // Stream rows straight from the source: tasks are (column tile, chunk of
    // output rows). Column tiles holding special values are redone whole by
    // the exact path afterwards (the source is intact: not in place).
    let tw = (STREAM_BYTES / std::mem::size_of::<T>()).min(inner);
    let ntiles = inner.div_ceil(tw);
    // van Herk works on blocks of k rows: keep chunks several blocks long
    let chunk_rows = if k <= DIRECT_MAX {
        CHUNK_ROWS
    } else {
        CHUNK_ROWS.max(4 * k)
    };
    let nchunks = n.div_ceil(chunk_rows);
    let flagged: Vec<AtomicBool> = (0..outer * ntiles)
        .map(|_| AtomicBool::new(false))
        .collect();
    let crow = vec![cv.t; tw];
    let crow = SendPtr(crow.as_ptr() as *mut T);
    let task_work = par::work(chunk_rows * tw, k.min(levels + 2), std::mem::size_of::<T>());
    par::for_each(
        outer * ntiles * nchunks,
        par::items_per_chunk(task_work),
        |task, ar| {
            let (src, dst, crow) = (srcp, dstp, crow);
            let chunk = task % nchunks;
            let tile_id = task / nchunks;
            let slab = (tile_id / ntiles) * n * inner;
            let c0 = (tile_id % ntiles) * tw;
            let w = tw.min(inner - c0);
            let y0 = chunk * chunk_rows;
            let y1 = (y0 + chunk_rows).min(n);
            let row = |q: usize| -> *const T {
                let t = tbl[q];
                if t < 0 {
                    crow.0 as *const T
                } else {
                    unsafe { src.0.add(slab + t as usize * inner + c0) }
                }
            };
            let check = |q: usize| -> bool {
                tbl[q] >= 0 && has_special(unsafe { std::slice::from_raw_parts(row(q), w) })
            };
            let out = |y: usize| -> *mut T { unsafe { dst.0.add(slab + y * inner + c0) } };
            if T::is_float() && (y0..y1 + k - 1).any(check) {
                flagged[tile_id].store(true, Ordering::Relaxed);
                return;
            }
            unsafe {
                if k <= DIRECT_MAX {
                    for y in y0..y1 {
                        direct(p.minimum, out(y), &row, y, k, w);
                    }
                } else {
                    van_herk(
                        p.minimum,
                        &row,
                        &out,
                        y0,
                        y1,
                        k,
                        w,
                        ar.a.take::<T>((k + 1) * w),
                    );
                }
            }
        },
    );
    let redo: Vec<usize> = (0..outer * ntiles)
        .filter(|&i| flagged[i].load(Ordering::Relaxed))
        .collect();
    par::for_each(redo.len(), 1, |i, ar| {
        let Arenas { a, b, c, d } = ar;
        let tile_id = redo[i];
        let (src, dst) = (srcp, dstp);
        let slab = (tile_id / ntiles) * n * inner;
        let c0 = (tile_id % ntiles) * tw;
        let w = tw.min(inner - c0);
        let ext = a.take::<T>(ext_len * w);
        let o = b.take::<T>(n * w);
        fill_ext(ext, ext_len, w, cv.t, |q| {
            let t = tbl[q];
            (t >= 0).then(|| unsafe { src.0.add(slab + t as usize * inner + c0) as *const T })
        });
        Harter::new(c, d, n, k).run_columns(ext, w, &tbl, cv.raw, p.minimum, o);
        for y in 0..n {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    o.as_ptr().add(y * w),
                    dst.0.add(slab + y * inner + c0),
                    w,
                )
            };
        }
    });
}

/// van Herk / Gil-Werman over rows: outputs y0..y1, window rows y .. y + k of
/// `row`. For blocks of k rows starting at b, out[b + i] = op(suffix of the
/// block from i, prefix of the next block up to i - 1). `buf` holds k + 1 rows.
#[allow(clippy::too_many_arguments)]
unsafe fn van_herk<T: HasVec>(
    minimum: bool,
    row: &impl Fn(usize) -> *const T,
    out: &impl Fn(usize) -> *mut T,
    y0: usize,
    y1: usize,
    k: usize,
    w: usize,
    buf: &mut [T],
) {
    let (suf, pre) = buf.split_at_mut(k * w);
    let sp = suf.as_mut_ptr();
    let pp = pre.as_mut_ptr();
    let mut b = y0;
    unsafe {
        while b < y1 {
            // suffix minima of rows b .. b + k
            std::ptr::copy_nonoverlapping(row(b + k - 1), sp.add((k - 1) * w), w);
            for i in (0..k - 1).rev() {
                combine(minimum, sp.add(i * w), row(b + i), sp.add((i + 1) * w), w);
            }
            std::ptr::copy_nonoverlapping(sp, out(b), w);
            let m = (y1 - b).min(k);
            if m > 1 {
                std::ptr::copy_nonoverlapping(row(b + k), pp, w);
                for i in 1..m {
                    combine(minimum, out(b + i), sp.add(i * w), pp, w);
                    if i + 1 < m {
                        combine(minimum, pp, pp, row(b + k + i), w);
                    }
                }
            }
            b += k;
        }
    }
}

/// SciPy's exact algorithm on extended lines held in `ext` (element values;
/// constant positions, where the table is -1, read SciPy's double `cval`).
struct Harter<'a> {
    line: &'a mut [f64],
    out: &'a mut [f64],
    ring: &'a mut [(f64, usize)],
}

impl<'a> Harter<'a> {
    fn new(f: &'a mut Arena, r: &'a mut Arena, n: usize, k: usize) -> Harter<'a> {
        let (line, out) = f.take::<f64>(2 * n + k - 1).split_at_mut(n + k - 1);
        Harter {
            line,
            out,
            ring: r.take::<(f64, usize)>(k),
        }
    }

    fn run_line<T: RElem>(self, ext: &[T], tbl: &[i64], cval: f64, minimum: bool, o: &mut [T]) {
        for (q, (d, &v)) in self.line.iter_mut().zip(ext).enumerate() {
            *d = if tbl[q] < 0 { cval } else { v.to_f64() };
        }
        harter(self.line, self.ring.len(), minimum, self.out, self.ring);
        for (d, &v) in o.iter_mut().zip(self.out.iter()) {
            *d = T::c_cast(v).expect("float conversion");
        }
    }

    /// Columns of a (units x w) block.
    fn run_columns<T: RElem>(
        self,
        ext: &[T],
        w: usize,
        tbl: &[i64],
        cval: f64,
        minimum: bool,
        o: &mut [T],
    ) {
        let n = self.out.len();
        for c in 0..w {
            for (q, d) in self.line.iter_mut().enumerate() {
                *d = if tbl[q] < 0 {
                    cval
                } else {
                    ext[q * w + c].to_f64()
                };
            }
            harter(self.line, self.ring.len(), minimum, self.out, self.ring);
            for y in 0..n {
                o[y * w + c] = T::c_cast(self.out[y]).expect("float conversion");
            }
        }
    }
}

/// Port of the MINLIST/MAXLIST loop of `NI_MinOrMaxFilter1D`.
fn harter(
    iline: &[f64],
    filter_size: usize,
    minimum: bool,
    oline: &mut [f64],
    ring: &mut [(f64, usize)],
) {
    let length = oline.len();
    if filter_size == 1 {
        oline.copy_from_slice(&iline[..length]);
        return;
    }
    let end = filter_size; // ring has filter_size entries
    let mut minpair = 0usize;
    ring[0] = (iline[0], filter_size);
    let mut last = 0usize;
    let mut oi = 0;
    for ll in 1..filter_size + length - 1 {
        let val = iline[ll];
        if ring[minpair].1 == ll {
            minpair += 1;
            if minpair >= end {
                minpair = 0;
            }
        }
        let mv = ring[minpair].0;
        if (minimum && val <= mv) || (!minimum && val >= mv) {
            ring[minpair] = (val, ll + filter_size);
            last = minpair;
        } else {
            loop {
                let lv = ring[last].0;
                let cont = if minimum { lv >= val } else { lv <= val };
                if !cont {
                    break;
                }
                // DECREASE_RING_PTR
                if last == 0 {
                    last = end;
                }
                last -= 1;
            }
            last += 1;
            if last == end {
                last = 0;
            }
            ring[last] = (val, ll + filter_size);
        }
        if ll >= filter_size - 1 {
            oline[oi] = ring[minpair].0;
            oi += 1;
        }
    }
}
