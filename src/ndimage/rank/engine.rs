//! Tiled drivers that evaluate a window function over every output pixel of
//! an n-d array.
//!
//! The output is split into tasks (outer index x band of rows along axis
//! ndim-2 x tile of columns). Each task copies the input region its windows
//! read, boundary-extended exactly like SciPy, into a small "slab" that stays
//! in cache, then runs a kernel on it. While copying, the task notes whether
//! the slab contains NaN or -0.0, so kernels can fall back to SciPy's exact
//! comparison order for that task only.

use super::boundary::axis_table;
use super::elem::RElem;
use super::nets::GenNet;
use super::par::{self, Arena, Arenas};
use super::network::{Net, CE, MAX, MIN};
use super::simd::Vx;
use crate::ndimage::common::Mode;

/// Raw pointer shared by worker threads that touch disjoint elements.
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// Slab bytes to aim for per task.
const SLAB_BYTES: usize = 96 * 1024;
/// Output columns per task, in bytes of element.
const TILE_BYTES: usize = 4096;
/// Extra elements after a slab so vector blocks may read past its end.
pub const SLACK: usize = 512;

/// Geometry of a filter call: output shape, window extent and the slab layout.
pub struct Plan {
    pub shape: Vec<usize>,
    /// window extent per axis
    pub span: Vec<usize>,
    /// padded coordinate -> input index, or -1 for the constant value
    tables: Vec<Vec<i64>>,
    /// output rows per task along axis ndim-2 (even), or 1 for 1-d
    pub band: usize,
    /// output columns per task
    pub tile: usize,
    /// slab dimensions and strides (elements)
    pub sdims: Vec<usize>,
    pub sstrides: Vec<usize>,
    /// whether any slab can contain the constant value
    pub uses_cval: bool,
}

impl Plan {
    pub fn new(shape: &[usize], before: &[usize], after: &[usize], mode: Mode, elem_size: usize) -> Plan {
        let nd = shape.len();
        let span: Vec<usize> = (0..nd).map(|i| before[i] + after[i] + 1).collect();
        let tables: Vec<Vec<i64>> = (0..nd).map(|i| axis_table(mode, shape[i], before[i], after[i])).collect();
        let uses_cval = tables.iter().any(|t| t.iter().any(|&v| v < 0));
        let nx = shape[nd - 1];
        let tile = (TILE_BYTES / elem_size).max(64).min(nx);
        let row_bytes = (tile + span[nd - 1] - 1) * elem_size;
        let (band, sdims) = if nd == 1 {
            (1, vec![tile + span[0] - 1])
        } else {
            let a = nd - 2;
            let outer_rows: usize = span[..a].iter().product();
            let rows_fit = SLAB_BYTES / (row_bytes * outer_rows).max(1);
            let mut band = rows_fit.saturating_sub(span[a] - 1).clamp(2, 64) & !1;
            band = band.min(shape[a].next_multiple_of(2)).max(2);
            // enough tasks to keep every thread busy on mid-sized images
            let others = shape[..a].iter().product::<usize>() * nx.div_ceil(tile);
            let want = 4 * crate::ndimage::common::num_threads();
            while band > 2 && others * shape[a].div_ceil(band) < want {
                band = (band / 2).max(2) & !1;
            }
            let mut sd: Vec<usize> = span[..a].to_vec();
            sd.push(band + span[a] - 1);
            sd.push(tile + span[nd - 1] - 1);
            (band, sd)
        };
        let mut sstrides = vec![1usize; nd];
        for i in (0..nd - 1).rev() {
            sstrides[i] = sstrides[i + 1] * sdims[i + 1];
        }
        Plan { shape: shape.to_vec(), span, tables, band, tile, sdims, sstrides, uses_cval }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Slab offset of a window element with per-axis coordinates `c` (0..span).
    pub fn offset(&self, c: &[usize]) -> isize {
        c.iter().zip(&self.sstrides).map(|(&a, &s)| (a * s) as isize).sum()
    }

    /// Slab offset between the windows of two outputs one row apart along
    /// axis ndim-2 (pairing), or None for 1-d.
    pub fn row_shift(&self) -> Option<isize> {
        let nd = self.ndim();
        (nd >= 2).then(|| self.sstrides[nd - 2] as isize)
    }

    fn slab_len(&self) -> usize {
        self.sdims.iter().product::<usize>() + SLACK
    }
}

/// One unit of work, as seen by a kernel.
pub struct Task {
    /// linear output index of the first pixel (row 0, column 0 of the task)
    pub out0: usize,
    /// output rows (along axis ndim-2) and their stride in the output
    pub rows: usize,
    pub out_row_stride: usize,
    /// slab stride between consecutive output rows
    pub slab_row_stride: usize,
    /// output columns
    pub cols: usize,
    /// the slab holds NaN or -0.0 (float types only)
    pub special: bool,
}

/// Build slabs for all tasks in parallel and call `kernel` on each.
/// `kernel(slab, task, scratch)` must only write the task's output pixels.
/// `work_per_px` (operations per output pixel) sizes the parallel chunks.
pub fn run_tasks<T, K>(plan: &Plan, src: &[T], cval: T, work_per_px: usize, kernel: K)
where
    T: RElem,
    K: Fn(&[T], &Task, &mut Arena) + Sync + Send,
{
    let nd = plan.ndim();
    let nx = plan.shape[nd - 1];
    let ntiles = nx.div_ceil(plan.tile);
    let (nouter, ny) = if nd == 1 {
        (1, 1)
    } else {
        (plan.shape[..nd - 2].iter().product::<usize>(), plan.shape[nd - 2])
    };
    let nbands = ny.div_ceil(plan.band);
    let ntasks = nouter * nbands * ntiles;
    let cval_special = T::is_float() && plan.uses_cval && cval.is_special();
    let slab_len = plan.slab_len();
    let task_work = par::work(plan.band * plan.tile, work_per_px, std::mem::size_of::<T>());
    par::for_each(ntasks, par::items_per_chunk(task_work), |t, ar| {
        let Arenas { a, b, .. } = ar;
        let slab = a.take::<T>(slab_len);
        let tile_i = t % ntiles;
        let band_i = (t / ntiles) % nbands;
        let o = t / (ntiles * nbands);
        let x0 = tile_i * plan.tile;
        let cols = plan.tile.min(nx - x0);
        let y0 = band_i * plan.band;
        let rows = if nd == 1 { 1 } else { plan.band.min(ny - y0) };
        let special = fill_slab(plan, src, cval, slab, o, y0, rows, x0, cols) || cval_special;
        let out0 = (o * ny + y0) * nx + x0;
        let task = Task {
            out0,
            rows,
            out_row_stride: nx,
            slab_row_stride: if nd == 1 { 0 } else { plan.sstrides[nd - 2] },
            cols,
            special,
        };
        kernel(slab, &task, b);
    });
}

/// Copy the boundary-extended input region of a task into `slab`.
/// Returns true if a copied input value is NaN or -0.0.
#[allow(clippy::too_many_arguments)]
fn fill_slab<T: RElem>(
    plan: &Plan,
    src: &[T],
    cval: T,
    slab: &mut [T],
    o: usize,
    y0: usize,
    rows: usize,
    x0: usize,
    cols: usize,
) -> bool {
    let nd = plan.ndim();
    let nx = plan.shape[nd - 1];
    let xt = &plan.tables[nd - 1];
    let row_len = cols + plan.span[nd - 1] - 1; // used part of a slab row
    // input strides
    let mut istr = vec![1usize; nd];
    for i in (0..nd - 1).rev() {
        istr[i] = istr[i + 1] * plan.shape[i + 1];
    }
    // output coordinates of the task origin along the leading axes
    let mut oc = vec![0usize; nd.saturating_sub(1)];
    if nd >= 2 {
        let mut r = o;
        for i in (0..nd - 2).rev() {
            oc[i] = r % plan.shape[i];
            r /= plan.shape[i];
        }
        oc[nd - 2] = y0;
    }
    // number of slab rows used: leading spans x (rows + span - 1)
    let lead_dims: Vec<usize> = if nd == 1 {
        vec![]
    } else {
        let mut d = plan.span[..nd - 2].to_vec();
        d.push(rows + plan.span[nd - 2] - 1);
        d
    };
    let nrows: usize = lead_dims.iter().product();
    // segments of the slab row: (dst start, src start or -1 for cval, len)
    let tab = &xt[x0..x0 + row_len];
    let mut segs: Vec<(usize, i64, usize)> = Vec::new();
    let mut q = 0;
    while q < row_len {
        let s = tab[q];
        let mut e = q + 1;
        if s < 0 {
            while e < row_len && tab[e] < 0 {
                e += 1;
            }
        } else {
            while e < row_len && tab[e] == s + (e - q) as i64 {
                e += 1;
            }
        }
        segs.push((q, s, e - q));
        q = e;
    }
    let mut special = false;
    let mut rc = vec![0usize; lead_dims.len()];
    for r in 0..nrows {
        // decode slab row coordinates
        let mut rem = r;
        for i in (0..lead_dims.len()).rev() {
            rc[i] = rem % lead_dims[i];
            rem /= lead_dims[i];
        }
        let mut soff = 0usize;
        let mut ioff = 0usize;
        let mut outside = false;
        for i in 0..lead_dims.len() {
            soff += rc[i] * plan.sstrides[i];
            let s = plan.tables[i][oc[i] + rc[i]];
            if s < 0 {
                outside = true;
            } else {
                ioff += s as usize * istr[i];
            }
        }
        let dst = &mut slab[soff..soff + row_len];
        if outside {
            dst.fill(cval);
            continue;
        }
        let line = &src[ioff..ioff + nx];
        for &(d, s, n) in &segs {
            if s < 0 {
                dst[d..d + n].fill(cval);
            } else {
                let chunk = &line[s as usize..s as usize + n];
                dst[d..d + n].copy_from_slice(chunk);
                if T::is_float() && !special {
                    special = any_special(chunk);
                }
            }
        }
    }
    special
}

#[inline]
fn any_special<T: RElem>(v: &[T]) -> bool {
    let mut sp = false;
    for c in v.chunks(64) {
        for &x in c {
            sp |= x.is_special();
        }
        if sp {
            return true;
        }
    }
    sp
}

// ---------------------------------------------------------------------------
// Interpreted networks

/// Slot block: `U` vectors (U * V::LANES consecutive output pixels).
#[derive(Clone, Copy)]
pub struct Blk<V, const U: usize>(pub [V; U]);

#[inline(always)]
fn bmin<T, V: Vx<T>, const U: usize>(x: &Blk<V, U>, y: &Blk<V, U>) -> Blk<V, U> {
    let mut r = x.0;
    for i in 0..U {
        r[i] = x.0[i].mn(y.0[i]);
    }
    Blk(r)
}

#[inline(always)]
fn bmax<T, V: Vx<T>, const U: usize>(x: &Blk<V, U>, y: &Blk<V, U>) -> Blk<V, U> {
    let mut r = x.0;
    for i in 0..U {
        r[i] = x.0[i].mx(y.0[i]);
    }
    Blk(r)
}

/// Run the ops of a network on the slot blocks.
#[inline(always)]
unsafe fn exec_ops<T, V: Vx<T>, const U: usize>(net: &Net, s: *mut Blk<V, U>) {
    for op in &net.ops {
        unsafe {
            let pa = s.add(op.a as usize);
            let pb = s.add(op.b as usize);
            match op.kind {
                CE => {
                    let x = *pa;
                    let y = *pb;
                    *pa = bmin(&x, &y);
                    *pb = bmax(&x, &y);
                }
                MIN => *pa = bmin(&*pa, &*pb),
                MAX => *pb = bmax(&*pa, &*pb),
                _ => *pb = *pa,
            }
        }
    }
}

#[inline(always)]
unsafe fn load_blk<T, V: Vx<T>, const U: usize>(p: *const T) -> Blk<V, U> {
    unsafe {
        let mut r: [std::mem::MaybeUninit<V>; U] = std::mem::MaybeUninit::uninit().assume_init();
        for (i, v) in r.iter_mut().enumerate() {
            v.write(V::load(p.add(i * V::LANES)));
        }
        Blk(std::mem::transmute_copy(&r))
    }
}

/// Evaluate a network over the pixels of a task: `pair` computes two output
/// rows at once, `single` one. Network input offsets are slab offsets.
///
/// # Safety
/// `out` must be valid for the task's output pixels.
pub unsafe fn net_task<T: RElem, V: Vx<T>, const U: usize>(
    ar: &mut Arena,
    slab: &[T],
    t: &Task,
    single: &Net,
    pair: Option<&Net>,
    out: SendPtr<T>,
) {
    let need = single.nslots.max(pair.map_or(0, |p| p.nslots));
    let s = ar.take::<Blk<V, U>>(need);
    let lanes = U * V::LANES;
    let mut tail = [T::default(); 512];
    let sp = s.as_mut_ptr();
    let mut y = 0;
    while y < t.rows {
        let (net, nrows) = match pair {
            Some(p) if y + 1 < t.rows => (p, 2),
            _ => (single, 1),
        };
        let base = y * t.slab_row_stride;
        let mut x0 = 0;
        while x0 < t.cols {
            unsafe {
                for &(slot, off) in &net.inputs {
                    let src = (base as isize + off + x0 as isize) as usize;
                    debug_assert!(src + lanes <= slab.len());
                    *sp.add(slot as usize) = load_blk::<T, V, U>(slab.as_ptr().add(src));
                }
                exec_ops::<T, V, U>(net, sp);
                let n = lanes.min(t.cols - x0);
                for (j, &o) in net.outputs.iter().enumerate() {
                    let dst = out.0.add(t.out0 + (y + j) * t.out_row_stride + x0);
                    let v = &*sp.add(o as usize);
                    if n == lanes {
                        for (i, v) in v.0.iter().enumerate() {
                            v.store(dst.add(i * V::LANES));
                        }
                    } else {
                        for (i, v) in v.0.iter().enumerate() {
                            v.store(tail.as_mut_ptr().add(i * V::LANES));
                        }
                        std::ptr::copy_nonoverlapping(tail.as_ptr(), dst, n);
                    }
                }
            }
            x0 += lanes;
        }
        y += nrows;
    }
}

/// Evaluate a generated network over the pixels of a task. `soff`/`poff`
/// are the slab offsets of the single / pair network inputs.
///
/// # Safety
/// `out` must be valid for the task's output pixels.
pub unsafe fn gen_task<T: RElem, V: Vx<T>, N: GenNet>(slab: &[T], t: &Task, soff: &[isize], poff: &[isize], r: usize, out: SendPtr<T>) {
    let lanes = V::LANES;
    let mut tail = [T::default(); 512];
    let ostride = t.out_row_stride as isize;
    let mut y = 0;
    while y < t.rows {
        let pair = y + 1 < t.rows;
        unsafe {
            let base = slab.as_ptr().add(y * t.slab_row_stride);
            let orow = out.0.add(t.out0 + y * t.out_row_stride);
            let mut x = 0;
            while x + lanes <= t.cols {
                if pair {
                    N::pair::<T, V>(base.add(x), poff, orow.add(x), ostride, r);
                } else {
                    N::single::<T, V>(base.add(x), soff, orow.add(x), ostride, r);
                }
                x += lanes;
            }
            if x < t.cols {
                let n = t.cols - x;
                let tp = tail.as_mut_ptr();
                if pair {
                    N::pair::<T, V>(base.add(x), poff, tp, lanes as isize, r);
                    std::ptr::copy_nonoverlapping(tp, orow.add(x), n);
                    std::ptr::copy_nonoverlapping(tp.add(lanes), orow.add(x).offset(ostride), n);
                } else {
                    N::single::<T, V>(base.add(x), soff, tp, lanes as isize, r);
                    std::ptr::copy_nonoverlapping(tp, orow.add(x), n);
                }
            }
        }
        y += if pair { 2 } else { 1 };
    }
}

/// Offsets for min/max filters: one output (`single`), or two outputs one
/// row apart sharing the reduction over `core`.
pub struct Reduce {
    single: Vec<isize>,
    pair: Option<(Vec<isize>, Vec<isize>, Vec<isize>)>,
}

impl Reduce {
    pub fn new(offsets: &[isize], shift: Option<isize>) -> Reduce {
        let pair = shift.map(|s| {
            let a: std::collections::HashSet<isize> = offsets.iter().copied().collect();
            let core: Vec<isize> = offsets.iter().copied().filter(|o| a.contains(&(o - s))).collect();
            let ea: Vec<isize> = offsets.iter().copied().filter(|o| !a.contains(&(o - s))).collect();
            let eb: Vec<isize> = offsets.iter().map(|o| o + s).filter(|o| !a.contains(o)).collect();
            (core, ea, eb)
        });
        Reduce { single: offsets.to_vec(), pair }
    }
}

#[inline(always)]
unsafe fn reduce_into<T, V: Vx<T>>(minimum: bool, acc: Option<V>, p: *const T, offs: &[isize]) -> Option<V> {
    let mut it = offs.iter();
    let mut acc = match acc {
        Some(a) => a,
        None => unsafe { V::load(p.offset(*it.next()?)) },
    };
    unsafe {
        if minimum {
            for &o in it {
                acc = acc.mn(V::load(p.offset(o)));
            }
        } else {
            for &o in it {
                acc = acc.mx(V::load(p.offset(o)));
            }
        }
    }
    Some(acc)
}

/// Min/max filter over the pixels of a task, accumulating in registers.
///
/// # Safety
/// `out` must be valid for the task's output pixels.
pub unsafe fn reduce_task<T: RElem, V: Vx<T>>(slab: &[T], t: &Task, r: &Reduce, minimum: bool, out: SendPtr<T>) {
    let lanes = V::LANES;
    let mut tail = [T::default(); 1024];
    let mut y = 0;
    while y < t.rows {
        let pair = r.pair.as_ref().filter(|_| y + 1 < t.rows);
        unsafe {
            let base = slab.as_ptr().add(y * t.slab_row_stride);
            let orow = out.0.add(t.out0 + y * t.out_row_stride);
            let mut x = 0;
            while x < t.cols {
                let p = base.add(x);
                let n = lanes.min(t.cols - x);
                let mut emit = |v: V, row: usize| {
                    let dst = orow.add(row * t.out_row_stride + x);
                    if n == lanes {
                        v.store(dst);
                    } else {
                        v.store(tail.as_mut_ptr());
                        std::ptr::copy_nonoverlapping(tail.as_ptr(), dst, n);
                    }
                };
                match pair {
                    Some((core, ea, eb)) => {
                        let c = reduce_into::<T, V>(minimum, None, p, core);
                        emit(reduce_into::<T, V>(minimum, c, p, ea).unwrap(), 0);
                        emit(reduce_into::<T, V>(minimum, c, p, eb).unwrap(), 1);
                    }
                    None => emit(reduce_into::<T, V>(minimum, None, p, &r.single).unwrap(), 0),
                }
                x += lanes;
            }
        }
        y += if pair.is_some() { 2 } else { 1 };
    }
}

/// Window offsets for sliding along the last axis: all elements, and those
/// leaving / entering when the window moves one step (relative to the old
/// window position).
pub struct Slide {
    all: Vec<isize>,
    leave: Vec<isize>,
    enter: Vec<isize>,
}

impl Slide {
    pub fn new(offsets: &[isize]) -> Slide {
        let set: std::collections::HashSet<isize> = offsets.iter().copied().collect();
        let leave = offsets.iter().copied().filter(|o| !set.contains(&(o - 1))).collect();
        let enter = offsets.iter().map(|o| o + 1).filter(|o| !set.contains(o)).collect();
        Slide { all: offsets.to_vec(), leave, enter }
    }
    /// Histogram updates per output pixel.
    pub fn edge(&self) -> usize {
        self.leave.len() + self.enter.len()
    }
}

/// Rank filter of 8-bit data with a sliding histogram (Huang): per pixel,
/// update the bins of the leaving / entering elements and the count of
/// elements below the current value `m`, then move `m` to the rank.
///
/// # Safety
/// `out` must be valid for the task's output pixels.
pub unsafe fn huang_task<T: RElem>(slab: &[T], t: &Task, s: &Slide, r: usize, out: SendPtr<T>) {
    let mut h = [0u32; 257];
    let r = r as u32;
    let maxo = s.all.iter().chain(&s.enter).copied().max().unwrap_or(0) as usize;
    for y in 0..t.rows {
        h.fill(0);
        let base = y * t.slab_row_stride;
        assert!(base + t.cols + maxo <= slab.len());
        let w = slab[base..].as_ptr();
        unsafe {
            for &o in &s.all {
                *h.get_unchecked_mut((*w.offset(o)).to_bin()) += 1;
            }
            // initial value: scan from the bottom
            let mut m = 0usize;
            let mut lt = 0u32; // elements below m
            while lt + h[m] <= r {
                lt += h[m];
                m += 1;
            }
            let orow = out.0.add(t.out0 + y * t.out_row_stride);
            for x in 0..t.cols {
                *orow.add(x) = T::from_bin(m);
                if x + 1 == t.cols {
                    break;
                }
                let wx = w.add(x);
                for &o in &s.leave {
                    let v = (*wx.offset(o)).to_bin();
                    *h.get_unchecked_mut(v) -= 1;
                    lt -= (v < m) as u32;
                }
                for &o in &s.enter {
                    let v = (*wx.offset(o)).to_bin();
                    *h.get_unchecked_mut(v) += 1;
                    lt += (v < m) as u32;
                }
                while lt > r {
                    m -= 1;
                    lt -= *h.get_unchecked(m);
                }
                while lt + *h.get_unchecked(m) <= r {
                    lt += *h.get_unchecked(m);
                    m += 1;
                }
            }
        }
    }
}

/// Evaluate `f` on the gathered window (in `offsets` order) of each pixel of
/// a task.
///
/// # Safety
/// `out` must be valid for the task's output pixels.
pub unsafe fn pixel_task<T: RElem, F>(ar: &mut Arena, slab: &[T], t: &Task, offsets: &[isize], out: SendPtr<T>, f: &F)
where
    F: Fn(&mut [T]) -> T,
{
    let buf = ar.take::<T>(offsets.len());
    for y in 0..t.rows {
        let base = y * t.slab_row_stride;
        for x in 0..t.cols {
            let b = (base + x) as isize;
            for (dst, &off) in buf.iter_mut().zip(offsets) {
                *dst = slab[(b + off) as usize];
            }
            unsafe { *out.0.add(t.out0 + y * t.out_row_stride + x) = f(buf) };
        }
    }
}

/// SciPy's `NI_Select` (Hoare-partition quickselect, pivot = first element),
/// reproduced exactly so results with NaN or signed zeros match.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn ni_select<T: PartialOrd + Copy>(buf: &mut [T], rank: usize) -> T {
    let mut min = 0isize;
    let mut max = buf.len() as isize - 1;
    let mut rank = rank as isize;
    loop {
        if min == max {
            return buf[min as usize];
        }
        let x = buf[min as usize];
        let mut ii = min - 1;
        let mut jj = max + 1;
        loop {
            loop {
                jj -= 1;
                if !(buf[jj as usize] > x) {
                    break;
                }
            }
            loop {
                ii += 1;
                if !(buf[ii as usize] < x) {
                    break;
                }
            }
            if ii < jj {
                buf.swap(ii as usize, jj as usize);
            } else {
                break;
            }
        }
        let cnt = jj - min + 1;
        if rank < cnt {
            max = jj;
        } else {
            min = jj + 1;
            rank -= cnt;
        }
    }
}

/// SciPy's sequential min/max over the window in footprint order.
#[inline]
pub fn seq_minmax<T: PartialOrd + Copy>(buf: &[T], minimum: bool) -> T {
    let mut res = buf[0];
    if minimum {
        for &v in &buf[1..] {
            if v < res {
                res = v;
            }
        }
    } else {
        for &v in &buf[1..] {
            if v > res {
                res = v;
            }
        }
    }
    res
}
