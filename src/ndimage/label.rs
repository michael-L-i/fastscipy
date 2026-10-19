//! `ndimage.label`: run-based two-scan connected components with SciPy's numbering.
//!
//! SciPy numbers components by the raster-scan position of their first pixel. Provisional
//! labels here are run indices in scan order, unions always keep the smaller root, so the
//! final numbering (rank of roots) is identical. Blocks of lines are labelled in parallel
//! and stitched across block boundaries.

use numpy::{PyArrayDyn, PyUntypedArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ndimage::common::{Elem, pool, slice, slice_mut, typed};

trait Nz: Copy + Send + Sync {
    fn nz(self) -> bool;
}
macro_rules! nz_int { ($($t:ty),*) => {$(impl Nz for $t { #[inline(always)] fn nz(self) -> bool { self != 0 } })*}; }
nz_int!(u8, i8, u16, i16, u32, i32, u64, i64);
impl Nz for f32 {
    #[inline(always)]
    fn nz(self) -> bool {
        self != 0.0
    }
}
impl Nz for f64 {
    #[inline(always)]
    fn nz(self) -> bool {
        self != 0.0
    }
}

pub(crate) type Run = [u32; 2];
pub(crate) static PAR_MIN: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(1 << 18);

#[pyfunction]
fn set_par_min(n: usize) {
    PAR_MIN.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) struct Nbr {
    pub(crate) off: usize,      // backwards distance in line index
    pub(crate) delta: Vec<i64>, // per non-last dim offset (-1..1)
    adj: bool,
    prev: bool,
    next: bool,
}

pub(crate) struct Geo {
    pub(crate) w: usize,
    pub(crate) nlines: usize,
    pub(crate) dims: Vec<usize>, // non-last dims
    pub(crate) use_prev: bool,
    pub(crate) nbrs: Vec<Nbr>,
}

impl Geo {
    pub(crate) fn new(shape: &[usize], st: &[u8]) -> Geo {
        let nd = shape.len();
        let w = shape[nd - 1];
        let dims: Vec<usize> = shape[..nd - 1].to_vec();
        let nlines: usize = dims.iter().product();
        // line-space strides
        let mut ls = vec![1usize; dims.len()];
        for d in (0..dims.len().saturating_sub(1)).rev() {
            ls[d] = ls[d + 1] * dims[d + 1];
        }
        let sidx = |o: &[i64], k: usize| -> usize {
            let mut idx = 0usize;
            for &c in o {
                idx = idx * 3 + (c + 1) as usize;
            }
            idx * 3 + k
        };
        let center: Vec<i64> = vec![0; dims.len()];
        let use_prev = st[sidx(&center, 0)] != 0;
        let mut nbrs = Vec::new();
        // enumerate offsets in C order; backward = lexicographically negative
        let total = 3usize.pow(dims.len() as u32);
        for code in 0..total {
            let mut o = vec![0i64; dims.len()];
            let mut c = code;
            for d in (0..dims.len()).rev() {
                o[d] = (c % 3) as i64 - 1;
                c /= 3;
            }
            let first = o.iter().find(|&&x| x != 0);
            if !matches!(first, Some(-1)) {
                continue;
            }
            let (p, a, n) = (
                st[sidx(&o, 0)] != 0,
                st[sidx(&o, 1)] != 0,
                st[sidx(&o, 2)] != 0,
            );
            if !(p || a || n) {
                continue;
            }
            let mut off = 0i64;
            for d in 0..dims.len() {
                off += o[d] * ls[d] as i64;
            }
            if off >= 0 {
                continue; // unrealisable when a size-1 dimension is involved
            }
            nbrs.push(Nbr {
                off: (-off) as usize,
                delta: o,
                adj: a,
                prev: p,
                next: n,
            });
        }
        Geo {
            w,
            nlines,
            dims,
            use_prev,
            nbrs,
        }
    }
}

pub(crate) struct Block {
    pub(crate) start: usize, // first line
    pub(crate) runs: Vec<Run>,
    pub(crate) line_start: Vec<u32>, // len = nl + 1
}

impl Block {
    #[inline(always)]
    pub(crate) fn line_runs(&self, line: usize) -> (&[Run], usize) {
        let i = line - self.start;
        let (a, b) = (self.line_start[i] as usize, self.line_start[i + 1] as usize);
        (&self.runs[a..b], a)
    }
}

#[inline(always)]
fn extract_scalar<T: Nz>(line: &[T], out: &mut Vec<Run>) {
    let n = line.len();
    let mut i = 0usize;
    while i < n {
        while i < n && !line[i].nz() {
            i += 1;
        }
        if i == n {
            break;
        }
        let s = i;
        while i < n && line[i].nz() {
            i += 1;
        }
        out.push([s as u32, i as u32]);
    }
}

/// Bit i set when byte i of the 64 at `p` is non-zero.
#[inline(always)]
unsafe fn mask64(p: *const u8) -> u64 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        const BM: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bm = vld1q_u8(BM.as_ptr());
        let z = vdupq_n_u8(0);
        let c0 = vandq_u8(vmvnq_u8(vceqq_u8(vld1q_u8(p), z)), bm);
        let c1 = vandq_u8(vmvnq_u8(vceqq_u8(vld1q_u8(p.add(16)), z)), bm);
        let c2 = vandq_u8(vmvnq_u8(vceqq_u8(vld1q_u8(p.add(32)), z)), bm);
        let c3 = vandq_u8(vmvnq_u8(vceqq_u8(vld1q_u8(p.add(48)), z)), bm);
        let s0 = vpaddq_u8(c0, c1);
        let s1 = vpaddq_u8(c2, c3);
        let s = vpaddq_u8(s0, s1);
        let s = vpaddq_u8(s, s);
        vgetq_lane_u64(vreinterpretq_u64_u8(s), 0)
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        let mut m = 0u64;
        for i in 0..64 {
            m |= ((*p.add(i) != 0) as u64) << i;
        }
        m
    }
}

/// Runs of non-zero bytes via 64-byte bitmasks and edge extraction.
pub(crate) fn extract_bytes(line: &[u8], out: &mut Vec<Run>) {
    let n = line.len();
    let mut start = 0usize;
    let mut carry = 0u64; // previous byte non-zero
    let mut x = 0usize;
    while x < n {
        let (m, valid) = if x + 64 <= n {
            (unsafe { mask64(line.as_ptr().add(x)) }, 64usize)
        } else {
            let mut buf = [0u8; 64];
            buf[..n - x].copy_from_slice(&line[x..]);
            (unsafe { mask64(buf.as_ptr()) }, n - x)
        };
        let mut e = m ^ ((m << 1) | carry);
        if valid < 64 {
            e &= (1u64 << valid) - 1;
        }
        let mut inrun = carry != 0;
        while e != 0 {
            let p = e.trailing_zeros() as usize;
            e &= e - 1;
            if inrun {
                out.push([start as u32, (x + p) as u32]);
            } else {
                start = x + p;
            }
            inrun = !inrun;
        }
        carry = (m >> (valid - 1)) & 1;
        x += 64;
    }
    if carry != 0 {
        out.push([start as u32, n as u32]);
    }
}

#[inline(always)]
fn extract_line<T: Nz>(line: &[T], split: bool, out: &mut Vec<Run>) {
    if split {
        for (x, &v) in line.iter().enumerate() {
            if v.nz() {
                out.push([x as u32, x as u32 + 1]);
            }
        }
    } else if std::mem::size_of::<T>() == 1 {
        // u8 / i8: same non-zero test on the raw bytes
        let b = unsafe { std::slice::from_raw_parts(line.as_ptr() as *const u8, line.len()) };
        extract_bytes(b, out);
    } else {
        extract_scalar(line, out);
    }
}

fn extract_block<T: Nz>(g: &Geo, data: &[T], start: usize, end: usize) -> Block {
    let mut runs: Vec<Run> = Vec::new();
    let mut line_start = Vec::with_capacity(end - start + 1);
    for l in start..end {
        line_start.push(runs.len() as u32);
        extract_line(&data[l * g.w..(l + 1) * g.w], !g.use_prev, &mut runs);
    }
    line_start.push(runs.len() as u32);
    Block {
        start,
        runs,
        line_start,
    }
}

struct Uf<'a> {
    p: &'a mut [u32],
    base: u32,
}
impl<'a> Uf<'a> {
    #[inline(always)]
    fn find(&mut self, mut x: u32) -> u32 {
        loop {
            let px = self.p[(x - self.base) as usize];
            if px == x {
                return x;
            }
            let ppx = self.p[(px - self.base) as usize];
            self.p[(x - self.base) as usize] = ppx;
            x = ppx;
        }
    }
    #[inline(always)]
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra < rb {
            self.p[(rb - self.base) as usize] = ra;
        } else if rb < ra {
            self.p[(ra - self.base) as usize] = rb;
        }
    }
}

/// Union overlapping runs of `cur` and `nb`, where neighbour runs cover `[s - lo, e + hi)`.
#[inline(always)]
fn sweep(cur: &[Run], cb: u32, nb: &[Run], nbb: u32, lo: i64, hi: i64, uf: &mut Uf) {
    // expanded neighbour intervals can overlap each other (single-pixel runs), so scan forward
    // from the first interval that ends after the current run starts without consuming it
    let mut j = 0usize;
    for (i, r) in cur.iter().enumerate() {
        let (cs, ce) = (r[0] as i64, r[1] as i64);
        while j < nb.len() && nb[j][1] as i64 + hi <= cs {
            j += 1;
        }
        let mut k = j;
        while k < nb.len() && (nb[k][0] as i64 - lo) < ce {
            uf.union(cb + i as u32, nbb + k as u32);
            k += 1;
        }
    }
}

fn link(cur: &[Run], cb: u32, nb: &[Run], nbb: u32, n: &Nbr, uf: &mut Uf) {
    if cur.is_empty() || nb.is_empty() {
        return;
    }
    if n.adj {
        sweep(cur, cb, nb, nbb, n.next as i64, n.prev as i64, uf);
    } else {
        // neighbour pixel j connects current pixel j+1 (prev flag) or j-1 (next flag)
        if n.prev {
            sweep_shift(cur, cb, nb, nbb, 1, uf);
        }
        if n.next {
            sweep_shift(cur, cb, nb, nbb, -1, uf);
        }
    }
}
fn sweep_shift(cur: &[Run], cb: u32, nb: &[Run], nbb: u32, d: i64, uf: &mut Uf) {
    let (mut i, mut j) = (0usize, 0usize);
    while i < cur.len() && j < nb.len() {
        let (cs, ce) = (cur[i][0] as i64, cur[i][1] as i64);
        let (ns, ne) = (nb[j][0] as i64 + d, nb[j][1] as i64 + d);
        if ce <= ns {
            i += 1;
        } else if ne <= cs {
            j += 1;
        } else {
            uf.union(cb + i as u32, nbb + j as u32);
            if ce < ne { i += 1 } else { j += 1 }
        }
    }
}

pub(crate) fn coords(g: &Geo, line: usize, c: &mut [i64]) {
    let mut r = line;
    for d in (0..g.dims.len()).rev() {
        c[d] = (r % g.dims[d]) as i64;
        r /= g.dims[d];
    }
}

#[inline(always)]
fn nbr_line(g: &Geo, n: &Nbr, line: usize, c: &[i64]) -> Option<usize> {
    for d in 0..g.dims.len() {
        let v = c[d] + n.delta[d];
        if v < 0 || v >= g.dims[d] as i64 {
            return None;
        }
    }
    Some(line - n.off)
}

fn union_block(g: &Geo, blk: &Block, base: u32, parent: &mut [u32]) {
    let mut uf = Uf { p: parent, base };
    let nl = blk.line_start.len() - 1;
    let mut c = vec![0i64; g.dims.len()];
    for l in blk.start..blk.start + nl {
        let (cur, ca) = blk.line_runs(l);
        if cur.is_empty() {
            continue;
        }
        coords(g, l, &mut c);
        for n in &g.nbrs {
            if let Some(nl_) = nbr_line(g, n, l, &c)
                && nl_ >= blk.start
            {
                let (nb, na) = blk.line_runs(nl_);
                link(cur, base + ca as u32, nb, base + na as u32, n, &mut uf);
            }
        }
    }
}

fn write_block<T: Elem + FromU32>(g: &Geo, blk: &Block, base: u32, lab: &[u32], out: &mut [T]) {
    let nl = blk.line_start.len() - 1;
    let zero = T::from_u32(0);
    for li in 0..nl {
        let o = &mut out[li * g.w..(li + 1) * g.w];
        let (a, b) = (blk.line_start[li] as usize, blk.line_start[li + 1] as usize);
        let mut x = 0usize;
        for r in a..b {
            let [s, e] = blk.runs[r];
            let v = T::from_u32(lab[(base as usize) + r]);
            o[x..s as usize].fill(zero);
            o[s as usize..e as usize].fill(v);
            x = e as usize;
        }
        o[x..].fill(zero);
    }
}

pub trait FromU32 {
    fn from_u32(v: u32) -> Self;
    const MAXV: u64;
}
macro_rules! from_u32 { ($($t:ty),*) => {$(impl FromU32 for $t { #[inline(always)] fn from_u32(v: u32) -> Self { v as $t } const MAXV: u64 = <$t>::MAX as u64; })*}; }
from_u32!(u8, i8, u16, i16, u32, i32, u64, i64);

/// Union-find over all runs: unions inside blocks in parallel, then stitches block borders.
/// Returns the parent array (roots are the smallest run id of each component).
pub(crate) fn connect(
    g: &Geo,
    blocks: &[Block],
    bounds: &[(usize, usize)],
    bases: &[u32],
    total: usize,
) -> Vec<u32> {
    let nblocks = blocks.len();
    let mut parent = vec![0u32; total];
    {
        let mut slices: Vec<&mut [u32]> = Vec::with_capacity(nblocks);
        let mut rest = parent.as_mut_slice();
        for b in blocks {
            let (h, t) = rest.split_at_mut(b.runs.len());
            slices.push(h);
            rest = t;
        }
        slices
            .par_iter_mut()
            .zip(blocks.par_iter())
            .zip(bases.par_iter())
            .for_each(|((sl, blk), &base)| {
                for (i, p) in sl.iter_mut().enumerate() {
                    *p = base + i as u32;
                }
                union_block(g, blk, base, sl);
            });
    }
    if nblocks > 1 {
        let mut uf = Uf {
            p: &mut parent,
            base: 0,
        };
        let maxback = g.nbrs.iter().map(|n| n.off).max().unwrap_or(0);
        let mut c = vec![0i64; g.dims.len()];
        for b in 1..nblocks {
            let blk = &blocks[b];
            let (s, e) = bounds[b];
            for l in s..e.min(s + maxback) {
                let (cur, ca) = blk.line_runs(l);
                if cur.is_empty() {
                    continue;
                }
                coords(g, l, &mut c);
                for n in &g.nbrs {
                    if let Some(nl) = nbr_line(g, n, l, &c)
                        && nl < s
                    {
                        let nbk = bounds.partition_point(|&(_, be)| be <= nl);
                        let (nb, na) = blocks[nbk].line_runs(nl);
                        link(
                            cur,
                            bases[b] + ca as u32,
                            nb,
                            bases[nbk] + na as u32,
                            n,
                            &mut uf,
                        );
                    }
                }
            }
        }
    }
    parent
}

/// Plain path compression to roots, in place.
pub(crate) fn find_root(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        let g = p[p[x as usize] as usize];
        p[x as usize] = g;
        x = g;
    }
    x
}

fn label_impl<I: Nz + Elem, O: Elem + FromU32>(
    shape: &[usize],
    st: &[u8],
    data: &[I],
    out: &mut [O],
    nthreads: usize,
) -> Option<u64> {
    let g = Geo::new(shape, st);
    if g.w >= u32::MAX as usize / 2 || data.len() >= (u32::MAX as usize) - 2 {
        return None;
    }
    let nblocks =
        if nthreads <= 1 || data.len() < PAR_MIN.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            (nthreads * 4).min(g.nlines)
        };
    let bl = g.nlines.div_ceil(nblocks);
    let nblocks = g.nlines.div_ceil(bl);
    let bounds: Vec<(usize, usize)> = (0..nblocks)
        .map(|b| (b * bl, ((b + 1) * bl).min(g.nlines)))
        .collect();

    let run = || -> Option<u64> {
        let blocks: Vec<Block> = bounds
            .par_iter()
            .map(|&(s, e)| extract_block(&g, data, s, e))
            .collect();
        let mut bases = Vec::with_capacity(nblocks);
        let mut total = 0usize;
        for b in &blocks {
            bases.push(total as u32);
            total += b.runs.len();
        }
        let mut parent = connect(&g, &blocks, &bounds, &bases, total);
        // compact: roots numbered in scan order
        let mut next = 0u32;
        for r in 0..total {
            let p = parent[r];
            if p as usize == r {
                next += 1;
                parent[r] = next;
            } else {
                parent[r] = parent[p as usize];
            }
        }
        if (next as u64) > O::MAXV {
            return None;
        }
        let lab = &parent;
        let mut chunks: Vec<&mut [O]> = Vec::with_capacity(nblocks);
        let mut rest = out;
        for &(s, e) in &bounds {
            let (h, t) = rest.split_at_mut((e - s) * g.w);
            chunks.push(h);
            rest = t;
        }
        chunks
            .par_iter_mut()
            .zip(blocks.par_iter())
            .zip(bases.par_iter())
            .for_each(|((o, blk), &base)| {
                write_block(&g, blk, base, lab, o);
            });
        Some(next as u64)
    };
    pool().install(run)
}

/// label_core(input, structure_u8, output) -> max label, or None when declined.
#[pyfunction]
fn label_core<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    structure: Vec<u8>,
    output: &Bound<'py, PyUntypedArray>,
) -> PyResult<Option<u64>> {
    use numpy::PyUntypedArrayMethods;
    if !input.is_c_contiguous() || !output.is_c_contiguous() {
        return Ok(None);
    }
    let shape: Vec<usize> = input.shape().to_vec();
    let nthreads = crate::ndimage::common::num_threads();
    let od = output.dtype();
    macro_rules! with_out {
        ($I:ty, $inp:expr) => {{
            let inp = typed::<$I>($inp)?;
            let data = unsafe { slice(&inp) };
            macro_rules! go {
                ($O:ty) => {{
                    let o = typed::<$O>(output)?;
                    let os = unsafe { slice_mut(&o) };
                    let st = &structure;
                    let sh = &shape;
                    return Ok(py.detach(|| label_impl::<$I, $O>(sh, st, data, os, nthreads)));
                }};
            }
            if od.is_equiv_to(&numpy::dtype::<i32>(py)) {
                go!(i32)
            } else if od.is_equiv_to(&numpy::dtype::<u32>(py)) {
                go!(u32)
            } else if od.is_equiv_to(&numpy::dtype::<i64>(py)) {
                go!(i64)
            } else if od.is_equiv_to(&numpy::dtype::<u64>(py)) {
                go!(u64)
            } else if od.is_equiv_to(&numpy::dtype::<i16>(py)) {
                go!(i16)
            } else if od.is_equiv_to(&numpy::dtype::<u16>(py)) {
                go!(u16)
            } else if od.is_equiv_to(&numpy::dtype::<i8>(py)) {
                go!(i8)
            } else if od.is_equiv_to(&numpy::dtype::<u8>(py)) {
                go!(u8)
            } else {
                return Ok(None);
            }
        }};
    }
    crate::dispatch_elem!(py, input.dtype(), T => { with_out!(T, input) } else Ok(None))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(label_core, m)?)?;
    m.add_function(wrap_pyfunction!(set_par_min, m)?)?;
    Ok(())
}

#[allow(dead_code)]
fn _unused(_: &PyArrayDyn<u8>) {}
