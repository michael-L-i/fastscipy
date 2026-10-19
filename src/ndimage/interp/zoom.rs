//! `NI_ZoomShift`: zoom and/or shift, where every output axis maps to one
//! input axis, so tap indices and weights are tabulated per axis.
//!
//! SciPy computes every output point as
//! `t = 0; t += ((v * w_0) * w_1) ... * w_last` over the taps in C order.
//! Here, for each output row, the leading
//! products `(v[c] * w_0) * ... * w_{r-2}` are tabulated per input column `c`
//! of the last axis; every output point then adds `table[c] * w_last` over
//! its taps in SciPy's order. That is the same sequence of roundings, so the
//! results are bit-identical.
//!
//! Work items cover `L` output rows: the tables of those rows are interleaved
//! (`table[c][lane]`), and since all rows share the last-axis taps, each tap
//! is one contiguous load of `L` lanes, a multiply by the tap weight and an
//! add, with no gathers.

use std::cell::RefCell;

use super::support::{
    OutElem, SendPtr, for_each_index, map_coordinate, spline_mode, spline_start, spline_weights,
};
use crate::ndimage::common::{Elem, Mode};

/// Output rows per work item (SIMD lanes).
const ROWS: usize = 4;
/// Target size of the interleaved product table, in doubles.
const TABLE_TARGET: usize = 12 * 1024;
/// Largest table allowed (elements); larger calls are declined.
const MAX_TABLE: usize = 1 << 22;

pub struct Params {
    pub order: usize,
    pub mode: Mode,
    pub cval: f64,
    pub npad: i64,
    pub grid_mode: bool,
}

/// Per-axis tables, indexed by output coordinate `k` and tap `j`.
struct Axis {
    /// Constant mode: the coordinate is outside the input, output is cval.
    zero: Vec<bool>,
    /// Input index of each tap (already mapped at the borders).
    idx: Vec<u32>,
    /// grid-constant: the tap lies outside the input and reads cval.
    cv: Vec<bool>,
    /// Spline weights (order > 0).
    w: Vec<f64>,
}

fn build_axis(
    out_len: usize,
    in_len: usize,
    zoom: Option<f64>,
    shift: Option<f64>,
    p: &Params,
) -> Option<Axis> {
    let order = p.order;
    let taps = order + 1;
    let len = in_len as i64;
    let smode = spline_mode(p.mode);
    let mut ax = Axis {
        zero: vec![false; out_len],
        idx: vec![0; out_len * taps],
        cv: vec![false; out_len * taps],
        w: vec![0.0; out_len * taps],
    };
    let mapped = p.mode != Mode::GridConstant && p.mode != Mode::Nearest;
    for k in 0..out_len {
        let mut cc = k as f64;
        if let Some(s) = shift {
            cc += s;
        }
        if let Some(z) = zoom {
            if p.grid_mode {
                cc += 0.5;
                cc *= z;
                cc -= 0.5;
            } else {
                cc *= z;
            }
        }
        cc += p.npad as f64;
        if mapped {
            cc = map_coordinate(cc, len, p.mode);
        }
        if cc > -1.0 || !mapped {
            // Huge coordinates overflow SciPy's integer arithmetic.
            if cc.abs() > 1e15 {
                return None;
            }
            let start = spline_start(cc, order);
            let edge = start < 0 || start + order as i64 >= len;
            for j in 0..taps {
                let i = start + j as i64;
                let slot = k * taps + j;
                if !edge {
                    ax.idx[slot] = i as u32;
                } else if p.mode == Mode::GridConstant {
                    if i < 0 || i >= len {
                        ax.cv[slot] = true;
                    } else {
                        ax.idx[slot] = i as u32;
                    }
                } else {
                    let m = map_coordinate(i as f64, len, smode) as i64;
                    if m < 0 || m >= len {
                        return None;
                    }
                    ax.idx[slot] = m as u32;
                }
            }
            if order > 0 {
                let mut w = [0.0; 6];
                spline_weights(cc, order, &mut w);
                ax.w[k * taps..k * taps + taps].copy_from_slice(&w[..taps]);
            }
        } else if p.mode == Mode::Constant {
            ax.zero[k] = true;
        } else {
            // SciPy dereferences a NULL table here.
            return None;
        }
    }
    Some(ax)
}

/// Last-axis taps for one block of output points `x0..x0 + nx`.
struct LastBlock {
    x0: usize,
    nx: usize,
    /// First input column read.
    lo: usize,
    /// Columns read; table index `span` holds the cval product.
    span: usize,
    /// `rel[x * taps + j]`: table column of tap `j` of point `x0 + x`.
    rel: Vec<u32>,
    /// `w[x * taps + j]`: its weight.
    w: Vec<f64>,
    /// Points that are cval (constant mode, outside the input).
    zeros: Vec<u32>,
}

fn last_block(ax: &Axis, x0: usize, nx: usize, taps: usize) -> LastBlock {
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    for x in x0..x0 + nx {
        if ax.zero[x] {
            continue;
        }
        for j in 0..taps {
            if !ax.cv[x * taps + j] {
                lo = lo.min(ax.idx[x * taps + j]);
                hi = hi.max(ax.idx[x * taps + j]);
            }
        }
    }
    if lo > hi {
        (lo, hi) = (0, 0);
    }
    let span = (hi - lo + 1) as usize;
    let mut b = LastBlock {
        x0,
        nx,
        lo: lo as usize,
        span,
        rel: vec![span as u32; nx * taps],
        w: vec![0.0; nx * taps],
        zeros: Vec::new(),
    };
    for xx in 0..nx {
        let x = x0 + xx;
        if ax.zero[x] {
            b.zeros.push(xx as u32);
            continue;
        }
        for j in 0..taps {
            let slot = x * taps + j;
            if !ax.cv[slot] {
                b.rel[xx * taps + j] = ax.idx[slot] - lo;
            }
            b.w[xx * taps + j] = ax.w[slot];
        }
    }
    b
}

#[derive(Default)]
struct Scratch {
    table: Vec<f64>,
    t: Vec<f64>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Returns false (without writing) when the call is outside what is supported.
pub fn zoom_shift<I: Elem, O: OutElem>(
    src: &[I],
    in_shape: &[usize],
    dst: &mut [O],
    out_shape: &[usize],
    zooms: Option<&[f64]>,
    shifts: Option<&[f64]>,
    p: &Params,
    par: bool,
) -> bool {
    let rank = in_shape.len();
    if rank == 0 || rank > 64 || out_shape.len() != rank || p.order > 5 {
        return false;
    }
    if in_shape.iter().any(|&n| n == 0 || n > i32::MAX as usize) || dst.is_empty() {
        return false;
    }
    let mut axes = Vec::with_capacity(rank);
    for d in 0..rank {
        match build_axis(
            out_shape[d],
            in_shape[d],
            zooms.map(|z| z[d]),
            shifts.map(|s| s[d]),
            p,
        ) {
            Some(a) => axes.push(a),
            None => return false,
        }
    }
    let taps = p.order + 1;
    let combos = taps.pow(rank as u32 - 1);
    let in_last = in_shape[rank - 1];
    let out_last = out_shape[rank - 1];
    let nrows = dst.len() / out_last;
    let lanes = if nrows >= ROWS { ROWS } else { 1 };
    if combos.saturating_mul(in_last + 1).saturating_mul(lanes) > MAX_TABLE {
        return false;
    }
    let mut strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        strides[d] = strides[d + 1] * in_shape[d + 1];
    }

    // Split the last axis so that a work item's table stays cache-resident.
    let cols = (TABLE_TARGET / (combos * lanes)).max(16);
    let per_col = out_last as f64 / in_last as f64;
    let block = ((cols as f64 * per_col) as usize)
        .clamp(32, 4096)
        .min(out_last);
    let blocks: Vec<LastBlock> = (0..out_last.div_ceil(block))
        .map(|b| {
            let x0 = b * block;
            last_block(&axes[rank - 1], x0, block.min(out_last - x0), taps)
        })
        .collect();

    let ctx = Ctx {
        src,
        out_shape,
        strides: &strides,
        axes: &axes,
        p,
        combos,
        nrows,
    };
    let groups = nrows.div_ceil(lanes);
    let nb = blocks.len();
    let out = SendPtr(dst.as_mut_ptr());
    for_each_index(groups * nb, par, |item| {
        let (g, blk) = (item / nb, &blocks[item % nb]);
        SCRATCH.with(|s| {
            let s = &mut *s.borrow_mut();
            // SAFETY: work items write disjoint (row, block) ranges of `dst`.
            unsafe {
                if lanes == ROWS {
                    ctx.item::<O, ROWS>(g, blk, s, out)
                } else {
                    ctx.item::<O, 1>(g, blk, s, out)
                }
            }
        })
    });
    true
}

struct Ctx<'a, I> {
    src: &'a [I],
    out_shape: &'a [usize],
    strides: &'a [usize],
    axes: &'a [Axis],
    p: &'a Params,
    combos: usize,
    nrows: usize,
}

impl<I: Elem> Ctx<'_, I> {
    /// Rows `g * L ..` (lanes past the last row repeat it and are discarded),
    /// points of block `blk`.
    unsafe fn item<O: OutElem, const L: usize>(
        &self,
        g: usize,
        blk: &LastBlock,
        s: &mut Scratch,
        out: SendPtr<O>,
    ) {
        let rank = self.out_shape.len();
        let lead = rank - 1;
        let order = self.p.order;
        let taps = order + 1;
        let cval = self.p.cval;
        let s1 = blk.span + 1;
        let nx = blk.nx;
        let rows: [usize; L] = std::array::from_fn(|l| (g * L + l).min(self.nrows - 1));
        let nvalid = L.min(self.nrows - g * L);

        // Output coordinates of each row along the leading axes. Rows outside
        // the input in constant mode are cval; their lanes are computed from
        // valid (zero) table entries and discarded.
        let mut k = [[0usize; 64]; L];
        let mut zero_row = [false; L];
        for l in 0..L {
            let mut rem = rows[l];
            for d in (0..lead).rev() {
                k[l][d] = rem % self.out_shape[d];
                rem /= self.out_shape[d];
            }
            zero_row[l] = (0..lead).any(|d| self.axes[d].zero[k[l][d]]);
        }

        let need = self.combos * s1 * L;
        if s.table.len() < need {
            s.table.resize(need, 0.0);
        }
        if s.t.len() < nx * L {
            s.t.resize(nx * L, 0.0);
        }
        if zero_row[..nvalid].iter().any(|&z| !z) {
            self.fill_table::<L>(&k, blk, &mut s.table[..need]);
            let (table, t) = (&s.table[..need], &mut s.t[..nx * L]);
            let (rel, w, c) = (&blk.rel, &blk.w, self.combos);
            match order {
                0 => sum_taps::<0, L>(table, c, s1, rel, w, t),
                1 => sum_taps::<1, L>(table, c, s1, rel, w, t),
                2 => sum_taps::<2, L>(table, c, s1, rel, w, t),
                3 => sum_taps::<3, L>(table, c, s1, rel, w, t),
                4 => sum_taps::<4, L>(table, c, s1, rel, w, t),
                _ => sum_taps::<5, L>(table, c, s1, rel, w, t),
            }
        }
        let _ = taps;

        let out_last = self.out_shape[rank - 1];
        let cv = O::from_interp(cval);
        for l in 0..nvalid {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(out.get().add(rows[l] * out_last + blk.x0), nx)
            };
            if zero_row[l] {
                dst.fill(cv);
                continue;
            }
            let t = &s.t[l * nx..(l + 1) * nx];
            for (d, &v) in dst.iter_mut().zip(t) {
                *d = O::from_interp(v);
            }
            for &x in &blk.zeros {
                dst[x as usize] = cv;
            }
        }
    }

    /// `table[o][c][lane] = (v[c] * w_0) * ... * w_{r-2}` for every
    /// combination `o` of leading taps (C order) and column `c` of the block,
    /// and the same product for cval at column `span`.
    fn fill_table<const L: usize>(&self, k: &[[usize; 64]; L], blk: &LastBlock, table: &mut [f64]) {
        let lead = self.out_shape.len() - 1;
        let taps = self.p.order + 1;
        let weighted = self.p.order > 0;
        let span = blk.span;
        let s1 = span + 1;
        let mut f = [0usize; 64];
        for o in 0..self.combos {
            let mut off = [blk.lo; L];
            let mut is_cval = [false; L];
            let mut ws = [[0.0f64; L]; 64];
            for l in 0..L {
                for d in 0..lead {
                    let slot = k[l][d] * taps + f[d];
                    let a = &self.axes[d];
                    is_cval[l] |= a.cv[slot];
                    off[l] += a.idx[slot] as usize * self.strides[d];
                    ws[d][l] = a.w[slot];
                }
            }
            let tab = &mut table[o * s1 * L..(o + 1) * s1 * L];
            let rows: [&[I]; L] = std::array::from_fn(|l| &self.src[off[l]..off[l] + span]);
            let (cells, last) = tab.split_at_mut(span * L);
            if !weighted || lead == 0 {
                for (c, cell) in cells.as_chunks_mut::<L>().0.iter_mut().enumerate() {
                    for l in 0..L {
                        cell[l] = rows[l][c].to_f64();
                    }
                }
            } else if lead == 1 {
                let w0 = ws[0];
                for (c, cell) in cells.as_chunks_mut::<L>().0.iter_mut().enumerate() {
                    for l in 0..L {
                        cell[l] = rows[l][c].to_f64() * w0[l];
                    }
                }
            } else if lead == 2 {
                let (w0, w1) = (ws[0], ws[1]);
                for (c, cell) in cells.as_chunks_mut::<L>().0.iter_mut().enumerate() {
                    for l in 0..L {
                        cell[l] = rows[l][c].to_f64() * w0[l] * w1[l];
                    }
                }
            } else {
                for (c, cell) in cells.as_chunks_mut::<L>().0.iter_mut().enumerate() {
                    for l in 0..L {
                        let mut p = rows[l][c].to_f64();
                        for w in &ws[..lead] {
                            p *= w[l];
                        }
                        cell[l] = p;
                    }
                }
            }
            for l in 0..L {
                let mut pc = self.p.cval;
                if weighted {
                    for w in &ws[..lead] {
                        pc *= w[l];
                    }
                }
                last[l] = pc;
                if is_cval[l] {
                    for cell in cells.as_chunks_mut::<L>().0 {
                        cell[l] = pc;
                    }
                }
            }
            // Next combination of leading taps (last leading axis fastest).
            let mut d = lead;
            while d > 0 {
                d -= 1;
                f[d] += 1;
                if f[d] < taps {
                    break;
                }
                f[d] = 0;
            }
        }
    }
}

/// `t[lane][x] = sum over combos o, taps j of table[o][rel[x][j]][lane] * w[x][j]`,
/// accumulated from 0.0 in SciPy's order.
#[inline(never)]
fn sum_taps<const ORDER: usize, const L: usize>(
    table: &[f64],
    combos: usize,
    s1: usize,
    rel: &[u32],
    w: &[f64],
    t: &mut [f64],
) {
    let taps = ORDER + 1;
    let nx = t.len() / L;
    assert!(rel.len() >= nx * taps && w.len() >= nx * taps && table.len() >= combos * s1 * L);
    for x in 0..nx {
        // Tap columns and weights stay in registers across combinations.
        let r: [usize; 6] = std::array::from_fn(|j| {
            if j < taps {
                rel[x * taps + j] as usize * L
            } else {
                0
            }
        });
        let wx: [f64; 6] = std::array::from_fn(|j| if j < taps { w[x * taps + j] } else { 0.0 });
        let mut acc = [0.0f64; L];
        for o in 0..combos {
            let tab = &table[o * s1 * L..];
            for j in 0..taps {
                let c = r[j];
                // SAFETY: rel entries are <= span < s1.
                let v = unsafe { tab.get_unchecked(c..c + L) };
                for l in 0..L {
                    acc[l] += if ORDER > 0 { v[l] * wx[j] } else { v[l] };
                }
            }
        }
        for l in 0..L {
            t[l * nx + x] = acc[l];
        }
    }
}
