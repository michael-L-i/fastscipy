//! N-D `correlate` (SciPy's `NI_Correlate`), used by `correlate` and `convolve`.
//!
//! Each output is `0.0 + w0*v0 + w1*v1 + ...` over the weights with
//! `|w| > DBL_EPSILON` in C order, where `v` is the input at the mapped
//! position (`NI_InitFilterOffsets`) or cval. Outputs are computed a row
//! (last axis) at a time with one SIMD lane per column: for every distinct
//! leading-axes offset of the kernel ("group"), the needed input row is
//! converted to double and extended along the last axis once, and kept in a
//! small cache while consecutive output rows reuse it.

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;

use super::cast::{self, LoadFn, Raw, StoreFn, for_each_task};
use super::ext::nd_map;
use super::kern::{self, Tap};
use crate::ndimage::common::Mode;

struct Job {
    src: Raw,
    dst: Raw,
    load: LoadFn,
    store: StoreFn,
    /// Leading (all but last) dimensions of the array.
    lead: Vec<usize>,
    /// Length of the last axis.
    n: usize,
    /// Kernel length along the last axis.
    k: usize,
    /// For each group, the kernel offset along the leading axes minus origin.
    groups: Vec<Vec<isize>>,
    taps: Vec<Tap>,
    /// Extended position of column 0 (kernel origin along the last axis).
    orgn: usize,
    mode: Mode,
    cval: f64,
    fused: bool,
    /// For 2-D kernels of width <= ND_PAIR_MAX_M with every weight kept: the
    /// weights, for the two-rows-at-a-time kernel.
    dense: Option<Vec<f64>>,
}

impl Job {
    fn rows(&self) -> usize {
        self.lead.iter().product()
    }

    /// Source row index (linear over the leading axes) for output row `i`
    /// (given as a multi-index) and group `g`, or None for a cval row.
    fn source_row(&self, idx: &[usize], g: usize) -> Option<usize> {
        let mut row = 0usize;
        for (d, (&i, &off)) in idx.iter().zip(&self.groups[g]).enumerate() {
            let m = nd_map(i as isize + off, self.lead[d] as isize, self.mode)?;
            row = row * self.lead[d] + m as usize;
        }
        Some(row)
    }

    /// Fill `ext[q - j0]` for q in j0..j0 + cnt from source row `row`.
    unsafe fn fill(&self, row: Option<usize>, j0: usize, cnt: usize, ext: &mut [f64]) {
        let Some(row) = row else {
            ext[..cnt].fill(self.cval);
            return;
        };
        let base = row * self.n;
        let end = j0 + cnt;
        // Positions orgn..orgn+n map to columns 0..n; the rest go through the map.
        let (lo, hi) = (
            self.orgn.max(j0).min(end),
            (self.orgn + self.n).max(j0).min(end),
        );
        let one = |q: usize, ext: &mut [f64]| {
            // Source column (or None for cval) of extended position q.
            let c = nd_map(q as isize - self.orgn as isize, self.n as isize, self.mode);
            ext[q - j0] = match c {
                None => self.cval,
                Some(c) => {
                    let mut v = 0.0;
                    unsafe { (self.load)(self.src.at(base + c as usize), 1, &mut v) };
                    v
                }
            };
        };
        for q in j0..lo {
            one(q, ext);
        }
        if hi > lo {
            unsafe {
                (self.load)(
                    self.src.at(base + lo - self.orgn),
                    hi - lo,
                    ext.as_mut_ptr().add(lo - j0),
                )
            };
        }
        for q in hi.max(lo)..end {
            one(q, ext);
        }
    }
}

const EMPTY: isize = -2;

/// Bytes of cached extended rows per task.
const L1_BUDGET: usize = 48 * 1024;

struct Cache {
    keys: Vec<isize>,
    stamps: Vec<u64>,
    bufs: Vec<Vec<f64>>,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn corr_correlate(
    py: Python<'_>,
    input: &Bound<'_, PyUntypedArray>,
    weights: &Bound<'_, PyUntypedArray>,
    output: &Bound<'_, PyUntypedArray>,
    mode: i64,
    cval: f64,
    origins: Vec<i64>,
    fused: bool,
) -> PyResult<bool> {
    let (Some(src), Some(dst), Some(wr)) = (
        cast::raw(py, input, false),
        cast::raw(py, output, true),
        cast::raw(py, weights, false),
    ) else {
        return Ok(false);
    };
    let shape = input.shape().to_vec();
    let nd = shape.len();
    let fshape = weights.shape().to_vec();
    if nd == 0
        || output.shape() != shape.as_slice()
        || wr.dt != cast::DType::F64
        || fshape.len() != nd
        || origins.len() != nd
        || fshape.contains(&0)
        || shape.contains(&0)
        || src.overlaps(&dst)
    {
        return Ok(false);
    }
    let Some(mode) = Mode::from_code(mode) else {
        return Ok(false);
    };
    let mut orgn = vec![0isize; nd];
    for d in 0..nd {
        let f = fshape[d] as i64;
        if origins[d] < -(f / 2) || origins[d] > (f - 1) / 2 {
            return Ok(false);
        }
        orgn[d] = (f / 2 + origins[d]) as isize;
    }
    // SciPy's reflect mapping sends coordinates that are an even multiple
    // (>= 2) of twice the axis length before the start to -1, and then reads
    // outside the array; leave those calls to SciPy.
    for d in 0..nd {
        let len = shape[d] as isize;
        let lo = -orgn[d];
        let hi = len + fshape[d] as isize - 2 - orgn[d];
        let outside = (lo..0).chain(len..=hi);
        if outside
            .into_iter()
            .any(|cc| nd_map(cc, len, mode).is_some_and(|m| m < 0 || m >= len))
        {
            return Ok(false);
        }
    }
    let w = unsafe { std::slice::from_raw_parts(wr.ptr as *const f64, fshape.iter().product()) };
    let k = fshape[nd - 1];
    let n = shape[nd - 1];

    // Taps in C order, grouped by their leading-axes offset.
    let mut groups: Vec<Vec<isize>> = Vec::new();
    let mut taps = Vec::new();
    let lead_f = &fshape[..nd - 1];
    for (li, chunk) in w.chunks(k).enumerate() {
        let mut gi = None;
        for (b, &wv) in chunk.iter().enumerate() {
            if !(wv.abs() > f64::EPSILON) {
                continue;
            }
            let g = *gi.get_or_insert_with(|| {
                let mut off = vec![0isize; nd - 1];
                let mut rem = li;
                for d in (0..nd - 1).rev() {
                    off[d] = (rem % lead_f[d]) as isize - orgn[d];
                    rem /= lead_f[d];
                }
                groups.push(off);
                groups.len() - 1
            });
            taps.push(Tap {
                row: g,
                off: b,
                w: wv,
            });
        }
    }
    let dense = (nd == 2 && k <= kern::ND_PAIR_MAX_M && taps.len() == w.len() && fshape[0] <= 64)
        .then(|| w.to_vec());
    let job = Job {
        src,
        dst,
        load: cast::loader(src.dt),
        store: cast::store_safe(dst.dt),
        lead: shape[..nd - 1].to_vec(),
        n,
        k,
        groups,
        taps,
        orgn: orgn[nd - 1] as usize,
        mode,
        cval,
        fused,
        dense,
    };
    py.detach(|| run(&job));
    Ok(true)
}

fn run(job: &Job) {
    let rows = job.rows();
    let n = job.n;
    let threads = cast::threads_for(rows * n * (job.taps.len().min(64) + 4));
    let ng = job.groups.len();
    let cap = ng + 3;
    // Column chunks small enough for the cached rows to stay in L1.
    let fit = (L1_BUDGET / (8 * cap)).saturating_sub(job.k).max(256);
    let nchunks = n.div_ceil(fit);
    let chunk = n.div_ceil(nchunks);
    // Row blocks: enough tasks to balance threads, long enough to reuse rows.
    let per = (rows * nchunks / (4 * threads)).clamp(1, 256).min(rows);
    let nblocks = rows.div_ceil(per);
    for_each_task(
        nblocks * nchunks,
        threads,
        || {
            let cache = Cache {
                keys: vec![EMPTY; cap],
                stamps: vec![0; cap],
                bufs: (0..cap).map(|_| vec![0.0f64; chunk + job.k - 1]).collect(),
            };
            (
                cache,
                vec![std::ptr::null::<f64>(); ng + 1],
                vec![0.0f64; 2 * chunk],
                vec![0usize; job.lead.len()],
            )
        },
        |(cache, ptrs, out, idx), task| {
            let blk = task / nchunks;
            let j0 = (task % nchunks) * chunk;
            let cnt = chunk.min(n - j0);
            let ecnt = cnt + job.k - 1;
            cache.keys.fill(EMPTY);
            let r0 = blk * per;
            let r1 = (r0 + per).min(rows);
            // Extended row for source row `key` (-1: cval), kept while `stamp` is current.
            let mut get = |key: isize, stamp: u64| -> *const f64 {
                let slot = match cache.keys.iter().position(|&kk| kk == key) {
                    Some(s) => s,
                    None => {
                        // Evict the least recently used entry not used by this step.
                        let s = (0..cap)
                            .filter(|&s| cache.stamps[s] != stamp || cache.keys[s] == EMPTY)
                            .min_by_key(|&s| {
                                if cache.keys[s] == EMPTY {
                                    0
                                } else {
                                    cache.stamps[s]
                                }
                            })
                            .unwrap();
                        let r = if key < 0 { None } else { Some(key as usize) };
                        unsafe { job.fill(r, j0, ecnt, &mut cache.bufs[s]) };
                        cache.keys[s] = key;
                        s
                    }
                };
                cache.stamps[slot] = stamp;
                cache.bufs[slot].as_ptr()
            };
            let (out0, out1) = out.split_at_mut(chunk);
            let mut row = r0;
            while row < r1 {
                let mut rem = row;
                for d in (0..job.lead.len()).rev() {
                    idx[d] = rem % job.lead[d];
                    rem /= job.lead[d];
                }
                let stamp = row as u64 + 1;
                if let (Some(dense), true) = (&job.dense, row + 1 < r1) {
                    // Rows R_0..=R_k0: groups 0..k0 of this row, then group
                    // k0 - 1 of the next row.
                    for g in 0..ng {
                        ptrs[g] = get(job.source_row(idx, g).map_or(-1, |r| r as isize), stamp);
                    }
                    idx[0] += 1;
                    ptrs[ng] = get(
                        job.source_row(idx, ng - 1).map_or(-1, |r| r as isize),
                        stamp,
                    );
                    unsafe {
                        kern::nd_pair(
                            dense,
                            job.k,
                            &job.taps,
                            &ptrs[..=ng],
                            cnt,
                            job.fused,
                            out0.as_mut_ptr(),
                            out1.as_mut_ptr(),
                        );
                        (job.store)(out0.as_ptr(), cnt, job.dst.at(row * n + j0));
                        (job.store)(out1.as_ptr(), cnt, job.dst.at((row + 1) * n + j0));
                    }
                    row += 2;
                    continue;
                }
                for g in 0..ng {
                    ptrs[g] = get(job.source_row(idx, g).map_or(-1, |r| r as isize), stamp);
                }
                unsafe {
                    kern::nd_correlate(&job.taps, &ptrs[..ng], cnt, job.fused, out0.as_mut_ptr());
                    (job.store)(out0.as_ptr(), cnt, job.dst.at(row * n + j0));
                }
                row += 1;
            }
        },
    );
}
