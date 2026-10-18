//! `binary_propagation` / `binary_fill_holes`: the fixed point of iterated masked dilation,
//! computed directly.
//!
//! The dilation step is out[q] = OR over d in D of in[q - d]. With 0 in D, iterating it under
//! a mask (pixels outside the mask keep their value) converges to the closure
//!   out = in  U  { q in mask : q reachable from an `in` pixel (or a border seed) through
//!                  mask pixels, one step being a member of D }.
//! Symmetric D within one pixel is done with run-based connected components on
//! V = in | mask (a component is filled when it holds an `in` pixel or a border seed). Any
//! other D uses a breadth-first flood fill.

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ndimage::common::pool;
use crate::ndimage::label::{connect, coords, extract_bytes, find_root, Block, Geo, Run, PAR_MIN};

/// Runs of V = in | mask on one line, plus the S runs of `in` for seeding.
fn extract_block_fill(
    g: &Geo, inp: &[u8], mask: Option<&[u8]>, start: usize, end: usize,
    seed_ends: bool, border_offs: &[Vec<i64>], border_value: bool,
) -> (Block, Vec<u8>) {
    let w = g.w;
    let mut runs: Vec<Run> = Vec::new();
    let mut seeds: Vec<u8> = Vec::new();
    let mut line_start = Vec::with_capacity(end - start + 1);
    let mut vbuf = vec![0u8; w];
    let mut sruns: Vec<Run> = Vec::new();
    let mut c = vec![0i64; g.dims.len()];
    for l in start..end {
        line_start.push(runs.len() as u32);
        let il = &inp[l * w..(l + 1) * w];
        let first = runs.len();
        sruns.clear();
        // Split lines into single-pixel runs when the structure has no in-line connection.
        let any_in = il.iter().any(|&v| v != 0);
        if any_in {
            extract_bytes(il, &mut sruns);
        }
        match mask {
            Some(m) => {
                let ml = &m[l * w..(l + 1) * w];
                for x in 0..w {
                    vbuf[x] = il[x] | ml[x];
                }
                extract_bytes(&vbuf, &mut runs);
            }
            None => {
                // everything passable
                runs.push([0, w as u32]);
            }
        }
        let n = runs.len() - first;
        seeds.resize(runs.len(), 0);
        // mark runs containing an `in` run
        let mut j = first;
        for s in &sruns {
            while runs[j][1] <= s[0] {
                j += 1;
            }
            seeds[j] = 1;
        }
        if border_value && n > 0 {
            coords(g, l, &mut c);
            let mut line_border = false;
            for o in border_offs {
                for d in 0..c.len() {
                    let v = c[d] + o[d];
                    if v < 0 || v >= g.dims[d] as i64 {
                        line_border = true;
                    }
                }
            }
            if line_border {
                for r in first..runs.len() {
                    seeds[r] = 1;
                }
            } else if seed_ends {
                if runs[first][0] == 0 {
                    seeds[first] = 1;
                }
                if runs[runs.len() - 1][1] as usize == w {
                    let k = runs.len() - 1;
                    seeds[k] = 1;
                }
            }
        }
    }
    line_start.push(runs.len() as u32);
    (Block { start, runs, line_start }, seeds)
}

fn fill_symmetric(
    shape: &[usize], st3: &[u8], inp: &[u8], mask: Option<&[u8]>, out: &mut [u8],
    border_value: bool, nthreads: usize,
) {
    let g = Geo::new(shape, st3);
    // fully connected in-line: required (checked by the caller)
    let n = inp.len();
    let nblocks = if nthreads <= 1 || n < PAR_MIN.load(std::sync::atomic::Ordering::Relaxed) { 1 } else { (nthreads * 4).min(g.nlines) };
    let bl = g.nlines.div_ceil(nblocks);
    let nblocks = g.nlines.div_ceil(bl);
    let bounds: Vec<(usize, usize)> = (0..nblocks).map(|b| (b * bl, ((b + 1) * bl).min(g.nlines))).collect();
    // lines that see the border: any non-zero other-dim offset present in the structure
    let nd = shape.len();
    let od = nd - 1;
    let mut border_offs: Vec<Vec<i64>> = Vec::new();
    let total_o = 3usize.pow(od as u32);
    let mut seed_ends = false;
    for code in 0..total_o {
        let mut o = vec![0i64; od];
        let mut cc = code;
        for d in (0..od).rev() {
            o[d] = (cc % 3) as i64 - 1;
            cc /= 3;
        }
        let base = code * 3;
        let (p, a, nx) = (st3[base] != 0, st3[base + 1] != 0, st3[base + 2] != 0);
        if o.iter().all(|&x| x == 0) {
            if p || nx { seed_ends = true; }
        } else if p || a || nx {
            border_offs.push(o);
            if p || nx { seed_ends = true; }
        }
    }
    pool().install(|| {
        let parts: Vec<(Block, Vec<u8>)> = bounds
            .par_iter()
            .map(|&(s, e)| extract_block_fill(&g, inp, mask, s, e, seed_ends, &border_offs, border_value))
            .collect();
        let mut blocks = Vec::with_capacity(nblocks);
        let mut seeds = Vec::with_capacity(nblocks);
        for (b, s) in parts {
            blocks.push(b);
            seeds.push(s);
        }
        let mut bases = Vec::with_capacity(nblocks);
        let mut total = 0usize;
        for b in &blocks {
            bases.push(total as u32);
            total += b.runs.len();
        }
        let mut parent = connect(&g, &blocks, &bounds, &bases, total);
        let mut seeded = vec![0u8; total];
        for (b, sd) in seeds.iter().enumerate() {
            for (i, &v) in sd.iter().enumerate() {
                if v != 0 {
                    let r = find_root(&mut parent, bases[b] + i as u32);
                    seeded[r as usize] = 1;
                }
            }
        }
        // resolve every run's root once (sequential, cheap)
        for r in 0..total as u32 {
            let root = find_root(&mut parent, r);
            parent[r as usize] = root;
        }
        let mut chunks: Vec<&mut [u8]> = Vec::with_capacity(nblocks);
        let mut rest = out;
        for &(s, e) in &bounds {
            let (h, t) = rest.split_at_mut((e - s) * g.w);
            chunks.push(h);
            rest = t;
        }
        let parent = &parent;
        let seeded = &seeded;
        chunks.par_iter_mut().zip(blocks.par_iter()).zip(bases.par_iter()).for_each(|((o, blk), &base)| {
            let nl = blk.line_start.len() - 1;
            for li in 0..nl {
                let ol = &mut o[li * g.w..(li + 1) * g.w];
                ol.fill(0);
                for r in blk.line_start[li] as usize..blk.line_start[li + 1] as usize {
                    let root = parent[base as usize + r] as usize;
                    if seeded[root] != 0 {
                        let [s, e] = blk.runs[r];
                        ol[s as usize..e as usize].fill(1);
                    }
                }
            }
        });
    });
}

/// Breadth-first closure for arbitrary offset sets D (0 in D).
fn fill_bfs(
    shape: &[usize], offs: &[Vec<i64>], inp: &[u8], mask: Option<&[u8]>, out: &mut [u8], border_value: bool,
) {
    let nd = shape.len();
    let n = inp.len();
    let mut stride = vec![1i64; nd];
    for d in (0..nd.saturating_sub(1)).rev() {
        stride[d] = stride[d + 1] * shape[d + 1] as i64;
    }
    let lin: Vec<i64> = offs.iter().map(|o| o.iter().zip(&stride).map(|(a, b)| a * b).sum()).collect();
    let mut queue: Vec<u64> = Vec::new();
    let mut c = vec![0i64; nd];
    for i in 0..n {
        out[i] = (inp[i] != 0) as u8;
    }
    // border seeds
    if border_value {
        for i in 0..n {
            if out[i] != 0 || mask.is_some_and(|m| m[i] == 0) {
                // seeded pixels still matter as sources only if they are in `in`
            }
        }
        for i in 0..n {
            let pass = mask.is_none_or(|m| m[i] != 0);
            if !pass || out[i] != 0 {
                continue;
            }
            let mut r = i as i64;
            for d in 0..nd {
                c[d] = r / stride[d];
                r %= stride[d];
            }
            'o: for o in offs {
                for d in 0..nd {
                    let p = c[d] - o[d]; // source pixel q - d
                    if p < 0 || p >= shape[d] as i64 {
                        out[i] = 1;
                        queue.push(i as u64);
                        break 'o;
                    }
                }
            }
        }
    }
    for i in 0..n {
        if inp[i] != 0 {
            queue.push(i as u64);
        }
    }
    let mut head = 0usize;
    while head < queue.len() {
        let p = queue[head] as i64;
        head += 1;
        let mut r = p;
        for d in 0..nd {
            c[d] = r / stride[d];
            r %= stride[d];
        }
        for (k, o) in offs.iter().enumerate() {
            let mut ok = true;
            for d in 0..nd {
                let q = c[d] + o[d];
                if q < 0 || q >= shape[d] as i64 {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let q = (p + lin[k]) as usize;
            if out[q] == 0 && mask.is_none_or(|m| m[q] != 0) {
                out[q] = 1;
                queue.push(q as u64);
            }
        }
    }
}

/// fill_core(input_u8, mask_u8_or_None, offsets (list of tuples), border_value, output_u8) -> bool path_symmetric
#[pyfunction]
fn fill_core<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    mask: Option<&Bound<'py, PyUntypedArray>>,
    offsets: Vec<Vec<i64>>,
    border_value: bool,
    output: &Bound<'py, PyUntypedArray>,
    allow_symmetric: bool,
) -> PyResult<bool> {
    use crate::ndimage::common::{slice, slice_mut, typed};
    let shape: Vec<usize> = input.shape().to_vec();
    let nd = shape.len();
    let inp = typed::<u8>(input)?;
    let o = typed::<u8>(output)?;
    let m = match mask {
        Some(m) => Some(typed::<u8>(m)?),
        None => None,
    };
    let (inp_s, out_s) = unsafe { (slice(&inp), slice_mut(&o)) };
    let mask_s = m.as_ref().map(|m| unsafe { slice(m) });
    // symmetric and within one pixel?
    let set: std::collections::HashSet<Vec<i64>> = offsets.iter().cloned().collect();
    let sym = allow_symmetric
        && offsets.iter().all(|d| d.iter().all(|&x| x.abs() <= 1) && set.contains(&d.iter().map(|x| -x).collect::<Vec<_>>()))
        && shape[nd - 1] < (1 << 30);
    let nthreads = crate::ndimage::common::num_threads();
    if sym {
        let mut st3 = vec![0u8; 3usize.pow(nd as u32)];
        for d in &offsets {
            let mut idx = 0usize;
            for &x in d {
                idx = idx * 3 + (x + 1) as usize;
            }
            st3[idx] = 1;
        }
        // run-based path needs connected in-line neighbours (d_last = +-1 on the centre line)
        let cl = {
            let mut z = vec![0i64; nd];
            z[nd - 1] = 1;
            set.contains(&z)
        };
        if cl && inp_s.len() < (u32::MAX as usize) - 2 {
            py.detach(|| fill_symmetric(&shape, &st3, inp_s, mask_s, out_s, border_value, nthreads));
            return Ok(true);
        }
    }
    py.detach(|| fill_bfs(&shape, &offsets, inp_s, mask_s, out_s, border_value));
    Ok(false)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fill_core, m)?)?;
    Ok(())
}
