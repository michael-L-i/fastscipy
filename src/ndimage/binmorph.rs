//! Port of SciPy's NI_BinaryErosion / NI_BinaryErosion2 (ni_morphology.c).
//!
//! NI_BinaryErosion is a pure point operation, so lines can be processed in any
//! order; NI_BinaryErosion2 propagates a list of changed pixels level by level
//! (the result set does not depend on the order within a level).

use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use std::sync::atomic::{AtomicU8, Ordering};

use crate::ndimage::common::{num_threads, pool};

struct Geom {
    shape: Vec<usize>,
    /// every true structure element: offset per axis (neighbour = x + off)
    elems: Vec<Vec<isize>>,
}

fn geom(shape: &[usize], sshape: &[usize], sdata: &[u8], origin: &[isize]) -> Geom {
    let n = shape.len();
    let mut elems = Vec::new();
    let total: usize = sshape.iter().product();
    let mut idx = vec![0usize; n];
    for flat in 0..total {
        if sdata[flat] != 0 {
            let off: Vec<isize> = (0..n)
                .map(|k| idx[k] as isize - (sshape[k] / 2) as isize - origin[k])
                .collect();
            elems.push(off);
        }
        for k in (0..n).rev() {
            idx[k] += 1;
            if idx[k] < sshape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    Geom {
        shape: shape.to_vec(),
        elems,
    }
}

/// Single pass. Returns (changed, list of flat indices of changed pixels).
#[allow(clippy::too_many_arguments)]
fn erode_pass(
    g: &Geom,
    input: &[u8],
    mask: Option<&[u8]>,
    output: &mut [u8],
    bdr_value: bool,
    invert: bool,
    cit: bool,
    want_list: bool,
) -> (bool, Vec<u64>) {
    let n = g.shape.len();
    let len = g.shape[n - 1];
    let lead_shape = &g.shape[..n - 1];
    let inv = invert as u8;
    // C: bdr_value flipped when inverted; `if (!border_value) out = _false`
    let bdr_ok = if invert { !bdr_value } else { bdr_value };
    // group elements by their leading offsets
    let mut groups: Vec<(Vec<isize>, Vec<isize>)> = Vec::new();
    for e in &g.elems {
        let lead = e[..n - 1].to_vec();
        match groups.iter_mut().find(|(l, _)| *l == lead) {
            Some((_, dxs)) => dxs.push(e[n - 1]),
            None => groups.push((lead, vec![e[n - 1]])),
        }
    }
    // line strides in lines
    let mut lstride = vec![1usize; n - 1];
    for k in (0..n - 1).rev() {
        lstride[k] = if k + 1 < n - 1 {
            lstride[k + 1] * lead_shape[k + 1]
        } else {
            1
        };
    }
    let lines_per_chunk = (65536 / len.max(1)).max(1);
    let chunk = lines_per_chunk * len;
    let results: Vec<(bool, Vec<u64>)> = pool().install(|| {
        output
            .par_chunks_mut(chunk)
            .enumerate()
            .map(|(ci, out_chunk)| {
                let mut acc = vec![0u8; len];
                let mut ok = vec![0u8; len];
                let mut dl = vec![0u8; if want_list { len } else { 0 }];
                let mut changed = false;
                let mut list = Vec::new();
                let first_line = ci * lines_per_chunk;
                let nl = out_chunk.len() / len;
                let mut coord = vec![0usize; n - 1];
                for li in 0..nl {
                    let line = first_line + li;
                    let mut r = line;
                    for k in (0..n - 1).rev() {
                        coord[k] = r % lead_shape[k];
                        r /= lead_shape[k];
                    }
                    for a in acc.iter_mut() {
                        *a = 1;
                    }
                    'groups: for (lead, dxs) in &groups {
                        let mut src_line = 0usize;
                        let mut inside = true;
                        for k in 0..n - 1 {
                            let c = coord[k] as isize + lead[k];
                            if c < 0 || c >= lead_shape[k] as isize {
                                inside = false;
                                break;
                            }
                            src_line += c as usize * lstride[k];
                        }
                        if !inside {
                            if !bdr_ok {
                                for a in acc.iter_mut() {
                                    *a = 0;
                                }
                                break 'groups;
                            }
                            continue;
                        }
                        let src = &input[src_line * len..(src_line + 1) * len];
                        for (o, &s) in ok.iter_mut().zip(src) {
                            *o = ((s != 0) as u8) ^ inv;
                        }
                        for &dx in dxs {
                            let lo = (-dx).max(0).min(len as isize) as usize;
                            let hi = (len as isize - dx).max(0).min(len as isize) as usize;
                            let hi = hi.max(lo);
                            if !bdr_ok {
                                for a in &mut acc[..lo] {
                                    *a = 0;
                                }
                                for a in &mut acc[hi..] {
                                    *a = 0;
                                }
                            }
                            let cnt = hi - lo;
                            if cnt == 0 {
                                continue;
                            }
                            let s0 = (lo as isize + dx) as usize;
                            let a = &mut acc[lo..hi];
                            let o = &ok[s0..s0 + cnt];
                            for (a, &o) in a.iter_mut().zip(o) {
                                *a &= o;
                            }
                        }
                    }
                    let base = line * len;
                    let in_line = &input[base..base + len];
                    let out_line = &mut out_chunk[li * len..(li + 1) * len];
                    let mline = mask.map(|m| &m[base..base + len]);
                    let cit_b = cit as u8;
                    let mut diff = 0u8;
                    match mline {
                        None => {
                            for ((o, &i), &a) in out_line.iter_mut().zip(in_line).zip(acc.iter()) {
                                let in01 = (i != 0) as u8;
                                let out = a ^ inv;
                                let keep = cit_b & ((in01 == inv) as u8);
                                let out = if keep != 0 { in01 } else { out };
                                diff |= out ^ in01;
                                *o = out;
                            }
                        }
                        Some(ml) => {
                            for (((o, &i), &a), &mm) in
                                out_line.iter_mut().zip(in_line).zip(acc.iter()).zip(ml)
                            {
                                let in01 = (i != 0) as u8;
                                let out = a ^ inv;
                                let keep = (cit_b & ((in01 == inv) as u8)) | ((mm == 0) as u8);
                                let out = if keep != 0 { in01 } else { out };
                                diff |= out ^ in01;
                                *o = out;
                            }
                        }
                    }
                    if diff != 0 {
                        changed = true;
                        if want_list {
                            for ((d, &o), &i) in dl.iter_mut().zip(out_line.iter()).zip(in_line) {
                                *d = o ^ ((i != 0) as u8);
                            }
                            let mut x = 0;
                            while x + 8 <= len {
                                let w = u64::from_le_bytes(dl[x..x + 8].try_into().unwrap());
                                let mut w = w;
                                while w != 0 {
                                    let j = (w.trailing_zeros() / 8) as usize;
                                    list.push((base + x + j) as u64);
                                    w &= w - 1;
                                }
                                x += 8;
                            }
                            while x < len {
                                if dl[x] != 0 {
                                    list.push((base + x) as u64);
                                }
                                x += 1;
                            }
                        }
                    }
                }
                (changed, list)
            })
            .collect()
    });
    let mut changed = false;
    let mut list = Vec::new();
    for (c, l) in results {
        changed |= c;
        if want_list {
            list.extend(l);
        }
    }
    (changed, list)
}

/// NI_BinaryErosion2. `array` is in/out (bool bytes), `mask` is the int8 array
/// SciPy builds (modified in place, as in C).
#[allow(clippy::too_many_arguments)]
fn erode2(
    g: &Geom,
    array: &mut [u8],
    mut mask: Option<&mut [i8]>,
    niter: i64,
    invert: bool,
    list: Vec<u64>,
) {
    let n = g.shape.len();
    let t: u8 = if invert { 0 } else { 1 };
    let f: u8 = 1 - t;
    if let Some(m) = mask.as_deref_mut() {
        for (a, m) in array.iter_mut().zip(m.iter_mut()) {
            if *m != 0 {
                *m = -1;
            } else {
                *m = *a as i8;
                *a = f;
            }
        }
    }
    let mut strides = vec![1isize; n];
    for k in (0..n - 1).rev() {
        strides[k] = strides[k + 1] * g.shape[k + 1] as isize;
    }
    let flat: Vec<isize> = g
        .elems
        .iter()
        .map(|e| e.iter().zip(&strides).map(|(a, s)| a * s).sum())
        .collect();
    let mut lo_m = vec![0isize; n];
    let mut hi_m = vec![0isize; n];
    for e in &g.elems {
        for k in 0..n {
            lo_m[k] = lo_m[k].max(-e[k]);
            hi_m[k] = hi_m[k].max(e[k]);
        }
    }
    let shape: Vec<isize> = g.shape.iter().map(|&s| s as isize).collect();
    let mut cur = list;
    let mut jj: i64 = 0;
    let nthreads = num_threads();
    while !cur.is_empty() {
        if !(niter <= 0 || jj < niter) {
            break;
        }
        jj += 1;
        let mklist = niter <= 0 || jj < niter;
        let process = |chunk: &[u64], arr: &[AtomicU8], next: &mut Vec<u64>| {
            let mut coord = vec![0isize; n];
            for &p in chunk {
                let mut r = p as usize;
                let mut interior = true;
                for k in (0..n).rev() {
                    let c = (r % g.shape[k]) as isize;
                    r /= g.shape[k];
                    coord[k] = c;
                    if c < lo_m[k] || c + hi_m[k] >= shape[k] {
                        interior = false;
                    }
                }
                'e: for (e, &fo) in g.elems.iter().zip(&flat) {
                    if !interior {
                        for k in 0..n {
                            let c = coord[k] + e[k];
                            if c < 0 || c >= shape[k] {
                                continue 'e;
                            }
                        }
                    }
                    let q = (p as isize + fo) as usize;
                    // exactly one thread wins the swap, as in the serial C code
                    if arr[q].load(Ordering::Relaxed) == t
                        && arr[q].swap(f, Ordering::Relaxed) == t
                        && mklist
                    {
                        next.push(q as u64);
                    }
                }
            }
        };
        let arr_at: &[AtomicU8] =
            unsafe { std::slice::from_raw_parts(array.as_ptr() as *const AtomicU8, array.len()) };
        let next: Vec<u64> = if cur.len() < 20000 || nthreads == 1 {
            let mut next = Vec::new();
            process(&cur, arr_at, &mut next);
            next
        } else {
            let parts: Vec<Vec<u64>> = pool().install(|| {
                cur.par_chunks(4096)
                    .map(|c| {
                        let mut v = Vec::new();
                        process(c, arr_at, &mut v);
                        v
                    })
                    .collect()
            });
            parts.concat()
        };
        cur = next;
    }
    if let Some(m) = mask.as_deref() {
        for (a, &m) in array.iter_mut().zip(m.iter()) {
            if m >= 0 {
                *a = m as u8;
            }
        }
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn binary_erosion_pass<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyArrayDyn<u8>>,
    structure: &Bound<'py, PyArrayDyn<u8>>,
    mask: Option<&Bound<'py, PyArrayDyn<u8>>>,
    output: &Bound<'py, PyArrayDyn<u8>>,
    border_value: bool,
    origin: Vec<isize>,
    invert: bool,
    cit: bool,
    want_list: bool,
) -> PyResult<(bool, Option<Bound<'py, PyArray1<u64>>>)> {
    let shape = input.shape().to_vec();
    let sshape = structure.shape().to_vec();
    let (inp, st, out) = unsafe {
        (
            input.as_slice()?,
            structure.as_slice()?,
            output.as_slice_mut()?,
        )
    };
    let msk = match mask {
        Some(m) => Some(unsafe { m.as_slice()? }),
        None => None,
    };
    let g = geom(&shape, &sshape, st, &origin);
    let (changed, list) =
        py.detach(|| erode_pass(&g, inp, msk, out, border_value, invert, cit, want_list));
    Ok((
        changed,
        if want_list {
            Some(PyArray1::from_vec(py, list))
        } else {
            None
        },
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn binary_erosion2<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyArrayDyn<u8>>,
    structure: &Bound<'py, PyArrayDyn<u8>>,
    mask: Option<&Bound<'py, PyArrayDyn<i8>>>,
    niter: i64,
    origin: Vec<isize>,
    invert: bool,
    list: &Bound<'py, PyArray1<u64>>,
) -> PyResult<()> {
    let shape = array.shape().to_vec();
    let sshape = structure.shape().to_vec();
    let st = unsafe { structure.as_slice()? };
    let arr = unsafe { array.as_slice_mut()? };
    let msk = match mask {
        Some(m) => Some(unsafe { m.as_slice_mut()? }),
        None => None,
    };
    let l: Vec<u64> = unsafe { list.as_slice()? }.to_vec();
    let g = geom(&shape, &sshape, st, &origin);
    py.detach(|| erode2(&g, arr, msk, niter, invert, l));
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_erosion_pass, m)?)?;
    m.add_function(wrap_pyfunction!(binary_erosion2, m)?)?;
    Ok(())
}
