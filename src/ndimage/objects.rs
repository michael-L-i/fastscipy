//! `find_objects`: per-label bounding boxes in one pass, run-compressed along the last axis.

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyTuple};
use rayon::prelude::*;

use crate::ndimage::common::{pool, slice, typed, Elem};

trait Lab: Copy + Send + Sync {
    /// label index (value - 1) as i64, or -1 when out of the representable range
    fn idx(self) -> i64;
}
macro_rules! lab_int { ($($t:ty),*) => {$(impl Lab for $t {
    #[inline(always)] fn idx(self) -> i64 { let v = self as i128 - 1; if v < 0 || v > i64::MAX as i128 { -1 } else { v as i64 } }
})*}; }
lab_int!(u8, i8, u16, i16, u32, i32, u64, i64);

/// regions layout per label: [min_0..min_{nd-1}, max_0..max_{nd-1}] with max exclusive; min = -1 when unseen.
fn scan<T: Lab>(shape: &[usize], data: &[T], max_label: usize, nthreads: usize) -> Vec<i64> {
    let nd = shape.len();
    let w = shape[nd - 1];
    let nlines = data.len() / w;
    let per = 2 * nd;
    let mem_threads = ((256usize << 20) / (max_label.max(1) * per * 8)).max(1);
    let nchunks = nthreads.min(mem_threads).min(nlines).max(1);
    let chunk_lines = nlines.div_ceil(nchunks);
    let nchunks = nlines.div_ceil(chunk_lines);
    let scan_chunk = |c: usize| -> Vec<i64> {
        let mut reg = vec![-1i64; max_label * per];
        let (l0, l1) = (c * chunk_lines, ((c + 1) * chunk_lines).min(nlines));
        let mut coord = vec![0i64; nd - 1];
        {
            let mut r = l0;
            for d in (0..nd - 1).rev() {
                coord[d] = (r % shape[d]) as i64;
                r /= shape[d];
            }
        }
        for l in l0..l1 {
            let line = &data[l * w..(l + 1) * w];
            let mut x = 0usize;
            while x < w {
                let idx = line[x].idx();
                let mut e = x + 1;
                // run of identical raw values; compare by label index
                while e < w && line[e].idx() == idx {
                    e += 1;
                }
                if idx >= 0 && (idx as usize) < max_label {
                    let base = idx as usize * per;
                    let r = &mut reg[base..base + per];
                    if r[0] < 0 {
                        for d in 0..nd - 1 {
                            r[d] = coord[d];
                            r[d + nd] = coord[d] + 1;
                        }
                        r[nd - 1] = x as i64;
                        r[2 * nd - 1] = e as i64;
                    } else {
                        for d in 0..nd - 1 {
                            if coord[d] < r[d] { r[d] = coord[d]; }
                            if coord[d] + 1 > r[d + nd] { r[d + nd] = coord[d] + 1; }
                        }
                        if (x as i64) < r[nd - 1] { r[nd - 1] = x as i64; }
                        if e as i64 > r[2 * nd - 1] { r[2 * nd - 1] = e as i64; }
                    }
                }
                x = e;
            }
            for d in (0..nd - 1).rev() {
                coord[d] += 1;
                if coord[d] < shape[d] as i64 { break; }
                coord[d] = 0;
            }
        }
        reg
    };
    let parts: Vec<Vec<i64>> = pool().install(|| (0..nchunks).into_par_iter().map(scan_chunk).collect());
    let mut it = parts.into_iter();
    let mut out = it.next().unwrap();
    for p in it {
        for k in 0..max_label {
            let b = k * per;
            if p[b] < 0 { continue; }
            if out[b] < 0 {
                out[b..b + per].copy_from_slice(&p[b..b + per]);
            } else {
                for d in 0..nd {
                    if p[b + d] < out[b + d] { out[b + d] = p[b + d]; }
                    if p[b + d + nd] > out[b + d + nd] { out[b + d + nd] = p[b + d + nd]; }
                }
            }
        }
    }
    out
}

/// find_objects_core(input, max_label) -> list | None
#[pyfunction]
fn find_objects_core<'py>(py: Python<'py>, input: &Bound<'py, PyUntypedArray>, max_label: usize) -> PyResult<Option<Bound<'py, PyList>>> {
    use numpy::PyArrayDescrMethods;
    if !input.is_c_contiguous() || input.ndim() == 0 || input.len() == 0 {
        return Ok(None);
    }
    let shape: Vec<usize> = input.shape().to_vec();
    let nd = shape.len();
    let nthreads = crate::ndimage::common::num_threads();
    let d = input.dtype();
    macro_rules! go { ($T:ty) => {{
        let a = typed::<$T>(input)?;
        let s = unsafe { slice(&a) };
        py.detach(|| scan::<$T>(&shape, s, max_label, nthreads))
    }}; }
    let reg = if d.is_equiv_to(&numpy::dtype::<u8>(py)) { go!(u8) }
    else if d.is_equiv_to(&numpy::dtype::<i8>(py)) { go!(i8) }
    else if d.is_equiv_to(&numpy::dtype::<u16>(py)) { go!(u16) }
    else if d.is_equiv_to(&numpy::dtype::<i16>(py)) { go!(i16) }
    else if d.is_equiv_to(&numpy::dtype::<u32>(py)) { go!(u32) }
    else if d.is_equiv_to(&numpy::dtype::<i32>(py)) { go!(i32) }
    else if d.is_equiv_to(&numpy::dtype::<u64>(py)) { go!(u64) }
    else if d.is_equiv_to(&numpy::dtype::<i64>(py)) { go!(i64) }
    else { return Ok(None); };
    let list = PyList::empty(py);
    let slice_t = py.get_type::<PySlice>();
    for k in 0..max_label {
        let b = k * 2 * nd;
        if reg[b] < 0 {
            list.append(py.None())?;
        } else {
            let mut items = Vec::with_capacity(nd);
            for dd in 0..nd {
                items.push(slice_t.call1((reg[b + dd], reg[b + dd + nd]))?);
            }
            list.append(PyTuple::new(py, items)?)?;
        }
    }
    Ok(Some(list))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_objects_core, m)?)?;
    Ok(())
}
#[allow(dead_code)]
fn _e<T: Elem>() {}
