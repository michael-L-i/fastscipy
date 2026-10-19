//! Literal port of `NI_WatershedIFT` (ni_measure.c), including its queue quirks:
//! * object markers go to the front of bucket 0, background markers to the end;
//! * an element is only unlinked from its old bucket when it has a next or prev link;
//! * neighbour bounds are checked on the *linear* index (rows wrap around);
//! * labels are stored in the output dtype and read back from it.

use numpy::PyUntypedArray;
use pyo3::prelude::*;

use crate::ndimage::common::{slice, slice_mut, typed};

const NONE: u32 = u32::MAX;

trait Val: Copy + Send + Sync {
    fn v(self) -> i32;
}
impl Val for u8 {
    #[inline(always)]
    fn v(self) -> i32 {
        self as i32
    }
}
impl Val for u16 {
    #[inline(always)]
    fn v(self) -> i32 {
        self as i32
    }
}

#[inline(always)]
fn wrap(label: i32, bits: u32) -> i32 {
    match bits {
        8 => label as i8 as i32,
        16 => label as i16 as i32,
        _ => label,
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct El {
    next: u32,
    prev: u32,
    cost: u32,
    done: u32,
}

fn watershed<I: Val>(input: &[I], markers: &[i32], nstrides: &[i64], lab: &mut [i32], bits: u32) {
    let size = input.len();
    let maxval = input.iter().map(|x| x.v()).max().unwrap_or(0) as usize;
    let mut e = vec![
        El {
            next: NONE,
            prev: NONE,
            cost: maxval as u32 + 1,
            done: 0
        };
        size
    ];
    let mut first = vec![NONE; maxval + 2];
    let mut last = vec![NONE; maxval + 2];
    for j in 0..size {
        let label = markers[j];
        lab[j] = wrap(label, bits);
        if label != 0 {
            e[j].cost = 0;
            let ju = j as u32;
            if first[0] == NONE {
                first[0] = ju;
                e[j].next = NONE;
                e[j].prev = NONE;
                last[0] = ju;
            } else if label > 0 {
                e[j].next = first[0];
                e[j].prev = NONE;
                e[first[0] as usize].prev = ju;
                first[0] = ju;
            } else {
                e[j].next = NONE;
                e[j].prev = last[0];
                e[last[0] as usize].next = ju;
                last[0] = ju;
            }
        }
    }
    let sz = size as i64;
    for jj in 0..=maxval {
        while first[jj] != NONE {
            let v = first[jj] as usize;
            first[jj] = e[v].next;
            if first[jj] != NONE {
                e[first[jj] as usize].prev = NONE;
            }
            e[v].prev = NONE;
            e[v].next = NONE;
            e[v].done = 1;
            let vv = input[v].v();
            let vcost = e[v].cost as i32;
            for &ns in nstrides {
                let pi = v as i64 + ns;
                if pi < 0 || pi >= sz {
                    continue;
                }
                let p = pi as usize;
                if e[p].done != 0 {
                    continue;
                }
                let mut wvp = input[p].v() - vv;
                if wvp < 0 {
                    wvp = -wvp;
                }
                let pcost = e[p].cost as i32;
                let max = if vcost > wvp { vcost } else { wvp };
                if max < pcost {
                    e[p].cost = max as u32;
                    let label = lab[v];
                    lab[p] = label;
                    if e[p].next != NONE || e[p].prev != NONE {
                        let (pv, nx) = (e[p].prev, e[p].next);
                        let pc = pcost as usize;
                        if first[pc] == p as u32 {
                            first[pc] = nx;
                        }
                        if last[pc] == p as u32 {
                            last[pc] = pv;
                        }
                        if pv != NONE {
                            e[pv as usize].next = nx;
                        }
                        if nx != NONE {
                            e[nx as usize].prev = pv;
                        }
                    }
                    let m = max as usize;
                    let pu = p as u32;
                    if label < 0 {
                        e[p].prev = last[m];
                        e[p].next = NONE;
                        if last[m] != NONE {
                            e[last[m] as usize].next = pu;
                        }
                        last[m] = pu;
                        if first[m] == NONE {
                            first[m] = pu;
                        }
                    } else {
                        e[p].next = first[m];
                        e[p].prev = NONE;
                        if first[m] != NONE {
                            e[first[m] as usize].prev = pu;
                        }
                        first[m] = pu;
                        if last[m] == NONE {
                            last[m] = pu;
                        }
                    }
                }
            }
        }
    }
}

/// watershed_core(input u8/u16, markers int32, nstrides, bits, lab int32 out) -> bool
#[pyfunction]
fn watershed_core<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    markers: &Bound<'py, PyUntypedArray>,
    nstrides: Vec<i64>,
    bits: u32,
    lab: &Bound<'py, PyUntypedArray>,
) -> PyResult<bool> {
    use numpy::{PyArrayDescrMethods, PyUntypedArrayMethods};
    let m = typed::<i32>(markers)?;
    let l = typed::<i32>(lab)?;
    let (ms, ls) = unsafe { (slice(&m), slice_mut(&l)) };
    if ms.len() >= (u32::MAX as usize) - 1 {
        return Ok(false);
    }
    if input.dtype().is_equiv_to(&numpy::dtype::<u8>(py)) {
        let i = typed::<u8>(input)?;
        let is = unsafe { slice(&i) };
        py.detach(|| watershed(is, ms, &nstrides, ls, bits));
        Ok(true)
    } else if input.dtype().is_equiv_to(&numpy::dtype::<u16>(py)) {
        let i = typed::<u16>(input)?;
        let is = unsafe { slice(&i) };
        py.detach(|| watershed(is, ms, &nstrides, ls, bits));
        Ok(true)
    } else {
        Ok(false)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(watershed_core, m)?)?;
    Ok(())
}
