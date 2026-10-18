//! scipy.sparse.csgraph kernels, ported from `_shortest_path.pyx` and `_traversal.pyx`
//! (SciPy 1.18). Arrays arrive contiguous with int32 indices; the Python layer checks that.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::pool;

mod components;
mod paths;

type Ro<'py, T> = PyReadonlyArray1<'py, T>;
type Rw<'py, T> = PyReadwriteArray1<'py, T>;

/// A CSR graph, checked so that kernels can index without bounds surprises.
pub struct Csr<'a> {
    indptr: &'a [i32],
    indices: &'a [i32],
    data: Option<&'a [f64]>, // None: every weight is 1
}

impl<'a> Csr<'a> {
    fn new(indptr: &'a [i32], indices: &'a [i32], data: Option<&'a [f64]>, n: usize) -> PyResult<Self> {
        let ok = indptr.len() == n + 1
            && indptr[0] >= 0
            && indptr.windows(2).all(|w| w[0] <= w[1])
            && indptr[n] as usize <= indices.len()
            && data.is_none_or(|d| d.len() >= indptr[n] as usize)
            && indices[..indptr[n] as usize].iter().all(|&j| (j as u32 as usize) < n);
        if ok { Ok(Csr { indptr, indices, data }) } else { Err(PyValueError::new_err("malformed csgraph")) }
    }

    #[inline]
    fn row(&self, v: usize) -> std::ops::Range<usize> {
        self.indptr[v] as usize..self.indptr[v + 1] as usize
    }

    #[inline]
    fn weight(&self, e: usize) -> f64 {
        self.data.map_or(1.0, |d| d[e])
    }
}

/// A raw pointer that workers use on disjoint rows.
#[derive(Clone, Copy)]
struct Ptr<T>(*mut T);
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}
impl<T> Ptr<T> {
    /// Row `i` of `width` items; `width` 0 gives an empty slice (output disabled).
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, i: usize, width: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(i * width), width) }
    }
}

/// Run `f(0..rows)` on the shared pool when `parallel`, releasing the GIL.
fn for_rows(py: Python<'_>, rows: usize, parallel: bool, f: impl Fn(usize) + Sync + Send) {
    py.detach(|| {
        if parallel && pool::num_threads() > 1 && rows > 1 {
            pool::pool().install(|| (0..rows).into_par_iter().for_each(f));
        } else {
            (0..rows).for_each(f);
        }
    });
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paths::dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(paths::bellman_ford, m)?)?;
    m.add_function(wrap_pyfunction!(paths::johnson_potentials, m)?)?;
    m.add_function(wrap_pyfunction!(paths::add_weights, m)?)?;
    m.add_function(wrap_pyfunction!(components::weak, m)?)?;
    m.add_function(wrap_pyfunction!(components::strong, m)?)?;
    Ok(())
}
