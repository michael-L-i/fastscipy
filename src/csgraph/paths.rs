use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicI64, Ordering::Relaxed};

use pyo3::prelude::*;

use super::*;

const EPS: f64 = 1e-15;
/// Per-source work below which threads are not worth waking.
const PARALLEL_WORK: usize = 1 << 15;

/// C++ `pair<double, unsigned>` ordering in a max-heap (keys are never NaN).
#[derive(PartialEq)]
struct Key(f64, u32);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal).then(self.1.cmp(&o.1))
    }
}

/// Outputs of one search. Empty `pred` / `src` mean "not requested".
struct Out<'a> {
    dist: &'a mut [f64],
    pred: &'a mut [i32],
    src: &'a mut [i32],
}

impl Out<'_> {
    #[inline]
    fn relax(&mut self, v: usize, j: usize, next: f64) {
        self.dist[j] = next;
        if !self.pred.is_empty() {
            self.pred[j] = v as i32;
        }
        if !self.src.is_empty() {
            self.src[j] = self.src[v];
        }
    }
}

/// `_dijkstra` with a binary heap, popping equal distances in descending node order.
fn heap_search(srcs: &[i32], gs: &[&Csr], out: &mut Out, limit: f64) {
    let mut heap = BinaryHeap::new();
    for &s in srcs {
        out.dist[s as usize] = 0.0;
        heap.push(Key(-out.dist[s as usize], s as u32));
        if !out.src.is_empty() {
            out.src[s as usize] = s;
        }
    }
    while let Some(Key(nd, v)) = heap.pop() {
        let (d, v) = (-nd, v as usize);
        if out.dist[v] < d {
            continue;
        }
        for g in gs {
            for e in g.row(v) {
                let j = g.indices[e] as usize;
                let next = d + g.weight(e);
                if next <= limit && out.dist[j] > next {
                    out.relax(v, j, next);
                    heap.push(Key(-next, j as u32));
                }
            }
        }
    }
}

/// The same result for unit weights: Dijkstra pops one distance level at a time,
/// highest node first, so a level-by-level search in that order sets the same predecessors.
fn level_search(srcs: &[i32], gs: &[&Csr], out: &mut Out, limit: f64) {
    let mut level: Vec<u32> = srcs.iter().map(|&s| s as u32).collect();
    for &s in srcs {
        out.dist[s as usize] = 0.0;
        if !out.src.is_empty() {
            out.src[s as usize] = s;
        }
    }
    let mut next_level = Vec::new();
    let mut next = 0.0;
    while !level.is_empty() {
        level.sort_unstable_by(|a, b| b.cmp(a));
        level.dedup();
        next += 1.0;
        if next > limit {
            break;
        }
        for &v in &level {
            for g in gs {
                for e in g.row(v as usize) {
                    let j = g.indices[e] as usize;
                    if out.dist[j] > next {
                        out.relax(v as usize, j, next);
                        next_level.push(j as u32);
                    }
                }
            }
        }
        std::mem::swap(&mut level, &mut next_level);
        next_level.clear();
    }
}

/// Dijkstra from each source into its own row of `dist` (or all sources into one row
/// when `min_only`). `dist`/`pred`/`src` are flattened rows of length N; `pred` and
/// `src` are empty when not requested. `tindptr` empty means a directed graph.
#[pyfunction]
#[pyo3(signature = (srcs, indptr, indices, data, tindptr, tindices, tdata, dist, pred, src, limit, min_only, unit))]
#[allow(clippy::too_many_arguments)]
pub fn dijkstra(
    py: Python<'_>, srcs: Ro<i32>, indptr: Ro<i32>, indices: Ro<i32>, data: Ro<f64>, tindptr: Ro<i32>,
    tindices: Ro<i32>, tdata: Ro<f64>, mut dist: Rw<f64>, mut pred: Rw<i32>, mut src: Rw<i32>, limit: f64,
    min_only: bool, unit: bool,
) -> PyResult<()> {
    let (srcs, n) = (srcs.as_slice()?, indptr.len()?.saturating_sub(1));
    fn w<'a>(a: &'a Ro<f64>, unit: bool) -> PyResult<Option<&'a [f64]>> {
        Ok(if unit { None } else { Some(a.as_slice()?) })
    }
    let g = Csr::new(indptr.as_slice()?, indices.as_slice()?, w(&data, unit)?, n)?;
    let gt = if tindptr.len()? == 0 {
        None
    } else {
        Some(Csr::new(tindptr.as_slice()?, tindices.as_slice()?, w(&tdata, unit)?, n)?)
    };
    let (dist, pred, src) = (dist.as_slice_mut()?, pred.as_slice_mut()?, src.as_slice_mut()?);
    let rows = if min_only { 1 } else { srcs.len() };
    let (pw, sw) = (if pred.is_empty() { 0 } else { n }, if src.is_empty() { 0 } else { n });
    if srcs.iter().any(|&s| s < 0 || s as usize >= n) || dist.len() != rows * n || pred.len() != rows * pw
        || src.len() != rows * sw || limit.is_nan() {
        return Err(PyValueError::new_err("bad dijkstra arguments"));
    }
    let gs: Vec<&Csr> = std::iter::once(&g).chain(gt.as_ref()).collect();
    let (pd, pp, ps) = (Ptr(dist.as_mut_ptr()), Ptr(pred.as_mut_ptr()), Ptr(src.as_mut_ptr()));
    let work = (g.indices.len() + n) * if min_only { 1 } else { srcs.len() };
    for_rows(py, rows, work >= PARALLEL_WORK, |i| {
        let mut out = unsafe { Out { dist: pd.row(i, n), pred: pp.row(i, pw), src: ps.row(i, sw) } };
        let s = if min_only { srcs } else { &srcs[i..i + 1] };
        if unit { level_search(s, &gs, &mut out, limit) } else { heap_search(s, &gs, &mut out, limit) }
    });
    Ok(())
}

/// Relax from `source` until a pass changes nothing (further passes could not either).
/// Returns true if distances still improve after N-1 passes by more than EPS (negative cycle).
fn bellman_ford_row(g: &Csr, undirected: bool, out: &mut Out) -> bool {
    let n = out.dist.len();
    let return_pred = !out.pred.is_empty();
    for _ in 0..n.saturating_sub(1) {
        let mut changed = false;
        for j in 0..n {
            let mut d1 = out.dist[j];
            for k in g.row(j) {
                let (w, c) = (g.weight(k), g.indices[k] as usize);
                let mut d2 = out.dist[c];
                if d1 + w < d2 {
                    d2 = d1 + w;
                    out.dist[c] = d2;
                    changed = true;
                    if return_pred {
                        out.pred[c] = j as i32;
                    }
                }
                if undirected && d2 + w < d1 {
                    d1 = d2 + w;
                    out.dist[j] = d1;
                    changed = true;
                    if return_pred {
                        out.pred[j] = c as i32;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    (0..n).any(|j| {
        let d1 = out.dist[j];
        g.row(j).any(|k| {
            let (w, d2) = (g.weight(k), out.dist[g.indices[k] as usize]);
            if undirected { (d2 - d1).abs() > w + EPS } else { d1 + w + EPS < d2 }
        })
    })
}

/// Bellman-Ford into the rows of `dist` (and `pred` if non-empty). Returns the first
/// source (in order) that finds a negative cycle, or -1.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bellman_ford(
    py: Python<'_>, srcs: Ro<i32>, indptr: Ro<i32>, indices: Ro<i32>, data: Ro<f64>, undirected: bool,
    mut dist: Rw<f64>, mut pred: Rw<i32>,
) -> PyResult<i64> {
    let (srcs, n) = (srcs.as_slice()?, indptr.len()?.saturating_sub(1));
    let g = Csr::new(indptr.as_slice()?, indices.as_slice()?, Some(data.as_slice()?), n)?;
    let (dist, pred) = (dist.as_slice_mut()?, pred.as_slice_mut()?);
    let pw = if pred.is_empty() { 0 } else { n };
    if dist.len() != srcs.len() * n || pred.len() != srcs.len() * pw {
        return Err(PyValueError::new_err("bad bellman_ford arguments"));
    }
    let (pd, pp) = (Ptr(dist.as_mut_ptr()), Ptr(pred.as_mut_ptr()));
    let first_bad = AtomicI64::new(i64::MAX);
    let work = (g.indices.len() + n) * n.min(64) * srcs.len();
    for_rows(py, srcs.len(), work >= PARALLEL_WORK, |i| {
        if (i as i64) < first_bad.load(Relaxed) {
            let mut out = unsafe { Out { dist: pd.row(i, n), pred: pp.row(i, pw), src: &mut [] } };
            if bellman_ford_row(&g, undirected, &mut out) {
                first_bad.fetch_min(i as i64, Relaxed);
            }
        }
    });
    let bad = first_bad.load(Relaxed);
    Ok(if bad == i64::MAX { -1 } else { srcs[bad as usize] as i64 })
}

/// Johnson's first stage: Bellman-Ford from a virtual source joined to every node by
/// a zero-weight edge. `h` (zeros on entry) receives the potentials. Returns the node
/// that reveals a negative cycle, or -1.
#[pyfunction]
pub fn johnson_potentials(
    py: Python<'_>, indptr: Ro<i32>, indices: Ro<i32>, data: Ro<f64>, undirected: bool, mut h: Rw<f64>,
) -> PyResult<i64> {
    let h = h.as_slice_mut()?;
    let g = Csr::new(indptr.as_slice()?, indices.as_slice()?, Some(data.as_slice()?), h.len())?;
    Ok(py.detach(|| {
        let n = h.len();
        for _ in 0..n {
            let mut changed = false;
            for j in 0..n {
                let mut d1 = h[j];
                for k in g.row(j) {
                    let (w, c) = (g.weight(k), g.indices[k] as usize);
                    let d2 = h[c];
                    if d1 + w < d2 {
                        h[c] = d1 + w;
                        changed = true;
                    }
                    if undirected && d2 + w < d1 {
                        d1 = d2 + w;
                        h[j] = d1;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        (0..n)
            .find(|&j| {
                g.row(j).any(|k| {
                    let (w, d2) = (g.weight(k), h[g.indices[k] as usize]);
                    if undirected { (d2 - h[j]).abs() > w + EPS } else { h[j] + w + EPS < d2 }
                })
            })
            .map_or(-1, |j| j as i64)
    }))
}

/// `w(u, v) += h(u); w(u, v) -= h(v)` over a CSR matrix, as `_johnson_add_weights`.
#[pyfunction]
pub fn add_weights(indptr: Ro<i32>, indices: Ro<i32>, mut data: Rw<f64>, h: Ro<f64>) -> PyResult<()> {
    let (ip, ix, h) = (indptr.as_slice()?, indices.as_slice()?, h.as_slice()?);
    let w = data.as_slice_mut()?;
    Csr::new(ip, ix, Some(w), h.len())?;
    for j in 0..h.len() {
        for k in ip[j] as usize..ip[j + 1] as usize {
            w[k] += h[j];
            w[k] -= h[ix[k] as usize];
        }
    }
    Ok(())
}
