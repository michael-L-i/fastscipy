use pyo3::prelude::*;

use super::*;

fn find(parent: &mut [u32], mut v: u32) -> u32 {
    while parent[v as usize] != v {
        parent[v as usize] = parent[parent[v as usize] as usize]; // path halving
        v = parent[v as usize];
    }
    v
}

/// Weak components. SciPy numbers them in order of their smallest node, which
/// union-find with "smaller root wins" reproduces exactly.
#[pyfunction]
pub fn weak(
    py: Python<'_>,
    indptr: Ro<i32>,
    indices: Ro<i32>,
    mut labels: Rw<i32>,
) -> PyResult<i32> {
    let (ip, ix) = (indptr.as_slice()?, indices.as_slice()?);
    let lab = labels.as_slice_mut()?;
    let g = Csr::new(ip, ix, None, lab.len())?;
    Ok(py.detach(|| {
        let n = lab.len();
        let mut parent: Vec<u32> = (0..n as u32).collect();
        for v in 0..n {
            for e in g.row(v) {
                let (a, b) = (
                    find(&mut parent, v as u32),
                    find(&mut parent, g.indices[e] as u32),
                );
                if a != b {
                    parent[a.max(b) as usize] = a.min(b);
                }
            }
        }
        let mut count = 0;
        for v in 0..n {
            let r = find(&mut parent, v as u32) as usize;
            lab[v] = if r == v {
                count += 1;
                count - 1
            } else {
                lab[r]
            };
        }
        count
    }))
}

/// Strong components: the iterative Tarjan variant of `_connected_components_directed2`.
#[pyfunction]
pub fn strong(
    py: Python<'_>,
    indptr: Ro<i32>,
    indices: Ro<i32>,
    mut labels: Rw<i32>,
) -> PyResult<i32> {
    const NONLEAD: i32 = i32::MIN;
    let (ip, ix) = (indptr.as_slice()?, indices.as_slice()?);
    let lab = labels.as_slice_mut()?;
    Csr::new(ip, ix, None, lab.len())?;
    Ok(py.detach(|| {
        let n = lab.len();
        lab.fill(0);
        let node = |v: i32| (v & 0x7FFF_FFFF) as usize;
        let mut succ = vec![-1i32; n];
        let (mut dfs, mut comp): (Vec<i32>, Vec<i32>) =
            (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut index, mut n_comp) = (n as i32, 0);
        for root in 0..n {
            if succ[root] != -1 {
                continue;
            }
            let root_hl = index;
            dfs.push(root as i32);
            lab[root] = index;
            index -= 1;
            succ[root] = ip[root];
            while let Some(&top) = dfs.last() {
                let v = node(top);
                if succ[v] < ip[v + 1] {
                    let w = ix[succ[v] as usize] as usize;
                    succ[v] += 1;
                    if succ[w] == -1 {
                        dfs.push(w as i32);
                        lab[w] = index;
                        index -= 1;
                        succ[w] = ip[w];
                    } else if lab[v] < lab[w] {
                        *dfs.last_mut().unwrap() = v as i32 | NONLEAD;
                        lab[v] = lab[w];
                        if lab[v] == root_hl && index == 0 {
                            lab[v] = n_comp;
                            comp.iter().for_each(|&c| lab[c as usize] = n_comp);
                            dfs[..dfs.len() - 1]
                                .iter()
                                .for_each(|&d| lab[node(d)] = n_comp);
                            return n_comp + 1;
                        }
                    }
                } else if top >= 0 {
                    dfs.pop();
                    while let Some(&t) = comp.last() {
                        if lab[v] < lab[t as usize] {
                            break;
                        }
                        lab[t as usize] = n_comp;
                        comp.pop();
                        index += 1;
                    }
                    lab[v] = n_comp;
                    index += 1;
                    n_comp += 1;
                } else {
                    dfs.pop();
                    comp.push(v as i32);
                    if let Some(p) = dfs.last_mut() {
                        let parent = node(*p);
                        if lab[parent] < lab[v] {
                            *p = parent as i32 | NONLEAD;
                            lab[parent] = lab[v];
                        }
                    }
                }
            }
        }
        n_comp
    }))
}
