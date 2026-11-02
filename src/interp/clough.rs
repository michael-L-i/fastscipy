//! Clough–Tocher evaluation with triangle geometry shared across output channels.
//! Scalar expressions follow SciPy's interpolate/_interpnd.pyx (BSD-3-Clause),
//! Copyright (C) Pauli Virtanen, 2010. See LICENSES/scipy-LICENSE.txt.
//! Geometry and polynomial weights depend on the query, not the output channel.
//! Keep coefficient and accumulation order unchanged when sharing this work.

use super::madd;
use numpy::{
    PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

fn barycentric<const F: bool>(t: &[f64], x: &[f64]) -> [f64; 3] {
    let mut b = [0.0, 0.0, 1.0];
    for i in 0..2 {
        for j in 0..2 {
            b[i] = madd::<F>(t[2 * i + j], x[j] - t[4 + j], b[i]);
        }
        b[2] -= b[i];
    }
    b
}

fn weights(b: [f64; 3]) -> [f64; 19] {
    let mut min = b[0];
    for v in b {
        if v < min {
            min = v;
        }
    }
    let (b1, b2, b3, b4) = (b[0] - min, b[1] - min, b[2] - min, 3.0 * min);
    [
        b1.powf(3.0),
        3.0 * (b1 * b1) * b2,
        3.0 * (b1 * b1) * b3,
        3.0 * (b1 * b1) * b4,
        3.0 * b1 * (b2 * b2),
        6.0 * b1 * b2 * b4,
        3.0 * b1 * (b3 * b3),
        6.0 * b1 * b3 * b4,
        3.0 * b1 * (b4 * b4),
        b2.powf(3.0),
        3.0 * (b2 * b2) * b3,
        3.0 * (b2 * b2) * b4,
        3.0 * b2 * (b3 * b3),
        6.0 * b2 * b3 * b4,
        3.0 * b2 * (b4 * b4),
        b3.powf(3.0),
        3.0 * (b3 * b3) * b4,
        3.0 * b3 * (b4 * b4),
        b4.powf(3.0),
    ]
}

// Keep the shared geometry and weights visible to the channel loop optimizer.
#[inline(always)]
fn value<const F: bool>(
    edge: &[f64; 6],
    g: &[f64; 3],
    w: &[f64; 19],
    f: &[f64; 3],
    df: &[f64; 6],
) -> f64 {
    let [e12x, e12y, e23x, e23y, e31x, e31y] = *edge;
    let [c3000, c0300, c0030] = *f;
    // For two products SciPy contracts the left one; later additions contract
    // the new term. Unary negation must stay outside the directional derivative.
    let df12 = madd::<F>(df[0], e12x, df[1] * e12y);
    let df21 = -madd::<F>(df[2], e12x, df[3] * e12y);
    let df23 = madd::<F>(df[2], e23x, df[3] * e23y);
    let df32 = -madd::<F>(df[4], e23x, df[5] * e23y);
    let df31 = madd::<F>(df[4], e31x, df[5] * e31y);
    let df13 = -madd::<F>(df[0], e31x, df[1] * e31y);
    let c2100 = madd::<F>(3.0, c3000, df12) / 3.0;
    let c2010 = madd::<F>(3.0, c3000, df13) / 3.0;
    let c1200 = madd::<F>(3.0, c0300, df21) / 3.0;
    let c0210 = madd::<F>(3.0, c0300, df23) / 3.0;
    let c1020 = madd::<F>(3.0, c0030, df31) / 3.0;
    let c0120 = madd::<F>(3.0, c0030, df32) / 3.0;
    let c2001 = (c2100 + c2010 + c3000) / 3.0;
    let c0201 = (c1200 + c0300 + c0210) / 3.0;
    let c0021 = (c1020 + c0120 + c0030) / 3.0;
    let c0111 = madd::<F>(
        g[0],
        madd::<F>(-3.0, c0120, madd::<F>(3.0, c0210, -c0300)) + c0030,
        madd::<F>(2.0, c0210, -c0300) - c0120 + c0021 + c0201,
    ) / 2.0;
    let c1011 = madd::<F>(
        g[1],
        madd::<F>(-3.0, c2010, madd::<F>(3.0, c1020, -c0030)) + c3000,
        madd::<F>(2.0, c1020, -c0030) - c2010 + c2001 + c0021,
    ) / 2.0;
    let c1101 = madd::<F>(
        g[2],
        madd::<F>(-3.0, c1200, madd::<F>(3.0, c2100, -c3000)) + c0300,
        madd::<F>(2.0, c2100, -c3000) - c1200 + c2001 + c0201,
    ) / 2.0;
    let c1002 = (c1101 + c1011 + c2001) / 3.0;
    let c0102 = (c1101 + c0111 + c0201) / 3.0;
    let c0012 = (c1011 + c0111 + c0021) / 3.0;
    let c0003 = (c1002 + c0102 + c0012) / 3.0;
    let c = [
        c3000, c2100, c2010, c2001, c1200, c1101, c1020, c1011, c1002, c0300, c0210, c0201, c0120,
        c0111, c0102, c0030, c0021, c0012, c0003,
    ];
    let mut result = madd::<F>(w[0], c[0], w[1] * c[1]);
    for i in 2..19 {
        result = madd::<F>(w[i], c[i], result);
    }
    result
}

struct Clough<'a> {
    simplex: &'a [i32],
    transform: &'a [f64],
    simplices: &'a [i32],
    neighbors: &'a [i32],
    points: &'a [f64],
    values: &'a [f64],
    gradients: &'a [f64],
    queries: &'a [f64],
    channels: usize,
    fill: f64,
}

impl Clough<'_> {
    fn evaluate<const BASIS: bool, const F: bool>(&self, out: &mut [f64]) {
        let nv = self.channels;
        super::fill(out, nv, 1 << 12, |start, out| {
            for (i, row) in out.chunks_mut(nv).enumerate() {
                let s = self.simplex[start + i];
                if s < 0 {
                    row.fill(self.fill);
                    continue;
                }
                let s = s as usize;
                let vertices = &self.simplices[s * 3..][..3];
                let p = |vertex: i32, axis: usize| self.points[vertex as usize * 2 + axis];
                let mut edge = [0.0; 6];
                for j in 0..3 {
                    for axis in 0..2 {
                        edge[j * 2 + axis] = p(vertices[(j + 1) % 3], axis) - p(vertices[j], axis);
                    }
                }
                let t = &self.transform[s * 6..][..6];
                let mut g = [-0.5; 3];
                for (k, gk) in g.iter_mut().enumerate() {
                    let neighbor = self.neighbors[s * 3 + k];
                    if neighbor < 0 {
                        continue;
                    }
                    let v = &self.simplices[neighbor as usize * 3..][..3];
                    let y = [
                        (p(v[0], 0) + p(v[1], 0) + p(v[2], 0)) / 3.0,
                        (p(v[0], 1) + p(v[1], 1) + p(v[2], 1)) / 3.0,
                    ];
                    let c = barycentric::<BASIS>(t, &y);
                    let a = c[(k + 2) % 3];
                    let b = c[(k + 1) % 3];
                    *gk =
                        (madd::<F>(2.0, a, b) - 1.0) / madd::<F>(-3.0, b, madd::<F>(-3.0, a, 2.0));
                }
                let b = barycentric::<BASIS>(t, &self.queries[(start + i) * 2..][..2]);
                let w = weights(b);
                for (channel, output) in row.iter_mut().enumerate() {
                    let mut f = [0.0; 3];
                    let mut df = [0.0; 6];
                    for j in 0..3 {
                        let offset = vertices[j] as usize * nv + channel;
                        f[j] = self.values[offset];
                        df[j * 2..j * 2 + 2]
                            .copy_from_slice(&self.gradients[offset * 2..offset * 2 + 2]);
                    }
                    *output = value::<F>(&edge, &g, &w, &f, &df);
                }
            }
        });
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn clough_eval<'py>(
    py: Python<'py>,
    simplex: PyReadonlyArray1<'py, i32>,
    transform: PyReadonlyArray3<'py, f64>,
    simplices: PyReadonlyArray2<'py, i32>,
    neighbors: PyReadonlyArray2<'py, i32>,
    points: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray2<'py, f64>,
    gradients: PyReadonlyArray3<'py, f64>,
    queries: PyReadonlyArray2<'py, f64>,
    fill: f64,
    arithmetic: u8,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (n, nv, nt, np) = (
        simplex.len(),
        values.shape()[1],
        simplices.shape()[0],
        points.shape()[0],
    );
    if arithmetic > 3
        || nv == 0
        || transform.shape() != [nt, 3, 2]
        || simplices.shape()[1] != 3
        || neighbors.shape() != [nt, 3]
        || points.shape()[1] != 2
        || values.shape()[0] != np
        || gradients.shape() != [np, nv, 2]
        || queries.shape() != [n, 2]
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "invalid Clough–Tocher inputs",
        ));
    }
    let data = Clough {
        simplex: simplex.as_slice()?,
        transform: transform.as_slice()?,
        simplices: simplices.as_slice()?,
        neighbors: neighbors.as_slice()?,
        points: points.as_slice()?,
        values: values.as_slice()?,
        gradients: gradients.as_slice()?,
        queries: queries.as_slice()?,
        channels: nv,
        fill,
    };
    if data
        .simplex
        .iter()
        .any(|&s| s < -1 || s as usize >= nt && s != -1)
        || data
            .neighbors
            .iter()
            .any(|&s| s < -1 || s as usize >= nt && s != -1)
        || data.simplices.iter().any(|&s| s < 0 || s as usize >= np)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "invalid triangle index",
        ));
    }
    let out = PyArray2::<f64>::zeros(py, (n, nv), false);
    // SAFETY: newly allocated contiguous output, not exposed until evaluation finishes.
    let result = unsafe { out.as_slice_mut()? };
    py.detach(|| match arithmetic {
        0 => data.evaluate::<false, false>(result),
        1 => data.evaluate::<true, false>(result),
        2 => data.evaluate::<false, true>(result),
        _ => data.evaluate::<true, true>(result),
    });
    Ok(out)
}
