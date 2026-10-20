//! Port of SciPy's NI_EuclideanFeatureTransform (Maurer et al., ni_morphology.c).
//!
//! The recursive C driver is equivalent to: initialise, then for d = 0..rank run
//! `_VoronoiFT` along every line of axis d. Lines of one pass are independent,
//! so they run in parallel. Arithmetic follows the C code, including the FMA
//! contraction used on macOS and Windows arm64. Official Linux wheels use
//! separate multiply/add instructions on both x86-64 and aarch64.

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ndimage::common::pool;

// Windows arm64 reference: the wheel SHA256 in interp/support.rs. Voronoi
// squared-distance sums use fmadd at RVAs 0x158b8/0x15a60, and the predicate
// uses fnmul/fmadd/fmsub/fnmul/fmadd at 0x1591c..0x1592c, matching the macOS
// expression tree below. Full Windows byte identity still requires CI.

#[inline(always)]
fn madd(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "windows")
    )) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

#[derive(Clone, Copy)]
struct Ptr<T>(*mut T);
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn voronoi_line<const R: usize, const S: bool>(
    ft: *mut i32,
    base: usize,
    stride: usize,
    cstride: usize,
    len: usize,
    rank_dyn: usize,
    d: usize,
    coor: &[i64],
    f: &mut Vec<i64>,
    g: &mut Vec<usize>,
    sampling_dyn: Option<&[f64]>,
) {
    let rank = if R == 0 { rank_dyn } else { R };
    let sampling = if S { sampling_dyn } else { None };
    f.resize(len * rank, 0);
    g.resize(len, 0);
    for ii in 0..len {
        for jj in 0..rank {
            f[ii * rank + jj] = unsafe { *ft.add(base + ii * stride + jj * cstride) } as i64;
        }
    }
    let mut l: isize = -1;
    for ii in 0..len {
        if unsafe { *ft.add(base + ii * stride) } >= 0 {
            let fd = f[ii * rank + d] as f64;
            let mut wr = 0.0f64;
            for jj in 0..rank {
                if jj != d {
                    let mut tw = (f[ii * rank + jj] - coor[jj]) as f64;
                    if let Some(s) = sampling {
                        tw *= s[jj];
                    }
                    wr = madd(tw, tw, wr);
                }
            }
            while l >= 1 {
                let idx1 = g[l as usize];
                let f1 = f[idx1 * rank + d] as f64;
                let idx2 = g[l as usize - 1];
                let mut a = f1 - f[idx2 * rank + d] as f64;
                let mut b = fd - f1;
                if let Some(s) = sampling {
                    a *= s[d];
                    b *= s[d];
                }
                let c = a + b;
                let (mut ur, mut vr) = (0.0f64, 0.0f64);
                for jj in 0..rank {
                    if jj != d {
                        let cc = coor[jj];
                        let mut tu = (f[idx2 * rank + jj] - cc) as f64;
                        let mut tv = (f[idx1 * rank + jj] - cc) as f64;
                        if let Some(s) = sampling {
                            tu *= s[jj];
                            tv *= s[jj];
                        }
                        ur = madd(tu, tu, ur);
                        vr = madd(tv, tv, vr);
                    }
                }
                // c*vR - b*uR - a*wR - a*b*c <= 0, as contracted by clang
                let t3 = if cfg!(all(
                    target_arch = "aarch64",
                    any(target_os = "macos", target_os = "windows")
                )) {
                    let t1 = madd(c, vr, -(b * ur));
                    let t2 = madd(-a, wr, t1);
                    madd(-(a * b), c, t2)
                } else {
                    c * vr - b * ur - a * wr - a * b * c
                };
                if t3 <= 0.0 {
                    break;
                }
                l -= 1;
            }
            l += 1;
            g[l as usize] = ii;
        }
    }
    let maxl = l;
    if maxl >= 0 {
        let maxl = maxl as usize;
        let mut l = 0usize;
        for ii in 0..len {
            let dist = |gi: usize, sampling: Option<&[f64]>| -> f64 {
                let mut delta = 0.0f64;
                for jj in 0..rank {
                    let mut t = if jj == d {
                        (f[gi * rank + jj] - ii as i64) as f64
                    } else {
                        (f[gi * rank + jj] - coor[jj]) as f64
                    };
                    if let Some(s) = sampling {
                        t *= s[jj];
                    }
                    delta = madd(t, t, delta);
                }
                delta
            };
            let mut delta1 = dist(g[l], sampling);
            while l < maxl {
                let delta2 = dist(g[l + 1], sampling);
                if delta1 <= delta2 {
                    break;
                }
                delta1 = delta2;
                l += 1;
            }
            let idx1 = g[l];
            for jj in 0..rank {
                unsafe {
                    *ft.add(base + ii * stride + jj * cstride) = f[idx1 * rank + jj] as i32;
                }
            }
        }
    }
}

fn feature_transform(input: &[i8], shape: &[usize], ft: &mut [i32], sampling: Option<&[f64]>) {
    let rank = shape.len();
    let n: usize = shape.iter().product();
    let mut strides = vec![1usize; rank];
    for k in (0..rank - 1).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    let ptr = Ptr(ft.as_mut_ptr());
    pool().install(|| {
        // init
        let last = shape[rank - 1];
        let nlines = n / last;
        (0..nlines)
            .into_par_iter()
            .with_min_len(64)
            .for_each(|line| {
                let p = ptr;
                let mut coord = vec![0usize; rank];
                let mut r = line;
                for k in (0..rank - 1).rev() {
                    coord[k] = r % shape[k];
                    r /= shape[k];
                }
                let base = line * last;
                for x in 0..last {
                    if input[base + x] != 0 {
                        unsafe {
                            *p.0.add(base + x) = -1;
                        }
                    } else {
                        coord[rank - 1] = x;
                        for k in 0..rank {
                            unsafe {
                                *p.0.add(base + x + k * n) = coord[k] as i32;
                            }
                        }
                    }
                }
            });
        for d in 0..rank {
            let len = shape[d];
            let nl = n / len;
            (0..nl)
                .into_par_iter()
                .with_min_len(8)
                .fold(
                    || (Vec::<i64>::new(), Vec::<usize>::new()),
                    |(mut f, mut g), line| {
                        let p = ptr;
                        // decode the line index over all axes except d (last axis fastest)
                        let mut coor = vec![0i64; rank];
                        let mut r = line;
                        let mut base = 0usize;
                        for k in (0..rank).rev() {
                            if k == d {
                                continue;
                            }
                            let c = r % shape[k];
                            r /= shape[k];
                            coor[k] = c as i64;
                            base += c * strides[k];
                        }
                        unsafe {
                            match (rank, sampling.is_some()) {
                                (1, false) => voronoi_line::<1, false>(
                                    p.0, base, strides[d], n, len, rank, d, &coor, &mut f, &mut g,
                                    sampling,
                                ),
                                (1, true) => voronoi_line::<1, true>(
                                    p.0, base, strides[d], n, len, rank, d, &coor, &mut f, &mut g,
                                    sampling,
                                ),
                                (2, false) => voronoi_line::<2, false>(
                                    p.0, base, strides[d], n, len, rank, d, &coor, &mut f, &mut g,
                                    sampling,
                                ),
                                (2, true) => voronoi_line::<2, true>(
                                    p.0, base, strides[d], n, len, rank, d, &coor, &mut f, &mut g,
                                    sampling,
                                ),
                                (3, false) => voronoi_line::<3, false>(
                                    p.0, base, strides[d], n, len, rank, d, &coor, &mut f, &mut g,
                                    sampling,
                                ),
                                (3, true) => voronoi_line::<3, true>(
                                    p.0, base, strides[d], n, len, rank, d, &coor, &mut f, &mut g,
                                    sampling,
                                ),
                                (_, false) => voronoi_line::<0, false>(
                                    p.0, base, strides[d], n, len, rank, d, &coor, &mut f, &mut g,
                                    sampling,
                                ),
                                (_, true) => voronoi_line::<0, true>(
                                    p.0, base, strides[d], n, len, rank, d, &coor, &mut f, &mut g,
                                    sampling,
                                ),
                            }
                        }
                        (f, g)
                    },
                )
                .for_each(|_| {});
        }
    });
}

#[pyfunction]
fn euclidean_feature_transform<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyArrayDyn<i8>>,
    sampling: Option<&Bound<'py, PyArrayDyn<f64>>>,
    ft: &Bound<'py, PyArrayDyn<i32>>,
) -> PyResult<()> {
    let shape = input.shape().to_vec();
    let inp = unsafe { input.as_slice()? };
    let out = unsafe { ft.as_slice_mut()? };
    let s = match sampling {
        Some(s) => Some(unsafe { s.as_slice()? }),
        None => None,
    };
    py.detach(|| feature_transform(inp, &shape, out, s));
    Ok(())
}

/// dt = sqrt(sum_j ((ft_j - index_j) * sampling_j)^2), numpy's evaluation order
#[pyfunction]
fn edt_distances<'py>(
    py: Python<'py>,
    ft: &Bound<'py, PyArrayDyn<i32>>,
    sampling: Option<&Bound<'py, PyArrayDyn<f64>>>,
    out: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<()> {
    let shape = out.shape().to_vec();
    let rank = shape.len();
    let n: usize = shape.iter().product();
    let f = unsafe { ft.as_slice()? };
    let o = unsafe { out.as_slice_mut()? };
    let s = match sampling {
        Some(s) => Some(unsafe { s.as_slice()? }),
        None => None,
    };
    let last = shape[rank - 1];
    py.detach(|| {
        pool().install(|| {
            o.par_chunks_mut(last).enumerate().for_each(|(line, row)| {
                let mut coord = vec![0i64; rank];
                let mut r = line;
                for k in (0..rank - 1).rev() {
                    coord[k] = (r % shape[k]) as i64;
                    r /= shape[k];
                }
                let base = line * last;
                for (x, o) in row.iter_mut().enumerate() {
                    coord[rank - 1] = x as i64;
                    let mut acc = 0.0f64;
                    for k in 0..rank {
                        let diff = (f[base + x + k * n] as i64 - coord[k]) as i32;
                        let mut v = diff as f64;
                        if let Some(s) = s {
                            v *= s[k];
                        }
                        let sq = v * v;
                        acc = if k == 0 { sq } else { acc + sq };
                    }
                    *o = acc.sqrt();
                }
            });
        });
    });
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(euclidean_feature_transform, m)?)?;
    m.add_function(wrap_pyfunction!(edt_distances, m)?)?;
    Ok(())
}
