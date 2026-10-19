//! Kernels for the interp family: spline prefiltering (`spline_filter1d`),
//! zoom/shift (`zoom_shift`) and `geometric_transform` without a Python
//! mapping (affine transforms and `map_coordinates`).
//!
//! Every entry point returns `Ok(false)`, without touching the output, when
//! the arguments are outside what it supports; the shim then calls SciPy.

use numpy::{PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;

mod band;
mod geom;
mod spline;
mod support;
mod zoom;

use crate::dispatch_elem;
use crate::ndimage::common::{self, Elem, Mode, typed};
use support::{OutElem, int_output_risky, worth_parallel};

/// Address range of an array's data, in bytes.
fn byte_range(a: &Bound<'_, PyUntypedArray>) -> (usize, usize) {
    let start = unsafe { (*a.as_array_ptr()).fields.data as usize };
    (start, start + a.len() * a.dtype().itemsize())
}

fn overlap(a: &Bound<'_, PyUntypedArray>, b: &Bound<'_, PyUntypedArray>) -> bool {
    let (a0, a1) = byte_range(a);
    let (b0, b1) = byte_range(b);
    a0 < b1 && b0 < a1
}

fn mode_of(code: i64) -> Option<Mode> {
    Mode::from_code(code)
}

/// `_nd_image.spline_filter1d(input, order, axis, output, mode)`.
/// `input` and `output` must be C-contiguous with the same shape; they may be
/// the same array (in-place filtering).
#[pyfunction]
fn interp_spline_filter1d<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    order: i64,
    axis: i64,
    output: &Bound<'py, PyUntypedArray>,
    mode: i64,
) -> PyResult<bool> {
    let Some(mode) = mode_of(mode) else {
        return Ok(false);
    };
    let Some(poles) = (0..=5)
        .contains(&order)
        .then(|| support::filter_poles(order as usize))
        .flatten()
    else {
        return Ok(false);
    };
    let ndim = input.ndim();
    if ndim == 0
        || axis < 0
        || axis as usize >= ndim
        || input.shape() != output.shape()
        || !input.is_c_contiguous()
        || !output.is_c_contiguous()
    {
        return Ok(false);
    }
    let in_place = byte_range(input) == byte_range(output);
    if in_place {
        if !input.dtype().is_equiv_to(&output.dtype()) {
            return Ok(false);
        }
    } else if overlap(input, output) {
        return Ok(false);
    }
    let shape = input.shape();
    let axis = axis as usize;
    let lines = spline::Lines {
        outer: shape[..axis].iter().product(),
        n: shape[axis],
        inner: shape[axis + 1..].iter().product(),
    };
    dispatch_elem!(py, input.dtype(), I => {
        dispatch_elem!(py, output.dtype(), O => {
            if !cfg!(target_arch = "aarch64") && <O as OutElem>::IS_INT {
                // Out-of-range C casts were only verified on aarch64.
                return Ok(false);
            }
            let src = typed::<I>(input)?.data() as *const I;
            let dst = typed::<O>(output)?.data();
            let (src, dst) = (support::SendConstPtr(src), support::SendPtr(dst));
            let work = lines.outer * lines.n * lines.inner * poles.len().max(1) * 4;
            if worth_parallel(work) {
                py.detach(|| {
                    let (src, dst) = (src, dst);
                    common::pool().install(|| unsafe {
                        spline::spline_filter1d::<I, O>(src.get(), dst.get(), lines, poles, mode, true)
                    })
                });
            } else {
                unsafe { spline::spline_filter1d::<I, O>(src.get(), dst.get(), lines, poles, mode, false) };
            }
            Ok(true)
        } else Ok(false))
    } else Ok(false))
}

fn read_f64(a: &Bound<'_, PyUntypedArray>) -> PyResult<Option<Vec<f64>>> {
    if a.ndim() != 1 || !a.is_c_contiguous() {
        return Ok(None);
    }
    let a = match a.cast::<numpy::PyArray1<f64>>() {
        Ok(a) => a,
        Err(_) => return Ok(None),
    };
    Ok(Some(a.readonly().as_slice()?.to_vec()))
}

/// `_nd_image.zoom_shift(input, zoom, shift, output, order, mode, cval, npad,
/// grid_mode)`. `zoom` and `shift` are None or 1-D float64 arrays.
#[pyfunction]
#[pyo3(signature = (input, zoom, shift, output, order, mode, cval, npad, grid_mode))]
#[allow(clippy::too_many_arguments)]
fn interp_zoom_shift<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    zoom: Option<&Bound<'py, PyUntypedArray>>,
    shift: Option<&Bound<'py, PyUntypedArray>>,
    output: &Bound<'py, PyUntypedArray>,
    order: i64,
    mode: i64,
    cval: f64,
    npad: i64,
    grid_mode: bool,
) -> PyResult<bool> {
    let Some(mode) = mode_of(mode) else {
        return Ok(false);
    };
    let rank = input.ndim();
    if !(0..=5).contains(&order)
        || !(0..=1 << 20).contains(&npad)
        || rank == 0
        || output.ndim() != rank
        || !input.is_c_contiguous()
        || !output.is_c_contiguous()
        || overlap(input, output)
    {
        return Ok(false);
    }
    let zooms = match zoom {
        Some(z) => match read_f64(z)? {
            Some(v) if v.len() == rank => Some(v),
            _ => return Ok(false),
        },
        None => None,
    };
    let shifts = match shift {
        Some(s) => match read_f64(s)? {
            Some(v) if v.len() == rank => Some(v),
            _ => return Ok(false),
        },
        None => None,
    };
    let params_finite = cval.is_finite()
        && zooms
            .iter()
            .chain(shifts.iter())
            .flatten()
            .all(|v| v.is_finite());
    let p = zoom::Params {
        order: order as usize,
        mode,
        cval,
        npad,
        grid_mode,
    };
    let in_shape = input.shape().to_vec();
    let out_shape = output.shape().to_vec();
    dispatch_elem!(py, input.dtype(), I => {
        dispatch_elem!(py, output.dtype(), O => {
            if int_output_risky::<O>(<I as Elem>::IS_FLOAT, params_finite) {
                return Ok(false);
            }
            let src = typed::<I>(input)?;
            let dst = typed::<O>(output)?;
            let (src, dst) = unsafe { (common::slice(&src), common::slice_mut(&dst)) };
            let (z, sh) = (zooms.as_deref(), shifts.as_deref());
            let work = dst.len() * (p.order + 1).pow(rank as u32) * 2;
            let ok = if worth_parallel(work) {
                py.detach(|| {
                    common::pool().install(|| {
                        zoom::zoom_shift(src, &in_shape, dst, &out_shape, z, sh, &p, true)
                    })
                })
            } else {
                zoom::zoom_shift(src, &in_shape, dst, &out_shape, z, sh, &p, false)
            };
            Ok(ok)
        } else Ok(false))
    } else Ok(false))
}

/// `_nd_image.geometric_transform` with `matrix` (float64, shape
/// `(input.ndim, output.ndim)`) and `offset` (float64, `input.ndim`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn interp_affine<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    matrix: &Bound<'py, PyUntypedArray>,
    offset: &Bound<'py, PyUntypedArray>,
    output: &Bound<'py, PyUntypedArray>,
    order: i64,
    mode: i64,
    cval: f64,
    npad: i64,
) -> PyResult<bool> {
    let Some(mode) = mode_of(mode) else {
        return Ok(false);
    };
    let (irank, orank) = (input.ndim(), output.ndim());
    if !(0..=5).contains(&order)
        || !(0..=1 << 20).contains(&npad)
        || matrix.ndim() != 2
        || matrix.shape() != [irank, orank]
        || !matrix.is_c_contiguous()
        || !input.is_c_contiguous()
        || !output.is_c_contiguous()
        || overlap(input, output)
    {
        return Ok(false);
    }
    let Ok(m) = matrix.cast::<numpy::PyArray2<f64>>() else {
        return Ok(false);
    };
    let m = m.readonly().as_slice()?.to_vec();
    let Some(off) = read_f64(offset)? else {
        return Ok(false);
    };
    if off.len() != irank {
        return Ok(false);
    }
    let params_finite = cval.is_finite() && m.iter().chain(off.iter()).all(|v| v.is_finite());
    let p = geom::Params {
        order: order as usize,
        mode,
        cval,
        npad,
    };
    let in_shape = input.shape().to_vec();
    let out_shape = output.shape().to_vec();
    dispatch_elem!(py, input.dtype(), I => {
        dispatch_elem!(py, output.dtype(), O => {
            if int_output_risky::<O>(<I as Elem>::IS_FLOAT, params_finite) {
                return Ok(false);
            }
            let src = typed::<I>(input)?;
            let dst = typed::<O>(output)?;
            let (src, dst) = unsafe { (common::slice(&src), common::slice_mut(&dst)) };
            let work = dst.len() * (p.order + 1).pow(irank as u32) * 4;
            let ok = if worth_parallel(work) {
                py.detach(|| {
                    common::pool().install(|| geom::affine(src, &in_shape, dst, &out_shape, &m, &off, &p, true))
                })
            } else {
                geom::affine(src, &in_shape, dst, &out_shape, &m, &off, &p, false)
            };
            Ok(ok)
        } else Ok(false))
    } else Ok(false))
}

/// `_nd_image.geometric_transform` with `coordinates`: float64, C-contiguous,
/// shape `(input.ndim,) + output.shape`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn interp_map_coordinates<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    coordinates: &Bound<'py, PyUntypedArray>,
    output: &Bound<'py, PyUntypedArray>,
    order: i64,
    mode: i64,
    cval: f64,
    npad: i64,
) -> PyResult<bool> {
    let Some(mode) = mode_of(mode) else {
        return Ok(false);
    };
    let rank = input.ndim();
    let cshape = coordinates.shape();
    if !(0..=5).contains(&order)
        || !(0..=1 << 20).contains(&npad)
        || cshape.is_empty()
        || cshape[0] != rank
        || cshape[1..] != *output.shape()
        || !coordinates.is_c_contiguous()
        || !input.is_c_contiguous()
        || !output.is_c_contiguous()
        || overlap(input, output)
        || overlap(coordinates, output)
    {
        return Ok(false);
    }
    let Ok(coords) = coordinates.cast::<numpy::PyArrayDyn<f64>>() else {
        return Ok(false);
    };
    let coords = coords.readonly();
    let coords = coords.as_slice()?;
    let p = geom::Params {
        order: order as usize,
        mode,
        cval,
        npad,
    };
    let in_shape = input.shape().to_vec();
    dispatch_elem!(py, input.dtype(), I => {
        dispatch_elem!(py, output.dtype(), O => {
            if !cfg!(target_arch = "aarch64") && <O as OutElem>::IS_INT {
                let finite = cval.is_finite() && geom::all_finite(coords);
                if int_output_risky::<O>(<I as Elem>::IS_FLOAT, finite) {
                    return Ok(false);
                }
            }
            let src = typed::<I>(input)?;
            let dst = typed::<O>(output)?;
            let (src, dst) = unsafe { (common::slice(&src), common::slice_mut(&dst)) };
            let work = dst.len() * (p.order + 1).pow(rank as u32) * 4;
            let ok = if worth_parallel(work) {
                py.detach(|| common::pool().install(|| geom::map_coords(src, &in_shape, dst, coords, &p, true)))
            } else {
                geom::map_coords(src, &in_shape, dst, coords, &p, false)
            };
            Ok(ok)
        } else Ok(false))
    } else Ok(false))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interp_spline_filter1d, m)?)?;
    m.add_function(wrap_pyfunction!(interp_zoom_shift, m)?)?;
    m.add_function(wrap_pyfunction!(interp_affine, m)?)?;
    m.add_function(wrap_pyfunction!(interp_map_coordinates, m)?)?;
    Ok(())
}
