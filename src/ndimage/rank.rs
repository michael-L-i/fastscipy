//! Rank family: `rank_filter`, `min_or_max_filter`, `min_or_max_filter1d`.
//!
//! Every kernel reproduces SciPy's result bit for bit or declines:
//! * Fast paths assume no NaN / negative zero among the values a window can
//!   select. Then every order statistic is a unique bit pattern and any exact
//!   selection algorithm matches SciPy.
//! * Where float input does contain NaN or -0.0 (detected per task while
//!   copying the input), an exact path replays SciPy's own comparison order
//!   (`NI_Select`, sequential min/max, Harter's algorithm on whole lines).
//! * Conversions to the output type follow SciPy's C casts; calls where C
//!   would hit undefined behaviour (out-of-range float -> int, 64-bit ints
//!   beyond 2^53, SciPy's out-of-bounds reflect quirk) decline.
//!
//! Layout:
//! * `boundary`: SciPy's boundary extension (`NI_ExtendLine`,
//!   `NI_InitFilterOffsets`).
//! * `engine`: tiled execution of n-d filters. Each task copies its
//!   boundary-extended input region into a cache-sized slab, then runs one of:
//!   generated networks (`nets`, from `gen_networks.py`: box 3x3/5x5/7x7,
//!   3x3x3 and disks of radius 1-5, every rank), interpreted selection
//!   networks (`network`, any footprint up to 2048 elements), a sliding
//!   histogram (8-bit data, large windows), register min/max reduction, or
//!   per-pixel select. Networks run on SIMD vectors of adjacent pixels and
//!   compute two output rows at once, sharing the rows their windows have in
//!   common.
//! * `minmax1d`: separable min/max along one axis (direct reduction, van
//!   Herk / Gil-Werman, doubling), in place when SciPy passes output = input.
//! * `simd`: NEON vectors on aarch64, portable arrays elsewhere.

use numpy::{
    PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use crate::dispatch_elem;
use crate::ndimage::common::{Mode, slice, slice_mut, typed, usable};

mod boundary;
mod elem;
mod engine;
mod minmax1d;
mod nets;
mod network;
mod par;
mod simd;

use boundary::reflect_quirk;
use elem::{OutElem, RElem, scan};
use engine::{
    Plan, Reduce, SendPtr, Slide, Task, gen_task, huang_task, net_task, ni_select, pixel_task,
    reduce_task, run_tasks, seq_minmax,
};
use nets::{GenNet, Visit};
use network::Net;
use simd::{HasVec, X2};

/// Slot block size of the network interpreter, in native vectors (8 x 16 B).
const INTERP_VECTORS: usize = 8;

/// Largest window evaluated with a selection network; bigger windows use a
/// per-pixel select.
const NET_MAX_K: usize = 2048;

/// Smallest window considered for the sliding histogram (8-bit data).
const HIST_MIN_K: usize = 100;

/// For 8-bit data: is a sliding histogram (cost ~ edge updates + search)
/// expected to beat an interpreted network (cost ~ k log^2 k / lanes)?
fn hist_better(k: usize, edge: usize) -> bool {
    k > HIST_MIN_K && edge * 4 < k
}

/// What to compute for an n-d filter, and the window in slab offsets.
struct NdKernel {
    op: NdOp,
    /// window offsets in footprint (C) order, for per-pixel paths
    offsets: Vec<isize>,
    /// window offsets grouped in rows, and the pairing shift (networks)
    rows: Vec<Vec<isize>>,
    shift: Option<isize>,
    /// offset lists for the fast path of min/max filters
    reduce: Option<Reduce>,
    /// sliding histogram for large windows of 8-bit data
    slide: Option<Slide>,
    /// (window coordinates, rank) to look up a generated network
    gen_key: Option<(Vec<[u8; 3]>, usize)>,
}

impl NdKernel {
    fn new(spec: &NdSpec, plan: &Plan, op: NdOp) -> NdKernel {
        let offsets: Vec<isize> = spec.coords.iter().map(|c| plan.offset(c)).collect();
        let shift = spec.pair_shift(plan);
        let reduce = matches!(op, NdOp::Min | NdOp::Max).then(|| Reduce::new(&offsets, shift));
        let slide = if matches!(op, NdOp::Rank(_)) && spec.k() > HIST_MIN_K {
            Some(Slide::new(&offsets)).filter(|s| hist_better(spec.k(), s.edge()))
        } else {
            None
        };
        NdKernel {
            op,
            offsets,
            rows: spec.rows(plan),
            shift,
            reduce,
            slide,
            gen_key: spec.gen_key(op),
        }
    }

    /// Interpreted networks (single row, row pair) for a rank filter, unless
    /// the window is too large for them.
    fn nets(&self) -> Option<(Net, Option<Net>)> {
        let NdOp::Rank(r) = self.op else { return None };
        if self.offsets.len() > NET_MAX_K {
            return None;
        }
        let single = network::build(&self.rows, None, r);
        let pair = self.shift.map(|s| network::build(&self.rows, Some(s), r));
        Some((single, pair))
    }
}

fn exact_fn<T: RElem>(op: NdOp) -> impl Fn(&mut [T]) -> T {
    move |b: &mut [T]| match op {
        NdOp::Rank(r) => ni_select(b, r),
        NdOp::Min => seq_minmax(b, true),
        NdOp::Max => seq_minmax(b, false),
    }
}

/// Runs a generated network over all tasks (from `nets::dispatch`).
struct GenRun<'a, T> {
    plan: &'a Plan,
    src: &'a [T],
    cval: T,
    k: &'a NdKernel,
    rank: usize,
    out: SendPtr<T>,
}

impl<T: HasVec> Visit for GenRun<'_, T> {
    type Out = ();
    fn visit<N: GenNet>(self) {
        let GenRun {
            plan,
            src,
            cval,
            k,
            rank,
            out,
        } = self;
        let nd = plan.ndim();
        let map = |inp: &[[u8; 3]]| -> Vec<isize> {
            inp.iter()
                .map(|c| {
                    let mut v = vec![0usize; nd];
                    for i in 0..3.min(nd) {
                        v[nd - 1 - i] = c[2 - i] as usize;
                    }
                    plan.offset(&v)
                })
                .collect()
        };
        let soff = map(N::SINGLE_IN);
        let poff = map(N::PAIR_IN);
        let exact = exact_fn::<T>(k.op);
        run_tasks(
            plan,
            src,
            cval,
            k.offsets.len(),
            |slab, t: &Task, ar| unsafe {
                let out = out;
                if t.special {
                    pixel_task(ar, slab, t, &k.offsets, out, &exact);
                } else {
                    gen_task::<T, N::W<T, T::V>, N>(slab, t, &soff, &poff, rank, out);
                }
            },
        );
    }
}

fn run_nd_impl<T: HasVec, const U: usize>(
    plan: &Plan,
    src: &[T],
    cval: T,
    k: &NdKernel,
    out: &mut [T],
) {
    let outp = SendPtr(out.as_mut_ptr());
    if T::GEN
        && let Some((win, r)) = &k.gen_key
    {
        let g = GenRun {
            plan,
            src,
            cval,
            k,
            rank: *r,
            out: outp,
        };
        if nets::dispatch(win, *r, g).is_some() {
            return;
        }
    }
    let slide = k.slide.as_ref().filter(|_| T::BINS == 256);
    let nets = if slide.is_none() { k.nets() } else { None };
    let exact = exact_fn::<T>(k.op);
    let select = |b: &mut [T]| match k.op {
        NdOp::Rank(r) => {
            b.select_nth_unstable_by(r, T::cmp_fast);
            b[r]
        }
        _ => unreachable!(),
    };
    run_tasks(plan, src, cval, k.offsets.len(), |slab, t, ar| unsafe {
        let outp = outp;
        if t.special {
            pixel_task(ar, slab, t, &k.offsets, outp, &exact);
        } else if let (Some(s), NdOp::Rank(r)) = (slide, k.op) {
            huang_task(slab, t, s, r, outp);
        } else if let Some((single, pair)) = &nets {
            net_task::<T, T::V, U>(ar, slab, t, single, pair.as_ref(), outp);
        } else if let Some(r) = &k.reduce {
            reduce_task::<T, X2<X2<T::V>>>(slab, t, r, k.op == NdOp::Min, outp);
        } else {
            pixel_task(ar, slab, t, &k.offsets, outp, &select);
        }
    });
}

fn is_constant(mode: Mode) -> bool {
    matches!(mode, Mode::Constant | Mode::GridConstant)
}

/// Footprint and geometry of an n-d filter call, validated.
struct NdSpec {
    shape: Vec<usize>,
    /// true footprint elements in C order: per-axis offsets into the padded
    /// window, i.e. c - origin_shift + before
    coords: Vec<Vec<usize>>,
    before: Vec<usize>,
    after: Vec<usize>,
    mode: Mode,
}

fn prepare_nd(
    shape: &[usize],
    fshape: &[usize],
    mask: &[bool],
    origins: &[i64],
    mode: i64,
) -> Option<NdSpec> {
    let nd = shape.len();
    let mode = Mode::from_code(mode)?;
    if nd == 0 || fshape.len() != nd || origins.len() != nd {
        return None;
    }
    if shape.iter().any(|&n| n == 0 || n > (1 << 30))
        || fshape.iter().any(|&f| f == 0 || f > (1 << 20))
    {
        return None;
    }
    if mask.len() != fshape.iter().product::<usize>() {
        return None;
    }
    let mut orgn = vec![0i64; nd];
    for i in 0..nd {
        orgn[i] = (fshape[i] / 2) as i64 + origins[i];
        if orgn[i] < 0 || orgn[i] >= fshape[i] as i64 {
            return None;
        }
    }
    // true elements, C order
    let mut raw: Vec<Vec<i64>> = Vec::new();
    let mut c = vec![0usize; nd];
    for &m in mask {
        if m {
            raw.push(c.iter().map(|&x| x as i64).collect());
        }
        for i in (0..nd).rev() {
            c[i] += 1;
            if c[i] < fshape[i] {
                break;
            }
            c[i] = 0;
        }
    }
    if raw.is_empty() {
        return None;
    }
    let mut before = vec![0usize; nd];
    let mut after = vec![0usize; nd];
    for i in 0..nd {
        let cmin = raw.iter().map(|c| c[i]).min().unwrap();
        let cmax = raw.iter().map(|c| c[i]).max().unwrap();
        if reflect_quirk(mode, orgn[i] - cmin, shape[i] as i64) {
            return None;
        }
        before[i] = (orgn[i] - cmin).max(0) as usize;
        after[i] = (cmax - orgn[i]).max(0) as usize;
    }
    let coords = raw
        .iter()
        .map(|c| {
            (0..nd)
                .map(|i| (c[i] - orgn[i] + before[i] as i64) as usize)
                .collect()
        })
        .collect();
    Some(NdSpec {
        shape: shape.to_vec(),
        coords,
        before,
        after,
        mode,
    })
}

impl NdSpec {
    fn k(&self) -> usize {
        self.coords.len()
    }
    /// Window elements grouped by leading coordinates (rows along the last axis).
    fn rows(&self, plan: &Plan) -> Vec<Vec<isize>> {
        let nd = self.shape.len();
        let mut rows: Vec<(Vec<usize>, Vec<isize>)> = Vec::new();
        for c in &self.coords {
            let off = plan.offset(c);
            let lead = c[..nd - 1].to_vec();
            match rows.iter_mut().find(|(l, _)| *l == lead) {
                Some((_, v)) => v.push(off),
                None => rows.push((lead, vec![off])),
            }
        }
        rows.into_iter().map(|(_, v)| v).collect()
    }
    /// For a rank filter whose window spans only the last two or three axes:
    /// its coordinates as sorted `[z, y, x]` (looked up in the generated
    /// networks), and the rank.
    fn gen_key(&self, op: NdOp) -> Option<(Vec<[u8; 3]>, usize)> {
        let NdOp::Rank(r) = op else { return None };
        let nd = self.shape.len();
        if nd < 2 {
            return None;
        }
        let span: Vec<usize> = (0..nd)
            .map(|i| self.before[i] + self.after[i] + 1)
            .collect();
        let lead = nd.saturating_sub(3);
        if span[..lead].iter().any(|&s| s > 1) || span.iter().any(|&s| s > 255) {
            return None;
        }
        let mut win: Vec<[u8; 3]> = self
            .coords
            .iter()
            .map(|c| {
                let z = if nd >= 3 { c[nd - 3] } else { 0 };
                [z as u8, c[nd - 2] as u8, c[nd - 1] as u8]
            })
            .collect();
        win.sort_unstable();
        Some((win, r))
    }
    /// Shift for pairing outputs along axis ndim-2, when the window extends
    /// along that axis (so the two windows share elements).
    fn pair_shift(&self, plan: &Plan) -> Option<isize> {
        let nd = self.shape.len();
        if nd < 2 || self.shape[nd - 2] < 2 || plan.span[nd - 2] < 2 {
            return None;
        }
        plan.row_shift()
    }
}

/// Read a bool footprint array.
fn read_footprint(fp: &Bound<'_, PyUntypedArray>) -> PyResult<Option<(Vec<usize>, Vec<bool>)>> {
    let Ok(a) = fp.cast::<PyArrayDyn<bool>>() else {
        return Ok(None);
    };
    if !a.is_c_contiguous() {
        return Ok(None);
    }
    let shape = a.shape().to_vec();
    let ro = a.readonly();
    let data = ro.as_slice().map(|s| s.to_vec()).ok();
    Ok(data.map(|d| (shape, d)))
}

/// Convert T results to the output array with SciPy's double round trip.
/// Returns false (output untouched) when some value would be undefined in C.
fn convert<T: RElem, O: OutElem>(src: &[T], dst: &mut [O], line: bool) -> bool {
    let conv = if line { O::conv_line } else { O::conv_filter };
    let sc = scan(src, true);
    if sc.lo <= sc.hi && (conv(sc.lo).is_none() || conv(sc.hi).is_none()) {
        return false;
    }
    let has_nan = |c: &[T]| c.iter().any(|v| v.partial_cmp(v).is_none());
    if sc.special
        && conv(f64::NAN).is_none()
        && par::map_reduce(src, 1 << 15, has_nan, || false, |a, b| a || b)
    {
        return false;
    }
    par::zip_chunks(dst, src, 1 << 15, |d, s| {
        for (d, &s) in d.iter_mut().zip(s) {
            *d = conv(s.to_f64()).unwrap();
        }
    });
    true
}

fn convert_to_output<T: RElem>(
    py: Python<'_>,
    tmp: &[T],
    output: &Bound<'_, PyUntypedArray>,
    line: bool,
    parallel: bool,
) -> PyResult<bool> {
    dispatch_elem!(py, output.dtype(), O => {
        let dst = typed::<O>(output)?;
        let dst = unsafe { slice_mut(&dst) };
        Ok(py.detach(|| par::run(parallel, || convert::<T, O>(tmp, dst, line))))
    } else Ok(false))
}

/// For 64-bit integers SciPy works in double precision; only accept values
/// that round-trip exactly.
fn ints_exact<T: RElem>(data: &[T], cval: Option<T>) -> bool {
    if T::EXACT_F64 {
        return true;
    }
    let lim = 9007199254740992.0; // 2^53
    let sc = scan(data, true);
    let ok = |v: f64| v.abs() <= lim;
    (sc.lo > sc.hi || (ok(sc.lo) && ok(sc.hi))) && cval.is_none_or(|c| ok(c.to_f64()))
}

#[derive(Clone, Copy, PartialEq)]
enum NdOp {
    Rank(usize),
    Min,
    Max,
}

/// Shared driver for rank_filter and min_or_max_filter.
fn nd_filter<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    output: &Bound<'py, PyUntypedArray>,
    spec: &NdSpec,
    op: NdOp,
    cval: f64,
) -> PyResult<bool> {
    if !usable(input) || !usable(output) || input.shape() != output.shape() {
        return Ok(false);
    }
    let same = input.dtype().is_equiv_to(&output.dtype());
    dispatch_elem!(py, input.dtype(), T => {
        nd_filter_typed::<T>(py, input, output, spec, op, cval, same)
    } else Ok(false))
}

fn nd_filter_typed<'py, T: HasVec>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    output: &Bound<'py, PyUntypedArray>,
    spec: &NdSpec,
    op: NdOp,
    cval: f64,
    same: bool,
) -> PyResult<bool> {
    let constant = is_constant(spec.mode);
    let cval_t: T = if constant {
        match T::c_cast(cval) {
            Some(v) => v,
            None => return Ok(false),
        }
    } else {
        T::default()
    };
    let src = typed::<T>(input)?;
    let src = unsafe { slice(&src) };
    let parallel = par::worth(par::work(src.len(), spec.k(), std::mem::size_of::<T>()));
    if !py.detach(|| par::run(parallel, || ints_exact(src, constant.then_some(cval_t)))) {
        return Ok(false);
    }
    let compute = |out: &mut [T]| {
        par::run(parallel, || {
            let plan = Plan::new(
                &spec.shape,
                &spec.before,
                &spec.after,
                spec.mode,
                std::mem::size_of::<T>(),
            );
            let kernel = NdKernel::new(spec, &plan, op);
            run_nd_impl::<T, INTERP_VECTORS>(&plan, src, cval_t, &kernel, out);
        })
    };
    if same {
        let dst = typed::<T>(output)?;
        let out = unsafe { slice_mut(&dst) };
        py.detach(|| compute(out));
        Ok(true)
    } else {
        let mut tmp: Vec<T> = vec![T::default(); src.len()];
        py.detach(|| compute(&mut tmp));
        convert_to_output(py, &tmp, output, false, parallel)
    }
}

#[pyfunction]
#[pyo3(name = "rank_filter")]
#[allow(clippy::too_many_arguments)]
fn py_rank_filter<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    rank: i64,
    footprint: &Bound<'py, PyUntypedArray>,
    output: &Bound<'py, PyUntypedArray>,
    mode: i64,
    cval: f64,
    origins: Vec<i64>,
) -> PyResult<bool> {
    let Some((fshape, mask)) = read_footprint(footprint)? else {
        return Ok(false);
    };
    let Some(spec) = prepare_nd(input.shape(), &fshape, &mask, &origins, mode) else {
        return Ok(false);
    };
    if rank < 0 || rank as usize >= spec.k() {
        return Ok(false);
    }
    nd_filter(py, input, output, &spec, NdOp::Rank(rank as usize), cval)
}

#[pyfunction]
#[pyo3(name = "min_or_max_filter")]
#[allow(clippy::too_many_arguments)]
fn py_min_or_max_filter<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    footprint: &Bound<'py, PyUntypedArray>,
    structure: Option<&Bound<'py, PyAny>>,
    output: &Bound<'py, PyUntypedArray>,
    mode: i64,
    cval: f64,
    origins: Vec<i64>,
    minimum: i64,
) -> PyResult<bool> {
    if structure.is_some_and(|s| !s.is_none()) {
        return Ok(false);
    }
    let Some((fshape, mask)) = read_footprint(footprint)? else {
        return Ok(false);
    };
    let Some(spec) = prepare_nd(input.shape(), &fshape, &mask, &origins, mode) else {
        return Ok(false);
    };
    let op = if minimum != 0 { NdOp::Min } else { NdOp::Max };
    nd_filter(py, input, output, &spec, op, cval)
}

#[pyfunction]
#[pyo3(name = "min_or_max_filter1d")]
#[allow(clippy::too_many_arguments)]
fn py_min_or_max_filter1d<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    size: i64,
    axis: i64,
    output: &Bound<'py, PyUntypedArray>,
    mode: i64,
    cval: f64,
    origin: i64,
    minimum: i64,
) -> PyResult<bool> {
    let Some(mode) = Mode::from_code(mode) else {
        return Ok(false);
    };
    let shape = input.shape().to_vec();
    let nd = shape.len();
    if nd == 0 || axis < 0 || axis as usize >= nd || size < 1 || input.shape() != output.shape() {
        return Ok(false);
    }
    if !usable(input)
        || !usable(output)
        || shape.iter().any(|&n| n == 0 || n > (1 << 30))
        || size > (1 << 30)
    {
        return Ok(false);
    }
    let before = size / 2 + origin;
    if before < 0 || before >= size {
        return Ok(false);
    }
    let p = minmax1d::Params {
        size: size as usize,
        before: before as usize,
        mode,
        minimum: minimum != 0,
    };
    let same = input.dtype().is_equiv_to(&output.dtype());
    dispatch_elem!(py, input.dtype(), T => {
        filter1d_typed::<T>(py, input, output, &shape, axis as usize, p, cval, same)
    } else Ok(false))
}

#[allow(clippy::too_many_arguments)]
fn filter1d_typed<'py, T: HasVec>(
    py: Python<'py>,
    input: &Bound<'py, PyUntypedArray>,
    output: &Bound<'py, PyUntypedArray>,
    shape: &[usize],
    axis: usize,
    p: minmax1d::Params,
    cval: f64,
    same: bool,
) -> PyResult<bool> {
    let constant = is_constant(p.mode);
    let src_arr = typed::<T>(input)?;
    let srcp = src_arr.data() as *const T;
    let n: usize = shape.iter().product();
    // The constant value: SciPy compares it as a double. Find an element of T
    // giving the same output, or force the exact path, or decline.
    let mut cv = minmax1d::Cval {
        t: T::default(),
        raw: cval,
        special: false,
    };
    if constant {
        if cval.is_nan() {
            if !T::is_float() {
                return Ok(false);
            }
            cv.special = true;
        } else {
            match T::c_cast(cval) {
                // exact, or float output of the same type: (T) commutes with min/max
                Some(c) if c.to_f64().to_bits() == cval.to_bits() || (same && T::is_float()) => {
                    cv.t = c;
                    cv.special = c.is_special();
                }
                // integer, same type: truncation commutes with min/max
                Some(c) if same => cv.t = c,
                // never selected: every window holds an array element
                None if same && p.minimum && cval > T::HIGHEST.to_f64() => cv.t = T::HIGHEST,
                None if same && !p.minimum && cval < T::LOWEST.to_f64() => cv.t = T::LOWEST,
                _ => return Ok(false),
            }
        }
    }
    let parallel = par::worth(par::work(n, 4, std::mem::size_of::<T>()));
    if !T::EXACT_F64 {
        // SAFETY: read-only view, nothing is written before this check
        let view = unsafe { std::slice::from_raw_parts(srcp, n) };
        if !py.detach(|| par::run(parallel, || ints_exact(view, constant.then_some(cv.t)))) {
            return Ok(false);
        }
    }
    let srcp = srcp as usize;
    if same {
        let dst = typed::<T>(output)?;
        let dstp = dst.data() as usize;
        let inplace = dstp == srcp;
        py.detach(|| {
            par::run(parallel, || unsafe {
                minmax1d::filter::<T>(
                    srcp as *const T,
                    dstp as *mut T,
                    shape,
                    axis,
                    p,
                    cv,
                    inplace,
                )
            })
        });
        Ok(true)
    } else {
        let mut tmp = vec![T::default(); n];
        let tp = tmp.as_mut_ptr() as usize;
        py.detach(|| {
            par::run(parallel, || unsafe {
                minmax1d::filter::<T>(srcp as *const T, tp as *mut T, shape, axis, p, cv, false)
            })
        });
        convert_to_output(py, &tmp, output, true, parallel)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_rank_filter, m)?)?;
    m.add_function(wrap_pyfunction!(py_min_or_max_filter, m)?)?;
    m.add_function(wrap_pyfunction!(py_min_or_max_filter1d, m)?)?;
    Ok(())
}
