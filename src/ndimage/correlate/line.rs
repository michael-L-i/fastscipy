//! `correlate1d` and `uniform_filter1d`: filters along one axis.
//!
//! The array is viewed as `(outer, len, inner)` with the filtered axis in the
//! middle. Every output line depends only on its own input line, extended at
//! both ends as `NI_ExtendLine` does.
//!
//! * `inner > 1` (non-last axis): a column strip of every `(len, inner)` slab
//!   is processed top to bottom, keeping the extended input rows it still
//!   needs, converted to double, in a ring buffer; output rows are computed
//!   two at a time with one SIMD lane per column. With several threads, rows
//!   are also split into blocks.
//! * `inner == 1` (last axis): each line is copied into an extended double
//!   buffer, then correlated with one lane per output position. The uniform
//!   filter's running sum is sequential along a line, so groups of lines are
//!   advanced in lockstep, two lines per vector.
//!
//! Within a strip or line, every input row is read into the buffers before
//! the output row with the same index is written, so `output` may be `input`
//! (in place), which is how SciPy chains passes in `gaussian_filter`,
//! `uniform_filter` and `sobel`. Row blocks that run concurrently in place
//! first copy the rows they need from other blocks (see `snapshot`).

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;

use super::cast::{self, LoadFn, Ptr, Raw, StoreFn, for_each_task};
use super::ext::{self, CONST};
use super::kern::{self, Prog};
use super::simd::Lane;
use crate::ndimage::common::Mode;

/// The filter applied along the axis.
enum Op {
    Correlate(Prog),
    Uniform(usize),
}

impl Op {
    fn size(&self) -> usize {
        match self {
            Op::Correlate(p) => p.fs,
            Op::Uniform(n) => *n,
        }
    }
}

struct Job {
    src: Raw,
    dst: Raw,
    load: LoadFn,
    store: StoreFn,
    outer: usize,
    len: usize,
    inner: usize,
    /// Source index (or CONST) of each of the `len + fs - 1` extended positions.
    map: ext::ExtMap,
    before: usize,
    cval: f64,
    /// Output is the input (same buffer and layout).
    inplace: bool,
    op: Op,
}

impl Job {
    fn lext(&self) -> usize {
        self.map.lext()
    }

    /// Rough element-operation count, to size the parallelism.
    fn work(&self) -> usize {
        self.outer * self.len * self.inner * (self.op.size().min(64) + 4)
    }

    /// Element pointer of input row `row` of slab `o`, column `c`.
    #[inline(always)]
    fn src_at(&self, o: usize, row: usize, c: usize) -> *const u8 {
        self.src.at((o * self.len + row) * self.inner + c)
    }

    #[inline(always)]
    fn dst_at(&self, o: usize, row: usize, c: usize) -> *mut u8 {
        self.dst.at((o * self.len + row) * self.inner + c)
    }
}

/// Largest filter handled along the last axis (line buffers of len + size).
const MAX_LINE_WINDOW: i64 = 1 << 24;
/// Largest filter handled along other axes (each thread keeps size + 1 rows
/// of a strip).
const MAX_STRIP_WINDOW: i64 = 1 << 15;

/// Validate arguments and set up a job, or None to decline.
#[allow(clippy::too_many_arguments)]
fn setup(
    py: Python<'_>,
    input: &Bound<'_, PyUntypedArray>,
    axis: i64,
    output: &Bound<'_, PyUntypedArray>,
    mode: i64,
    cval: f64,
    origin: i64,
    op: Op,
) -> Option<Job> {
    let src = cast::raw(py, input, false)?;
    let dst = cast::raw(py, output, true)?;
    let shape = input.shape();
    if shape != output.shape() || shape.is_empty() {
        return None;
    }
    let ndim = shape.len();
    if axis < 0 || axis as usize >= ndim {
        return None;
    }
    let axis = axis as usize;
    let size: usize = shape.iter().product();
    if size == 0 {
        return None;
    }
    // In place is fine (see module docs); any other overlap is declined.
    if src.overlaps(&dst) && (src.ptr != dst.ptr || src.dt != dst.dt) {
        return None;
    }
    let mode = Mode::from_code(mode)?;
    let fs = op.size() as i64;
    // Huge windows: the buffers would not fit (SciPy then raises MemoryError
    // or takes its time); leave them to SciPy.
    let inner: usize = shape[axis + 1..].iter().product();
    if fs > MAX_LINE_WINDOW || (inner > 1 && fs > MAX_STRIP_WINDOW) {
        return None;
    }
    let size1 = fs / 2;
    let size2 = fs - size1 - 1;
    // Validated by SciPy's wrappers; out-of-range origins would make the C
    // code read outside its buffers.
    if origin < -size1 || origin > size2 {
        return None;
    }
    let before = (size1 + origin) as usize;
    let after = (size2 - origin) as usize;
    let len = shape[axis];
    let outer: usize = shape[..axis].iter().product();
    let map = ext::ExtMap::new(len, before, after, mode);
    Some(Job {
        src,
        dst,
        load: cast::loader(src.dt),
        store: cast::store_line(dst.dt),
        outer,
        len,
        inner,
        map,
        before,
        cval,
        inplace: src.ptr == dst.ptr,
        op,
    })
}

fn run(py: Python<'_>, job: Job) {
    py.detach(|| {
        if job.inner == 1 {
            match &job.op {
                Op::Correlate(p) => last_axis_correlate(&job, p),
                Op::Uniform(n) => last_axis_uniform(&job, *n),
            }
        } else {
            strips(&job)
        }
    });
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn corr_correlate1d(
    py: Python<'_>,
    input: &Bound<'_, PyUntypedArray>,
    weights: &Bound<'_, PyUntypedArray>,
    axis: i64,
    output: &Bound<'_, PyUntypedArray>,
    mode: i64,
    cval: f64,
    origin: i64,
    scheme: u32,
) -> PyResult<bool> {
    let wr = match cast::raw(py, weights, false) {
        Some(w) if w.dt == cast::DType::F64 && weights.ndim() == 1 => w,
        _ => return Ok(false),
    };
    // SciPy reads the weights while writing the output; don't emulate that.
    if cast::raw(py, output, false).is_some_and(|o| o.overlaps(&wr)) {
        return Ok(false);
    }
    let w = unsafe { std::slice::from_raw_parts(wr.ptr as *const f64, weights.len()) };
    let Some(prog) = Prog::new(w, scheme) else {
        return Ok(false);
    };
    let Some(job) = setup(
        py,
        input,
        axis,
        output,
        mode,
        cval,
        origin,
        Op::Correlate(prog),
    ) else {
        return Ok(false);
    };
    run(py, job);
    Ok(true)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn corr_uniform_filter1d(
    py: Python<'_>,
    input: &Bound<'_, PyUntypedArray>,
    size: i64,
    axis: i64,
    output: &Bound<'_, PyUntypedArray>,
    mode: i64,
    cval: f64,
    origin: i64,
) -> PyResult<bool> {
    if size < 1 {
        return Ok(false);
    }
    let Some(job) = setup(
        py,
        input,
        axis,
        output,
        mode,
        cval,
        origin,
        Op::Uniform(size as usize),
    ) else {
        return Ok(false);
    };
    run(py, job);
    Ok(true)
}

// ---------------------------------------------------------------------------
// Non-last axis: column strips with a ring buffer of extended rows.

/// Ring slots: extended row q lives in slot q % P, and the `fs + 1`
/// consecutive rows a step needs are all resident.
fn ring_rows(fs: usize) -> usize {
    fs + 1
}

/// Bytes of ring buffer per strip.
const L1_BUDGET: usize = 64 * 1024;

/// Rows to prefetch ahead in a strip.
const PREFETCH_ROWS: usize = 8;

/// Output rows `r0..r1` of slab `o`. When the output is the input and the
/// slab is split into several blocks, `snap` holds the input rows this block
/// reads but other blocks write ("foreign" rows), copied before any block
/// starts writing.
struct Block {
    o: usize,
    r0: usize,
    r1: usize,
    snap: Option<Snapshot>,
}

struct Snapshot {
    /// Sorted source rows held, and their bytes (`inner` elements each).
    rows: Vec<usize>,
    data: Vec<u8>,
}

impl Block {
    /// Pointer to element `c` of source row `m`.
    fn src_row(&self, job: &Job, m: usize, c: usize) -> *const u8 {
        if let Some(s) = &self.snap
            && (m < self.r0 || m >= self.r1)
        {
            let k = s.rows.binary_search(&m).expect("foreign row in snapshot");
            let isz = job.src.dt.size();
            return s.data[(k * job.inner + c) * isz..].as_ptr();
        }
        job.src_at(self.o, m, c)
    }
}

fn strips(job: &Job) {
    let fs = job.op.size();
    let threads = cast::threads_for(job.work());
    let mut w = ((L1_BUDGET / 8) / ring_rows(fs)).clamp(32, 1024) & !7;
    let mut nblocks = 1;
    if threads > 1 {
        let want = 4 * threads;
        match job.op {
            // A running sum must start at row 0: split columns only.
            Op::Uniform(_) => {
                while w > 32 && job.outer * job.inner.div_ceil(w) < want {
                    w /= 2;
                }
            }
            // Correlations can also split rows (wider strips access memory better).
            Op::Correlate(_) => {
                let tasks = job.outer * job.inner.div_ceil(w);
                if tasks < want {
                    nblocks = want.div_ceil(tasks).min(job.len / (2 * fs).max(16)).max(1);
                }
                while w > 32 && job.outer * job.inner.div_ceil(w) * nblocks < want {
                    w /= 2;
                }
            }
        }
    }
    let w = w.min(job.inner);
    let nstrips = job.inner.div_ceil(w);
    let per = job.len.div_ceil(nblocks);
    let nblocks = job.len.div_ceil(per);
    let mut blocks: Vec<Block> = (0..job.outer * nblocks)
        .map(|k| {
            let (o, b) = (k / nblocks, k % nblocks);
            Block {
                o,
                r0: b * per,
                r1: ((b + 1) * per).min(job.len),
                snap: None,
            }
        })
        .collect();
    if job.inplace && nblocks > 1 {
        snapshot(job, &mut blocks, threads);
    }
    for_each_task(
        blocks.len() * nstrips,
        threads,
        || Scratch::new(ring_rows(fs), w, fs),
        |scr, task| {
            let blk = &blocks[task / nstrips];
            let c0 = (task % nstrips) * w;
            let cw = w.min(job.inner - c0);
            unsafe { strip(job, blk, scr, c0, cw) };
        },
    );
}

/// Copy, for every block, the source rows it reads but does not own.
fn snapshot(job: &Job, blocks: &mut [Block], threads: usize) {
    let fs = job.op.size();
    let isz = job.src.dt.size();
    let row_bytes = job.inner * isz;
    let bp = cast::Ptr(blocks.as_mut_ptr());
    for_each_task(
        blocks.len(),
        threads,
        || (),
        |_, k| {
            let bp = bp;
            let blk = unsafe { &mut *bp.0.add(k) };
            let mut rows: Vec<usize> = (blk.r0..blk.r1 + fs - 1)
                .filter_map(|q| {
                    let m = job.map.get(q);
                    (m != CONST && ((m as usize) < blk.r0 || m as usize >= blk.r1))
                        .then_some(m as usize)
                })
                .collect();
            rows.sort_unstable();
            rows.dedup();
            let mut data = vec![0u8; rows.len() * row_bytes];
            for (i, &m) in rows.iter().enumerate() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        job.src_at(blk.o, m, 0),
                        data.as_mut_ptr().add(i * row_bytes),
                        row_bytes,
                    )
                };
            }
            blk.snap = Some(Snapshot { rows, data });
        },
    );
}

struct Scratch {
    ring: Vec<f64>,
    tail: Vec<f64>,
    out: Vec<f64>,
    tmp: Vec<f64>,
    ptrs: Vec<*const f64>,
}

impl Scratch {
    fn new(slots: usize, w: usize, fs: usize) -> Scratch {
        Scratch {
            ring: vec![0.0; slots * w],
            tail: vec![0.0; fs * w],
            out: vec![0.0; 2 * w],
            tmp: vec![0.0; w],
            ptrs: vec![std::ptr::null(); slots],
        }
    }
}

/// Process columns `c0..c0+cw` of output rows `blk.r0..blk.r1`.
///
/// Output row `i` needs extended rows `i..i+fs`; extended row `q` lives in
/// ring slot `q % P` (P = ring_rows), and the kernels get the rows of a step
/// as a pointer table.
unsafe fn strip(job: &Job, blk: &Block, scr: &mut Scratch, c0: usize, cw: usize) {
    let w = cw;
    let fs = job.op.size();
    let pr = ring_rows(fs);
    let o = blk.o;
    let (r0, r1) = (blk.r0, blk.r1);
    let qend = r1 + fs - 1; // extended rows r0..qend are needed
    let before = job.before;
    let tail0 = (before + job.len).max(r0);
    let ringp = scr.ring.as_mut_ptr();
    let tailp = scr.tail.as_mut_ptr();
    let isz = job.src.dt.size();
    let osz = job.dst.dt.size();
    // Fill extended row `q` into `dst` from the input (or cval).
    let fill = |q: usize, dst: *mut f64| unsafe {
        let m = job.map.get(q);
        if m == CONST {
            std::slice::from_raw_parts_mut(dst, w).fill(job.cval);
        } else {
            (job.load)(blk.src_row(job, m as usize, c0), w, dst);
        }
    };
    // Rows past the end of the line may map to rows that are overwritten by
    // then (in place); read them first.
    for q in tail0..qend {
        fill(q, unsafe { tailp.add((q - tail0) * w) });
    }
    let mut next = r0; // next extended row to load
    let mut next_slot = r0 % pr;
    let mut ensure = |upto: usize| {
        while next < upto {
            let dst = unsafe { ringp.add(next_slot * w) };
            if next >= tail0 {
                unsafe { std::ptr::copy_nonoverlapping(tailp.add((next - tail0) * w), dst, w) };
            } else {
                let ahead = next + PREFETCH_ROWS;
                if ahead < tail0 && ahead >= before {
                    cast::prefetch_read(job.src_at(o, ahead - before, c0), w * isz);
                }
                fill(next, dst);
            }
            next += 1;
            next_slot += 1;
            if next_slot == pr {
                next_slot = 0;
            }
        }
    };
    let ptrs = scr.ptrs.as_mut_ptr();
    // Point ptrs[t] at extended row `first + t`, for t <= fs.
    let window = |first: usize| unsafe {
        let s = first % pr;
        for t in 0..pr {
            let slot = if s + t < pr { s + t } else { s + t - pr };
            *ptrs.add(t) = ringp.add(slot * w);
        }
        kern::Ptrs(ptrs)
    };
    let out0 = scr.out.as_mut_ptr();
    let out1 = unsafe { out0.add(w) };
    let store = |i: usize, src: *const f64| unsafe {
        if i + PREFETCH_ROWS < r1 {
            cast::prefetch_write(job.dst_at(o, i + PREFETCH_ROWS, c0), w * osz);
        }
        (job.store)(src, w, job.dst_at(o, i, c0));
    };
    match &job.op {
        Op::Correlate(p) => {
            let mut i = r0;
            while i + 2 <= r1 {
                ensure(i + fs + 1);
                unsafe {
                    let rows = window(i);
                    if !kern::correlate2_spec(p, rows, w, out0, out1) {
                        kern::correlate2(p, rows, w, out0, out1);
                    }
                }
                store(i, out0);
                store(i + 1, out1);
                i += 2;
            }
            if i < r1 {
                ensure(i + fs);
                unsafe { kern::correlate(p, window(i), w, out0) };
                store(i, out0);
            }
        }
        Op::Uniform(n) => {
            debug_assert!(r0 == 0 && r1 == job.len);
            let size = *n as f64;
            ensure(fs);
            unsafe {
                kern::uniform_init(window(0), fs, w, &mut scr.tmp);
                kern::uniform_out(&scr.tmp[..w], size, out0);
            }
            store(0, out0);
            // Step i adds row i-1+fs and drops row i-1.
            let (mut leave, mut enter) = (0usize, fs % pr);
            for i in 1..r1 {
                ensure(i + fs);
                unsafe {
                    kern::uniform_step(
                        ringp.add(enter * w),
                        ringp.add(leave * w),
                        w,
                        &mut scr.tmp,
                        size,
                        out0,
                    )
                };
                store(i, out0);
                leave = if leave + 1 == pr { 0 } else { leave + 1 };
                enter = if enter + 1 == pr { 0 } else { enter + 1 };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Last axis.

/// Copy line `l` into `ext` (length lext), extended per the map.
unsafe fn fill_line(job: &Job, l: usize, ext: &mut [f64]) {
    let before = job.before;
    let len = job.len;
    unsafe { (job.load)(job.src_at(l, 0, 0), len, ext.as_mut_ptr().add(before)) };
    for q in (0..before).chain(before + len..job.lext()) {
        let m = job.map.get(q);
        ext[q] = if m == CONST {
            job.cval
        } else {
            ext[before + m as usize]
        };
    }
}

fn last_axis_correlate(job: &Job, p: &Prog) {
    let lines = job.outer;
    let len = job.len;
    let lext = job.lext();
    let threads = cast::threads_for(job.work());
    if lines >= 4 * threads || len < 4096 {
        let per = (lines / (8 * threads)).clamp(1, 64);
        for_each_task(
            lines.div_ceil(per),
            threads,
            || (vec![0.0f64; lext], vec![0.0f64; len]),
            |(ext, out), chunk| {
                for l in chunk * per..((chunk + 1) * per).min(lines) {
                    unsafe {
                        fill_line(job, l, ext);
                        let rows = kern::Strided {
                            base: ext.as_ptr(),
                            stride: 1,
                        };
                        if !kern::correlate_spec(p, rows, len, out.as_mut_ptr()) {
                            kern::correlate(p, rows, len, out.as_mut_ptr());
                        }
                        (job.store)(out.as_ptr(), len, job.dst_at(l, 0, 0));
                    }
                }
            },
        );
    } else {
        // Few long lines: split each line's outputs across threads.
        let mut ext = vec![0.0f64; lext];
        const CH: usize = 2048;
        for l in 0..lines {
            unsafe { fill_line(job, l, &mut ext) };
            let e = Ptr(ext.as_mut_ptr());
            for_each_task(
                len.div_ceil(CH),
                threads,
                || vec![0.0f64; CH],
                |out, k| {
                    let a = k * CH;
                    let n = CH.min(len - a);
                    let e = e;
                    unsafe {
                        let rows = kern::Strided {
                            base: e.0.add(a),
                            stride: 1,
                        };
                        if !kern::correlate_spec(p, rows, n, out.as_mut_ptr()) {
                            kern::correlate(p, rows, n, out.as_mut_ptr());
                        }
                        (job.store)(out.as_ptr(), n, job.dst_at(l, 0, a));
                    }
                },
            );
        }
    }
}

/// Lines filtered together by the last-axis uniform filter.
const R: usize = 8;

/// The running sum of a line is one long dependency chain, so `R` lines are
/// advanced in lockstep to overlap their chains.
fn last_axis_uniform(job: &Job, fs: usize) {
    let lines = job.outer;
    let len = job.len;
    let lext = job.lext();
    let size = fs as f64;
    let groups = lines.div_ceil(R);
    for_each_task(
        groups,
        cast::threads_for(job.work()),
        || {
            // Buffers for a group of R lines, or one line if there are fewer.
            let nb = if lines >= R { R } else { 1 };
            (vec![0.0f64; lext * nb], vec![0.0f64; len * nb])
        },
        |(ext, out), g| {
            let l0 = g * R;
            let nl = R.min(lines - l0);
            if nl < R {
                // Leftover lines, one at a time.
                for l in l0..l0 + nl {
                    unsafe {
                        fill_line(job, l, &mut ext[..lext]);
                        uniform_line(&ext[..lext], &mut out[..len], fs, size);
                        (job.store)(out.as_ptr(), len, job.dst_at(l, 0, 0));
                    }
                }
                return;
            }
            for r in 0..R {
                unsafe { fill_line(job, l0 + r, &mut ext[r * lext..(r + 1) * lext]) };
            }
            let ep: [*const f64; R] = std::array::from_fn(|r| ext[r * lext..].as_ptr());
            let op: [*mut f64; R] = std::array::from_fn(|r| out[r * len..].as_mut_ptr());
            unsafe { uniform_lines(&ep, &op, fs, len, size) };
            for r in 0..R {
                unsafe { (job.store)(op[r], len, job.dst_at(l0 + r, 0, 0)) };
            }
        },
    );
}

/// `NI_UniformFilter1D`'s loop on one extended line.
fn uniform_line(ext: &[f64], out: &mut [f64], fs: usize, size: f64) {
    let mut tmp = 0.0f64;
    for &v in &ext[..fs] {
        tmp = tmp.add_ordered(v);
    }
    out[0] = tmp / size;
    let n = out.len();
    // Step i adds ext[i - 1 + fs] and drops ext[i - 1].
    let (enter, leave) = (&ext[fs..fs + n - 1], &ext[..n - 1]);
    for ((o, &e), &l) in out[1..].iter_mut().zip(enter).zip(leave) {
        tmp = tmp.add_ordered(e - l);
        *o = tmp / size;
    }
}

/// `NI_UniformFilter1D`'s loop on `R` extended lines at once.
///
/// Lines are paired into the two lanes of a vector (so the divisions, the
/// slowest operation, run two at a time); steps are taken two at a time so
/// the pairing is a 2x2 transpose of contiguous loads.
#[inline(never)]
unsafe fn uniform_lines(
    ext: &[*const f64; R],
    out: &[*mut f64; R],
    fs: usize,
    len: usize,
    size: f64,
) {
    use super::simd::{Lane, V, zip};
    const P: usize = R / 2;
    unsafe {
        let mut tmp = [V::splat(0.0); P];
        for q in 0..fs {
            for p in 0..P {
                let values = [*ext[2 * p].add(q), *ext[2 * p + 1].add(q)];
                tmp[p] = tmp[p].add_ordered(V::load(values.as_ptr()));
            }
        }
        let mut t = [0.0f64; R];
        for p in 0..P {
            tmp[p].store(t.as_mut_ptr().add(2 * p));
        }
        for r in 0..R {
            *out[r] = t[r] / size;
        }
        let vsize = V::splat(size);
        let mut i = 1;
        while i + 2 <= len {
            for p in 0..P {
                let (r0, r1) = (2 * p, 2 * p + 1);
                // Steps i and i + 1: enter ext[i-1+fs], ext[i+fs]; leave ext[i-1], ext[i].
                let (e0, e1) = zip(
                    V::load(ext[r0].add(i - 1 + fs)),
                    V::load(ext[r1].add(i - 1 + fs)),
                );
                let (l0, l1) = zip(V::load(ext[r0].add(i - 1)), V::load(ext[r1].add(i - 1)));
                tmp[p] = tmp[p].add_ordered(e0.sub(l0));
                let o0 = tmp[p].div(vsize);
                tmp[p] = tmp[p].add_ordered(e1.sub(l1));
                let o1 = tmp[p].div(vsize);
                let (a, b) = zip(o0, o1);
                a.store(out[r0].add(i));
                b.store(out[r1].add(i));
            }
            i += 2;
        }
        if i < len {
            let mut s = [0.0f64; R];
            for p in 0..P {
                tmp[p].store(s.as_mut_ptr().add(2 * p));
            }
            for r in 0..R {
                s[r] = s[r].add_ordered(*ext[r].add(i - 1 + fs) - *ext[r].add(i - 1));
                *out[r].add(i) = s[r] / size;
            }
        }
    }
}
