//! `scipy.signal._peak_finding_utils._peak_prominences` and `_peak_widths`.
//!
//! Prominence without a window is found by scanning outward from each peak, as
//! SciPy does, unless the scans would walk more than a few passes over the
//! signal (a random walk makes them quadratic); then two monotonic-stack sweeps
//! give every peak's bases in O(n). Ties go to the nearest index, as in the scan.
//! Both functions return `None` for input SciPy would raise on.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::pool;

type Prom = (f64, isize, isize);

/// SciPy's scan for one peak: `(prominence, left base, right base, steps walked)`.
fn scan(x: &[f64], peak: usize, wlen: isize) -> (Prom, usize) {
    let (n, p) = (x.len() as isize, peak as isize);
    let (mut lo, mut hi) = (0, n - 1);
    if wlen >= 2 {
        (lo, hi) = ((p - wlen / 2).max(0), (p + wlen / 2).min(n - 1));
    }
    let xp = x[peak];
    let (mut i, mut lb, mut lmin) = (p, p, xp);
    while lo <= i && x[i as usize] <= xp {
        if x[i as usize] < lmin {
            (lmin, lb) = (x[i as usize], i);
        }
        i -= 1;
    }
    let steps = (p - i) as usize;
    let (mut i, mut rb, mut rmin) = (p, p, xp);
    while i <= hi && x[i as usize] <= xp {
        if x[i as usize] < rmin {
            (rmin, rb) = (x[i as usize], i);
        }
        i += 1;
    }
    (
        (xp - if rmin > lmin { rmin } else { lmin }, lb, rb),
        steps + (i - p) as usize,
    )
}

/// For each slot-marked index, the minimum of `x` over the maximal run around it
/// (towards the sweep origin) on which `x <= x[i]`, and its nearest index.
fn sweep(x: &[f64], slot: &[u32], rev: bool) -> (Vec<f64>, Vec<isize>) {
    let n = x.len();
    let count = slot.iter().filter(|&&s| s != u32::MAX).count();
    let (mut mins, mut idx) = (vec![0.0; count], vec![0isize; count]);
    let mut stack: Vec<(f64, f64, usize)> = Vec::new(); // (x value, run minimum, its index)
    for t in 0..n {
        let i = if rev { n - 1 - t } else { t };
        let v = x[i];
        let (mut cmin, mut cidx) = (v, i);
        while let Some(&(xv, m, mi)) = stack.last() {
            if xv > v {
                break;
            }
            stack.pop();
            if m < cmin {
                (cmin, cidx) = (m, mi);
            }
        }
        if slot[i] != u32::MAX {
            mins[slot[i] as usize] = cmin;
            idx[slot[i] as usize] = cidx as isize;
        }
        stack.push((v, cmin, cidx));
    }
    (mins, idx)
}

fn sweeps(x: &[f64], peaks: &[isize]) -> Vec<Prom> {
    let mut slot = vec![u32::MAX; x.len()];
    let mut unique = 0u32;
    for &p in peaks {
        if slot[p as usize] == u32::MAX {
            slot[p as usize] = unique;
            unique += 1;
        }
    }
    let (left, right) = if pool::num_threads() > 1 && x.len() >= 1 << 16 {
        pool::pool().install(|| rayon::join(|| sweep(x, &slot, false), || sweep(x, &slot, true)))
    } else {
        (sweep(x, &slot, false), sweep(x, &slot, true))
    };
    peaks
        .iter()
        .map(|&p| {
            let s = slot[p as usize] as usize;
            let (l, r) = (left.0[s], right.0[s]);
            (
                x[p as usize] - if r > l { r } else { l },
                left.1[s],
                right.1[s],
            )
        })
        .collect()
}

/// Per-peak scans of at most `wlen` samples, on all threads when there is enough work.
fn windowed(
    x: &[f64],
    peaks: &[isize],
    wlen: isize,
    out: (&mut [f64], &mut [isize], &mut [isize]),
) {
    const BLOCK: usize = 1024;
    let (prom, left, right) = out;
    let block = |(b, ((p, l), r)): (usize, ((&mut [f64], &mut [isize]), &mut [isize]))| {
        for k in 0..p.len() {
            (p[k], l[k], r[k]) = scan(x, peaks[b * BLOCK + k] as usize, wlen).0;
        }
    };
    if pool::num_threads() > 1 && peaks.len() as isize * wlen >= 1 << 20 {
        pool::pool().install(|| {
            prom.par_chunks_mut(BLOCK)
                .zip(left.par_chunks_mut(BLOCK))
                .zip(right.par_chunks_mut(BLOCK))
                .enumerate()
                .for_each(block)
        });
    } else {
        prom.chunks_mut(BLOCK)
            .zip(left.chunks_mut(BLOCK))
            .zip(right.chunks_mut(BLOCK))
            .enumerate()
            .for_each(block);
    }
}

#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn peak_prominences<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    peaks: PyReadonlyArray1<'py, isize>,
    wlen: isize,
) -> PyResult<
    Option<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<isize>>,
        Bound<'py, PyArray1<isize>>,
    )>,
> {
    let (x, peaks) = (x.as_slice()?, peaks.as_slice()?);
    if peaks.iter().any(|&p| p < 0 || p as usize >= x.len()) {
        return Ok(None);
    }
    let m = peaks.len();
    let (prom, left, right) = (
        PyArray1::<f64>::zeros(py, m, false),
        PyArray1::<isize>::zeros(py, m, false),
        PyArray1::<isize>::zeros(py, m, false),
    );
    // SAFETY: freshly created, contiguous, and only written inside this call.
    let (p, l, r) = unsafe {
        (
            prom.as_slice_mut()?,
            left.as_slice_mut()?,
            right.as_slice_mut()?,
        )
    };
    py.detach(|| {
        if wlen >= 2 {
            return windowed(x, peaks, wlen, (p, l, r));
        }
        // A scan step costs about 2.5 ns, a sweep pass over the signal about 7 ns per sample: scanning
        // wins below about 5 steps per sample. Estimate the steps from a spread of peaks, and keep a
        // budget in case the estimate was low. The sweeps order NaN differently, so with NaN scan on.
        let stride = (peaks.len() / 4096).max(1);
        let sweep_wins = |steps: usize| steps > 5 * x.len() && !x.iter().any(|v| v.is_nan());
        let mut sample = 0;
        for &pk in peaks.iter().step_by(stride) {
            sample += scan(x, pk as usize, wlen).1;
            if sample * stride > 5 * x.len() {
                break; // the full estimate is at least this
            }
        }
        let mut use_sweeps = sweep_wins(sample * stride);
        let mut steps = 0;
        if !use_sweeps {
            for (k, &pk) in peaks.iter().enumerate() {
                let (res, s) = scan(x, pk as usize, wlen);
                (p[k], l[k], r[k]) = res;
                steps += s;
                if steps > 12 * x.len() && sweep_wins(steps) {
                    use_sweeps = true;
                    break;
                }
            }
        }
        if use_sweeps {
            for (k, res) in sweeps(x, peaks).into_iter().enumerate() {
                (p[k], l[k], r[k]) = res;
            }
        }
    });
    Ok(Some((prom, left, right)))
}

fn width(x: &[f64], peak: usize, rel: f64, prom: f64, lo: usize, hi: usize) -> [f64; 4] {
    let height = (-prom).mul_add(rel, x[peak]); // clang fuses x - prom * rel
    let mut i = peak;
    while lo < i && height < x[i] {
        i -= 1;
    }
    let mut left = i as f64;
    if x[i] < height && x[i + 1] != x[i] {
        left += (height - x[i]) / (x[i + 1] - x[i]);
    }
    let mut i = peak;
    while i < hi && height < x[i] {
        i += 1;
    }
    let mut right = i as f64;
    if x[i] < height && x[i - 1] != x[i] {
        right -= (height - x[i]) / (x[i - 1] - x[i]);
    }
    [right - left, height, left, right]
}

#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn peak_widths<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    peaks: PyReadonlyArray1<'py, isize>,
    rel_height: f64,
    prominences: PyReadonlyArray1<'py, f64>,
    left: PyReadonlyArray1<'py, isize>,
    right: PyReadonlyArray1<'py, isize>,
) -> PyResult<Option<[Bound<'py, PyArray1<f64>>; 4]>> {
    let (x, peaks, prom) = (x.as_slice()?, peaks.as_slice()?, prominences.as_slice()?);
    let (left, right) = (left.as_slice()?, right.as_slice()?);
    let ok = |p: usize| {
        let (l, pk, r) = (left[p], peaks[p], right[p]);
        0 <= l && l <= pk && pk <= r && (r as usize) < x.len()
    };
    if [prom.len(), left.len(), right.len()]
        .iter()
        .any(|&n| n != peaks.len())
        || !(0..peaks.len()).all(ok)
    {
        return Ok(None);
    }
    let one = |p: usize| {
        width(
            x,
            peaks[p] as usize,
            rel_height,
            prom[p],
            left[p] as usize,
            right[p] as usize,
        )
    };
    let rows: Vec<[f64; 4]> = py.detach(|| {
        if pool::num_threads() > 1 && peaks.len() >= 1 << 15 {
            pool::pool().install(|| {
                (0..peaks.len())
                    .into_par_iter()
                    .with_min_len(4096)
                    .map(one)
                    .collect()
            })
        } else {
            (0..peaks.len()).map(one).collect()
        }
    });
    Ok(Some([0, 1, 2, 3].map(|k| {
        PyArray1::from_iter(py, rows.iter().map(|r| r[k]))
    })))
}
