//! Parallel loops sized by work, with per-thread reusable buffers.
//!
//! Handing work to the thread pool costs tens of microseconds (waking
//! workers), more than SciPy needs for a small image. `run(false, f)` runs `f`
//! on the calling thread with every loop of this module serial; `run(true, f)`
//! runs it in the shared pool. Loops split their items into chunks of a
//! minimum amount of work, so mid-sized inputs wake only a few threads, and
//! buffers come from thread-local arenas instead of per-call allocations.

use std::cell::{Cell, RefCell};

use rayon::prelude::*;

thread_local! {
    static SERIAL: Cell<bool> = const { Cell::new(false) };
    static ARENAS: RefCell<Arenas> = RefCell::new(Arenas::default());
}

/// Work is counted in 16-byte vector operations (see `work`).
/// Below this, a call runs on the calling thread (~100-200 us of work; waking
/// pool threads costs tens of microseconds each).
const PARALLEL_MIN_WORK: usize = 1 << 20;
/// Work per parallel chunk (~100 us): short calls wake few threads.
const CHUNK_WORK: usize = 1 << 19;

/// Work estimate in vector operations: `elems` elements of `elem_size` bytes,
/// `ops` operations each.
pub fn work(elems: usize, ops: usize, elem_size: usize) -> usize {
    elems.saturating_mul(ops.max(1)).saturating_mul(elem_size) / 16 + 1
}

/// Whether `work` (vector operations) is worth the thread pool.
pub fn worth(work: usize) -> bool {
    work >= PARALLEL_MIN_WORK && crate::ndimage::common::num_threads() > 1
}

/// Run `f` in the pool (`parallel`) or serially on this thread.
pub fn run<R: Send>(parallel: bool, f: impl FnOnce() -> R + Send) -> R {
    if parallel {
        crate::ndimage::common::pool().install(f)
    } else {
        let prev = SERIAL.with(|s| s.replace(true));
        let r = f();
        SERIAL.with(|s| s.set(prev));
        r
    }
}

fn serial() -> bool {
    SERIAL.with(|s| s.get())
}

/// A grow-only buffer handing out slices of plain-data types.
#[derive(Default)]
pub struct Arena(Vec<u128>);

impl Arena {
    /// `n` elements of `U`, with unspecified (but initialized) contents.
    /// `U` must be valid for every bit pattern: numbers, SIMD vectors, and
    /// arrays / tuples of these.
    pub fn take<U: Copy>(&mut self, n: usize) -> &mut [U] {
        assert!(std::mem::align_of::<U>() <= 16);
        let words = (n * std::mem::size_of::<U>()).div_ceil(16);
        if self.0.len() < words {
            self.0.resize(words, 0);
        }
        // SAFETY: the buffer is initialized, 16-byte aligned and large enough;
        // U accepts any bit pattern (caller contract).
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr() as *mut U, n) }
    }
}

/// Per-thread buffers for one work item.
#[derive(Default)]
pub struct Arenas {
    pub a: Arena,
    pub b: Arena,
    pub c: Arena,
    pub d: Arena,
}

/// `f(i, arenas)` for i in 0..n, in chunks of at least `min_items` items.
pub fn for_each<F>(n: usize, min_items: usize, f: F)
where
    F: Fn(usize, &mut Arenas) + Sync + Send,
{
    let per = min_items.max(1);
    let nchunks = n.div_ceil(per);
    let body = |c: usize| {
        ARENAS.with(|a| {
            let mut a = a.borrow_mut();
            for i in c * per..((c + 1) * per).min(n) {
                f(i, &mut a);
            }
        })
    };
    if serial() || nchunks <= 1 {
        (0..nchunks).for_each(body);
    } else {
        (0..nchunks).into_par_iter().for_each(body);
    }
}

/// Items per chunk for items of `item_work` element-operations each.
pub fn items_per_chunk(item_work: usize) -> usize {
    CHUNK_WORK.div_ceil(item_work.max(1))
}

/// Map chunks of `data` and combine the results.
pub fn map_reduce<T: Sync, R: Send>(
    data: &[T],
    chunk: usize,
    map: impl Fn(&[T]) -> R + Sync + Send,
    identity: impl Fn() -> R + Sync + Send,
    reduce: impl Fn(R, R) -> R + Sync + Send,
) -> R {
    if serial() {
        data.chunks(chunk).map(&map).fold(identity(), &reduce)
    } else {
        data.par_chunks(chunk).map(map).reduce(identity, reduce)
    }
}

/// `f(dst_chunk, src_chunk)` over matching chunks of two slices.
pub fn zip_chunks<A: Send, B: Sync>(
    dst: &mut [A],
    src: &[B],
    chunk: usize,
    f: impl Fn(&mut [A], &[B]) + Sync + Send,
) {
    if serial() {
        for (d, s) in dst.chunks_mut(chunk).zip(src.chunks(chunk)) {
            f(d, s);
        }
    } else {
        dst.par_chunks_mut(chunk)
            .zip(src.par_chunks(chunk))
            .for_each(|(d, s)| f(d, s));
    }
}
