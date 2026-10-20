//! The one thread pool every kernel uses.
//!
//! It is built lazily and rebuilt when the thread count changes or the process
//! id changes. A forked child has none of the parent's worker threads, so the
//! parent's pool is leaked there rather than joined.

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct Slot {
    pid: u32,
    threads: AtomicUsize,
    pool: Mutex<Option<Arc<rayon::ThreadPool>>>,
}

// Published slots live for the process lifetime. A forked child replaces the
// entire slot before locking anything; the inherited mutex may be locked by
// a parent thread that no longer exists in the child.
static SLOT: AtomicPtr<Slot> = AtomicPtr::new(std::ptr::null_mut());

fn default_threads() -> usize {
    std::env::var("FASTSCIPY_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

fn slot() -> &'static Slot {
    let pid = std::process::id();
    loop {
        let previous = SLOT.load(Ordering::Acquire);
        if !previous.is_null() {
            // SAFETY: published slots are never freed, and pid is immutable.
            let current = unsafe { &*previous };
            if current.pid == pid {
                return current;
            }
        }
        let threads = if previous.is_null() {
            default_threads()
        } else {
            // The budget is readable without the inherited pool mutex.
            unsafe { &*previous }.threads.load(Ordering::Relaxed)
        };
        let fresh = Box::into_raw(Box::new(Slot {
            pid,
            threads: AtomicUsize::new(threads),
            pool: Mutex::new(None),
        }));
        match SLOT.compare_exchange(previous, fresh, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                // Leak the old slot only in the forked child. In particular,
                // never join its inherited Rayon workers or lock its mutex.
                return unsafe { &*fresh };
            }
            Err(_) => {
                // SAFETY: this unpublished allocation was never accessible
                // through SLOT, so another initializer cannot be using it.
                drop(unsafe { Box::from_raw(fresh) });
            }
        }
    }
}

/// The shared pool. Kernels run parallel work with `pool().install(|| ...)`.
pub fn pool() -> Arc<rayon::ThreadPool> {
    let s = slot();
    let mut pool = s.pool.lock().unwrap_or_else(|e| e.into_inner());
    let threads = s.threads.load(Ordering::Relaxed);
    pool.get_or_insert_with(|| {
        Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("fastscipy-{i}"))
                .build()
                .expect("failed to build thread pool"),
        )
    })
    .clone()
}

/// Threads kernels may use. 1 means run serially.
pub fn num_threads() -> usize {
    slot().threads.load(Ordering::Relaxed)
}

#[pyfunction]
fn set_num_threads(threads: usize) -> PyResult<()> {
    if threads == 0 {
        return Err(PyValueError::new_err("threads must be >= 1"));
    }
    let s = slot();
    let mut pool = s.pool.lock().unwrap_or_else(|e| e.into_inner());
    if s.threads.swap(threads, Ordering::Relaxed) != threads {
        *pool = None;
    }
    Ok(())
}

#[pyfunction]
fn get_num_threads() -> usize {
    num_threads()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    Ok(())
}
