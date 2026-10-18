//! The one thread pool every kernel uses.
//!
//! It is built lazily and rebuilt when the thread count changes or the process
//! id changes. A forked child has none of the parent's worker threads, so the
//! parent's pool is leaked there rather than joined.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct State {
    pid: u32,
    threads: usize,
    pool: Option<Arc<rayon::ThreadPool>>,
}

static STATE: Mutex<State> = Mutex::new(State { pid: 0, threads: 0, pool: None });

fn default_threads() -> usize {
    std::env::var("FASTSCIPY_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

fn state() -> std::sync::MutexGuard<'static, State> {
    let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if s.threads == 0 {
        s.threads = default_threads();
    }
    let pid = std::process::id();
    if s.pid != pid {
        std::mem::forget(s.pool.take());
        s.pid = pid;
    }
    s
}

/// The shared pool. Kernels run parallel work with `pool().install(|| ...)`.
pub fn pool() -> Arc<rayon::ThreadPool> {
    let mut s = state();
    let threads = s.threads;
    s.pool
        .get_or_insert_with(|| {
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
    state().threads
}

#[pyfunction]
fn set_num_threads(threads: usize) -> PyResult<()> {
    if threads == 0 {
        return Err(PyValueError::new_err("threads must be >= 1"));
    }
    let mut s = state();
    if s.threads != threads {
        s.threads = threads;
        s.pool = None;
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
