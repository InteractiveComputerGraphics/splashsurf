//! Internal helper functions and types

use log::info;
use rayon::prelude::*;

/// Wrapper type to make any type Send + Sync
pub(crate) struct SendSyncWrapper<T>(T);

impl<T> SendSyncWrapper<T> {
    #[inline(always)]
    pub unsafe fn new(value: T) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub fn get(&self) -> &T {
        &self.0
    }
}

unsafe impl<T> Sync for SendSyncWrapper<T> {}
unsafe impl<T> Send for SendSyncWrapper<T> {}

/// Ensure that at least the specified total capacity is reserved for the given vector
pub(crate) fn reserve_total<T>(vec: &mut Vec<T>, total_capacity: usize) {
    if total_capacity > vec.capacity() {
        vec.reserve(total_capacity - vec.capacity());
    }
}

pub(crate) fn resize_and_fill<T: Clone + Send + Sync>(
    vec: &mut Vec<T>,
    new_len: usize,
    value: T,
    par: bool,
) {
    if par {
        par_resize_and_fill(vec, new_len, value);
    } else {
        seq_resize_and_fill(vec, new_len, value);
    }
}

pub(crate) fn seq_resize_and_fill<T: Clone>(vec: &mut Vec<T>, new_len: usize, value: T) {
    let old_len = vec.len();
    vec.iter_mut()
        .take(old_len.min(new_len))
        .for_each(|v| *v = value.clone());
    vec.resize(new_len, value);
}

pub(crate) fn par_resize_and_fill<T: Clone + Send + Sync>(
    vec: &mut Vec<T>,
    new_len: usize,
    value: T,
) {
    let old_len = vec.len();
    vec.par_iter_mut()
        .with_min_len(8)
        .take(old_len.min(new_len))
        .for_each(|v| *v = value.clone());
    vec.resize(new_len, value);
}

pub struct ParallelPolicy {
    pub min_task_size: usize,
    pub tasks_per_thread: usize,
}

impl Default for ParallelPolicy {
    fn default() -> Self {
        Self {
            min_task_size: 256,
            tasks_per_thread: 8,
        }
    }
}

pub(crate) struct ChunkSize {
    pub num_items: usize,
    pub num_chunks: usize,
    pub chunk_size: usize,
}

impl ChunkSize {
    pub(crate) fn new(parallel_policy: &ParallelPolicy, num_items: usize) -> Self {
        let num_threads = rayon::current_num_threads();

        // Chunks size for 1 chunk per thread
        let equal_distribution = num_items / num_threads;

        let chunk_size = if parallel_policy.min_task_size > equal_distribution {
            // Ensure that at every thread gets some data
            equal_distribution
        } else {
            // Ensure that there are the desired amount of tasks/chunks per thread
            let num_tasks = parallel_policy.tasks_per_thread * num_threads;
            let task_size = (num_items / num_tasks).max(parallel_policy.min_task_size);
            task_size
        }
        // Ensure that we don't have less than a minimum number of items per thread
        .max(16);

        // Compute the number of chunks needed
        let num_chunks = if num_items % chunk_size == 0 {
            num_items / chunk_size
        } else {
            num_items / chunk_size + 1
        };

        Self {
            num_items,
            num_chunks,
            chunk_size,
        }
    }

    pub(crate) fn log<S1: AsRef<str>, S2: AsRef<str>>(&self, item_name: S1, purpose: S2) {
        info!(
            "Splitting {} {} into {} chunks (with {} particles each) for {}",
            self.num_items,
            item_name.as_ref(),
            self.num_chunks,
            self.chunk_size,
            purpose.as_ref(),
        );
    }

    pub(crate) fn with_log<S1: AsRef<str>, S2: AsRef<str>>(
        self,
        item_name: S1,
        purpose: S2,
    ) -> Self {
        self.log(item_name, purpose);
        self
    }
}
