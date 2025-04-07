//! Internal helper functions and types

use log::info;
use rayon::prelude::*;
use std::cell::UnsafeCell;

/// Macro version of Option::map that allows using e.g. using the ?-operator in the map expression
///
/// For example:
/// ```ignore
/// let four: f64 = map_option!(Some(2.0_f64), val => val? * 2.0);
/// ```
macro_rules! map_option {
    ($some_optional:expr, $value_identifier:ident => $value_transformation:expr) => {
        match $some_optional {
            Some($value_identifier) => Some($value_transformation),
            None => None,
        }
    };
}

/// Wrapper for unsafe shared mutable access to a slice, disjoint access has to be ensured separately
/// Implementation based on: <https://stackoverflow.com/a/65182786/929037>
#[derive(Copy, Clone)]
pub(crate) struct UnsafeSlice<'a, T> {
    slice: &'a [UnsafeCell<T>],
}

unsafe impl<T: Send + Sync> Send for UnsafeSlice<'_, T> {}
unsafe impl<T: Send + Sync> Sync for UnsafeSlice<'_, T> {}

impl<'a, T> UnsafeSlice<'a, T> {
    /// Wraps a slice to be able to share mutable access between threads
    pub fn new(slice: &'a mut [T]) -> Self {
        // SAFETY: `&mut` ensures unique access.
        //  See `Cell::from_mut`: https://doc.rust-lang.org/std/cell/struct.Cell.html#method.from_mut
        // SAFETY: `UnsafeCell<T>` has the same memory layout as `T`.
        //  See `Cell::as_slice_of_cells`: https://doc.rust-lang.org/std/cell/struct.Cell.html#method.as_slice_of_cells
        let ptr = slice as *mut [T] as *const [UnsafeCell<T>];
        Self {
            slice: unsafe { &*ptr },
        }
    }

    /// Returns the length of the wrapped slice
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Returns a mutable reference to an element of the wrapped slice without doing bounds checking, simultaneous access has to be disjoint!
    /// SAFETY: It is unsound to obtain two mutable references to the same index.
    /// SAFETY: Calling this method with an out-of-bounds index is undefined behavior.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut_unchecked(&self, i: usize) -> &mut T {
        debug_assert!(i < self.len(), "index out of bounds");
        unsafe { &mut *self.slice.get_unchecked(i).get() }
    }
}

/// Ensures that at least the specified total capacity is reserved for the given vector
pub(crate) fn reserve_total<T>(vec: &mut Vec<T>, total_capacity: usize) {
    if total_capacity > vec.capacity() {
        vec.reserve(total_capacity - vec.capacity());
    }
}

/// Resizes the given vector to the given length and fills new entries with `value.clone()`, parallel or sequential depending on runtime parameter
#[allow(unused)]
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

/// Resizes the given vector to the given length and fills new entries with `value.clone()`, sequential version
#[allow(unused)]
pub(crate) fn seq_resize_and_fill<T: Clone>(vec: &mut Vec<T>, new_len: usize, value: T) {
    let old_len = vec.len();
    vec.iter_mut()
        .take(old_len.min(new_len))
        .for_each(|v| *v = value.clone());
    vec.resize(new_len, value);
}

/// Resizes the given vector to the given length and fills new entries with `value.clone()`, parallel version
#[allow(unused)]
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

            (num_items / num_tasks).max(parallel_policy.min_task_size)
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
