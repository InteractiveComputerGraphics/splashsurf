//! Internal helper functions and types

use anyhow::{anyhow, Context};
use log::info;
use nalgebra::{SVector, Scalar};
use num_traits::Zero;
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::fmt::Debug;

/// Converts a slice of scalar values to a vector of the same length, returns an error if conversion fails
pub fn try_convert_scalar_slice<
    ScalarFrom: Copy + Debug,
    ScalarTo,
    F: Fn(ScalarFrom) -> Option<ScalarTo>,
>(
    values: &[ScalarFrom],
    f: F,
) -> Result<Vec<ScalarTo>, anyhow::Error> {
    values
        .iter()
        .copied()
        .map(|v| {
            f(v).ok_or_else(|| {
                anyhow!(
                    "failed to convert value {:?} from type {} to {}",
                    v,
                    std::any::type_name::<ScalarFrom>(),
                    std::any::type_name::<ScalarTo>()
                )
            })
        })
        .try_collect_with_capacity(values.len())
}

/// Converts a slice of scalar values to a vector of [`nalgebra::SVector`], returns an error if conversion fails or the input slice's length is not a multiple of the vector length.
pub fn try_convert_scalar_slice_to_vectors<
    const N: usize,
    ScalarFrom: Copy + Debug,
    ScalarTo: Scalar + Zero,
    F: Fn(ScalarFrom) -> Option<ScalarTo>,
>(
    values: &[ScalarFrom],
    f: F,
) -> Result<Vec<SVector<ScalarTo, N>>, anyhow::Error> {
    {
        if values.len() % N != 0 {
            Err(anyhow!("input slice length is not a multiple of {}", N))
        } else {
            values
                .chunks_exact(N)
                .map(|v| {
                    let mut v_out = SVector::zeros();
                    for i in 0..N {
                        v_out[i] = f(v[i]).ok_or_else(|| {
                            anyhow!(
                                "failed to convert value {:?} from type {} to {}",
                                v,
                                std::any::type_name::<ScalarFrom>(),
                                std::any::type_name::<ScalarTo>()
                            )
                        })?;
                    }
                    Ok(v_out)
                })
                .try_collect_with_capacity(values.len() / N)
        }
    }
    .context(anyhow!(
        "failed to convert scalar slice to vectors of length {}",
        N
    ))
}

#[cfg(test)]
mod tests {
    use num_traits::FromPrimitive;
    #[test]
    fn test_try_convert_scalar_slice() {
        let values = vec![1, -1];
        assert!(super::try_convert_scalar_slice(&values, u64::from_i32).is_err());
        let values = vec![1, -1];
        assert_eq!(
            super::try_convert_scalar_slice(&values, f32::from_i64).unwrap(),
            vec![1.0, -1.0]
        );
    }
}

/// "Convert" an empty vector to preserve allocated memory if size and alignment matches
/// See https://users.rust-lang.org/t/pattern-how-to-reuse-a-vec-str-across-loop-iterations/61657/5
/// See https://github.com/rust-lang/rfcs/pull/2802
#[allow(unused)]
pub(crate) fn recycle<A, B>(mut v: Vec<A>) -> Vec<B> {
    v.clear();
    v.into_iter().map(|_| unreachable!()).collect()
}

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

/// Useful extension methods for iterators
pub(crate) trait IteratorExt {
    /// Tries to collect the items of the iterator into a `Vec` that reserves the given capacity and stops as soon as an error is encountered
    ///
    /// Motivation: <https://github.com/rust-lang/rust/issues/48994>
    fn try_collect_with_capacity<T, E>(self, capacity: usize) -> Result<Vec<T>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>;
}

impl<Iter: Iterator> IteratorExt for Iter {
    fn try_collect_with_capacity<T, E>(mut self, capacity: usize) -> Result<Vec<T>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
    {
        self.try_fold(Vec::with_capacity(capacity), |mut vec, item| {
            vec.push(item?);
            Ok(vec)
        })
    }
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
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut_unchecked(&self, i: usize) -> &mut T {
        debug_assert!(i < self.len(), "index out of bounds");
        &mut *self.slice.get_unchecked(i).get()
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
