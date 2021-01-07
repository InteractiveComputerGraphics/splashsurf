use log::info;

/// Wrapper type to make any type Send + Sync
pub struct SendSyncWrapper<T>(T);

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
