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
    pub min_chunk_size: usize,
    pub chunks_per_cpu: usize,
    pub count_logical_cpus: bool,
}

impl Default for ParallelPolicy {
    fn default() -> Self {
        Self {
            min_chunk_size: 16,
            chunks_per_cpu: 4,
            count_logical_cpus: true,
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
        let min_chunk_size = parallel_policy.min_chunk_size.max(num_items);
        let chunks_per_cpu = parallel_policy.chunks_per_cpu;

        let num_cpus = if parallel_policy.count_logical_cpus {
            num_cpus::get()
        } else {
            num_cpus::get_physical()
        };

        let num_chunks = chunks_per_cpu * num_cpus;
        let chunk_size = (num_items / num_chunks).min(min_chunk_size);

        Self {
            num_items,
            num_chunks,
            chunk_size,
        }
    }

    pub(crate) fn log<S: AsRef<str>>(&self, item_name: S) {
        info!(
            "Splitting {} {} into {} chunks (with {} particles each) for density map generation",
            self.num_items,
            item_name.as_ref(),
            self.num_chunks,
            self.chunk_size
        );
    }

    pub(crate) fn with_log<S: AsRef<str>>(self, item_name: S) -> Self {
        self.log(item_name);
        self
    }
}
