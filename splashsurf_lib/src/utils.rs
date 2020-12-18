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

pub(crate) struct ChunkSize {
    pub num_items: usize,
    pub num_chunks: usize,
    pub chunk_size: usize,
}

impl ChunkSize {
    pub(crate) fn new(num_items: usize) -> Self {
        let min_chunk_size = 100.max(num_items);
        let chunks_per_cpu = 10;

        let num_cpus = num_cpus::get();
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
            self.num_items, item_name.as_ref(), self.num_chunks, self.chunk_size
        );
    }

    pub(crate) fn with_log<S: AsRef<str>>(self, item_name: S) -> Self {
        self.log(item_name);
        self
    }
}
