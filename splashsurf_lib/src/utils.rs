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
