use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicU64, Ordering};

/// Wraps a global allocator and counts the currently and peak allocated memory
pub struct CountingAllocator<A: GlobalAlloc> {
    pub allocator: A,
    pub current_allocation: AtomicU64,
    pub peak_allocation: AtomicU64,
}

// TODO: Is Ordering::AcqRel ok to use?

unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAllocator<A> {
    /// Allocates memory and counts the number of bytes allocated (current and peak allocation).
    /// Safety: See [`GlobalAlloc::alloc`].
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = unsafe { self.allocator.alloc(layout) };
        if !ret.is_null() {
            let old_allocation = self
                .current_allocation
                .fetch_add(layout.size() as u64, Ordering::AcqRel);
            let current_allocation = old_allocation + layout.size() as u64;
            self.peak_allocation
                .fetch_max(current_allocation, Ordering::AcqRel);
        }
        ret
    }

    /// Deallocates memory and counts the number of bytes deallocated.
    /// Safety: See [`GlobalAlloc::dealloc`].
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.allocator.dealloc(ptr, layout) };
        self.current_allocation
            .fetch_sub(layout.size() as u64, Ordering::AcqRel);
    }
}

#[allow(unused)]
impl<A: GlobalAlloc> CountingAllocator<A> {
    /// Returns the currently allocated memory in bytes
    pub fn get_current(&self) -> u64 {
        self.current_allocation.load(Ordering::Acquire)
    }

    /// Returns the peak allocated memory since program start in bytes
    pub fn get_peak(&self) -> u64 {
        self.peak_allocation.load(Ordering::Acquire)
    }
}

pub trait GetPeakAllocatedMemory {
    fn get_peak_allocated_memory(&self) -> Option<u64>;
}

impl<T: GlobalAlloc> GetPeakAllocatedMemory for CountingAllocator<T> {
    fn get_peak_allocated_memory(&self) -> Option<u64> {
        Some(self.get_peak())
    }
}

impl GetPeakAllocatedMemory for () {
    fn get_peak_allocated_memory(&self) -> Option<u64> {
        None
    }
}

/// Registers and initializes a `CountingAllocator` in the static
macro_rules! register_counting_allocator {
    ($static_name:ident, enable = true) => {
        #[global_allocator]
        static $static_name: crate::allocator::CountingAllocator<std::alloc::System> =
            crate::allocator::CountingAllocator {
                allocator: std::alloc::System,
                current_allocation: std::sync::atomic::AtomicU64::new(0),
                peak_allocation: std::sync::atomic::AtomicU64::new(0),
            };
    };
    ($static_name:ident, enable = false) => {
        static $static_name: () = ();
    };
}
