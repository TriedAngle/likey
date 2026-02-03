#![allow(unused)]
pub mod dataset;
pub mod fasta;
mod system;

use std::{
    alloc::Layout,
    cell::UnsafeCell,
    marker::PhantomData,
    ptr::{self, NonNull},
};

use system::{map_memory, unmap_memory};

struct BumpArenaInner {
    start: NonNull<u8>,
    ptr: *mut u8,
    end: *mut u8,
    total_size: usize,
}

pub struct BumpArena {
    inner: UnsafeCell<BumpArenaInner>,
    _marker: PhantomData<*mut u8>,
}

impl BumpArena {
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "Size must be > 0");
        let start_ptr = map_memory(size).expect("BumpArena: mmap failed");
        let raw_start = start_ptr.as_ptr();
        let end_ptr = unsafe { raw_start.add(size) };

        Self {
            inner: UnsafeCell::new(BumpArenaInner {
                start: start_ptr,
                ptr: raw_start,
                end: end_ptr,
                total_size: size,
            }),
            _marker: PhantomData,
        }
    }

    /// Helper to get mutable access to inner state from a shared reference.
    /// # Safety:
    /// This is not thread-safe. Caller must ensure single-threaded access
    #[inline(always)]
    unsafe fn get_inner_mut(&self) -> &mut BumpArenaInner {
        // SAFETY: caller ensures single threaded access
        unsafe { self.inner.get().as_mut().unwrap_unchecked() }
    }

    /// Allocates space for a layout.
    pub fn alloc_layout(&self, layout: Layout) -> *mut u8 {
        let inner = unsafe { self.get_inner_mut() };

        let align = layout.align();
        let size = layout.size();
        let offset = inner.ptr.align_offset(align);

        if offset == usize::MAX {
            panic!("BumpArena: Fatal alignment error");
        }

        unsafe {
            let aligned_ptr = inner.ptr.add(offset);
            let new_ptr = aligned_ptr.add(size);

            if new_ptr > inner.end {
                panic!(
                    "BumpArena: Out of Memory! Capacity: {}, Used: {}, Requested: {}",
                    inner.total_size,
                    self.used(),
                    size
                );
            }

            inner.ptr = new_ptr;
            aligned_ptr
        }
    }

    /// Allocate a value.
    #[inline(always)]
    pub fn alloc<T>(&self, value: T) -> &mut T {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_layout(layout) as *mut T;
        unsafe {
            ptr::write(ptr, value);
            &mut *ptr
        }
    }

    /// Clone a slice into the arena.
    pub fn alloc_slice<T: Copy>(&self, src: &[T]) -> &mut [T] {
        let layout = Layout::for_value(src);
        let dst_ptr = self.alloc_layout(layout) as *mut T;
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
            std::slice::from_raw_parts_mut(dst_ptr, src.len())
        }
    }

    /// Clone a str into the arena.
    pub fn alloc_str(&self, src: &str) -> &mut str {
        let buf = self.alloc_slice(src.as_bytes());
        unsafe { std::str::from_utf8_unchecked_mut(buf) }
    }

    /// Reset.
    pub fn reset(&mut self) {
        let inner = self.inner.get_mut(); // Safe because we have &mut self
        inner.ptr = inner.start.as_ptr();
    }

    pub fn used(&self) -> usize {
        let inner = unsafe { self.get_inner_mut() };
        (inner.ptr as usize) - (inner.start.as_ptr() as usize)
    }

    pub fn capacity(&self) -> usize {
        let inner = unsafe { self.get_inner_mut() };
        inner.total_size
    }

    pub fn start(&self) -> *mut u8 {
        let inner = unsafe { self.get_inner_mut() };
        inner.start.as_ptr()
    }

    pub fn bump(&self) -> *mut u8 {
        let inner = unsafe { self.get_inner_mut() };
        inner.ptr
    }
}

impl Drop for BumpArena {
    fn drop(&mut self) {
        let inner = self.inner.get_mut();
        unmap_memory(inner.start, inner.total_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    #[test]
    fn test_basic_allocation() {
        let arena = BumpArena::new(1024); // 1KB

        let val_a = arena.alloc(10u64);
        let val_b = arena.alloc(20u64);

        assert_eq!(*val_a, 10);
        assert_eq!(*val_b, 20);

        *val_a = 99;
        assert_eq!(*val_a, 99);
        assert_eq!(*val_b, 20);
    }

    #[test]
    fn test_contiguous_memory() {
        let arena = BumpArena::new(1024);

        let a = arena.alloc(1u32);
        let b = arena.alloc(2u32);
        let c = arena.alloc(3u32);

        let ptr_a = a as *const u32 as usize;
        let ptr_b = b as *const u32 as usize;
        let ptr_c = c as *const u32 as usize;

        assert_eq!(ptr_b - ptr_a, 4, "Allocations are not contiguous!");
        assert_eq!(ptr_c - ptr_b, 4, "Allocations are not contiguous!");

        assert_eq!(arena.used(), 12);
    }

    #[test]
    fn test_alignment_padding() {
        let arena = BumpArena::new(1024);

        let _byte = arena.alloc(0xFFu8);
        assert_eq!(arena.used(), 1);

        let _long = arena.alloc(12345u64);

        assert_eq!(arena.used(), 16);

        let ptr_addr = _long as *const u64 as usize;
        assert_eq!(ptr_addr % 8, 0, "Pointer is not aligned to 8 bytes!");
    }

    #[test]
    fn test_reset_behavior() {
        let mut arena = BumpArena::new(1024);

        let ptr1_addr = arena.alloc(10u64) as *const u64;
        assert_eq!(arena.used(), 8);

        arena.reset();
        assert_eq!(arena.used(), 0);

        let ptr2 = arena.alloc(20u64);

        assert_eq!(ptr1_addr, ptr2 as *const u64);
        assert_eq!(*ptr2, 20);
    }

    #[test]
    #[should_panic(expected = "BumpArena: Out of Memory!")]
    fn test_oom_panic() {
        let arena = BumpArena::new(16);

        arena.alloc(0u64);
        arena.alloc(0u64);

        arena.alloc(0u8);
    }

    #[test]
    #[should_panic(expected = "BumpArena: Out of Memory!")]
    fn test_oom_via_padding() {
        let arena = BumpArena::new(10);

        arena.alloc(0u8);
        arena.alloc(0u64);
    }

    #[test]
    fn test_alloc_layout_raw() {
        let arena = BumpArena::new(1024);

        let layout = Layout::from_size_align(100, 16).unwrap();

        let ptr = arena.alloc_layout(layout);

        assert_eq!(ptr as usize % 16, 0);

        unsafe {
            std::ptr::write_bytes(ptr, 0xAA, 100);
        }
    }

    #[test]
    fn test_alloc_slice_u8() {
        let arena = BumpArena::new(1024);
        let original = [0xAAu8, 0xBB, 0xCC, 0xDD];

        let stored = arena.alloc_slice(&original);

        assert_eq!(stored, &original);

        stored[0] = 0x00;
        assert_eq!(stored[0], 0x00);
        assert_eq!(original[0], 0xAA);

        let arena_start = arena.start() as usize;
        let stored_addr = stored.as_ptr() as usize;
        assert!(stored_addr >= arena_start);
        assert!(stored_addr < arena_start + 1024);
    }

    #[test]
    fn test_alloc_slice_u64_alignment() {
        let arena = BumpArena::new(1024);

        arena.alloc(0u8);

        let numbers = [100u64, 200, 300];
        let stored = arena.alloc_slice(&numbers);

        let addr = stored.as_ptr() as usize;
        assert_eq!(addr % 8, 0, "Slice address must be 8-byte aligned");

        assert_eq!(stored, &numbers);
    }

    #[test]
    fn test_alloc_str() {
        let arena = BumpArena::new(1024);
        let original = "SELECT * FROM users";

        let stored_str = arena.alloc_str(original);

        assert_eq!(stored_str, original);

        assert_ne!(stored_str.as_ptr(), original.as_ptr());

        unsafe {
            let bytes = stored_str.as_bytes_mut();
            bytes[0] = b's';
        }

        assert_eq!(stored_str, "sELECT * FROM users");
    }

    #[test]
    fn test_multiple_clones() {
        let arena = BumpArena::new(4096);

        let s1 = arena.alloc_str("Key");
        let s2 = arena.alloc_str("Value");
        let nums = arena.alloc_slice(&[1, 2, 3, 4]);

        let p1 = s1.as_ptr() as usize;
        let p2 = s2.as_ptr() as usize;

        assert_eq!(p2 - p1, 3);

        let p3 = nums.as_ptr() as usize;
        assert_eq!(p3 - p2, 5);
    }
}
