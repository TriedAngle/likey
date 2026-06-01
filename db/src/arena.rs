//! Append-only arena and relative typed slices.
//!
//! The builder uses a `Vec<u8>` and freezes into a separately aligned allocation
//! so typed slices such as `&[u64]` can be safely viewed. Replacing
//! [`FrozenArena`] with a mmap-backed byte region later should not affect the
//! table/query APIs.
//!
//! This implementation is process-local and host-endian. For a durable file
//! format, store integers explicitly as little-endian bytes and add a real file
//! header/version.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ptr::NonNull;
use std::slice;

/// Marker trait for types that may be copied as raw bytes into the arena.
///
/// Keep this narrow. The base implementation only uses primitive integer arrays.
/// Adding custom POD structs is fine, but they must not contain references,
/// pointers, padding you care about, or platform-dependent layout unless you
/// accept those constraints.
///
/// # Safety
///
/// Implementors must be plain data that can be copied to and from raw bytes
/// without running destructors or invalidating Rust aliasing/lifetime rules.
/// Types with references, ownership, invalid bit patterns, or layout-sensitive
/// padding must not implement this trait.
pub unsafe trait Pod: Copy + 'static {}

unsafe impl Pod for u8 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for u64 {}
unsafe impl Pod for usize {}
unsafe impl Pod for i8 {}
unsafe impl Pod for i16 {}
unsafe impl Pod for i32 {}
unsafe impl Pod for i64 {}
unsafe impl Pod for isize {}

/// A relative typed slice inside the arena.
///
/// `offset` is a byte offset from the arena base. `len` is a count of `T`, not a
/// byte length.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RelSlice<T> {
    pub offset: u64,
    pub len: u64,
    _marker: PhantomData<T>,
}

impl<T> RelSlice<T> {
    pub const fn new(offset: u64, len: u64) -> Self {
        Self {
            offset,
            len,
            _marker: PhantomData,
        }
    }

    pub const fn empty() -> Self {
        Self::new(0, 0)
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn byte_len(self) -> Option<usize> {
        (self.len as usize).checked_mul(size_of::<T>())
    }
}

impl<T> fmt::Debug for RelSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelSlice")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    OffsetOverflow,
    LengthOverflow,
    OutOfBounds,
    Misaligned { offset: usize, align: usize },
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArenaError::OffsetOverflow => write!(f, "arena offset overflow"),
            ArenaError::LengthOverflow => write!(f, "arena length overflow"),
            ArenaError::OutOfBounds => write!(f, "relative slice out of arena bounds"),
            ArenaError::Misaligned { offset, align } => {
                write!(
                    f,
                    "relative slice at offset {offset} is not aligned to {align}"
                )
            }
        }
    }
}

impl std::error::Error for ArenaError {}

/// Append-only arena builder.
#[derive(Debug, Default)]
pub struct ArenaBuilder {
    bytes: Vec<u8>,
}

impl ArenaBuilder {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn clear(&mut self) {
        self.bytes.clear();
    }

    pub fn reserve(&mut self, additional: usize) {
        self.bytes.reserve(additional);
    }

    /// Append bytes with no additional alignment.
    pub fn append_bytes(&mut self, bytes: &[u8]) -> RelSlice<u8> {
        self.append_bytes_aligned(bytes, 1)
    }

    /// Append bytes after aligning the section start.
    pub fn append_bytes_aligned(&mut self, bytes: &[u8], align: usize) -> RelSlice<u8> {
        self.align_to(align);
        let offset = self.bytes.len();
        self.bytes.extend_from_slice(bytes);
        RelSlice::new(offset as u64, bytes.len() as u64)
    }

    /// Append a typed POD slice after aligning to `align_of::<T>()`.
    pub fn append_slice<T: Pod>(&mut self, values: &[T]) -> RelSlice<T> {
        self.align_to(align_of::<T>());
        let offset = self.bytes.len();
        let byte_len = values.len() * size_of::<T>();

        if byte_len != 0 {
            let raw = unsafe { slice::from_raw_parts(values.as_ptr() as *const u8, byte_len) };
            self.bytes.extend_from_slice(raw);
        }

        RelSlice::new(offset as u64, values.len() as u64)
    }

    /// Pad with zeros until the current cursor is aligned.
    pub fn align_to(&mut self, align: usize) {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        let padded = align_up(self.bytes.len(), align);
        self.bytes.resize(padded, 0);
    }

    /// Freeze into an allocation whose base pointer is at least 64-byte aligned.
    pub fn freeze(self) -> FrozenArena {
        FrozenArena::from_vec_aligned(self.bytes, 64)
    }
}

/// Read-only aligned arena.
pub struct FrozenArena {
    ptr: NonNull<u8>,
    len: usize,
    layout: Option<Layout>,
}

impl FrozenArena {
    pub fn from_vec_aligned(bytes: Vec<u8>, align: usize) -> Self {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        let len = bytes.len();
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                layout: None,
            };
        }

        let layout = Layout::from_size_align(len, align).expect("valid arena layout");
        let raw = unsafe { alloc(layout) };
        let ptr = match NonNull::new(raw) {
            Some(ptr) => ptr,
            None => handle_alloc_error(layout),
        };

        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), len);
        }

        Self {
            ptr,
            len,
            layout: Some(layout),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn bytes(&self, rel: RelSlice<u8>) -> &[u8] {
        self.try_bytes(rel).expect("invalid arena byte slice")
    }

    pub fn try_bytes(&self, rel: RelSlice<u8>) -> Result<&[u8], ArenaError> {
        let start = usize::try_from(rel.offset).map_err(|_| ArenaError::OffsetOverflow)?;
        let len = usize::try_from(rel.len).map_err(|_| ArenaError::LengthOverflow)?;
        let end = start.checked_add(len).ok_or(ArenaError::LengthOverflow)?;
        if end > self.len {
            return Err(ArenaError::OutOfBounds);
        }
        Ok(&self.as_bytes()[start..end])
    }

    pub fn slice<T: Pod>(&self, rel: RelSlice<T>) -> &[T] {
        self.try_slice(rel).expect("invalid arena typed slice")
    }

    pub fn try_slice<T: Pod>(&self, rel: RelSlice<T>) -> Result<&[T], ArenaError> {
        let start = usize::try_from(rel.offset).map_err(|_| ArenaError::OffsetOverflow)?;
        let len = usize::try_from(rel.len).map_err(|_| ArenaError::LengthOverflow)?;
        let byte_len = len
            .checked_mul(size_of::<T>())
            .ok_or(ArenaError::LengthOverflow)?;
        let end = start
            .checked_add(byte_len)
            .ok_or(ArenaError::LengthOverflow)?;
        if end > self.len {
            return Err(ArenaError::OutOfBounds);
        }
        let align = align_of::<T>();
        if start % align != 0 {
            return Err(ArenaError::Misaligned {
                offset: start,
                align,
            });
        }

        let ptr = unsafe { self.ptr.as_ptr().add(start) as *const T };
        Ok(unsafe { slice::from_raw_parts(ptr, len) })
    }
}

impl Drop for FrozenArena {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            unsafe { dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

impl fmt::Debug for FrozenArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrozenArena")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

unsafe impl Send for FrozenArena {}
unsafe impl Sync for FrozenArena {}

fn align_up(x: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
}
