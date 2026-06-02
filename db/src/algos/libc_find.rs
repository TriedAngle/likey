use core::ffi::c_void;

use crate::{LiteralAlgorithm, RowLiteralSearch};
use crate::{Utf8Column, Utf8Row};

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "android")))]
use super::naive::naive_find_scalar;

use super::utf8_shared::{
    ByteNeedle, byte_index_symbols, byte_literal_len, compile_byte_literal, matches_at_bytes,
    utf8_row_len,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct LibcMemmem;

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "android"))]
unsafe extern "C" {
    fn memmem(
        haystack: *const c_void,
        haystacklen: usize,
        needle: *const c_void,
        needlelen: usize,
    ) -> *mut c_void;
}

impl LiteralAlgorithm for LibcMemmem {
    type Needle = ByteNeedle;
    type State = ();

    const SUPPORTS_UNDERSCORE: bool = false;

    #[inline]
    fn compile_literal(src: &str) -> Option<Self::Needle> {
        compile_byte_literal(src)
    }

    #[inline]
    fn build_state(_needle: &Self::Needle) -> Self::State {
        ()
    }

    #[inline]
    fn literal_len(needle: &Self::Needle) -> u32 {
        byte_literal_len(needle)
    }

    #[inline]
    fn index_symbols(needle: &Self::Needle) -> Option<Box<[u8]>> {
        byte_index_symbols(needle)
    }
}

impl<'db> RowLiteralSearch<Utf8Column<'db>> for LibcMemmem {
    #[inline]
    fn row_len<'r>(row: &Utf8Row<'r>) -> u32 {
        utf8_row_len(row)
    }

    #[inline(always)]
    fn matches_at<'r>(
        row: &Utf8Row<'r>,
        pos: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> bool {
        matches_at_bytes(row, pos, needle)
    }

    #[inline]
    fn find_from<'r>(
        row: &Utf8Row<'r>,
        from: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> Option<u32> {
        let text = row.bytes();
        let pat = needle.bytes();
        let from = from as usize;

        if from > text.len() {
            return None;
        }
        memmem_find(&text[from..], pat).map(|pos| (pos + from) as u32)
    }
}

#[inline]
pub fn memmem_find(text: &[u8], needle: &[u8]) -> Option<usize> {
    let m = needle.len();
    let n = text.len();

    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "android"))]
    {
        let ptr = unsafe {
            memmem(
                text.as_ptr().cast::<c_void>(),
                n,
                needle.as_ptr().cast::<c_void>(),
                m,
            )
        };

        if ptr.is_null() {
            None
        } else {
            let base = text.as_ptr() as usize;
            Some((ptr as usize) - base)
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "android")))]
    {
        naive_find_scalar(text, needle)
    }
}
