use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::dna2::{Dna2Column, Dna2Row, DnaBase};

const DNA_WILDCARD: u8 = 0xFF;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dna2Needle {
    symbols: Box<[u8]>,
    has_wildcard: bool,
}

impl Dna2Needle {
    #[inline]
    pub fn symbols(&self) -> &[u8] {
        &self.symbols
    }

    #[inline]
    pub fn has_wildcard(&self) -> bool {
        self.has_wildcard
    }
}

/// Baseline DNA2 literal search over packed rows.
///
/// This is storage-compatible with the packed `Dna2Column`, but it is still a
/// simple scalar logical-base matcher. The important API point is that `_` can
/// stay inside literal fragments and be interpreted as one wildcard DNA base.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dna2NaiveWildcard;

impl LiteralAlgorithm for Dna2NaiveWildcard {
    type Needle = Dna2Needle;
    type State = ();

    const SUPPORTS_UNDERSCORE: bool = true;

    fn compile_literal(src: &str) -> Option<Self::Needle> {
        let mut symbols = Vec::with_capacity(src.len());
        let mut has_wildcard = false;

        for &b in src.as_bytes() {
            if b == b'_' {
                symbols.push(DNA_WILDCARD);
                has_wildcard = true;
            } else {
                symbols.push(DnaBase::from_ascii(b).ok()?.code());
            }
        }

        Some(Dna2Needle {
            symbols: symbols.into_boxed_slice(),
            has_wildcard,
        })
    }

    #[inline]
    fn build_state(_needle: &Self::Needle) -> Self::State {
        ()
    }

    #[inline]
    fn literal_len(needle: &Self::Needle) -> u32 {
        needle.symbols.len() as u32
    }

    #[inline]
    fn index_symbols(needle: &Self::Needle) -> Option<Box<[u8]>> {
        if needle.has_wildcard {
            None
        } else {
            Some(needle.symbols.clone())
        }
    }
}

impl<'db> RowLiteralSearch<Dna2Column<'db>> for Dna2NaiveWildcard {
    #[inline]
    fn row_len<'r>(row: &Dna2Row<'r>) -> u32 {
        row.len_bases()
    }

    #[inline]
    fn matches_at<'r>(
        row: &Dna2Row<'r>,
        pos: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> bool {
        let len = needle.symbols.len() as u32;
        let Some(end) = pos.checked_add(len) else {
            return false;
        };
        if end > row.len_bases() {
            return false;
        }

        for (i, &want) in needle.symbols.iter().enumerate() {
            if want == DNA_WILDCARD {
                continue;
            }
            if row.base_code_at(pos + i as u32) != want {
                return false;
            }
        }
        true
    }

    #[inline]
    fn find_from<'r>(
        row: &Dna2Row<'r>,
        from: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> Option<u32> {
        let text_len = row.len_bases();
        let needle_len = needle.symbols.len() as u32;

        if from > text_len {
            return None;
        }
        if needle_len == 0 {
            return Some(from);
        }
        if needle_len > text_len.saturating_sub(from) {
            return None;
        }

        let last_start = text_len - needle_len;
        let mut pos = from;
        while pos <= last_start {
            if Self::matches_at(row, pos, needle, state) {
                return Some(pos);
            }
            pos += 1;
        }
        None
    }
}
