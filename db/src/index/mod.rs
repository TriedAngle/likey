//! Index extension traits and small baseline index implementations.
//!
//! The database itself does not need to know about concrete indexes. The seam is
//! [`CandidateProvider`](crate::CandidateProvider): a trigram index, FM-index,
//! equality index, or benchmark fixture exposes a probe object that yields row
//! candidates for [`execute_like`](crate::execute_like).

pub mod fm;

pub use fm::{FmIndex, FmIndexError, FmProbe};

use crate::RowId;
use crate::query::CandidateProvider;
use crate::storage::Column;

/// Optional build hook for reusable index types.
///
/// You do not need this trait to use `execute_like`; it is only a convention.
pub trait BuildIndex<C: Column>: Sized {
    fn build(column: &C) -> Self;
}

/// Marker trait for index probes.
///
/// Implementing [`CandidateProvider`] is the important part. This alias-style
/// trait is useful when naming APIs such as `fn probe(...) -> impl IndexProbe`.
pub trait IndexProbe: CandidateProvider {}

impl<T: CandidateProvider> IndexProbe for T {}

/// Intersect two sorted, deduplicated row-id lists into `out`.
///
/// This is handy for row-list trigram indexes where a LIKE pattern has several
/// required grams. The output is also sorted and deduplicated if the inputs are.
pub fn intersect_sorted_rowids(a: &[RowId], b: &[RowId], out: &mut Vec<RowId>) {
    out.clear();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
}
