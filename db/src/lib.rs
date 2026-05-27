//! Tiny dense string database base implementation for LIKE/search experiments.
//!
//! This crate implements storage, typed table/database views, candidate
//! iteration, result sinks, a small LIKE verifier module, and baseline FM/trigram
//! indexes that plug into the candidate API.

pub mod algos;
mod arena;
mod db;
mod index;
mod like;
mod query;
mod storage;

pub use crate::algos::{
    BM, BMState, ByteNeedle, ByteWildcardNeedle, ByteWildcardState, DNA_WILDCARD, Dna2, Dna2Needle,
    Dna2PackedChunk, Dna2PackedNeedle, Dna2PackedScalar, Dna2PackedState, Dna2PackedVectorized,
    FftNeedle, FftState0, FftState1, FftStr0, FftStr1, LibcMemmem, Naive, NaiveAuto,
    NaiveAutoWildcard, NaiveAvx2, NaiveAvx2V2, NaiveAvx2V2Wildcard, NaiveAvx2Wildcard, NaiveAvx512,
    NaiveAvx512V2, NaiveAvx512V2Wildcard, NaiveAvx512Wildcard, NaiveMixed, NaiveMixedWildcard,
    NaiveScalar, NaiveScalarWildcard, NaiveVectorized, NaiveVectorizedV2,
    NaiveVectorizedV2Wildcard, NaiveVectorizedWildcard, NaiveWildcard, StdSearch, TwoWay, TwoWay2,
    TwoWay2State, TwoWayState, Utf8Kmp, bm_find, bytes_eq_same_len, bytes_match_wildcard_same_len,
    eq_at_bytes, kmp_find, kmp_find_from, matches_at_bytes, matches_at_bytes_wildcard, memmem_find,
    naive_find, naive_find_auto, naive_find_avx2, naive_find_avx2_v2, naive_find_avx512,
    naive_find_avx512_v2, naive_find_mixed, naive_find_scalar, naive_find_vectorized,
    naive_find_vectorized_v2, naive_find_wildcard, naive_find_wildcard_auto,
    naive_find_wildcard_avx2, naive_find_wildcard_avx2_v2, naive_find_wildcard_avx512,
    naive_find_wildcard_avx512_v2, naive_find_wildcard_mixed, naive_find_wildcard_scalar,
    naive_find_wildcard_vectorized, naive_find_wildcard_vectorized_v2, two_way_find, two_way2_find,
};
pub use crate::arena::{ArenaBuilder, ArenaError, FrozenArena, Pod, RelSlice};
pub use crate::db::{Db, DbBuilder, DbError, TableBuilder, TableDesc, TableKind, TableRef};
pub use crate::index::{
    BuildIndex, Dna2TrigramDomain, Fixed64PostingStore, FmIndex, FmIndexError, FmProbe,
    FsstDecodedTrigramDomain, HasTrigramIndex, HashMapPostingStore, IndexProbe, TrigramDomain,
    TrigramIndex, TrigramPostingStore, TrigramProbe, TypedTrigramIndex, Utf8ByteTrigramDomain,
    dna2_trigram_key, intersect_sorted_rowids, trigram_key, trigram_keys,
};
pub use crate::like::{
    LikeCompileError, LikeCompileOptions, LikePattern, LikeToken, LiteralAlgorithm, MatchStrategy,
    RowLiteralSearch,
};
pub use crate::query::{
    AcceptAll, BitmapSink, CandidateBatch, CandidateProvider, CandidateScratch, CountSink,
    FullScan, QueryScratch, QueryStats, ResultSink, RowVerifier, SortedRowsProbe, VerifyScratch,
    execute_like,
};
pub use crate::storage::Column;
pub use crate::storage::dna2::{
    Dna2Column, Dna2ColumnBuilder, Dna2ColumnDesc, Dna2Iter, Dna2Row, Dna2RowEntry, Dna2Table,
    Dna2TableBuilder, Dna2TableDesc, DnaBase, DnaError,
};
pub use crate::storage::fsst::{
    FsstCodec, FsstColumn, FsstColumnBuilder, FsstColumnDesc, FsstRow, FsstRowEntry, FsstTable,
    FsstTableBuilder, FsstTableDesc,
};
pub use crate::storage::utf8::{
    Utf8Column, Utf8ColumnBuilder, Utf8ColumnDesc, Utf8Row, Utf8RowEntry, Utf8Table,
    Utf8TableBuilder, Utf8TableDesc,
};

/// Physical dense row ordinal.
///
/// `RowId` is the internal primary key for a table. All columns in a table use
/// the same physical row ordering, and indexes should return these IDs.
pub type RowId = u64;

/// Stable table identifier within one [`Db`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableId(pub u32);

/// Stable column identifier within one table.
///
/// The current UTF-8, FSST, and DNA2 table types each have one searchable
/// column, but keeping a column ID in the public vocabulary makes it easier to
/// add proper multi-column tables later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColumnId(pub u32);

/// Logical-length constraint used by query verifiers.
///
/// For `Utf8Column` and `FsstColumn` with byte LIKE semantics this is decoded
/// bytes. For `Dna2Column`, this is bases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LenConstraint {
    pub min: u32,
    pub max: Option<u32>,
}

impl LenConstraint {
    pub const fn any() -> Self {
        Self { min: 0, max: None }
    }

    pub const fn exact(n: u32) -> Self {
        Self {
            min: n,
            max: Some(n),
        }
    }

    pub const fn at_least(n: u32) -> Self {
        Self { min: n, max: None }
    }

    pub const fn between(min: u32, max: u32) -> Self {
        Self {
            min,
            max: Some(max),
        }
    }

    #[inline]
    pub fn matches(self, len: u32) -> bool {
        len >= self.min && self.max.map_or(true, |max| len <= max)
    }
}

impl Default for LenConstraint {
    fn default() -> Self {
        Self::any()
    }
}
