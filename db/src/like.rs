//! Minimal LIKE compiler/verifier integration.
//!
//! This module is deliberately small:
//! - compile a LIKE pattern into `Literal`, `Skip(_)`, and `Any` tokens;
//! - build literal-search state once per literal;
//! - implement `RowVerifier<C>` so the compiled pattern plugs into `execute_like`;
//! - keep indexes separate: they still only produce candidate `RowId`s.
//!
//! `_` handling is controlled by the literal algorithm. Algorithms that do not
//! support `_` receive it as `Skip(1)`. Algorithms that do support `_` can keep
//! it inside literal fragments and interpret it directly.

use std::marker::PhantomData;

use crate::query::{RowVerifier, VerifyScratch};
use crate::storage::Column;
use crate::{LenConstraint, RowId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LikeToken {
    /// Index into `LikePattern::literals`.
    Literal(usize),
    /// SQL `_` lowered to a logical-unit skip.
    Skip(u32),
    /// SQL `%`.
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchStrategy {
    /// Pattern is `%`, `%%`, etc.
    All,
    /// No `%` and no lowered `_` skips. `None` means empty pattern.
    Exact { literal_idx: Option<usize> },
    /// `literal%`.
    Prefix { literal_idx: usize },
    /// `%literal`.
    Suffix { literal_idx: usize },
    /// `%literal%`.
    Contains { literal_idx: usize },
    /// Everything else.
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LikeCompileOptions {
    /// If true, algorithms with `SUPPORTS_UNDERSCORE = true` receive `_` inside
    /// literal fragments. If false, `_` is always lowered to `Skip(1)`.
    pub pass_underscore_to_algorithm: bool,
}

impl Default for LikeCompileOptions {
    fn default() -> Self {
        Self {
            pass_underscore_to_algorithm: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LikeCompileError {
    UnsupportedLiteral { fragment: String },
    LengthOverflow,
}

impl std::fmt::Display for LikeCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LikeCompileError::UnsupportedLiteral { fragment } => {
                write!(
                    f,
                    "LIKE literal fragment is not supported by this algorithm: {fragment:?}"
                )
            }
            LikeCompileError::LengthOverflow => {
                write!(f, "LIKE pattern logical length exceeds u32")
            }
        }
    }
}

impl std::error::Error for LikeCompileError {}

/// Literal-level algorithm state.
///
/// This trait is independent of the table/column lifetime. It is only about
/// compiling literal fragments once. For example, KMP stores a byte needle plus
/// its prefix table here.
pub trait LiteralAlgorithm {
    type Needle;
    type State;

    /// Whether `_` may remain inside literal fragments.
    ///
    /// If false, the compiler lowers `_` to `LikeToken::Skip(1)`. If true and
    /// `LikeCompileOptions::pass_underscore_to_algorithm` is true, `_` stays in
    /// the literal and the algorithm decides what it means.
    const SUPPORTS_UNDERSCORE: bool = false;

    fn compile_literal(src: &str) -> Option<Self::Needle>;
    fn build_state(needle: &Self::Needle) -> Self::State;
    fn literal_len(needle: &Self::Needle) -> u32;

    /// Exact logical symbols usable by an index, if this literal has no
    /// algorithm-level wildcard characters.
    ///
    /// For UTF-8 byte KMP this is the literal bytes. For a DNA wildcard literal
    /// such as `A_G`, this returns `None` because the middle position is not an
    /// exact symbol.
    fn index_symbols(_needle: &Self::Needle) -> Option<Box<[u8]>> {
        None
    }
}

/// Row-level operations for a literal algorithm on one concrete dense column.
///
/// Implement this for the column types on which the algorithm is legal. KMP is
/// implemented only for `Utf8Column<'_>` in `algos::kmp`, so it cannot
/// accidentally be used on DNA2 rows.
pub trait RowLiteralSearch<C>: LiteralAlgorithm
where
    C: Column<Symbol = u8>,
{
    fn row_len<'r>(row: &C::Row<'r>) -> u32;

    fn matches_at<'r>(
        row: &C::Row<'r>,
        pos: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> bool;

    fn find_from<'r>(
        row: &C::Row<'r>,
        from: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> Option<u32>;

    fn advance<'r>(row: &C::Row<'r>, pos: u32, count: u32) -> Option<u32> {
        let next = pos.checked_add(count)?;
        if next <= Self::row_len(row) {
            Some(next)
        } else {
            None
        }
    }
}

struct CompiledLiteral<A: LiteralAlgorithm> {
    source: Box<str>,
    needle: A::Needle,
    state: A::State,
    len: u32,
    index_symbols: Option<Box<[u8]>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SegmentAnchor {
    literal_idx: usize,
    /// Logical offset of this literal from the start of the `%`-free segment.
    offset: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SegmentPlan {
    /// Half-open token range. This range never contains `LikeToken::Any`.
    token_start: usize,
    token_end: usize,
    /// Fixed logical length of this segment: literals plus lowered `_` skips.
    len: u32,
    /// First literal in this segment. Used by the direct implementation of the
    /// proposed adaptive algorithm.
    first_anchor: Option<SegmentAnchor>,
    /// Best static literal anchor in this segment. Today this means the longest
    /// literal fragment, with exact/indexable fragments winning ties.
    best_anchor: Option<SegmentAnchor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentVerify {
    Match,
    FailedLiteral { anchor: SegmentAnchor, pos: u32 },
    FailedUnanchored,
}

pub struct LikePattern<A: LiteralAlgorithm> {
    tokens: Box<[LikeToken]>,
    literals: Box<[CompiledLiteral<A>]>,
    segments: Box<[SegmentPlan]>,
    strategy: MatchStrategy,
    min_len: u32,
    has_any: bool,
    has_skip: bool,
    _marker: PhantomData<A>,
}

impl<A> LikePattern<A>
where
    A: LiteralAlgorithm,
{
    pub fn compile(pattern: &str) -> Result<Self, LikeCompileError> {
        Self::compile_with_options(pattern, LikeCompileOptions::default())
    }

    pub fn compile_with_options(
        pattern: &str,
        options: LikeCompileOptions,
    ) -> Result<Self, LikeCompileError> {
        let mut tokens = Vec::new();
        let mut literals = Vec::<CompiledLiteral<A>>::new();
        let mut start_idx = 0usize;
        let mut min_len = 0u32;

        let pass_underscore = A::SUPPORTS_UNDERSCORE && options.pass_underscore_to_algorithm;

        for (idx, ch) in pattern.char_indices() {
            let underscore_is_token = ch == '_' && !pass_underscore;
            let breaks_literal = ch == '%' || underscore_is_token;
            if !breaks_literal {
                continue;
            }

            if idx > start_idx {
                push_literal::<A>(
                    &mut tokens,
                    &mut literals,
                    &mut min_len,
                    &pattern[start_idx..idx],
                )?;
            }

            match ch {
                '%' => {
                    if tokens.last() != Some(&LikeToken::Any) {
                        tokens.push(LikeToken::Any);
                    }
                }
                '_' => {
                    if let Some(LikeToken::Skip(count)) = tokens.last_mut() {
                        *count = count
                            .checked_add(1)
                            .ok_or(LikeCompileError::LengthOverflow)?;
                    } else {
                        tokens.push(LikeToken::Skip(1));
                    }
                    min_len = min_len
                        .checked_add(1)
                        .ok_or(LikeCompileError::LengthOverflow)?;
                }
                _ => unreachable!("only % and _ break literals"),
            }

            start_idx = idx + ch.len_utf8();
        }

        if start_idx < pattern.len() {
            push_literal::<A>(
                &mut tokens,
                &mut literals,
                &mut min_len,
                &pattern[start_idx..],
            )?;
        }

        let has_any = tokens.iter().any(|t| matches!(t, LikeToken::Any));
        let has_skip = tokens.iter().any(|t| matches!(t, LikeToken::Skip(_)));
        let segments = build_segments::<A>(&tokens, &literals);
        let strategy = derive_strategy(&tokens, has_skip);

        Ok(Self {
            tokens: tokens.into_boxed_slice(),
            literals: literals.into_boxed_slice(),
            segments,
            strategy,
            min_len,
            has_any,
            has_skip,
            _marker: PhantomData,
        })
    }

    pub fn tokens(&self) -> &[LikeToken] {
        &self.tokens
    }

    pub fn strategy(&self) -> MatchStrategy {
        self.strategy
    }

    pub fn min_len(&self) -> u32 {
        self.min_len
    }

    pub fn has_any(&self) -> bool {
        self.has_any
    }

    pub fn has_skip(&self) -> bool {
        self.has_skip
    }

    pub fn literal_count(&self) -> usize {
        self.literals.len()
    }

    pub fn literal_source(&self, literal_idx: usize) -> &str {
        &self.literals[literal_idx].source
    }

    /// Exact literal fragments that an index may use for candidate generation.
    pub fn indexable_literals(&self) -> impl Iterator<Item = &[u8]> + '_ {
        self.literals
            .iter()
            .filter_map(|lit| lit.index_symbols.as_deref())
    }

    /// A convenient simple index hint: usually the longest exact fragment gives
    /// the most selective trigrams/FM-index probes.
    pub fn longest_indexable_literal(&self) -> Option<&[u8]> {
        self.indexable_literals().max_by_key(|lit| lit.len())
    }

    fn len_constraint_internal(&self) -> LenConstraint {
        if self.has_any {
            LenConstraint::at_least(self.min_len)
        } else {
            LenConstraint::exact(self.min_len)
        }
    }

    pub fn matches_row<'r, C>(&self, row: &C::Row<'r>) -> bool
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let text_len = A::row_len(row);
        if !self.len_constraint_internal().matches(text_len) {
            return false;
        }

        match self.strategy {
            MatchStrategy::All => true,
            MatchStrategy::Exact { literal_idx } => match literal_idx {
                None => text_len == 0,
                Some(idx) => {
                    text_len == self.literal_len(idx) && self.literal_matches_at::<C>(row, 0, idx)
                }
            },
            MatchStrategy::Prefix { literal_idx } => {
                self.literal_matches_at::<C>(row, 0, literal_idx)
            }
            MatchStrategy::Suffix { literal_idx } => {
                let len = self.literal_len(literal_idx);
                text_len >= len && self.literal_matches_at::<C>(row, text_len - len, literal_idx)
            }
            MatchStrategy::Contains { literal_idx } => {
                self.find_literal_from::<C>(row, 0, literal_idx).is_some()
            }
            MatchStrategy::General => self.match_general_segmented::<C>(row, text_len),
        }
    }

    /// General LIKE verification using the improved `%`-split segment plan.
    ///
    /// This is the default `General` implementation used by `matches_row`.
    /// It splits the pattern into fixed-width segments separated by `%`, picks
    /// the best literal anchor inside each segment, searches for that anchor,
    /// and verifies the whole segment around each candidate hit.
    pub fn matches_row_general_segmented<'r, C>(&self, row: &C::Row<'r>) -> bool
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let text_len = A::row_len(row);
        if !self.len_constraint_internal().matches(text_len) {
            return false;
        }

        match self.strategy {
            MatchStrategy::General => self.match_general_segmented::<C>(row, text_len),
            _ => self.matches_row::<C>(row),
        }
    }

    /// Direct implementation of the proposed adaptive restart idea.
    ///
    /// For each fixed-width segment, it starts by searching for the first
    /// literal fragment. If verification later fails at another literal, that
    /// failed literal becomes the next search anchor. This is useful for
    /// benchmarking the proposed heuristic against the static best-anchor plan.
    pub fn matches_row_general_pure_proposed<'r, C>(&self, row: &C::Row<'r>) -> bool
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let text_len = A::row_len(row);
        if !self.len_constraint_internal().matches(text_len) {
            return false;
        }

        match self.strategy {
            MatchStrategy::General => self.match_general_pure_proposed::<C>(row, text_len),
            _ => self.matches_row::<C>(row),
        }
    }

    /// Reference implementation of the old recursive/backtracking verifier.
    ///
    /// Kept public so tests/benchmarks can compare the new segment-based
    /// implementations against the original behavior without keeping dead
    /// private code around.
    pub fn matches_row_recursive_reference<'r, C>(&self, row: &C::Row<'r>) -> bool
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let text_len = A::row_len(row);
        if !self.len_constraint_internal().matches(text_len) {
            return false;
        }

        match self.strategy {
            MatchStrategy::General => self.match_from::<C>(row, 0, 0, text_len),
            _ => self.matches_row::<C>(row),
        }
    }

    #[inline]
    fn literal_len(&self, literal_idx: usize) -> u32 {
        self.literals[literal_idx].len
    }

    #[inline]
    fn literal_matches_at<'r, C>(&self, row: &C::Row<'r>, pos: u32, literal_idx: usize) -> bool
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let lit = &self.literals[literal_idx];
        A::matches_at(row, pos, &lit.needle, &lit.state)
    }

    #[inline]
    fn find_literal_from<'r, C>(
        &self,
        row: &C::Row<'r>,
        from: u32,
        literal_idx: usize,
    ) -> Option<u32>
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let lit = &self.literals[literal_idx];
        A::find_from(row, from, &lit.needle, &lit.state)
    }

    fn match_general_segmented<'r, C>(&self, row: &C::Row<'r>, text_len: u32) -> bool
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        self.match_general_segments::<C>(row, text_len, false)
    }

    fn match_general_pure_proposed<'r, C>(&self, row: &C::Row<'r>, text_len: u32) -> bool
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        self.match_general_segments::<C>(row, text_len, true)
    }

    fn match_general_segments<'r, C>(
        &self,
        row: &C::Row<'r>,
        text_len: u32,
        pure_proposed: bool,
    ) -> bool
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        if self.segments.is_empty() {
            // The length constraint has already handled `%`/`_`-only patterns.
            return true;
        }

        let starts_with_any = matches!(self.tokens.first(), Some(LikeToken::Any));
        let ends_with_any = matches!(self.tokens.last(), Some(LikeToken::Any));

        let mut first_middle = 0usize;
        let mut last_middle = self.segments.len();
        let mut lower = 0u32;
        let mut upper = text_len;

        // SQL LIKE is implicitly anchored at the start unless the pattern starts
        // with `%`.
        if !starts_with_any {
            let segment = &self.segments[0];
            if !self.segment_matches_at::<C>(row, segment, 0, text_len) {
                return false;
            }
            lower = segment.len;
            first_middle = 1;
        }

        // SQL LIKE is implicitly anchored at the end unless the pattern ends
        // with `%`. If there is only one segment, the prefix check above has
        // already checked it; the global length constraint enforces exactness.
        if !ends_with_any && last_middle > first_middle {
            let suffix_idx = last_middle - 1;
            let segment = &self.segments[suffix_idx];
            if text_len < segment.len {
                return false;
            }
            let suffix_start = text_len - segment.len;
            if suffix_start < lower {
                return false;
            }
            if !self.segment_matches_at::<C>(row, segment, suffix_start, text_len) {
                return false;
            }
            upper = suffix_start;
            last_middle = suffix_idx;
        }

        // Middle segments are ordered by `%` separators. Taking the earliest
        // valid occurrence is safe because `%` imposes only a lower bound on the
        // next segment; an earlier match leaves at least as much room for the
        // remaining suffix.
        for segment_idx in first_middle..last_middle {
            let start = if pure_proposed {
                self.find_segment_from_pure_proposed::<C>(row, segment_idx, lower, upper, text_len)
            } else {
                self.find_segment_from_best_anchor::<C>(row, segment_idx, lower, upper, text_len)
            };

            let Some(start) = start else {
                return false;
            };
            let Some(next_lower) = start.checked_add(self.segments[segment_idx].len) else {
                return false;
            };
            lower = next_lower;
        }

        true
    }

    fn segment_matches_at<'r, C>(
        &self,
        row: &C::Row<'r>,
        segment: &SegmentPlan,
        start: u32,
        text_len: u32,
    ) -> bool
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let Some(end) = start.checked_add(segment.len) else {
            return false;
        };
        if end > text_len {
            return false;
        }

        let mut pos = start;
        for token in &self.tokens[segment.token_start..segment.token_end] {
            match *token {
                LikeToken::Literal(literal_idx) => {
                    let literal_len = self.literal_len(literal_idx);
                    if pos > text_len || text_len - pos < literal_len {
                        return false;
                    }
                    if !self.literal_matches_at::<C>(row, pos, literal_idx) {
                        return false;
                    }
                    let Some(next) = pos.checked_add(literal_len) else {
                        return false;
                    };
                    pos = next;
                }
                LikeToken::Skip(count) => {
                    let Some(next) = A::advance(row, pos, count) else {
                        return false;
                    };
                    pos = next;
                }
                LikeToken::Any => unreachable!("segments never contain `%` tokens"),
            }
        }

        pos == end
    }

    fn verify_segment_or_first_failed_anchor<'r, C>(
        &self,
        row: &C::Row<'r>,
        segment: &SegmentPlan,
        start: u32,
        text_len: u32,
    ) -> SegmentVerify
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let Some(end) = start.checked_add(segment.len) else {
            return SegmentVerify::FailedUnanchored;
        };
        if end > text_len {
            return SegmentVerify::FailedUnanchored;
        }

        let mut pos = start;
        let mut offset = 0u32;
        for token in &self.tokens[segment.token_start..segment.token_end] {
            match *token {
                LikeToken::Literal(literal_idx) => {
                    let literal_len = self.literal_len(literal_idx);
                    let anchor = SegmentAnchor {
                        literal_idx,
                        offset,
                    };
                    if pos > text_len || text_len - pos < literal_len {
                        return SegmentVerify::FailedLiteral { anchor, pos };
                    }
                    if !self.literal_matches_at::<C>(row, pos, literal_idx) {
                        return SegmentVerify::FailedLiteral { anchor, pos };
                    }

                    let Some(next_pos) = pos.checked_add(literal_len) else {
                        return SegmentVerify::FailedUnanchored;
                    };
                    let Some(next_offset) = offset.checked_add(literal_len) else {
                        return SegmentVerify::FailedUnanchored;
                    };
                    pos = next_pos;
                    offset = next_offset;
                }
                LikeToken::Skip(count) => {
                    let Some(next_pos) = A::advance(row, pos, count) else {
                        return SegmentVerify::FailedUnanchored;
                    };
                    let Some(next_offset) = offset.checked_add(count) else {
                        return SegmentVerify::FailedUnanchored;
                    };
                    pos = next_pos;
                    offset = next_offset;
                }
                LikeToken::Any => unreachable!("segments never contain `%` tokens"),
            }
        }

        if pos == end {
            SegmentVerify::Match
        } else {
            SegmentVerify::FailedUnanchored
        }
    }

    fn find_segment_from_best_anchor<'r, C>(
        &self,
        row: &C::Row<'r>,
        segment_idx: usize,
        lower: u32,
        upper: u32,
        text_len: u32,
    ) -> Option<u32>
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let segment = &self.segments[segment_idx];
        let max_start = max_segment_start(lower, upper, segment.len)?;

        let Some(anchor) = segment.best_anchor else {
            return Some(lower);
        };

        let mut search_from = lower.checked_add(anchor.offset)?;
        loop {
            let hit = self.find_literal_from::<C>(row, search_from, anchor.literal_idx)?;
            if hit < anchor.offset {
                search_from = A::advance(row, hit, 1)?;
                continue;
            }

            let start = hit - anchor.offset;
            if start < lower {
                search_from = A::advance(row, hit, 1)?;
                continue;
            }
            if start > max_start {
                return None;
            }

            if self.segment_matches_at::<C>(row, segment, start, text_len) {
                return Some(start);
            }

            search_from = A::advance(row, hit, 1)?;
        }
    }

    fn find_segment_from_pure_proposed<'r, C>(
        &self,
        row: &C::Row<'r>,
        segment_idx: usize,
        lower: u32,
        upper: u32,
        text_len: u32,
    ) -> Option<u32>
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        let segment = &self.segments[segment_idx];
        let max_start = max_segment_start(lower, upper, segment.len)?;

        let Some(mut anchor) = segment.first_anchor else {
            return Some(lower);
        };

        let mut search_from = lower.checked_add(anchor.offset)?;
        loop {
            let hit = self.find_literal_from::<C>(row, search_from, anchor.literal_idx)?;
            if hit < anchor.offset {
                search_from = A::advance(row, hit, 1)?;
                continue;
            }

            let start = hit - anchor.offset;
            if start < lower {
                search_from = A::advance(row, hit, 1)?;
                continue;
            }
            if start > max_start {
                return None;
            }

            match self.verify_segment_or_first_failed_anchor::<C>(row, segment, start, text_len) {
                SegmentVerify::Match => return Some(start),
                SegmentVerify::FailedLiteral {
                    anchor: failed_anchor,
                    pos,
                } => {
                    anchor = failed_anchor;
                    search_from = pos;
                }
                SegmentVerify::FailedUnanchored => {
                    search_from = A::advance(row, hit, 1)?;
                }
            }
        }
    }

    fn match_from<'r, C>(&self, row: &C::Row<'r>, token_idx: usize, pos: u32, text_len: u32) -> bool
    where
        C: Column<Symbol = u8>,
        A: RowLiteralSearch<C>,
    {
        if pos > text_len {
            return false;
        }
        if token_idx == self.tokens.len() {
            return pos == text_len;
        }

        match self.tokens[token_idx] {
            LikeToken::Literal(literal_idx) => {
                if !self.literal_matches_at::<C>(row, pos, literal_idx) {
                    return false;
                }
                let next = pos + self.literal_len(literal_idx);
                self.match_from::<C>(row, token_idx + 1, next, text_len)
            }
            LikeToken::Skip(count) => {
                let Some(next) = A::advance(row, pos, count) else {
                    return false;
                };
                self.match_from::<C>(row, token_idx + 1, next, text_len)
            }
            LikeToken::Any => {
                if token_idx + 1 == self.tokens.len() {
                    return true;
                }

                // Optimization: if `%` is followed by a literal, jump between
                // literal occurrences instead of trying every position.
                if let Some(LikeToken::Literal(next_lit)) = self.tokens.get(token_idx + 1).copied()
                {
                    let mut search_from = pos;
                    loop {
                        let Some(hit) = self.find_literal_from::<C>(row, search_from, next_lit)
                        else {
                            return false;
                        };
                        if self.match_from::<C>(row, token_idx + 1, hit, text_len) {
                            return true;
                        }
                        let Some(next_search) = A::advance(row, hit, 1) else {
                            return false;
                        };
                        search_from = next_search;
                    }
                }

                // Otherwise keep it simple and try every logical position.
                let mut p = pos;
                loop {
                    if self.match_from::<C>(row, token_idx + 1, p, text_len) {
                        return true;
                    }
                    if p == text_len {
                        return false;
                    }
                    let Some(next) = A::advance(row, p, 1) else {
                        return false;
                    };
                    p = next;
                }
            }
        }
    }
}

impl<C, A> RowVerifier<C> for LikePattern<A>
where
    C: Column<Symbol = u8>,
    A: RowLiteralSearch<C>,
{
    fn len_constraint(&self) -> LenConstraint {
        self.len_constraint_internal()
    }

    fn verify(&self, column: &C, row: RowId, _scratch: &mut VerifyScratch) -> bool {
        let row = column.row(row);
        self.matches_row::<C>(&row)
    }
}

fn push_literal<A>(
    tokens: &mut Vec<LikeToken>,
    literals: &mut Vec<CompiledLiteral<A>>,
    min_len: &mut u32,
    src: &str,
) -> Result<(), LikeCompileError>
where
    A: LiteralAlgorithm,
{
    let Some(needle) = A::compile_literal(src) else {
        return Err(LikeCompileError::UnsupportedLiteral {
            fragment: src.to_owned(),
        });
    };
    let len = A::literal_len(&needle);
    let state = A::build_state(&needle);
    let index_symbols = A::index_symbols(&needle);
    let idx = literals.len();
    *min_len = min_len
        .checked_add(len)
        .ok_or(LikeCompileError::LengthOverflow)?;

    literals.push(CompiledLiteral {
        source: src.into(),
        needle,
        state,
        len,
        index_symbols,
    });
    tokens.push(LikeToken::Literal(idx));
    Ok(())
}

fn build_segments<A>(tokens: &[LikeToken], literals: &[CompiledLiteral<A>]) -> Box<[SegmentPlan]>
where
    A: LiteralAlgorithm,
{
    let mut segments = Vec::new();
    let mut start = 0usize;

    while start < tokens.len() {
        let mut end = start;
        while end < tokens.len() && !matches!(tokens[end], LikeToken::Any) {
            end += 1;
        }

        if start < end {
            segments.push(build_segment(tokens, literals, start, end));
        }

        // `end` is either the first `%` after this segment, or `tokens.len()`.
        // Adjacent `%` tokens are collapsed by the compiler, but this also
        // handles leading/trailing `%` by simply skipping empty segments.
        start = end.saturating_add(1);
    }

    segments.into_boxed_slice()
}

fn build_segment<A>(
    tokens: &[LikeToken],
    literals: &[CompiledLiteral<A>],
    token_start: usize,
    token_end: usize,
) -> SegmentPlan
where
    A: LiteralAlgorithm,
{
    let mut len = 0u32;
    let mut first_anchor = None;
    let mut best_anchor = None;
    let mut best_score = (0u32, false);

    for token in &tokens[token_start..token_end] {
        match *token {
            LikeToken::Literal(literal_idx) => {
                let anchor = SegmentAnchor {
                    literal_idx,
                    offset: len,
                };
                first_anchor.get_or_insert(anchor);

                let literal = &literals[literal_idx];
                let exact = literal.index_symbols.is_some();
                let score = (literal.len, exact);
                if best_anchor.is_none() || score > best_score {
                    best_anchor = Some(anchor);
                    best_score = score;
                }

                len = len
                    .checked_add(literal.len)
                    .expect("pattern length was checked while compiling");
            }
            LikeToken::Skip(count) => {
                len = len
                    .checked_add(count)
                    .expect("pattern length was checked while compiling");
            }
            LikeToken::Any => unreachable!("segment builder stops before `%` tokens"),
        }
    }

    SegmentPlan {
        token_start,
        token_end,
        len,
        first_anchor,
        best_anchor,
    }
}

fn max_segment_start(lower: u32, upper: u32, segment_len: u32) -> Option<u32> {
    if lower > upper {
        return None;
    }
    let max_start = upper.checked_sub(segment_len)?;
    if lower <= max_start {
        Some(max_start)
    } else {
        None
    }
}

fn derive_strategy(tokens: &[LikeToken], has_skip: bool) -> MatchStrategy {
    if tokens.is_empty() {
        return MatchStrategy::Exact { literal_idx: None };
    }

    let mut literal_idx = None;
    let mut literal_count = 0usize;
    for token in tokens {
        if let LikeToken::Literal(idx) = *token {
            literal_idx = Some(idx);
            literal_count += 1;
        }
    }

    if literal_count == 0 {
        // The row-level length constraint already distinguishes `_`, `___`,
        // `%_`, `%__%`, etc. from plain `%`. No row contents need scanning.
        return MatchStrategy::All;
    }
    if has_skip {
        return MatchStrategy::General;
    }
    if literal_count > 1 {
        return MatchStrategy::General;
    }

    let literal_idx = literal_idx.expect("literal_count is one");
    let starts_with_any = matches!(tokens.first(), Some(LikeToken::Any));
    let ends_with_any = matches!(tokens.last(), Some(LikeToken::Any));

    match (starts_with_any, ends_with_any) {
        (true, true) => MatchStrategy::Contains { literal_idx },
        (true, false) => MatchStrategy::Suffix { literal_idx },
        (false, true) => MatchStrategy::Prefix { literal_idx },
        (false, false) => MatchStrategy::Exact {
            literal_idx: Some(literal_idx),
        },
    }
}
