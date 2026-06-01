#![allow(unsafe_op_in_unsafe_fn)]

use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::dna2::{Dna2Column, Dna2Row, DnaBase};

/// Algorithm-level wildcard symbol used by DNA2 literal needles.
///
/// Pattern byte `_` compiles to this symbol and matches any DNA base.
pub const DNA_WILDCARD: u8 = 0xFF;

const MAX_BYTE_ANCHORS: usize = 6;
const TARGET_FIXED_BASES: usize = 8;
const SCALAR_LANES: u32 = 8;
const SCALAR_BLOCK_BASES: u32 = SCALAR_LANES * 4;
const BYTE_REPEAT: u64 = 0x0101_0101_0101_0101;
const BYTE_HIGH_BITS: u64 = 0x8080_8080_8080_8080;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dna2PackedChunk {
    /// Number of bases in this chunk, in 1..=32 except there are no chunks for
    /// an empty literal.
    pub bases: u8,
    /// Packed pattern bits. Layout matches `Dna2Row::load_2bit_window`: the
    /// first base is the highest pair in the low `2 * bases` bits.
    pub bits: u64,
    /// `0b11` for bases that must match, `0b00` for wildcard bases.
    pub care_mask: u64,
}

/// Compiled DNA2 literal for packed-byte matching.
///
/// `symbols` stores DNA base codes `0..=3` plus [`DNA_WILDCARD`] for `_`.
/// `chunks` are used by the exact verifier after byte-anchor candidate search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dna2PackedNeedle {
    symbols: Box<[u8]>,
    chunks: Box<[Dna2PackedChunk]>,
    has_wildcard: bool,
}

/// Backwards-compatible needle name for DNA2 literal matching.
pub type Dna2Needle = Dna2PackedNeedle;

impl Dna2PackedNeedle {
    #[inline]
    pub fn symbols(&self) -> &[u8] {
        &self.symbols
    }

    #[inline]
    pub fn chunks(&self) -> &[Dna2PackedChunk] {
        &self.chunks
    }

    #[inline]
    pub fn has_wildcard(&self) -> bool {
        self.has_wildcard
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dna2ByteAnchorPhase {
    /// Byte offset from the candidate start byte for this absolute start phase.
    pub byte_offset: u32,
    /// Pattern bits that must match in the current byte.
    pub cur_pat: u8,
    /// Care mask for the current byte.
    pub cur_mask: u8,
    /// Pattern bits that must match in the next byte when the anchor straddles.
    pub next_pat: u8,
    /// Care mask for the next byte.
    pub next_mask: u8,
}

impl Dna2ByteAnchorPhase {
    pub const EMPTY: Self = Self {
        byte_offset: 0,
        cur_pat: 0,
        cur_mask: 0,
        next_pat: 0,
        next_mask: 0,
    };
}

impl Default for Dna2ByteAnchorPhase {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dna2ByteAnchor {
    /// Pattern offset, in bases, of this up-to-four-base anchor window.
    pub offset: u32,
    /// Number of bases covered by the window, in 1..=4.
    pub bases: u8,
    /// Number of non-wildcard bases in the window.
    pub fixed_count: u8,
    /// Pre-shifted masks for absolute candidate-start phases 0..=3.
    pub phases: [Dna2ByteAnchorPhase; 4],
}

impl Dna2ByteAnchor {
    pub const EMPTY: Self = Self {
        offset: 0,
        bases: 0,
        fixed_count: 0,
        phases: [Dna2ByteAnchorPhase::EMPTY; 4],
    };
}

impl Default for Dna2ByteAnchor {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Search-time state for the packed-byte DNA2 matcher.
///
/// The old single-base/SWAR anchors are intentionally gone. The new state uses
/// up to six byte anchors, where each anchor compares up to four adjacent DNA
/// bases directly against the raw big-endian DNA2 byte stream.
pub struct Dna2PackedState {
    byte_anchors: [Dna2ByteAnchor; MAX_BYTE_ANCHORS],
    byte_anchor_count: u8,
    fixed_base_count: u8,
}

impl Default for Dna2PackedState {
    #[inline]
    fn default() -> Self {
        Self {
            byte_anchors: [Dna2ByteAnchor::EMPTY; MAX_BYTE_ANCHORS],
            byte_anchor_count: 0,
            fixed_base_count: 0,
        }
    }
}

impl Dna2PackedState {
    #[inline]
    pub fn byte_anchors(&self) -> &[Dna2ByteAnchor] {
        &self.byte_anchors[..self.byte_anchor_count as usize]
    }

    /// Compatibility accessor for code that previously inspected packed state anchors.
    #[inline]
    pub fn anchors(&self) -> &[Dna2ByteAnchor] {
        self.byte_anchors()
    }

    #[inline]
    pub fn has_fixed_bases(&self) -> bool {
        self.byte_anchor_count != 0
    }

    #[inline]
    pub fn fixed_base_count(&self) -> u8 {
        self.fixed_base_count
    }
}

/// Scalar packed-byte DNA2 matcher.
///
/// `find_from` scans candidates in 32-base blocks using raw packed bytes and a
/// scalar zero-byte test. Candidate hits are verified by packed 32-base chunks.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dna2PackedScalar;

/// Runtime-dispatched SIMD packed-byte DNA2 matcher.
///
/// On x86-64 this uses AVX2 when available. On AArch64 it uses NEON. If the
/// binary is compiled with AVX-512BW enabled, an AVX-512BW path is also used.
/// All paths share the same scalar packed verifier and wildcard semantics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dna2PackedVectorized;

/// Preferred short name for the new DNA2 literal backend.
pub type Dna2 = Dna2PackedVectorized;

macro_rules! impl_packed_literal_algorithm {
    ($ty:ty) => {
        impl LiteralAlgorithm for $ty {
            type Needle = Dna2PackedNeedle;
            type State = Dna2PackedState;

            const SUPPORTS_UNDERSCORE: bool = true;

            fn compile_literal(src: &str) -> Option<Self::Needle> {
                let (symbols, has_wildcard) = compile_symbols(src)?;
                let chunks = build_packed_chunks(&symbols);
                Some(Dna2PackedNeedle {
                    symbols,
                    chunks,
                    has_wildcard,
                })
            }

            #[inline]
            fn build_state(needle: &Self::Needle) -> Self::State {
                build_packed_state(needle.symbols())
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
    };
}

impl_packed_literal_algorithm!(Dna2PackedScalar);
impl_packed_literal_algorithm!(Dna2PackedVectorized);

impl<'db> RowLiteralSearch<Dna2Column<'db>> for Dna2PackedScalar {
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
        packed_matches_at(row, pos, needle)
    }

    #[inline]
    fn find_from<'r>(
        row: &Dna2Row<'r>,
        from: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> Option<u32> {
        packed_find_from_scalar(row, from, needle, state)
    }
}

impl<'db> RowLiteralSearch<Dna2Column<'db>> for Dna2PackedVectorized {
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
        packed_matches_at(row, pos, needle)
    }

    #[inline]
    fn find_from<'r>(
        row: &Dna2Row<'r>,
        from: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> Option<u32> {
        packed_find_from_vectorized(row, from, needle, state)
    }
}

#[inline]
fn compile_symbols(src: &str) -> Option<(Box<[u8]>, bool)> {
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

    Some((symbols.into_boxed_slice(), has_wildcard))
}

#[inline]
fn build_packed_chunks(symbols: &[u8]) -> Box<[Dna2PackedChunk]> {
    if symbols.is_empty() {
        return Vec::<Dna2PackedChunk>::new().into_boxed_slice();
    }

    let mut out = Vec::with_capacity((symbols.len() + 31) / 32);
    for chunk in symbols.chunks(32) {
        let mut bits = 0u64;
        let mut care_mask = 0u64;

        for &sym in chunk {
            bits <<= 2;
            care_mask <<= 2;
            if sym != DNA_WILDCARD {
                debug_assert!(sym < 4);
                bits |= u64::from(sym);
                care_mask |= 0b11;
            }
        }

        out.push(Dna2PackedChunk {
            bases: chunk.len() as u8,
            bits,
            care_mask,
        });
    }

    out.into_boxed_slice()
}

fn build_packed_state(symbols: &[u8]) -> Dna2PackedState {
    let mut state = Dna2PackedState::default();
    if symbols.is_empty() {
        return state;
    }

    let fixed_total = symbols.iter().filter(|&&sym| sym != DNA_WILDCARD).count();
    if fixed_total == 0 {
        return state;
    }

    let mut covered = vec![false; symbols.len()];
    let mut selected_fixed = 0usize;

    while (state.byte_anchor_count as usize) < MAX_BYTE_ANCHORS && selected_fixed < fixed_total {
        let mut best_offset = 0usize;
        let mut best_bases = 0usize;
        let mut best_new_fixed = 0usize;
        let mut best_fixed = 0usize;

        for offset in 0..symbols.len() {
            let bases = core::cmp::min(4, symbols.len() - offset);
            let mut fixed = 0usize;
            let mut new_fixed = 0usize;

            for i in 0..bases {
                if symbols[offset + i] != DNA_WILDCARD {
                    fixed += 1;
                    if !covered[offset + i] {
                        new_fixed += 1;
                    }
                }
            }

            if new_fixed == 0 {
                continue;
            }

            let better = new_fixed > best_new_fixed
                || (new_fixed == best_new_fixed && fixed > best_fixed)
                || (new_fixed == best_new_fixed
                    && fixed == best_fixed
                    && prefer_anchor_offset(offset, symbols.len(), best_offset));

            if better {
                best_offset = offset;
                best_bases = bases;
                best_new_fixed = new_fixed;
                best_fixed = fixed;
            }
        }

        if best_new_fixed == 0 {
            break;
        }

        let anchor = build_byte_anchor(symbols, best_offset, best_bases);
        let slot = state.byte_anchor_count as usize;
        state.byte_anchors[slot] = anchor;
        state.byte_anchor_count += 1;
        state.fixed_base_count = state.fixed_base_count.saturating_add(anchor.fixed_count);

        for i in 0..best_bases {
            if symbols[best_offset + i] != DNA_WILDCARD && !covered[best_offset + i] {
                covered[best_offset + i] = true;
                selected_fixed += 1;
            }
        }

        if selected_fixed >= TARGET_FIXED_BASES {
            break;
        }
    }

    state
}

#[inline]
fn prefer_anchor_offset(candidate: usize, len: usize, current: usize) -> bool {
    fn rank(offset: usize, len: usize) -> usize {
        let mid = len / 2;
        let d_mid = offset.abs_diff(mid);
        let d_start = offset;
        let d_end = len.saturating_sub(1).saturating_sub(offset);
        core::cmp::min(d_mid, core::cmp::min(d_start, d_end))
    }

    rank(candidate, len) < rank(current, len)
        || (rank(candidate, len) == rank(current, len) && candidate < current)
}

fn build_byte_anchor(symbols: &[u8], offset: usize, bases: usize) -> Dna2ByteAnchor {
    debug_assert!((1..=4).contains(&bases));
    debug_assert!(offset + bases <= symbols.len());

    let mut fixed_count = 0u8;
    let mut phases = [Dna2ByteAnchorPhase::EMPTY; 4];

    for abs_start_phase in 0..4usize {
        let mut phase = Dna2ByteAnchorPhase {
            byte_offset: ((abs_start_phase + offset) / 4) as u32,
            ..Dna2ByteAnchorPhase::EMPTY
        };

        let first_slot = (abs_start_phase + offset) & 3;
        for i in 0..bases {
            let sym = symbols[offset + i];
            if sym == DNA_WILDCARD {
                continue;
            }

            if abs_start_phase == 0 {
                fixed_count += 1;
            }

            let slot = first_slot + i;
            let physical_slot = slot & 3;
            let shift = 6 - 2 * physical_slot;
            let pat = sym << shift;
            let mask = 0b11u8 << shift;

            if slot < 4 {
                phase.cur_pat |= pat;
                phase.cur_mask |= mask;
            } else {
                phase.next_pat |= pat;
                phase.next_mask |= mask;
            }
        }

        phases[abs_start_phase] = phase;
    }

    Dna2ByteAnchor {
        offset: offset as u32,
        bases: bases as u8,
        fixed_count,
        phases,
    }
}

#[inline]
fn checked_search_bounds(
    row: &Dna2Row<'_>,
    from: u32,
    needle_len: usize,
) -> Option<(u32, u32, u32)> {
    let text_len = row.len_bases();
    let needle_len = needle_len as u32;

    if from > text_len {
        return None;
    }
    if needle_len == 0 {
        return Some((text_len, 0, from));
    }
    if needle_len > text_len.saturating_sub(from) {
        return None;
    }

    Some((text_len, needle_len, text_len - needle_len))
}

#[inline(always)]
fn has_full_block(pos: u32, last_start: u32, block_bases: u32) -> bool {
    debug_assert!(block_bases > 0);
    pos <= last_start && last_start - pos >= block_bases - 1
}

#[inline]
fn packed_matches_at(row: &Dna2Row<'_>, pos: u32, needle: &Dna2PackedNeedle) -> bool {
    let len = needle.symbols.len() as u32;
    let Some(end) = pos.checked_add(len) else {
        return false;
    };
    if end > row.len_bases() {
        return false;
    }

    for (chunk_idx, chunk) in needle.chunks.iter().enumerate() {
        if chunk.care_mask == 0 {
            continue;
        }

        let chunk_pos = pos + (chunk_idx as u32) * 32;
        let Some(row_bits) = row.load_2bit_window(chunk_pos, u32::from(chunk.bases)) else {
            return false;
        };

        if ((row_bits ^ chunk.bits) & chunk.care_mask) != 0 {
            return false;
        }
    }

    true
}

fn packed_find_from_scalar(
    row: &Dna2Row<'_>,
    from: u32,
    needle: &Dna2PackedNeedle,
    state: &Dna2PackedState,
) -> Option<u32> {
    let (_, needle_len, last_start) = checked_search_bounds(row, from, needle.symbols.len())?;
    if needle_len == 0 {
        return Some(from);
    }
    if !state.has_fixed_bases() {
        return Some(from);
    }

    let mut pos = from;
    match scan_until_phase_zero(row, pos, last_start, needle, state) {
        PrefixScan::Found(found) => return Some(found),
        PrefixScan::Continue(next) => pos = next,
        PrefixScan::Exhausted => return None,
    }

    while has_full_block(pos, last_start, SCALAR_BLOCK_BASES) {
        let masks = unsafe { candidate_masks_block_scalar8(row, pos, state) };
        if let Some(found) = first_verified_from_phase_masks(row, pos, last_start, needle, masks) {
            return Some(found);
        }
        pos += SCALAR_BLOCK_BASES;
    }

    scan_tail_scalar(row, pos, last_start, needle, state)
}

#[inline]
fn packed_find_from_vectorized(
    row: &Dna2Row<'_>,
    from: u32,
    needle: &Dna2PackedNeedle,
    state: &Dna2PackedState,
) -> Option<u32> {
    if !state.has_fixed_bases() || needle.symbols.is_empty() {
        return packed_find_from_scalar(row, from, needle, state);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        if std::is_x86_feature_detected!("avx512bw") {
            return unsafe { packed_find_from_avx512bw(row, from, needle, state) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { packed_find_from_avx2(row, from, needle, state) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { packed_find_from_neon(row, from, needle, state) };
    }

    #[allow(unreachable_code)]
    packed_find_from_scalar(row, from, needle, state)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixScan {
    Found(u32),
    Continue(u32),
    Exhausted,
}

fn scan_until_phase_zero(
    row: &Dna2Row<'_>,
    mut pos: u32,
    last_start: u32,
    needle: &Dna2PackedNeedle,
    state: &Dna2PackedState,
) -> PrefixScan {
    while pos <= last_start && ((row.start_base_offset() + u64::from(pos)) & 3) != 0 {
        if byte_anchors_match_at_scalar(row, pos, state) && packed_matches_at(row, pos, needle) {
            return PrefixScan::Found(pos);
        }
        if pos == last_start {
            return PrefixScan::Exhausted;
        }
        pos += 1;
    }

    if pos <= last_start {
        PrefixScan::Continue(pos)
    } else {
        PrefixScan::Exhausted
    }
}

fn scan_tail_scalar(
    row: &Dna2Row<'_>,
    mut pos: u32,
    last_start: u32,
    needle: &Dna2PackedNeedle,
    state: &Dna2PackedState,
) -> Option<u32> {
    while pos <= last_start {
        if byte_anchors_match_at_scalar(row, pos, state) && packed_matches_at(row, pos, needle) {
            return Some(pos);
        }
        if pos == last_start {
            break;
        }
        pos += 1;
    }

    None
}

#[inline]
fn byte_anchors_match_at_scalar(row: &Dna2Row<'_>, pos: u32, state: &Dna2PackedState) -> bool {
    let absolute = row.start_base_offset() + u64::from(pos);
    let phase = (absolute & 3) as usize;
    let base_byte = (absolute >> 2) as usize;
    let payload = row.packed_payload();

    for anchor in state.byte_anchors() {
        let p = anchor.phases[phase];
        let idx = base_byte + p.byte_offset as usize;

        if p.cur_mask != 0 {
            let Some(&cur) = payload.get(idx) else {
                return false;
            };
            if ((cur ^ p.cur_pat) & p.cur_mask) != 0 {
                return false;
            }
        }

        if p.next_mask != 0 {
            let Some(&next) = payload.get(idx + 1) else {
                return false;
            };
            if ((next ^ p.next_pat) & p.next_mask) != 0 {
                return false;
            }
        }
    }

    true
}

unsafe fn candidate_masks_block_scalar8(
    row: &Dna2Row<'_>,
    pos: u32,
    state: &Dna2PackedState,
) -> [u64; 4] {
    debug_assert_eq!((row.start_base_offset() + u64::from(pos)) & 3, 0);

    let base_byte = ((row.start_base_offset() + u64::from(pos)) >> 2) as usize;
    let payload = row.packed_payload();
    let ptr = payload.as_ptr().add(base_byte);
    let all = (1u64 << SCALAR_LANES) - 1;
    let mut out = [0u64; 4];

    for phase in 0..4usize {
        let mut mask = all;
        for anchor in state.byte_anchors() {
            let p = anchor.phases[phase];
            let anchor_ptr = ptr.add(p.byte_offset as usize);
            mask &= u64::from(cmp_phase_anchor_scalar8(anchor_ptr, p));
            if mask == 0 {
                break;
            }
        }
        out[phase] = mask;
    }

    out
}

#[inline(always)]
unsafe fn cmp_phase_anchor_scalar8(ptr: *const u8, p: Dna2ByteAnchorPhase) -> u8 {
    let mut diff = 0u64;

    if p.cur_mask != 0 {
        let cur = load_u64_le(ptr);
        diff |= (cur ^ repeat_byte(p.cur_pat)) & repeat_byte(p.cur_mask);
    }

    if p.next_mask != 0 {
        let next = load_u64_le(ptr.add(1));
        diff |= (next ^ repeat_byte(p.next_pat)) & repeat_byte(p.next_mask);
    }

    zero_byte_mask8(diff)
}

#[inline(always)]
unsafe fn load_u64_le(ptr: *const u8) -> u64 {
    let mut bytes = [0u8; 8];
    core::ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), 8);
    u64::from_le_bytes(bytes)
}

#[inline(always)]
fn repeat_byte(byte: u8) -> u64 {
    u64::from(byte) * BYTE_REPEAT
}

#[inline(always)]
fn zero_byte_mask8(x: u64) -> u8 {
    let high = x.wrapping_sub(BYTE_REPEAT) & !x & BYTE_HIGH_BITS;
    let mut mask = 0u8;
    let mut lane = 0u32;
    while lane < 8 {
        if ((high >> (lane * 8 + 7)) & 1) != 0 {
            mask |= 1u8 << lane;
        }
        lane += 1;
    }
    mask
}

fn first_verified_from_phase_masks(
    row: &Dna2Row<'_>,
    block_pos: u32,
    last_start: u32,
    needle: &Dna2PackedNeedle,
    mut masks: [u64; 4],
) -> Option<u32> {
    loop {
        let mut best_phase = 4usize;
        let mut best_lane = 0u32;
        let mut best_pos = u32::MAX;

        for (phase, &mask) in masks.iter().enumerate() {
            if mask == 0 {
                continue;
            }
            let lane = mask.trailing_zeros();
            let cand = block_pos + phase as u32 + 4 * lane;
            if cand < best_pos {
                best_pos = cand;
                best_phase = phase;
                best_lane = lane;
            }
        }

        if best_phase == 4 {
            return None;
        }

        masks[best_phase] &= !(1u64 << best_lane);
        if best_pos <= last_start && packed_matches_at(row, best_pos, needle) {
            return Some(best_pos);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn packed_find_from_avx2(
    row: &Dna2Row<'_>,
    from: u32,
    needle: &Dna2PackedNeedle,
    state: &Dna2PackedState,
) -> Option<u32> {
    const LANES: u32 = 32;
    const BLOCK_BASES: u32 = LANES * 4;

    let (_, needle_len, last_start) = checked_search_bounds(row, from, needle.symbols.len())?;
    if needle_len == 0 {
        return Some(from);
    }
    if !state.has_fixed_bases() {
        return Some(from);
    }

    let mut pos = match scan_until_phase_zero(row, from, last_start, needle, state) {
        PrefixScan::Found(found) => return Some(found),
        PrefixScan::Continue(next) => next,
        PrefixScan::Exhausted => return None,
    };

    while has_full_block(pos, last_start, BLOCK_BASES) {
        let masks = candidate_masks_block_avx2(row, pos, state);
        if let Some(found) = first_verified_from_phase_masks(row, pos, last_start, needle, masks) {
            return Some(found);
        }
        pos += BLOCK_BASES;
    }

    scan_tail_scalar(row, pos, last_start, needle, state)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn candidate_masks_block_avx2(
    row: &Dna2Row<'_>,
    pos: u32,
    state: &Dna2PackedState,
) -> [u64; 4] {
    debug_assert_eq!((row.start_base_offset() + u64::from(pos)) & 3, 0);

    let base_byte = ((row.start_base_offset() + u64::from(pos)) >> 2) as usize;
    let payload = row.packed_payload();
    let ptr = payload.as_ptr().add(base_byte);
    let mut out = [0u64; 4];

    for phase in 0..4usize {
        let mut mask = u32::MAX;
        for anchor in state.byte_anchors() {
            let p = anchor.phases[phase];
            let anchor_ptr = ptr.add(p.byte_offset as usize);
            mask &= cmp_phase_anchor_avx2(anchor_ptr, p);
            if mask == 0 {
                break;
            }
        }
        out[phase] = u64::from(mask);
    }

    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cmp_phase_anchor_avx2(ptr: *const u8, p: Dna2ByteAnchorPhase) -> u32 {
    use core::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let mut diff = zero;

    if p.cur_mask != 0 {
        let cur = _mm256_loadu_si256(ptr.cast::<__m256i>());
        let d = _mm256_and_si256(
            _mm256_xor_si256(cur, _mm256_set1_epi8(p.cur_pat as i8)),
            _mm256_set1_epi8(p.cur_mask as i8),
        );
        diff = _mm256_or_si256(diff, d);
    }

    if p.next_mask != 0 {
        let next = _mm256_loadu_si256(ptr.add(1).cast::<__m256i>());
        let d = _mm256_and_si256(
            _mm256_xor_si256(next, _mm256_set1_epi8(p.next_pat as i8)),
            _mm256_set1_epi8(p.next_mask as i8),
        );
        diff = _mm256_or_si256(diff, d);
    }

    _mm256_movemask_epi8(_mm256_cmpeq_epi8(diff, zero)) as u32
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn packed_find_from_avx512bw(
    row: &Dna2Row<'_>,
    from: u32,
    needle: &Dna2PackedNeedle,
    state: &Dna2PackedState,
) -> Option<u32> {
    const LANES: u32 = 64;
    const BLOCK_BASES: u32 = LANES * 4;

    let (_, needle_len, last_start) = checked_search_bounds(row, from, needle.symbols.len())?;
    if needle_len == 0 {
        return Some(from);
    }
    if !state.has_fixed_bases() {
        return Some(from);
    }

    let mut pos = match scan_until_phase_zero(row, from, last_start, needle, state) {
        PrefixScan::Found(found) => return Some(found),
        PrefixScan::Continue(next) => next,
        PrefixScan::Exhausted => return None,
    };

    while has_full_block(pos, last_start, BLOCK_BASES) {
        let masks = candidate_masks_block_avx512bw(row, pos, state);
        if let Some(found) = first_verified_from_phase_masks(row, pos, last_start, needle, masks) {
            return Some(found);
        }
        pos += BLOCK_BASES;
    }

    scan_tail_scalar(row, pos, last_start, needle, state)
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn candidate_masks_block_avx512bw(
    row: &Dna2Row<'_>,
    pos: u32,
    state: &Dna2PackedState,
) -> [u64; 4] {
    debug_assert_eq!((row.start_base_offset() + u64::from(pos)) & 3, 0);

    let base_byte = ((row.start_base_offset() + u64::from(pos)) >> 2) as usize;
    let payload = row.packed_payload();
    let ptr = payload.as_ptr().add(base_byte);
    let mut out = [0u64; 4];

    for phase in 0..4usize {
        let mut mask = u64::MAX;
        for anchor in state.byte_anchors() {
            let p = anchor.phases[phase];
            let anchor_ptr = ptr.add(p.byte_offset as usize);
            mask &= cmp_phase_anchor_avx512bw(anchor_ptr, p);
            if mask == 0 {
                break;
            }
        }
        out[phase] = mask;
    }

    out
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn cmp_phase_anchor_avx512bw(ptr: *const u8, p: Dna2ByteAnchorPhase) -> u64 {
    use core::arch::x86_64::*;

    let mut mask = u64::MAX;

    if p.cur_mask != 0 {
        let cur = _mm512_loadu_si512(ptr.cast::<__m512i>());
        let k = _mm512_testn_epi8_mask(
            _mm512_xor_si512(cur, _mm512_set1_epi8(p.cur_pat as i8)),
            _mm512_set1_epi8(p.cur_mask as i8),
        );
        mask &= k as u64;
    }

    if p.next_mask != 0 {
        let next = _mm512_loadu_si512(ptr.add(1).cast::<__m512i>());
        let k = _mm512_testn_epi8_mask(
            _mm512_xor_si512(next, _mm512_set1_epi8(p.next_pat as i8)),
            _mm512_set1_epi8(p.next_mask as i8),
        );
        mask &= k as u64;
    }

    mask
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn packed_find_from_neon(
    row: &Dna2Row<'_>,
    from: u32,
    needle: &Dna2PackedNeedle,
    state: &Dna2PackedState,
) -> Option<u32> {
    const LANES: u32 = 16;
    const BLOCK_BASES: u32 = LANES * 4;

    let (_, needle_len, last_start) = checked_search_bounds(row, from, needle.symbols.len())?;
    if needle_len == 0 {
        return Some(from);
    }
    if !state.has_fixed_bases() {
        return Some(from);
    }

    let mut pos = match scan_until_phase_zero(row, from, last_start, needle, state) {
        PrefixScan::Found(found) => return Some(found),
        PrefixScan::Continue(next) => next,
        PrefixScan::Exhausted => return None,
    };

    while has_full_block(pos, last_start, BLOCK_BASES) {
        let masks = candidate_masks_block_neon(row, pos, state);
        if let Some(found) = first_verified_from_phase_masks(row, pos, last_start, needle, masks) {
            return Some(found);
        }
        pos += BLOCK_BASES;
    }

    scan_tail_scalar(row, pos, last_start, needle, state)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn candidate_masks_block_neon(
    row: &Dna2Row<'_>,
    pos: u32,
    state: &Dna2PackedState,
) -> [u64; 4] {
    debug_assert_eq!((row.start_base_offset() + u64::from(pos)) & 3, 0);

    let base_byte = ((row.start_base_offset() + u64::from(pos)) >> 2) as usize;
    let payload = row.packed_payload();
    let ptr = payload.as_ptr().add(base_byte);
    let mut out = [0u64; 4];

    for phase in 0..4usize {
        let mut mask = 0xFFFFu16;
        for anchor in state.byte_anchors() {
            let p = anchor.phases[phase];
            let anchor_ptr = ptr.add(p.byte_offset as usize);
            mask &= cmp_phase_anchor_neon(anchor_ptr, p);
            if mask == 0 {
                break;
            }
        }
        out[phase] = u64::from(mask);
    }

    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cmp_phase_anchor_neon(ptr: *const u8, p: Dna2ByteAnchorPhase) -> u16 {
    use core::arch::aarch64::*;

    let zero = vdupq_n_u8(0);
    let mut diff = zero;

    if p.cur_mask != 0 {
        let cur = vld1q_u8(ptr);
        let d = vandq_u8(veorq_u8(cur, vdupq_n_u8(p.cur_pat)), vdupq_n_u8(p.cur_mask));
        diff = vorrq_u8(diff, d);
    }

    if p.next_mask != 0 {
        let next = vld1q_u8(ptr.add(1));
        let d = vandq_u8(
            veorq_u8(next, vdupq_n_u8(p.next_pat)),
            vdupq_n_u8(p.next_mask),
        );
        diff = vorrq_u8(diff, d);
    }

    neon_movemask_u8(vceqq_u8(diff, zero))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_movemask_u8(v: core::arch::aarch64::uint8x16_t) -> u16 {
    use core::arch::aarch64::*;

    let mut tmp = [0u8; 16];
    vst1q_u8(tmp.as_mut_ptr(), v);

    let mut mask = 0u16;
    for (idx, &byte) in tmp.iter().enumerate() {
        if byte & 0x80 != 0 {
            mask |= 1u16 << idx;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::like::{LikePattern, RowLiteralSearch};
    use crate::storage::dna2::Dna2TableBuilder;
    use crate::storage::Column;
    use crate::{execute_like, DbBuilder, FullScan, QueryScratch, RowId};

    fn one_row(seq: &str) -> (crate::Db, crate::TableId) {
        let mut table = Dna2TableBuilder::new("dna");
        table.push_str(seq).unwrap();
        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(table).unwrap();
        (dbb.freeze(), id)
    }

    fn reference_matches_at(row: &Dna2Row<'_>, pos: u32, symbols: &[u8]) -> bool {
        let len = symbols.len() as u32;
        let Some(end) = pos.checked_add(len) else {
            return false;
        };
        if end > row.len_bases() {
            return false;
        }

        for (idx, &want) in symbols.iter().enumerate() {
            if want == DNA_WILDCARD {
                continue;
            }
            if row.base_code_at(pos + idx as u32) != want {
                return false;
            }
        }
        true
    }

    fn reference_find_from(row: &Dna2Row<'_>, from: u32, symbols: &[u8]) -> Option<u32> {
        let text_len = row.len_bases();
        let needle_len = symbols.len() as u32;

        if from > text_len {
            return None;
        }
        if needle_len == 0 {
            return Some(from);
        }
        if needle_len > text_len.saturating_sub(from) {
            return None;
        }

        let last = text_len - needle_len;
        let mut pos = from;
        while pos <= last {
            if reference_matches_at(row, pos, symbols) {
                return Some(pos);
            }
            if pos == last {
                break;
            }
            pos += 1;
        }
        None
    }

    fn check_algo<A>()
    where
        A: LiteralAlgorithm<Needle = Dna2PackedNeedle, State = Dna2PackedState>,
        for<'db> A: RowLiteralSearch<Dna2Column<'db>>,
    {
        let cases = [
            ("ACGTACGT", "ACG"),
            ("ACGTACGT", "A_G"),
            ("ACGTACGT", "A__T"),
            ("ACGTACGT", "_CG_"),
            ("ACGTACGT", "TAC"),
            ("AAAAAA", "AAA"),
            ("AAAAAA", "A_A"),
            ("ACGT", "____"),
            ("ACGT", "TTT"),
            ("", ""),
            ("", "_"),
            ("ACGTACGTACGTACGT", "ACGTACGT"),
            ("ACGTACGTACGTACGT", "A___A___"),
            ("TTTTACGTAAAA", "ACGT"),
        ];

        for (text, pat) in cases {
            let (db, id) = one_row(text);
            let table = db.dna2_table(id).unwrap();
            let col = table.sequence();
            let row = col.row_view(0);
            let needle = A::compile_literal(pat).unwrap();
            let state = A::build_state(&needle);

            for from in 0..=(text.len() as u32 + 1) {
                let got = A::find_from(&row, from, &needle, &state);
                let expect = reference_find_from(&row, from, needle.symbols());
                assert_eq!(
                    got,
                    expect,
                    "find mismatch: algo={}, text={text:?}, pat={pat:?}, from={from}",
                    core::any::type_name::<A>()
                );
            }

            for pos in 0..=(text.len() as u32 + 1) {
                let got = A::matches_at(&row, pos, &needle, &state);
                let expect = reference_matches_at(&row, pos, needle.symbols());
                assert_eq!(
                    got,
                    expect,
                    "match mismatch: algo={}, text={text:?}, pat={pat:?}, pos={pos}",
                    core::any::type_name::<A>()
                );
            }
        }
    }

    #[test]
    fn packed_scalar_matches_reference() {
        check_algo::<Dna2PackedScalar>();
    }

    #[test]
    fn packed_vectorized_matches_reference() {
        check_algo::<Dna2PackedVectorized>();
    }

    #[test]
    fn unaligned_rows_match_reference() {
        let mut table = Dna2TableBuilder::new("dna");
        table.push_str("A").unwrap();
        table.push_str("ACGTACGTACGTACGTACGT").unwrap();
        table.push_str("AC").unwrap();
        table.push_str("TTTACGTAAAACCCGGGTTT").unwrap();

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(table).unwrap();
        let db = dbb.freeze();
        let table = db.dna2_table(id).unwrap();
        let col = table.sequence();

        let patterns = ["ACGT", "A__T", "_CG_", "TTTA", "AAAAC", "GGG_T"];
        for row_id in 0..col.row_count() {
            let row = col.row_view(row_id);
            for pat in patterns {
                let needle = Dna2PackedVectorized::compile_literal(pat).unwrap();
                let state = Dna2PackedVectorized::build_state(&needle);
                for from in 0..=(row.len_bases() + 1) {
                    let got = Dna2PackedVectorized::find_from(&row, from, &needle, &state);
                    let expect = reference_find_from(&row, from, needle.symbols());
                    assert_eq!(got, expect, "row={row_id}, pat={pat:?}, from={from}");
                }
            }
        }
    }

    #[test]
    fn random_scalar_and_vectorized_match_reference() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *seed
        }

        fn base(seed: &mut u64) -> u8 {
            b"ACGT"[(next(seed) & 3) as usize]
        }

        fn pat_byte(seed: &mut u64) -> u8 {
            if next(seed) % 5 == 0 {
                b'_'
            } else {
                base(seed)
            }
        }

        let mut seed = 7u64;
        for case_idx in 0..2_000 {
            let text_len = (next(&mut seed) % 192) as usize;
            let pat_len = (next(&mut seed) % 48) as usize;
            let from = if text_len == 0 {
                0u32
            } else {
                (next(&mut seed) % (text_len as u64 + 2)) as u32
            };

            let mut text = Vec::with_capacity(text_len);
            for _ in 0..text_len {
                text.push(base(&mut seed));
            }

            let mut pat = Vec::with_capacity(pat_len);
            for _ in 0..pat_len {
                pat.push(pat_byte(&mut seed));
            }

            let text_s = std::str::from_utf8(&text).unwrap();
            let pat_s = std::str::from_utf8(&pat).unwrap();
            let (db, id) = one_row(text_s);
            let table = db.dna2_table(id).unwrap();
            let col = table.sequence();
            let row = col.row_view(0);

            let scalar_needle = Dna2PackedScalar::compile_literal(pat_s).unwrap();
            let scalar_state = Dna2PackedScalar::build_state(&scalar_needle);
            let vector_needle = Dna2PackedVectorized::compile_literal(pat_s).unwrap();
            let vector_state = Dna2PackedVectorized::build_state(&vector_needle);

            let expect = reference_find_from(&row, from, scalar_needle.symbols());
            let got_scalar = Dna2PackedScalar::find_from(&row, from, &scalar_needle, &scalar_state);
            let got_vector =
                Dna2PackedVectorized::find_from(&row, from, &vector_needle, &vector_state);

            assert_eq!(
                got_scalar, expect,
                "scalar random mismatch case={case_idx}, text={text_s:?}, pat={pat_s:?}, from={from}"
            );
            assert_eq!(
                got_vector, expect,
                "vector random mismatch case={case_idx}, text={text_s:?}, pat={pat_s:?}, from={from}"
            );
        }
    }

    #[test]
    fn like_integration() {
        let mut reads = Dna2TableBuilder::new("reads");
        reads.push_str("ACGT").unwrap();
        reads.push_str("AGGT").unwrap();
        reads.push_str("TTTT").unwrap();
        reads.push_str("AACGAAAA").unwrap();

        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(reads).unwrap();
        let db = dbb.freeze();
        let table = db.dna2_table(id).unwrap();
        let col = table.sequence();

        let expected: Vec<RowId> = vec![0, 1];

        let like = LikePattern::<Dna2PackedVectorized>::compile("A_G%").unwrap();
        let mut scan = FullScan::new(col.row_count(), 16);
        let mut scratch = QueryScratch::default();
        let mut matches = Vec::<RowId>::new();
        execute_like(&col, &mut scan, &like, &mut scratch, &mut matches);
        assert_eq!(matches, expected);

        let like = LikePattern::<Dna2PackedScalar>::compile("A_G%").unwrap();
        let mut scan = FullScan::new(col.row_count(), 16);
        let mut scratch = QueryScratch::default();
        let mut matches = Vec::<RowId>::new();
        execute_like(&col, &mut scan, &like, &mut scratch, &mut matches);
        assert_eq!(matches, expected);
    }
}
