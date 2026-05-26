use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::dna2::{Dna2Column, Dna2Row, DnaBase};

pub const DNA_WILDCARD: u8 = 0xFF;
const LOW_2BIT_LANES: u64 = 0x5555_5555_5555_5555;

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

/// Reference DNA2 matcher.
///
/// This intentionally remains the simple baseline: it reads one logical base at
/// a time from the packed row. Keep it around for correctness and performance
/// comparisons against the packed matchers below.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dna2NaiveWildcard;

impl LiteralAlgorithm for Dna2NaiveWildcard {
    type Needle = Dna2Needle;
    type State = ();

    const SUPPORTS_UNDERSCORE: bool = true;

    fn compile_literal(src: &str) -> Option<Self::Needle> {
        let (symbols, has_wildcard) = compile_symbols(src)?;
        Some(Dna2Needle {
            symbols,
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
        naive_matches_at(row, pos, needle.symbols())
    }

    #[inline]
    fn find_from<'r>(
        row: &Dna2Row<'r>,
        from: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> Option<u32> {
        naive_find_from(row, from, needle.symbols())
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dna2PackedNeedle {
    symbols: Box<[u8]>,
    chunks: Box<[Dna2PackedChunk]>,
    has_wildcard: bool,
}

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
pub struct Dna2PackedState {
    anchors: [(u32, u8); 8],
    anchor_count: u8,
}

impl Dna2PackedState {
    #[inline]
    pub fn anchors(&self) -> &[(u32, u8)] {
        &self.anchors[..self.anchor_count as usize]
    }

    #[inline]
    pub fn has_fixed_bases(&self) -> bool {
        self.anchor_count != 0
    }
}

/// Fast scalar packed DNA2 matcher.
///
/// `matches_at` compares packed 2-bit chunks with
/// `((row_bits ^ needle_bits) & care_mask) == 0`. `find_from` still walks
/// candidate starts one-by-one, but it uses a few fixed-base anchors before the
/// packed verification.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dna2PackedScalar;

/// Fast packed DNA2 matcher with 32-start candidate blocks.
///
/// The search loop keeps a 32-lane candidate bit mask over packed 2-bit row
/// windows. On x86_64 it uses SSE2 for two-anchor candidate-mask updates and
/// two-chunk verification. On aarch64 it uses NEON for the same jobs. Other
/// targets use the same packed SWAR logic without platform intrinsics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dna2PackedVectorized;

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
        packed_matches_at_scalar(row, pos, needle)
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
        packed_matches_at_vectorized(row, pos, needle)
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
fn naive_matches_at(row: &Dna2Row<'_>, pos: u32, symbols: &[u8]) -> bool {
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

#[inline]
fn naive_find_from(row: &Dna2Row<'_>, from: u32, symbols: &[u8]) -> Option<u32> {
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

    let last_start = text_len - needle_len;
    let mut pos = from;
    while pos <= last_start {
        if naive_matches_at(row, pos, symbols) {
            return Some(pos);
        }
        pos += 1;
    }
    None
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

#[inline]
fn build_packed_state(symbols: &[u8]) -> Dna2PackedState {
    let mut anchors = [(0u32, 0u8); 8];
    let mut count = 0u8;

    fn push_anchor(anchors: &mut [(u32, u8); 8], count: &mut u8, offset: usize, symbol: u8) {
        if symbol == DNA_WILDCARD || (*count as usize) >= anchors.len() {
            return;
        }
        let off = offset as u32;
        if anchors[..*count as usize]
            .iter()
            .any(|&(old, _)| old == off)
        {
            return;
        }
        anchors[*count as usize] = (off, symbol);
        *count += 1;
    }

    if !symbols.is_empty() {
        let len = symbols.len();
        let preferred = [0usize, len - 1, len / 2, len / 4, (len * 3) / 4];
        for &idx in &preferred {
            push_anchor(&mut anchors, &mut count, idx, symbols[idx]);
        }

        if (count as usize) < anchors.len() {
            for (idx, &sym) in symbols.iter().enumerate() {
                push_anchor(&mut anchors, &mut count, idx, sym);
                if (count as usize) == anchors.len() {
                    break;
                }
            }
        }
    }

    Dna2PackedState {
        anchors,
        anchor_count: count,
    }
}

#[inline]
fn packed_matches_at_scalar(row: &Dna2Row<'_>, pos: u32, needle: &Dna2PackedNeedle) -> bool {
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

#[inline]
fn packed_matches_at_vectorized(row: &Dna2Row<'_>, pos: u32, needle: &Dna2PackedNeedle) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is baseline on x86_64.
        return unsafe { packed_matches_at_sse2(row, pos, needle) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        return unsafe { packed_matches_at_neon(row, pos, needle) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        packed_matches_at_scalar(row, pos, needle)
    }
}

#[inline]
fn packed_find_from_scalar(
    row: &Dna2Row<'_>,
    from: u32,
    needle: &Dna2PackedNeedle,
    state: &Dna2PackedState,
) -> Option<u32> {
    let (text_len, needle_len, last_start) =
        checked_search_bounds(row, from, needle.symbols.len())?;
    if needle_len == 0 {
        return Some(from);
    }
    if !state.has_fixed_bases() {
        return Some(from);
    }

    let mut pos = from;
    while pos <= last_start {
        if anchors_match_scalar(row, pos, state) && packed_matches_at_scalar(row, pos, needle) {
            return Some(pos);
        }
        pos += 1;
    }

    let _ = text_len;
    None
}

#[inline]
fn packed_find_from_vectorized(
    row: &Dna2Row<'_>,
    from: u32,
    needle: &Dna2PackedNeedle,
    state: &Dna2PackedState,
) -> Option<u32> {
    let (_text_len, needle_len, last_start) =
        checked_search_bounds(row, from, needle.symbols.len())?;
    if needle_len == 0 {
        return Some(from);
    }
    if !state.has_fixed_bases() {
        return Some(from);
    }

    let mut pos = from;

    while pos <= last_start && pos.saturating_add(31) <= last_start {
        let mut mask = candidate_mask_block(row, pos, state);

        while mask != 0 {
            let lane = first_lane_from_sparse_mask(mask);
            let cand = pos + lane;
            if packed_matches_at_vectorized(row, cand, needle) {
                return Some(cand);
            }
            mask &= !sparse_lane_bit(lane);
        }

        pos += 32;
    }

    while pos <= last_start {
        if anchors_match_scalar(row, pos, state) && packed_matches_at_vectorized(row, pos, needle) {
            return Some(pos);
        }
        pos += 1;
    }

    None
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

#[inline]
fn anchors_match_scalar(row: &Dna2Row<'_>, pos: u32, state: &Dna2PackedState) -> bool {
    for &(offset, want) in state.anchors() {
        let Some(bits) = row.load_2bit_window(pos + offset, 1) else {
            return false;
        };
        if bits as u8 != want {
            return false;
        }
    }
    true
}

#[inline]
fn candidate_mask_block(row: &Dna2Row<'_>, pos: u32, state: &Dna2PackedState) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is baseline on x86_64. The caller guarantees 32 valid
        // candidate starts, and each anchor offset is within the literal.
        return unsafe { candidate_mask_block_sse2(row, pos, state) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64. The caller guarantees 32 valid
        // candidate starts, and each anchor offset is within the literal.
        return unsafe { candidate_mask_block_neon(row, pos, state) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        candidate_mask_block_scalar(row, pos, state)
    }
}

#[inline]
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn candidate_mask_block_scalar(row: &Dna2Row<'_>, pos: u32, state: &Dna2PackedState) -> u64 {
    let mut mask = LOW_2BIT_LANES;

    for &(offset, want) in state.anchors() {
        let bits = row
            .load_2bit_window(pos + offset, 32)
            .expect("candidate block must be in-bounds");
        mask &= eq_2bit_lanes_sparse(bits, want);
        if mask == 0 {
            break;
        }
    }

    mask
}

#[inline(always)]
fn repeat_2bit(base: u8) -> u64 {
    debug_assert!(base < 4);
    u64::from(base) * LOW_2BIT_LANES
}

#[inline(always)]
fn eq_2bit_lanes_sparse(bits: u64, base: u8) -> u64 {
    let diff = bits ^ repeat_2bit(base);
    eq_2bit_lanes_sparse_from_diff(diff)
}

#[inline(always)]
fn eq_2bit_lanes_sparse_from_diff(diff: u64) -> u64 {
    !(diff | (diff >> 1)) & LOW_2BIT_LANES
}

#[inline(always)]
fn sparse_lane_bit(lane: u32) -> u64 {
    debug_assert!(lane < 32);
    1u64 << (2 * (31 - lane))
}

#[inline(always)]
fn first_lane_from_sparse_mask(mask: u64) -> u32 {
    debug_assert!(mask != 0);
    let bit = 63 - mask.leading_zeros();
    31 - (bit / 2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn packed_matches_at_sse2(row: &Dna2Row<'_>, pos: u32, needle: &Dna2PackedNeedle) -> bool {
    use core::arch::x86_64::*;

    let len = needle.symbols.len() as u32;
    let Some(end) = pos.checked_add(len) else {
        return false;
    };
    if end > row.len_bases() {
        return false;
    }

    let chunks = needle.chunks();
    let zero = _mm_setzero_si128();
    let mut idx = 0usize;

    while idx + 1 < chunks.len() {
        let c0 = chunks[idx];
        let c1 = chunks[idx + 1];
        let p0 = pos + (idx as u32) * 32;
        let p1 = pos + ((idx + 1) as u32) * 32;
        let r0 = row
            .load_2bit_window(p0, u32::from(c0.bases))
            .expect("chunk is in bounds");
        let r1 = row
            .load_2bit_window(p1, u32::from(c1.bases))
            .expect("chunk is in bounds");

        let row_v = _mm_set_epi64x(r1 as i64, r0 as i64);
        let bits_v = _mm_set_epi64x(c1.bits as i64, c0.bits as i64);
        let care_v = _mm_set_epi64x(c1.care_mask as i64, c0.care_mask as i64);
        let diff = _mm_and_si128(_mm_xor_si128(row_v, bits_v), care_v);
        let eq = _mm_cmpeq_epi8(diff, zero);
        if _mm_movemask_epi8(eq) != 0xFFFF {
            return false;
        }

        idx += 2;
    }

    if idx < chunks.len() {
        let chunk = chunks[idx];
        if chunk.care_mask != 0 {
            let row_bits = row
                .load_2bit_window(pos + (idx as u32) * 32, u32::from(chunk.bases))
                .expect("chunk is in bounds");
            if ((row_bits ^ chunk.bits) & chunk.care_mask) != 0 {
                return false;
            }
        }
    }

    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn candidate_mask_block_sse2(row: &Dna2Row<'_>, pos: u32, state: &Dna2PackedState) -> u64 {
    use core::arch::x86_64::*;

    let anchors = state.anchors();
    let mut mask = LOW_2BIT_LANES;
    let mut idx = 0usize;

    while idx + 1 < anchors.len() {
        let (off0, want0) = anchors[idx];
        let (off1, want1) = anchors[idx + 1];
        let b0 = row
            .load_2bit_window(pos + off0, 32)
            .expect("candidate block must be in-bounds");
        let b1 = row
            .load_2bit_window(pos + off1, 32)
            .expect("candidate block must be in-bounds");

        let bits_v = _mm_set_epi64x(b1 as i64, b0 as i64);
        let reps_v = _mm_set_epi64x(repeat_2bit(want1) as i64, repeat_2bit(want0) as i64);
        let diff_v = _mm_xor_si128(bits_v, reps_v);
        let bad_v = _mm_or_si128(diff_v, _mm_srli_epi64(diff_v, 1));
        let low_v = _mm_set1_epi64x(LOW_2BIT_LANES as i64);
        let eq_v = _mm_andnot_si128(bad_v, low_v);

        let mut lanes = [0u64; 2];
        unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast::<__m128i>(), eq_v) };
        mask &= lanes[0] & lanes[1];
        if mask == 0 {
            return 0;
        }

        idx += 2;
    }

    if idx < anchors.len() {
        let (off, want) = anchors[idx];
        let bits = row
            .load_2bit_window(pos + off, 32)
            .expect("candidate block must be in-bounds");
        mask &= eq_2bit_lanes_sparse(bits, want);
    }

    mask
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn packed_matches_at_neon(row: &Dna2Row<'_>, pos: u32, needle: &Dna2PackedNeedle) -> bool {
    use core::arch::aarch64::*;

    let len = needle.symbols.len() as u32;
    let Some(end) = pos.checked_add(len) else {
        return false;
    };
    if end > row.len_bases() {
        return false;
    }

    let chunks = needle.chunks();
    let mut idx = 0usize;

    while idx + 1 < chunks.len() {
        let c0 = chunks[idx];
        let c1 = chunks[idx + 1];
        let p0 = pos + (idx as u32) * 32;
        let p1 = pos + ((idx + 1) as u32) * 32;
        let r0 = row
            .load_2bit_window(p0, u32::from(c0.bases))
            .expect("chunk is in bounds");
        let r1 = row
            .load_2bit_window(p1, u32::from(c1.bases))
            .expect("chunk is in bounds");

        let row_v = vcombine_u64(vcreate_u64(r0), vcreate_u64(r1));
        let bits_v = vcombine_u64(vcreate_u64(c0.bits), vcreate_u64(c1.bits));
        let care_v = vcombine_u64(vcreate_u64(c0.care_mask), vcreate_u64(c1.care_mask));
        let diff = vandq_u64(veorq_u64(row_v, bits_v), care_v);
        if (vgetq_lane_u64::<0>(diff) | vgetq_lane_u64::<1>(diff)) != 0 {
            return false;
        }

        idx += 2;
    }

    if idx < chunks.len() {
        let chunk = chunks[idx];
        if chunk.care_mask != 0 {
            let row_bits = row
                .load_2bit_window(pos + (idx as u32) * 32, u32::from(chunk.bases))
                .expect("chunk is in bounds");
            if ((row_bits ^ chunk.bits) & chunk.care_mask) != 0 {
                return false;
            }
        }
    }

    true
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn candidate_mask_block_neon(row: &Dna2Row<'_>, pos: u32, state: &Dna2PackedState) -> u64 {
    use core::arch::aarch64::*;

    let anchors = state.anchors();
    let mut mask = LOW_2BIT_LANES;
    let mut idx = 0usize;

    while idx + 1 < anchors.len() {
        let (off0, want0) = anchors[idx];
        let (off1, want1) = anchors[idx + 1];
        let b0 = row
            .load_2bit_window(pos + off0, 32)
            .expect("candidate block must be in-bounds");
        let b1 = row
            .load_2bit_window(pos + off1, 32)
            .expect("candidate block must be in-bounds");

        let bits_v = vcombine_u64(vcreate_u64(b0), vcreate_u64(b1));
        let reps_v = vcombine_u64(
            vcreate_u64(repeat_2bit(want0)),
            vcreate_u64(repeat_2bit(want1)),
        );
        let diff_v = veorq_u64(bits_v, reps_v);
        let bad_v = vorrq_u64(diff_v, vshrq_n_u64::<1>(diff_v));
        let eq_v = vbicq_u64(vdupq_n_u64(LOW_2BIT_LANES), bad_v);
        mask &= vgetq_lane_u64::<0>(eq_v) & vgetq_lane_u64::<1>(eq_v);
        if mask == 0 {
            return 0;
        }

        idx += 2;
    }

    if idx < anchors.len() {
        let (off, want) = anchors[idx];
        let bits = row
            .load_2bit_window(pos + off, 32)
            .expect("candidate block must be in-bounds");
        mask &= eq_2bit_lanes_sparse(bits, want);
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::like::{LikePattern, RowLiteralSearch};
    use crate::storage::Column;
    use crate::storage::dna2::Dna2TableBuilder;
    use crate::{DbBuilder, FullScan, QueryScratch, RowId, execute_like};

    fn one_row(seq: &str) -> (crate::Db, crate::TableId) {
        let mut table = Dna2TableBuilder::new("dna");
        table.push_str(seq).unwrap();
        let mut dbb = DbBuilder::new();
        let id = dbb.add_dna2_table(table).unwrap();
        (dbb.freeze(), id)
    }

    fn check_algo<A>()
    where
        A: LiteralAlgorithm,
        for<'db> A: RowLiteralSearch<Dna2Column<'db>>,
    {
        let cases = [
            ("ACGTACGT", "ACG"),
            ("ACGTACGT", "A_G"),
            ("ACGTACGT", "TAC"),
            ("AAAAAA", "AAA"),
            ("AAAAAA", "A_A"),
            ("ACGT", "____"),
            ("ACGT", "TTT"),
            ("", ""),
        ];

        for (text, pat) in cases {
            let (db, id) = one_row(text);
            let table = db.dna2_table(id).unwrap();
            let col = table.sequence();
            let row = col.row_view(0);
            let needle = A::compile_literal(pat).unwrap();
            let state = A::build_state(&needle);
            let reference = Dna2NaiveWildcard::compile_literal(pat).unwrap();
            for from in 0..=(text.len() as u32 + 1) {
                let got = A::find_from(&row, from, &needle, &state);
                let expect = naive_find_from(&row, from, reference.symbols());
                assert_eq!(got, expect, "text={text:?}, pat={pat:?}, from={from}");
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
    }
}
