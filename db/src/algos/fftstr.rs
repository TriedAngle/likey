//! FFT/NTT-backed byte-string literal search with `_` wildcard support.
//!
//! This is a port of the older `FftStr0`/`FftStr1` searchers into the current
//! `LiteralAlgorithm + RowLiteralSearch<Utf8Column>` API. Unlike KMP/BM/TwoWay,
//! these algorithms can keep `_` inside literal fragments and interpret it as a
//! one-byte wildcard.

use std::cell::RefCell;
use std::cmp;
use std::ops::{Add, Mul, Sub};

use crate::like::{LiteralAlgorithm, RowLiteralSearch};
use crate::storage::utf8::{Utf8Column, Utf8Row};

use super::utf8_shared::{bytes_eq_same_len, utf8_row_len};

const FFT_WILDCARD: u8 = b'_';

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftNeedle {
    bytes: Box<[u8]>,
    has_wildcard: bool,
}

impl FftNeedle {
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[inline]
    pub fn has_wildcard(&self) -> bool {
        self.has_wildcard
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FftStr0;

#[derive(Debug, Clone, Copy, Default)]
pub struct FftStr1;

#[derive(Debug)]
pub struct FftState0 {
    impl_kind: ImplKind0,
}

#[derive(Debug)]
pub struct FftState1 {
    impl_actual: RefCell<ImplActual<Value2>>,
}

impl LiteralAlgorithm for FftStr0 {
    type Needle = FftNeedle;
    type State = FftState0;

    const SUPPORTS_UNDERSCORE: bool = true;

    fn compile_literal(src: &str) -> Option<Self::Needle> {
        validate_fft0_literal(src.as_bytes())?;
        Some(FftNeedle {
            bytes: src.as_bytes().into(),
            has_wildcard: src.as_bytes().contains(&FFT_WILDCARD),
        })
    }

    fn build_state(needle: &Self::Needle) -> Self::State {
        let log2n = validate_fft0_literal(needle.bytes()).expect("validated FFT0 literal");
        let impl_kind = if log2n < 7 {
            ImplKind0::Small(RefCell::new(ImplActual::<Value0>::new(
                log2n,
                needle.bytes(),
                FFT_WILDCARD,
            )))
        } else {
            ImplKind0::Large(RefCell::new(ImplActual::<Value1>::new(
                log2n,
                needle.bytes(),
                FFT_WILDCARD,
            )))
        };
        FftState0 { impl_kind }
    }

    fn literal_len(needle: &Self::Needle) -> u32 {
        needle.bytes().len() as u32
    }

    fn index_symbols(needle: &Self::Needle) -> Option<Box<[u8]>> {
        if needle.has_wildcard() {
            None
        } else {
            Some(needle.bytes().into())
        }
    }
}

impl LiteralAlgorithm for FftStr1 {
    type Needle = FftNeedle;
    type State = FftState1;

    const SUPPORTS_UNDERSCORE: bool = true;

    fn compile_literal(src: &str) -> Option<Self::Needle> {
        validate_fft1_literal(src.as_bytes())?;
        Some(FftNeedle {
            bytes: src.as_bytes().into(),
            has_wildcard: src.as_bytes().contains(&FFT_WILDCARD),
        })
    }

    fn build_state(needle: &Self::Needle) -> Self::State {
        let log2n = validate_fft1_literal(needle.bytes()).expect("validated FFT1 literal");
        FftState1 {
            impl_actual: RefCell::new(ImplActual::<Value2>::new(
                log2n,
                needle.bytes(),
                FFT_WILDCARD,
            )),
        }
    }

    fn literal_len(needle: &Self::Needle) -> u32 {
        needle.bytes().len() as u32
    }

    fn index_symbols(needle: &Self::Needle) -> Option<Box<[u8]>> {
        if needle.has_wildcard() {
            None
        } else {
            Some(needle.bytes().into())
        }
    }
}

impl<'db> RowLiteralSearch<Utf8Column<'db>> for FftStr0 {
    #[inline]
    fn row_len<'r>(row: &Utf8Row<'r>) -> u32 {
        utf8_row_len(row)
    }

    #[inline]
    fn matches_at<'r>(
        row: &Utf8Row<'r>,
        pos: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> bool {
        fft_matches_at(row, pos, needle)
    }

    #[inline]
    fn find_from<'r>(
        row: &Utf8Row<'r>,
        from: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> Option<u32> {
        let text = row.bytes();
        let from = from as usize;
        if from > text.len() {
            return None;
        }
        if needle.bytes().is_empty() {
            return Some(from as u32);
        }
        if needle.bytes().len() > text.len().saturating_sub(from) {
            return None;
        }

        let hit = match &state.impl_kind {
            ImplKind0::Small(inner) => inner.borrow_mut().find_first(&text[from..]),
            ImplKind0::Large(inner) => inner.borrow_mut().find_first(&text[from..]),
        }?;
        Some((from + hit) as u32)
    }
}

impl<'db> RowLiteralSearch<Utf8Column<'db>> for FftStr1 {
    #[inline]
    fn row_len<'r>(row: &Utf8Row<'r>) -> u32 {
        utf8_row_len(row)
    }

    #[inline]
    fn matches_at<'r>(
        row: &Utf8Row<'r>,
        pos: u32,
        needle: &Self::Needle,
        _state: &Self::State,
    ) -> bool {
        fft_matches_at(row, pos, needle)
    }

    #[inline]
    fn find_from<'r>(
        row: &Utf8Row<'r>,
        from: u32,
        needle: &Self::Needle,
        state: &Self::State,
    ) -> Option<u32> {
        let text = row.bytes();
        let from = from as usize;
        if from > text.len() {
            return None;
        }
        if needle.bytes().is_empty() {
            return Some(from as u32);
        }
        if needle.bytes().len() > text.len().saturating_sub(from) {
            return None;
        }

        state
            .impl_actual
            .borrow_mut()
            .find_first(&text[from..])
            .map(|hit| (from + hit) as u32)
    }
}

#[inline]
fn fft_matches_at(row: &Utf8Row<'_>, pos: u32, needle: &FftNeedle) -> bool {
    let pos = pos as usize;
    let pat = needle.bytes();
    let Some(end) = pos.checked_add(pat.len()) else {
        return false;
    };
    let Some(candidate) = row.bytes().get(pos..end) else {
        return false;
    };

    if !needle.has_wildcard() {
        return bytes_eq_same_len(candidate, pat);
    }

    candidate
        .iter()
        .zip(pat)
        .all(|(&got, &want)| want == FFT_WILDCARD || got == want)
}

fn validate_fft0_literal(pattern: &[u8]) -> Option<u32> {
    let required = pattern.len().checked_mul(3)?;
    if required <= 1 {
        return None;
    }
    let log2n = log2ceil(required as u64);
    if log2n > 7 {
        return None;
    }
    Some(log2n)
}

fn validate_fft1_literal(pattern: &[u8]) -> Option<u32> {
    let required = pattern.len().checked_mul(3)?;
    if required <= 1 {
        return None;
    }
    let log2n = log2ceil(required as u64);
    if log2n > 27 {
        return None;
    }
    Some(log2n)
}

#[derive(Debug)]
enum ImplKind0 {
    Small(RefCell<ImplActual<Value0>>),
    Large(RefCell<ImplActual<Value1>>),
}

fn log2ceil(value: u64) -> u32 {
    debug_assert!(value > 1);
    let v = value - 1;
    64 - v.leading_zeros()
}

trait FftValue:
    Copy
    + Clone
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Eq
    + std::fmt::Debug
{
    type Base: Copy + Eq + std::fmt::Debug;
    type Higher: Copy + std::fmt::Debug;

    const OMEGA: Self::Base;
    const INV_OMEGA: Self::Base;
    const OMEGA_POWER: u64;

    fn from_base(value: Self::Base) -> Self;
    fn from_u64(value: u64) -> Self;
    fn base(self) -> Self::Base;

    fn mod_higher(value: Self::Higher) -> Self::Base;
    fn to_higher(value: Self::Base) -> Self::Higher;
    fn add_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher;
    fn sub_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher;
    fn shl1(a: Self::Higher) -> Self::Higher;
    fn twice_m() -> Self::Higher;
    fn m_minus(value: Self::Base) -> Self::Base;
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
struct Value0 {
    v: u64,
}

impl Value0 {
    const M_BASE: u64 = (1u64 << 32) + 1;

    #[inline(always)]
    fn mod_u64(value: u64) -> u64 {
        let a = value as u32 as u64;
        let b = value >> 32;
        let c = a as i64 - b as i64;
        if c < 0 {
            (c + Self::M_BASE as i64) as u64
        } else {
            c as u64
        }
    }
}

impl Add for Value0 {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self {
            v: Self::mod_u64(self.v.wrapping_add(other.v)),
        }
    }
}

impl Sub for Value0 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self {
            v: Self::mod_u64(self.v.wrapping_add(Self::M_BASE).wrapping_sub(other.v)),
        }
    }
}

impl Mul for Value0 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        let product = (self.v as u128) * (other.v as u128);
        let res = product as u64;
        let overflow = (product >> 64) as u64;
        Self {
            v: Self::mod_u64(res.wrapping_add(overflow)),
        }
    }
}

impl FftValue for Value0 {
    type Base = u64;
    type Higher = u64;

    const OMEGA: Self::Base = 2;
    const INV_OMEGA: Self::Base = 2147483649u64;
    const OMEGA_POWER: u64 = 64;

    #[inline(always)]
    fn from_base(value: Self::Base) -> Self {
        Self {
            v: Self::mod_u64(value),
        }
    }

    #[inline(always)]
    fn from_u64(value: u64) -> Self {
        Self {
            v: Self::mod_u64(value),
        }
    }

    #[inline(always)]
    fn base(self) -> Self::Base {
        self.v
    }

    #[inline(always)]
    fn mod_higher(value: Self::Higher) -> Self::Base {
        Self::mod_u64(value)
    }

    #[inline(always)]
    fn to_higher(value: Self::Base) -> Self::Higher {
        value
    }

    #[inline(always)]
    fn add_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_add(b)
    }

    #[inline(always)]
    fn sub_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_sub(b)
    }

    #[inline(always)]
    fn shl1(a: Self::Higher) -> Self::Higher {
        a << 1
    }

    #[inline(always)]
    fn twice_m() -> Self::Higher {
        Self::M_BASE * 2
    }

    #[inline(always)]
    fn m_minus(value: Self::Base) -> Self::Base {
        Self::M_BASE.wrapping_sub(value)
    }
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
struct Value1 {
    v: u128,
}

impl Value1 {
    const M_BASE: u128 = (1u128 << 64) + 1;

    #[inline(always)]
    fn mod_u128(value: u128) -> u128 {
        let a = value as u64 as u128;
        let b = value >> 64;
        let c = a as i128 - b as i128;
        if c < 0 {
            (c + Self::M_BASE as i128) as u128
        } else {
            c as u128
        }
    }
}

impl Add for Value1 {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self {
            v: Self::mod_u128(self.v.wrapping_add(other.v)),
        }
    }
}

impl Sub for Value1 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self {
            v: Self::mod_u128(self.v.wrapping_add(Self::M_BASE).wrapping_sub(other.v)),
        }
    }
}

impl Mul for Value1 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        let res = (self.v as u64 as u128) * (other.v as u64 as u128);
        let high_a = self.v >> 64;
        let high_b = other.v >> 64;
        if high_a != 0 && high_b != 0 {
            Self { v: 1 }
        } else {
            Self {
                v: Self::mod_u128(res),
            }
        }
    }
}

impl FftValue for Value1 {
    type Base = u128;
    type Higher = u128;

    const OMEGA: Self::Base = 2;
    const INV_OMEGA: Self::Base = 9223372036854775809u128;
    const OMEGA_POWER: u64 = 128;

    #[inline(always)]
    fn from_base(value: Self::Base) -> Self {
        Self {
            v: Self::mod_u128(value),
        }
    }

    #[inline(always)]
    fn from_u64(value: u64) -> Self {
        Self {
            v: Self::mod_u128(value as u128),
        }
    }

    #[inline(always)]
    fn base(self) -> Self::Base {
        self.v
    }

    #[inline(always)]
    fn mod_higher(value: Self::Higher) -> Self::Base {
        Self::mod_u128(value)
    }

    #[inline(always)]
    fn to_higher(value: Self::Base) -> Self::Higher {
        value
    }

    #[inline(always)]
    fn add_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_add(b)
    }

    #[inline(always)]
    fn sub_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_sub(b)
    }

    #[inline(always)]
    fn shl1(a: Self::Higher) -> Self::Higher {
        a << 1
    }

    #[inline(always)]
    fn twice_m() -> Self::Higher {
        Self::M_BASE * 2
    }

    #[inline(always)]
    fn m_minus(value: Self::Base) -> Self::Base {
        Self::M_BASE.wrapping_sub(value)
    }
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
struct Value2 {
    v: u32,
}

impl Value2 {
    const M_BASE: u32 = 2013265921u32;
    const FASTMOD_M: u64 = 0xFFFFFFFFFFFFFFFFu64 / Self::M_BASE as u64 + 1;

    #[inline(always)]
    fn fastmod_u32(value: u32) -> u32 {
        let lowbits = Self::FASTMOD_M.wrapping_mul(value as u64);
        (((lowbits as u128) * (Self::M_BASE as u128)) >> 64) as u32
    }

    #[inline(always)]
    fn mod_u64(value: u64) -> u32 {
        (value % (Self::M_BASE as u64)) as u32
    }
}

impl Add for Value2 {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let sum = self.v.wrapping_add(other.v);
        Self {
            v: Self::fastmod_u32(sum),
        }
    }
}

impl Sub for Value2 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        let diff = Self::M_BASE.wrapping_add(self.v).wrapping_sub(other.v);
        Self {
            v: Self::fastmod_u32(diff),
        }
    }
}

impl Mul for Value2 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        let product = self.v as u64 * other.v as u64;
        Self {
            v: Self::mod_u64(product),
        }
    }
}

impl FftValue for Value2 {
    type Base = u32;
    type Higher = u64;

    const OMEGA: Self::Base = 1985266761u32;
    const INV_OMEGA: Self::Base = 1885204058u32;
    const OMEGA_POWER: u64 = 1u64 << 27;

    #[inline(always)]
    fn from_base(value: Self::Base) -> Self {
        Self {
            v: Self::mod_u64(value as u64),
        }
    }

    #[inline(always)]
    fn from_u64(value: u64) -> Self {
        Self {
            v: Self::mod_u64(value),
        }
    }

    #[inline(always)]
    fn base(self) -> Self::Base {
        self.v
    }

    #[inline(always)]
    fn mod_higher(value: Self::Higher) -> Self::Base {
        Self::mod_u64(value)
    }

    #[inline(always)]
    fn to_higher(value: Self::Base) -> Self::Higher {
        value as u64
    }

    #[inline(always)]
    fn add_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_add(b)
    }

    #[inline(always)]
    fn sub_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_sub(b)
    }

    #[inline(always)]
    fn shl1(a: Self::Higher) -> Self::Higher {
        a << 1
    }

    #[inline(always)]
    fn twice_m() -> Self::Higher {
        (Self::M_BASE as u64) * 2
    }

    #[inline(always)]
    fn m_minus(value: Self::Base) -> Self::Base {
        Self::M_BASE.wrapping_sub(value)
    }
}

#[derive(Debug)]
struct Fft<V: FftValue> {
    log2n: u32,
    n: usize,
    twiddles: Vec<V>,
    itwiddles: Vec<V>,
}

impl<V: FftValue> Fft<V> {
    fn new(log2n: u32) -> Self {
        let n = 1usize << log2n;
        let omega = V::from_base(V::OMEGA);
        let inv_omega = V::from_base(V::INV_OMEGA);
        let phi = Self::fast_pow(omega, V::OMEGA_POWER >> log2n).base();
        let iphi = Self::fast_pow(inv_omega, V::OMEGA_POWER >> log2n).base();

        let twiddles = Self::get_twiddles(n, V::from_base(phi));
        let itwiddles = Self::get_twiddles(n, V::from_base(iphi));

        Self {
            log2n,
            n,
            twiddles,
            itwiddles,
        }
    }

    fn get_p(&self) -> u64 {
        V::OMEGA_POWER >> self.log2n
    }

    fn fast_pow(mut base: V, mut pw: u64) -> V {
        let mut res = V::from_u64(1);
        while pw != 0 {
            if pw & 1 == 1 {
                res = base * res;
            }
            pw >>= 1;
            base = base * base;
        }
        res
    }

    fn get_twiddles(size: usize, t: V) -> Vec<V> {
        let mut res = vec![V::default(); size];
        if size == 0 {
            return res;
        }
        res[0] = V::from_u64(1);
        for i in 1..size {
            res[i] = res[i - 1] * t;
        }
        res
    }

    #[inline(always)]
    fn ct_dif_bf2(s: usize, out: &mut [V], w: V) {
        let a = out[0];
        let b = out[s];
        out[0] = a + b;
        out[s] = (a - b) * w;
    }

    #[inline(always)]
    fn ct_dit_bf2(s: usize, out: &mut [V], w: V) {
        let a = out[0];
        let b = out[s] * w;
        out[0] = a + b;
        out[s] = a - b;
    }

    fn fft(&self, x: &mut [V]) {
        for jp in 0..self.log2n {
            let j = self.log2n - jp - 1;
            let s = self.log2n - j - 1;
            let l = 1usize << j;

            for i in 0..(1usize << s) {
                let base = i << (j + 1);
                let t = &mut x[base..base + (l << 1)];
                for k in 0..l {
                    let w = self.twiddles[(k << s) % self.n];
                    Self::ct_dif_bf2(l, &mut t[k..], w);
                }
            }
        }
    }

    fn ifft(&self, x: &mut [V]) {
        for jp in 0..self.log2n {
            let j = jp;
            let s = self.log2n - j - 1;
            let l = 1usize << j;

            for i in 0..(1usize << s) {
                let base = i << (j + 1);
                let t = &mut x[base..base + (l << 1)];
                for k in 0..l {
                    let w = self.itwiddles[(k << s) % self.n];
                    Self::ct_dit_bf2(l, &mut t[k..], w);
                }
            }
        }
    }
}

#[derive(Debug)]
struct ImplActual<V: FftValue> {
    pattern_size: usize,
    n: usize,
    p0: Vec<V>,
    p1: Vec<V>,
    p2: Vec<V>,
    t1: Vec<V>,
    t2: Vec<V>,
    fft: Fft<V>,
}

impl<V: FftValue> ImplActual<V> {
    fn new(log2n: u32, pattern: &[u8], wildcard: u8) -> Self {
        let n = 1usize << log2n;
        let mut p0 = vec![V::default(); n];
        let mut p1 = vec![V::default(); n];
        let mut p2 = vec![V::default(); n];
        let t1 = vec![V::default(); n];
        let t2 = vec![V::default(); n];
        let fft = Fft::<V>::new(log2n);

        assert!(fft.get_p() != 0, "invalid fft parameters");

        let pattern_size = pattern.len();
        assert!(pattern_size * 3 <= n);

        p0[0] = V::from_u64(0);
        for i in 0..pattern_size {
            let c = pattern[i] as u64;
            let active = if pattern[i] != wildcard { 1u64 } else { 0u64 };
            let value = c * c * active * (n as u64);
            p0[i + 1] = p0[i] + V::from_u64(value);
        }

        for i in 0..pattern_size {
            let c = pattern[pattern_size - i - 1] as u64;
            let active = if c as u8 != wildcard { 1u64 } else { 0u64 };
            p1[i] = V::from_u64(active);
            p2[i] = V::from_u64(c * active);
        }

        let mut impl_actual = Self {
            pattern_size,
            n,
            p0,
            p1,
            p2,
            t1,
            t2,
            fft,
        };

        impl_actual.fft.fft(&mut impl_actual.p1);
        impl_actual.fft.fft(&mut impl_actual.p2);
        impl_actual
    }

    fn compute(&mut self, text: &[u8]) {
        for (idx, &byte) in text.iter().enumerate() {
            let val = byte as u64;
            self.t1[idx] = V::from_u64(val * val);
            self.t2[idx] = V::from_u64(val);
        }
        for i in text.len()..self.n {
            self.t1[i] = V::from_u64(0);
            self.t2[i] = V::from_u64(0);
        }

        self.fft.fft(&mut self.t1);
        self.fft.fft(&mut self.t2);

        for i in 0..self.n {
            self.t1[i] = self.t1[i] * self.p1[i];
            self.t2[i] = self.t2[i] * self.p2[i];
        }

        self.fft.ifft(&mut self.t1);
        self.fft.ifft(&mut self.t2);
    }

    fn iterate<F>(&mut self, text: &[u8], mut callback: F)
    where
        F: FnMut(usize) -> bool,
    {
        if text.len() < self.pattern_size {
            return;
        }

        let ts = self.n + 1 - self.pattern_size;
        let match_start = self.pattern_size - 1;
        let matchable = ts - self.pattern_size + 1;

        let mut offset = 0usize;
        let limit = text.len() - self.pattern_size + 1;
        while offset < limit {
            let end = cmp::min(offset + ts, text.len());
            let tt = &text[offset..end];
            self.compute(tt);

            let remaining = tt.len().saturating_sub(self.pattern_size) + 1;
            let max_j = cmp::min(matchable, remaining);

            for j in 0..max_j {
                let idx = j + match_start;
                let t1v = self.t1[idx].base();
                let t2v = self.t2[idx].base();
                let lhs = V::mod_higher(V::sub_higher(
                    V::add_higher(V::to_higher(t1v), V::twice_m()),
                    V::shl1(V::to_higher(t2v)),
                ));

                let remaining = tt.len() - j;
                let p0_idx = cmp::min(remaining, self.pattern_size);
                let rhs = V::m_minus(self.p0[p0_idx].base());

                if lhs == rhs && !callback(offset + j) {
                    return;
                }
            }

            offset += matchable;
        }
    }

    fn find_first(&mut self, text: &[u8]) -> Option<usize> {
        let mut found = None;
        self.iterate(text, |pos| {
            found = Some(pos);
            false
        });
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RowId;
    use crate::db::DbBuilder;
    use crate::like::LikePattern;
    use crate::query::{FullScan, QueryScratch, execute_like};
    use crate::storage::Column;
    use crate::storage::utf8::Utf8TableBuilder;

    fn one_row(text: &str) -> (crate::Db, crate::TableId) {
        let mut t = Utf8TableBuilder::new("t");
        t.push_str(text);
        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(t).unwrap();
        (dbb.freeze(), id)
    }

    fn assert_find<A>(text: &str, pat: &str, expected: Option<u32>)
    where
        A: LiteralAlgorithm<Needle = FftNeedle>,
        for<'db> A: RowLiteralSearch<Utf8Column<'db>>,
    {
        let (db, id) = one_row(text);
        let table = db.utf8_table(id).unwrap();
        let col = table.text();
        let row = col.row(0);
        let needle = A::compile_literal(pat).unwrap();
        let state = A::build_state(&needle);
        assert_eq!(A::find_from(&row, 0, &needle, &state), expected);
    }

    #[test]
    fn fftstr0_basic_and_wildcard() {
        assert_find::<FftStr0>("ababcabcabababd", "ababd", Some(10));
        assert_find::<FftStr0>("hello world", "rust", None);
        assert_find::<FftStr0>("zzabczz", "a_c", Some(2));
    }

    #[test]
    fn fftstr1_basic_and_wildcard() {
        assert_find::<FftStr1>("ababcabcabababd", "ababd", Some(10));
        assert_find::<FftStr1>("zzabczz", "a_c", Some(2));
    }

    #[test]
    fn fft_like_integration_passes_underscore() {
        let mut table = Utf8TableBuilder::new("docs");
        for row in ["ACGT", "AGGT", "ATTT", "TTTT"] {
            table.push_str(row);
        }
        let mut dbb = DbBuilder::new();
        let id = dbb.add_utf8_table(table).unwrap();
        let db = dbb.freeze();
        let table = db.utf8_table(id).unwrap();
        let col = table.text();

        let like = LikePattern::<FftStr1>::compile("A_G%").unwrap();
        let mut scan = FullScan::new(col.row_count(), 16);
        let mut scratch = QueryScratch::default();
        let mut matches = Vec::<RowId>::new();
        execute_like(&col, &mut scan, &like, &mut scratch, &mut matches);
        assert_eq!(matches, vec![0, 1]);
    }
}
