use std::cell::RefCell;
use std::cmp;
use std::ops::{Add, Mul, Sub};

use crate::StringSearch;

#[derive(Debug, Clone)]
pub struct FftConfig {
    pattern: Vec<u8>,
    wildcard: u8,
}

impl FftConfig {
    pub fn new(pattern: impl AsRef<[u8]>) -> Self {
        Self {
            pattern: pattern.as_ref().to_vec(),
            wildcard: b'_',
        }
    }

    pub fn from_str(pattern: &str) -> Self {
        Self::new(pattern.as_bytes())
    }

    pub fn pattern(&self) -> &[u8] {
        &self.pattern
    }

    pub fn wildcard(&self) -> u8 {
        self.wildcard
    }
}

#[derive(Debug)]
pub struct FftState0 {
    impl_kind: ImplKind0,
}

#[derive(Debug)]
pub struct FftState1 {
    impl_actual: RefCell<ImplActual<Value2>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FftStr0;

#[derive(Debug, Clone, Copy, Default)]
pub struct FftStr1;

impl StringSearch for FftStr0 {
    type Config = FftConfig;
    type State = FftState0;

    fn build(config: &Self::Config) -> Self::State {
        let pattern_size = config.pattern.len();
        let required = pattern_size.saturating_mul(3);

        if required <= 1 {
            panic!("pattern size too small");
        }

        let log2n = log2ceil(required as u64);
        if log2n >= 12 {
            panic!("pattern too large for fft size");
        }

        let impl_kind = if log2n < 7 {
            ImplKind0::Small(RefCell::new(ImplActual::<Value0>::new(
                log2n,
                &config.pattern,
                config.wildcard,
            )))
        } else {
            ImplKind0::Large(RefCell::new(ImplActual::<Value1>::new(
                log2n,
                &config.pattern,
                config.wildcard,
            )))
        };

        Self::State { impl_kind }
    }

    fn find_bytes(config: &Self::Config, state: &Self::State, text: &[u8]) -> Option<usize> {
        match &state.impl_kind {
            ImplKind0::Small(inner) => inner.borrow_mut().find_first(text, config.wildcard),
            ImplKind0::Large(inner) => inner.borrow_mut().find_first(text, config.wildcard),
        }
    }
}

impl StringSearch for FftStr1 {
    type Config = FftConfig;
    type State = FftState1;

    fn build(config: &Self::Config) -> Self::State {
        let pattern_size = config.pattern.len();
        let required = pattern_size.saturating_mul(3);

        if required <= 1 {
            panic!("pattern size too small");
        }

        let log2n = log2ceil(required as u64);
        if log2n > 27 {
            panic!("pattern too large for fft size");
        }

        let impl_actual = ImplActual::<Value2>::new(log2n, &config.pattern, config.wildcard);
        Self::State {
            impl_actual: RefCell::new(impl_actual),
        }
    }

    fn find_bytes(config: &Self::Config, state: &Self::State, text: &[u8]) -> Option<usize> {
        state
            .impl_actual
            .borrow_mut()
            .find_first(text, config.wildcard)
    }
}

#[derive(Debug)]
enum ImplKind0 {
    Small(RefCell<ImplActual<Value0>>),
    Large(RefCell<ImplActual<Value1>>),
}

fn log2ceil(value: u64) -> u32 {
    assert!(value > 1);
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

    fn add(self, other: Self) -> Self {
        Self {
            v: Self::mod_u64(self.v.wrapping_add(other.v)),
        }
    }
}

impl Sub for Value0 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            v: Self::mod_u64(self.v.wrapping_add(Self::M_BASE).wrapping_sub(other.v)),
        }
    }
}

impl Mul for Value0 {
    type Output = Self;

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

    fn from_base(value: Self::Base) -> Self {
        Self {
            v: Self::mod_u64(value),
        }
    }

    fn from_u64(value: u64) -> Self {
        Self {
            v: Self::mod_u64(value),
        }
    }

    fn base(self) -> Self::Base {
        self.v
    }

    fn mod_higher(value: Self::Higher) -> Self::Base {
        Self::mod_u64(value)
    }

    fn to_higher(value: Self::Base) -> Self::Higher {
        value
    }

    fn add_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_add(b)
    }

    fn sub_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_sub(b)
    }

    fn shl1(a: Self::Higher) -> Self::Higher {
        a << 1
    }

    fn twice_m() -> Self::Higher {
        Self::M_BASE * 2
    }

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

    fn add(self, other: Self) -> Self {
        Self {
            v: Self::mod_u128(self.v.wrapping_add(other.v)),
        }
    }
}

impl Sub for Value1 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            v: Self::mod_u128(self.v.wrapping_add(Self::M_BASE).wrapping_sub(other.v)),
        }
    }
}

impl Mul for Value1 {
    type Output = Self;

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

    fn from_base(value: Self::Base) -> Self {
        Self {
            v: Self::mod_u128(value),
        }
    }

    fn from_u64(value: u64) -> Self {
        Self {
            v: Self::mod_u128(value as u128),
        }
    }

    fn base(self) -> Self::Base {
        self.v
    }

    fn mod_higher(value: Self::Higher) -> Self::Base {
        Self::mod_u128(value)
    }

    fn to_higher(value: Self::Base) -> Self::Higher {
        value
    }

    fn add_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_add(b)
    }

    fn sub_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_sub(b)
    }

    fn shl1(a: Self::Higher) -> Self::Higher {
        a << 1
    }

    fn twice_m() -> Self::Higher {
        Self::M_BASE * 2
    }

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
    // Lemire's fastmod: precomputed constant for M_BASE
    // See: https://lemire.me/blog/2019/02/08/faster-remainders-when-the-divisor-is-a-constant-beating-compilers-and-libdivide/
    const FASTMOD_M: u64 = 0xFFFFFFFFFFFFFFFFu64 / Self::M_BASE as u64 + 1;

    // FastMod for u32 inputs (used by add/sub where result < 2*M < 2^32)
    #[inline(always)]
    fn fastmod_u32(value: u32) -> u32 {
        let lowbits = Self::FASTMOD_M.wrapping_mul(value as u64);
        (((lowbits as u128) * (Self::M_BASE as u128)) >> 64) as u32
    }

    // Regular mod for u64 inputs (used by mul where result can be up to M^2)
    #[inline(always)]
    fn mod_u64(value: u64) -> u32 {
        (value % (Self::M_BASE as u64)) as u32
    }
}

impl Add for Value2 {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // Sum of two values < M is < 2M < 2^32, so fits in u32
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
        // M + a - b where a, b < M, result is in [1, 2M-1] < 2^32
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

    fn from_base(value: Self::Base) -> Self {
        Self {
            v: Self::mod_u64(value as u64),
        }
    }

    fn from_u64(value: u64) -> Self {
        Self {
            v: Self::mod_u64(value),
        }
    }

    fn base(self) -> Self::Base {
        self.v
    }

    fn mod_higher(value: Self::Higher) -> Self::Base {
        Self::mod_u64(value)
    }

    fn to_higher(value: Self::Base) -> Self::Higher {
        value as u64
    }

    fn add_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_add(b)
    }

    fn sub_higher(a: Self::Higher, b: Self::Higher) -> Self::Higher {
        a.wrapping_sub(b)
    }

    fn shl1(a: Self::Higher) -> Self::Higher {
        a << 1
    }

    fn twice_m() -> Self::Higher {
        (Self::M_BASE as u64) * 2
    }

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
                    let w = self.twiddles[((k << s) % self.n) as usize];
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
                    let w = self.itwiddles[((k << s) % self.n) as usize];
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

        if fft.get_p() == 0 {
            panic!("invalid fft parameters");
        }

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

                if lhs == rhs {
                    if !callback(offset + j) {
                        return;
                    }
                }
            }

            offset += matchable;
        }
    }

    fn find_first(&mut self, text: &[u8], _wildcard: u8) -> Option<usize> {
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

    #[test]
    fn test_fftstr0_basic() {
        let config = FftConfig::from_str("ababd");
        let state = FftStr0::build(&config);
        assert_eq!(
            FftStr0::find_str(&config, &state, "ababcabcabababd"),
            Some(10)
        );
    }

    #[test]
    fn test_fftstr0_not_found() {
        let config = FftConfig::from_str("rust");
        let state = FftStr0::build(&config);
        assert_eq!(FftStr0::find_str(&config, &state, "hello world"), None);
    }

    #[test]
    fn test_fftstr0_wildcard() {
        let config = FftConfig::from_str("a_c");
        let state = FftStr0::build(&config);
        assert_eq!(FftStr0::find_str(&config, &state, "zzabczz"), Some(2));
    }

    #[test]
    fn test_fftstr0_find_all() {
        let config = FftConfig::from_str("aba");
        let state = FftStr0::build(&config);
        let matches = FftStr0::find_all_bytes(&config, &state, b"ababa");
        assert_eq!(matches, vec![0, 2]);
    }

    #[test]
    fn test_fftstr1_basic() {
        let config = FftConfig::from_str("ababd");
        let state = FftStr1::build(&config);
        assert_eq!(
            FftStr1::find_str(&config, &state, "ababcabcabababd"),
            Some(10)
        );
    }

    #[test]
    fn test_fftstr1_wildcard() {
        let config = FftConfig::from_str("a_c");
        let state = FftStr1::build(&config);
        assert_eq!(FftStr1::find_str(&config, &state, "zzabczz"), Some(2));
    }

    #[test]
    fn test_fftstr1_find_all() {
        let config = FftConfig::from_str("aba");
        let state = FftStr1::build(&config);
        let matches = FftStr1::find_all_bytes(&config, &state, b"ababa");
        assert_eq!(matches, vec![0, 2]);
    }

    #[test]
    fn test_fftstr1_large_pattern() {
        let pattern = "a".repeat(50);
        let text = format!("zz{}zz", pattern);
        let config = FftConfig::from_str(&pattern);
        let state = FftStr1::build(&config);
        assert_eq!(FftStr1::find_str(&config, &state, &text), Some(2));
    }

    #[test]
    #[should_panic]
    fn test_fftstr0_large_pattern_panics() {
        let pattern = "a".repeat(50);
        let config = FftConfig::from_str(&pattern);
        let _ = FftStr0::build(&config);
    }
}
