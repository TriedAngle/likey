use crate::StringSearch;
use core::cmp::max;
use core::marker::PhantomData;

pub struct TwoWay<'a>(PhantomData<&'a ()>);

#[derive(Clone, Copy, Debug)]
pub struct TwoWayState {
    // crit = ell + 1 (so crit==0 corresponds to ell==-1)
    crit: usize,
    period: usize,
    is_periodic: bool,
    // shift used in the non-periodic case (C's recomputed `per`)
    shift: usize,
}

impl<'a> StringSearch for TwoWay<'a> {
    type Config = &'a [u8];
    type State = TwoWayState;

    #[inline]
    fn build(pattern: &Self::Config) -> Self::State {
        build_state(pattern)
    }

    #[inline]
    fn find_bytes(pattern: &Self::Config, state: &Self::State, text: &[u8]) -> Option<usize> {
        two_way_find(text, pattern, state)
    }
}

#[inline(always)]
fn maximal_suffix(pattern: &[u8], reversed: bool) -> (isize, usize) {
    let m = pattern.len();
    let ptr = pattern.as_ptr();

    let mut ms: isize = -1;
    let mut j: usize = 0;
    let mut k: usize = 1;
    let mut p: usize = 1;

    while j + k < m {
        unsafe {
            let a = *ptr.add(j + k);
            let b = *ptr.add((ms + k as isize) as usize);

            if (!reversed && a < b) || (reversed && a > b) {
                j += k;
                k = 1;
                p = (j as isize - ms) as usize;
            } else if a == b {
                if k != p {
                    k += 1;
                } else {
                    j += p;
                    k = 1;
                }
            } else {
                // (!reversed && a > b) || (reversed && a < b)
                ms = j as isize;
                j = (ms + 1) as usize;
                k = 1;
                p = 1;
            }
        }
    }

    (ms, p)
}

#[inline]
fn build_state(pattern: &[u8]) -> TwoWayState {
    let m = pattern.len();

    if m == 0 {
        return TwoWayState {
            crit: 0,
            period: 1,
            is_periodic: true,
            shift: 1,
        };
    }
    if m == 1 {
        return TwoWayState {
            crit: 0,
            period: 1,
            is_periodic: false,
            shift: 1,
        };
    }

    let (ms1, p1) = maximal_suffix(pattern, false);
    let (ms2, p2) = maximal_suffix(pattern, true);

    let (ell, period) = if ms1 > ms2 { (ms1, p1) } else { (ms2, p2) };
    let crit = (ell + 1) as usize; // 0..=m

    // C: if (memcmp(x, x + per, ell + 1) == 0) ...
    // => compare length = crit, starting at `period`
    let is_periodic = period < m && crit <= (m - period) && pattern[..crit] == pattern[period..period + crit];

    // C non-periodic: per = MAX(ell + 1, m - ell - 1) + 1
    // with crit = ell + 1, and m - ell - 1 = m - crit
    let shift = max(crit, m - crit) + 1;

    TwoWayState {
        crit,
        period,
        is_periodic,
        shift,
    }
}

#[inline]
fn two_way_find(text: &[u8], pattern: &[u8], state: &TwoWayState) -> Option<usize> {
    let n = text.len();
    let m = pattern.len();

    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }
    if m == 1 {
        let needle = pattern[0];
        return text.iter().position(|&b| b == needle);
    }
    if m == 2 {
        let a = pattern[0];
        let b = pattern[1];
        // Simple tight loop; avoids two-way overhead for tiny needles.
        let mut i = 0usize;
        while i + 1 < n {
            if text[i] == a && text[i + 1] == b {
                return Some(i);
            }
            i += 1;
        }
        return None;
    }

    let crit = state.crit; // = ell + 1
    let pat = pattern.as_ptr();
    let txt = text.as_ptr();

    let last_start = n - m;
    let mut pos = 0usize;

    unsafe {
        if state.is_periodic {
            // memory = previous `memory + 1` (so 0 means C's memory == -1)
            let mut memory = 0usize;

            while pos <= last_start {
                // C: i = MAX(ell, memory) + 1
                // here: i = MAX(ell+1, memory+1) = MAX(crit, memory)
                let mut i = max(crit, memory);

                while i < m && *pat.add(i) == *txt.add(pos + i) {
                    i += 1;
                }

                if i >= m {
                    // C: i = ell; while (i > memory && ...) --i;
                    // Use i1 = i+1; start at crit (= ell+1).
                    let mut i1 = crit;
                    while i1 > memory && *pat.add(i1 - 1) == *txt.add(pos + i1 - 1) {
                        i1 -= 1;
                    }

                    if i1 <= memory {
                        return Some(pos);
                    }

                    pos += state.period;
                    // C: memory = m - per - 1  => memory+1 = m - per
                    memory = m - state.period;
                } else {
                    // C: j += (i - ell) ; with ell = crit - 1 => shift = i + 1 - crit
                    pos += i + 1 - crit;
                    memory = 0;
                }
            }
        } else {
            while pos <= last_start {
                // C: i = ell + 1 => i = crit
                let mut i = crit;

                while i < m && *pat.add(i) == *txt.add(pos + i) {
                    i += 1;
                }

                if i >= m {
                    // C: i = ell; while (i >= 0 && ...) --i;
                    // Use i1 = i+1; start at crit and go down to 0.
                    let mut i1 = crit;
                    while i1 > 0 && *pat.add(i1 - 1) == *txt.add(pos + i1 - 1) {
                        i1 -= 1;
                    }

                    if i1 == 0 {
                        return Some(pos);
                    }

                    pos += state.shift;
                } else {
                    pos += i + 1 - crit;
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_way_basic() {
        let hay = b"ababcabcabababd";
        let pat: &[u8] = b"ababd";
        assert_eq!(
            TwoWay::find(&pat, core::str::from_utf8(hay).unwrap()),
            Some(10)
        );
    }

    #[test]
    fn test_two_way_not_found() {
        let hay = b"hello world";
        let pat: &[u8] = b"rust";
        assert_eq!(TwoWay::find(&pat, core::str::from_utf8(hay).unwrap()), None);
    }

    #[test]
    fn test_two_way_empty_pattern() {
        let hay = b"abc";
        let pat: &[u8] = b"";
        assert_eq!(
            TwoWay::find(&pat, core::str::from_utf8(hay).unwrap()),
            Some(0)
        );
    }

    #[test]
    fn test_two_way_utf8() {
        let hay = "🌍hello🌍hello";
        let pat = "🌍hello";
        assert_eq!(TwoWay::find(&pat.as_bytes(), hay), Some(0));
    }

    #[test]
    fn test_two_way_matches_std_many() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *seed
        }

        let mut seed = 1u64;

        for _ in 0..5000 {
            let text_len = (next(&mut seed) % 96) as usize;
            let pat_len = (next(&mut seed) % 24) as usize;

            let mut text = Vec::with_capacity(text_len);
            let mut pat = Vec::with_capacity(pat_len);

            for _ in 0..text_len {
                let b = b'a' + (next(&mut seed) % 26) as u8;
                text.push(b);
            }
            for _ in 0..pat_len {
                let b = b'a' + (next(&mut seed) % 26) as u8;
                pat.push(b);
            }

            let text_s = core::str::from_utf8(&text).unwrap();
            let pat_s = core::str::from_utf8(&pat).unwrap();

            let got = TwoWay::find(&pat.as_slice(), text_s);
            let expect = text_s.find(pat_s);
            assert_eq!(got, expect, "text={text_s:?}, pat={pat_s:?}");
        }
    }

    #[test]
    fn test_two_way_matches_std_dna_many() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            *seed
        }

        const ALPHABET: &[u8] = b"ACGTN";
        let mut seed = 7u64;

        for _ in 0..30_000 {
            let text_len = (next(&mut seed) % 512) as usize;
            let pat_len = (next(&mut seed) % 32) as usize;

            let mut text = Vec::with_capacity(text_len);
            let mut pat = Vec::with_capacity(pat_len);

            for _ in 0..text_len {
                let idx = (next(&mut seed) as usize) % ALPHABET.len();
                text.push(ALPHABET[idx]);
            }
            for _ in 0..pat_len {
                let idx = (next(&mut seed) as usize) % ALPHABET.len();
                pat.push(ALPHABET[idx]);
            }

            let text_s = core::str::from_utf8(&text).unwrap();
            let pat_s = core::str::from_utf8(&pat).unwrap();
            let got = TwoWay::find(&pat.as_slice(), text_s);
            let expect = text_s.find(pat_s);
            assert_eq!(got, expect, "text={text_s:?}, pat={pat_s:?}");
        }
    }
}
