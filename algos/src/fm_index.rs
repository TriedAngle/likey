#![allow(dead_code)]
// FM-index over a byte corpus for fast substring search.
// This is a baseline implementation: naive suffix array construction and full
// Occ tables. It is correct but not optimized for memory or build time.
// The caller is responsible for inserting row separators and a sentinel that
// do not appear in the input text.
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct FMIndex {
    text: Vec<u8>,
    sa: Vec<usize>,
    bwt: Vec<u8>,
    c: Vec<usize>,
    counts: Vec<usize>,
    occ: Vec<Vec<u32>>,
    checkpoint: usize,
    byte_to_rank: [i16; 256],
    rank_to_byte: Vec<u8>,
    sentinel_rank: usize,
    separator_rank: Option<usize>,
    sentinel: u8,
    separator: Option<u8>,
}

impl FMIndex {
    pub fn new(mut text: Vec<u8>, sentinel: u8, separator: Option<u8>) -> Self {
        if let Some(sep) = separator {
            assert!(sep != sentinel, "separator must differ from sentinel");
        }

        if text.iter().any(|&b| b == sentinel) {
            panic!("sentinel byte appears in text");
        }

        text.push(sentinel);

        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa, sentinel);

        let (byte_to_rank, rank_to_byte, counts) = build_alphabet(&text);
        let sentinel_rank = byte_to_rank[sentinel as usize]
            .try_into()
            .expect("sentinel rank must exist");
        let separator_rank = separator.and_then(|sep| {
            let rank = byte_to_rank[sep as usize];
            if rank < 0 {
                None
            } else {
                Some(rank as usize)
            }
        });
        let c = build_c(&counts);
        let bwt = remap_bwt(&bwt, &byte_to_rank);

        let checkpoint = 128usize;
        let occ = build_occ(&bwt, counts.len(), checkpoint);

        Self {
            text,
            sa,
            bwt,
            c,
            counts,
            occ,
            checkpoint,
            byte_to_rank,
            rank_to_byte,
            sentinel_rank,
            separator_rank,
            sentinel,
            separator,
        }
    }

    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn backward_search(&self, pattern: &[u8]) -> Option<(usize, usize)> {
        if pattern.is_empty() {
            return Some((0, self.len()));
        }

        let mut top = 0usize;
        let mut bottom = self.len();

        for &ch in pattern.iter().rev() {
            let rank = match self.rank_for_byte(ch) {
                Some(rank) => rank,
                None => return None,
            };
            if self.counts[rank] == 0 {
                return None;
            }

            top = self.c[rank] + self.occ_at(rank, top);
            bottom = self.c[rank] + self.occ_at(rank, bottom);

            if top >= bottom {
                return None;
            }
        }

        Some((top, bottom))
    }

    pub fn search(&self, pattern: &[u8]) -> Vec<usize> {
        match self.backward_search(pattern) {
            Some((top, bottom)) => {
                let mut out = self.sa[top..bottom].to_vec();
                out.sort_unstable();
                out
            }
            None => Vec::new(),
        }
    }

    pub fn search_with_underscore(&self, pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() {
            return (0..self.len()).collect();
        }

        let mut results = HashSet::new();

        fn rec(
            fm: &FMIndex,
            pattern: &[u8],
            idx: isize,
            top: usize,
            bottom: usize,
            results: &mut HashSet<usize>,
        ) {
            if idx < 0 {
                for &pos in &fm.sa[top..bottom] {
                    results.insert(pos);
                }
                return;
            }

            let ch = pattern[idx as usize];
            if ch == b'_' {
                let mut seen = vec![false; fm.counts.len()];
                for &rank in &fm.bwt[top..bottom] {
                    let r = rank as usize;
                    if seen[r] {
                        continue;
                    }
                    seen[r] = true;
                    if r == fm.sentinel_rank {
                        continue;
                    }
                    if let Some(sep_rank) = fm.separator_rank {
                        if r == sep_rank {
                            continue;
                        }
                    }
                    if fm.counts[r] == 0 {
                        continue;
                    }

                    let new_top = fm.c[r] + fm.occ_at(r, top);
                    let new_bottom = fm.c[r] + fm.occ_at(r, bottom);
                    if new_top < new_bottom {
                        rec(fm, pattern, idx - 1, new_top, new_bottom, results);
                    }
                }
            } else {
                let rank = match fm.rank_for_byte(ch) {
                    Some(rank) => rank,
                    None => return,
                };
                if fm.counts[rank] == 0 {
                    return;
                }
                let new_top = fm.c[rank] + fm.occ_at(rank, top);
                let new_bottom = fm.c[rank] + fm.occ_at(rank, bottom);
                if new_top < new_bottom {
                    rec(fm, pattern, idx - 1, new_top, new_bottom, results);
                }
            }
        }

        rec(
            self,
            pattern,
            (pattern.len() as isize) - 1,
            0,
            self.len(),
            &mut results,
        );

        let mut out: Vec<usize> = results.into_iter().collect();
        out.sort_unstable();
        out
    }

    fn occ_at(&self, rank: usize, index: usize) -> usize {
        let capped = index.min(self.len());
        let base_idx = capped / self.checkpoint;
        let base_pos = base_idx * self.checkpoint;
        let mut count = self.occ[base_idx][rank] as usize;
        for &r in &self.bwt[base_pos..capped] {
            if r as usize == rank {
                count += 1;
            }
        }
        count
    }

    fn rank_for_byte(&self, ch: u8) -> Option<usize> {
        let rank = self.byte_to_rank[ch as usize];
        if rank < 0 {
            None
        } else {
            Some(rank as usize)
        }
    }
}

fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    let mut sa: Vec<usize> = (0..text.len()).collect();
    sa.sort_by(|&a, &b| text[a..].cmp(&text[b..]));
    sa
}

fn build_bwt(text: &[u8], sa: &[usize], sentinel: u8) -> Vec<u8> {
    let mut bwt = Vec::with_capacity(sa.len());
    for &pos in sa {
        if pos == 0 {
            bwt.push(sentinel);
        } else {
            bwt.push(text[pos - 1]);
        }
    }
    bwt
}

fn build_alphabet(text: &[u8]) -> ([i16; 256], Vec<u8>, Vec<usize>) {
    let mut counts_by_byte = [0usize; 256];
    for &b in text {
        counts_by_byte[b as usize] += 1;
    }

    let mut byte_to_rank = [-1i16; 256];
    let mut rank_to_byte = Vec::new();
    let mut counts = Vec::new();

    for byte in 0..256u16 {
        let b = byte as u8;
        let count = counts_by_byte[b as usize];
        if count == 0 {
            continue;
        }
        let rank = rank_to_byte.len();
        byte_to_rank[b as usize] = rank as i16;
        rank_to_byte.push(b);
        counts.push(count);
    }

    (byte_to_rank, rank_to_byte, counts)
}

fn build_c(counts: &[usize]) -> Vec<usize> {
    let mut c = Vec::with_capacity(counts.len());
    let mut total = 0usize;
    for &count in counts {
        c.push(total);
        total += count;
    }
    c
}

fn remap_bwt(bwt: &[u8], byte_to_rank: &[i16; 256]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bwt.len());
    for &b in bwt {
        let rank = byte_to_rank[b as usize];
        if rank < 0 {
            panic!("missing rank for bwt byte");
        }
        out.push(rank as u8);
    }
    out
}

fn build_occ(bwt: &[u8], sigma: usize, checkpoint: usize) -> Vec<Vec<u32>> {
    let mut occ = Vec::new();
    let mut counts = vec![0u32; sigma];
    occ.push(counts.clone());

    for (idx, &rank) in bwt.iter().enumerate() {
        counts[rank as usize] += 1;
        if (idx + 1) % checkpoint == 0 {
            occ.push(counts.clone());
        }
    }

    if bwt.len() % checkpoint != 0 {
        occ.push(counts);
    }

    occ
}

#[cfg(test)]
mod tests {
    use super::FMIndex;

    const SEP: u8 = 0x1F;
    const SENTINEL: u8 = 0x00;

    fn sample_index() -> FMIndex {
        let text = b"banana\x1fbandana\x1fapple".to_vec();
        FMIndex::new(text, SENTINEL, Some(SEP))
    }

    #[test]
    fn test_exact_search() {
        let fm = sample_index();
        let matches = fm.search(b"ana");
        assert_eq!(matches, vec![1, 3, 11]);
    }

    #[test]
    fn test_search_with_underscore() {
        let fm = sample_index();
        let matches = fm.search_with_underscore(b"b_n");
        assert_eq!(matches, vec![0, 7]);
    }

    #[test]
    fn test_complex_wildcard() {
        let fm = sample_index();
        let matches = fm.search_with_underscore(b"a__le");
        assert_eq!(matches, vec![15]);
    }
}
