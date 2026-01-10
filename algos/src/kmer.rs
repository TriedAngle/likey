use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::StringSearch;

pub struct KmerConfig {
    pub pattern: Vec<u8>,
    pub k: usize,
    pub min_hits: usize,
}

#[derive(Debug, Clone)]
pub struct KmerIndex {
    inner: Arc<KmerIndexInner>,
}

#[derive(Debug)]
pub struct KmerIndexInner {
    map: HashMap<Vec<u8>, Vec<usize>>,
    k: usize,
    min_hits: usize,
}

pub struct KmerSearch;

impl StringSearch for KmerSearch {
    type Config = KmerConfig;
    type State = KmerIndex;

    fn build(config: Self::Config) -> Self::State {
        let mut map = HashMap::<Vec<u8>, Vec<usize>>::new();
        let k = config.k;

        if k > 0 && config.pattern.len() >= k {
            for i in 0..=config.pattern.len() - k {
                let kmer = config.pattern[i..i + k].to_vec();
                map.entry(kmer).or_default().push(i);
            }
        }

        let inner = KmerIndexInner {
            map,
            k: config.k,
            min_hits: config.min_hits,
        };

        KmerIndex {
            inner: Arc::new(inner),
        }
    }

    fn find_bytes(state: &Self::State, text: &[u8], _pattern: &[u8]) -> Option<usize> {
        let state = state.inner.clone();
        if state.map.is_empty() || text.len() < state.k {
            return None;
        }

        let mut diagonal_counts: HashMap<isize, usize> = HashMap::new();

        for text_pos in 0..=text.len() - state.k {
            let kmer = &text[text_pos..text_pos + state.k];

            if let Some(query_positions) = state.map.get(kmer) {
                for &query_pos in query_positions {
                    let diagonal = text_pos as isize - query_pos as isize;

                    if diagonal < 0 {
                        continue;
                    }

                    let count = diagonal_counts.entry(diagonal).or_insert(0);
                    *count += 1;

                    if *count >= state.min_hits {
                        return Some(diagonal as usize);
                    }
                }
            }
        }

        None
    }

    fn find_all_bytes(state: &Self::State, text: &[u8], _pattern: &[u8]) -> Vec<usize> {
        let state = state.inner.clone();
        if state.map.is_empty() || text.len() < state.k {
            return Vec::new();
        }

        let mut diagonal_counts: HashMap<isize, usize> = HashMap::new();
        let mut found_diagonals: HashSet<isize> = HashSet::new();
        let mut results = Vec::new();

        for text_pos in 0..=text.len() - state.k {
            let kmer = &text[text_pos..text_pos + state.k];

            if let Some(query_positions) = state.map.get(kmer) {
                for &query_pos in query_positions {
                    let diagonal = text_pos as isize - query_pos as isize;

                    if diagonal < 0 {
                        continue;
                    }

                    let count = diagonal_counts.entry(diagonal).or_insert(0);
                    *count += 1;

                    if *count >= state.min_hits {
                        if found_diagonals.insert(diagonal) {
                            results.push(diagonal as usize);
                        }
                    }
                }
            }
        }

        results.sort();
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interface_compliance() {
        let pattern = b"ACGTACGT".to_vec();
        let k = 3;
        // Total k-mers in pattern: 8 - 3 + 1 = 6.
        // We set min_hits to 6 to require a full match.
        let min_hits = 6;

        let config = KmerConfig {
            pattern: pattern.clone(),
            k,
            min_hits,
        };
        let index = KmerSearch::build(config);

        let text_single = b"__ACGTACGT__";
        // The match "ACGTACGT" starts at index 2 in text_single
        let found = KmerSearch::find_bytes(&index, text_single, &[]);
        assert_eq!(found, Some(2));

        // Create text with two occurrences: index 0 and index 10
        let text_multi = b"ACGTACGT__ACGTACGT";
        let all_found = KmerSearch::find_all_bytes(&index, text_multi, &[]);
        assert_eq!(all_found, vec![0, 10]);

        let text_none = b"ZZZZZZZZZZ";
        let none_found = KmerSearch::find_bytes(&index, text_none, &[]);
        assert_eq!(none_found, None);
    }

    // this test illustrates that the amount of kmers should be >= min_hits
    #[test]
    fn test_partial_hits_threshold() {
        let pattern = b"AAAAA".to_vec();
        let config = KmerConfig {
            pattern,
            k: 2,
            min_hits: 3,
        };
        let index = KmerSearch::build(config);

        // Text has 3 'A's -> 2 kmers (AA, AA). Should fail (2 < 3).
        let text_fail = b"AAA";
        assert_eq!(KmerSearch::find_bytes(&index.clone(), text_fail, &[]), None);

        // Text has 4 'A's -> 3 kmers (AA, AA, AA). Should pass (3 >= 3).
        let text_pass = b"AAAA";
        assert_eq!(KmerSearch::find_bytes(&index, text_pass, &[]), Some(0));
    }
}
