use std::{collections::HashMap, collections::HashSet, marker::PhantomData, sync::Arc};

use crate::StringSearch;

pub struct KmerConfig<'a> {
    pub pattern: &'a [u8],
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

pub struct KmerSearch<'a>(PhantomData<&'a ()>);

impl<'a> StringSearch for KmerSearch<'a> {
    type Config = KmerConfig<'a>;
    type State = KmerIndex;

    fn build(config: &Self::Config) -> Self::State {
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

    fn find_bytes(config: &Self::Config, state: &Self::State, text: &[u8]) -> Option<usize> {
        let state = &state.inner;
        let pattern = config.pattern;

        if state.map.is_empty() || text.len() < state.k || pattern.is_empty() {
            return None;
        }

        let mut diagonal_counts: HashMap<isize, usize> = HashMap::new();
        let mut candidates: Vec<isize> = Vec::new();
        let mut candidate_seen: HashSet<isize> = HashSet::new();

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
                        if !candidate_seen.contains(&diagonal) {
                            candidates.push(diagonal);
                            candidate_seen.insert(diagonal);
                        }
                    }
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        candidates.sort_unstable();

        for diagonal in candidates {
            let start = diagonal as usize;
            if start + pattern.len() > text.len() {
                continue;
            }
            if &text[start..start + pattern.len()] == pattern {
                return Some(start);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interface_compliance() {
        let pattern = b"ACGTACGT";
        let k = 3;
        // Total k-mers in pattern: 8 - 3 + 1 = 6.
        // We set min_hits to 6 to require a full match.
        let min_hits = 6;

        let config = KmerConfig {
            pattern,
            k,
            min_hits,
        };

        // 1. Build State
        let state = KmerSearch::build(&config);

        let text_single = b"__ACGTACGT__";
        // The match "ACGTACGT" starts at index 2 in text_single
        let found = KmerSearch::find_bytes(&config, &state, text_single);
        assert_eq!(found, Some(2));

        // 2. Test find_all (provided by trait)
        // Create text with two occurrences: index 0 and index 10
        let text_multi = b"ACGTACGT__ACGTACGT";
        let all_found = KmerSearch::find_all_bytes(&config, &state, text_multi);
        assert_eq!(all_found, vec![0, 10]);

        // 3. Test None
        let text_none = b"ZZZZZZZZZZ";
        let none_found = KmerSearch::find_bytes(&config, &state, text_none);
        assert_eq!(none_found, None);
    }

    #[test]
    fn test_partial_hits_threshold() {
        let pattern = b"AAAAA";
        let config = KmerConfig {
            pattern,
            k: 2,
            min_hits: 3,
        };
        let index = KmerSearch::build(&config);

        // Text has 3 'A's -> 2 kmers (AA, AA). Should fail (2 < 3).
        let text_fail = b"AAA";
        assert_eq!(KmerSearch::find_bytes(&config, &index, text_fail), None);

        // Text has 4 'A's -> 3 kmers (AA, AA, AA). Should pass (3 >= 3).
        let text_pass = b"AAAA";
        assert_eq!(KmerSearch::find_bytes(&config, &index, text_pass), None);
    }
}
