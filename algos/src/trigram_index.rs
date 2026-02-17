use std::collections::{HashMap, HashSet};

#[derive(Debug, Default)]
pub struct TrigramIndex<'a> {
    index: HashMap<u32, Vec<u32>>,
    documents: Vec<&'a str>,
}

impl<'a> TrigramIndex<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, text: &'a str) -> u32 {
        let doc_id = self.documents.len() as u32;
        self.documents.push(text);

        let mut seen = HashSet::new();
        for tri in trigrams(text.as_bytes()) {
            if seen.insert(tri) {
                self.index.entry(tri).or_default().push(doc_id);
            }
        }

        doc_id
    }

    pub fn search_literal(&self, literal: &str) -> Option<Vec<u32>> {
        let grams = trigrams(literal.as_bytes());
        if grams.is_empty() {
            return None;
        }

        let mut lists: Vec<&[u32]> = Vec::new();
        for tri in grams {
            let list = self.index.get(&tri)?;
            lists.push(list);
        }

        lists.sort_by_key(|list| list.len());
        let mut result: Vec<u32> = lists[0].to_vec();
        for list in lists.iter().skip(1) {
            result = intersect_sorted(&result, list);
            if result.is_empty() {
                break;
            }
        }

        Some(result)
    }

    pub fn document(&self, doc_id: u32) -> Option<&'a str> {
        self.documents.get(doc_id as usize).copied()
    }
}

fn trigrams(bytes: &[u8]) -> Vec<u32> {
    if bytes.len() < 3 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(bytes.len() - 2);
    for i in 0..=bytes.len() - 3 {
        let b0 = bytes[i] as u32;
        let b1 = bytes[i + 1] as u32;
        let b2 = bytes[i + 2] as u32;
        let key = (b0 << 16) | (b1 << 8) | b2;
        out.push(key);
    }

    out
}

fn intersect_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() && j < b.len() {
        let va = a[i];
        let vb = b[j];
        if va == vb {
            out.push(va);
            i += 1;
            j += 1;
        } else if va < vb {
            i += 1;
        } else {
            j += 1;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::TrigramIndex;

    #[test]
    fn test_trigram_candidates() {
        let mut idx = TrigramIndex::new();
        let docs = [
            "apple",
            "applet",
            "pineapple",
            "application",
            "banana",
            "bandana",
        ];

        for doc in docs.iter() {
            idx.add(doc);
        }

        let ids = idx.search_literal("appl").unwrap();
        let results: Vec<&str> = ids
            .into_iter()
            .map(|id| idx.document(id).unwrap())
            .collect();
        assert_eq!(results, vec!["apple", "applet", "pineapple", "application"]);

        let ids = idx.search_literal("ana").unwrap();
        let results: Vec<&str> = ids
            .into_iter()
            .map(|id| idx.document(id).unwrap())
            .collect();
        assert_eq!(results, vec!["banana", "bandana"]);

        let ids = idx.search_literal("pine").unwrap();
        let results: Vec<&str> = ids
            .into_iter()
            .map(|id| idx.document(id).unwrap())
            .collect();
        assert_eq!(results, vec!["pineapple"]);
    }

    #[test]
    fn test_trigram_short_literal() {
        let mut idx = TrigramIndex::new();
        idx.add("abc");
        assert!(idx.search_literal("an").is_none());
    }
}
