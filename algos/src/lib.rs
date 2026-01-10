mod bm;
mod kmer;
mod kmp;
mod naive;

pub mod compare;

pub trait StringSearch {
    type Config;
    type State;

    fn build(_config: Self::Config) -> Self::State {
        unimplemented!("this algorithm doesnt use build");
    }
    fn find_bytes(state: &Self::State, text: &[u8], pattern: &[u8]) -> Option<usize>;
    fn find_all_bytes(state: &Self::State, text: &[u8], pattern: &[u8]) -> Vec<usize>;
    fn find(state: &Self::State, text: &str, pattern: &str) -> Option<usize> {
        let text_bytes = text.as_bytes();
        let pattern_bytes = pattern.as_bytes();
        Self::find_bytes(state, text_bytes, pattern_bytes)
    }
    fn find_all(state: &Self::State, text: &str, pattern: &str) -> Vec<usize> {
        let text_bytes = text.as_bytes();
        let pattern_bytes = pattern.as_bytes();
        Self::find_all_bytes(state, text_bytes, pattern_bytes)
    }
}

pub use bm::BM;
pub use kmer::{KmerConfig, KmerIndex, KmerSearch};
pub use kmp::KMP;
pub use naive::{Naive, NaiveScalar, NaiveVectorized};

pub struct StdSearch;
impl StringSearch for StdSearch {
    type State = ();
    type Config = ();
    fn build(_config: Self::Config) -> Self::State {
        ()
    }
    fn find_bytes(_state: &Self::State, _text: &[u8], _pattern: &[u8]) -> Option<usize> {
        unimplemented!()
    }
    fn find_all_bytes(_state: &Self::State, _text: &[u8], _pattern: &[u8]) -> Vec<usize> {
        unimplemented!()
    }
    fn find(_state: &Self::State, text: &str, pattern: &str) -> Option<usize> {
        text.find(pattern)
    }
    fn find_all(_state: &Self::State, _text: &str, _pattern: &str) -> Vec<usize> {
        unimplemented!()
    }
}
