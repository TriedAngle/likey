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
    fn find_bytes(state: Self::State, text: &[u8], pattern: &[u8]) -> Option<usize>;
    fn find_all_bytes(state: Self::State, text: &[u8], pattern: &[u8]) -> Vec<usize>;
    fn find(state: Self::State, text: &str, pattern: &str) -> Option<usize> {
        let text_bytes = text.as_bytes();
        let pattern_bytes = pattern.as_bytes();
        Self::find_bytes(state, text_bytes, pattern_bytes)
    }
    fn find_all(state: Self::State, text: &str, pattern: &str) -> Vec<usize> {
        let text_bytes = text.as_bytes();
        let pattern_bytes = pattern.as_bytes();
        Self::find_all_bytes(state, text_bytes, pattern_bytes)
    }
}

pub use naive::{Naive, NaiveScalar, NaiveVectorized};
pub use kmp::KMP;
pub use bm::BM;
pub use kmer::{KmerIndex, KmerConfig, KmerSearch};

