mod bm;
mod kmer;
mod kmp;
mod naive;

pub mod compare;

pub trait StringSearch {
    type Config;
    type State;

    fn build(config: &Self::Config) -> Self::State;

    fn find_bytes(config: &Self::Config, state: &Self::State, text: &[u8]) -> Option<usize>;

    fn find_str(config: &Self::Config, state: &Self::State, text: &str) -> Option<usize> {
        Self::find_bytes(config, state, text.as_bytes())
    }

    fn find(config: &Self::Config, text: &str) -> Option<usize> {
        let state = Self::build(config);
        Self::find_str(config, &state, text)
    }

    fn find_all_bytes(config: &Self::Config, state: &Self::State, text: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let mut cursor = 0;
        let len = text.len();

        while cursor <= len {
            let search_window = &text[cursor..];

            match Self::find_bytes(config, state, search_window) {
                Some(relative_pos) => {
                    let absolute_pos = cursor + relative_pos;
                    results.push(absolute_pos);

                    cursor = absolute_pos + 1;
                }
                None => break,
            }
        }

        results
    }

    fn find_all(config: &Self::Config, text: &str) -> Vec<usize> {
        let state = Self::build(config);
        Self::find_all_bytes(config, &state, text.as_bytes())
    }
}

use std::marker::PhantomData;

pub use bm::BM;
pub use kmer::{KmerConfig, KmerIndex, KmerSearch};
pub use kmp::KMP;
pub use naive::{Naive, NaiveScalar, NaiveVectorized};

pub struct StdSearch<'a>(PhantomData<&'a ()>);

impl<'a> StringSearch for StdSearch<'a> {
    type Config = &'a str;
    type State = ();

    fn build(_config: &Self::Config) -> Self::State {
        ()
    }

    fn find_bytes(_config: &Self::Config, _state: &Self::State, _text: &[u8]) -> Option<usize> {
        unimplemented!("StdSearch works on str only")
    }

    fn find_str(config: &Self::Config, _state: &Self::State, text: &str) -> Option<usize> {
        text.find(*config)
    }
}
