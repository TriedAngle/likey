use std::marker::PhantomData;

use algos::StringSearch;

#[derive(Debug, PartialEq)]
pub enum Token<'a> {
    Literal(&'a str),
    Skip(usize), // Optimized '_' (matches N characters)
    Any,         // '%'
}

pub struct Pattern<'a, S: StringSearch> {
    tokens: Box<[Token<'a>]>,
    min_len: usize,

    literal_configs: Box<[S::Config]>,
    literal_states: Box<[S::State]>,

    literal_underscore_is_wildcard: bool,

    _marker: PhantomData<S>,
}

#[derive(Debug, Clone, Copy)]
pub struct CompileOptions {
    pub treat_underscore_as_literal: bool,
    pub literal_underscore_is_wildcard: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            treat_underscore_as_literal: false,
            literal_underscore_is_wildcard: false,
        }
    }
}

pub fn compile_pattern<'a, S, D, F>(
    pattern: &'a str,
    user_data: D,
    config_factory: F,
) -> Pattern<'a, S>
where
    S: StringSearch,
    F: FnMut(&mut D, &'a str) -> S::Config,
{
    compile_pattern_with_options(
        pattern,
        user_data,
        config_factory,
        CompileOptions::default(),
    )
}

pub fn compile_pattern_with_options<'a, S, D, F>(
    pattern: &'a str,
    mut user_data: D,
    mut config_factory: F,
    options: CompileOptions,
) -> Pattern<'a, S>
where
    S: StringSearch,
    F: FnMut(&mut D, &'a str) -> S::Config,
{
    if options.literal_underscore_is_wildcard && !options.treat_underscore_as_literal {
        panic!("literal underscore wildcard requires treat_underscore_as_literal");
    }

    let mut tokens = Vec::new();
    let mut literal_configs = Vec::new();
    let mut literal_states = Vec::new();
    let mut start_idx = 0;
    let mut min_len = 0;

    for (idx, c) in pattern.char_indices() {
        let is_wildcard = c == '%' || (c == '_' && !options.treat_underscore_as_literal);
        if is_wildcard {
            if idx > start_idx {
                let lit = &pattern[start_idx..idx];
                tokens.push(Token::Literal(lit));

                // Generate config and state
                let config = config_factory(&mut user_data, lit);
                let state = S::build(&config);

                literal_configs.push(config);
                literal_states.push(state);

                min_len += lit.len();
            }

            if c == '%' {
                if tokens.last() != Some(&Token::Any) {
                    tokens.push(Token::Any);
                }
            } else if c == '_' {
                if let Some(Token::Skip(count)) = tokens.last_mut() {
                    *count += 1;
                } else {
                    tokens.push(Token::Skip(1));
                }
                min_len += 1;
            }
            start_idx = idx + c.len_utf8();
        }
    }

    if start_idx < pattern.len() {
        let lit = &pattern[start_idx..];
        tokens.push(Token::Literal(lit));

        let config = config_factory(&mut user_data, lit);
        let state = S::build(&config);

        literal_configs.push(config);
        literal_states.push(state);

        min_len += lit.len();
    }

    Pattern {
        tokens: tokens.into_boxed_slice(),
        min_len,
        literal_configs: literal_configs.into_boxed_slice(),
        literal_states: literal_states.into_boxed_slice(),
        literal_underscore_is_wildcard: options.literal_underscore_is_wildcard,
        _marker: PhantomData,
    }
}

#[inline(always)]
fn slice_from(text: &str, idx: usize) -> &str {
    debug_assert!(idx <= text.len());
    unsafe { text.get_unchecked(idx..) }
}

#[inline(always)]
fn literal_matches_at<S: StringSearch>(
    pattern: &Pattern<S>,
    lit: &str,
    text: &str,
    idx: usize,
    state_idx: usize,
) -> bool {
    if pattern.literal_underscore_is_wildcard && lit.as_bytes().contains(&b'_') {
        let config = &pattern.literal_configs[state_idx];
        let state = &pattern.literal_states[state_idx];
        S::find_str(config, state, slice_from(text, idx)) == Some(0)
    } else {
        slice_from(text, idx).starts_with(lit)
    }
}

// skip if pattern locally bigger or optimize around that
// also suffix match
pub fn like_match<S: StringSearch>(pattern: &Pattern<S>, text: &str) -> bool {
    if text.len() < pattern.min_len {
        return false;
    }

    let tokens = &pattern.tokens;
    let starts_with_any = matches!(tokens.first(), Some(Token::Any));
    let ends_with_any = matches!(tokens.last(), Some(Token::Any));
    let mut t_idx = 0;
    let mut s_idx = 0;

    let mut state_idx = 0;

    let mut last_wildcard_t_idx = None;
    let mut last_wildcard_state_idx = 0;
    let mut match_s_idx = 0;

    if !starts_with_any {
        if let Some(Token::Literal(lit)) = tokens.first() {
            if !lit.is_empty() {
                if !literal_matches_at(pattern, lit, text, 0, 0) {
                    return false;
                }
                s_idx = lit.len();
                t_idx = 1;
                state_idx = 1;
            }
        }
    }

    if !ends_with_any {
        if let Some(Token::Literal(lit)) = tokens.last() {
            if !lit.is_empty() {
                let last_state_idx = pattern.literal_configs.len().saturating_sub(1);
                if text.len() < lit.len()
                    || !literal_matches_at(
                        pattern,
                        lit,
                        text,
                        text.len() - lit.len(),
                        last_state_idx,
                    )
                {
                    return false;
                }
            }
        }
    }

    while s_idx < text.len() {
        if t_idx < tokens.len() {
            match tokens[t_idx] {
                Token::Literal(lit) => {
                    if literal_matches_at(pattern, lit, text, s_idx, state_idx) {
                        s_idx += lit.len();
                        t_idx += 1;
                        state_idx += 1;
                        continue;
                    }
                }
                Token::Skip(count) => {
                    let mut chars = slice_from(text, s_idx).chars();
                    let mut advanced = 0;
                    let mut met = 0;
                    while met < count {
                        if let Some(c) = chars.next() {
                            advanced += c.len_utf8();
                            met += 1;
                        } else {
                            break;
                        }
                    }
                    if met == count {
                        s_idx += advanced;
                        t_idx += 1;
                        continue;
                    }
                }
                Token::Any => {
                    last_wildcard_t_idx = Some(t_idx);
                    last_wildcard_state_idx = state_idx;

                    // OPTIMIZATION: Smart Jump
                    if let Some(Token::Literal(_)) = tokens.get(t_idx + 1) {
                        let next_config = &pattern.literal_configs[state_idx];
                        let next_state = &pattern.literal_states[state_idx];

                        if let Some(found_offset) =
                            S::find_str(next_config, next_state, slice_from(text, s_idx))
                        {
                            match_s_idx = s_idx + found_offset;
                            s_idx = match_s_idx;
                            t_idx += 1;
                            continue;
                        } else {
                            return false;
                        }
                    }

                    t_idx += 1;
                    match_s_idx = s_idx;
                    continue;
                }
            }
        }

        // Backtracking
        if let Some(wildcard_idx) = last_wildcard_t_idx {
            t_idx = wildcard_idx + 1;
            state_idx = last_wildcard_state_idx;

            if let Some(Token::Literal(_)) = tokens.get(t_idx) {
                // Smart Jump Backtracking
                let search_start = match_s_idx + 1;
                if search_start >= text.len() {
                    return false;
                }

                let next_config = &pattern.literal_configs[state_idx];
                let next_state = &pattern.literal_states[state_idx];

                if let Some(found_offset) =
                    S::find_str(next_config, next_state, slice_from(text, search_start))
                {
                    match_s_idx = search_start + found_offset;
                    s_idx = match_s_idx;
                    continue;
                } else {
                    return false;
                }
            } else {
                // Dumb Backtracking
                if let Some(c) = slice_from(text, match_s_idx).chars().next() {
                    match_s_idx += c.len_utf8();
                    s_idx = match_s_idx;
                    continue;
                }
            }
        }

        return false;
    }

    while t_idx < tokens.len() {
        if let Token::Any = tokens[t_idx] {
            t_idx += 1;
        } else {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use algos::{FftConfig, FftStr1, Naive, StdSearch, BM, KMP};

    fn run_test_suite<S, F>(factory: F)
    where
        S: StringSearch,
        F: FnMut(&mut (), &'static str) -> S::Config + Clone,
    {
        // Wrapper to simplify the calls inside tests
        macro_rules! compile {
            ($pat:expr) => {
                compile_pattern::<S, _, _>($pat, (), factory.clone())
            };
        }

        // Test 1: Basic Exact Match
        let p = compile!("hello");
        assert!(like_match(&p, "hello"), "Basic match failed");
        assert!(
            !like_match(&p, "hello world"),
            "Basic prefix match incorrectly passed"
        );

        // Test 2: % Wildcard
        let p = compile!("h%o");
        assert!(like_match(&p, "hello"), "h%o -> hello");
        assert!(like_match(&p, "ho"), "h%o -> ho");
        assert!(like_match(&p, "h_long_string_o"), "h%o -> long string");
        assert!(!like_match(&p, "h"), "h%o -> h (missing o)");

        // Test 3: _ Wildcard
        let p = compile!("h_t");
        assert!(like_match(&p, "hat"));
        assert!(!like_match(&p, "heat"));

        // Test 4: Backtracking
        let p = compile!("%a");
        assert!(like_match(&p, "banana"), "%a -> banana (needs backtrack)");
        assert!(like_match(&p, "pizza"), "%a -> pizza");

        let p = compile!("a%b");
        assert!(like_match(&p, "abb"), "a%b -> abb (needs backtrack)");

        // Test 5: Complex Mix
        let p = compile!("a_%_b");
        assert!(like_match(&p, "ax_b"));
        assert!(like_match(&p, "a_long___b"));
        assert!(!like_match(&p, "ab"));

        // Test 6: UTF-8
        let p = compile!("_%");
        assert!(like_match(&p, "💩"));
        assert!(like_match(&p, "💩more"));
    }

    #[test]
    fn test_std_algorithm() {
        run_test_suite::<StdSearch, _>(|_, pat| unsafe { std::mem::transmute::<&str, &str>(pat) });
    }

    #[test]
    fn test_kmp_algorithm() {
        run_test_suite::<KMP, _>(|_, pat| unsafe {
            std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes())
        });
    }

    #[test]
    fn test_naive_algorithm() {
        run_test_suite::<Naive, _>(|_, pat| unsafe {
            std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes())
        });
    }

    #[test]
    fn test_bm_algorithm() {
        run_test_suite::<BM, _>(|_, pat| unsafe {
            std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes())
        });
    }

    #[test]
    fn test_underscore_literal_option() {
        let options = CompileOptions {
            treat_underscore_as_literal: true,
            literal_underscore_is_wildcard: false,
        };
        let pattern = compile_pattern_with_options::<StdSearch, _, _>(
            "%a_c%",
            (),
            |_, pat| unsafe { std::mem::transmute::<&str, &str>(pat) },
            options,
        );

        assert!(like_match(&pattern, "zza_czz"));
        assert!(!like_match(&pattern, "zzabczz"));
    }

    #[test]
    #[ignore]
    fn reproduce_fftstr_underscore_literal_mismatch() {
        let options = CompileOptions {
            treat_underscore_as_literal: true,
            literal_underscore_is_wildcard: true,
        };
        let pattern = compile_pattern_with_options::<FftStr1, _, _>(
            "%a_%_b%",
            (),
            |_, pat| FftConfig::from_str(pat),
            options,
        );
        let text = "zzaXfooYbzz";
        let first_config = &pattern.literal_configs[0];
        let first_state = &pattern.literal_states[0];
        let second_config = &pattern.literal_configs[1];
        let second_state = &pattern.literal_states[1];

        assert_eq!(FftStr1::find_str(first_config, first_state, text), Some(2));
        assert_eq!(
            FftStr1::find_str(first_config, first_state, &text[2..]),
            Some(0)
        );
        assert_eq!(
            FftStr1::find_str(second_config, second_state, &text[4..]),
            Some(3)
        );
        assert_eq!(
            FftStr1::find_str(second_config, second_state, &text[7..]),
            Some(0)
        );
        assert!(literal_matches_at(&pattern, "a_", text, 2, 0));
        assert!(literal_matches_at(&pattern, "_b", text, 7, 1));

        assert!(like_match(&pattern, text));
    }
}
