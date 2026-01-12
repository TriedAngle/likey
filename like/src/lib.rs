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

    _marker: PhantomData<S>,
}

pub fn compile_pattern<'a, S, D, F>(
    pattern: &'a str,
    mut user_data: D,
    mut config_factory: F,
) -> Pattern<'a, S>
where
    S: StringSearch,
    F: FnMut(&mut D, &'a str) -> S::Config,
{
    let mut tokens = Vec::new();
    let mut literal_configs = Vec::new();
    let mut literal_states = Vec::new();
    let mut start_idx = 0;
    let mut min_len = 0;

    for (idx, c) in pattern.char_indices() {
        if c == '%' || c == '_' {
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

            match c {
                '%' => {
                    if tokens.last() != Some(&Token::Any) {
                        tokens.push(Token::Any);
                    }
                }
                '_' => {
                    if let Some(Token::Skip(count)) = tokens.last_mut() {
                        *count += 1;
                    } else {
                        tokens.push(Token::Skip(1));
                    }
                    min_len += 1;
                }
                _ => unreachable!(),
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
        _marker: PhantomData,
    }
}

pub fn like_match<S: StringSearch>(pattern: &Pattern<S>, text: &str) -> bool {
    if text.len() < pattern.min_len {
        return false;
    }

    let tokens = &pattern.tokens;
    let mut t_idx = 0;
    let mut s_idx = 0;

    let mut state_idx = 0;

    let mut last_wildcard_t_idx = None;
    let mut last_wildcard_state_idx = 0;
    let mut match_s_idx = 0;

    while s_idx < text.len() {
        if t_idx < tokens.len() {
            match tokens[t_idx] {
                Token::Literal(lit) => {
                    if text[s_idx..].starts_with(lit) {
                        s_idx += lit.len();
                        t_idx += 1;
                        state_idx += 1;
                        continue;
                    }
                }
                Token::Skip(count) => {
                    let mut chars = text[s_idx..].chars();
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
                            S::find_str(next_config, next_state, &text[s_idx..])
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
                    S::find_str(next_config, next_state, &text[search_start..])
                {
                    match_s_idx = search_start + found_offset;
                    s_idx = match_s_idx;
                    continue;
                } else {
                    return false;
                }
            } else {
                // Dumb Backtracking
                if let Some(c) = text[match_s_idx..].chars().next() {
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
    use algos::{BM, KMP, Naive, StdSearch};

    fn run_test_suite<S, F>(factory: F)
    where
        S: StringSearch,
        F: FnMut(&mut (), &str) -> S::Config + Clone,
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
}
