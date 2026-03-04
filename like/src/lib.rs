use std::marker::PhantomData;

use algos::StringSearch;

#[derive(Debug, PartialEq)]
pub enum Token<'a> {
    Literal(&'a str),
    Skip(usize), // Optimized '_' (matches N characters)
    Any,         // '%'
}

#[derive(Debug, Clone, Copy)]
enum MatchStrategy<'a> {
    All,
    Exact {
        lit: &'a str,
        state_idx: Option<usize>,
    },
    Prefix {
        lit: &'a str,
        state_idx: usize,
    },
    Suffix {
        lit: &'a str,
        state_idx: usize,
    },
    Contains {
        state_idx: usize,
    },
    General,
}

#[derive(Debug, Clone, Copy)]
struct LiteralPart<'a> {
    lit: &'a str,
    state_idx: usize,
}

pub struct Pattern<'a, S: StringSearch> {
    tokens: Box<[Token<'a>]>,
    min_len: usize,
    strategy: MatchStrategy<'a>,
    starts_with_any: bool,
    ends_with_any: bool,
    has_skip: bool,
    literal_parts: Box<[LiteralPart<'a>]>,

    literal_configs: Box<[S::Config]>,
    literal_states: Box<[S::State]>,

    literal_underscore_is_wildcard: bool,
    ascii_mode: bool,

    _marker: PhantomData<S>,
}

#[derive(Debug, Clone, Copy)]
pub struct CompileOptions {
    pub treat_underscore_as_literal: bool,
    pub literal_underscore_is_wildcard: bool,
    pub ascii_mode: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            treat_underscore_as_literal: false,
            literal_underscore_is_wildcard: false,
            ascii_mode: true,
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

    let starts_with_any = matches!(tokens.first(), Some(Token::Any));
    let ends_with_any = matches!(tokens.last(), Some(Token::Any));

    let mut has_skip = false;
    let mut literal_parts = Vec::new();
    let mut state_idx = 0usize;
    for token in tokens.iter() {
        match token {
            Token::Literal(lit) => {
                literal_parts.push(LiteralPart { lit, state_idx });
                state_idx += 1;
            }
            Token::Skip(_) => has_skip = true,
            Token::Any => {}
        }
    }

    let strategy = derive_strategy(pattern, options);

    Pattern {
        tokens: tokens.into_boxed_slice(),
        min_len,
        strategy,
        starts_with_any,
        ends_with_any,
        has_skip,
        literal_parts: literal_parts.into_boxed_slice(),
        literal_configs: literal_configs.into_boxed_slice(),
        literal_states: literal_states.into_boxed_slice(),
        literal_underscore_is_wildcard: options.literal_underscore_is_wildcard,
        ascii_mode: options.ascii_mode,
        _marker: PhantomData,
    }
}

#[inline(always)]
fn advance_units(text: &str, idx: usize, count: usize, ascii_mode: bool) -> Option<usize> {
    if ascii_mode {
        let next = idx + count;
        if next <= text.len() {
            Some(next)
        } else {
            None
        }
    } else {
        let mut chars = slice_from(text, idx).chars();
        let mut advanced = 0usize;
        let mut met = 0usize;
        while met < count {
            if let Some(c) = chars.next() {
                advanced += c.len_utf8();
                met += 1;
            } else {
                return None;
            }
        }
        Some(idx + advanced)
    }
}

#[inline(always)]
fn advance_one_unit(text: &str, idx: usize, ascii_mode: bool) -> Option<usize> {
    advance_units(text, idx, 1, ascii_mode)
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

fn derive_strategy<'a>(pattern: &'a str, options: CompileOptions) -> MatchStrategy<'a> {
    if pattern.is_empty() {
        return MatchStrategy::Exact {
            lit: "",
            state_idx: None,
        };
    }

    if !options.treat_underscore_as_literal && pattern.contains('_') {
        return MatchStrategy::General;
    }

    let literals: Vec<&str> = pattern.split('%').filter(|s| !s.is_empty()).collect();
    if literals.is_empty() {
        return MatchStrategy::All;
    }
    if literals.len() > 1 {
        return MatchStrategy::General;
    }

    let lit = literals[0];
    let starts = pattern.starts_with('%');
    let ends = pattern.ends_with('%');
    match (starts, ends) {
        (true, true) => MatchStrategy::Contains { state_idx: 0 },
        (true, false) => MatchStrategy::Suffix { lit, state_idx: 0 },
        (false, true) => MatchStrategy::Prefix { lit, state_idx: 0 },
        (false, false) => MatchStrategy::Exact {
            lit,
            state_idx: Some(0),
        },
    }
}

#[inline(always)]
fn try_match_percent_only<S: StringSearch>(pattern: &Pattern<S>, text: &str) -> Option<bool> {
    if pattern.has_skip {
        return None;
    }

    let parts = &pattern.literal_parts;
    if parts.is_empty() {
        return Some(true);
    }

    let mut from = 0usize;
    let mut first = 0usize;
    let mut after_last = parts.len();

    if !pattern.starts_with_any {
        let first_part = parts[0];
        if !literal_matches_at(pattern, first_part.lit, text, 0, first_part.state_idx) {
            return Some(false);
        }
        from = first_part.lit.len();
        first = 1;
    }

    let mut end_limit = text.len();
    if !pattern.ends_with_any {
        let last_part = parts[parts.len() - 1];
        if text.len() < last_part.lit.len() {
            return Some(false);
        }

        let suffix_idx = text.len() - last_part.lit.len();
        if !literal_matches_at(
            pattern,
            last_part.lit,
            text,
            suffix_idx,
            last_part.state_idx,
        ) {
            return Some(false);
        }

        end_limit = suffix_idx;
        after_last = parts.len() - 1;
    }

    for part in &parts[first..after_last] {
        if from > end_limit || part.lit.len() > end_limit.saturating_sub(from) {
            return Some(false);
        }

        let max_start = end_limit - part.lit.len();
        let config = &pattern.literal_configs[part.state_idx];
        let state = &pattern.literal_states[part.state_idx];

        let Some(found) = S::find_str(config, state, slice_from(text, from)) else {
            return Some(false);
        };

        let hit = from + found;
        if hit > max_start {
            return Some(false);
        }

        from = hit + part.lit.len();
    }

    Some(true)
}

// skip if pattern locally bigger or optimize around that
// also suffix match
pub fn like_match<S: StringSearch>(pattern: &Pattern<S>, text: &str) -> bool {
    match pattern.strategy {
        MatchStrategy::All => return true,
        MatchStrategy::Exact { lit, state_idx } => {
            if text.len() != lit.len() {
                return false;
            }
            if let Some(idx) = state_idx {
                return literal_matches_at(pattern, lit, text, 0, idx);
            }
            return true;
        }
        MatchStrategy::Prefix { lit, state_idx } => {
            if text.len() < lit.len() {
                return false;
            }
            return literal_matches_at(pattern, lit, text, 0, state_idx);
        }
        MatchStrategy::Suffix { lit, state_idx } => {
            if text.len() < lit.len() {
                return false;
            }
            return literal_matches_at(pattern, lit, text, text.len() - lit.len(), state_idx);
        }
        MatchStrategy::Contains { state_idx } => {
            let config = &pattern.literal_configs[state_idx];
            let state = &pattern.literal_states[state_idx];
            return S::find_str(config, state, text).is_some();
        }
        MatchStrategy::General => {
            if let Some(matched) = try_match_percent_only(pattern, text) {
                return matched;
            }
        }
    }

    if text.len() < pattern.min_len {
        return false;
    }

    let tokens = &pattern.tokens;
    let starts_with_any = pattern.starts_with_any;
    let ends_with_any = pattern.ends_with_any;
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
                    if let Some(next) = advance_units(text, s_idx, count, pattern.ascii_mode) {
                        s_idx = next;
                        t_idx += 1;
                        continue;
                    }
                }
                Token::Any => {
                    if t_idx + 1 >= tokens.len() {
                        return true;
                    }

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
                if let Some(next) = advance_one_unit(text, match_s_idx, pattern.ascii_mode) {
                    match_s_idx = next;
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
    use algos::{FftConfig, FftStr1, Naive, StdSearch, TwoWay, BM, KMP};

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

        let p = compile!("%ell%");
        assert!(like_match(&p, "hello"), "%ell% -> hello");

        let p = compile!("abc%");
        assert!(like_match(&p, "abcXYZ"), "abc% -> abcXYZ");

        let p = compile!("%");
        assert!(like_match(&p, "anything"), "% should match all");

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
    fn test_two_way_algorithm() {
        run_test_suite::<TwoWay, _>(|_, pat| unsafe {
            std::mem::transmute::<&[u8], &[u8]>(pat.as_bytes())
        });
    }

    #[test]
    fn test_underscore_literal_option() {
        let options = CompileOptions {
            treat_underscore_as_literal: true,
            literal_underscore_is_wildcard: false,
            ascii_mode: true,
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
    fn test_ascii_mode_underscore_is_byte() {
        let pattern = compile_pattern_with_options::<StdSearch, _, _>(
            "_",
            (),
            |_, pat| unsafe { std::mem::transmute::<&str, &str>(pat) },
            CompileOptions {
                ascii_mode: true,
                ..CompileOptions::default()
            },
        );

        assert!(!like_match(&pattern, "💩"));
        assert!(like_match(&pattern, "a"));
    }

    #[test]
    fn test_utf8_mode_underscore_is_char() {
        let pattern = compile_pattern_with_options::<StdSearch, _, _>(
            "_",
            (),
            |_, pat| unsafe { std::mem::transmute::<&str, &str>(pat) },
            CompileOptions {
                ascii_mode: false,
                ..CompileOptions::default()
            },
        );

        assert!(like_match(&pattern, "💩"));
        assert!(like_match(&pattern, "a"));
    }
}
