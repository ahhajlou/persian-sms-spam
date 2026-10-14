use fancy_regex::{Captures, Regex, RegexBuilder};

/// Backtracking steps one regex run may take. Input is SMS-sized (a long multi-part message
/// is under 2,000 characters) and normal text needs a few thousand steps, so this only
/// stops crafted input.
pub(crate) const BACKTRACK_LIMIT: usize = 200_000;

pub(crate) fn compile(pattern: &str) -> Regex {
    RegexBuilder::new(pattern)
        .backtrack_limit(BACKTRACK_LIMIT)
        .build()
        .unwrap_or_else(|e| panic!("invalid rule {pattern:?}: {e}"))
}

/// Replace every match with what `f` returns (`None` keeps the match as it is).
///
/// If the regex gives up (backtrack limit), the whole rule is skipped: the text comes back
/// unchanged from this call and `limit_hits` goes up by one.
pub(crate) fn replace_all<F>(re: &Regex, text: &str, limit_hits: &mut usize, mut f: F) -> String
where
    F: FnMut(&Captures) -> Option<String>,
{
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for caps in re.captures_iter(text) {
        let Ok(caps) = caps else {
            *limit_hits += 1;
            return text.to_string();
        };
        let m = caps.get(0).expect("group 0 always exists");
        out.push_str(&text[last..m.start()]);
        match f(&caps) {
            Some(replacement) => out.push_str(&replacement),
            None => out.push_str(m.as_str()),
        }
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}
