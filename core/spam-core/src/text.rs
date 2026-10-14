//! Character rules that must agree with Python and sklearn.

/// Python's `str.isspace()` (and so `\s`, `str.split()` and `str.strip()`). Wider than Rust's
/// `char::is_whitespace`: it also accepts U+001C–U+001F.
pub(crate) fn is_py_space(c: char) -> bool {
    matches!(c,
        '\u{9}'..='\u{d}' | '\u{1c}'..='\u{20}' | '\u{85}' | '\u{a0}' | '\u{1680}'
        | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_space_differs_from_rust_whitespace() {
        assert!(is_py_space('\u{1c}') && !'\u{1c}'.is_whitespace());
        assert!(is_py_space('\u{2009}') && is_py_space('\u{202f}'));
        assert!(!is_py_space('\u{200c}') && !is_py_space('\u{200b}'));
    }
}
