use crate::text::is_py_space;

/// Lowercase, then sklearn's `char_wb` n-grams for n = 2..=4.
///
/// Each whitespace-separated word is padded with one space on both sides and every window of
/// 2, 3 and 4 characters is taken. A word shorter than the window gives one window and ends
/// the n loop. Whitespace is Python's (`str.split()`), which includes U+2009 and U+202F.
pub fn ngrams(text: &str) -> Vec<String> {
    const MIN_N: usize = 2;
    const MAX_N: usize = 4;
    let lower = text.to_lowercase();
    let mut out = Vec::new();
    for word in lower.split(is_py_space).filter(|w| !w.is_empty()) {
        let w: Vec<char> = std::iter::once(' ')
            .chain(word.chars())
            .chain(std::iter::once(' '))
            .collect();
        for n in MIN_N..=MAX_N {
            let mut offset = 0;
            out.push(w[..n.min(w.len())].iter().collect());
            while offset + n < w.len() {
                offset += 1;
                out.push(w[offset..offset + n].iter().collect());
            }
            if offset == 0 {
                break; // count a short word only once
            }
        }
    }
    out
}
