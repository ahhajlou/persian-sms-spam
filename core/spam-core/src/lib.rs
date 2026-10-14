//! Persian SMS preprocessing shared by training (Python) and the NoSpam app (Android).
//!
//! ```text
//! raw SMS ─► anonymize ─► mask URLs ─► mask numbers ─► normalize ─► lowercase ─► char n-grams
//! ```
//!
//! The order matters, and the reasons are in `common/src/spam_common/pipeline.py`'s docstring:
//! anonymize needs the raw digits, URLs are masked before hazm's spacing rules can break them.
//! The model math (idf, L2 norm, dot product) is not part of this crate.
//!
//! Every regex rule lives in `rules/` as data and uses explicit character classes only, so
//! the behavior does not depend on what `\d \w \s \b` mean in an engine.
//!
//! Bump [`PIPELINE_VERSION`] on any change that alters the output of any function here; the
//! golden files in `tests/golden/` fail `cargo test` until you do.

use std::collections::BTreeMap;

pub mod anonymize;
#[doc(hidden)]
pub mod golden;
mod hazm;
mod ngrams;
mod regex_util;
mod rules;
mod text;
mod urls;

pub use anonymize::{anonymize, card_ok, national_id_ok, Anonymized};
pub use hazm::{Lexicons, Normalizer};
pub use ngrams::ngrams;
pub use urls::{mask_numbers, mask_urls};

/// The output contract. 1–4 were the Python pipeline; 5 is the first Rust one
/// (hazm `rules` normalizer, ASCII-only digit classes in the anonymizer).
pub const PIPELINE_VERSION: u32 = 5;

/// Only this many characters of a message are used; the rest is dropped before any step runs.
/// The longest real message in the dataset has 941, so real output does not change; it bounds
/// the time a crafted message can take.
pub const MAX_CHARS: usize = 2000;

/// Output after each text step, for tests and the debug screen.
#[derive(Debug, Clone, PartialEq)]
pub struct Stages {
    pub anonymized: String,
    pub urls: String,
    pub numbers: String,
    pub normalized: String,
    /// How many values the anonymizer masked, per tag.
    pub counts: BTreeMap<String, usize>,
    /// Regex runs that hit the backtracking limit and were skipped (0 for normal text).
    pub limit_hits: usize,
}

/// Run every text step and keep each step's output. Only the first [`MAX_CHARS`] characters
/// of `text` are used.
pub fn stages(text: &str, normalizer: &Normalizer) -> Stages {
    let text = match text.char_indices().nth(MAX_CHARS) {
        Some((end, _)) => &text[..end],
        None => text,
    };
    let mut limit_hits = 0;
    let anon = anonymize(text);
    limit_hits += anon.limit_hits;
    let urls = urls::mask_urls_counted(&anon.text, &mut limit_hits);
    let numbers = urls::mask_numbers_counted(&urls, &mut limit_hits);
    let normalized = normalizer.normalize_counted(&numbers, &mut limit_hits);
    Stages {
        anonymized: anon.text,
        urls,
        numbers,
        normalized,
        counts: anon.counts,
        limit_hits,
    }
}

/// The text a model sees (of the first [`MAX_CHARS`] characters).
pub fn preprocess(text: &str, normalizer: &Normalizer) -> String {
    stages(text, normalizer).normalized
}

/// Raw SMS to TF-IDF features: [`preprocess`], then [`ngrams`].
pub fn features(text: &str, normalizer: &Normalizer) -> Vec<String> {
    ngrams(&preprocess(text, normalizer))
}
