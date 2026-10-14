#![allow(dead_code)]
use std::path::{Path, PathBuf};

use serde_json::Value;
use spam_core::{Lexicons, Normalizer};

pub fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn case_file(name: &str) -> Value {
    let path = manifest_dir().join("tests/cases").join(name);
    serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}")),
    )
    .unwrap_or_else(|e| panic!("{path:?}: {e}"))
}

pub fn pairs(v: &Value) -> Vec<(&str, &str)> {
    v.as_array()
        .expect("array of pairs")
        .iter()
        .map(|p| (p[0].as_str().unwrap(), p[1].as_str().unwrap()))
        .collect()
}

pub fn strings(v: &Value) -> Vec<&str> {
    v.as_array()
        .expect("array of strings")
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect()
}

/// `make lexicons` writes the word lists here (gitignored); `SPAM_CORE_LEXICONS` overrides.
pub fn lexicon_dir() -> PathBuf {
    std::env::var_os("SPAM_CORE_LEXICONS")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir().join("../data/lexicons"))
}

/// The `full` normalizer, or None (with a note) when the word lists were not generated.
/// Set `SPAM_CORE_REQUIRE_FULL=1` to fail instead of skipping.
pub fn full_normalizer() -> Option<Normalizer> {
    let dir: &Path = &lexicon_dir();
    match Lexicons::from_dir(dir) {
        Ok(l) => Some(Normalizer::Full(l)),
        Err(e) => {
            assert!(
                std::env::var_os("SPAM_CORE_REQUIRE_FULL").is_none(),
                "full mode needs the word lists ({dir:?}: {e}); run `make lexicons`"
            );
            eprintln!("SKIPPED full mode: no word lists in {dir:?} ({e}); run `make lexicons`");
            None
        }
    }
}
