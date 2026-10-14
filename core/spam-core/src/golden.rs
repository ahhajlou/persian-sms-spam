//! Snapshots of the whole pipeline on a fixed battery, for the golden files.
//! Only the tests and `examples/golden.rs` use this.

use serde_json::{json, Value};

use crate::{ngrams, stages, Normalizer, PIPELINE_VERSION};

/// Every stage's output for each input, tagged with the pipeline version and normalizer mode.
pub fn snapshot(inputs: &[String], normalizer: &Normalizer) -> Value {
    let cases: Vec<Value> = inputs
        .iter()
        .map(|text| {
            let s = stages(text, normalizer);
            json!({
                "anonymized": s.anonymized,
                "urls": s.urls,
                "numbers": s.numbers,
                "normalized": s.normalized,
                "ngrams": ngrams(&s.normalized),
                "limit_hits": s.limit_hits,
            })
        })
        .collect();
    json!({
        "pipeline_version": PIPELINE_VERSION,
        "normalizer": normalizer.mode(),
        "cases": cases,
    })
}

/// File name of the golden file for the current version, e.g. `v5-rules.json`.
pub fn file_name(normalizer: &Normalizer) -> String {
    format!("v{PIPELINE_VERSION}-{}.json", normalizer.mode())
}
