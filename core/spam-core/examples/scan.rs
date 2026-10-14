//! Run the pipeline over JSONL files and print counts and timing only, never message text:
//!
//!     cargo run --release -p spam-core --example scan -- ../../data/sms_labeled.jsonl

use std::time::Instant;

use spam_core::{stages, Normalizer};

fn main() {
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).expect("readable file");
        let (mut rows, mut hits, mut empty_after) = (0usize, 0usize, 0usize);
        let mut slowest = std::time::Duration::ZERO;
        let start = Instant::now();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let row: serde_json::Value = serde_json::from_str(line).expect("JSON row");
            let Some(sms) = row["text"].as_str() else {
                continue;
            };
            let t = Instant::now();
            let s = stages(sms, &Normalizer::Rules);
            slowest = slowest.max(t.elapsed());
            rows += 1;
            hits += s.limit_hits;
            empty_after += s.normalized.trim().is_empty() as usize;
        }
        let total = start.elapsed();
        println!(
            "{path}: {rows} rows, {hits} backtrack-limit hits, {empty_after} empty results, {:.0} µs/row, slowest {:.1} ms",
            total.as_secs_f64() * 1e6 / rows.max(1) as f64,
            slowest.as_secs_f64() * 1e3
        );
    }
}
