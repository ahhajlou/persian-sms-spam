//! The pipeline's output on the battery must equal the golden file of the current
//! PIPELINE_VERSION. Changing the output without bumping the version cannot pass.

mod common;

use common::*;
use spam_core::{golden, Normalizer, PIPELINE_VERSION};

fn check(normalizer: &Normalizer) {
    let battery: Vec<String> = serde_json::from_value(case_file("battery.json")).unwrap();
    let path = manifest_dir()
        .join("tests/golden")
        .join(golden::file_name(normalizer));
    let expected: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "no golden file {path:?}. If you changed the pipeline output, bump PIPELINE_VERSION, then run \
             `cargo run -p spam-core --example golden -- {}`",
            normalizer.mode()
        )
    }))
    .unwrap();
    let actual = golden::snapshot(&battery, normalizer);

    let (a, e) = (
        actual["cases"].as_array().unwrap(),
        expected["cases"].as_array().unwrap(),
    );
    assert_eq!(a.len(), e.len(), "battery size changed");
    for (i, (a, e)) in a.iter().zip(e).enumerate() {
        for stage in [
            "anonymized",
            "urls",
            "numbers",
            "normalized",
            "ngrams",
            "limit_hits",
        ] {
            assert_eq!(
                a[stage], e[stage],
                "battery[{i}] differs at stage {stage:?} from {path:?}: the output changed, so bump PIPELINE_VERSION \
                 (now {PIPELINE_VERSION}) and write a new golden file"
            );
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn golden_rules() {
    check(&Normalizer::Rules);
}

#[test]
fn golden_full() {
    if let Some(n) = full_normalizer() {
        check(&n);
    }
}

#[test]
fn battery_hits_no_limits() {
    let battery: Vec<String> = serde_json::from_value(case_file("battery.json")).unwrap();
    for text in &battery {
        assert_eq!(spam_core::stages(text, &Normalizer::Rules).limit_hits, 0);
    }
}
