//! The shared case files in tests/cases/, one per stage. pytest reads the same files.

mod common;

use common::*;
use spam_core::*;

#[test]
fn anonymize_masked() {
    let cases = case_file("anonymize.json");
    for (input, expected) in pairs(&cases["masked"]) {
        assert_eq!(anonymize(input).text, expected, "input: {input:?}");
    }
}

#[test]
fn anonymize_unchanged() {
    let cases = case_file("anonymize.json");
    for input in strings(&cases["unchanged"]) {
        assert_eq!(anonymize(input).text, input);
    }
}

#[test]
fn anonymize_is_idempotent() {
    let cases = case_file("anonymize.json");
    for (input, _) in pairs(&cases["masked"]) {
        let once = anonymize(input).text;
        assert_eq!(anonymize(&once).text, once, "input: {input:?}");
    }
}

#[test]
fn anonymize_counts_tags() {
    let a = anonymize("کارت 6037-9975-1234-5678 و 603799*1234 مانده:50,051,391");
    assert_eq!(a.counts.get("CARD"), Some(&2));
    assert_eq!(a.counts.get("AMOUNT"), Some(&1));
    assert_eq!(a.limit_hits, 0);
    // A rejected match is not counted: this is not a valid national ID.
    assert!(!anonymize("شماره 0023191588").counts.contains_key("NATID"));
}

#[test]
fn validators() {
    let cases = case_file("anonymize.json");
    for v in strings(&cases["national_id"]["valid"]) {
        assert!(national_id_ok(v), "{v}");
    }
    for v in strings(&cases["national_id"]["invalid"]) {
        assert!(!national_id_ok(v), "{v}");
    }
    for v in strings(&cases["card"]["valid"]) {
        assert!(card_ok(v), "{v}");
    }
    for v in strings(&cases["card"]["invalid"]) {
        assert!(!card_ok(v), "{v}");
    }
}

#[test]
fn urls_masked() {
    let cases = case_file("urls.json");
    for (input, expected) in pairs(&cases["masked"]) {
        assert_eq!(mask_urls(input), expected, "input: {input:?}");
    }
}

#[test]
fn urls_unchanged() {
    let cases = case_file("urls.json");
    for input in strings(&cases["unchanged"]) {
        assert_eq!(mask_urls(input), input);
    }
}

#[test]
fn numbers() {
    for (input, expected) in pairs(&case_file("numbers.json")) {
        assert_eq!(mask_numbers(input), expected, "input: {input:?}");
    }
}

#[test]
fn normalize_rules() {
    let cases = case_file("normalize.json");
    let n = Normalizer::Rules;
    let failures: Vec<_> = pairs(&cases["rules"])
        .into_iter()
        .filter(|(i, e)| n.normalize(i) != *e)
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} differ, first: {:?}",
        failures.len(),
        cases["rules"].as_array().unwrap().len(),
        failures.first().map(|(i, e)| (i, e, n.normalize(i)))
    );
}

#[test]
fn normalize_full() {
    let Some(n) = full_normalizer() else { return };
    let cases = case_file("normalize.json");
    let failures: Vec<_> = pairs(&cases["full"])
        .into_iter()
        .filter(|(i, e)| n.normalize(i) != *e)
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} differ, first: {:?}",
        failures.len(),
        cases["full"].as_array().unwrap().len(),
        failures.first().map(|(i, e)| (i, e, n.normalize(i)))
    );
}

#[test]
fn ngrams_match_sklearn() {
    for (input, expected) in case_file("ngrams.json")
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_str().unwrap(), strings(&p[1])))
    {
        assert_eq!(ngrams(input), expected, "input: {input:?}");
    }
}

#[test]
fn ngrams_split_on_thin_space() {
    // Java's (?u) split only on ASCII whitespace; sklearn (and so the core) also splits here.
    assert_eq!(ngrams("a\u{2009}b"), ngrams("a b"));
    assert_eq!(ngrams("a\u{202f}b"), ngrams("a b"));
}

#[test]
fn stages_chain() {
    let s = stages("کد: 123456 سایت x.ir", &Normalizer::Rules);
    assert_eq!(s.anonymized, "کد: [CODE] سایت x.ir");
    assert!(s.urls.contains("URLTOKEN") && !s.urls.contains("x.ir"));
    assert_eq!(
        preprocess("کد: 123456 سایت x.ir", &Normalizer::Rules),
        s.normalized
    );
    assert_eq!(
        features("کد: 123456 سایت x.ir", &Normalizer::Rules),
        ngrams(&s.normalized)
    );
}
