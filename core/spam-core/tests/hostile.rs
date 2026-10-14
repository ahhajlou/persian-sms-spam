//! A crafted SMS must not hang the phone: a regex that backtracks too long is skipped.

use std::time::{Duration, Instant};

use spam_core::*;

/// Dots and letters with no '@' make the email rule retry from every start.
fn crafted() -> String {
    "a.".repeat(20_000)
}

#[test]
fn backtracking_is_cut_off_and_counted() {
    let text = crafted();
    let start = Instant::now();
    let a = anonymize(&text);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "took {:?}",
        start.elapsed()
    );
    assert!(
        a.limit_hits >= 1,
        "expected the email rule to hit the limit"
    );
    // The skipped rule leaves the text as it was; later rules still ran.
    assert!(a.text.starts_with("a.a.a."));
}

#[test]
fn only_the_first_2000_characters_are_used() {
    let n = Normalizer::Rules;
    // Characters, not bytes: Persian letters are two bytes each.
    let head = "سلام ".repeat(400); // exactly 2000 characters
    assert_eq!(head.chars().count(), MAX_CHARS);
    let long = format!("{head}someone@example.com کد 123456 {}", crafted());
    assert_eq!(preprocess(&long, &n), preprocess(&head, &n));
    assert_eq!(stages(&long, &n), stages(&head, &n));
    assert_eq!(features(&long, &n), features(&head, &n));
    // Nothing past the cut is masked or counted.
    assert!(stages(&long, &n).counts.is_empty());
    // The cut falls inside "x@y.ir": only the "x" before it survives.
    let at_limit = format!("{}{}", "ا".repeat(MAX_CHARS - 1), "x@y.ir");
    assert_eq!(
        stages(&at_limit, &n).normalized,
        stages(&"ا".repeat(MAX_CHARS - 1), &n).normalized.clone() + "x"
    );
}

#[test]
fn a_long_crafted_message_finishes_quickly() {
    let start = Instant::now();
    let s = stages(&crafted(), &Normalizer::Rules);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "took {:?}",
        start.elapsed()
    );
    assert!(!s.normalized.is_empty());
}

#[test]
fn normal_messages_never_hit_the_limit() {
    for text in [
        "کد تایید شما: 123456 https://x.ir/a?b=1",
        "برداشت از حساب 0303438303006 مبلغ 50,000 مانده 1,000,000",
        &"سلام ".repeat(300),
        &"1234 ".repeat(300),
    ] {
        assert_eq!(
            stages(text, &Normalizer::Rules).limit_hits,
            0,
            "{:?}",
            &text[..text.len().min(30)]
        );
    }
}
