//! Replace sensitive values with typed placeholders ([CARD], [NATID], [PASSWORD], ...).
//! Rules only, no model: names and street addresses are not detected.

use std::collections::BTreeMap;

use crate::regex_util::replace_all;
use crate::rules::{anonymize_rules, Validator};

#[derive(Debug, Clone, PartialEq)]
pub struct Anonymized {
    pub text: String,
    /// How many values were masked, per tag.
    pub counts: BTreeMap<String, usize>,
    /// Rules skipped because the regex hit the backtracking limit (0 for normal text).
    pub limit_hits: usize,
}

/// Steps, in order: translate digits and Arabic letters; format rules (the whole match
/// becomes `[TAG]` unless its validator rejects it); keyword rules (group 1 kept, group 2
/// becomes `[TAG]`); then any remaining run of 8+ digits becomes `[NUMBER]`.
pub fn anonymize(text: &str) -> Anonymized {
    let rules = anonymize_rules();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut limit_hits = 0;

    let mut text: String = text
        .chars()
        .map(|c| *rules.translate.get(&c).unwrap_or(&c))
        .collect();

    for rule in &rules.formats {
        text = replace_all(&rule.re, &text, &mut limit_hits, |caps| {
            let value = caps.get(0)?.as_str();
            let ok = match rule.validator {
                None => true,
                Some(Validator::Luhn) => card_ok(value),
                Some(Validator::NationalId) => national_id_ok(value),
            };
            ok.then(|| {
                *counts.entry(rule.tag.clone()).or_default() += 1;
                format!("[{}]", rule.tag)
            })
        });
    }

    for rule in &rules.keywords {
        text = replace_all(&rule.re, &text, &mut limit_hits, |caps| {
            *counts.entry(rule.tag.clone()).or_default() += 1;
            Some(format!(
                "{}[{}]",
                caps.get(1).map_or("", |m| m.as_str()),
                rule.tag
            ))
        });
    }

    let mut numbers = 0;
    text = replace_all(&rules.long_number, &text, &mut limit_hits, |_| {
        numbers += 1;
        Some("[NUMBER]".to_string())
    });
    if numbers > 0 {
        *counts.entry("NUMBER".to_string()).or_default() += numbers;
    }

    Anonymized {
        text,
        counts,
        limit_hits,
    }
}

/// Iranian national ID: 10 digits, not all equal, and the check digit matches.
pub fn national_id_ok(s: &str) -> bool {
    let d: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    if d.len() != 10 || s.chars().count() != 10 || d.iter().all(|&x| x == d[0]) {
        return false;
    }
    let r = (0..9).map(|i| d[i] * (10 - i as u32)).sum::<u32>() % 11;
    d[9] == if r < 2 { r } else { 11 - r }
}

/// Grouped 4-4-4-4 digits (with spaces or dashes) are always a card; 16 bare digits only if
/// they pass Luhn, as every bank card does (16-digit tracking numbers usually don't).
pub fn card_ok(s: &str) -> bool {
    if !s.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    let total: u32 = s
        .chars()
        .rev()
        .filter_map(|c| c.to_digit(10))
        .enumerate()
        .map(|(i, d)| {
            if i % 2 == 1 {
                if d > 4 {
                    d * 2 - 9
                } else {
                    d * 2
                }
            } else {
                d
            }
        })
        .sum();
    total.is_multiple_of(10)
}
