//! The rules files, embedded at build time and compiled once.

use std::collections::HashMap;
use std::sync::OnceLock;

use fancy_regex::Regex;
use serde::Deserialize;

use crate::regex_util::compile;

pub const ANONYMIZE_TOML: &str = include_str!("../rules/anonymize.toml");
pub const URLS_TOML: &str = include_str!("../rules/urls.toml");
pub const HAZM_JSON: &str = include_str!("../rules/hazm.json");

// --- anonymize.toml ---------------------------------------------------------

#[derive(Deserialize)]
struct AnonymizeFile {
    translate_src: String,
    translate_dst: String,
    format: Vec<FormatFile>,
    keyword: Vec<KeywordFile>,
    long_number: PatternFile,
}

#[derive(Deserialize)]
struct FormatFile {
    tag: String,
    pattern: String,
    validator: Option<String>,
}

#[derive(Deserialize)]
struct KeywordFile {
    tag: String,
    pattern: String,
}

#[derive(Deserialize)]
struct PatternFile {
    pattern: String,
}

#[derive(Clone, Copy)]
pub(crate) enum Validator {
    Luhn,
    NationalId,
}

pub(crate) struct FormatRule {
    pub tag: String,
    pub re: Regex,
    pub validator: Option<Validator>,
}

pub(crate) struct KeywordRule {
    pub tag: String,
    pub re: Regex,
}

pub(crate) struct AnonymizeRules {
    pub translate: HashMap<char, char>,
    pub formats: Vec<FormatRule>,
    pub keywords: Vec<KeywordRule>,
    pub long_number: Regex,
}

pub(crate) fn anonymize_rules() -> &'static AnonymizeRules {
    static RULES: OnceLock<AnonymizeRules> = OnceLock::new();
    RULES.get_or_init(|| {
        let f: AnonymizeFile = toml::from_str(ANONYMIZE_TOML).expect("anonymize.toml");
        let (src, dst): (Vec<char>, Vec<char>) = (
            f.translate_src.chars().collect(),
            f.translate_dst.chars().collect(),
        );
        assert_eq!(
            src.len(),
            dst.len(),
            "translate_src and translate_dst differ in length"
        );
        AnonymizeRules {
            // Later pairs win, like str.maketrans.
            translate: src.into_iter().zip(dst).collect(),
            formats: f
                .format
                .into_iter()
                .map(|r| FormatRule {
                    re: compile(&r.pattern),
                    validator: r.validator.as_deref().map(|v| match v {
                        "luhn" => Validator::Luhn,
                        "natid" => Validator::NationalId,
                        other => panic!("unknown validator {other:?}"),
                    }),
                    tag: r.tag,
                })
                .collect(),
            keywords: f
                .keyword
                .into_iter()
                .map(|r| KeywordRule {
                    re: compile(&r.pattern),
                    tag: r.tag,
                })
                .collect(),
            long_number: compile(&f.long_number.pattern),
        }
    })
}

// --- urls.toml --------------------------------------------------------------

#[derive(Deserialize)]
struct MaskFile {
    pattern: String,
    replacement: String,
}

#[derive(Deserialize)]
struct UrlsFile {
    url: MaskFile,
    number: MaskFile,
}

pub(crate) struct Mask {
    pub re: Regex,
    pub replacement: String,
}

pub(crate) struct UrlRules {
    pub url: Mask,
    pub number: Mask,
}

pub(crate) fn url_rules() -> &'static UrlRules {
    static RULES: OnceLock<UrlRules> = OnceLock::new();
    RULES.get_or_init(|| {
        let f: UrlsFile = toml::from_str(URLS_TOML).expect("urls.toml");
        let mask = |m: MaskFile| Mask {
            re: compile(&m.pattern),
            replacement: m.replacement,
        };
        UrlRules {
            url: mask(f.url),
            number: mask(f.number),
        }
    })
}
