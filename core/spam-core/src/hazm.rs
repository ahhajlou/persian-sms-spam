//! hazm's `Normalizer` as data (`rules/hazm.json`): character tables and regex rules.
//!
//! Two modes:
//! - `Normalizer::Rules` runs everything that needs no word list.
//! - `Normalizer::Full(lexicons)` adds hazm's word-list steps: joining tokens that form a
//!   known word, reducing repeated letters, and the می‌ separation.
//!
//! The steps and their order are hazm 0.12's `Normalizer.normalize`, as ported in
//! `classic/export_android_assets.py`.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;
use std::{fs, io};

use fancy_regex::{Captures, Regex};
use serde::Deserialize;

use crate::regex_util::{compile, replace_all};
use crate::rules::HAZM_JSON;
use crate::text::is_py_space;

const ZWNJ: char = '\u{200c}';

/// hazm's word lists. About 14 MB, so the core only has them when asked.
#[derive(Debug, Default)]
pub struct Lexicons {
    words: HashSet<String>,
    /// Words with a rank above zero (gate for joining two tokens).
    words_positive: HashSet<String>,
    verbs: HashSet<String>,
}

impl Lexicons {
    /// Each argument is one entry per line, as `make lexicons` writes them.
    pub fn from_texts(words: &str, words_positive: &str, verbs: &str) -> Self {
        let set = |t: &str| {
            t.lines()
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect()
        };
        Lexicons {
            words: set(words),
            words_positive: set(words_positive),
            verbs: set(verbs),
        }
    }

    /// Reads `words.txt`, `words_positive.txt` and `verbs.txt` from `dir`.
    pub fn from_dir(dir: &Path) -> io::Result<Self> {
        let read = |name: &str| fs::read_to_string(dir.join(name));
        Ok(Self::from_texts(
            &read("words.txt")?,
            &read("words_positive.txt")?,
            &read("verbs.txt")?,
        ))
    }
}

#[derive(Debug)]
pub enum Normalizer {
    Rules,
    Full(Lexicons),
}

impl Normalizer {
    /// "rules" or "full": what a model file records next to the pipeline version.
    pub fn mode(&self) -> &'static str {
        match self {
            Normalizer::Rules => "rules",
            Normalizer::Full(_) => "full",
        }
    }

    pub fn normalize(&self, text: &str) -> String {
        self.normalize_counted(text, &mut 0)
    }

    pub(crate) fn normalize_counted(&self, text: &str, limit_hits: &mut usize) -> String {
        let lex = match self {
            Normalizer::Rules => None,
            Normalizer::Full(l) => Some(l),
        };
        normalize(text, lex, limit_hits)
    }
}

// --- rules ------------------------------------------------------------------

#[derive(Deserialize)]
struct Template {
    l: Option<String>,
    g: Option<usize>,
}

#[derive(Deserialize)]
struct RuleFile {
    p: String,
    r: Vec<Template>,
}

#[derive(Deserialize)]
struct HazmFile {
    translation_src: String,
    translation_dst: String,
    numbers_src: String,
    numbers_dst: String,
    suffixes: Vec<String>,
    persian_style: Vec<RuleFile>,
    diacritics: Vec<RuleFile>,
    extra_space: Vec<RuleFile>,
    affix_spacing: Vec<RuleFile>,
    punctuation_spacing: Vec<RuleFile>,
    special_chars: Vec<RuleFile>,
    unicode_replacements: Vec<(String, String)>,
    tokenize_pattern: String,
    repeat_chars_pattern: String,
    repeat_more_than_two_pattern: String,
    seperate_mi_pattern: String,
}

enum Part {
    Literal(String),
    Group(usize),
}

struct Rule {
    re: Regex,
    parts: Vec<Part>,
}

struct Hazm {
    translation: HashMap<char, char>,
    numbers: HashMap<char, char>,
    suffixes: HashSet<String>,
    persian_style: Vec<Rule>,
    diacritics: Vec<Rule>,
    extra_space: Vec<Rule>,
    affix_spacing: Vec<Rule>,
    punctuation_spacing: Vec<Rule>,
    special_chars: Vec<Rule>,
    unicode_replacements: Vec<(String, String)>,
    tokenize: Regex,
    repeat_chars: Regex,
    repeat_more_than_two: Regex,
    seperate_mi: Regex,
}

fn table(src: &str, dst: &str) -> HashMap<char, char> {
    let (src, dst): (Vec<char>, Vec<char>) = (src.chars().collect(), dst.chars().collect());
    assert_eq!(src.len(), dst.len(), "character table lengths differ");
    src.into_iter().zip(dst).collect() // later pairs win, like str.maketrans
}

fn rules(files: Vec<RuleFile>) -> Vec<Rule> {
    files
        .into_iter()
        .map(|f| Rule {
            re: compile(&f.p),
            parts: f
                .r
                .into_iter()
                .map(|t| match (t.l, t.g) {
                    (Some(l), None) => Part::Literal(l),
                    (None, Some(g)) => Part::Group(g),
                    _ => panic!("a template part is either a literal or a group"),
                })
                .collect(),
        })
        .collect()
}

fn hazm() -> &'static Hazm {
    static HAZM: OnceLock<Hazm> = OnceLock::new();
    HAZM.get_or_init(|| {
        let f: HazmFile = serde_json::from_str(HAZM_JSON).expect("hazm.json");
        Hazm {
            translation: table(&f.translation_src, &f.translation_dst),
            numbers: table(&f.numbers_src, &f.numbers_dst),
            suffixes: f.suffixes.into_iter().collect(),
            persian_style: rules(f.persian_style),
            diacritics: rules(f.diacritics),
            extra_space: rules(f.extra_space),
            affix_spacing: rules(f.affix_spacing),
            punctuation_spacing: rules(f.punctuation_spacing),
            special_chars: rules(f.special_chars),
            unicode_replacements: f.unicode_replacements,
            tokenize: compile(&f.tokenize_pattern),
            repeat_chars: compile(&f.repeat_chars_pattern),
            repeat_more_than_two: compile(&f.repeat_more_than_two_pattern),
            seperate_mi: compile(&f.seperate_mi_pattern),
        }
    })
}

// --- steps ------------------------------------------------------------------

fn translate(text: &str, table: &HashMap<char, char>) -> String {
    text.chars().map(|c| *table.get(&c).unwrap_or(&c)).collect()
}

fn expand(parts: &[Part], caps: &Captures) -> String {
    let mut out = String::new();
    for part in parts {
        match part {
            Part::Literal(l) => out.push_str(l),
            Part::Group(g) => out.push_str(caps.get(*g).map_or("", |m| m.as_str())),
        }
    }
    out
}

fn apply(rules: &[Rule], text: String, limit_hits: &mut usize) -> String {
    rules.iter().fold(text, |text, rule| {
        replace_all(&rule.re, &text, limit_hits, |caps| {
            Some(expand(&rule.parts, caps))
        })
    })
}

/// Split into words and punctuation runs ("a،b" -> a ، b).
fn tokenize(line: &str, h: &Hazm, limit_hits: &mut usize) -> Vec<String> {
    let line = line.replace(['\n', '\t'], " ");
    let spaced = replace_all(&h.tokenize, &line, limit_hits, |caps| {
        Some(format!(" {} ", caps.get(1).map_or("", |m| m.as_str())))
    });
    spaced
        .split(' ')
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// Join neighbouring tokens that form a known word or verb (needs the word lists).
fn token_spacing(tokens: Vec<String>, h: &Hazm, lex: Option<&Lexicons>) -> Vec<String> {
    let Some(lex) = lex.filter(|l| !l.words.is_empty()) else {
        return tokens;
    };
    let mut result: Vec<String> = Vec::new();
    for (t, token) in tokens.iter().enumerate() {
        let mut joined = false;
        let mut pair = String::new();
        if let Some(prev) = result.last() {
            pair = format!("{prev}{ZWNJ}{token}");
            if (!lex.verbs.is_empty() && lex.verbs.contains(&pair))
                || lex.words_positive.contains(&pair)
            {
                joined = true;
                if t + 1 < tokens.len()
                    && !lex.verbs.is_empty()
                    && lex.verbs.contains(&format!("{token}_{}", tokens[t + 1]))
                {
                    joined = false;
                }
            } else if h.suffixes.contains(token) && lex.words.contains(prev) {
                joined = true;
            }
        }
        if joined {
            *result.last_mut().expect("joined implies a previous token") = pair;
        } else {
            result.push(token.clone());
        }
    }
    result
}

/// "سلاممم" -> "سلام" when the word list knows which form is right (needs the word lists).
fn decrease_repeated(
    text: String,
    h: &Hazm,
    lex: Option<&Lexicons>,
    limit_hits: &mut usize,
) -> String {
    let Some(lex) = lex.filter(|l| !l.words.is_empty()) else {
        return text;
    };
    let mut spans = Vec::new();
    for m in h.repeat_chars.find_iter(&text) {
        match m {
            Ok(m) => spans.push((m.start(), m.end())),
            Err(_) => {
                *limit_hits += 1;
                return text;
            }
        }
    }
    let mut text = text;
    for (start, end) in spans.into_iter().rev() {
        let word = text[start..end].to_string();
        if lex.words.contains(&word) {
            continue;
        }
        let keep_group = |times: usize| {
            replace_all(&h.repeat_more_than_two, &word, &mut 0, |caps| {
                Some(caps.get(1).map_or("", |m| m.as_str()).repeat(times))
            })
        };
        let (no_repeat, two_repeat) = (keep_group(1), keep_group(2));
        let replacement = if lex.words.contains(&no_repeat) != lex.words.contains(&two_repeat) {
            if lex.words.contains(&no_repeat) {
                no_repeat
            } else {
                two_repeat
            }
        } else {
            two_repeat
        };
        text.replace_range(start..end, &replacement);
    }
    text
}

/// "میخوام" -> "می‌خوام" when the result is a known verb form (needs the word lists).
fn seperate_mi(text: String, h: &Hazm, lex: Option<&Lexicons>, limit_hits: &mut usize) -> String {
    let Some(lex) = lex.filter(|l| !l.verbs.is_empty()) else {
        return text;
    };
    replace_all(&h.seperate_mi, &text, limit_hits, |caps| {
        let word = caps.get(0)?.as_str();
        let candidate = if word.starts_with("نمی") {
            format!("نمی{ZWNJ}{}", word.chars().skip(3).collect::<String>())
        } else if word.starts_with("می") {
            format!("می{ZWNJ}{}", word.chars().skip(2).collect::<String>())
        } else {
            word.to_string()
        };
        Some(if lex.verbs.contains(&candidate) {
            candidate
        } else {
            word.to_string()
        })
    })
}

fn normalize(text: &str, lex: Option<&Lexicons>, limit_hits: &mut usize) -> String {
    let h = hazm();
    let mut text = translate(text, &h.translation);
    text = apply(&h.persian_style, text, limit_hits);
    text = translate(&text, &h.numbers);
    text = apply(&h.diacritics, text, limit_hits);

    text = apply(&h.extra_space, text, limit_hits);
    let lines: Vec<String> = text
        .split('\n')
        .map(|line| {
            if line.chars().all(is_py_space) {
                return line.to_string(); // blank lines are kept as they are
            }
            token_spacing(tokenize(line, h, limit_hits), h, lex).join(" ")
        })
        .collect();
    text = lines.join("\n");
    text = apply(&h.affix_spacing, text, limit_hits);
    text = apply(&h.punctuation_spacing, text, limit_hits);

    for (old, new) in &h.unicode_replacements {
        text = text.replace(old.as_str(), new);
    }
    text = apply(&h.special_chars, text, limit_hits);

    text = decrease_repeated(text, h, lex, limit_hits);
    seperate_mi(text, h, lex, limit_hits)
}
