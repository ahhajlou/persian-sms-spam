//! Checks on the rules files themselves.

mod common;

use common::manifest_dir;
use serde_json::Value;

/// Every regex in the rules files: (file, where, pattern).
fn all_patterns() -> Vec<(String, String, String)> {
    let dir = manifest_dir().join("rules");
    let mut out = Vec::new();

    for file in ["anonymize.toml", "urls.toml"] {
        let doc: toml::Table = std::fs::read_to_string(dir.join(file))
            .unwrap()
            .parse()
            .unwrap();
        fn walk(file: &str, path: &str, v: &toml::Value, out: &mut Vec<(String, String, String)>) {
            match v {
                toml::Value::Table(t) => {
                    for (k, v) in t {
                        if k == "pattern" {
                            out.push((
                                file.to_string(),
                                path.to_string(),
                                v.as_str().unwrap().to_string(),
                            ));
                        } else {
                            walk(file, &format!("{path}.{k}"), v, out);
                        }
                    }
                }
                toml::Value::Array(a) => {
                    for (i, v) in a.iter().enumerate() {
                        walk(file, &format!("{path}[{i}]"), v, out);
                    }
                }
                _ => {}
            }
        }
        walk(file, "", &toml::Value::Table(doc), &mut out);
    }

    let hazm: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("hazm.json")).unwrap()).unwrap();
    for (key, v) in hazm.as_object().unwrap() {
        match v {
            Value::String(s) if key.ends_with("_pattern") => {
                out.push(("hazm.json".into(), key.clone(), s.clone()))
            }
            Value::Array(rules) if rules.first().is_some_and(|r| r.get("p").is_some()) => {
                for (i, r) in rules.iter().enumerate() {
                    out.push((
                        "hazm.json".into(),
                        format!("{key}[{i}]"),
                        r["p"].as_str().unwrap().to_string(),
                    ));
                }
            }
            _ => {}
        }
    }
    out
}

#[test]
fn the_rules_files_have_patterns() {
    let n = all_patterns().len();
    assert!(
        n >= 40,
        "found only {n} patterns; did the file layout change?"
    );
}

/// The shorthand classes (and negations) used in `pattern`, e.g. ["\\d"]. An escaped
/// backslash followed by a letter (`\\\\d`) is not one.
fn shorthand(pattern: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(e) = chars.next() {
                if "dDwWsSbB".contains(e) {
                    found.push(format!("\\{e}"));
                }
            }
        }
    }
    found
}

#[test]
fn shorthand_scanner_finds_them() {
    assert_eq!(shorthand(r"[\d.]\bx\s"), ["\\d", "\\b", "\\s"]);
    assert!(shorthand(r"\\d \p{Nd} \x{20} \. \n").is_empty());
}

/// \d \w \s \b mean different things in Python, Java, ICU and Rust, so rules spell the
/// classes out.
#[test]
fn no_shorthand_classes() {
    let mut bad = Vec::new();
    for (file, place, pattern) in all_patterns() {
        for s in shorthand(&pattern) {
            bad.push(format!("{file} {place}: {s}"));
        }
    }
    assert!(bad.is_empty(), "shorthand classes in rules: {bad:#?}");
}

#[test]
fn every_pattern_compiles() {
    // fancy-regex is what the core uses; compile each rule on its own to name the broken one.
    for (file, place, pattern) in all_patterns() {
        fancy_regex::Regex::new(&pattern).unwrap_or_else(|e| panic!("{file} {place}: {e}"));
    }
}
