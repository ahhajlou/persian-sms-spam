//! Write the golden file for the current PIPELINE_VERSION:
//!
//!     cargo run -p spam-core --example golden -- rules
//!     cargo run -p spam-core --example golden -- full      (needs `make lexicons`)
//!
//! It refuses to overwrite a version's file. If `cargo test` fails on a golden file, the
//! pipeline output changed: bump PIPELINE_VERSION in src/lib.rs, then run this again.

use std::path::PathBuf;

use spam_core::{golden, Lexicons, Normalizer};

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let normalizer = match mode.as_str() {
        "rules" => Normalizer::Rules,
        "full" => Normalizer::Full(
            Lexicons::from_dir(&root.join("../data/lexicons"))
                .expect("word lists missing; run `make lexicons`"),
        ),
        _ => {
            eprintln!("usage: golden rules|full");
            std::process::exit(2);
        }
    };

    let battery: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(root.join("tests/cases/battery.json")).unwrap(),
    )
    .unwrap();
    let path = root
        .join("tests/golden")
        .join(golden::file_name(&normalizer));
    if path.exists() {
        eprintln!(
            "{} exists. Output changed? Bump PIPELINE_VERSION first.",
            path.display()
        );
        std::process::exit(1);
    }
    let snapshot = golden::snapshot(&battery, &normalizer);
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&snapshot).unwrap() + "\n",
    )
    .unwrap();
    println!("wrote {} ({} inputs)", path.display(), battery.len());
}
