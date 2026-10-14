# Rust core

The text preprocessing that training and the NoSpam app share, so a message becomes the same
model input in both. Design and decisions: [docs/preprocessing-core-plan.md](../docs/preprocessing-core-plan.md).

```
raw SMS → anonymize → mask URLs → mask numbers → normalize → lowercase → char n-grams (2-4)
```

- `spam-core/`: the library. Every regex rule is data in `rules/` (`anonymize.toml`, `urls.toml`,
  `hazm.json` from hazm, MIT, notice in `HAZM-LICENSE.txt`) and uses explicit character classes.
  `PIPELINE_VERSION` in `src/lib.rs` is the output contract; bump it on any output change.
  Only the first 2,000 characters of a message are used.
- `spam-core-py/`: Python binding (PyO3, built by maturin), module `spam_core`. `uv` rebuilds it
  when the core changes.
- `spam-core-jni/`: JNI binding for the Android app (`libspam_core_jni.so`).
- `tools/`: `gen_hazm_data.py` (`make lexicons`) regenerates `hazm.json` and the optional word
  lists; `core_diff.py` (`make core-diff`) compares the core with the old Python pipeline;
  `rx.py` rewrites Python regexes for Rust.
- `data/lexicons/`: hazm's word lists for the `full` normalizer (generated, gitignored).

```bash
make core-test      # cargo test: shared case files for every stage, golden files, hostile input, JNI
make lexicons       # word lists for the `full` normalizer tests
cd core && cargo test -p spam-core
```

The test case files in `spam-core/tests/cases/` are shared with pytest. `tests/golden/` holds the
whole pipeline's output per version; if `cargo test` fails there, the output changed: bump
`PIPELINE_VERSION`, then `cargo run -p spam-core --example golden -- rules` (and `full`).
The JNI tests start a JVM and need a JDK (JAVA_HOME or `java` on PATH); without one they print
`SKIPPED`, or fail with `SPAM_CORE_REQUIRE_JVM=1`.
