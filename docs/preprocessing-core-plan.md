# Plan: one Rust preprocessing core for training and the NoSpam app

Status: Phase 3a and Phase 4 (this repo) done; next: Phase 3b in NoSpam.

## How to work through this plan

- One phase at a time, in order. Stop at the phase's "Done when", report the results, and wait
  for the user before starting the next phase.
- Don't reopen the decisions below; they were made with the user. Ask if something seems wrong.
- After the user accepts a phase, commit it. NoSpam changes happen only in Phase 3b.
- `data/` is private: scripts that read it print counts and metrics, never message text.
- "Implementation notes" near the end lists the details that are easy to get wrong.

## Goal

The phone must turn an SMS into exactly the same model input as training does.
Today the pipeline exists twice (Python here, Kotlin in NoSpam `core:ml`) and runs on
three regex engines (Python `re`, desktop Java, Android's ICU). The Kotlin side is still
at pipeline v1: no anonymizer, old URL rule.

After this plan, the text processing exists once, in Rust, and both projects call it:

```
raw SMS ─► [ Rust core: shared by both projects ]                  ─► [ separate in each project ]
           anonymize → mask URLs → mask numbers → normalize            TF-IDF weights + SVM score
           → lowercase → char n-grams (2-4, char_wb)                   Python: sklearn
                                                                       Android: Kotlin
```

- The model math (counts, idf, sublinear tf, L2 norm, dot product) stays separate:
  sklearn in training, the existing Kotlin code on the phone. It is plain arithmetic, and the
  parity test checks the scores (1e-6).
- fastText and ParsBERT keep using the preprocessed text (`preprocess()`), now from the core.
- ONNX is not part of this plan. Add it when a neural model goes on the phone.

## Decisions already made

| Topic | Decision | Why |
|---|---|---|
| Single source | Rust core, Python and Android bindings | One implementation and one regex engine on both sides |
| Scope of the core | All text steps, through char n-grams | The n-gram step has its own Unicode rules (see U+2009 below) |
| Scoring | Stays separate (sklearn / Kotlin) | Plain arithmetic, already verified by the score parity test |
| hazm | Keep its regex rules and character tables as data; word lists optional (see below) | Experiment below |
| Kotlin anonymizer port | Not done | It would be thrown away |
| Where the core lives | `core/` in this repo | Every rule change means retraining, which happens here |
| How NoSpam gets the native library | Built from source in NoSpam's Gradle build, core pulled in as a git submodule | Both projects fully open source: anyone can check the binary matches the source, and F-Droid accepts it (it rejects prebuilt binaries) |
| Android binding | Hand-written JNI | Three or four functions; UniFFI would add JNA to the APK |

### Prerequisites for building from source

- persian-sms-spam must be public (it has no git remote yet). Its history was checked:
  no data or model files were ever committed; the deleted notebook has no saved outputs.
- License: GPL-3.0, like NoSpam (chosen 2026-10-14). `rules/hazm.json` comes from hazm (MIT):
  keep its copyright notice (`rules/HAZM-LICENSE.txt`).

## What is public, what stays private

| What | Where | Public? |
|---|---|---|
| SMS dataset (`data/`) | this repo, gitignored | No, never |
| Trained models (`models/`, `*.joblib`, ...) | this repo, gitignored | No, never |
| Training code and the Rust core | this repo | Yes |
| TF-IDF weights (`spam_model.json`) and the test fixture | committed in NoSpam | Yes: the app ships them |

- **Anyone can build NoSpam** from its repo and the submodule. No dataset needed.
- **Nobody else can regenerate these weights**: that needs the private dataset.
- **Anyone can train their own model** with their own SMS export and the same workflow, then
  export it into their NoSpam checkout. It works as long as it was trained with the same
  `PIPELINE_VERSION` and normalizer as the core NoSpam builds.
- **Licenses:** publishing weights without the dataset breaks no license. GPL-3.0 covers the
  source code, and the dataset is personal data under no license. Good practice (and what
  OSI's Open Source AI Definition asks for) is a description of the data in the README instead:
  where it comes from, size, how it was labeled, why it is not published.
- **Weights privacy:** the weights contain short text fragments (2-4 character n-grams) from the
  messages, and the anonymizer does not mask names. Not changed for now; see "Future: min_df"
  at the end.

## Workflow after the migration

**persian-sms-spam**
1. Export SMS from a phone.
2. Prepare the data: `anonymize` → `model-label` → check by hand (`make label`) → `merge`.
3. `make test-tfidf`: split, train and score the model the app uses (`make test-all` for all
   three). Every message goes through the Rust core first;
   `uv` rebuilds the core whenever its code changed.
4. `make export-android`: writes `spam_model.json` and the fixture into NoSpam. It refuses if
   `core/` has uncommitted changes, so the commit it records exists.
5. Push this repo.

**NoSpam**
1. Move the submodule to the commit the model was exported from.
2. Build: Gradle compiles the Rust core as part of the normal build.
3. Run the tests: the fixture checks the phone matches training.
4. Commit the submodule move and the new model together.

### Evidence behind the hazm decision

TF-IDF, stratified 5-fold CV repeated 20 times, 8,747 unique labeled messages
(`sms_labeled.jsonl` plus the hand-checked Galaxy A8 and A26 labels).

| Normalize step | Errors | vs full hazm |
|---|---|---|
| full hazm (today) | 287.9 | — |
| hazm rules + char tables, no word lists | 290.1 | +2.3, not significant (p = 0.36) |
| char table only | 297.0 | +9.1, probably real (p = 0.10) |
| none | 297.9 | +10.0, real (p = 0.03) |

The 14 MB of word lists (346k verb forms) add nothing measurable. The regex rules do.

### The U+2009 bug (why engines matter)

`(?u)` in `HazmNormalizer.kt` / `TfidfPreprocessor.kt` is Java's UNICODE_CASE flag, not
Unicode character classes (that is `(?U)`). So desktop Kotlin splits n-grams only on ASCII
whitespace, while sklearn splits on U+2009 and U+202F, which survive the pipeline. No
fixture input and no labeled message contains them, so the parity test passes anyway.

## The normalizer option

The core has two normalizer modes:

| Mode | What runs | Data it needs | APK cost |
|---|---|---|---|
| `rules` (default) | hazm char maps, style, diacritics, spacing, punctuation, affix and special-char rules | ~6 KB, compiled into the core | none |
| `full` | everything in `rules` + token spacing, repeated-letter reduction, می‌ separation | hazm word lists, ~14 MB | ~14 MB assets |

- **Rust:** `enum Normalizer { Rules, Full(Lexicons) }`. `Full` can only be built by passing
  the word lists in, so the core never carries them unless asked.
- **Setting:** `NORMALIZER = "rules"` in `common/src/spam_common/pipeline.py`. That file is in
  the Makefile's `$(PIPELINE)`, so changing the mode retrains every model and rebuilds the split.
- **Model file:** records `"pipeline": {"version": N, "normalizer": "rules"}`. The app refuses a
  model whose pipeline differs from its core, and a `full` model when the word lists are not
  bundled.
- **Word lists for `full`:** generated from the installed hazm by `make lexicons` into
  `core/data/lexicons/` (gitignored, 14 MB). hazm stays only as an optional tool dependency for
  this. The pipeline itself never imports hazm.
- **Both modes are tested:** the golden fixture has expected output for each mode.

## Repository layout

New Cargo workspace inside this repo:

```
core/
  Cargo.toml                workspace
  spam-core/                pure Rust: stages, validators, rules, n-grams, PIPELINE_VERSION
    rules/anonymize.toml    anonymizer format + keyword rules
    rules/urls.toml         URL and number masks
    rules/hazm.json         hazm 0.12 rules + char tables (MIT, attribution kept)
    tests/                  golden fixture, shared anonymizer cases
  spam-core-py/             PyO3 binding, built by maturin, Python module `spam_core`
  spam-core-jni/            JNI binding (cdylib) for Android
  data/lexicons/            generated, gitignored (full mode only)
```

- `common` depends on `spam-core-py` as an editable path dependency, with
  `tool.uv.cache-keys` set to the `.rs`, `.toml` and rules files, so `uv run` rebuilds the
  extension after an edit.
- `spam_common.pipeline` stays the Python entry point (code placement rule unchanged). It
  becomes a thin wrapper over `spam_core`.
- `$(PIPELINE)` in the Makefile becomes the core's sources and rules files plus `pipeline.py`.
- `core/rust-toolchain.toml` pins the Rust version and the Android targets, and `Cargo.lock` is committed, so the
  library builds the same everywhere.
- NoSpam adds this repo as a git submodule pinned to a commit; that commit is the core version
  the app uses. NoSpam's Gradle build compiles `spam-core-jni` from it. `make export-android`
  copies only the model and the fixture, never binaries.

## Phases

### Phase 0: data (before any retraining)

1. Merge the two hand-checked Galaxy files, so v5 trains on ~8,700 messages instead of 874:
   `make merge IN=data/Galaxy_A8/sms_export_20260908_233032_anon_model_fixed.jsonl`, then
   `make merge IN=data/Galaxy_A26/sms_export_20261014_134847_anon_model.jsonl`.
   Use A8's `_fixed` file, not `_anon_model.jsonl`. The labels equal the model's verdict on most
   rows, but the user checked every row by hand.
2. `make split`, `make test-all`: v4 baseline numbers on the bigger data.
3. Move the anonymizer and URL test cases from `common/tests/*.py` into data files
   (`core/spam-core/tests/cases/*.json`) that pytest, the fixture export and `cargo test` all read.

Done when: v4 results on the merged data are in `models/results/`; pytest reads the case files.

### Phase 1: the core in Rust, with Python as the reference

1. Stages: anonymize (rules from `anonymize.toml`, Luhn and national-ID validators in code),
   mask URLs, mask numbers, normalize (`rules` / `full`), lowercase + char_wb n-grams.
2. Regex: `fancy-regex` (lookaround, backtracking; delegates simple parts to `regex`). All 16
   anonymizer/URL patterns use lookaround, so a lookaround-free rewrite would turn rules
   into code. Input is SMS-sized; Python's backtracking engine already does ~177 µs/message.
3. Set fancy-regex's backtrack limit and define one behavior when it is hit (skip that rule,
   count it). A hostile SMS reaches this code. Test it with a crafted input.
   Also, `preprocess()` keeps only the first 2,000 characters of a message (part of v5; the
   longest real message has 941, so real outputs don't change). It bounds the hostile case
   to about 1 s on a desktop, and the app waits at most 3 s on a slower phone.
4. Rules use explicit character classes only, never `\d \w \s \b`: Python, Java, ICU and
   Rust disagree on them (Rust's `\w` even includes ZWNJ and diacritics). hazm's `\b`/`\B` rules
   are rewritten to explicit `[\p{L}\p{N}_]` lookarounds, as `pythonBoundaries()` does in Kotlin.
   A test rejects the shorthand classes in every rules file.
5. Expose: `preprocess(text)`, `stages(text)`, `anonymize(text) -> (text, tag counts)`,
   `ngrams(text)`, `PIPELINE_VERSION`, normalizer mode.

Done when: `cargo test` passes the shared case files for every stage.

### Phase 2: Python switches to the core

1. Differential test (`make core-diff`): Python (today's code, hazm) vs Rust per stage on the
   fixture battery, a fuzz generator, and all local data. Prints counts and the first
   differing stage only, never message text.
   - `full` must match real hazm; `rules` must match the export script's port with empty word lists.
   - Each difference is a Rust bug (fix it) or an intended change (write it down).
2. Flip: `spam_common.pipeline` and `spam_common.anonymize` call `spam_core`; the CLI tools stay.
   TF-IDF uses `TfidfVectorizer(analyzer=spam_common.pipeline.ngrams)`.
3. Delete the Python regex rules and the hazm port in `export_android_assets.py`.
4. `PIPELINE_VERSION = 5` (now a Rust constant; Python re-exports it), `NORMALIZER = "rules"`.
   Retrain and score TF-IDF with `make test-tfidf` and compare with its Phase 0 numbers.
   `evaluate()` also records `NORMALIZER` next to `pipeline_version` in the results.
5. fastText and ParsBERT are optional: they are not on the phone. Their code stays and keeps
   working through `preprocess()`; run `make test-fasttext` / `make test-parsbert` only when
   wanted. `make compare` marks results from an older pipeline with `*`.

Done when: no unexplained differences, and v5 TF-IDF is at least as good as v4 TF-IDF.

Phase 0 baseline (v4, TF-IDF, 1,756 test messages): accuracy 0.977, macro F1 0.976,
spam F1 0.969, 16 missed spam, 24 blocked ham.

Phase 2 results (2026-10-14):

- `make core-diff` compares the Rust core with the Python pipeline as of commit `45fe347`
  (read from git, so it still runs after the Python rules were deleted): 512 battery inputs,
  150,000 fuzz inputs (seed 7) and all 31,235 distinct local messages. No unexpected differences.
  `full` equals real hazm; `rules` equals the export script's port with empty word lists; n-grams
  equal sklearn's `char_wb`. The 2,000-character cut changed none of the local messages.
- Intended differences (a difference counts as intended only if it disappears when the
  characters below are replaced, so nothing else hides behind them). No local message is affected:
  - Digits of other scripts (fullwidth, Devanagari, ...): the anonymizer's classes are `[0-9]`,
    where Python's `\d` matched every decimal digit. (The number mask still uses `\p{Nd}`.)
  - U+0130 (İ) and U+0131 (ı): Python's `re.I` lets them match `i`; Rust's `(?i)` does not.
    Affects the URL, keyword and IBAN rules when such a letter touches a match.
- TF-IDF, same split as the baseline (7,023 train, 1,756 test): v5 accuracy 0.9772 (same), macro F1
  0.97562 (v4: 0.97564), 17 missed spam / 23 blocked ham (v4: 16 / 24). Both make 40 errors; one
  message moved from blocked ham to missed spam.
- `export_android_assets.py` lost the hazm port and now writes only the model (with its `pipeline`)
  and a fixture of stage outputs, n-grams and scores. Its output is not shippable before Phase 3.

### Phase 3: the phone uses the core

Two parts. 3a is done in this repo. 3b is done in a separate session started in
`~/Github/NoSpam`, after 3a, because it builds on the commit 3a leaves here. The step numbers
below are kept from the original plan.

#### Phase 3a: this repo (steps 1 and 4)

1. `spam-core-jni`: `pipelineVersion()`, `normalizerMode()`, `ngrams(text)`, `stages(text)`
   (for `feature:mldebug`). Hand-written JNI, not UniFFI: four functions don't justify adding JNA.
   The Rust function names encode the Kotlin class (`com.nospam.nospam.core.ml.NativeCore`),
   so renaming that class means renaming them. A host-side Rust test calls the exported
   functions through a JNI environment; cross-compiling for Android is checked in 3b.
4. `make export-android` writes the model (with `pipeline` and this repo's git SHA) and the
   fixture into NoSpam. Word lists only for a `full` model. No `.so` files. It refuses to run
   when `core/` has uncommitted changes, so the SHA it records names the code that made the
   model.

Done when: the JNI crate's test passes on the host; `make export-android` refuses with
uncommitted changes in `core/` and records the SHA otherwise.

#### Phase 3b: NoSpam (steps 2, 3, 5, 6, plus its CI, README build section and CLAUDE.md)

2. NoSpam: add this repo as a git submodule (e.g. `third_party/persian-sms-spam`), pinned to
   the commit the model was exported from.
3. NoSpam `core:ml` Gradle: a task runs `cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64` on the
   submodule into the build directory's `jniLibs` (not into `src/`), and a host (linux-x86_64)
   build for the JVM unit tests. Both run before `merge*JniLibFolders` / the unit test task.
5. NoSpam `core:ml`: add `NativeCore.kt`; `TfidfSpamClassifier` gets n-grams from it and keeps
   its scoring math. Delete `HazmNormalizer.kt`, `HazmResources.kt`, `TfidfPreprocessor.kt` and
   the hazm assets.
6. Tests:
   - JVM unit test loads the host `.so` and checks the fixture (stages, n-grams, scores). This now
     tests the same code the phone runs.
   - Instrumented test runs the same fixture on the phone (`connectedDebugAndroidTest`)
     before each release.
   - Unit test: model `pipeline` (version and normalizer) equals `NativeCore`'s. The git SHA is
     not checked, so a model someone trained on their own fork still works; `feature:mldebug`
     shows it.
7. NoSpam's workflows (see "CI"), the README build section (draft below) and its `CLAUDE.md`.

Done when: both tests pass on the Galaxy A26; release APK classifies with the v5 model.

### Phase 4: cleanup

This repo's part is done:

- hazm is out of `classic`'s dependencies and out of `common`'s normal ones. `common` has it in the
  `reference` group (with scikit-learn and joblib), used by `make lexicons` and `make core-diff`.
  `classic` keeps it in a `fasttext` group, because `baseline_fasttext.py` tokenizes with hazm's
  `WordTokenizer`; the Makefile's fastText targets pass `--group fasttext`, so TF-IDF work never
  installs hazm. (Replacing that tokenizer would change fastText's results, so it was left.)
- `LICENSE` (GPL-3.0), a dataset description and the Rust requirement in the README, a `core/README.md`,
  and `CLAUDE.md` and the folder READMEs corrected.

NoSpam's part is in Phase 3b: its README build section (draft below), `CLAUDE.md` and the design page.

Phase 3a results (2026-10-14): `spam-core-jni` exports the four functions with `-Xcheck:jni` tests
through a JVM started on the host (needs a JDK; skipped with a note otherwise). The model file carries
`coreGitSha` next to `pipeline`; `make export-android` refuses to run while `core/` has uncommitted
or untracked files. The release profile (LTO, size) is left for 3b, when the APK size is known.

### Draft: build section for NoSpam's README

````markdown
## Build

Requirements: JDK 25 (JetBrains), the Android SDK and NDK, and Rust (rustup). The spam
filter's text processing is Rust code from
[persian-sms-spam](https://github.com/<owner>/persian-sms-spam), included as a git submodule
and compiled by Gradle.

```bash
git clone --recurse-submodules https://github.com/<owner>/NoSpam
cd NoSpam
cargo install cargo-ndk        # once; rustup installs the Rust version and Android targets from rust-toolchain.toml
./gradlew :app:assembleDebug
```

Already cloned without submodules: `git submodule update --init`.

The spam model (`core/ml/src/main/assets/spam_model.json`) is trained on a private SMS
dataset and committed here, so building needs no dataset. To use your own model, train it
with persian-sms-spam ("Train your own model" in its README) and run its
`make export-android` against this checkout.
````

## Implementation notes

**Shared test cases (Phase 0).** One JSON file per stage, e.g. `cases/anonymize.json`:
`{"masked": [["input", "expected"], ...], "unchanged": ["input", ...]}`. Fill them from the
`MASKED`/`UNCHANGED`/`CASES` lists in `common/tests/`, then make those tests load the files.

**Anonymizer rules as data (Phase 1).**
- Generate the first `anonymize.toml` from the Python module rather than translating by hand:
  write out each compiled `pattern.pattern` (`FORMAT_RULES`, `_KEYWORD_RES`, `LONG_NUMBER`).
  These are the final regexes after `_NB`, `_filler()`, `_TOKEN` etc. were filled in.
- Schema sketch: `[[format]] tag, pattern, validator = "luhn" | "natid" (optional)` and
  `[[keyword]] tag, pattern` (flags inline, e.g. `(?i)` for the keyword rules and the URL rule).
- Order and semantics, as in `anonymize()`: translate digits and Arabic letters first
  (`_TRANSLATE`); then format rules in file order (the whole match becomes `[TAG]`, unless
  its validator rejects it, then the match stays and scanning continues after it); then keyword
  rules (group 1 kept, group 2 becomes `[TAG]`); then `LONG_NUMBER` → `[NUMBER]`. Count tags.
- After `_TRANSLATE` the anonymizer only sees ASCII digits, so `\d` → `[0-9]` is exact for
  Persian/Arabic-Indic digits. Other digit scripts (e.g. fullwidth) change behavior; write that
  down as an intended change.

**hazm rules (Phase 1).**
- `rules/hazm.json` = output of `export_resources()` in `classic/export_android_assets.py`
  (replacement templates already parsed into `{"l": literal}` / `{"g": group}` parts).
- Port `normalize_ha()` and its helpers from the same file, step by step and in the same order.
  `HazmNormalizer.kt` is a second reference, but its `(?u)` flags are wrong (see U+2009 above).
- `rules` mode = `normalize_ha()` with empty word lists; `full` mode = with the lists.

**n-grams (Phase 1).** Port sklearn's `char_wb` exactly, as `TfidfPreprocessor.kt` does: lowercase,
collapse whitespace runs to one space, split on whitespace, pad each word with one space,
for n in 2..=4 slide a window, and a word shorter than n gives one window and ends the n loop.

**Python binding (Phase 2).**
- `spam-core-py/pyproject.toml`: `build-backend = "maturin"`, `[tool.maturin] module-name = "spam_core"`,
  and `[tool.uv] cache-keys` covering `Cargo.toml`, `../spam-core/src/**/*.rs`,
  `../spam-core/rules/*` and its own `src/**/*.rs`.
- `common/pyproject.toml`: depend on `spam-core` with `[tool.uv.sources] spam-core = { path =
  "../core/spam-core-py", editable = true }`. Check that `classic`, `parsbert` and `data-prep`
  still resolve it through `common`; if uv does not apply `common`'s source there, add the same
  source line to each of them.
- The TF-IDF analyzer must be a module-level Python function (`spam_common.pipeline.ngrams`)
  so joblib can pickle the trained pipeline. With a callable analyzer sklearn ignores
  `lowercase` and `ngram_range`, so the core does both; drop those fields from the export.
- `make check` runs `cargo test` (in `core/`) as well as pytest. `$(PIPELINE)` in the Makefile
  lists the core's `.rs` and rules files plus `pipeline.py`.
- The fixture battery for the differential test is `build_battery()` in
  `classic/export_android_assets.py`.

**Android (Phase 3).**
- JVM unit tests find the host library through `java.library.path`: set it with
  `systemProperty` on the `Test` tasks in `core/ml/build.gradle.kts`.
- Point cargo-ndk at the NDK AGP uses (`android.ndkVersion`; `ANDROID_NDK_HOME`).
- The instrumented test needs the fixture as an androidTest asset, e.g. add
  `src/test/resources` to the `androidTest` assets source set.
- `TfidfSpamClassifier` stops lowercasing and building n-grams itself; it counts the n-grams
  `NativeCore` returns and keeps the idf / L2 / dot-product code.

## Versioning

- **`PIPELINE_VERSION`** (Rust constant) = the output contract. Bump on any output change.
- **Enforced, not by discipline:** golden files `tests/golden/v{N}-{mode}.json`. `cargo test` fails
  when output differs from the file for the current version; the regenerate command refuses
  to overwrite an existing version's file. Changing output without bumping cannot pass tests.
- **Core version in NoSpam** = the submodule commit. Updating the model means moving the
  submodule to the commit the model was exported from, in the same NoSpam commit.
- **Model ↔ core:** the model file carries `pipeline {version, normalizer}`, checked by a test,
  and the git SHA it was exported from, for traceability only. Model and library ship in the
  same APK, so a mismatch is caught by tests, not by users.
- No crate publishing or semver needed while the only consumers are this repo and NoSpam.

## CI

- This repo has no CI. `make check` runs `cargo test` + pytest locally. Optional later: a
  workflow with `cargo test`, pytest, and `cargo ndk build` to catch cross-compile breakage.
- NoSpam's workflows (`android.yml`, `release.yml`): check out with submodules, install the
  Rust toolchain from `rust-toolchain.toml`, the Android targets and `cargo-ndk`, and cache
  `~/.cargo` and `target/`. The NDK version is pinned in `core/ml/build.gradle.kts`.

## Revisit these decisions if

| Decision | Revisit when |
|---|---|
| Core in this repo | A third consumer (iOS, a server) appears: move `core/` to its own repo |
| Hand-written JNI | The API grows to objects and callbacks: switch to UniFFI |
| `rules` normalizer | A larger labeled set shows the word lists help: switch to `full` |

## Future: min_df (not decided, keep `min_df=1` for now)

`min_df` is the number of different training messages an n-gram must appear in to be kept in
the model. Today it is 1, so the published weights include n-grams that come from a single
message, often pieces of names. Measured on 2026-10-14 (v5 pipeline, TF-IDF, nothing changed):

| min_df | Errors, test set (1,756) | Errors per CV repeat (8,779, 10x5-fold) | n-grams | Export size |
|---|---|---|---|---|
| 1 (today) | 40 | 302.7 | 50,033 | 2.7 MB |
| 2 | 40 | 300.9 (-1.8, noise) | 28,251 | 1.6 MB |
| 3 | 38 | 300.8 (-1.9, noise) | 21,552 | 1.2 MB |
| 5 | 43 | 295.9 (-6.8, p = 0.14) | 15,978 | 0.9 MB |

- 2 or 3 cost no accuracy, make the model 40-55% smaller and drop every n-gram seen in only
  one message (~23,900 when trained on all data). 5 is unclear (CV better, test set worse).
- Changing it does not change the pipeline (no `PIPELINE_VERSION` bump), only the model
  file: set `min_df` in `classic/baseline_tfidf.py`, retrain, re-export.
- Decide before the repos are published; the user wants to think about it first.
