# Persian SMS Spam

Spam filters for Persian (Farsi) SMS. The models are trained on real messages exported
from Android phones and are built to run on the phone itself: the TF-IDF model is used
by the NoSpam Android app, and ParsBERT can be exported to TFLite.

The repo covers the whole path from a phone export to a model on the phone:

```
phone export → extract → anonymize → first-round labels → manual check → merge
            → train/test split → train three kinds of model → compare → export to Android
```

The text preprocessing (anonymizer, URL and number masks, Persian normalizer, character
n-grams) is a small Rust library in `core/` that training and the NoSpam app both use, so a
message becomes exactly the same model input in both places. The code is GPL-3.0 licensed
(see [License](#license)); the dataset it was trained on is private (see [The dataset](#the-dataset)).

## Models

| Model | Directory | What it is | On the phone |
|---|---|---|---|
| TF-IDF + Linear SVM | `classic/` | character n-grams with a linear classifier; small and fast | yes (used by the app) |
| fastText | `classic/` | word embeddings + linear classifier; optional pretrained Persian vectors | quantized `.ftz` |
| ParsBERT | `parsbert/` | fine-tuned `HooshvareLab/bert-fa-zwnj-base` transformer | TFLite export |

All models train on the same data and are scored on the same held-out test set, so their
results can be compared directly. Latest results on 1,756 unseen test messages:

| Model | Accuracy | Macro F1 | Missed spam | Real messages blocked |
|---|---|---|---|---|
| TF-IDF | 0.977 | 0.976 | 17 | 23 |
| fastText | 0.967 | 0.965 | 27 | 31 |
| ParsBERT | 0.961 | 0.958 | 45 | 24 |
| fastText quantized | 0.954 | 0.951 | 12 | 69 |

TF-IDF was scored with the current preprocessing (pipeline version 5). The other three were
last scored with version 4 on the same split and are shown here as they were; re-run them
with `make test-fasttext` / `make test-parsbert` for a like-for-like table.

For a spam filter, "real messages blocked" is the costliest error: those are messages
the user never sees.

## The dataset

The dataset is private and is not in this repo (see [Privacy](#privacy)). What it is:

- **Source:** SMS exported from two personal Android phones with NoSpam's Export screen.
  The two exports hold 20,091 and 2,375 messages; after removing messages that are the same
  once preprocessed (the same SMS is often received many times, or differs only in a code),
  8,779 are labeled and used.
- **Labels:** 5,529 ham and 3,250 spam. The first-round label of each message came from a
  trained model (`make model-label`), and every message was then checked and corrected by hand
  in the label GUI (`make label`).
- **Split:** 80/20, stratified, fixed seed: 7,023 messages for training and 1,756 for testing,
  with no message (after preprocessing) in both.
- **Privacy:** passwords, codes, national IDs, card, IBAN and account numbers, phone numbers
  and emails were masked by the anonymizer before labeling. It does not detect names or street
  addresses.
- **Why it is not published:** these are real personal messages. Masking the values the rules
  know does not make them safe to share, so only the trained TF-IDF weights are published
  (inside NoSpam), never the messages.


## Repository layout

```
core/        the Rust preprocessing core (spam-core), its Python and JNI bindings, rules and tests
common/      spam_common: the pipeline entry point, data helpers, split, merge, metrics
data-prep/   tools that make the dataset: SMS extraction, regex first-round labels, label GUI, augmentation
classic/     TF-IDF and fastText models, Android asset export
parsbert/    ParsBERT fine-tuning, evaluation and TFLite conversion
docs/        design plan for the Rust core
Makefile     the single entry point for every command
LICENSE      GPL-3.0
data/        your SMS data (not committed)
models/      trained models and results (not committed)
```

Each of `common`, `data-prep`, `classic` and `parsbert` is a separate
[uv](https://docs.astral.sh/uv/) project with its own environment, because their
dependencies conflict (fastText needs `numpy<2`, ParsBERT needs torch). The other three
install `common` as an editable dependency, so shared code exists only once:
anything a model or the split depends on lives in `common/`; tools that only produce or
edit data files live in `data-prep/`.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), GNU Make 4.3+ and Rust
([rustup](https://rustup.rs/); it installs the version pinned in `core/rust-toolchain.toml` by
itself). A JDK is optional: without one the JNI tests in `make check` print `SKIPPED`. The label GUI uses Qt
(PySide6, installed by `make sync`); on an X11 session Fedora may also need `xcb-util-cursor`.
A CUDA GPU speeds up ParsBERT but is optional.

```bash
make sync     # create every environment and build the Rust core (the first run downloads torch)
make check    # run the unit tests: cargo test for the core, pytest for the Python code
make          # list all commands
```

## Usage

Run everything from the repo root. `make` picks the right environment for each step.

### Train and compare the models

```bash
make split                 # data/sms_labeled.jsonl → data/train.jsonl + data/test.jsonl
make test-all              # train what is out of date, score every model, print the table
make compare               # print the table again from the saved results
```

Models are retrained only when the training data, their training script or the
preprocessing pipeline (the Rust core's code or rules) changed; otherwise `make` reuses them.

### One model at a time

```bash
make train-tfidf                          # or train-fasttext / train-parsbert
make test-tfidf                           # or test-fasttext / test-parsbert
make predict-tfidf TEXT="متن پیامک"        # or predict-fasttext
make run-parsbert                         # type messages, see ParsBERT's verdict
make train-fasttext-pretrained FASTTEXT_VEC=classic/cc.fa.300.vec
```

### Export to Android

```bash
make export-android ANDROID_MODULE=/path/to/NoSpam/core/ml   # TF-IDF model + parity fixture
make export-tflite                                           # ParsBERT → models/parsbert_spam.tflite
```

`export-android` refuses to run while `core/` has uncommitted changes, and records the commit it
exported from in the model (`coreGitSha`): NoSpam builds the text processing from that commit.

## Train your own model

The SMS dataset behind the published model is private and is not in this repo (see
[Privacy](#privacy)). You can build the NoSpam app without it, since NoSpam ships the trained
weights. To train a model yourself, use your own messages:

1. **Export your SMS** into `data/<phone>/`: NoSpam's Export screen writes
   `sms_export_<time>.jsonl`; an adb dump works too (step 0 below). Each row needs a `text` field.
2. **Anonymize:** `make anonymize IN=data/<phone>/sms_export_<time>.jsonl`.
3. **Label.** You have no model yet, so either:
   - `make regex-label IN=...` with your own spam rules in `data/regex_rules.json`
     (`{"rules": [{"pattern": "...", "isBlacklist": true}, ...]}`, as SpamBlocker exports them),
     then check the labels with `make label`, or
   - label every message by hand: `make label DATA=data/<phone>/sms_export_<time>_anon.jsonl`
     (the **Label** filter's **null** option shows the unlabelled ones).
4. **Create the dataset:** `make merge IN=<reviewed file>` writes `data/sms_labeled.jsonl`.
5. **Train and score:** `make test-tfidf` trains only the model the phone uses; `make test-all`
   trains all three (ParsBERT is slow without a GPU).
6. **Next exports:** your model can now give the first-round labels (`make model-label`, below).

### Use your model in NoSpam

```bash
make export-android ANDROID_MODULE=/path/to/NoSpam/core/ml
```

writes your weights and a test fixture into your NoSpam checkout; then build NoSpam as usual.

**Not yet possible:** NoSpam's on-device preprocessing is still at pipeline version 1, while
this repo is at version 5. A model exported now fails NoSpam's parity test and would see
different input on the phone than in training. The shared Rust core is finished in this repo;
NoSpam switching to it is the last step of
[docs/preprocessing-core-plan.md](docs/preprocessing-core-plan.md) (Phase 3b). Until that is
merged, keep the model NoSpam ships.

## Adding data from a phone

Every step reads one file and writes a new one next to it, so you can always go back to
an earlier step. No step overwrites an existing file unless you add `FORCE=1`. The data
tools only change a message's `text` and `label` (plus a review-only `model_score`, see
step 2); every other field stays as exported.

```bash
# 0. Only for a raw adb dump (exports from an SMS export app are already JSONL):
adb shell content query --uri content://sms/ > data/sms_dump.txt
make extract                                               # → data/sms_unlabeled.jsonl

# 1. Mask private data: passwords, codes, national IDs, card/IBAN/account numbers, phones, emails
make anonymize   IN=data/<phone>/export.jsonl              # → export_anon.jsonl

# 2. First-round labels from a trained model: tfidf (default), fasttext or parsbert
make model-label IN=data/<phone>/export_anon.jsonl MODEL=tfidf   # → export_anon_model.jsonl

# 3. Check and fix the labels by hand, then save in the GUI
make label     DATA=data/<phone>/export_anon_model.jsonl

# 4. Add the reviewed messages to the dataset
make merge       IN=data/<phone>/export_anon_model.jsonl   # → data/sms_labeled.jsonl

# 5. Re-split, retrain and compare
make test-all
```

`merge` skips messages that are still unlabelled or already in the dataset, and keeps the
previous dataset as `data/sms_labeled.jsonl.bak`. It cannot correct labels of messages
already in the dataset; fix those with `make label DATA=data/sms_labeled.jsonl`.

**Step 2** gives messages that are already in the dataset their existing, checked label.
Every other message gets the model's label and a `model_score`: how sure the model is
that it is spam (0–1). In the GUI, set **Model score** to *Uncertain* or **Sort** to *Most
uncertain first* and review those messages carefully; confident ones only need a skim.
Still review before merging: unchecked model labels would teach the model its own mistakes.
`merge` drops `model_score`, so it never reaches the dataset.

| `MODEL` | Speed | Use it for |
|---|---|---|
| `tfidf` | seconds | everyday labelling; the same model as on the phone |
| `fasttext` | seconds | a second opinion |
| `parsbert` | about a minute on a GPU | exports with many new kinds of messages |

`make model-label` first retrains the chosen model if it is out of date. Before you have
a trained model, `make regex-label IN=...` labels with your SpamBlocker rules in
`data/regex_rules.json` instead (→ `export_anon_regex.jsonl`); use one of the two, not both.

The label GUI's **Label** filter has a **null** option that shows unlabelled messages.

`make augment N=200` can add synthetic spam generated by a local Ollama model.

## How it works

### Preprocessing

Every model trains and predicts on the output of one function,
`spam_common.pipeline.preprocess()`, which calls the Rust core in `core/spam-core`:

```
raw SMS → anonymize → mask URLs → mask numbers → normalize → model
                                                          └→ character n-grams (TF-IDF)
```

Only the first 2,000 characters of a message are used. The rules are data files in
`core/spam-core/rules/`.

- **anonymize** replaces private values with placeholders such as `[CODE]`, `[CARD]` or
  `[NATID]` (national IDs are detected by their check digit), so the model learns
  "this is an OTP message", not the code itself. `core/spam-core/tests/cases/anonymize.json` holds
  the cases it must mask and must leave alone (pytest and `cargo test` both read it); add one
  for every mistake you find.
- **mask URLs / numbers** turns links (any scheme, `www.`, or a bare domain like `t.me/x`)
  and remaining numbers into `URLTOKEN` and `NUM_TOKEN`.
- **normalize** applies hazm's regex rules and character tables to Persian text (letters,
  digits, spacing, half-spaces). hazm's word lists (14 MB) are optional (`make lexicons`) and
  not used: they made no measurable difference. hazm itself is only needed to regenerate
  `rules/hazm.json` and for `make core-diff`.

The Android app will run this same Rust core instead of a Kotlin copy, built from the commit
recorded in the model. `make export-android` writes a fixture with the expected output of each
step for about 500 test inputs, which the app's tests check, so the phone and training can't
drift apart silently. The Kotlin port in NoSpam is still at pipeline version 1 until
[Phase 3b of the plan](docs/preprocessing-core-plan.md) is done.

Bump `PIPELINE_VERSION` (a constant in `core/spam-core/src/lib.rs`) whenever a change alters
the output: `cargo test` fails until you do, and changing the pipeline retrains every model.
`make core-diff` compares the core with the old Python pipeline it replaced.

### Evaluation

- The split removes duplicates first, compared after preprocessing: the same SMS is often
  received many times or differs only in a code, and a copy in both train and test would
  inflate the scores.
- It keeps the ham/spam ratio in both parts (stratified, 80/20, fixed seed).
- Each model's report is saved to `models/results/<model>.json`; `make compare` shows them
  side by side and marks with `*` any result scored under another pipeline version or normalizer.
- Only TF-IDF runs on the phone. `make test-tfidf` is enough when you change the data or the
  pipeline; train fastText and ParsBERT only when you want to compare.

## Privacy

`data/` holds real personal messages and `models/` is trained on them. Both are in
`.gitignore` and must never be committed or published. The anonymizer is rule-based: it
does not detect person names or street addresses, so check a sample by hand before
sharing any data.

The only trained artifact that is published is the TF-IDF weights that `make export-android`
writes into NoSpam (`spam_model.json`): the app needs them on the phone. The dataset itself
stays private. Note that the weights contain short text fragments (2-4 character n-grams) of the
training messages; see "Future: min_df" in the plan.

## License

GPL-3.0, see [LICENSE](LICENSE). `core/spam-core/rules/hazm.json` is derived from
[hazm](https://github.com/roshan-research/hazm) (MIT); its copyright notice is in
`core/spam-core/rules/HAZM-LICENSE.txt`.
