# Persian SMS Spam

Spam filters for Persian (Farsi) SMS. The models are trained on real messages exported
from Android phones and are built to run on the phone itself: the TF-IDF model is used
by the NoSpam Android app, and ParsBERT can be exported to TFLite.

The repo covers the whole path from a phone export to a model on the phone:

```
phone export → extract → anonymize → first-round labels → manual check → merge
            → train/test split → train three kinds of model → compare → export to Android
```

## Models

| Model | Directory | What it is | On the phone |
|---|---|---|---|
| TF-IDF + Linear SVM | `classic/` | character n-grams with a linear classifier; small and fast | yes (used by the app) |
| fastText | `classic/` | word embeddings + linear classifier; optional pretrained Persian vectors | quantized `.ftz` |
| ParsBERT | `parsbert/` | fine-tuned `HooshvareLab/bert-fa-zwnj-base` transformer | TFLite export |

All models train on the same data and are scored on the same held-out test set, so their
results can be compared directly. Latest results on 219 unseen test messages:

| Model | Accuracy | Macro F1 | Missed spam | Real messages blocked |
|---|---|---|---|---|
| fastText | 0.977 | 0.972 | 3 | 2 |
| ParsBERT | 0.973 | 0.967 | 2 | 4 |
| TF-IDF | 0.973 | 0.966 | 3 | 3 |
| fastText quantized | 0.941 | 0.929 | 3 | 10 |

For a spam filter, "real messages blocked" is the costliest error: those are messages
the user never sees.

## Repository layout

```
common/      spam_common: shared preprocessing pipeline, anonymizer, data helpers, split, merge, metrics
data-prep/   tools that make the dataset: SMS extraction, regex first-round labels, label GUI, augmentation
classic/     TF-IDF and fastText models, Android asset export
parsbert/    ParsBERT fine-tuning, evaluation and TFLite conversion
Makefile     the single entry point for every command
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

Requirements: [uv](https://docs.astral.sh/uv/) and GNU Make 4.3+. The label GUI uses Qt
(PySide6, installed by `make sync`); on an X11 session Fedora may also need `xcb-util-cursor`.
A CUDA GPU speeds up ParsBERT but is optional.

```bash
make sync     # create every environment (first run downloads torch, so it takes a while)
make check    # run the unit tests
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
preprocessing pipeline changed; otherwise `make` reuses them.

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
make export-android ANDROID_MODULE=/path/to/NoSpam/core/ml   # TF-IDF model + hazm resources + parity fixture
make export-tflite                                           # ParsBERT → models/parsbert_spam.tflite
```

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
this repo is at version 4. A model exported now fails NoSpam's parity test and would see
different input on the phone than in training. The shared Rust core in
[docs/preprocessing-core-plan.md](docs/preprocessing-core-plan.md) fixes this; until then, keep
the model NoSpam ships.

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
`spam_common.pipeline.preprocess()`:

```
raw SMS → anonymize → mask URLs → mask numbers → hazm normalize → model
```

- **anonymize** replaces private values with placeholders such as `[CODE]`, `[CARD]` or
  `[NATID]` (national IDs are detected by their check digit), so the model learns
  "this is an OTP message", not the code itself. `common/tests/test_anonymize.py` holds
  the cases it must mask and must leave alone; add one for every mistake you find.
- **mask URLs / numbers** turns links (any scheme, `www.`, or a bare domain like `t.me/x`)
  and remaining numbers into `URLTOKEN` and `NUM_TOKEN`.
- **hazm** normalizes Persian text (letters, digits, spacing, half-spaces).

The Android app runs the same steps in Kotlin. `make export-android` writes a fixture with
the expected output of each step for some 500 test inputs, which the app's JVM test
checks, so the phone and training can't drift apart silently. The Kotlin port is currently
at pipeline version 1; [docs/preprocessing-core-plan.md](docs/preprocessing-core-plan.md)
replaces it with a Rust core shared by both projects. Bump `PIPELINE_VERSION` in
`pipeline.py` whenever a change alters the output; changing the pipeline retrains every
model.

### Evaluation

- The split removes duplicates first, compared after preprocessing: the same SMS is often
  received many times or differs only in a code, and a copy in both train and test would
  inflate the scores.
- It keeps the ham/spam ratio in both parts (stratified, 80/20, fixed seed).
- Each model's report is saved to `models/results/<model>.json`; `make compare` shows them
  side by side and marks with `*` any result scored under an older pipeline version.
- Only TF-IDF runs on the phone. `make test-tfidf` is enough when you change the data or the
  pipeline; train fastText and ParsBERT only when you want to compare.

## Privacy

`data/` holds real personal messages and `models/` is trained on them. Both are in
`.gitignore` and must never be committed or published. The anonymizer is rule-based: it
does not detect person names or street addresses, so check a sample by hand before
sharing any data.

The only trained artifact that is published is the TF-IDF weights that `make export-android`
writes into NoSpam (`spam_model.json`): the app needs them on the phone. The dataset itself
stays private.
