# Persian SMS Spam

Spam/ham classifiers for Persian SMS, trained on messages exported from Android phones.

| Directory | Approach | Main dependencies |
|---|---|---|
| [`common/`](common/) | Shared preprocessing pipeline, anonymizer, data loading, train/test split, metrics (`spam_common`) | hazm |
| [`data-prep/`](data-prep/) | SMS extraction, regex first-round labelling, label viewer GUI, synthetic spam via local LLM | requests, tkinter |
| [`classic/`](classic/) | TF-IDF + Linear SVM and fastText, exported as Android assets | scikit-learn, fasttext, hazm |
| [`parsbert/`](parsbert/) | Fine-tuned ParsBERT (`HooshvareLab/bert-fa-zwnj-base`), exported to TFLite | torch, transformers |

Each directory is an independent [uv](https://docs.astral.sh/uv/) project with its own
`pyproject.toml`, `uv.lock` and `.venv`, because their dependencies differ widely
(e.g. `classic` needs `numpy<2` for fastText, `parsbert` pulls in torch).
The others install `common` as an editable path dependency.

## Usage

The root `Makefile` is the single entry point; it runs each step in the right environment.

```bash
make                  # list all commands
make split            # data/sms_labeled.jsonl → data/train.jsonl + data/test.jsonl
make test-all         # train what's out of date, score every model, print a comparison
make train-tfidf      # or train-fasttext / train-parsbert
make predict-tfidf TEXT="متن پیامک"
make export-android ANDROID_MODULE=path/to/core/ml
```

Models are file targets: they are retrained only when `data/train.jsonl`, their
training script or the preprocessing pipeline changes. `make test-all` therefore fine-tunes ParsBERT when it is out of date.

### Preprocessing

Every model trains and predicts on the output of `spam_common.pipeline.preprocess()`:

```
raw SMS → anonymize → mask URLs → mask numbers → hazm normalize → model
```

The Android app runs the same steps in Kotlin. `make export-android` writes a fixture
with the expected output of every step for its parity test. Bump `PIPELINE_VERSION`
whenever a change alters the output. Changing the pipeline retrains every model.

### Evaluation

Every model trains on `data/train.jsonl` and is scored on the same `data/test.jsonl`.
The split drops duplicate messages first, compared after preprocessing (the same SMS is
often received many times, or differs only in a code), so no test message is seen during
training, and keeps the ham/spam ratio in both parts.
Results are saved to `models/results/<model>.json`; `make compare` prints them side by side.

## Shared directories (not committed)

```
data/     # raw SMS dumps and labelled datasets (sms_labeled.jsonl, train/test split, ...)
models/   # trained models, checkpoints, exports and results/
```

Both contain or are derived from personal messages and are listed in `.gitignore`.

## Adding data from a phone

```bash
make anonymize   IN=data/<phone>/export.jsonl              # mask private data
make regex-label IN=data/<phone>/export_anon.jsonl         # first-round labels
make label     DATA=data/<phone>/export_anon_regex.jsonl   # fix labels by hand in the GUI
make merge       IN=data/<phone>/export_anon_regex.jsonl   # add to data/sms_labeled.jsonl
make test-all                                              # re-split, retrain, compare
```

`merge` skips rows that are still unlabelled and messages already in the dataset, and
keeps the previous dataset as `sms_labeled.jsonl.bak`.

Exports from an SMS export app are already JSONL. For a raw `adb` dump, extract it first:

```bash
adb shell content query --uri content://sms/ > data/sms_dump.txt
make extract          # -> data/sms_unlabeled.jsonl, then continue with make anonymize
```
