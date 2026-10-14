# Persian SMS Spam

Spam/ham classifiers for Persian SMS, trained on messages exported from Android phones.

| Directory | Approach | Main dependencies |
|---|---|---|
| [`common/`](common/) | Shared data paths, loading, train/test split and metrics (`spam_common`) | none |
| [`data-prep/`](data-prep/) | Regex first-round labelling, label viewer GUI, synthetic spam augmentation via local LLM | requests, tkinter |
| [`classic/`](classic/) | TF-IDF + Linear SVM and fastText, exported as Android assets | scikit-learn, fasttext, hazm |
| [`parsbert/`](parsbert/) | Fine-tuned ParsBERT (`HooshvareLab/bert-fa-zwnj-base`), exported to TFLite | torch, transformers |

Each directory is an independent [uv](https://docs.astral.sh/uv/) project with its own
`pyproject.toml`, `uv.lock` and `.venv`, because their dependencies differ widely
(e.g. `classic` needs `numpy<2` for fastText, `parsbert` pulls in torch).
`classic` and `parsbert` install `common` as an editable path dependency.

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

Models are file targets: they are retrained only when `data/train.jsonl` or their
training script changes. `make test-all` therefore fine-tunes ParsBERT when it is out of date.

### Evaluation

Every model trains on `data/train.jsonl` and is scored on the same `data/test.jsonl`.
The split drops duplicate messages first (the same SMS is often received many times),
so no test message is seen during training, and keeps the ham/spam ratio in both parts.
Results are saved to `models/results/<model>.json`; `make compare` prints them side by side.

## Shared directories (not committed)

```
data/     # raw SMS dumps and labelled datasets (sms_labeled.jsonl, train/test split, ...)
models/   # trained models, checkpoints, exports and results/
```

Both contain or are derived from personal messages and are listed in `.gitignore`.

## Collecting data

```bash
adb shell content query --uri content://sms/ > data/sms_dump.txt
```

Extraction and regex labelling (steps 1–2) are in `parsbert/spam-filter-pipeline.ipynb`
and write `data/sms_unlabeled.jsonl` and `data/sms_labeled.jsonl`.
