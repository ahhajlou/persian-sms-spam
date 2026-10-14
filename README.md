# Persian SMS Spam

Spam/ham classifiers for Persian SMS, trained on messages exported from Android phones.

| Directory | Approach | Main dependencies |
|---|---|---|
| [`data-prep/`](data-prep/) | Label viewer GUI, synthetic spam augmentation via local LLM | requests, tkinter |
| [`parsbert/`](parsbert/) | Fine-tuned ParsBERT (`HooshvareLab/bert-fa-zwnj-base`), exported to TFLite | torch, transformers |
| [`classic/`](classic/) | TF-IDF + Linear SVM and fastText, exported as Android assets | scikit-learn, fasttext, hazm |

Each directory is an independent [uv](https://docs.astral.sh/uv/) project with its own
`pyproject.toml`, `uv.lock` and `.venv`, because their dependencies differ widely
(e.g. `classic` needs `numpy<2` for fastText, `parsbert` pulls in torch).

```bash
cd classic && uv sync     # or parsbert / data-prep
```

## Shared directories (not committed)

```
data/     # raw SMS dumps and labelled datasets (sms_labeled.jsonl, ...)
models/   # trained models, checkpoints and exports
```

Both contain or are derived from personal messages and are listed in `.gitignore`.

## Collecting data

```bash
adb shell content query --uri content://sms/ > data/sms_dump.txt
```

Extraction and regex labelling (steps 1–2) are in `parsbert/spam-filter-pipeline.ipynb`
and write `data/sms_unlabeled.jsonl` and `data/sms_labeled.jsonl`.
