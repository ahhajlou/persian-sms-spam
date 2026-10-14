# spam_common

Shared by every model project (installed editable via `[tool.uv.sources]`):

- `data.py`: repo paths (`TRAIN_FILE`, `TEST_FILE`, `MODELS_DIR`, ...), labels, `load_texts_labels`, `clean_text`
- `split.py`: `python -m spam_common.split` builds the deduplicated, stratified train/test split
- `metrics.py`: `evaluate()` prints and saves a report; `python -m spam_common.metrics` compares all saved results

No third-party dependencies, so it works in any model's environment.
