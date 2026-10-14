# spam_common

Shared by every model project (installed editable via `[tool.uv.sources]`):

- `data.py`: repo paths (`TRAIN_FILE`, `TEST_FILE`, `MODELS_DIR`, ...), labels, `load_texts_labels`, `clean_text`
- `pipeline.py`: `preprocess()`, the one text pipeline every model trains and predicts on
  (anonymize → URL mask → number mask → hazm), mirrored in Kotlin on Android
- `split.py`: `python -m spam_common.split` builds the deduplicated, stratified train/test split
- `merge.py`: `python -m spam_common.merge` adds a reviewed file to the labelled dataset
- `metrics.py`: `evaluate()` prints and saves a report; `python -m spam_common.metrics` compares all saved results
- `anonymize.py`: `python -m spam_common.anonymize` masks sensitive data in a phone export with typed placeholders

Its only dependency is hazm (for the normalizer), so it works in any model's environment.
