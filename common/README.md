# spam_common

Code that more than one project depends on, installed editable into `classic`, `parsbert`
and `data-prep` via `[tool.uv.sources]`. Its only dependency is hazm.

- `data.py`: repo paths (`DATA_DIR`, `TRAIN_FILE`, `MODELS_DIR`, ...), labels, JSONL helpers, `output_path()`
- `pipeline.py`: `preprocess()`, the one text pipeline every model trains and predicts on
  (anonymize → URL mask → number mask → hazm), mirrored in Kotlin on Android
- `anonymize.py`: rule-based masking of passwords, codes, IDs, card/account numbers, phones, emails
- `split.py`: deduplicated, stratified train/test split
- `merge.py`: adds a reviewed file to the labelled dataset
- `metrics.py`: `evaluate()` saves a model's test report; running it compares all saved reports

Rule: anything a model or the split depends on lives here, so the logic exists once.
Commands: `make help` in the repo root.
