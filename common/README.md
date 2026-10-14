# spam_common

Code that more than one project depends on, installed editable into `classic`, `parsbert`
and `data-prep` via `[tool.uv.sources]`. Its only dependency is `spam-core`, the Python binding
of the Rust core in `../core` (built by maturin). hazm is an optional dependency group
(`reference`), used only by `make lexicons` and `make core-diff`.

- `data.py`: repo paths (`DATA_DIR`, `TRAIN_FILE`, `MODELS_DIR`, ...), labels, JSONL helpers, `output_path()`
- `pipeline.py`: `preprocess()`, the one text pipeline every model trains and predicts on
  (anonymize → URL mask → number mask → normalize), and `ngrams()` for TF-IDF; wrappers over the
  Rust core, which the Android app will build too. `PIPELINE_VERSION` is the core's constant
- `anonymize.py`: the `anonymize` CLI and wrapper; the rules (passwords, codes, IDs, card/account
  numbers, phones, emails) are in `core/spam-core/rules/anonymize.toml`
- `split.py`: deduplicated, stratified train/test split
- `merge.py`: adds a reviewed file to the labelled dataset
- `model_label.py`: first-round labels with a trained model (each model's `label` command)
- `metrics.py`: `evaluate()` saves a model's test report; running it compares all saved reports

Rule: anything a model or the split depends on lives here, so the logic exists once.
Commands: `make help` in the repo root.
