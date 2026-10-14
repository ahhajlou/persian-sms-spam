# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Persian (Farsi) SMS spam filters (TF-IDF + Linear SVM, fastText, ParsBERT) trained on real phone
exports and built to run on Android (the NoSpam app, a separate repo, uses the TF-IDF model).

## Commands

The root `Makefile` is the single entry point; run everything from the repo root. `make` (or `make help`) lists every target and variable.

```bash
make sync                    # create/update all four uv environments
make check                   # unit tests (common/tests)
uv run --project common --with pytest pytest -q common/tests/test_anonymize.py -k masked   # one test
make split                   # data/sms_labeled.jsonl -> data/train.jsonl + data/test.jsonl
make test-all                # (re)train what is out of date, score every model, print comparison
make test-tfidf | train-fasttext | predict-tfidf TEXT="..."   # one model
make label DATA=path.jsonl   # PySide6 label review GUI (opens empty without DATA)
```

No linter or formatter is configured. Only `common/` has tests. Test the GUI headless with
`QT_QPA_PLATFORM=offscreen uv run --project data-prep python ...`.

## Architecture

**Four separate uv projects** (`common`, `data-prep`, `classic`, `parsbert`), each with its own
`.venv`, because their dependencies conflict (fastText needs `numpy<2`, ParsBERT needs torch).
`data-prep`, `classic` and `parsbert` install `common` (package `spam_common`) as an editable
path dependency. Run a script with `uv run --project <dir> python ...`; the Makefile variables
`COMMON` / `CLASSIC` / `PARSBERT_PY` / `DATAPREP` do this. A dependency change means
`uv lock` in that project's directory.

**Code placement rule:** anything a model or the split depends on goes in `common/` so it
exists once. Standalone file tools that only produce or edit data files go in `data-prep/`.
Repo paths (`DATA_DIR`, `LABELED_FILE`, `TRAIN_FILE`, `MODELS_DIR`, `RESULTS_DIR`) come from
`spam_common/data.py`, which finds them relative to the repo root. Don't hard-code paths.

**One preprocessing pipeline.** Every model trains and predicts on `spam_common.pipeline.preprocess()`:
anonymize → mask URLs (`URLTOKEN`) → mask numbers (`NUM_TOKEN`) → hazm normalize. The order matters,
and the reasons are in the module docstring. Any change that alters pipeline or anonymizer
output must bump `PIPELINE_VERSION` (and add a line to the history comment above it). Every anonymizer
mistake found gets a regression case in `common/tests/test_anonymize.py` (masked or must-stay-unchanged lists).

**Android parity, and the planned Rust core.** The NoSpam app (`core:ml`) reimplements the pipeline
in Kotlin and checks a parity fixture written by `make export-android`. The Kotlin side is still at
pipeline v1 (no anonymizer, old URL rule), so a model exported from this repo now fails NoSpam's
parity test and must not ship. Don't port the anonymizer to Kotlin: `docs/preprocessing-core-plan.md`
replaces both implementations with one Rust core (`core/`) that both repos build, keeping hazm's
regex rules and character tables but not its word lists by default. Follow that plan for
pipeline work that touches Android.

**Makefile as build graph.** Model files and the split are real file targets that depend on the
training data, their script and `$(PIPELINE)` (`pipeline.py` + `anonymize.py`). So editing the
pipeline makes every model out of date and rebuilds the split; `make` retrains only the models you ask for.
Only TF-IDF runs in NoSpam, so `make test-tfidf` is enough for pipeline work; fastText and ParsBERT
stay in the repo and are trained only on request.
`train.jsonl`/`test.jsonl` come from `spam_common.split`. It deduplicates on `dedup_key()` (the preprocessed
text), so repeated SMS or SMS differing only in a code never land in both train and test, and it
uses a stratified 80/20 split with seed 42. `merge` and `model_label` use the same `dedup_key()`. Test reports
go to `models/results/<model>.json` with the `pipeline_version` they were scored under, and
`spam_common.metrics` (`make compare`) tabulates them, marking results from another version with `*`.

**Data workflow** (phone export → dataset), each step one file in, one new file out:
`extract` (adb dump) → `anonymize` (`*_anon.jsonl`) → `model-label MODEL=tfidf|fasttext|parsbert`
(`*_model.jsonl`) or `regex-label` before any model exists → manual review with `make label` → `merge`
into `data/sms_labeled.jsonl` (keeps a `.bak`). Step tools pick their output with
`spam_common.data.output_path()`, which writes `<input>_<suffix>.jsonl` and refuses to overwrite
without `--force` (`FORCE=1` in make). `model_label()` is shared by every model's `label`
command. A model passes it a function returning spam probabilities. Rows already in the dataset keep
their checked label with `model_score: null`. `model_score` is review-only (`UNCERTAIN = (0.3, 0.7)`
drives the GUI's filter and sort) and `merge` drops it. `merge` skips rows whose label is not
`ham`/`spam`.

**JSONL rows** look like `{"address": ..., "text": ..., "label": "ham"|"spam"|null, ...}`, and
phone exports carry extra fields. Data tools may change only `text` and `label` (plus the
review-only `model_score`). Every other field must be written back exactly as exported.

## Privacy

`data/` holds real personal SMS and `models/` is trained on it. Both are gitignored and must never
be committed, published or pasted into external services. Experiments on the data print counts
and metrics, never message text. The anonymizer is rule-based and does not catch person names or
street addresses.

The one trained artifact that is published is the TF-IDF weights `make export-android` writes
into NoSpam (`spam_model.json`, committed there because the app ships it). The `.joblib` and
everything else in `models/` stay local. Others can build NoSpam with those weights, or train
their own model with their own SMS; the dataset itself is never published.
