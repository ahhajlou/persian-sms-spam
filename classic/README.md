# Classic models

Lightweight models that run on-device without a neural network runtime.

- `baseline_tfidf.py`: TF-IDF (char n-grams) + Linear SVM; the model the Android app uses
- `baseline_fasttext.py`: fastText, optionally with pretrained cc.fa.300 vectors; also writes a quantized `.ftz`
- `download_fasttext.py`: downloads the cc.fa.300 vectors
- `export_android_assets.py`: exports the TF-IDF model, hazm resources and the parity fixture to the Android module

Run them via the root Makefile (`make help`), or directly from this directory,
e.g. `uv run baseline_tfidf.py predict ../models/model-tfidf.joblib "متن پیامک"`.
Each script's docstring lists its commands; both models also have a `label` command
used by `make model-label`.
