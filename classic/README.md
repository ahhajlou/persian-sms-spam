# Classic models

Lightweight baselines that run on-device without a neural network runtime.

```bash
# TF-IDF (char n-grams) + Linear SVM
uv run baseline_tfidf.py train ../data/sms_labeled.jsonl ../models/model-tfidf.joblib
uv run baseline_tfidf.py test  ../models/model-tfidf.joblib ../data/sms_labeled.jsonl
uv run baseline_tfidf.py predict ../models/model-tfidf.joblib "متن پیامک"

# fastText (from scratch, or with pretrained cc.fa.300 vectors)
uv run baseline_fasttext.py train ../data/sms_labeled.jsonl ../models/model-fasttext.bin
uv run download_fasttext.py
uv run fasttext_with_pretrained_vector.py train ../data/sms_labeled.jsonl ../models/model-fasttext.bin cc.fa.300.vec  # needs the .vec form, see docstring

# Export TF-IDF model + hazm normalizer to an Android module
uv run export_android_assets.py path/to/core/ml
```
