"""
Method #1: TF-IDF (char n-grams) + Linear SVM
Usage:
    python baseline_tfidf.py train train.jsonl model.joblib
    python baseline_tfidf.py test  model.joblib test.jsonl
    python baseline_tfidf.py predict model.joblib "متن پیامک"
    python baseline_tfidf.py label model.joblib export.jsonl [--force]   (see spam_common.model_label)

For Android, export with export_android_assets.py (make export-android).
"""
import math
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from spam_common.data import load_texts_labels
from spam_common.pipeline import preprocess
from spam_common.metrics import evaluate
from spam_common.model_label import model_label


# --- Data loading -----------------------------------------------------------
def load_jsonl(path):
    texts, labels = load_texts_labels(path)
    return [preprocess(t) for t in texts], labels


def build_pipeline():
    # char_wb + n-gram range (2,4) is robust to Persian word-spacing
    # inconsistencies (half-space/zwnj vs full space) since it doesn't
    # depend on clean tokenization at all.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,          # lower for tiny demo dataset; use 2-3 on real data
        sublinear_tf=True,
    )
    clf = LinearSVC(class_weight="balanced")
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train(data_path, model_path):
    # Evaluation happens separately on the shared test set (`test` command).
    texts, labels = load_jsonl(data_path)
    pipe = build_pipeline()
    pipe.fit(texts, labels)
    print(f"Trained on {len(texts)} rows from {data_path}")

    joblib.dump(pipe, model_path)
    print(f"\nSaved model to {model_path}")


def test(model_path, data_path):
    pipe = joblib.load(model_path)
    texts, labels = load_jsonl(data_path)
    preds = pipe.predict(texts)
    evaluate(Path(model_path).stem, labels, list(preds))


def predict(model_path, text):
    pipe = joblib.load(model_path)
    x = [preprocess(text)]
    print(f"label={pipe.predict(x)[0]}  score={pipe.decision_function(x)[0]:.3f}")


def label(model_path, input_path, force=False):
    pipe = joblib.load(model_path)
    sign = 1 if pipe.classes_[1] == "spam" else -1

    def spam_scores(texts):
        if not texts:
            return []
        margins = pipe.decision_function([preprocess(t) for t in texts])
        # LinearSVC gives a margin, not a probability; a sigmoid maps it to
        # 0..1 (0.5 = the decision boundary) as a confidence score for review.
        return [1 / (1 + math.exp(-2 * sign * m)) for m in margins]

    model_label(input_path, spam_scores, Path(model_path).stem, force=force)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2], sys.argv[3])
    elif cmd == "test":
        test(sys.argv[2], sys.argv[3])
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3])
    elif cmd == "label":
        label(sys.argv[2], sys.argv[3], force="--force" in sys.argv[4:])
    else:
        print(__doc__)
