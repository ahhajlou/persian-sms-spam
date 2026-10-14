"""
Method #1: TF-IDF (char n-grams) + Linear SVM
Usage:
    python baseline_tfidf.py train train.jsonl model.joblib
    python baseline_tfidf.py test  model.joblib test.jsonl
    python baseline_tfidf.py predict model.joblib "متن پیامک"
    python baseline_tfidf.py export model.joblib model.json
"""
import sys
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from spam_common.data import load_texts_labels
from spam_common.pipeline import preprocess
from spam_common.metrics import evaluate


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
    pred = pipe.predict([preprocess(text)])[0]
    score = pipe.decision_function([preprocess(text)])[0]
    print(f"label={pred}  score={score:.3f}")

def export_to_mobile(model_path, json_path):
    # Load the trained pipeline
    pipe = joblib.load(model_path)
    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    
    # Extract the core components
    export_data = {
        "vocab": vectorizer.vocabulary_,          # Dict: ngram -> index
        "weights": clf.coef_[0].tolist(),         # SVM weights (for binary classification)
        "bias": float(clf.intercept_[0]),         # SVM bias/intercept
        "classes": clf.classes_.tolist()          # e.g., ["ham", "spam"]
    }
    
    # Save to JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False)
    
    print(f"✅ Successfully exported to {json_path}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2], sys.argv[3])
    elif cmd == "test":
        test(sys.argv[2], sys.argv[3])
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3])
    elif cmd == "export":
        export_to_mobile(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
