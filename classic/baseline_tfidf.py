"""
Method #1: TF-IDF (char n-grams) + Linear SVM
Usage:
    python baseline_tfidf.py train data.jsonl model.joblib
    python baseline_tfidf.py test  model.joblib data.jsonl
    python baseline_tfidf.py predict model.joblib "متن پیامک"
"""
import sys
import json
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Persian-specific normalization ---------------------------------------
# hazm handles Arabic->Persian char normalization (ي->ی, ك->ک), digit
# normalization, and extra whitespace/diacritics. This matters a lot more
# for char n-grams than for word-level features, since inconsistent
# characters fragment the n-gram vocabulary.
from hazm import Normalizer

_normalizer = Normalizer()
_url_re = re.compile(
    r"(?:https?://[^\s]+)"
    r"|(?:www\.[^\s]+)"
    r"|(?:\b[a-zA-Z0-9][a-zA-Z0-9\-]*(?:\.[a-zA-Z0-9\-]+)+(?:/[^\s]*)?)"
)


def preprocess(text: str) -> str:
    # Mask URLs BEFORE normalizing: hazm's normalize() inserts spaces
    # around punctuation (e.g. "https://x.ir" -> "https: / / x. ir"),
    # which breaks the URL apart before a regex could ever match it as
    # one token. Masking first, then normalizing, keeps URLTOKEN intact
    # and collapses domain-specific text into one generic signal instead
    # of the model memorizing individual spam domains.
    text = _url_re.sub(" URLTOKEN ", text)
    text = re.sub(r"\d+", " NUM_TOKEN ", text)
    text = _normalizer.normalize(text)
    return text


# --- Data loading -----------------------------------------------------------
def load_jsonl(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(preprocess(row["text"]))
            labels.append(row["label"])
    return texts, labels


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
    texts, labels = load_jsonl(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print("=== Held-out evaluation ===")
    print(classification_report(y_test, preds, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    labels_sorted = sorted(set(labels))
    print(labels_sorted)
    print(confusion_matrix(y_test, preds, labels=labels_sorted))

    joblib.dump(pipe, model_path)
    print(f"\nSaved model to {model_path}")


def test(model_path, data_path):
    pipe = joblib.load(model_path)
    texts, labels = load_jsonl(data_path)
    preds = pipe.predict(texts)
    print(classification_report(labels, preds, digits=3))


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
