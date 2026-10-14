"""Export the TF-IDF + Linear SVM model and a parity fixture for the NoSpam app (core:ml).

Writes:
  src/main/assets/spam_model.json   vocabulary, idf, weights, bias and the `pipeline`
                                    (version + normalizer) the model was trained with
  src/test/resources/spam_fixture.json
                                    the core's battery inputs with every stage's output, the
                                    n-grams and the model score, for the app's parity tests

The text processing itself is not exported: the app builds the same Rust core
(core/spam-core, see docs/preprocessing-core-plan.md). The Kotlin side still expects the
old file names and pipeline v1, so what this writes is not shippable until Phase 3.

Usage:
    python export_android_assets.py /abs/path/to/NoSpam/core/ml [--model models/model-tfidf.joblib]
"""
import argparse
import json
import os
import sys

import joblib
import spam_core
from spam_common.data import MODELS_DIR, REPO_ROOT
from spam_common.pipeline import NORMALIZER, PIPELINE_VERSION, ngrams, preprocess_stages

BATTERY = REPO_ROOT / "core" / "spam-core" / "tests" / "cases" / "battery.json"


def export_model(pipe) -> dict:
    """The trained TfidfVectorizer + LinearSVC as plain data, enough to reproduce sklearn's
    score: counts of the core's n-grams -> sublinear tf -> idf -> L2 norm -> dot product + bias."""
    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    assert vectorizer.analyzer is ngrams, "the model was not trained with spam_common.pipeline.ngrams"
    return {
        "vocab": vectorizer.vocabulary_,
        "weights": clf.coef_[0].tolist(),
        "bias": float(clf.intercept_[0]),
        "classes": clf.classes_.tolist(),
        "idf": vectorizer.idf_.tolist(),
        "sublinearTf": bool(vectorizer.sublinear_tf),
        "norm": vectorizer.norm,
        "pipeline": {"version": PIPELINE_VERSION, "normalizer": NORMALIZER},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("module_dir", help="core/ml module dir (writes src/main/assets and src/test/resources)")
    ap.add_argument("--model", default=str(MODELS_DIR / "model-tfidf.joblib"), help="path to model-tfidf.joblib")
    args = ap.parse_args()

    out_assets = os.path.join(args.module_dir, "src", "main", "assets")
    out_tests = os.path.join(args.module_dir, "src", "test", "resources")
    os.makedirs(out_assets, exist_ok=True)
    os.makedirs(out_tests, exist_ok=True)

    pipe = joblib.load(args.model)
    model = export_model(pipe)

    battery = json.loads(BATTERY.read_text("utf-8"))
    cases = []
    for text in battery:
        stages = preprocess_stages(text)
        cases.append({
            "input": text,
            "anonymized": stages["anonymize"],
            "urlsMasked": stages["urls"],
            "numbersMasked": stages["numbers"],
            "output": stages["normalize"],
            "ngrams": spam_core.ngrams(stages["normalize"]),
            "score": float(pipe.decision_function([stages["normalize"]])[0]),
        })

    with open(os.path.join(out_assets, "spam_model.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=0)
    with open(os.path.join(out_tests, "spam_fixture.json"), "w", encoding="utf-8") as f:
        json.dump({"pipeline": model["pipeline"], "cases": cases}, f, ensure_ascii=False, indent=0)

    print("exported model to", out_assets, "- vocabulary", len(model["vocab"]))
    print("exported fixture to", out_tests, "- cases", len(cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
