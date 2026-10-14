"""First-round labels for a phone export with ParsBERT (see spam_common.model_label).

Usage: python 6-label.py export.jsonl [--force]
"""
import sys

import torch
from transformers import pipeline
from spam_common.data import clean_text
from spam_common.model_label import model_label
from spam_common.pipeline import preprocess

from config import FINAL_MODEL_DIR, MAX_LENGTH

classifier = pipeline(
    "text-classification",
    model=str(FINAL_MODEL_DIR),
    tokenizer=str(FINAL_MODEL_DIR),
    device=0 if torch.cuda.is_available() else -1,
)


def spam_scores(texts):
    if not texts:
        return []
    # Same preprocessing and max_length as 2-fine-tune.py; top_k=None returns
    # every label's probability.
    results = classifier([clean_text(preprocess(t)) for t in texts], batch_size=32,
                         truncation=True, max_length=MAX_LENGTH, top_k=None)
    return [next(r["score"] for r in result if r["label"] == "spam") for result in results]


if __name__ == "__main__":
    model_label(sys.argv[1], spam_scores, FINAL_MODEL_DIR.name, force="--force" in sys.argv[2:])
