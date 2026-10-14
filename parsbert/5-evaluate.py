import torch
from transformers import pipeline
from spam_common.data import TEST_FILE, clean_text, load_texts_labels
from spam_common.pipeline import preprocess

from config import FINAL_MODEL_DIR, MAX_LENGTH
from spam_common.metrics import evaluate

classifier = pipeline(
    "text-classification",
    model=str(FINAL_MODEL_DIR),
    tokenizer=str(FINAL_MODEL_DIR),
    device=0 if torch.cuda.is_available() else -1,
)

texts, labels = load_texts_labels(TEST_FILE)
# Same preprocessing and max_length as 2-fine-tune.py
results = classifier([clean_text(preprocess(t)) for t in texts], batch_size=32, truncation=True, max_length=MAX_LENGTH)
preds = [r["label"] for r in results]

evaluate(FINAL_MODEL_DIR.name, labels, preds)
