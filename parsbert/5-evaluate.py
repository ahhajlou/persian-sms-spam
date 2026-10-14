import torch
from transformers import pipeline
from spam_common.data import MODELS_DIR, TEST_FILE, clean_text, load_texts_labels
from spam_common.pipeline import preprocess
from spam_common.metrics import evaluate

MODEL_DIR = str(MODELS_DIR / "parsbert_spam_final")

classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    device=0 if torch.cuda.is_available() else -1,
)

texts, labels = load_texts_labels(TEST_FILE)
# Same preprocessing and max_length as 2-fine-tune.py
results = classifier([clean_text(preprocess(t)) for t in texts], batch_size=32, truncation=True, max_length=128)
preds = [r["label"] for r in results]

evaluate("parsbert_spam_final", labels, preds)
