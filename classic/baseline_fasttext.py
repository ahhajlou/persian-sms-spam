"""
Method #2: FastText supervised classifier
Usage:
    python baseline_fasttext.py train data.jsonl model.bin
    python baseline_fasttext.py test  model.bin data.jsonl
    python baseline_fasttext.py predict model.bin "متن پیامک"

Note: fasttext's supervised trainer reads its own plain-text format
(one example per line: "__label__<label> <space-separated tokens>"),
not jsonl directly — this script does the jsonl -> fasttext conversion.
"""
import os
import sys
import json
import re
import time
import fasttext
from hazm import Normalizer, WordTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    from tqdm import tqdm
except ImportError:  # progress bar is a nicety, not a hard dependency
    def tqdm(iterable, **kwargs):
        return iterable

_normalizer = Normalizer()
# BUG FIX: hazm's module-level `word_tokenize(text)` convenience function
# calls `WordTokenizer().tokenize(text)` internally — i.e. it builds a
# brand new WordTokenizer (which loads a multi-MB lexicon file from disk)
# on every single call. Looping that over ~2000 rows means ~2000 cold
# lexicon loads, which is what looked like a hang. Instantiating it once
# here and reusing `.tokenize()` fixes it.
_word_tokenizer = WordTokenizer()
_url_re = re.compile(r"https?://\S+|www\.\S+")


def preprocess(text: str) -> str:
    text = _normalizer.normalize(text)
    text = _url_re.sub(" URLTOKEN ", text)
    text = text.replace("\n", " ")
    tokens = _word_tokenizer.tokenize(text)
    return " ".join(tokens)


def load_jsonl(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    for line in tqdm(lines, desc="Preprocessing", unit="msg"):
        row = json.loads(line)
        texts.append(preprocess(row["text"]))
        labels.append(row["label"])
    return texts, labels


def write_fasttext_file(texts, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            f.write(f"__label__{label} {text}\n")


def train(data_path, model_path):
    t0 = time.time()
    print("Loading + preprocessing...")
    texts, labels = load_jsonl(data_path)
    print(f"  {len(texts)} rows in {time.time() - t0:.1f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    train_txt = model_path + ".train.txt"
    write_fasttext_file(X_train, y_train, train_txt)

    print(f"Training on {len(X_train)} rows (fastText prints its own "
          f"progress bar below)...")
    t1 = time.time()
    model = fasttext.train_supervised(
        input=train_txt,
        epoch=25,
        lr=0.5,            # safe now at real data volume (>100 rows);
                           # drop to ~0.1 only if you see NaN on a
                           # very small/toy dataset
        wordNgrams=2,      # bigrams help catch short spam phrases
        minCount=2,        # ignore singleton tokens once you have real volume
        dim=50,
        loss="softmax",
        # fastText spawns `thread` training threads regardless of core
        # count; on constrained/containerized CPUs this has been known
        # to crash ("Floating point exception") from a race in the
        # shared-state update. Pinning it to the actual core count
        # avoids that.
        thread=max(1, os.cpu_count() or 1),
        verbose=2,
    )
    print(f"Training took {time.time() - t1:.1f}s")

    preds = [model.predict(t)[0][0].replace("__label__", "") for t in X_test]
    print("=== Held-out evaluation ===")
    print(classification_report(y_test, preds, digits=3))

    model.save_model(model_path)
    # Quantize for on-device deployment: shrinks the model (product
    # quantization on embeddings) with a small, usually acceptable,
    # accuracy hit. This is the version you'd actually ship to Android.
    t2 = time.time()
    model.quantize(input=train_txt, retrain=True)
    print(f"Quantization took {time.time() - t2:.1f}s")
    model.save_model(model_path.replace(".bin", "_quantized.ftz"))
    print(f"Saved full model to {model_path}")
    print(f"Saved quantized model to {model_path.replace('.bin', '_quantized.ftz')}")
    print(f"Total wall time: {time.time() - t0:.1f}s")


def test(model_path, data_path):
    model = fasttext.load_model(model_path)
    texts, labels = load_jsonl(data_path)
    preds = [model.predict(t)[0][0].replace("__label__", "") for t in texts]
    print(classification_report(labels, preds, digits=3))


def predict(model_path, text):
    model = fasttext.load_model(model_path)
    label, prob = model.predict(preprocess(text))
    print(f"label={label[0].replace('__label__', '')}  prob={prob[0]:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2], sys.argv[3])
    elif cmd == "test":
        test(sys.argv[2], sys.argv[3])
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)