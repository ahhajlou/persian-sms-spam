"""
Method #2: FastText supervised classifier
Usage:
    python baseline_fasttext.py train train.jsonl model.bin [cc.fa.300.vec]
    python baseline_fasttext.py test  model.bin test.jsonl
    python baseline_fasttext.py predict model.bin "متن پیامک"

`train` also writes a quantized <model>_quantized.ftz for Android.

Optional pretrained vectors: pass Meta's Persian cc.fa.300 word vectors to
initialise the embeddings (dim becomes 300). Training needs the plain-text
.vec form, not the compressed .bin; `python download_fasttext.py` downloads
cc.fa.300.bin (several GB). Train once with and once without, then
`make compare` shows whether the vectors help on this data.

Note: fasttext's supervised trainer reads its own plain-text format
(one example per line: "__label__<label> <space-separated tokens>"),
not jsonl directly — this script does the jsonl -> fasttext conversion.
"""
import os
import sys
import time
from pathlib import Path

import fasttext
from hazm import WordTokenizer
from spam_common.data import load_texts_labels
from spam_common.pipeline import preprocess as pipeline_preprocess
from spam_common.metrics import evaluate

try:
    from tqdm import tqdm
except ImportError:  # progress bar is a nicety, not a hard dependency
    def tqdm(iterable, **kwargs):
        return iterable

_word_tokenizer = WordTokenizer()


def preprocess(text: str) -> str:
    # Shared pipeline (anonymize, URL/number masks, hazm), then fastText's
    # own tokenization: one line per example, space-separated tokens.
    text = pipeline_preprocess(text).replace("\n", " ")
    return " ".join(_word_tokenizer.tokenize(text))


def load_jsonl(path):
    texts, labels = load_texts_labels(path)
    return [preprocess(t) for t in tqdm(texts, desc="Preprocessing", unit="msg")], labels


def write_fasttext_file(texts, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            f.write(f"__label__{label} {text}\n")


def train(data_path, model_path, pretrained_vec_path=None):
    t0 = time.time()
    print("Loading + preprocessing...")
    texts, labels = load_jsonl(data_path)
    print(f"  {len(texts)} rows in {time.time() - t0:.1f}s")

    # Evaluation happens separately on the shared test set (`test` command).
    train_txt = model_path + ".train.txt"
    write_fasttext_file(texts, labels, train_txt)

    print(f"Training on {len(texts)} rows (fastText prints its own "
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
        # dim must match the pretrained vectors (300 for cc.fa.300).
        dim=300 if pretrained_vec_path else 50,
        pretrainedVectors=pretrained_vec_path or "",
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
    evaluate(Path(model_path).stem, labels, preds)


def predict(model_path, text):
    model = fasttext.load_model(model_path)
    t2 = time.time()
    label, prob = model.predict(preprocess(text))
    print(f"Prediction took {time.time() - t2:.1f}s")
    print(f"label={label[0].replace('__label__', '')}  prob={prob[0]:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "test":
        test(sys.argv[2], sys.argv[3])
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
