"""
Method #2: FastText supervised classifier
Usage:
    python fasttext_with_pretrained_vector.py train train.jsonl model.bin [path/to/cc.fa.300.vec]
    python fasttext_with_pretrained_vector.py test  model.bin test.jsonl
    python fasttext_with_pretrained_vector.py predict model.bin "متن پیامک"

If a pretrained vectors path is given to `train`, this trains BOTH a
from-scratch model and a pretrained-vector-initialized model on the same
training data, scores both on the shared test set (data/test.jsonl), and
prints a side-by-side comparison so you can see directly whether the
pretrained vectors actually help on your data.

To get pretrained Persian vectors:
    pip install fasttext
    python -c "import fasttext.util; fasttext.util.download_model('fa', if_exists='ignore')"
This downloads cc.fa.300.bin (several GB) in the current directory.
train_supervised's pretrainedVectors option needs the plain-text .vec
form, not the compressed .bin — if you only have the .bin, load it with
fasttext.load_model() and re-export word vectors to a .vec file, or look
for a Persian .vec download directly (search "cc.fa.300.vec").

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
from spam_common.data import TEST_FILE, load_texts_labels
from spam_common.pipeline import preprocess as pipeline_preprocess
from spam_common.metrics import evaluate as evaluate_preds

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


def train_one(train_txt, pretrained_vec_path=None, dim=50):
    kwargs = dict(
        input=train_txt,
        epoch=25,
        lr=0.5,          # safe at real data volume (>100 rows); drop to
                         # ~0.1 only if you see NaN on a tiny/toy dataset
        wordNgrams=2,    # bigrams help catch short spam phrases
        minCount=2,      # ignore singleton tokens once you have real volume
        loss="softmax",
        # fastText spawns `thread` training threads regardless of core
        # count; on constrained/containerized CPUs this has been known
        # to crash ("Floating point exception") from a race in the
        # shared-state update. Pinning it to the actual core count
        # avoids that.
        thread=max(1, os.cpu_count() or 1),
        verbose=2,
    )
    if pretrained_vec_path:
        # dim MUST match the pretrained vectors' dimension (300 for
        # Meta's cc.fa.300) or fastText will error out.
        kwargs["pretrainedVectors"] = pretrained_vec_path
        kwargs["dim"] = 300
    else:
        kwargs["dim"] = dim
    return fasttext.train_supervised(**kwargs)


def evaluate(model, name, X_test, y_test):
    preds = [model.predict(t)[0][0].replace("__label__", "") for t in X_test]
    return evaluate_preds(name, y_test, preds)


def train(data_path, model_path, pretrained_vec_path=None, test_path=TEST_FILE):
    t0 = time.time()
    print("Loading + preprocessing...")
    X_train, y_train = load_jsonl(data_path)
    # Both models below are scored on the shared test set, so the
    # comparison is fair and matches every other model in the repo.
    X_test, y_test = load_jsonl(test_path)
    print(f"  {len(X_train)} train / {len(X_test)} test rows in {time.time() - t0:.1f}s")

    train_txt = model_path + ".train.txt"
    write_fasttext_file(X_train, y_train, train_txt)

    print(f"\nTraining on {len(X_train)} rows...")
    t1 = time.time()
    model_scratch = train_one(train_txt)
    print(f"From-scratch training took {time.time() - t1:.1f}s")
    scores_scratch = evaluate(model_scratch, Path(model_path).stem, X_test, y_test)
    model_scratch.save_model(model_path)
    print(f"Saved to {model_path}")

    if pretrained_vec_path:
        print(f"\nTraining with pretrained vectors from {pretrained_vec_path}...")
        t2 = time.time()
        model_pretrained = train_one(train_txt, pretrained_vec_path=pretrained_vec_path)
        print(f"Pretrained-vector training took {time.time() - t2:.1f}s")
        pretrained_path = model_path.replace(".bin", "_pretrained.bin")
        scores_pretrained = evaluate(model_pretrained, Path(pretrained_path).stem, X_test, y_test)
        model_pretrained.save_model(pretrained_path)
        print(f"Saved to {pretrained_path}")

        print("\n=== Comparison ===")
        print(f"{'Metric':<12}{'Scratch':>10}{'Pretrained':>12}{'Delta':>10}")
        for key in ("accuracy", "f1_macro"):
            delta = scores_pretrained[key] - scores_scratch[key]
            print(f"{key:<12}{scores_scratch[key]:>10.3f}{scores_pretrained[key]:>12.3f}{delta:>+10.3f}")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


def test(model_path, data_path):
    model = fasttext.load_model(model_path)
    texts, labels = load_jsonl(data_path)
    evaluate(model, Path(model_path).stem, texts, labels)


def predict(model_path, text):
    model = fasttext.load_model(model_path)
    label, prob = model.predict(preprocess(text))
    print(f"label={label[0].replace('__label__', '')}  prob={prob[0]:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        pretrained = sys.argv[4] if len(sys.argv) > 4 else None
        train(sys.argv[2], sys.argv[3], pretrained_vec_path=pretrained)
    elif cmd == "test":
        test(sys.argv[2], sys.argv[3])
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
