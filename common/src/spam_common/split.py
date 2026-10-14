"""Build the fixed train/test split that every model trains and is scored on.

Usage:
    python -m spam_common.split [labeled.jsonl] [--test-size 0.2] [--seed 42]
"""
import argparse
import random
from collections import Counter, defaultdict

from spam_common.data import LABELED_FILE, TEST_FILE, TRAIN_FILE, clean_text, load_jsonl, write_jsonl


def make_split(rows, test_size=0.2, seed=42):
    # Deduplicate first: the same SMS is often received many times, and a copy
    # in both train and test would inflate every model's score.
    unique = {}
    for row in rows:
        unique.setdefault(clean_text(row["text"]), row)

    # Stratify by label so test keeps the same ham/spam ratio as the data.
    by_label = defaultdict(list)
    for row in unique.values():
        by_label[row["label"].lower()].append(row)

    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        n_test = round(len(group) * test_size)
        test += group[:n_test]
        train += group[n_test:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test, len(rows) - len(unique)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default=LABELED_FILE)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_jsonl(args.source)
    train, test, n_dupes = make_split(rows, args.test_size, args.seed)
    write_jsonl(train, TRAIN_FILE)
    write_jsonl(test, TEST_FILE)

    count = lambda part: dict(sorted(Counter(r["label"].lower() for r in part).items()))
    print(f"{len(rows)} rows from {args.source}, {n_dupes} duplicates dropped")
    print(f"train: {len(train):>5}  {count(train)}  -> {TRAIN_FILE}")
    print(f"test:  {len(test):>5}  {count(test)}  -> {TEST_FILE}")


if __name__ == "__main__":
    main()
