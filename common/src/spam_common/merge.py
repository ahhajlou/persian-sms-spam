"""Add a manually checked file to the labelled dataset.

Rows whose label is not ham/spam (e.g. null, still unreviewed) are skipped, as
are messages already in the dataset (compared after preprocessing, like the
split). The previous dataset is kept as <dataset>.bak.

Usage:
    python -m spam_common.merge reviewed.jsonl [--into data/sms_labeled.jsonl]
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

from spam_common.data import LABELED_FILE, LABELS, load_jsonl, write_jsonl
from spam_common.split import dedup_key


def merge(existing, new):
    """Return (rows to add, Counter of skipped reasons)."""
    seen = {dedup_key(row["text"]) for row in existing}
    added, skipped = [], Counter()
    for row in new:
        label = (row.get("label") or "").lower()
        if label not in LABELS:
            skipped["unlabelled"] += 1
            continue
        key = dedup_key(row.get("text") or "")
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        added.append({**row, "label": label})
    return added, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--into", type=Path, default=LABELED_FILE)
    args = ap.parse_args()

    existing = load_jsonl(args.into) if args.into.exists() else []
    added, skipped = merge(existing, load_jsonl(args.input))
    if added:
        if args.into.exists():
            shutil.copy2(args.into, args.into.with_name(args.into.name + ".bak"))
        write_jsonl(existing + added, args.into)

    counts = dict(sorted(Counter(row["label"] for row in added).items()))
    print(f"added {len(added)} {counts} to {args.into} ({len(existing) + len(added)} rows now)")
    for reason, n in skipped.items():
        print(f"skipped {n} {reason}")


if __name__ == "__main__":
    main()
