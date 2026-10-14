"""Remove sensitive data from SMS exports before labelling or training.

Sensitive values are replaced with typed placeholders ([CARD], [NATID],
[PASSWORD], ...) rather than deleted, so a classifier still sees what kind of
message it is (e.g. an OTP or a bank receipt). Rules only, no model: names and
street addresses are NOT detected.

    1. Normalize Persian/Arabic digits and letters.
    2. Formats: IBAN, bank cards (full and masked), credentials in URIs,
       emails, mobile/landline numbers, and national IDs (only bare 10-digit
       numbers that pass the check digit, so other 10-digit numbers are not
       mistaken for one).
    3. Keyword context: the token after رمز/کلمه عبور/نام کاربری/کد/code/...,
       balances and transaction amounts, account numbers.
    4. Any remaining run of 8+ digits becomes [NUMBER].

The rules live in core/spam-core/rules/anonymize.toml (Rust core).

Only `text` is changed; every other field of a row is kept as it is.

Usage:
    python -m spam_common.anonymize input.jsonl [output.jsonl] [--force]

The output defaults to <input>_anon.jsonl and is never overwritten without --force.
"""
import argparse
from collections import Counter
from pathlib import Path

import spam_core
from spam_core import card_ok, national_id_ok  # noqa: F401  (re-exported for the tests)

from spam_common.data import load_jsonl, output_path, write_jsonl


def anonymize(text, counts=None):
    """Return `text` with sensitive values replaced; tally tags into `counts`.

    The rules are in core/spam-core/rules/anonymize.toml. A rule that hit the regex
    backtracking limit is skipped and tallied as LIMIT_HIT (never for real messages).
    """
    counts = Counter() if counts is None else counts
    text, tags, limit_hits = spam_core.anonymize(text)
    counts.update(tags)
    if limit_hits:
        counts["LIMIT_HIT"] += limit_hits
    return text


def anonymize_row(row, counts=None):
    """Anonymize `text` only; all other fields are left untouched."""
    if row.get("text"):
        row = {**row, "text": anonymize(row["text"], counts)}
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path, nargs="?")
    ap.add_argument("--force", action="store_true", help="overwrite the output file if it exists")
    args = ap.parse_args()

    output = output_path(args.input, "anon", args.output, args.force)
    counts = Counter()
    rows = [anonymize_row(row, counts) for row in load_jsonl(args.input)]
    write_jsonl(rows, output)

    print(f"{len(rows)} messages -> {output}")
    for tag, n in counts.most_common():
        if n:
            print(f"  {tag:<9}{n:>6}")


if __name__ == "__main__":
    main()
