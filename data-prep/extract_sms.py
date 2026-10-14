"""Extract messages from an `adb` SMS dump into unlabelled JSONL.

Export the dump from a connected phone with:
    adb shell content query --uri content://sms/ > data/sms_dump.txt

Usage:
    python extract_sms.py [data/sms_dump.txt] [output.jsonl] [--force]

The output defaults to data/sms_unlabeled.jsonl and is never overwritten
without --force. Next step: `make anonymize IN=<output>`.
"""
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

from spam_common.data import DATA_DIR, output_path, write_jsonl

# Each message starts with a "Row: <n> " marker.
ROW_RE = re.compile(r"\nRow: \d+ ")
# The body can contain commas and newlines; it is followed by service_center.
BODY_RE = re.compile(r"body=(.*?)(?=, service_center=)", re.DOTALL)
ADDRESS_RE = re.compile(r"address=([^,]*)")
# `date` rather than `date_sent`: date_sent is 0/NULL for most received
# messages, while date is always set and is what the phone sorts by.
DATE_RE = re.compile(r"\bdate=(\d+)")


def parse_dump(raw):
    records = []
    for row in ROW_RE.split(raw):
        body, address, date = BODY_RE.search(row), ADDRESS_RE.search(row), DATE_RE.search(row)
        if not (body and address and date):
            continue
        timestamp_ms = int(date.group(1))
        records.append({
            "address": address.group(1).strip(),
            "text": body.group(1).strip(),
            "timestamp": timestamp_ms,
            # Human-readable copy for spot checks; sort by `timestamp`.
            "datetime": datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat(),
            "label": None,
        })
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, nargs="?", default=DATA_DIR / "sms_dump.txt")
    ap.add_argument("output", type=Path, nargs="?", default=DATA_DIR / "sms_unlabeled.jsonl")
    ap.add_argument("--force", action="store_true", help="overwrite the output file if it exists")
    args = ap.parse_args()

    output = output_path(args.input, "", args.output, args.force)
    records = parse_dump(args.input.read_text(encoding="utf-8"))
    write_jsonl(records, output)
    print(f"Extracted {len(records)} messages from {args.input} -> {output}")


if __name__ == "__main__":
    main()
