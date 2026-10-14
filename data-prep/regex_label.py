"""First-round labelling with SpamBlocker regex rules, before a manual check.

A message is labelled "spam" if any blacklist rule matches its text, otherwise
"ham". Review the output with `make label DATA=<output>`.

Usage:
    python regex_label.py input.jsonl [output.jsonl] [--rules data/regex_rules.json] [--force]

The output defaults to <input>_regex.jsonl next to the input, and an existing
file is never overwritten unless --force is given.
"""
import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_RULES = Path(__file__).resolve().parents[1] / "data" / "regex_rules.json"


def load_rules(path):
    with open(path, encoding="utf-8") as f:
        rules = json.load(f)["rules"]
    # re.DOTALL so '.*' also matches across newlines in multi-line SMS.
    return [re.compile(rule["pattern"], re.DOTALL) for rule in rules if rule.get("isBlacklist", True)]


def label(rows, rules):
    for row in rows:
        text = row.get("text") or ""
        row["label"] = "spam" if any(rule.search(text) for rule in rules) else "ham"
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path, nargs="?")
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    ap.add_argument("--force", action="store_true", help="overwrite the output file if it exists")
    args = ap.parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_regex.jsonl")
    if output.resolve() == args.input.resolve():
        sys.exit("Output must differ from input.")
    if output.exists() and not args.force:
        sys.exit(f"{output} already exists; pass --force to overwrite it.")

    rules = load_rules(args.rules)
    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    label(rows, rules)

    with open(output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_spam = sum(row["label"] == "spam" for row in rows)
    print(f"{len(rules)} rules, {len(rows)} messages: {n_spam} spam, {len(rows) - n_spam} ham -> {output}")


if __name__ == "__main__":
    main()
