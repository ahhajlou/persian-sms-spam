"""Remove sensitive data from SMS exports before labelling or training.

Sensitive values are replaced with typed placeholders ([CARD], [NATID],
[PASSWORD], ...) rather than deleted, so a classifier still sees what kind of
message it is (e.g. an OTP or a bank receipt). Rules only, no model: names and
street addresses are NOT detected.

    1. Normalize Persian/Arabic digits and letters.
    2. Formats: IBAN, bank cards (full and masked), emails, mobile/landline
       numbers, and national IDs (only bare 10-digit numbers that pass the
       check digit, so other 10-digit numbers are not mistaken for one).
    3. Keyword context: the token after رمز/کلمه عبور/نام کاربری/کد/code/...,
       balances and transaction amounts, account numbers.
    4. Any remaining run of 8+ digits becomes [NUMBER].

Phone-number senders in `address` become a stable pseudonym (PHONE_<hash>)
keyed by a salt in data/.anonymize_salt; service senders (e.g. "Hamta") stay.

Usage:
    python -m spam_common.anonymize input.jsonl [output.jsonl] [--force]

The output defaults to <input>_anon.jsonl and is never overwritten without --force.
"""
import argparse
import hashlib
import json
import re
import secrets
import sys
from collections import Counter
from pathlib import Path

from spam_common.data import DATA_DIR, load_jsonl, write_jsonl

SALT_FILE = DATA_DIR / ".anonymize_salt"
DROP_FIELDS = ("installId",)  # identifies the device

_TRANSLATE = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩يكى", "01234567890123456789یکی")


def normalize(text):
    return text.translate(_TRANSLATE)


# --- Check-digit validator --------------------------------------------------

def national_id_ok(s):
    d = [int(c) for c in s]
    if len(d) != 10 or len(set(d)) == 1:
        return False
    r = sum(d[i] * (10 - i) for i in range(9)) % 11
    return d[9] == (r if r < 2 else 11 - r)


# --- Rules ------------------------------------------------------------------

# Digits must not touch other digits; \b does not work between Persian letters and digits.
_NB, _NA = r"(?<!\d)", r"(?!\d)"

# (tag, pattern, validator) applied in order to the whole match.
FORMAT_RULES = [
    ("IBAN", re.compile(r"(?<![A-Za-z])IR\s?\d{2}(?:\s?\d){16,22}" + _NA, re.I), None),
    ("CARD", re.compile(_NB + r"\d{4}(?:[\s-]?\d{4}){3}" + _NA), None),
    # Masked card, e.g. 603799******1234 or 603799*1234; not USSD codes like *123*45#.
    ("CARD", re.compile(r"(?<![\d*#])\d{4,6}[\s-]?[*xX×]+[\s\-*xX×\d]{0,10}?\d{2,4}" + _NA), None),
    # Dotted account number, e.g. 123.4567.12345678.1
    ("ACCOUNT", re.compile(_NB + r"\d{3,4}\.\d{3,4}\.\d{6,10}\.\d{1,3}" + _NA), None),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.?[A-Za-z]{0,10}"), None),
    ("PHONE", re.compile(r"(?<![\d+])(?:\+98|0098|0)\s?9\d{2}[\s-]?\d{3}[\s-]?\d{4}" + _NA), None),
    ("PHONE", re.compile(_NB + r"0[1-8]\d{1,2}[\s-]?\d{7,8}" + _NA), None),
    ("NATID", re.compile(_NB + r"\d{10}" + _NA), national_id_ok),
]

# Text between a keyword and its value: no digits, newlines, placeholders or
# USSD codes (*123#, a spam signal rather than private data).
_FILLER = r"[^\d\n\[*#]{0,25}?"
_TOKEN = r"[A-Za-z0-9][A-Za-z0-9._@\-]"

# (tag, keywords, value pattern); the keyword is kept, the value replaced.
KEYWORD_RULES = [
    ("PASSWORD", r"رمز|کلمه\s?(?:ی\s)?عبور|پسورد|password|passcode|pin\s?code", _TOKEN + "{3,}"),
    ("USERNAME", r"نام\s?کاربری|شناسه\s?کاربری|username|user\s?name|user\s?id|login", _TOKEN + "{2,}"),
    ("CODE", r"کد|code|otp|پیگیری|رهگیری|مرجع|سفارش|مرسوله|@[\w.-]+\s*#", r"[A-Za-z0-9\-]*\d[A-Za-z0-9\-]*"),
    ("AMOUNT", r"مانده|موجودی|برداشت|واریز|مبلغ|balance", r"\d[\d,٬.]*"),
    ("ACCOUNT", r"حساب|سپرده|account", r"\d[\d.\-/]{3,}"),
]
_KEYWORD_RES = [
    (tag, re.compile(rf"((?<![A-Za-z])(?:{kw}){_FILLER})({value})", re.I)) for tag, kw, value in KEYWORD_RULES
]

LONG_NUMBER = re.compile(_NB + r"\d{8,}" + _NA)


def anonymize(text, counts=None):
    """Return `text` with sensitive values replaced; tally tags into `counts`."""
    counts = Counter() if counts is None else counts
    text = normalize(text)

    for tag, pattern, valid in FORMAT_RULES:
        def repl(m, tag=tag, valid=valid):
            if valid and not valid(m.group(0)):
                return m.group(0)
            counts[tag] += 1
            return f"[{tag}]"
        text = pattern.sub(repl, text)

    for tag, pattern in _KEYWORD_RES:
        def repl(m, tag=tag):
            counts[tag] += 1
            return f"{m.group(1)}[{tag}]"
        text = pattern.sub(repl, text)

    text, n = LONG_NUMBER.subn("[NUMBER]", text)
    counts["NUMBER"] += n
    return text


def load_salt():
    if not SALT_FILE.exists():
        SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SALT_FILE.write_text(secrets.token_hex(16), encoding="utf-8")
    return SALT_FILE.read_text(encoding="utf-8").strip()


def pseudonymize_address(address, salt):
    """Hash personal phone numbers; keep service senders and short codes."""
    plain = normalize(address or "")
    digits = re.sub(r"[\s\-+]", "", plain)
    if not (digits.isdigit() and len(digits) >= 10):
        return address
    canonical = "0" + digits[-10:]  # +98912..., 0098912..., 0912... -> same pseudonym
    return "PHONE_" + hashlib.sha256((salt + canonical).encode()).hexdigest()[:8]


def anonymize_row(row, salt, counts=None):
    row = {k: v for k, v in row.items() if k not in DROP_FIELDS}
    if row.get("text"):
        row["text"] = anonymize(row["text"], counts)
    if "address" in row:
        row["address"] = pseudonymize_address(row["address"], salt)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path, nargs="?")
    ap.add_argument("--force", action="store_true", help="overwrite the output file if it exists")
    args = ap.parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_anon.jsonl")
    if output.resolve() == args.input.resolve():
        sys.exit("Output must differ from input.")
    if output.exists() and not args.force:
        sys.exit(f"{output} already exists; pass --force to overwrite it.")

    salt = load_salt()
    counts = Counter()
    rows = [anonymize_row(row, salt, counts) for row in load_jsonl(args.input)]
    write_jsonl(rows, output)

    print(f"{len(rows)} messages -> {output}")
    for tag, n in counts.most_common():
        if n:
            print(f"  {tag:<9}{n:>6}")


if __name__ == "__main__":
    main()
