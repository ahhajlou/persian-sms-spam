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

Only `text` is changed; every other field of a row is kept as it is.

Usage:
    python -m spam_common.anonymize input.jsonl [output.jsonl] [--force]

The output defaults to <input>_anon.jsonl and is never overwritten without --force.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

from spam_common.data import load_jsonl, output_path, write_jsonl

_TRANSLATE = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩يكى", "01234567890123456789یکی")


def normalize(text):
    return text.translate(_TRANSLATE)


# --- Check-digit validators -------------------------------------------------

def national_id_ok(s):
    d = [int(c) for c in s]
    if len(d) != 10 or len(set(d)) == 1:
        return False
    r = sum(d[i] * (10 - i) for i in range(9)) % 11
    return d[9] == (r if r < 2 else 11 - r)


def card_ok(s):
    """Grouped 4-4-4-4 digits are always a card; 16 bare digits only if they pass
    Luhn, as every bank card does (16-digit tracking numbers usually don't)."""
    if not s.isdigit():
        return True
    total = 0
    for i, d in enumerate(int(c) for c in reversed(s)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


# --- Rules ------------------------------------------------------------------

# Digits must not touch other digits; \b does not work between Persian letters and digits.
_NB, _NA = r"(?<!\d)", r"(?!\d)"

# (tag, pattern, validator) applied in order to the whole match.
FORMAT_RULES = [
    ("IBAN", re.compile(r"(?<![A-Za-z])IR\s?\d{2}(?:\s?\d){16,22}" + _NA, re.I), None),
    ("CARD", re.compile(_NB + r"\d{4}(?:[\s-]?\d{4}){3}" + _NA), card_ok),
    # Masked card, e.g. 603799******1234, 603799*1234 or 6037-99**-****-1234;
    # not USSD codes like *123*45#.
    ("CARD", re.compile(r"(?<![\d*#])\d{4,6}[\s-]?\d{0,2}[*xX×]+[\s\-*xX×\d]{0,10}?\d{2,4}" + _NA), None),
    # Dotted account number, e.g. 123.4567.12345678.1
    ("ACCOUNT", re.compile(_NB + r"\d{3,4}\.\d{3,4}\.\d{6,10}\.\d{1,3}" + _NA), None),
    # Credentials in a URI before '@', e.g. the key in vless://<uuid>@host:443.
    ("SECRET", re.compile(r"(?<=://)[^\s/@]+(?=@)"), None),
    # Needs a dotted domain ending in letters, so device commands such as
    # #AMIR1827@HALL@ON# are not emails.
    ("EMAIL", re.compile(r"[\w.+-]+@(?:[\w-]+\.)+[A-Za-z]{2,}(?![\w-])"), None),
    ("PHONE", re.compile(r"(?<![\d+])(?:\+98|0098|0)\s?9\d{2}[\s-]?\d{3}[\s-]?\d{4}" + _NA), None),
    ("PHONE", re.compile(_NB + r"0[1-8]\d{1,2}[\s-]?\d{7,8}" + _NA), None),
    ("NATID", re.compile(_NB + r"\d{10}" + _NA), national_id_ok),
]


def _filler(stop=""):
    """Text allowed between a keyword and its value (at most 25 characters).

    Excludes digits, placeholders ('['), USSD codes (*123#, a spam signal
    rather than private data) and URL punctuation ('/?&'), so keywords inside
    links (login/..., Account.aspx?Req=...) do not match. The value may start
    the next line ("رمز عبور:\nAbc123"), but the filler never spans lines.
    `stop` lists words the filler must not cross.
    """
    char = r"[^\d\n\[*#/?&]"
    if stop:
        char = rf"(?:(?!{stop}){char})"
    return char + r"{0,25}?(?:[ \t]*\n[ \t]*)?"


# A value never starts inside a word. A password or username is never a link,
# a domain, or another keyword ("رمز اول کارت\nCode: 123456" is a code).
_START = r"(?<![A-Za-z0-9_])"
_TOKEN = _START + (
    r"(?!https?://|www\.|[\w-]+\.[A-Za-z]{2,}\b|(?:code|otp|pin|pass(?:word|code)?|user(?:name)?|login)\b)"
    r"[A-Za-z0-9][A-Za-z0-9._@\-]"
)

# (tag, keywords, value pattern, filler); the keyword is kept, the value replaced.
KEYWORD_RULES = [
    # Not رمزارز / رمز ارز (crypto) or رمزینه (a brand), common in ads.
    ("PASSWORD", r"رمز(?!\s?ارز|ینه)|کلمه\s?(?:ی\s)?عبور|گذرواژه|پسورد|password|passcode|pin\s?code",
     _TOKEN + "{3,}", _filler()),
    ("USERNAME", r"نام\s?کاربری|شناسه\s?کاربری|username|user\s?name|user\s?id",
     _TOKEN + "{2,}", _filler()),
    # Codes have at least 4 characters; shorter numbers are percentages, prices
    # or short codes (کد تخفیف 20, ارسال کد به 3000).
    ("CODE", r"کد|code|otp|پیگیری|رهگیری|مرجع|سفارش|مرسوله|@[\w.-]+\s*#",
     _START + r"(?=[A-Za-z0-9\-]{4})[A-Za-z0-9\-]*\d[A-Za-z0-9\-]*", _filler()),
    # Stops at حساب/کارت: in «برداشت از حساب 0123...» the number is an account.
    ("AMOUNT", r"مانده|موجودی|برداشت|واریز|مبلغ|balance", _START + r"\d[\d,٬.]*", _filler("حساب|کارت")),
    ("ACCOUNT", r"حساب|سپرده|account", _START + r"\d[\d.\-/]{3,}", _filler()),
]
# Keywords must not be part of a placeholder ([PASSWORD] would match
# "password" again) or of a longer Latin word, nor be followed by '.'.
_KEYWORD_RES = [
    (tag, re.compile(rf"((?<![\[A-Za-z])(?:{kw})(?![A-Za-z.]){filler})({value})", re.I))
    for tag, kw, value, filler in KEYWORD_RULES
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
