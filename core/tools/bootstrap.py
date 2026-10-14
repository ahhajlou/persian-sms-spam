"""One-time bootstrap for the Rust core (Phase 1), run while the Python pipeline still exists.

    uv run --project classic python core/tools/bootstrap.py

Writes, from the Python implementation (the reference until Phase 2):
  spam-core/rules/anonymize.toml   the anonymizer's final regexes, rewritten for Rust
  spam-core/rules/urls.toml        URL and number masks
  spam-core/tests/cases/{numbers,normalize,ngrams}.json   expected output per stage
  spam-core/tests/cases/battery.json                      inputs for the golden files
anonymize.json and urls.json (hand-written regression cases) are not touched.
Delete this script when Phase 2 removes the Python rules.
"""
import json
import re
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "tools"))
sys.path.insert(0, str(CORE.parent / "classic"))

import export_android_assets as ref
from hazm import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer

from rx import to_rust
from spam_common import anonymize as an
from spam_common import pipeline as pl

RULES = CORE / "spam-core" / "rules"
CASES = CORE / "spam-core" / "tests" / "cases"


# The same strings as spam_common.anonymize._TRANSLATE (checked below).
TRANSLATE_SRC = "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩يكى"
TRANSLATE_DST = "01234567890123456789یکی"
assert an._TRANSLATE == str.maketrans(TRANSLATE_SRC, TRANSLATE_DST)


def lit(s):
    assert "'''" not in s
    return f"'''{s}'''"


def write_anonymize_toml():
    validators = {an.card_ok: "luhn", an.national_id_ok: "natid", None: None}
    out = [
        "# Anonymizer rules, generated from spam_common.anonymize (Python) and rewritten with\n"
        "# core/tools/rx.py. Edit this file now; the Python rules go away in Phase 2.\n"
        "# Explicit character classes only (no \\d \\w \\s \\b); a test enforces it.\n\n"
        "# Step 1: character translation (Persian/Arabic digits -> ASCII, Arabic yeh/kaf -> Persian).\n"
        f"translate_src = {json.dumps(TRANSLATE_SRC, ensure_ascii=False)}\n"
        f"translate_dst = {json.dumps(TRANSLATE_DST, ensure_ascii=False)}\n\n"
        "# Step 2: format rules, in this order. The whole match becomes [TAG] unless the\n"
        "# validator rejects it (then it stays and scanning continues after it).\n"
        "# validator: \"luhn\" (16 bare digits must pass Luhn; grouped ones always pass)\n"
        "#            \"natid\" (Iranian national ID check digit)\n"
    ]
    for tag, pattern, valid in an.FORMAT_RULES:
        out.append("\n[[format]]\n")
        out.append(f'tag = "{tag}"\n')
        out.append(f"pattern = {lit(to_rust(pattern.pattern, ignore_case=bool(pattern.flags & re.I)))}\n")
        if validators[valid]:
            out.append(f'validator = "{validators[valid]}"\n')
    out.append(
        "\n# Step 3: keyword rules. Group 1 (the keyword and filler) is kept, group 2 (the value)\n"
        "# becomes [TAG].\n"
    )
    for tag, pattern in an._KEYWORD_RES:
        out.append("\n[[keyword]]\n")
        out.append(f'tag = "{tag}"\n')
        out.append(f"pattern = {lit(to_rust(pattern.pattern, ignore_case=bool(pattern.flags & re.I)))}\n")
    out.append(
        "\n# Step 4: any digit run this long that is left becomes [NUMBER].\n"
        f"[long_number]\npattern = {lit(to_rust(an.LONG_NUMBER.pattern))}\n"
    )
    (RULES / "anonymize.toml").write_text("".join(out), "utf-8")


def write_urls_toml():
    out = (
        "# Link and number masks (pipeline steps 2 and 3), generated from spam_common.pipeline.\n"
        "# A match is replaced by `replacement`, with the spaces, so the token stays a word.\n\n"
        "[url]\n"
        f"pattern = {lit(to_rust(pl.URL_RE.pattern, ignore_case=True))}\n"
        'replacement = " URLTOKEN "\n\n'
        "# \\p{Nd}: every decimal digit, as Python's \\d did (the anonymizer only sees ASCII).\n"
        "[number]\n"
        "pattern = '''\\p{Nd}+'''\n"
        'replacement = " NUM_TOKEN "\n'
    )
    (RULES / "urls.toml").write_text(out, "utf-8")


def dump(name, obj):
    (CASES / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", "utf-8")


NUMBER_INPUTS = [
    "", "abc", "123", "۱۲۳", "٤٥٦", "mixed ۱2٣", "a1b22c333", "1,000,000 تومان", "2.5", "14:30",
    "１２３ fullwidth", "ساعت ۱۴ و 30 دقیقه", "[NUMBER] [CODE]", "0912", "x\n12\n",
]
NGRAM_INPUTS = [
    "", " ", "a", "ab", "abc", "abcd", "abcde", "ab cd", "  ab   cd  ", "A B", "Hello World",
    "سلام دنیا", "سلام\u200cدنیا", "می\u200cخواهم برم", "کد تایید 123", "URLTOKEN NUM_TOKEN",
    "tab\tsep", "new\nline", "cr\rlf", "nbsp\u00a0here", "thin\u2009space", "narrow\u202fnbsp",
    "ideographic\u3000space", "zwsp\u200bhere", "ΑΣ ΑΣΑ Σ", "İstanbul", "ÀÉÎ", "ß straße",
    "😀 emoji 👍🏽", "a\u0301b", "x" * 40, "ی", "یک دو", "[EMAIL] و [CODE]",
]
NORMALIZE_EXTRA = [
    "", " ", "  leading and trailing  ", "trailing space  \n", "a  \n b", "\n\nx\n\n\n\ny\n",
    "line one \n  \n  line two", "\t tab \t", "a\u2009b", "a\u202fb",
    "کلمه\u200c\u200cها", "کلمه \u200cها", "کلمه\u200c ها", "\u200c", "\u200cx\u200c", "x\u200c y",
    "کتاب‌ها", "کتاب ها", "بزرگ تر", "بزرگ ترین", "ایران گری", "خانه ام", "خانه ها", "خانه‌ی من",
    "می خوام", "نمی خوام", "می‌خوام", "میخوام", "نمیخوام", "می‌", "می",
    "\"نقل قول\"", "\" فاصله \"", "«a» ( b ) [ c ] { d }", "a،b؛c؟d!e:f.g", "a ، b ؛ c", "...", "wait ...",
    "1.5 و 2.5", "۱.۵", "3 . 4", "abc123", "123abc", "ی۱", "۱ی", "50%", "۵۰٪",
    "ـکشیدهـ", "كي", "ٱأإؤ", "ۀةە", "ٍَُِّْ", "ﷲ ﷺ", "\u0640\r\r", "x\ry",
    "سلامممم", "جایزههههه", "دددد", "خیلییییی خوبه", "ببببب", "اااا", "ممم",
    "(a)", "(سلام)", "سلام(خوب)", "ab(cd", "ab)cd", "a.b", "a. b", "a .b", "x:y", "x: y",
    "https: / / x. ir", "mixed English and فارسی 123 ۴۵۶",
]


def main():
    CASES.mkdir(parents=True, exist_ok=True)
    # The reference Normalizer has to exist before the module-level helpers are used.
    normalizer = Normalizer()
    res = ref.export_resources(normalizer)
    words_full, words_positive, verbs = ref.export_lexicons(normalizer)
    words_full, words_positive, verbs = set(words_full), set(words_positive), set(verbs)

    write_anonymize_toml()
    write_urls_toml()

    battery = ref.build_battery()
    dump("battery.json", battery)

    dump("numbers.json", [[t, pl.mask_numbers(t)] for t in NUMBER_INPUTS])

    # normalize: raw battery text, the numbers-stage text, and hand-picked inputs.
    inputs = []
    for t in NORMALIZE_EXTRA + battery + [pl.preprocess_stages(b)["numbers"] for b in battery]:
        if t not in inputs:
            inputs.append(t)
    empty = set()
    dump("normalize.json", {
        "rules": [[t, ref.normalize_ha(t, res, empty, empty, empty)] for t in inputs],
        "full": [[t, normalizer.normalize(t)] for t in inputs],
    })
    # The port with the real lists must equal real hazm (the old self-check).
    bad = sum(ref.normalize_ha(t, res, words_full, words_positive, verbs) != normalizer.normalize(t) for t in inputs)
    assert bad == 0, f"{bad} inputs where the reference port differs from hazm"

    analyzer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True).build_analyzer()
    dump("ngrams.json", [[t, analyzer(t)] for t in NGRAM_INPUTS + battery[:60]])
    print("normalize inputs:", len(inputs), "battery:", len(battery))


if __name__ == "__main__":
    main()
