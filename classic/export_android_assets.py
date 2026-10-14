"""Export hazm 0.12 normalization resources + a parity fixture for the NoSpam
Android classifier (core:ml).

Generates, from the installed hazm in this venv:
  - hazm_normalizer.json      all regex tables/char maps used by Normalizer()
  - hazm_words.txt            full Persian word dictionary (membership checks)
  - hazm_words_positive.txt   words with rank > 0 (token_spacing rank gate)
  - hazm_verbs.txt            conjugated verb forms (seperate_mi/token_spacing)
  - hazm_fixture.json         curated + deterministic-fuzz inputs -> real hazm
                              output, consumed by the Kotlin JVM parity test

Then SELF-VERIFIES a faithful Python port of the same algorithm (consuming only
the exported data, mirroring what the Kotlin port will do) against real hazm
over a large fixed-seed battery. Exits non-zero on any mismatch.

Usage:
    python export_android_assets.py /abs/path/to/NoSpam/core/ml
"""
import argparse
import json
import os
import random
import re
import sys

import joblib
from hazm import Normalizer, WordTokenizer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import baseline_tfidf  # noqa: E402  (reuses _url_re / digit mask / preprocess)


# ---------------------------------------------------------------------------
# Resource extraction from the installed hazm
# ---------------------------------------------------------------------------

def parse_replacement(template: str) -> list:
    """Parse a Python re.sub replacement template into a portable part list.

    Parts are either {"l": <literal string>} or {"g": <group index>}.
    hazm only ever uses '\\1'..'\\9' group references plus literal chars; any
    other backslash sequence is an assertion failure so a silent mis-port is
    impossible.
    """
    parts = []
    i = 0
    n = len(template)
    if template == "":
        return []
    while i < n:
        c = template[i]
        if c == "\\" and i + 1 < n and template[i + 1].isdigit():
            j = i + 1
            while j < n and template[j].isdigit():
                j += 1
            parts.append({"g": int(template[i + 1:j])})
            i = j
        else:
            j = i
            while j < n and not (template[j] == "\\" and j + 1 < n and template[j + 1].isdigit()):
                j += 1
            if j > i:
                parts.append({"l": template[i:j]})
            i = j
    bad = re.findall(r"\\[^0-9]", template)
    assert not bad, f"unexpected backslash escape {bad!r} in replacement {template!r}"
    return parts


def export_resources(normalizer: Normalizer) -> dict:
    from hazm.constants import (
        AFFIX_SPACING_PATTERNS,
        DIACRITICS_PATTERNS,
        EXTRA_SPACE_PATTERNS,
        NUMBERS_DST,
        NUMBERS_SRC,
        PERSIAN_STYLE_PATTERNS,
        PUNCTUATION_SPACING_PATTERNS,
        SPECIAL_CHARS_PATTERNS,
        SUFFIXES,
        TRANSLATION_DST,
        TRANSLATION_SRC,
        UNICODE_REPLACEMENTS,
    )

    assert len(TRANSLATION_SRC) == len(TRANSLATION_DST)
    assert len(NUMBERS_SRC) == len(NUMBERS_DST)

    def plist(rows):
        return [{"p": p, "r": parse_replacement(r)} for p, r in rows]

    letters = normalizer.repeated_chars_pattern.pattern
    letters = letters[letters.index("["): letters.index("]") + 1]
    tokenizer = WordTokenizer(join_verb_parts=False)

    return {
        "translation_src": TRANSLATION_SRC,
        "translation_dst": TRANSLATION_DST,
        "numbers_src": NUMBERS_SRC,
        "numbers_dst": NUMBERS_DST,
        "suffixes": sorted(SUFFIXES),
        "persian_style": plist(PERSIAN_STYLE_PATTERNS),
        "diacritics": plist(DIACRITICS_PATTERNS),
        "extra_space": plist(EXTRA_SPACE_PATTERNS),
        "affix_spacing": plist(AFFIX_SPACING_PATTERNS),
        "punctuation_spacing": plist(PUNCTUATION_SPACING_PATTERNS),
        "special_chars": plist(SPECIAL_CHARS_PATTERNS),
        "unicode_replacements": [[o, n] for o, n in UNICODE_REPLACEMENTS],
        "tokenize_pattern": tokenizer.pattern.pattern,
        "repeat_chars_pattern": normalizer.repeated_chars_pattern.pattern,
        "repeat_more_than_two_pattern": normalizer.more_than_two_repeat_pattern.pattern,
        "seperate_mi_pattern": "\\bن?می" + letters + "+",
    }


def export_lexicons(normalizer: Normalizer):
    words = normalizer._tokenizer.words  # last-wins dict: word -> (rank, tags)
    words_full = sorted(words.keys())
    words_positive = sorted(k for k, v in words.items() if v[0] > 0)
    verbs = sorted(normalizer._verbs)
    return words_full, words_positive, verbs


def expand(parts: list, m) -> str:
    out = []
    for p in parts:
        if "l" in p:
            out.append(p["l"])
        else:
            g = m.group(p["g"]) or ""
            out.append(g)
    return "".join(out)


def apply(patterns: list, text: str) -> str:
    for rule in patterns:
        rx = re.compile(rule["p"])
        text = rx.sub(lambda m: expand(rule["r"], m), text)
    return text


def translate_charmap(text: str, src: str, dst: str) -> str:
    return text.translate(str.maketrans(src, dst))


def tokenize_ha(text: str, token_pattern: str) -> list:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(token_pattern, lambda m: " " + (m.group(1) or "") + " ", text)
    return [w for w in text.split(" ") if w]


def token_spacing_ha(tokens: list, words_full, words_positive, verbs, suffixes) -> list:
    result = []
    for t, token in enumerate(tokens):
        joined = False
        if result:
            token_pair = result[-1] + "\u200c" + token
            if words_full and (
                (verbs and token_pair in verbs)
                or (token_pair in words_positive)
            ):
                joined = True
                if (
                    t < len(tokens) - 1
                    and verbs
                    and token + "_" + tokens[t + 1] in verbs
                ):
                    joined = False
            elif words_full and token in suffixes and result[-1] in words_full:
                joined = True
        if joined:
            result[-1] = token_pair
        else:
            result.append(token)
    return result


def decrease_repeated_ha(text, words_full, repeat_chars, repeat_more) -> str:
    matches = list(re.compile(repeat_chars).finditer(text))
    for m in reversed(matches):
        word = m.group(0)
        if words_full and word not in words_full:
            no_repeat = re.compile(repeat_more).sub(r"\1", word)
            two_repeat = re.compile(repeat_more).sub(r"\1\1", word)
            if (no_repeat in words_full) != (two_repeat in words_full):
                r = no_repeat if no_repeat in words_full else two_repeat
            else:
                r = two_repeat
            text = text[: m.start()] + text[m.start(): m.end()].replace(word, r) + text[m.end():]
    return text


def seperate_mi_ha(text, verbs, seperate_mi_pattern) -> str:
    def repl(m):
        g0 = m.group(0)
        if g0.startswith("نمی"):
            cand = "نمی" + "\u200c" + g0[3:]
        elif g0.startswith("می"):
            cand = "می" + "\u200c" + g0[2:]
        else:
            cand = g0
        return cand if (verbs and cand in verbs) else g0

    return re.sub(seperate_mi_pattern, repl, text)


def normalize_ha(
    text, res, words_full, words_positive, verbs,
) -> str:
    text = translate_charmap(text, res["translation_src"], res["translation_dst"])
    text = apply(res["persian_style"], text)
    text = translate_charmap(text, res["numbers_src"], res["numbers_dst"])
    text = apply(res["diacritics"], text)

    text = apply(res["extra_space"], text)
    lines = text.split("\n")
    result = []
    for line in lines:
        if not line.strip():
            result.append(line)
            continue
        tokens = tokenize_ha(line, res["tokenize_pattern"])
        spaced = token_spacing_ha(tokens, words_full, words_positive, verbs, set(res["suffixes"]))
        result.append(" ".join(spaced))
    text = "\n".join(result)
    text = apply(res["affix_spacing"], text)
    text = apply(res["punctuation_spacing"], text)

    for old, new in res["unicode_replacements"]:
        text = text.replace(old, new)
    text = apply(res["special_chars"], text)

    text = decrease_repeated_ha(text, words_full, res["repeat_chars_pattern"], res["repeat_more_than_two_pattern"])
    text = seperate_mi_ha(text, verbs, res["seperate_mi_pattern"])
    return text


def preprocess_ha(text, res, words_full, words_positive, verbs) -> str:
    text = baseline_tfidf._url_re.sub(" URLTOKEN ", text)
    text = re.sub(r"\d+", " NUM_TOKEN ", text)
    return normalize_ha(text, res, words_full, words_positive, verbs)


def build_battery(seed: int = 20260910) -> list:
    curated = [
        "",
        "سلام دوست عزیز",
        "You won a prize click here",
        "برنده جایزه شدی روی لینک کلیک کن",
        "لینک واریز جایزه: https://bit.ly/xyz123",
        "کارت بانکی شما مسدود شد. برای رفع مسدودی با شماره ۰۹۱۲۳۴۵۶۷۸۹ تماس بگیرید",
        "مشتری گرامی، سامانه شاپرک ۱۰۰۰۰ تومان به حساب شما واریز کرد",
        "کد تخفیف شما 23456 است",
        "می‌خواهم بدانم فردا می‌آیی یا نه",
        "بـِه زودی تماس میگیرم",
        "جایزه ده میلیون تومانی برنده شدی!!!! برای دریافت با ما تماس بگیر",
        "کد تایید شما: ۱۲۳۴۵",
        "promo: 50% off today, buy now",
        "طرفدار رئال هستم و فوتبال تماشا می‌کنم",
        "این یک پیام کمپین تبلیغاتی است برای فروشگاه ما",
        "کارت بانگی شما قفل شده برای فعالسازی کد ۵۶۷ را ارسال کنید",
        "میخوام بدونم فردا میای یا نه؟!",
        "نمی‌خوام، ممنون.",
        "کتاب‌های من روی میز هست",
        "بهترین ها در این فروشگاه",
        "ارزان‌ترین قیمت را بزنید",
        "سلامممم به همه",
        "جایزههههه ۱,۰۰۰,۰۰۰ تومان!!!!!",
        "فروشگاه اینترنتی ما: تخفیف فوق‌العاده ۵۰٪",
        "با 50% تخفیف فقط امروز",
        "سرویس پیامک بانک: عملیات شما موفق بود کد رهگیری ۵۷۳۹",
        "هم اکنون ثبت نام کن",
        "مبل و یخچال میفروشم قیمت مناسب است تماس بگیرید ۰۹۱۹۰۰۰۰۰۰۰",
        "ارسال رایگان به سراسر کشور قیمت فقط ۹۹۰۰۰ تومان",
        "سایتgoogle.com را ببینید",
        "www.example.org/fa/page?x=1 و https://t.co/abc هم",
        "خط اول\nخط دوم\n\nخط چهارم",
        "a\u200cbc", "اب\u200cجد", "می\u200cخواهم", "می\u200cخواستم",
        "\u0640kashida\u0640", "واژه\u0640ای", "نیم\u200cفاصله",
        "«نقل قول» و \"دوگانه\"",
        "سلام، خوبی؟؛!",
        "۵۰٪", "50%", "abc.ir", "۱۲۳", "123", "mixed ۱۲3",
        "تکرار دددددرستی",
        "میز", "میزی", "می دهیم", "نمی دهم", "بفرمایید",
        "تلفن‌ها", "تلفن های من", "فیلما", "فیلمها",
        "پیامک با خط\nتازه\rقدیمی",
        "بدون آنکه بدانی ...",
        "https://example.com", "http://sub.domain.org/x/y",
        "عدد ۱٫۵ و 2.5 در جمله",
        "«گیومه» « و فاصله »",
        "قلبB و رنگها",
        "این پیام236 دارای شماره12است",
        "می‌شود", "میشود", "خواهد شد", "خواهدشد",
        "\u200e\u200fRTL\u200e\u200fآزمون",
    ]
    rng = random.Random(seed)
    letters = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیینمی"
    latin = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    punct = "،؛؟!:.»«()[]{}%\"'\u200c\u200f\u0640"
    digits = "0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩"
    diac = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0640\u0670"
    toks = [letters, latin, punct, digits, " ", "\n", "http://x.ir", "www.t.com/a", "می", "نمی", "ها", "های", "ترین", "گری", "ام", "اش", "یم", "ند"]
    for _ in range(400):
        parts = []
        for _ in range(rng.randint(1, 12)):
            pool = toks[rng.randrange(len(toks))]
            parts.append(rng.choice(pool) + rng.choice(pool))
        s = "".join(parts)
        s = s[: rng.randint(1, 60)]
        # sprinkle in intentional stress patterns
        if rng.random() < 0.3:
            s = s.replace("می", "می\u200c")
        if rng.random() < 0.2:
            s = s.rstrip() + "ی" + diac[rng.randrange(len(diac))]
        curated.append(s)
    return curated


def export_model(pipe) -> dict:
    """Full serialization of the trained TfidfVectorizer+LinearSVC pipeline so
    the Kotlin classifier can reproduce sklearn's transform + decision exactly
    (sublinear_tf -> idf -> L2 row norm -> coef dot + intercept)."""
    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    return {
        "vocab": vectorizer.vocabulary_,
        "weights": clf.coef_[0].tolist(),
        "bias": float(clf.intercept_[0]),
        "classes": clf.classes_.tolist(),
        "idf": vectorizer.idf_.tolist(),
        "ngramRange": list(vectorizer.ngram_range),
        "sublinearTf": bool(vectorizer.sublinear_tf),
        "norm": vectorizer.norm,
        "lowercase": bool(vectorizer.lowercase),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("module_dir", help="core/ml module dir (writes src/main/assets and src/test/resources)")
    ap.add_argument("--model", default=None, help="path to model-tfidf.joblib (default: ../models/model-tfidf.joblib)")
    args = ap.parse_args()

    out_assets = args.module_dir + "/src/main/assets"
    out_tests = args.module_dir + "/src/test/resources"
    os.makedirs(out_assets, exist_ok=True)
    os.makedirs(out_tests, exist_ok=True)
    model_path = args.model or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "model-tfidf.joblib")

    print("loading hazm lexicons (one-time, slow)...")
    normalizer = Normalizer()
    res = export_resources(normalizer)
    words_full, words_positive, verbs = export_lexicons(normalizer)

    battery = build_battery()
    expected_hazm = [baseline_tfidf.preprocess(s) for s in battery]
    expected_naive = [preprocess_ha(s, res, words_full, words_positive, verbs) for s in battery]

    diffs = 0
    for src, exp_h, exp_p in zip(battery, expected_hazm, expected_naive):
        if exp_h != exp_p:
            diffs += 1
            if diffs <= 20:
                print("MISMATCH src=%r\n  hazm=%r\n  port =%r" % (src, exp_h, exp_p))
    if diffs:
        print(f"SELF-CHECK FAILED: {diffs}/{len(battery)} mismatches (of which curated real-hazm vs port)")
        return 1

    # Save assets
    with open(out_assets + "/hazm_normalizer.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=0)
    with open(out_assets + "/hazm_words.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(words_full) + ("\n" if words_full else ""))
    with open(out_assets + "/hazm_words_positive.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(words_positive) + ("\n" if words_positive else ""))
    with open(out_assets + "/hazm_verbs.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(verbs) + ("\n" if verbs else ""))

    pipe = joblib.load(model_path)
    expected_scores = [float(pipe.decision_function([p])[0]) for p in expected_hazm]

    with open(out_tests + "/hazm_fixture.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": battery,
                "outputs": expected_hazm,
                "scores": expected_scores,
                "seed": 20260910,
            },
            f, ensure_ascii=False, indent=0,
        )
    with open(out_assets + "/spam_model.json", "w", encoding="utf-8") as f:
        json.dump(export_model(pipe), f, ensure_ascii=False, indent=0)

    sizes = {
        "words": len(words_full), "words_positive": len(words_positive), "verbs": len(verbs),
        "vocab": len(pipe.named_steps["tfidf"].vocabulary_),
    }
    print("exported to", out_assets)
    print("exported fixture to", out_tests, "entries:", len(battery))
    print("sizes:", sizes)
    print("SELF-CHECK OK: port matched real hazm on", len(battery), "inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())