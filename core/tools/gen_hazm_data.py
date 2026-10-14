"""Write the hazm data the Rust core needs, from the installed hazm (0.12.x, MIT).

    uv run --project common python core/tools/gen_hazm_data.py            # rules/hazm.json
    uv run --project common python core/tools/gen_hazm_data.py --lexicons  # + data/lexicons/

hazm.json (committed, ~6 KB of rules and character tables) drives the `rules` normalizer.
The word lists (gitignored, ~14 MB) are only for the `full` normalizer; `make lexicons`
runs this with --lexicons.
"""
import argparse
import json
import re
from pathlib import Path

import hazm
from hazm import Normalizer, WordTokenizer

from rx import to_rust

CORE = Path(__file__).resolve().parents[1]
RULES_FILE = CORE / "spam-core" / "rules" / "hazm.json"
LEXICON_DIR = CORE / "data" / "lexicons"


def parse_replacement(template):
    """Python re.sub template -> [{"l": literal} | {"g": group}]; only \\1..\\9 are allowed."""
    assert not re.findall(r"\\[^0-9]", template), template
    parts, pos = [], 0
    for m in re.finditer(r"\\(\d+)", template):
        if m.start() > pos:
            parts.append({"l": template[pos:m.start()]})
        parts.append({"g": int(m.group(1))})
        pos = m.end()
    if pos < len(template):
        parts.append({"l": template[pos:]})
    return parts


def export_rules(normalizer):
    from hazm.constants import (
        AFFIX_SPACING_PATTERNS, DIACRITICS_PATTERNS, EXTRA_SPACE_PATTERNS, NUMBERS_DST,
        NUMBERS_SRC, PERSIAN_STYLE_PATTERNS, PUNCTUATION_SPACING_PATTERNS,
        SPECIAL_CHARS_PATTERNS, SUFFIXES, TRANSLATION_DST, TRANSLATION_SRC,
        UNICODE_REPLACEMENTS,
    )
    assert len(TRANSLATION_SRC) == len(TRANSLATION_DST) and len(NUMBERS_SRC) == len(NUMBERS_DST)

    def rx(p):
        # hazm runs these on text that may still hold digits of any script, as Python's \d does.
        return to_rust(p, digits=r"\p{Nd}")

    def plist(rows):
        return [{"p": rx(p), "r": parse_replacement(r)} for p, r in rows]

    letters = normalizer.repeated_chars_pattern.pattern
    letters = letters[letters.index("["): letters.index("]") + 1]
    tokenizer = WordTokenizer(join_verb_parts=False)
    return {
        "hazm_version": hazm.__version__ if hasattr(hazm, "__version__") else "0.12.1",
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
        "tokenize_pattern": rx(tokenizer.pattern.pattern),
        "repeat_chars_pattern": rx(normalizer.repeated_chars_pattern.pattern),
        "repeat_more_than_two_pattern": rx(normalizer.more_than_two_repeat_pattern.pattern),
        "seperate_mi_pattern": rx("\\bن?می" + letters + "+"),
    }


def export_lexicons(normalizer):
    words = normalizer._tokenizer.words  # last-wins dict: word -> (rank, tags)
    return {
        "words.txt": sorted(words),
        "words_positive.txt": sorted(k for k, v in words.items() if v[0] > 0),
        "verbs.txt": sorted(normalizer._verbs),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicons", action="store_true", help="also write the word lists for the `full` normalizer")
    args = ap.parse_args()

    normalizer = Normalizer()
    RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    RULES_FILE.write_text(json.dumps(export_rules(normalizer), ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("wrote", RULES_FILE)
    if args.lexicons:
        LEXICON_DIR.mkdir(parents=True, exist_ok=True)
        for name, items in export_lexicons(normalizer).items():
            (LEXICON_DIR / name).write_text("\n".join(items) + "\n", "utf-8")
            print(f"wrote {LEXICON_DIR / name} ({len(items)} entries)")


if __name__ == "__main__":
    main()
