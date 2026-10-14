"""The one text preprocessing pipeline, used by every model and by the NoSpam app.

    raw SMS ─► anonymize ─► mask_urls ─► mask_numbers ─► normalize ─► model input
                                                                 └─► ngrams (TF-IDF)

All of it runs in the Rust core (core/spam-core, module `spam_core`); this module only
chooses the normalizer and keeps the names the models import. The rules are data in
core/spam-core/rules/ and the phone runs the same code.

The order matters:
- anonymize runs first because its rules need the raw digits (national ID
  check digit, card formats, codes after keywords).
- URLs are masked before the normalizer, because hazm's spacing rules insert spaces
  around punctuation ("https://x.ir" -> "https: / / x. ir") and would break
  them apart.

Only the first spam_core.MAX_CHARS (2,000) characters of a message are used.

PIPELINE_VERSION is a Rust constant (core/spam-core/src/lib.rs); bump it there on any
change that alters the output. cargo test fails until you do (golden files), and
everything trained on the old output is out of date. Keep the history below.
"""
import spam_core

from spam_common.data import REPO_ROOT

# 1: URL mask -> number mask -> hazm
# 2: anonymize added in front
# 3: anonymize fixes (emails need a domain, URI secrets, no keyword matches
#    inside placeholders or links, values on the next line, Luhn for bare cards)
# 4: URL fixes (any scheme in any case, domains right after Persian text,
#    links end at Persian text or trailing punctuation, bare domains need a
#    letter TLD so numbers like 2.5 are not links)
# 5: the Rust core. hazm's `rules` normalizer (no word lists), n-grams from the core,
#    anonymizer digit classes are [0-9] (other digit scripts are no longer digits there),
#    U+0130/U+0131 no longer match i in case-insensitive rules, 2,000-character cut
PIPELINE_VERSION = spam_core.PIPELINE_VERSION

# "rules": hazm's regex rules and character tables. "full" adds hazm's word-list steps
# (needs `make lexicons`, ~14 MB). Changing it changes the output: retrain, and bump the
# version if a model was shipped with the old one.
NORMALIZER = "rules"

_normalizer = spam_core.Normalizer(NORMALIZER, str(REPO_ROOT / "core" / "data" / "lexicons"))


def mask_urls(text):
    return spam_core.mask_urls(text)


def mask_numbers(text):
    return spam_core.mask_numbers(text)


def normalize(text):
    return _normalizer.normalize(text)


def ngrams(text):
    """Lowercase + char_wb n-grams (2-4) of preprocessed text: the TF-IDF analyzer.

    A module-level function, so joblib can pickle a trained model that uses it.
    """
    return spam_core.ngrams(text)


def preprocess_stages(text):
    """Output after each step, keyed by step name (for the Android fixture)."""
    s = spam_core.stages(text, _normalizer)
    return {
        "anonymize": s["anonymized"],
        "urls": s["urls"],
        "numbers": s["numbers"],
        "normalize": s["normalized"],
    }


def preprocess(text):
    return spam_core.preprocess(text, _normalizer)
