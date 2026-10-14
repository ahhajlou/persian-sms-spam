"""The one text preprocessing pipeline, used by every model and mirrored on Android.

    raw SMS ─► anonymize ─► mask_urls ─► mask_numbers ─► normalize ─► model input

The order matters:
- anonymize runs first because its rules need the raw digits (national ID
  check digit, card formats, codes after keywords).
- URLs are masked before hazm, because hazm's normalize() inserts spaces
  around punctuation ("https://x.ir" -> "https: / / x. ir") and would break
  them apart.

The Android app (NoSpam core:ml) reimplements these steps in Kotlin;
classic/export_android_assets.py writes a fixture of inputs and expected
outputs that its JVM test checks against. Bump PIPELINE_VERSION on any change
that alters the output, then retrain and re-export.
"""
import re

from spam_common.anonymize import anonymize

# 1: URL mask -> number mask -> hazm
# 2: anonymize added in front
# 3: anonymize fixes (emails need a domain, URI secrets, no keyword matches
#    inside placeholders or links, values on the next line, Luhn for bare cards)
# 4: URL fixes (any scheme in any case, domains right after Persian text,
#    links end at Persian text or trailing punctuation, bare domains need a
#    letter TLD so numbers like 2.5 are not links)
PIPELINE_VERSION = 4

# A link ends at whitespace, Persian/Arabic text (U+0600-U+06FF, which also
# covers «،» «؛» «؟»), ZWNJ, bidi marks or guillemets, and never on trailing
# punctuation, so "x.ir،سریع" and "(see x.ir)." keep the words around the link.
_URL_CHAR = r"[^\s\u0600-\u06FF\u200c-\u200f\u202a-\u202e«»]"
_URL_TAIL = _URL_CHAR + r"*(?<![.,!?:;'\")])"
URL_RE = re.compile(
    # scheme://..., any scheme (https, HTTPS, vless, tg, ...)
    r"(?<![A-Za-z0-9+.\-])[a-z][a-z0-9+.\-]*://" + _URL_TAIL
    # www.... or a bare domain with a letter TLD, optionally with a port and
    # path; may follow Persian text directly ("سایتgoogle.com").
    + r"|(?<![A-Za-z0-9\-.])(?:www\.)?[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.[a-z]{2,}(?![a-z0-9\-])"
    + r"(?::\d+)?(?:[/?#]" + _URL_TAIL + r")?",
    re.I,
)
NUMBER_RE = re.compile(r"\d+")

_normalizer = None


def mask_urls(text):
    return URL_RE.sub(" URLTOKEN ", text)


def mask_numbers(text):
    return NUMBER_RE.sub(" NUM_TOKEN ", text)


def normalize(text):
    # hazm loads its lexicons on first use (slow), so only when needed.
    global _normalizer
    if _normalizer is None:
        from hazm import Normalizer
        _normalizer = Normalizer()
    return _normalizer.normalize(text)


STEPS = (
    ("anonymize", anonymize),
    ("urls", mask_urls),
    ("numbers", mask_numbers),
    ("normalize", normalize),
)


def preprocess_stages(text):
    """Output after each step, keyed by step name (for the Android fixture)."""
    stages = {}
    for name, step in STEPS:
        text = step(text)
        stages[name] = text
    return stages


def preprocess(text):
    for _, step in STEPS:
        text = step(text)
    return text
