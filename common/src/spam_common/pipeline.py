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
PIPELINE_VERSION = 3

URL_RE = re.compile(
    r"(?:https?://[^\s]+)"
    r"|(?:www\.[^\s]+)"
    r"|(?:\b[a-zA-Z0-9][a-zA-Z0-9\-]*(?:\.[a-zA-Z0-9\-]+)+(?:/[^\s]*)?)"
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
