"""Link masking cases for spam_common.pipeline. The cases live in
core/spam-core/tests/cases/urls.json, shared with the Rust core. Run with `make check`."""
import json

import pytest

from spam_common.data import REPO_ROOT
from spam_common.pipeline import mask_urls

CASES = json.loads((REPO_ROOT / "core/spam-core/tests/cases/urls.json").read_text("utf-8"))


@pytest.mark.parametrize("text, expected", CASES["masked"])
def test_mask_urls(text, expected):
    assert mask_urls(text) == expected


@pytest.mark.parametrize("text", CASES["unchanged"])
def test_not_links(text):
    assert mask_urls(text) == text


# --- the binding to the Rust core ---------------------------------------------

def test_stages_chain_into_preprocess():
    from spam_common.pipeline import preprocess, preprocess_stages

    text = "کد: 123456 سایت x.ir 50% تخفیف"
    stages = preprocess_stages(text)
    assert list(stages) == ["anonymize", "urls", "numbers", "normalize"]
    assert stages["anonymize"] == "کد: [CODE] سایت x.ir 50% تخفیف"
    assert "URLTOKEN" in stages["urls"] and "NUM_TOKEN" in stages["numbers"]
    assert preprocess(text) == stages["normalize"]


def test_version_and_normalizer_come_from_the_core():
    import spam_core
    from spam_common.pipeline import NORMALIZER, PIPELINE_VERSION

    assert PIPELINE_VERSION == spam_core.PIPELINE_VERSION
    assert NORMALIZER in ("rules", "full")


def test_only_the_first_2000_characters_are_used():
    import spam_core
    from spam_common.pipeline import preprocess

    head = "سلام " * 400
    assert len(head) == spam_core.MAX_CHARS
    assert preprocess(head + "someone@example.com") == preprocess(head)


def test_ngrams_is_picklable_and_matches_sklearn():
    import pickle

    from spam_common.pipeline import ngrams

    assert pickle.loads(pickle.dumps(ngrams)) is ngrams  # a module-level function
    # ngrams.json holds sklearn's own char_wb output (made when the core was written).
    cases = json.loads((REPO_ROOT / "core/spam-core/tests/cases/ngrams.json").read_text("utf-8"))
    for text, expected in cases:
        assert ngrams(text) == expected


def test_anonymize_tallies_counts_and_limit_hits():
    from collections import Counter

    from spam_common.anonymize import anonymize

    counts = Counter()
    anonymize("کارت 6037-9975-1234-5678 و کارت 6037997599999993", counts)
    assert counts["CARD"] == 2
    anonymize("a." * 20000, counts)  # crafted: the email rule gives up
    assert counts["LIMIT_HIT"] >= 1
