"""Regression cases for spam_common.anonymize. The cases live in
core/spam-core/tests/cases/anonymize.json, shared with the Rust core. Add new cases there.
Run with `make check`."""
import json

import pytest

from spam_common.anonymize import anonymize, card_ok, national_id_ok
from spam_common.data import REPO_ROOT

CASES = json.loads((REPO_ROOT / "core/spam-core/tests/cases/anonymize.json").read_text("utf-8"))
MASKED = CASES["masked"]
UNCHANGED = CASES["unchanged"]


@pytest.mark.parametrize("text, expected", MASKED)
def test_masked(text, expected):
    assert anonymize(text) == expected


@pytest.mark.parametrize("text", UNCHANGED)
def test_unchanged(text):
    assert anonymize(text) == text


def test_idempotent():
    for text, _ in MASKED:
        once = anonymize(text)
        assert anonymize(once) == once


def test_validators():
    for value in CASES["national_id"]["valid"]:
        assert national_id_ok(value)
    for value in CASES["national_id"]["invalid"]:
        assert not national_id_ok(value)
    for value in CASES["card"]["valid"]:
        assert card_ok(value)
    for value in CASES["card"]["invalid"]:
        assert not card_ok(value)
