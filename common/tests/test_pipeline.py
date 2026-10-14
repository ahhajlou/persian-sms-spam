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
