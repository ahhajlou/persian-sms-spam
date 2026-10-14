"""Link masking cases for spam_common.pipeline. Run with `make check`."""
import pytest

from spam_common.pipeline import mask_urls

CASES = [
    ("لینک https://bit.ly/abc123 کلیک", "لینک  URLTOKEN  کلیک"),
    ("سایت www.digikala.com/p/123", "سایت  URLTOKEN "),
    ("ببینید example.ir/fa?x=1", "ببینید  URLTOKEN "),
    ("تلگرام t.me/channel", "تلگرام  URLTOKEN "),
    # any scheme, any case
    ("HTTPS://BANK.IR/X", " URLTOKEN "),
    ("vless://[SECRET]@mtn.example.space:443?path=/", " URLTOKEN "),
    ("tg://resolve?domain=x", " URLTOKEN "),
    # domain right after Persian text
    ("سایتgoogle.com را ببینید", "سایت URLTOKEN  را ببینید"),
    # the link ends at Persian text, Persian punctuation or trailing punctuation
    ("لینک:https://x.ir،سریع", "لینک: URLTOKEN ،سریع"),
    ("https://x.ir/a?b=1‌کلیک", " URLTOKEN ‌کلیک"),
    ("(see x.ir).", "(see  URLTOKEN )."),
    ("«x.ir»", "« URLTOKEN »"),
    ("x.ir:8080/path", " URLTOKEN "),
    # placeholders from anonymize stay inside the link; web-OTP domains are links
    ("https://my.example.ir/dlp?id=main&ph=[NUMBER]", " URLTOKEN "),
    ("@digikala.com #[CODE]", "@ URLTOKEN  #[CODE]"),
]

UNCHANGED = [
    "عدد 2.5 و 1.000.000 تومان",  # numbers are not links
    "[EMAIL] و [CODE]",
    "ساعت 14:30",
    "Hello. World",
]


@pytest.mark.parametrize("text, expected", CASES)
def test_mask_urls(text, expected):
    assert mask_urls(text) == expected


@pytest.mark.parametrize("text", UNCHANGED)
def test_not_links(text):
    assert mask_urls(text) == text
