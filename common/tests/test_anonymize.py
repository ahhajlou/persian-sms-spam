"""Regression cases for spam_common.anonymize, taken from real message shapes
(digits changed). Run with `make check`."""
import pytest

from spam_common.anonymize import anonymize, card_ok, national_id_ok

MASKED = [
    # passwords, usernames
    ("رمز شخصی شما 2062761400 است", "رمز شخصی شما [PASSWORD] است"),
    ("رمز موقت ثنا ۹۴۳۷۷۴ انقضا 1405", "رمز موقت ثنا [PASSWORD] انقضا 1405"),
    ("نام کاربری): 403135156 و کلمه عبور: Abc12345", "نام کاربری): [USERNAME] و کلمه عبور: [PASSWORD]"),
    ("نام کاربری admin", "نام کاربری [USERNAME]"),
    ("رمز عبور:\nGoi12Q3v", "رمز عبور:\n[PASSWORD]"),
    ("رمزعبور: Amir@1amir", "رمزعبور: [PASSWORD]"),
    ("گذرواژه: Secret99", "گذرواژه: [PASSWORD]"),
    ("Password: U1o2A3j45jIERkJn", "Password: [PASSWORD]"),
    # codes
    ("Code: 567560 do not share", "Code: [CODE] do not share"),
    ("رمز اول کارت\nCode: 123456\nزمان اعتبار", "رمز اول کارت\nCode: [CODE]\nزمان اعتبار"),
    ("code=CHO-24141973", "code=[CODE]"),
    ("@digikala.com #54321", "@digikala.com #[CODE]"),
    ("با شماره پیگیری ۱۲۳۴۵۶ ثبت شد", "با شماره پیگیری [CODE] ثبت شد"),
    # IDs and bank data
    ("کد ملی (کلمه عبور) : 0023191589", "کد ملی (کلمه عبور) : [NATID]"),
    ("مانده:50,051,391 برداشت:1,000,000", "مانده:[AMOUNT] برداشت:[AMOUNT]"),
    ("برداشت از حساب 0303438303006 مبلغ 50,000", "برداشت از حساب [ACCOUNT] مبلغ [AMOUNT]"),
    ("حساب:93009", "حساب:[ACCOUNT]"),
    ("حساب 123.4567.12345678.1 بانک", "حساب [ACCOUNT] بانک"),
    ("شبا : IR820540102680020817909002", "شبا : [IBAN]"),
    ("کارت 6037-9975-1234-5678 و 603799******1234 و 603799*1234", "کارت [CARD] و [CARD] و [CARD]"),
    ("کارت 6037997599999993", "کارت [CARD]"),  # passes Luhn
    ("کارت مبدا: \u200e6037-99**-****-1234", "کارت مبدا: \u200e[CARD]"),
    # contact data
    ("تماس 021-88776655 یا +989121234567 یا ۰۹۱۲ ۳۴۵ ۶۷۸۹", "تماس [PHONE] یا [PHONE] یا [PHONE]"),
    ("نام کاربری: someone@gmail.com", "نام کاربری: [EMAIL]"),
    ("ایمیل a.b@hut.ac.ir", "ایمیل [EMAIL]"),
    ("vless://a12f34ed-dcaa-56fc-7cba-db8f9c01dd23@mtn.example.space:443?path=/", "vless://[SECRET]@mtn.example.space:443?path=/"),
    # safety net
    ("مشترک گرامی 912345678901", "مشترک گرامی [NUMBER]"),
]

UNCHANGED = [
    # device commands are not emails (issue found in the Galaxy A8 export)
    "#AMIR1827@HALL@ON#",
    "#AM18@H1@OFF#",
    "#AMIR1827@",
    "Relay1 ON",
    # keywords inside links
    "https://example.ir/login/12345",
    "https://example.ir/login?utm_source=sms",
    "https://bank.ir/Account.aspx?Req=12345",
    # short numbers after code keywords are not codes
    "کد تخفیف 20 درصدی",
    "کد 12 را به 300 بفرستید",
    # spam signals stay
    "کد دستوری *780*1234# را بگیرید",
    "قیمت فقط 50000 تومان",
    "ارسال عدد 1 به 3000",
    # words that contain a keyword
    "خرید رمزارز با 1000000",
    "سرمایه گذاری در رمز ارز با 1000000",
    # links are not passwords, and values never come from unrelated lines
    "رمز عبور:\nhttps://example.ir/reset",
    "برای تغییر رمز عبور کلیک کنید:\n\nhttps://example.ir",
    "رمز اول کارت\nwww.bank.ir",
    "واریز در سایت\n\nsM5 کد",
    "حساب کاربری خود را شارژ کنید",
    "رمز دستگاه را وارد کنید",
]


@pytest.mark.parametrize("text, expected", MASKED)
def test_masked(text, expected):
    assert anonymize(text) == expected


@pytest.mark.parametrize("text", UNCHANGED)
def test_unchanged(text):
    assert anonymize(text) == text


def test_placeholders_are_not_keywords():
    # "[PASSWORD]" contains "password"; the word after it must stay.
    assert anonymize("رمز: 12345 انقضا 1405") == "رمز: [PASSWORD] انقضا 1405"
    assert anonymize("Code: 123456/12") == "Code: [CODE]/12"


def test_idempotent():
    for text, _ in MASKED:
        once = anonymize(text)
        assert anonymize(once) == once


def test_validators():
    assert national_id_ok("0023191589")
    assert not national_id_ok("0023191588")
    assert not national_id_ok("1111111111")
    assert card_ok("6037997599999993")
    assert not card_ok("6037997599999990")
    assert card_ok("6037-9975-1234-5678")  # grouped: always a card
