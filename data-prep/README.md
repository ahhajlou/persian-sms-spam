# Data prep

From the repo root:

```bash
make anonymize IN=data/phone/export.jsonl    # mask sensitive data -> export_anon.jsonl
make regex-label IN=data/phone/export_anon.jsonl  # first-round labels -> export_anon_regex.jsonl
make extract                                # adb dump data/sms_dump.txt -> data/sms_unlabeled.jsonl
make label                                  # label viewer GUI on data/sms_labeled.jsonl
make label DATA=data/other.jsonl            # ...or on another file
make augment N=200                          # synthetic spam via a local Ollama model
```

`anonymize` (in `common`, `spam_common.anonymize`) replaces passwords, usernames,
OTP/tracking codes, national IDs, card/IBAN/account numbers, balances, phone numbers
and emails with placeholders such as `[PASSWORD]` or `[CARD]`, and turns phone-number
senders into stable `PHONE_<hash>` pseudonyms. It is rule-based: person names and
street addresses are not detected.

`regex_label.py` labels a message spam if any SpamBlocker blacklist rule in
`data/regex_rules.json` matches it; everything else is ham. It never overwrites an
existing output unless given `--force`.

This environment uses the system Python (`python-preference = "only-system"`):
uv's bundled Python has a Tk build that crashes on Fedora/Wayland.
On Fedora, tkinter comes from the `python3-tkinter` package.
