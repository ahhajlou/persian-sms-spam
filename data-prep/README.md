# Data prep

From the repo root:

```bash
make regex-label IN=data/phone/export.jsonl  # first-round labels -> export_regex.jsonl
make label                                  # label viewer GUI on data/sms_labeled.jsonl
make label DATA=data/other.jsonl            # ...or on another file
make augment N=200                          # synthetic spam via a local Ollama model
```

`regex_label.py` labels a message spam if any SpamBlocker blacklist rule in
`data/regex_rules.json` matches it; everything else is ham. It never overwrites an
existing output unless given `--force`.

This environment uses the system Python (`python-preference = "only-system"`):
uv's bundled Python has a Tk build that crashes on Fedora/Wayland.
On Fedora, tkinter comes from the `python3-tkinter` package.
