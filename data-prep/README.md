# Data prep

Standalone tools that produce or edit dataset files; no model imports them.

- `extract_sms.py`: `adb` SMS dump → unlabelled JSONL (`make extract`)
- `regex_label.py`: first-round labels from the SpamBlocker rules in `data/regex_rules.json` (`make regex-label`)
- `label_viewer_gui.py`: review and fix labels by hand (`make label`)
- `augment_spam.py`: synthetic spam from a local Ollama model (`make augment`)

The full workflow is in the root README under "Adding data from a phone".

This environment uses the system Python (`python-preference = "only-system"`):
uv's bundled Python has a Tk build that crashes on Fedora/Wayland.
On Fedora, tkinter comes from the `python3-tkinter` package.
