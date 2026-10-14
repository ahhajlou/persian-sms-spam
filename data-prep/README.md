# Data prep

Standalone tools that produce or edit dataset files; no model imports them.

- `extract_sms.py`: `adb` SMS dump → unlabelled JSONL (`make extract`)
- `regex_label.py`: first-round labels from the SpamBlocker rules in `data/regex_rules.json` (`make regex-label`);
  once a model is trained, `make model-label` is usually the better first round
- `label_viewer_gui.py`: review and fix labels by hand (`make label`); filters and sorts by `model_score`.
  Keys: `S` spam, `H` ham (both move on to the next message), Up/Down to browse, `Ctrl+S` to save
- `augment_spam.py`: synthetic spam from a local Ollama model (`make augment`)

The full workflow is in the root README under "Adding data from a phone".
