# Data prep

```bash
uv run label_viewer_gui.py                                               # review / fix labels
uv run augment_spam.py ../data/sms_labeled.jsonl ../data/augmented.jsonl 200  # needs Ollama running locally
```
