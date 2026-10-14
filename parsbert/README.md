# ParsBERT

Fine-tunes `HooshvareLab/bert-fa-zwnj-base` on `../data/train.jsonl` and converts it to TFLite.

- `config.py`: base model, paths and `MAX_LENGTH`, shared by every script below
- `2-fine-tune.py`: fine-tune → `../models/parsbert_spam_final`
- `3-run.py`: interactive spam check
- `4-convert-model.py`: convert → `../models/parsbert_spam.tflite`
- `5-evaluate.py`: score on the shared test set → `../models/results/`
- `6-label.py`: first-round labels for a phone export (`make model-label MODEL=parsbert`)

Run them via the root Makefile (`make train-parsbert`, `make test-parsbert`, `make run-parsbert`,
`make export-tflite`), or directly from this directory with `uv run <script>`.
