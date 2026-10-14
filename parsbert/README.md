# ParsBERT

Fine-tunes `HooshvareLab/bert-fa-zwnj-base` on `../data/train.jsonl` and converts it to TFLite.
Usually run from the repo root via the Makefile (`make train-parsbert`, `make test-parsbert`, ...).
The equivalent direct commands, from this directory:

```bash
uv run 1-test-tokenizer.py   # sanity-check the tokenizer
uv run 2-fine-tune.py        # → ../models/parsbert_spam_final
uv run 3-run.py              # interactive spam check
uv run 4-convert-model.py    # → ../models/parsbert_spam.tflite
uv run 5-evaluate.py         # score on ../data/test.jsonl → ../models/results/
```
