# ParsBERT

Fine-tunes `HooshvareLab/bert-fa-zwnj-base` on `../data/sms_labeled.jsonl` and converts it to TFLite.

```bash
uv run 1-test-tokenizer.py   # sanity-check the tokenizer
uv run 2-fine-tune.py        # → ../models/parsbert_spam_final
uv run 3-run.py              # interactive spam check
uv run 4-convert-model.py    # → ../models/parsbert_spam.tflite
```

`spam-filter-pipeline.ipynb` runs the same steps, plus SMS extraction and regex labelling:

```bash
uv run --with jupyterlab jupyter lab
```
