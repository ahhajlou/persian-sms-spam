"""model_label and merge, with a fake model. Run with `make check`."""
import json

from spam_common.merge import merge
from spam_common.model_label import SCORE_FIELD, model_label


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_model_label(tmp_path):
    dataset = write(tmp_path / "labeled.jsonl", [{"text": "کد تایید شما 12345 است", "label": "ham"}])
    export = write(tmp_path / "export.jsonl", [
        {"id": 1, "text": "کد تایید شما 99999 است", "label": None},  # same model input as the dataset row
        {"id": 2, "text": "تخفیف ویژه", "label": "ham"},
        {"id": 3, "text": "سلام", "address": "x"},
    ])
    seen = []

    def fake_scores(texts):
        seen.extend(texts)
        return [0.9 if "تخفیف" in t else 0.1 for t in texts]

    model_label(export, fake_scores, "fake", dataset=dataset)
    out = read(tmp_path / "export_model.jsonl")

    assert seen == ["تخفیف ویژه", "سلام"]  # rows already in the dataset are not sent to the model
    assert [(r["label"], r[SCORE_FIELD]) for r in out] == [("ham", None), ("spam", 0.9), ("ham", 0.1)]
    # every other field is kept as it was
    assert [r["id"] for r in out] == [1, 2, 3] and out[2]["address"] == "x"


def test_merge_drops_model_score():
    added, _ = merge([], [{"text": "تخفیف ویژه", "label": "spam", SCORE_FIELD: 0.9, "address": "x"}])
    assert added == [{"text": "تخفیف ویژه", "label": "spam", "address": "x"}]
