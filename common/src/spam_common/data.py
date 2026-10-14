"""Dataset paths and loading, shared by every model."""
import json
from pathlib import Path

# common/src/spam_common/data.py -> repo root. Valid because every model
# project installs this package editable (see [tool.uv.sources]).
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = MODELS_DIR / "results"

LABELED_FILE = DATA_DIR / "sms_labeled.jsonl"
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

LABELS = ("ham", "spam")
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))


def clean_text(text: str) -> str:
    # Collapse newlines and repeated spaces. str.split() does not split on
    # ZWNJ (U+200C), so Persian half-spaces are preserved.
    return " ".join(text.split())


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_texts_labels(path) -> tuple[list[str], list[str]]:
    rows = load_jsonl(path)
    return [row["text"] for row in rows], [row["label"].lower() for row in rows]


def write_jsonl(rows, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
