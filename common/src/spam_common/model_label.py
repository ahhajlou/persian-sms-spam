"""First-round labelling of a phone export with a trained model.

Each model script calls `model_label()` from its `label` command with a
function that returns, for a list of texts, how sure the model is that each
one is spam (0..1). For every row:

- a message already in the labelled dataset (same model input, as in `merge`)
  gets its existing, checked label and `model_score: null`;
- any other message gets the model's label and `model_score`.

Review the output with `make label`: low-confidence messages first.
`model_score` is review-only; `merge` drops it, so it never reaches the dataset.
"""
from collections import Counter

from spam_common.data import LABELED_FILE, load_jsonl, output_path, write_jsonl
from spam_common.split import dedup_key

SCORE_FIELD = "model_score"
THRESHOLD = 0.5
# Scores in this range are reported as uncertain (worth reviewing first).
UNCERTAIN = (0.3, 0.7)


def known_labels(path=LABELED_FILE):
    if not path.exists():
        return {}
    return {dedup_key(row["text"]): row["label"].lower() for row in load_jsonl(path)}


def model_label(input_path, spam_scores, model_name, output=None, force=False, dataset=LABELED_FILE):
    output = output_path(input_path, "model", output, force)
    rows = load_jsonl(input_path)
    known = known_labels(dataset)

    keys = [dedup_key(row.get("text") or "") for row in rows]
    todo = [i for i, key in enumerate(keys) if key not in known]
    scores = dict(zip(todo, spam_scores([rows[i].get("text") or "" for i in todo])))

    stats = Counter()
    out = []
    for i, (row, key) in enumerate(zip(rows, keys)):
        old = (row.get("label") or "").lower() or None
        if key in known:
            label, score = known[key], None
            stats["from dataset"] += 1
        else:
            score = round(float(scores[i]), 3)
            label = "spam" if score >= THRESHOLD else "ham"
            stats["from model"] += 1
            stats["uncertain"] += UNCERTAIN[0] <= score <= UNCERTAIN[1]
        stats[label] += 1
        stats["changed"] += old is not None and old != label
        out.append({**row, "label": label, SCORE_FIELD: score})
    write_jsonl(out, output)

    print(f"{model_name}: {len(out)} messages -> {output}")
    print(f"  labels: {stats['spam']} spam, {stats['ham']} ham")
    print(f"  {stats['from dataset']} already in the dataset (existing label), {stats['from model']} labelled by the model")
    print(f"  {stats['uncertain']} uncertain (score {UNCERTAIN[0]}-{UNCERTAIN[1]}): review these first")
    if stats["changed"]:
        print(f"  {stats['changed']} labels differ from the input file's labels")
    return out
