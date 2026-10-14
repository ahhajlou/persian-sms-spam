"""Scoring shared by every model, so results are directly comparable.

`evaluate` prints a report and saves it to models/results/<name>.json, with the
PIPELINE_VERSION it was scored under; `python -m spam_common.metrics` prints all
saved results side by side and marks those from another pipeline version.
"""
import json

from spam_common.data import LABELS, RESULTS_DIR
from spam_common.pipeline import PIPELINE_VERSION


def evaluate(name, y_true, y_pred, positive="spam"):
    total = len(y_true)
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        predicted = sum(confusion[t][label] for t in LABELS)
        actual = sum(confusion[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}

    result = {
        "name": name,
        "pipeline_version": PIPELINE_VERSION,
        "n": total,
        "accuracy": sum(confusion[label][label] for label in LABELS) / total if total else 0.0,
        "f1_macro": sum(m["f1"] for m in per_label.values()) / len(LABELS),
        "per_label": per_label,
        "confusion": confusion,
        # The two errors that matter for an SMS filter.
        "missed_spam": sum(confusion[positive][p] for p in LABELS if p != positive),
        "blocked_ham": sum(confusion[t][positive] for t in LABELS if t != positive),
    }
    print_report(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {path}")
    return result


def print_report(result):
    print(f"=== {result['name']} ({result['n']} test messages) ===")
    print(f"{'':<8}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for label, m in result["per_label"].items():
        print(f"{label:<8}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10}")
    print(f"accuracy {result['accuracy']:.3f}   macro f1 {result['f1_macro']:.3f}")
    print("Confusion matrix (rows=true, cols=pred):", list(LABELS))
    for t in LABELS:
        print(f"  {t:<6}", [result["confusion"][t][p] for p in LABELS])


def compare():
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("*.json"))]
    if not results:
        print(f"No results in {RESULTS_DIR} — run `make test-all` first.")
        return
    results.sort(key=lambda r: r["f1_macro"], reverse=True)
    width = max(len(r["name"]) for r in results)
    print(f"{'model':<{width}}{'pipeline':>9}{'n':>6}{'accuracy':>10}{'macro f1':>10}{'spam f1':>9}{'missed spam':>13}{'blocked ham':>13}")
    for r in results:
        # Results saved before the version was recorded have none ("?").
        version = r.get("pipeline_version")
        mark = f"{'?' if version is None else version}{'' if version == PIPELINE_VERSION else '*'}"
        print(
            f"{r['name']:<{width}}{mark:>9}{r['n']:>6}{r['accuracy']:>10.3f}{r['f1_macro']:>10.3f}"
            f"{r['per_label']['spam']['f1']:>9.3f}{r['missed_spam']:>13}{r['blocked_ham']:>13}"
        )
    if any(r.get("pipeline_version") != PIPELINE_VERSION for r in results):
        print(f"\n* not scored with the current pipeline (version {PIPELINE_VERSION}); "
              "re-run that model's test, e.g. `make test-fasttext`.")
    if len({r["n"] for r in results}) > 1:
        print("\nWarning: results were computed on test sets of different sizes; re-run `make test-all`.")


if __name__ == "__main__":
    compare()
