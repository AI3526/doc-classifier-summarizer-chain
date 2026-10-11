"""분류 지표와 오분류 기록. 모든 후보가 같은 함수로 채점된다."""

import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_scores(actual, predicted, labels):
    # 0, 1, 2를 항상 같은 라벨 순서로 해석한다.
    ids = list(range(len(labels)))
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=ids,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=ids,
                                      average="weighted", zero_division=0)),
        "per_class": classification_report(actual, predicted, labels=ids,
                                           target_names=labels, output_dict=True,
                                           zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=ids).tolist(),
        "confusion_matrix_axes": {"rows": "actual", "columns": "predicted", "labels": labels},
    }


def save_errors(path, rows, predictions, labels):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "text", "actual", "predicted", "group_id"])
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            if row["label"] != labels[prediction]:
                writer.writerow({"id": row["id"], "text": row["text"],
                                 "actual": row["label"], "predicted": labels[prediction],
                                 "group_id": row["group_id"]})
