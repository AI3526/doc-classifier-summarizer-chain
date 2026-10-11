"""문자 TF-IDF와 로지스틱 회귀로 비교 기준 모델을 학습한다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step03_train_baseline.py
"""

import csv
import json
from pathlib import Path

import joblib
from lesson_settings import DATASET, LEARNER_DIR, ROOT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from step01_read_data import read_labels, read_rows
from step02_check_data import check_splits

DATA_DIR = ROOT / "data" / DATASET
NGRAM_RANGE = (2, 5)
SEED = 42


def classification_scores(actual, predicted, labels):
    # 0, 1, 2를 항상 같은 라벨 순서로 해석한다.
    ids = list(range(len(labels)))
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=ids,
                                   average="macro", zero_division=0)),
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


def train_baseline(data_dir=DATA_DIR, learner_dir=LEARNER_DIR,
                   ngram_range=NGRAM_RANGE, seed=SEED):
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    output = learner_dir / "baseline"
    if output.exists():
        raise FileExistsError(f"기준 모델이 이미 있습니다: {output}")
    labels = read_labels(data_dir)
    train = read_rows(data_dir / "train.csv")
    validation = read_rows(data_dir / "validation.csv")
    check_splits({"train": train, "validation": validation}, labels)
    label2id = {label: index for index, label in enumerate(labels)}

    # fit은 train에서만 수행한다. 어휘와 IDF도 이 단계에서 train으로 학습된다.
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=ngram_range, min_df=1),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    model.fit([row["text"] for row in train], [label2id[row["label"]] for row in train])
    predictions = model.predict([row["text"] for row in validation]).tolist()
    actual = [label2id[row["label"]] for row in validation]
    metrics = classification_scores(actual, predictions, labels)

    output.mkdir(parents=True)
    joblib.dump(model, output / "baseline.joblib")
    config = {"dataset": data_dir.name, "labels": labels,
              "ngram_range": list(ngram_range), "seed": seed}
    for filename, value in (("config.json", config), ("validation_metrics.json", metrics)):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    save_errors(output / "validation_errors.csv", validation, predictions, labels)
    print("기준 모델 validation macro F1:", round(metrics["macro_f1"], 4))
    print("저장 위치:", output)
    return metrics


def main():
    train_baseline()


if __name__ == "__main__":
    main()
