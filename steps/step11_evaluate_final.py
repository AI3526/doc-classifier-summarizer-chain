"""모델 선택을 마친 뒤 남겨 둔 test로 분류 성능을 한 번 평가한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step11_evaluate_final.py
"""

import json
from pathlib import Path

import joblib
from lesson_settings import DATA_DIR, LEARNER_DIR
from step01_read_data import read_labels, read_rows
from step02_check_data import check_splits
from step03_train_baseline import classification_scores, save_errors
from step06_predict import load_classifier, predict_text


def evaluate_final(data_dir=DATA_DIR, learner_dir=LEARNER_DIR):
    data_dir, learner_dir = Path(data_dir), Path(learner_dir)
    metrics_path = learner_dir / "test_metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"test 결과가 이미 있습니다. 덮어쓰지 않습니다: {metrics_path}")
    # test를 읽기 전에 선택한 모델과 기준 모델이 준비됐는지 확인한다.
    bundle = load_classifier(learner_dir)
    labels = read_labels(data_dir)
    if labels != bundle["labels"]:
        raise ValueError("데이터와 선택한 모델의 라벨 순서가 다릅니다.")
    train = read_rows(data_dir / "train.csv")
    validation = read_rows(data_dir / "validation.csv")
    # 자신이 step03에서 저장한 파일만 읽는다.
    baseline = joblib.load(bundle["run_dir"] / "baseline.joblib")

    # test 점수는 최종 보고에만 쓴다. 이 점수를 보고 모델이나 설정을 다시 고르지 않는다.
    test_rows = read_rows(data_dir / "test.csv")
    check_splits({"train": train, "validation": validation, "test": test_rows}, labels)
    label2id = {label: index for index, label in enumerate(labels)}
    actual = [label2id[row["label"]] for row in test_rows]
    predictions = [label2id[predict_text(row["text"], bundle)["label"]] for row in test_rows]
    baseline_predictions = baseline.predict([row["text"] for row in test_rows]).tolist()
    report = {"selected_run": bundle["run_dir"].name, "split": "test",
              "classifier": classification_scores(actual, predictions, labels),
              "baseline": classification_scores(actual, baseline_predictions, labels)}
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_errors(learner_dir / "test_errors.csv", test_rows, predictions, labels)
    print("분류기 test macro F1:", round(report["classifier"]["macro_f1"], 4))
    print("기준 모델 test macro F1:", round(report["baseline"]["macro_f1"], 4))
    print("저장 위치:", metrics_path)
    return report


def main():
    evaluate_final()


if __name__ == "__main__":
    main()
