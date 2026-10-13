"""모델 선택을 마친 뒤 남겨 둔 test로 분류 성능을 한 번 평가한다.

step05에서 고정한 선정 분류기와 기준 모델을 같은 test로 평가한다.
이 점수를 보고 모델이나 설정을 다시 고르지 않는다.

심화 트랙(--track advanced)은 artifacts/documents_advanced/selection/selected.json에 test 전에 고정한
결정(기본 데이터에서 선정한 분류기 + 본문 추출 전처리)으로 심화 test를 한 번 평가한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step11_evaluate_final.py
uv run python steps/step11_evaluate_final.py --track advanced
"""

import argparse
import csv
from pathlib import Path

from common.metrics import classification_scores, save_errors
from common.plots import plot_confusion_matrix
from common.preprocess import extract_body
from common.records import now_iso, read_json, write_json
from common.resources import measure_latency
from lesson_settings import DATA_DIR, OUTPUT_ROOT, ROOT, SELECTION_DIR
from step01_read_data import read_labels, read_rows
from step02_check_data import check_splits
from step06_predict import load_run, predict_text

REVIEW_CASES = 5
TRACKS = {"basic": (DATA_DIR, SELECTION_DIR),
          "advanced": (ROOT / "data" / "documents_advanced",
                       OUTPUT_ROOT / "documents_advanced" / "selection")}
PREPROCESSING = {None: lambda text: text, "extract_body": extract_body}


def evaluate_bundle(bundle, rows, label2id, labels, prepare=PREPROCESSING[None]):
    results = [predict_text(prepare(row["text"]), bundle) for row in rows]
    actual = [label2id[row["label"]] for row in rows]
    predictions = [label2id[result["label"]] for result in results]
    latency = measure_latency(lambda text: predict_text(prepare(text), bundle),
                              [row["text"] for row in rows])
    return {"scores": classification_scores(actual, predictions, labels),
            "inference_latency": {**latency, "device": "cpu"},
            "truncated_rows": sum(result["truncated"] for result in results)}, results


def save_predictions(directory, rows, results):
    fields = ["id", "text", "actual", "predicted", "confidence", "correct", "truncated"]
    table = [{"id": row["id"], "text": row["text"], "actual": row["label"],
              "predicted": result["label"], "confidence": round(result["confidence"], 4),
              "correct": row["label"] == result["label"], "truncated": result["truncated"]}
             for row, result in zip(rows, results, strict=True)]
    with (directory / "test_predictions.csv").open("w", encoding="utf-8-sig",
                                                     newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    # 대표 사례 후보: 확신하고 틀린 사례와 확신 없이 맞힌 사례. 분석 메모는 사람이 채운다.
    errors = sorted((row for row in table if not row["correct"]),
                    key=lambda row: -row["confidence"])[:REVIEW_CASES]
    unsure = sorted((row for row in table if row["correct"]),
                    key=lambda row: row["confidence"])[:REVIEW_CASES]
    with (directory / "test_case_review.csv").open("w", encoding="utf-8-sig",
                                                     newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_type", *fields, "analysis_note"])
        writer.writeheader()
        for case_type, group in (("confident_error", errors), ("low_confidence_correct", unsure)):
            for row in group:
                writer.writerow({"case_type": case_type, **row, "analysis_note": ""})


def evaluate_final(data_dir=DATA_DIR, selection_dir=SELECTION_DIR, mixed_group_labels=False):
    data_dir, selection_dir = Path(data_dir), Path(selection_dir)
    metrics_path = selection_dir / "test_metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"test 결과가 이미 있습니다. 덮어쓰지 않습니다: {metrics_path}")
    selection_path = selection_dir / "selected.json"
    if not selection_path.exists():
        raise RuntimeError("먼저 step05_select_model.py에서 사용할 실험을 고르세요.")
    selection = read_json(selection_path)
    preprocessing = selection.get("preprocessing")
    prepare = PREPROCESSING[preprocessing]

    # test를 읽기 전에 선택한 모델과 기준 모델을 모두 불러올 수 있는지 확인한다.
    bundles = {name: load_run(OUTPUT_ROOT / selection[name]["run_dir"])
               for name in ("selected", "baseline_reference")}
    labels = read_labels(data_dir)
    if any(bundle["labels"] != labels for bundle in bundles.values()):
        raise ValueError("데이터와 모델의 라벨 순서가 다릅니다.")
    train = read_rows(data_dir / "train.csv")
    validation = read_rows(data_dir / "validation.csv")

    # test 점수는 최종 보고에만 쓴다.
    test_rows = read_rows(data_dir / "test.csv")
    check_splits({"train": train, "validation": validation, "test": test_rows}, labels,
                 mixed_group_labels)
    label2id = {label: index for index, label in enumerate(labels)}

    report = {"split": "test", "rows": len(test_rows), "evaluated_at": now_iso(),
              "selection_rule": selection["rule"], "preprocessing": preprocessing}
    for name, bundle in bundles.items():
        result, predictions = evaluate_bundle(bundle, test_rows, label2id, labels, prepare)
        report[name] = {"run_dir": selection[name]["run_dir"], **result}
        plot_confusion_matrix(result["scores"]["confusion_matrix"], labels,
                              selection_dir / f"confusion_test_{name}.png",
                              f"{bundle['identity']['run_name']} (test)")
        if name == "selected":
            save_predictions(selection_dir, test_rows, predictions)
            save_errors(selection_dir / "test_errors.csv", test_rows,
                        [label2id[item["label"]] for item in predictions], labels)
    write_json(metrics_path, report)

    for name in bundles:
        scores = report[name]["scores"]
        print(f"{name}: accuracy {scores['accuracy']:.4f}, macro F1 {scores['macro_f1']:.4f}, "
              f"CPU {report[name]['inference_latency']['mean_ms']:.1f}ms")
    print("저장 위치:", metrics_path)
    print("test_case_review.csv의 analysis_note에 성공/실패 이유를 적으세요.")
    return report


def main():
    parser = argparse.ArgumentParser(description="고정한 분류기로 test 1회 평가")
    parser.add_argument("--track", choices=list(TRACKS), default="basic")
    track = parser.parse_args().track
    data_dir, selection_dir = TRACKS[track]
    # 심화 데이터는 한 상황 묶음이 공지·매뉴얼·회의록 3건으로 되어 있다.
    evaluate_final(data_dir, selection_dir, mixed_group_labels=track == "advanced")


if __name__ == "__main__":
    main()
