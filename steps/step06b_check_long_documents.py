"""선정한 분류기를 재학습 없이 긴 문서(심화 validation)에 적용한다.

같은 분류기에 원문 그대로와 본문만 추출한 입력을 넣어 비교한다(입력 처리 한 가지만 바꿈).
기준 모델(TF-IDF + 로지스틱 회귀)도 같은 두 조건으로 함께 기록한다.
심화 test는 읽지 않는다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step06b_check_long_documents.py
"""

import csv
from collections import Counter, defaultdict

from common.metrics import classification_scores
from common.plots import plot_confusion_matrix
from common.preprocess import document_layout, extract_body
from common.records import now_iso, read_json, stamp, write_json
from common.resources import environment_info, measure_latency
from lesson_settings import OUTPUT_ROOT, ROOT, SELECTION_DIR
from step01_read_data import read_labels, read_rows
from step06_predict import load_run, predict_text

ADVANCED = "documents_advanced"
DATA = ROOT / "data" / ADVANCED
CONDITIONS = {"raw": lambda text: text, "body": extract_body}


def evaluate(bundle, rows, labels, condition):
    prepare = CONDITIONS[condition]
    label2id = {label: index for index, label in enumerate(labels)}
    results = [predict_text(prepare(row["text"]), bundle) for row in rows]
    actual = [label2id[row["label"]] for row in rows]
    predicted = [label2id[result["label"]] for result in results]
    by_layout = defaultdict(lambda: [0, 0])
    for row, result in zip(rows, results, strict=True):
        layout = document_layout(row["text"])
        by_layout[layout][0] += row["label"] == result["label"]
        by_layout[layout][1] += 1
    latency = measure_latency(lambda text: predict_text(prepare(text), bundle),
                              [row["text"] for row in rows])
    summary = {"scores": classification_scores(actual, predicted, labels),
               "accuracy_by_layout": {key: {"correct": c, "total": n, "accuracy": c / n}
                                      for key, (c, n) in sorted(by_layout.items())},
               "truncated_rows": sum(result["truncated"] for result in results),
               "inference_latency": {**latency, "device": "cpu",
                                     "scope": latency["scope"] + " (본문 추출 포함)"}}
    return summary, results


def main():
    selected = read_json(SELECTION_DIR / "selected.json")
    labels = read_labels(DATA)
    rows = read_rows(DATA / "validation.csv")
    models = {"selected": selected["selected"]["run_dir"],
              "baseline": selected["baseline_reference"]["run_dir"]}
    out = OUTPUT_ROOT / ADVANCED / "transfer" / f"{stamp()}_existing_classifier"
    out.mkdir(parents=True)

    metrics, table = {}, []
    for role, run_dir in models.items():
        bundle = load_run(OUTPUT_ROOT / run_dir)
        if bundle["labels"] != labels:
            raise ValueError("기본 데이터와 심화 데이터의 라벨 순서가 다릅니다.")
        for condition in CONDITIONS:
            summary, results = evaluate(bundle, rows, labels, condition)
            key = f"{role}_{condition}"
            metrics[key] = {"run_dir": run_dir, "kind": bundle["kind"],
                            "max_length": bundle["max_length"], "condition": condition, **summary}
            scores = summary["scores"]
            plot_confusion_matrix(scores["confusion_matrix"], labels,
                                  out / f"confusion_{key}.png", f"{role} / {condition}")
            for row, result in zip(rows, results, strict=True):
                table.append({"model": role, "condition": condition, "id": row["id"],
                              "layout": document_layout(row["text"]), "actual": row["label"],
                              "predicted": result["label"],
                              "confidence": round(result["confidence"], 4),
                              "correct": row["label"] == result["label"],
                              "truncated": result["truncated"]})
            print(f"{key:18s} acc {scores['accuracy']:.3f}  macro-F1 {scores['macro_f1']:.3f}  "
                  f"잘림 {summary['truncated_rows']}  구조별 " + ", ".join(
                      f"{k} {v['correct']}/{v['total']}"
                      for k, v in summary["accuracy_by_layout"].items()))

    write_json(out / "config.json", {
        "purpose": "선정 분류기를 재학습 없이 긴 문서에 적용: 원문 vs 본문만",
        "dataset": ADVANCED, "split": "validation", "rows": len(rows),
        "layouts": dict(Counter(document_layout(row["text"]) for row in rows)),
        "conditions": {"raw": "원문 그대로", "body": "common.preprocess.extract_body로 참고 자료 제거"},
        "models": models, "environment": environment_info(), "created_at": now_iso()})
    write_json(out / "metrics.json", metrics)
    with (out / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print("저장:", out)


if __name__ == "__main__":
    main()
