"""팀 전체 실행의 validation 결과를 비교해 서비스에 사용할 분류기 하나를 고른다.

artifacts/{dataset}/*/runs/*/experiment.json을 모두 모아 기준 모델과 사전학습 모델을
같은 표로 비교한다. 선정 규칙은 결과를 보기 전에 정한 아래 규칙을 그대로 쓴다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step05_select_model.py
"""

import csv
import shutil
from pathlib import Path

from common.plots import plot_confusion_matrix
from common.records import now_iso, read_json, stamp, write_json
from lesson_settings import DATASET, OUTPUT_ROOT, SELECTION_DIR

# 선정 규칙: validation macro F1이 가장 높은 후보를 기준으로, 그보다 TOLERANCE 이내인
# 후보 중 CPU 1건 추론 시간이 가장 짧은 후보를 고른다. validation 120건에서 1건은 약 0.008이다.
# 결과를 본 뒤 이 값을 바꾸지 않는다.
TOLERANCE = 0.01
RULE = (f"validation macro F1 최고점에서 {TOLERANCE} 이내인 후보 중 "
        "CPU 1건 추론 시간이 가장 짧은 후보")


def collect_candidates(dataset_dir):
    candidates = []
    for path in sorted(Path(dataset_dir).glob("*/runs/*/experiment.json")):
        run_dir = path.parent
        experiment = read_json(path)
        config = read_json(run_dir / "config.json")
        metrics = experiment["evaluation_metrics"]
        resources = experiment["resource_and_time"]
        weights = run_dir / ("baseline.joblib" if config["kind"] == "baseline"
                             else "checkpoint/model.safetensors")
        candidates.append({
            "run_dir": run_dir.relative_to(OUTPUT_ROOT).as_posix(),
            "learner": config["learner"], "run_name": run_dir.name, "kind": config["kind"],
            "model": config["model"],
            "model_alias": config.get("model_alias", "baseline" if config["kind"] == "baseline"
                                      else config["model"].split("/")[-1]),
            "learning_rate": config.get("learning_rate"),
            "epochs": config.get("epochs"), "best_epoch": metrics.get("best_epoch"),
            "max_length": config.get("max_length"), "seed": config["seed"],
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_weighted_f1": metrics["weighted_f1"],
            "train_time_sec": resources["train_time_sec"],
            "cpu_latency_ms": resources["inference_latency_per_sample_ms"],
            "peak_vram_mb": resources["peak_vram_mb"],
            "params": experiment["model_info"]["model_size_params"],
            "weights_available": weights.exists(),
            "labels": config["labels"], "dataset": config["dataset"],
        })
    return candidates


def choose(candidates):
    best_f1 = max(row["validation_macro_f1"] for row in candidates)
    close = [row for row in candidates if best_f1 - row["validation_macro_f1"] <= TOLERANCE]
    # 동점이면 macro F1이 높은 쪽, 그다음 실행 이름 순으로 정해 결과가 늘 같게 한다.
    return min(close, key=lambda row: (row["cpu_latency_ms"], -row["validation_macro_f1"],
                                       row["run_dir"]))


def write_table(candidates, selected, directory):
    columns = ["learner", "run_name", "kind", "model", "learning_rate", "best_epoch",
               "validation_accuracy", "validation_macro_f1", "validation_weighted_f1",
               "train_time_sec", "cpu_latency_ms", "peak_vram_mb", "params",
               "weights_available"]
    ordered = sorted(candidates, key=lambda row: -row["validation_macro_f1"])
    with (directory / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)

    def cell(value):
        if isinstance(value, float):
            if 0 < value < 1e-3:  # 학습률처럼 아주 작은 값은 지수로 쓴다.
                return f"{value:.0e}"
            return f"{value:.4f}" if value < 10 else f"{value:.1f}"
        return "-" if value is None else str(value)

    lines = [f"# 분류 모델 비교 ({DATASET}, validation)", "",
             f"- 선정 규칙: {RULE}", f"- 선정: `{selected['run_dir']}`", f"- 작성: {now_iso()}",
             "", "| 선정 | " + " | ".join(columns) + " |",
             "|---|" + "---|" * len(columns)]
    for row in ordered:
        mark = "**✔**" if row is selected else ""
        lines.append(f"| {mark} | " + " | ".join(cell(row[key]) for key in columns) + " |")
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_model(dataset_dir=OUTPUT_ROOT / DATASET, selection_dir=SELECTION_DIR):
    selection_dir = Path(selection_dir)
    if (selection_dir / "test_metrics.json").exists():
        raise FileExistsError("이미 test 평가를 마쳤습니다. test를 본 뒤에는 다시 선정하지 않습니다.")
    candidates = collect_candidates(dataset_dir)
    kinds = {row["kind"] for row in candidates}
    if not {"baseline", "transformer"} <= kinds:
        raise ValueError("기준 모델(step03)과 사전학습 모델(step04) 실행이 각각 1개 이상 필요합니다.")
    if len({tuple(row["labels"]) for row in candidates}) != 1 or \
            len({row["dataset"] for row in candidates}) != 1:
        raise ValueError("같은 데이터와 라벨 순서로 학습한 실험끼리 비교하세요.")

    selected = choose(candidates)
    baselines = [row for row in candidates if row["kind"] == "baseline"]
    # test에서 함께 보고할 기준 모델도 지금 고정한다.
    baseline_reference = max(baselines, key=lambda row: (row["validation_macro_f1"],
                                                         -row["cpu_latency_ms"]))
    for row in candidates:
        print(f"{row['learner']:>13} {row['run_name']:<40} macro F1 "
              f"{row['validation_macro_f1']:.4f}  CPU {row['cpu_latency_ms']:.1f}ms")

    selection_dir.mkdir(parents=True, exist_ok=True)
    if (selection_dir / "selected.json").exists():
        # 이전 선정 결과는 지우지 않고 기록으로 남긴다.
        history = selection_dir / "history"
        history.mkdir(exist_ok=True)
        shutil.copy2(selection_dir / "selected.json", history / f"selected_{stamp()}.json")
    selection = {"selected": selected, "baseline_reference": baseline_reference,
                 "rule": RULE, "tolerance": TOLERANCE, "selected_by": "validation",
                 "candidates": candidates, "selected_at": now_iso()}
    write_json(selection_dir / "selected.json", selection)
    write_table(candidates, selected, selection_dir)
    for name, row in (("selected", selected), ("baseline_reference", baseline_reference)):
        metrics = read_json(OUTPUT_ROOT / row["run_dir"] / "validation_metrics.json")
        plot_confusion_matrix(metrics["confusion_matrix"], row["labels"],
                              selection_dir / f"confusion_validation_{name}.png",
                              f"{row['run_name']} (validation)")

    print("선택한 실험:", selected["run_dir"])
    if not selected["weights_available"]:
        print("주의: 선택한 실험의 가중치가 이 컴퓨터에 없습니다. 받아 오거나 같은 설정으로 다시 학습하세요.")
    print("비교표:", selection_dir / "comparison.md")
    return selection


def main():
    select_model()


if __name__ == "__main__":
    main()
