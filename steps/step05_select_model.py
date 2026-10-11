"""팀 전체 실행의 validation 결과를 비교해 서비스에 사용할 분류기 하나를 고른다.

artifacts/{dataset}/*/runs/*/experiment.json을 모두 모아 기준 모델과 사전학습 모델을
같은 표로 비교한다. seed만 다른 실행은 한 설정으로 묶어 평균으로 비교한다.
선정 규칙은 결과를 보기 전에 정한 아래 규칙을 그대로 쓴다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step05_select_model.py
"""

import csv
import json
import shutil
import statistics
from pathlib import Path

from common.plots import plot_confusion_matrix
from common.records import now_iso, read_json, stamp, write_json
from lesson_settings import DATASET, OUTPUT_ROOT, SELECTION_DIR

# 선정 규칙: 설정별 validation macro F1 평균(seed 평균)이 가장 높은 설정을 기준으로,
# 그보다 TOLERANCE 이내인 설정 중 CPU 1건 추론 시간이 가장 짧은 설정을 고른다.
# 고른 설정 안에서는 validation macro F1이 가장 높은 seed의 가중치를 쓴다.
# validation 120건에서 1건은 약 0.008이다. 결과를 본 뒤 이 값을 바꾸지 않는다.
TOLERANCE = 0.01
RULE = (f"설정별 validation macro F1 평균의 최고점에서 {TOLERANCE} 이내인 설정 중 "
        "CPU 1건 추론 시간이 가장 짧은 설정, 그 안에서 macro F1이 가장 높은 seed")
# seed를 제외하고 이 값이 모두 같으면 같은 설정으로 본다.
CONFIG_KEYS = ("kind", "model", "learning_rate", "epochs", "batch_size",
               "accumulation_steps", "max_length", "ngram_range")


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
            "epochs": config.get("epochs"), "batch_size": config.get("batch_size"),
            "accumulation_steps": config.get("accumulation_steps"),
            "max_length": config.get("max_length"), "ngram_range": config.get("ngram_range"),
            "seed": config["seed"], "best_epoch": metrics.get("best_epoch"),
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


def config_key(row):
    return json.dumps([row.get(key) for key in CONFIG_KEYS])


def group_candidates(candidates):
    """seed만 다른 실행을 한 설정으로 묶고 평균과 표준편차를 계산한다."""
    groups = {}
    for row in candidates:
        groups.setdefault(config_key(row), []).append(row)
    summaries = []
    for key, rows in groups.items():
        scores = [row["validation_macro_f1"] for row in rows]
        first = rows[0]
        summaries.append({
            "config_key": key, "kind": first["kind"], "model": first["model"],
            "model_alias": first["model_alias"], "learning_rate": first["learning_rate"],
            "seeds": sorted(row["seed"] for row in rows), "n_seeds": len(rows),
            "macro_f1_mean": statistics.mean(scores),
            # seed가 하나면 흩어짐을 알 수 없으므로 None으로 둔다.
            "macro_f1_std": statistics.stdev(scores) if len(scores) > 1 else None,
            "accuracy_mean": statistics.mean(row["validation_accuracy"] for row in rows),
            "cpu_latency_ms_mean": statistics.mean(row["cpu_latency_ms"] for row in rows),
            "train_time_sec_mean": statistics.mean(row["train_time_sec"] for row in rows),
            "peak_vram_mb_max": max((row["peak_vram_mb"] for row in rows
                                     if row["peak_vram_mb"] is not None), default=None),
            "params": first["params"],
            "runs": [row["run_dir"] for row in rows],
        })
    return summaries


def best_run(group, candidates):
    rows = [row for row in candidates if row["run_dir"] in group["runs"]]
    # 같은 점수면 seed가 작은 쪽을 써서 결과가 늘 같게 한다.
    return max(rows, key=lambda row: (row["validation_macro_f1"], -row["seed"]))


def choose_group(candidates):
    groups = group_candidates(candidates)
    best_f1 = max(group["macro_f1_mean"] for group in groups)
    close = [group for group in groups if best_f1 - group["macro_f1_mean"] <= TOLERANCE]
    # 동점이면 평균 macro F1이 높은 쪽, 그다음 설정 이름 순으로 정한다.
    return min(close, key=lambda group: (group["cpu_latency_ms_mean"],
                                         -group["macro_f1_mean"], group["config_key"]))


def choose(candidates):
    """선정 규칙에 따라 서비스에 쓸 실행 하나를 돌려준다."""
    return best_run(choose_group(candidates), candidates)


def cell(value):
    if isinstance(value, float):
        if 0 < value < 1e-3:  # 학습률처럼 아주 작은 값은 지수로 쓴다.
            return f"{value:.0e}"
        return f"{value:.4f}" if value < 10 else f"{value:.1f}"
    return "-" if value is None else str(value)


def markdown_table(rows, columns, is_selected):
    lines = ["| 선정 | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for row in rows:
        mark = "**✔**" if is_selected(row) else ""
        lines.append(f"| {mark} | " + " | ".join(cell(row[key]) for key in columns) + " |")
    return lines


def write_tables(candidates, groups, selected_group, selected, directory):
    run_columns = ["learner", "run_name", "kind", "model", "learning_rate", "seed", "best_epoch",
                   "validation_accuracy", "validation_macro_f1", "validation_weighted_f1",
                   "train_time_sec", "cpu_latency_ms", "peak_vram_mb", "params",
                   "weights_available"]
    group_columns = ["model_alias", "model", "learning_rate", "n_seeds", "macro_f1_mean",
                     "macro_f1_std", "accuracy_mean", "train_time_sec_mean",
                     "cpu_latency_ms_mean", "peak_vram_mb_max", "params"]
    runs = sorted(candidates, key=lambda row: -row["validation_macro_f1"])
    groups = sorted(groups, key=lambda group: -group["macro_f1_mean"])
    for filename, rows, columns in (("comparison.csv", runs, run_columns),
                                    ("comparison_by_config.csv", groups, group_columns)):
        with (directory / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    lines = [f"# 분류 모델 비교 ({DATASET}, validation)", "",
             f"- 선정 규칙: {RULE}", f"- 선정: `{selected['run_dir']}`", f"- 작성: {now_iso()}",
             "", "## 설정별 (seed 평균)", "",
             *markdown_table(groups, group_columns,
                             lambda group: group["config_key"] == selected_group["config_key"]),
             "", "## 실행별", "",
             *markdown_table(runs, run_columns,
                             lambda row: row["run_dir"] == selected["run_dir"])]
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

    groups = group_candidates(candidates)
    selected_group = choose_group(candidates)
    selected = best_run(selected_group, candidates)
    # test에서 함께 보고할 기준 모델도 지금 고정한다.
    baseline_groups = [group for group in groups if group["kind"] == "baseline"]
    baseline_group = max(baseline_groups, key=lambda group: (group["macro_f1_mean"],
                                                             -group["cpu_latency_ms_mean"]))
    baseline_reference = best_run(baseline_group, candidates)
    for group in sorted(groups, key=lambda group: -group["macro_f1_mean"]):
        spread = "" if group["macro_f1_std"] is None else f" ± {group['macro_f1_std']:.4f}"
        print(f"{group['model_alias']:>11} lr={cell(group['learning_rate']):>6} "
              f"seeds={group['n_seeds']}  macro F1 {group['macro_f1_mean']:.4f}{spread}  "
              f"CPU {group['cpu_latency_ms_mean']:.1f}ms")

    selection_dir.mkdir(parents=True, exist_ok=True)
    if (selection_dir / "selected.json").exists():
        # 이전 선정 결과는 지우지 않고 기록으로 남긴다.
        history = selection_dir / "history"
        history.mkdir(exist_ok=True)
        shutil.copy2(selection_dir / "selected.json", history / f"selected_{stamp()}.json")
    selection = {"selected": selected, "selected_group": selected_group,
                 "baseline_reference": baseline_reference, "rule": RULE,
                 "tolerance": TOLERANCE, "selected_by": "validation",
                 "candidates": candidates, "groups": groups, "selected_at": now_iso()}
    write_json(selection_dir / "selected.json", selection)
    write_tables(candidates, groups, selected_group, selected, selection_dir)
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
