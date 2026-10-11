"""저장된 실험의 validation 점수를 비교해 사용할 모델 하나를 고른다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step05_select_model.py
"""

import json
from pathlib import Path

from lesson_settings import LEARNER_DIR

RUN_NAMES = ["lr2e5", "lr5e5"]
# 다른 팀원의 실험은 "learner02/lr5e5"처럼 적는다.


def select_model(learner_dir=LEARNER_DIR, run_names=RUN_NAMES):
    learner_dir = Path(learner_dir)
    if len(set(run_names)) < 2:
        raise ValueError("서로 다른 실험을 두 개 이상 준비하세요.")
    candidates = []
    label_orders = set()
    for name in run_names:
        parts = name.split("/")
        if len(parts) == 1:
            run_learner, run_name = learner_dir.name, parts[0]
        elif len(parts) == 2:
            run_learner, run_name = parts
        else:
            raise ValueError("실험 이름은 run 또는 learner/run 형식이어야 합니다.")
        run = learner_dir.parent / run_learner / run_name
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "validation_metrics.json").read_text(encoding="utf-8"))
        if not (run / "checkpoint" / "model.safetensors").exists():
            raise FileNotFoundError(f"저장된 가중치가 없습니다: {run}")
        score = metrics["macro_f1"]
        if not 0 <= score <= 1:
            raise ValueError("validation macro F1이 올바른 범위가 아닙니다.")
        label_orders.add(tuple(config["labels"]))
        candidates.append({"run_name": run_name, "run_learner": run_learner,
                           "validation_macro_f1": score})
        print(run_learner, run_name, "validation macro F1:", round(score, 4))

    if len(label_orders) != 1:
        raise ValueError("같은 라벨 순서로 학습한 실험끼리 비교하세요.")
    # 선택에 test 점수를 사용하지 않는다. 동점이면 RUN_NAMES의 앞 후보가 남는다.
    best = max(candidates, key=lambda candidate: candidate["validation_macro_f1"])
    selection = {**best, "selected_by": "validation_macro_f1", "candidates": candidates}
    learner_dir.mkdir(parents=True, exist_ok=True)
    (learner_dir / "selected.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print("선택한 실험:", best["run_name"])
    return selection


def main():
    select_model()


if __name__ == "__main__":
    main()
