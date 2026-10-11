"""실행 폴더 만들기와 JSON 기록.

실행 폴더 이름 앞에 시각을 붙여 같은 학습자가 같은 이름으로 다시 실행해도 겹치지 않게 한다.
"""

import json
from datetime import datetime
from pathlib import Path


def now_iso():
    # 로컬 시간대를 붙여 팀원 기록의 시각을 비교할 수 있게 한다.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stamp():
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def new_run_dir(learner_dir, run_name):
    # artifacts/{dataset}/{learner}/runs/{YYYYMMDD-HHMMSS}_{run_name}
    run_dir = Path(learner_dir) / "runs" / f"{stamp()}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
