"""CSV 한 행을 파이썬 사전으로 읽고 문장과 라벨을 확인한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step01_read_data.py
"""

import csv
import json
from collections import Counter
from pathlib import Path

from lesson_settings import DATASET, ROOT

DATA_DIR = ROOT / "data" / DATASET
PREVIEW_ROWS = 3


def read_rows(path: Path) -> list[dict[str, str]]:
    # DictReader는 첫 줄을 키로 사용한다. 모든 셀은 문자열로 읽힌다.
    # utf-8-sig는 Excel용 BOM이 있는 CSV도 같은 방식으로 읽는다.
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_labels(data_dir: Path) -> list[str]:
    value = json.loads((Path(data_dir) / "labels.json").read_text(encoding="utf-8-sig"))
    labels = value["labels"]
    if (not isinstance(labels, list) or not labels
            or not all(isinstance(label, str) and label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("labels.json에는 중복 없는 라벨 문자열 목록이 필요합니다.")
    return labels


def main():
    rows = read_rows(DATA_DIR / "train.csv")
    labels = read_labels(DATA_DIR)
    print("읽은 파일:", DATA_DIR / "train.csv")
    print("행 수:", len(rows))
    print("라벨 순서:", labels)
    print("라벨별 행 수:", dict(Counter(row["label"] for row in rows)))
    for row in rows[:PREVIEW_ROWS]:
        print("\nID:", row["id"])
        print("문장:", row["text"])
        print("정답:", row["label"], "상황 묶음:", row["group_id"])


if __name__ == "__main__":
    main()
