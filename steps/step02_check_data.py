"""학습과 검증 데이터의 빈 값, 라벨, 상황 묶음 중복을 확인한다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step02_check_data.py
"""

import json
import unicodedata
from collections import Counter

from lesson_settings import DATASET, ROOT
from step01_read_data import read_labels, read_rows

DATA_DIR = ROOT / "data" / DATASET
REQUIRED_COLUMNS = {"id", "text", "label", "group_id", "source"}


def check_splits(splits: dict[str, list[dict]], labels: list[str]) -> dict:
    seen = {"id": set(), "group_id": set(), "text": set()}
    report = {}
    for split_name, rows in splits.items():
        if not rows:
            raise ValueError(f"{split_name}: 데이터가 비어 있습니다.")
        for row in rows:
            if not REQUIRED_COLUMNS.issubset(row):
                raise ValueError(f"{split_name}: 필수 열이 빠졌습니다.")
            if any(not isinstance(row[key], str) or not row[key].strip()
                   for key in REQUIRED_COLUMNS):
                raise ValueError(f"{split_name}: 필수 값이 비어 있습니다.")
        if {row["label"] for row in rows} != set(labels):
            raise ValueError(f"{split_name}: 라벨 목록과 데이터가 다릅니다.")

        # 검사할 때만 유니코드와 공백을 정규화한다. 원본 CSV는 수정하지 않는다.
        current = {
            "id": {row["id"] for row in rows},
            "group_id": {row["group_id"] for row in rows},
            "text": {" ".join(unicodedata.normalize("NFKC", row["text"]).split())
                     for row in rows},
        }
        if len(current["id"]) != len(rows):
            raise ValueError(f"{split_name}: 중복된 id가 있습니다.")
        group_labels = {}
        for row in rows:
            group_labels.setdefault(row["group_id"], set()).add(row["label"])
        if any(len(values) != 1 for values in group_labels.values()):
            raise ValueError(f"{split_name}: 한 상황 묶음에 서로 다른 라벨이 있습니다.")
        for key, previous in seen.items():
            overlap = previous & current[key]
            if overlap:
                raise ValueError(f"{split_name}: 분할 사이에 {key} 중복이 있습니다.")
            previous.update(current[key])
        report[split_name] = {
            "rows": len(rows), "groups": len(current["group_id"]),
            "class_counts": dict(Counter(row["label"] for row in rows)),
        }
    return report


def main():
    # test는 마지막 평가 단계에서 처음 읽는다.
    splits = {name: read_rows(DATA_DIR / f"{name}.csv")
              for name in ("train", "validation")}
    report = check_splits(splits, read_labels(DATA_DIR))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("train과 validation의 필수 값, 라벨, 분할 중복 검사를 통과했습니다.")


if __name__ == "__main__":
    main()
