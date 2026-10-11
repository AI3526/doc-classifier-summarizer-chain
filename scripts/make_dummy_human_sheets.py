"""미리보기용 가상 사람 채점 시트를 만든다. 실제 사람 채점이 아니다.

보고서와 README가 어떻게 보이는지 확인하려고, 비어 있는 human/ 시트를 human_dummy/로 복사한 뒤
채점자마다 LLM 판정을 바탕으로 일부 칸을 무작위로 바꿔 채운다. 실제 시트(human/)는 건드리지 않는다.
집계는 `step09b_score.py aggregate --dummy`로 하며 결과에 is_dummy=true가 남는다.

uv run python scripts/make_dummy_human_sheets.py --dir <비교 폴더>
"""

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steps"))

from step09b_score import FLAG_COLUMNS, STATUS_KO, latest_comparison, latest_judgements

POINT_FLIP = 0.12  # 핵심 내용 판정을 다른 값으로 바꿀 확률
FLAG_FLIP = 0.10  # 예/아니오 항목을 뒤집을 확률
SEED = 7


def main():
    parser = argparse.ArgumentParser(description="미리보기용 가상 사람 채점 시트 생성")
    parser.add_argument("--dir", type=Path)
    args = parser.parse_args()
    directory = args.dir or latest_comparison()
    source, target = directory / "human", directory / "human_dummy"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    (target / "README.md").write_text(
        "# 가상 채점 (미리보기용)\n\n이 폴더의 시트는 `scripts/make_dummy_human_sheets.py`가 LLM 판정에 "
        "무작위 변형을 더해 채운 **가상 데이터**입니다. 실제 사람 채점 결과로 쓰지 않습니다.\n",
        encoding="utf-8")
    key = json.loads((target / "human_key.json").read_text(encoding="utf-8"))
    judged = latest_judgements(directory)
    to_ko = {value: name for name, value in STATUS_KO.items()}
    for path in sorted(target.glob("human_sheet_rater*.csv")):
        rng = random.Random(f"{SEED}-{path.stem}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            check = judged[key["items"][row["item_id"]]]["checklist"]
            if check is None:  # LLM 채점이 실패한 항목은 비워 둔다.
                continue
            for item in check["required_points"]:
                status = item["status"]
                if rng.random() < POINT_FLIP:
                    status = rng.choice([s for s in to_ko if s != status])
                row[f"rp{item['index']}"] = to_ko[status]
            for item in check["forbidden_claims"]:
                violated = item["violated"] != (rng.random() < FLAG_FLIP / 2)
                row[f"fc{item['index']}"] = "1" if violated else "0"
            keep_unsupported = bool(check["unsupported_claims"]) != (rng.random() < FLAG_FLIP)
            row["unsupported_claims"] = (check["unsupported_claims"] or ["(가상) 근거 없는 내용"])[0] \
                if keep_unsupported else ""
            for field in FLAG_COLUMNS:
                row[field] = "1" if check[field] != (rng.random() < FLAG_FLIP) else "0"
            row["notes"] = "가상 채점"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("가상 시트:", target)


if __name__ == "__main__":
    main()
