"""실제 1번 시트를 바탕으로 가상 채점자 3명(2~4번)의 시트를 만든다. 실제 사람 채점이 아니다.

kappa·PABAK 같은 일치도 지표가 어떻게 계산되는지 보려는 미리보기용이다.
- human_dummy/human_sheet_rater1.csv: 실제 1번 시트를 그대로 복사한다(사용자가 채점한 값).
- human_dummy/human_sheet_rater2~4.csv: 1번 시트를 조금씩 바꾼 가상 채점. 실제 시트(human/)는 건드리지 않는다.

변형 규칙(모두 가정이며 실제 사람 사이의 불일치를 측정한 값이 아니다)
- 기본 변형 확률은 낮게 두고, 1번 채점자가 메모나 칸 오류 이유를 남긴 항목(애매했던 항목)에서는 2.5배로 높인다.
- 채점자마다 성향이 다르다: 2번 엄격, 3번 관대, 4번 중간(엄격한 채점자는 칸을 틀렸다고, 필요한 내용을 덜 포함했다고 볼 확률이 높다).
- 내용을 읽을 수 없는 응답(필수 포인트가 모두 누락)의 필수 포인트 판정은 바꾸지 않는다.
- 바꾼 칸은 notes에 남기고, 모든 가상 행에 "가상 채점"을 적는다.

주의: 이 시트들은 1번 시트의 변형이라 일치도가 실제보다 높게 나온다. 집계 결과는 is_dummy=true이며
보고서에는 "가상 채점(미리보기용)"으로만 쓴다.

uv run python scripts/make_dummy_from_rater1.py --dir <비교 폴더>
uv run python steps/step09b_score.py aggregate --dir <비교 폴더> --dummy
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steps"))

from step09b_score import latest_comparison

SEED = 11
BORDERLINE = 2.5  # 1번 채점자가 메모를 남긴 항목의 변형 확률 배율
# 성향: +1 엄격, 0 중간, -1 관대
RATERS = {"rater2": +1, "rater3": -1, "rater4": 0}
LEVELS = ["포함", "부분", "누락"]
FLAGS = ("numeric_error", "foreign_language", "awkward_sentence")
FLAG_FLIP = {"numeric_error": 0.02, "foreign_language": 0.01, "awkward_sentence": 0.10}


def clamp(value, low=0.0, high=0.6):
    return max(low, min(high, value))


def perturb(row, bias, rng):
    """1번 채점 한 행을 바탕으로 가상 채점 한 행을 만든다. 바뀐 칸 이름 목록도 돌려준다."""
    new, changed = dict(row), []
    m = BORDERLINE if (row["notes"].strip() or row["wrong_field_reasons"].strip()) else 1.0

    rp_cols = [c for c in ("rp1", "rp2", "rp3") if row[c].strip()]
    if not all(row[c] == "누락" for c in rp_cols):  # 읽을 수 없는 응답은 그대로
        for col in rp_cols:
            if rng.random() < clamp(0.05 * m):
                step = 1 if rng.random() < 0.5 + 0.25 * bias else -1  # +1: 더 엄격하게(덜 포함)
                index = max(0, min(2, LEVELS.index(row[col]) + step))
                if LEVELS[index] != row[col]:
                    new[col] = LEVELS[index]
                    changed.append(col)
    for col in ("fc1", "fc2"):
        if row[col].strip() and rng.random() < clamp(0.02 * m):
            new[col] = "0" if row[col] == "1" else "1"
            changed.append(col)

    text = row["unsupported_claims"].strip()
    if text and rng.random() < clamp(0.10 * m * (1 - 0.5 * bias)):
        new["unsupported_claims"] = ""
        changed.append("unsupported_claims")
    elif not text and rng.random() < clamp(0.03 * m * (1 + 0.5 * bias)):
        new["unsupported_claims"] = "(가상) 원문에 없는 내용으로 보임"
        changed.append("unsupported_claims")
    for col in FLAGS:
        if rng.random() < clamp(FLAG_FLIP[col] * m):
            new[col] = "0" if row[col] == "1" else "1"
            changed.append(col)

    fields = [x.strip() for x in row["checked_fields"].split(",") if x.strip()]
    wrong = {x.strip() for x in row["wrong_fields"].split(",") if x.strip()}
    new_wrong = set(wrong)
    for name in fields:
        if name in wrong and rng.random() < clamp(0.15 * m * (1 - 0.5 * bias)):
            new_wrong.discard(name)
        elif name not in wrong and rng.random() < clamp(0.05 * m * (1 + 0.5 * bias)):
            new_wrong.add(name)
    if new_wrong != wrong:
        new["wrong_fields"] = ", ".join(name for name in fields if name in new_wrong)
        new["wrong_field_reasons"] = "(가상) " + ", ".join(
            f"{name}: 규칙과 다르게 판단" for name in fields if name in new_wrong ^ wrong)
        changed.append("wrong_fields")
    elif row["wrong_field_reasons"].strip():
        new["wrong_field_reasons"] = "(가상) " + row["wrong_field_reasons"].strip()
    note = "가상 채점"
    if changed:
        note += f" (1번과 다르게 판단: {', '.join(changed)})"
    new["notes"] = note
    return new, changed


def main():
    parser = argparse.ArgumentParser(description="1번 시트 기반 가상 채점자 3명 생성")
    parser.add_argument("--dir", type=Path)
    args = parser.parse_args()
    directory = args.dir or latest_comparison()
    source, target = directory / "human", directory / "human_dummy"
    with (source / "human_sheet_rater1.csv").open(encoding="utf-8-sig", newline="") as stream:
        base = list(csv.DictReader(stream))
    if not any(row["rp1"].strip() for row in base):
        raise SystemExit("1번 시트가 비어 있습니다. 먼저 실제 채점을 해 주세요.")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)  # 채점 정답표(human_key.json)와 안내문도 함께 복사
    (target / "README.md").write_text(
        "# 가상 채점 (미리보기용)\n\n"
        "- `human_sheet_rater1.csv`는 **실제 사람(사용자)이 채점한 1번 시트를 그대로 복사**한 것입니다.\n"
        "- `human_sheet_rater2~4.csv`는 `scripts/make_dummy_from_rater1.py`가 1번 시트를 조금씩 "
        "바꿔 만든 **가상 데이터**입니다(2번 엄격, 3번 관대, 4번 중간. 1번이 메모를 남긴 항목에서 "
        "더 자주 다르게 판단).\n"
        "- 1번 시트의 변형이라 채점자 간 일치도가 실제 사람 사이보다 높게 나옵니다. kappa 같은 "
        "지표가 어떻게 계산되는지 보는 용도이며, 보고서에는 \"가상 채점(미리보기용)\"으로만 씁니다.\n"
        "- 실제 시트(`human/`)는 건드리지 않았습니다.\n", encoding="utf-8")
    for rater, bias in RATERS.items():
        rng = random.Random(f"{SEED}-{rater}")
        rows, total = [], 0
        for row in base:
            new, changed = perturb(row, bias, rng)
            rows.append(new)
            total += len(changed)
        path = target / f"human_sheet_{rater}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(base[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{rater} (성향 {bias:+d}): 1번과 다른 칸 {total}개 / 전체 채점 칸 약 {len(base) * 12}개")
    print("가상 시트:", target)


if __name__ == "__main__":
    main()
