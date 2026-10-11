"""저장한 생성 비교 기록을 검수한다. 모델을 호출하거나 답변을 다시 만들지 않는다.

uv run python scripts/verify_service.py                      # 가장 최근 비교 폴더
uv run python scripts/verify_service.py <비교 폴더 경로>
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

STEPS = Path(__file__).resolve().parents[1] / "steps"
sys.path.insert(0, str(STEPS))

from lesson_settings import LEARNER_DIR
from prompts.document_policies import parse_output
from step09_compare import summarize_target


def latest_comparison():
    folders = sorted((LEARNER_DIR / "comparisons").glob("*/summary.json"))
    if not folders:
        raise FileNotFoundError("비교 결과가 없습니다. step09_compare.py를 먼저 실행하세요.")
    return folders[-1].parent


def verify_comparison(directory):
    directory = Path(directory)
    records = [json.loads(line) for line in
               (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    warmups = json.loads((directory / "warmup.json").read_text(encoding="utf-8"))
    aliases = [target["alias"] for target in config["targets"]]
    expected_count = config["cases"] * config["repeats"] * len(aliases)
    assert len(records) == expected_count, "예상 호출 수와 저장된 행 수가 다릅니다."
    assert set(warmups) == set(aliases), "모델마다 준비 호출 기록이 있어야 합니다."

    case_ids = {row["case_id"] for row in records}
    assert len(case_ids) == config["cases"], "입력 개수가 다릅니다."
    for case_id in case_ids:
        case_records = [row for row in records if row["case_id"] == case_id]
        actual = Counter((row["repeat"], row["model_info"]["alias"]) for row in case_records)
        expected = Counter((repeat, alias) for repeat in range(1, config["repeats"] + 1)
                           for alias in aliases)
        assert actual == expected, "모델별 반복 횟수가 다릅니다."
        # 같은 입력에서는 분류 결과와 프롬프트를 모든 모델이 공유해야 한다.
        for field in ("source_text", "predicted_label", "prompt_template_used"):
            assert len({row["input_data"][field] for row in case_records}) == 1, \
                f"같은 입력의 {field} 값이 달라졌습니다."
        assert len({row["prompt"] for row in case_records}) == 1, "같은 입력의 프롬프트가 다릅니다."

    for row in records:
        result = row["result"]
        # 채점 기준이 프롬프트에 섞이지 않았는지 확인한다.
        for point in (row["scoring_reference"].get("required_points") or []) + \
                (row["scoring_reference"].get("forbidden_claims") or []):
            assert point not in row["prompt"], "채점 기준이 프롬프트에 들어갔습니다."
        parsed = parse_output(result["generated_text"], row["input_data"]["predicted_label"])
        assert result["format_valid_strict"] == parsed["strict"]
        assert result["format_valid_lenient"] == parsed["lenient"]
        latency = row["performance_metrics"]["total_time_sec"]
        assert math.isfinite(latency) and latency >= 0
        if result["status"] == "ERROR":
            assert result["generated_text"] == "", "실패가 임의의 답변으로 바뀌었습니다."
            assert result["generation_complete"] is False

    for alias in aliases:
        assert summary["targets"][alias] == summarize_target(records, alias, warmups[alias]), \
            f"{alias} 요약이 기록과 다릅니다."
    return {"directory": directory.name, "cases": len(case_ids), "records": len(records),
            "errors": sum(row["result"]["status"] == "ERROR" for row in records),
            "format_valid_strict": sum(row["result"]["format_valid_strict"] for row in records),
            "quality_scored": summary.get("quality_scored", False)}


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_comparison()
    print(json.dumps(verify_comparison(target), ensure_ascii=False, indent=2))
