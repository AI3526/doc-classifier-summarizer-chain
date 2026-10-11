"""저장한 생성 비교 기록을 검수한다. 모델을 호출하거나 답변을 다시 만들지 않는다.

교안의 실행 순서에는 포함하지 않는다. 튜터가 결과를 확인할 때 사용한다.
COMPARISON_DIR를 확인할 결과 폴더로 바꾼 뒤 이 파일을 실행한다.
"""

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

STEPS = Path(__file__).resolve().parents[1] / "steps"
sys.path.insert(0, str(STEPS))

from lesson_settings import LEARNER_DIR
from step07_call_models import valid_output
from step09_compare import summarize_provider

COMPARISON_DIR = LEARNER_DIR / "comparisons" / "development01"


def verify_comparison(directory):
    directory = Path(directory)
    records = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    expected_count = summary["cases"] * summary["repeats"] * 2
    assert len(records) == expected_count, "예상 호출 수와 저장된 행 수가 다릅니다."
    assert not (directory / "warmup.json").exists(), "현재 실습은 예열 호출을 하지 않습니다."

    case_ids = {row["case_id"] for row in records}
    assert len(case_ids) == summary["cases"], "입력 개수가 다릅니다."
    for case_id in case_ids:
        case_records = [row for row in records if row["case_id"] == case_id]
        actual = Counter((row["repeat"], row["provider"]) for row in case_records)
        expected = Counter(
            (repeat, provider)
            for repeat in range(1, summary["repeats"] + 1)
            for provider in ("ollama", "openai")
        )
        assert actual == expected, "제공자별 반복 횟수가 다릅니다."
        for field in ("input_text", "classification", "policy", "prompt"):
            values = {json.dumps(row[field], ensure_ascii=False, sort_keys=True)
                      for row in case_records}
            assert len(values) == 1, f"같은 입력의 {field} 값이 달라졌습니다."

    for row in records:
        assert row["text"] == row["response_text"], "응답 원문이 달라졌습니다."
        assert row["output_format_valid"] == valid_output(row["response_text"])
        assert math.isfinite(row["latency_seconds"]) and row["latency_seconds"] >= 0
        if row["error"]:
            assert row["response_text"] == "", "실패가 임의의 답변으로 바뀌었습니다."
            assert row["generation_complete"] is False
        assert not {"mock", "prompt_sha256", "metadata"}.intersection(row)

    for provider in ("ollama", "openai"):
        assert summary["providers"][provider] == summarize_provider(records, provider)

    with (directory / "human_scores.csv").open(encoding="utf-8-sig", newline="") as stream:
        scores = list(csv.DictReader(stream))
    assert len(scores) == len(records), "사람 검토 표의 행 수가 다릅니다."
    for record, score in zip(records, scores, strict=True):
        assert score["case_id"] == record["case_id"]
        assert score["provider"] == record["provider"]
        assert score["response"] == record["response_text"]
    return {
        "cases": len(case_ids), "records": len(records),
        "errors": sum(bool(row["error"]) for row in records),
        "format_valid": sum(row["output_format_valid"] for row in records),
        "generation_complete": sum(row["generation_complete"] for row in records),
        "human_quality_verified": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify_comparison(COMPARISON_DIR), ensure_ascii=False, indent=2))
