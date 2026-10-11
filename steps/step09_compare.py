"""09. 같은 입력과 프롬프트로 두 모델을 호출하고 비교 기록을 저장한다.

개발 입력 두 개로 코드를 확인한 뒤 최종 비교 시 INPUT_PATH를 변경한다.
최종 생성 평가 데이터의 답변을 보며 프롬프트를 고치지 않는다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step09_compare.py
"""

import csv
import json
import statistics
from pathlib import Path

from lesson_settings import DATA_DIR, DEVELOPMENT_INPUT, LEARNER_DIR
from step06_predict import load_classifier
from step07_call_models import generation_config
from step08_connect_chain import MAX_INPUT_CHARS, create_chain, generate_prepared

# 최종 비교를 시작할 때 INPUT_PATH = FINAL_INPUT_PATH로 바꾼다.
FINAL_INPUT_PATH = DATA_DIR / "generation_eval.jsonl"
INPUT_PATH = DEVELOPMENT_INPUT
REPEATS = 2
OUTPUT_NAME = "development01"
OUTPUT_DIR = LEARNER_DIR / "comparisons" / OUTPUT_NAME


def read_cases(path):
    cases = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    validate_cases(cases)
    return cases


def validate_cases(cases):
    # 빈 입력이나 중복 id를 발견하면 유료 호출 전에 멈춘다.
    if not cases:
        raise ValueError("비교 입력이 비어 있습니다.")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or any(not str(value).strip() for value in ids):
        raise ValueError("입력 id는 내용이 있어야 하고 서로 달라야 합니다.")
    for case in cases:
        text = case["text"]
        if not text.strip() or len(text) > MAX_INPUT_CHARS:
            raise ValueError("비교 입력의 text와 길이를 확인하세요.")


def summarize_provider(records, provider):
    subset = [record for record in records if record["provider"] == provider]
    success = [record for record in subset if not record["error"]]
    calls = len(subset)
    errors = calls - len(success)
    valid = sum(record["output_format_valid"] for record in subset)
    return {
        "calls": calls,
        "errors": errors,
        "error_rate_all_calls": errors / calls if calls else None,
        "mean_latency_seconds_success_only": statistics.mean(
            record["latency_seconds"] for record in success
        ) if success else None,
        "format_valid_rate_all_calls": valid / calls if calls else None,
    }


def write_human_scores(path, records):
    fields = [
        "case_id", "repeat", "provider", "model",
        "predicted_label", "expected_label", "latency_seconds", "input_tokens",
        "output_tokens", "output_format_valid", "generation_complete", "finish_reason",
        "error", "text", "response", "required_points", "forbidden_claims",
        "reviewer", "policy_correct_0_2", "required_points_0_2",
        "no_unsupported_claims_0_2", "clarity_0_2", "notes",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields}
            row.update(
                predicted_label=record["classification"]["label"],
                text=record["input_text"], response=record["response_text"],
                input_tokens=record["usage"]["input_tokens"],
                output_tokens=record["usage"]["output_tokens"],
                required_points=json.dumps(record["required_points"], ensure_ascii=False),
                forbidden_claims=json.dumps(record["forbidden_claims"], ensure_ascii=False),
            )
            # 점수는 빈칸으로 남긴다. 사람이 원문과 답변을 비교한 뒤 채운다.
            writer.writerow(row)


def run_comparison(chain, cases, output_dir, repeats=REPEATS):
    validate_cases(cases)
    if repeats < 1:
        raise ValueError("REPEATS는 1 이상이어야 합니다.")

    # 폴더가 이미 있으면 멈춘다. 앞선 결과를 덮어쓰지 않는다.
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("출력 폴더가 있습니다. OUTPUT_NAME을 새 이름으로 바꾸세요.")
    prepared = [chain.invoke(case["text"]) for case in cases]
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for case_index, (case, fixed) in enumerate(zip(cases, prepared, strict=True)):
            for repeat in range(repeats):
                # 준비한 프롬프트를 매번 그대로 사용한다. 예열 호출은 하지 않는다.
                for provider in ("ollama", "openai"):
                    generated = generate_prepared(fixed, provider)
                    record = {
                        "case_id": case["id"], "repeat": repeat + 1,
                        "input_text": fixed["text"],
                        "expected_label": case.get("expected_label"),
                        "classification": fixed["classification"],
                        "policy": fixed["policy"], "prompt": fixed["prompt"],
                        "required_points": case.get("required_points", []),
                        "forbidden_claims": case.get("forbidden_claims", []),
                        **generated, "response_text": generated["text"],
                    }
                    # 예외가 생긴 호출도 저장한다. 중단되더라도 끝난 호출 기록은 남긴다.
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    records.append(record)
                print(f"입력 {case_index + 1}/{len(cases)}, 반복 {repeat + 1} 저장")

    write_human_scores(output_dir / "human_scores.csv", records)
    summary = {
        "cases": len(cases), "repeats": repeats,
        "generation_config": generation_config(),
        "human_quality_scored": False,
        "providers": {
            provider: summarize_provider(records, provider)
            for provider in ("ollama", "openai")
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    cases = read_cases(INPUT_PATH)
    classifier = load_classifier()
    chain = create_chain(classifier)
    summary = run_comparison(chain, cases, OUTPUT_DIR)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("저장 위치:", OUTPUT_DIR)
    print("human_scores.csv에 원문과 답변을 비교한 점수와 이유를 적으세요.")


if __name__ == "__main__":
    main()
