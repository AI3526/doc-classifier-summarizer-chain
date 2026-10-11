"""08. 선택한 분류 모델과 정책, 실제 생성 호출을 LangChain으로 연결한다.

체인: 분류(학습한 모델) → 정책·프롬프트 선택(예측 라벨 기준) → 생성(LLM) → 출력 형식 검사
입력은 generation_development.jsonl의 text만 사용한다. 정답과 채점 기준 필드는 읽지 않는다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step08_connect_chain.py
uv run python steps/step08_connect_chain.py --index 3 --target qwen2.5-7b
"""

import argparse
import json

import lesson_settings as lesson
from langchain_core.runnables import RunnableLambda
from prompts.document_policies import POLICIES, PROMPT_VERSION, build_prompt, parse_output
from step01_read_data import read_jsonl
from step06_predict import load_classifier, predict_text
from step07_call_models import call_model, resolve_target

MAX_INPUT_CHARS = lesson.MAX_INPUT_CHARS
DEVELOPMENT_CASES = lesson.DATA_DIR / "generation_development.jsonl"


def apply_policy(item):
    # 정답 라벨이 아니라 분류기의 예측 라벨로 정책과 출력 형식을 고른다.
    label = item["classification"]["label"]
    return {**item, "policy": POLICIES[label], "prompt_version": PROMPT_VERSION,
            "prompt": build_prompt(item["text"], label)}


def create_chain(classifier):
    """분류 → 정책·프롬프트까지. 같은 준비 결과를 여러 생성 모델에 줄 때 쓴다."""
    def classify(text):
        text = text.strip()
        classification = predict_text(text, classifier)
        # 분류 결과만 반환하면 뒤 단계에서 원문을 잃으므로 함께 전달한다.
        return {"text": text, "classification": classification}

    # 왼쪽 함수가 반환한 딕셔너리가 오른쪽 함수의 인자가 된다.
    return RunnableLambda(classify) | RunnableLambda(apply_policy)


def generate_prepared(prepared, target):
    """체인에서 만든 프롬프트를 그대로 생성 모델에 보내고, 예측 라벨의 형식으로 검사한다."""
    result = call_model(target, prepared["prompt"])
    parsed = parse_output(result["text"], prepared["classification"]["label"])
    return {**result, "format_valid_strict": parsed["strict"],
            "format_valid_lenient": parsed["lenient"], "format_error": parsed["error"],
            "parsed": parsed["data"]}


def create_service_chain(classifier, target):
    """분류 → 정책·프롬프트 → 생성 전체 체인. 서비스는 이 체인 하나를 호출한다."""
    target = resolve_target(target) if isinstance(target, str) else target

    def generate(item):
        return {**item, "generation": generate_prepared(item, target)}

    return create_chain(classifier) | RunnableLambda(generate)


def compare_prepared(prepared, targets):
    # 같은 분류 결과와 프롬프트를 모든 생성 모델이 공유한다.
    return {"classification": prepared["classification"],
            "prompt_version": prepared["prompt_version"],
            "results": [generate_prepared(prepared, target) for target in targets]}


def main():
    parser = argparse.ArgumentParser(description="분류 → 정책 → 생성 체인 실행")
    parser.add_argument("--index", type=int, default=0,
                        help="generation_development.jsonl에서 사용할 행 번호")
    parser.add_argument("--target", default=lesson.GENERATION_DEFAULT,
                        help=f"후보: {', '.join(lesson.GENERATION_CANDIDATES)}")
    args = parser.parse_args()
    case = read_jsonl(DEVELOPMENT_CASES)[args.index]
    chain = create_service_chain(load_classifier(), args.target)
    result = chain.invoke(case["text"])  # text만 넘긴다.
    print("입력:", case["id"], "\n", result["text"])
    print("분류 결과:", json.dumps(result["classification"], ensure_ascii=False))
    print("선택한 정책:", result["policy"])
    print(json.dumps(result["generation"], ensure_ascii=False, indent=2))
    if result["generation"]["status"] == "ERROR":
        raise SystemExit("생성 호출이 실패했습니다. 연결 설정과 error를 확인하세요.")


if __name__ == "__main__":
    main()
