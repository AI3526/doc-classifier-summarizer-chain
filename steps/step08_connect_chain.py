"""08. 선택한 분류 모델과 정책, 실제 생성 호출을 LangChain으로 연결한다.

체인: 분류(학습한 모델) → 정책·프롬프트 선택(예측 라벨 기준) → 생성(LLM) → 출력 형식 검사
라우팅 체인: 기밀 문서는 로컬 모델 목록, 그 외는 기본 모델 목록으로 분기(RunnableBranch)하고,
생성 모델이 연결 실패·시간 초과면 목록의 다음 모델로 넘긴다(with_fallbacks).
입력은 generation_development.jsonl의 text만 사용한다. 정답과 채점 기준 필드는 읽지 않는다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step08_connect_chain.py
uv run python steps/step08_connect_chain.py --index 3 --target qwen2.5-7b
"""

import argparse
import json

import lesson_settings as lesson
from common.preprocess import extract_body
from langchain_core.runnables import RunnableBranch, RunnableLambda
from prompts.document_policies import (
    POLICIES,
    PROMPT_VERSION,
    build_prompt,
    parse_output,
    response_schema,
)
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


def create_chain(classifier, extract=lesson.EXTRACT_BODY):
    """분류 → 정책·프롬프트까지. 같은 준비 결과를 여러 생성 모델에 줄 때 쓴다.

    입력은 문서 문자열, 또는 {"text": ..., 그 밖의 값(예: confidential)} 딕셔너리.
    extract=True면 참고 자료 구간을 떼고 본문만 분류·요약한다(표시가 없으면 원문 그대로).
    """
    def classify(value):
        extra = {key: item for key, item in value.items() if key != "text"}             if isinstance(value, dict) else {}
        original = (value["text"] if isinstance(value, dict) else value).strip()
        text = extract_body(original) if extract else original
        preprocessing = {"body_extracted": text != original, "original_chars": len(original),
                         "used_chars": len(text)}
        classification = predict_text(text, classifier)
        # 분류 결과만 반환하면 뒤 단계에서 원문을 잃으므로 함께 전달한다.
        return {**extra, "text": text, "classification": classification,
                "preprocessing": preprocessing}

    # 왼쪽 함수가 반환한 딕셔너리가 오른쪽 함수의 인자가 된다.
    return RunnableLambda(classify) | RunnableLambda(apply_policy)


def generate_prepared(prepared, target, structured=False):
    """체인에서 만든 프롬프트를 그대로 생성 모델에 보내고, 예측 라벨의 형식으로 검사한다.

    structured=True면 예측 라벨의 출력 스키마로 구조화 출력을 요청한다.
    """
    label = prepared["classification"]["label"]
    result = call_model(target, prepared["prompt"],
                        response_format=response_schema(label) if structured else None)
    parsed = parse_output(result["text"], label)
    return {**result, "format_valid_strict": parsed["strict"],
            "format_valid_lenient": parsed["lenient"], "format_error": parsed["error"],
            "parsed": parsed["data"]}


def create_service_chain(classifier, target, structured=lesson.STRUCTURED_OUTPUT):
    """분류 → 정책·프롬프트 → 생성 전체 체인. 서비스는 이 체인 하나를 호출한다."""
    target = resolve_target(target) if isinstance(target, str) else target

    def generate(item):
        return {**item, "generation": generate_prepared(item, target, structured)}

    return create_chain(classifier) | RunnableLambda(generate)


# 연결 실패·시간 초과처럼 다른 모델로 넘겨 볼 만한 실패. 형식 오류 등은 넘기지 않는다.
RETRYABLE_ERRORS = {"ConnectError", "APIConnectionError", "AuthenticationError", "RuntimeError",
                    "ReadTimeout", "ConnectTimeout", "TimeoutException", "APITimeoutError"}


class GenerationUnavailable(Exception):
    """생성 모델이 응답하지 못함. with_fallbacks가 이 예외를 받아 다음 모델로 넘긴다."""

    def __init__(self, target, generation, attempts, item):
        super().__init__(f"{target}: {generation['error']}")
        self.target, self.generation, self.attempts = target, generation, attempts
        self.item = item  # 분류 결과를 실패 응답에도 돌려주기 위해 보관


def is_retryable(generation):
    error = generation["error"] or ""
    return generation["status"] == "ERROR" and (error in RETRYABLE_ERRORS or "Timeout" in error)


def generation_step(target, structured):
    """생성 단계 하나. 넘겨 볼 만한 실패면 예외를 내고, 시도 기록을 요청 단위 목록에 쌓는다.

    with_fallbacks는 모든 대체가 실패하면 첫 번째 예외를 다시 던지므로, 시도 기록은 예외가 아니라
    요청마다 하나씩 만드는 attempt_log 목록(모든 단계가 같은 목록을 공유)에 남긴다.
    """
    def generate(item):
        rest = {key: value for key, value in item.items() if key != "previous_error"}
        log = rest.setdefault("attempt_log", [])
        generation = generate_prepared(item, target, structured)
        log.append({"target": target, "status": generation["status"],
                    "error": generation["error"]})
        if is_retryable(generation):
            raise GenerationUnavailable(target, generation, log, rest)
        return {**rest, "generation": generation, "used_target": target, "attempts": list(log)}

    return RunnableLambda(generate)


def fallback_chain(targets, structured):
    first, *rest = [generation_step(target, structured) for target in targets]
    if not rest:
        return first
    return first.with_fallbacks(rest, exceptions_to_handle=(GenerationUnavailable,),
                                exception_key="previous_error")


def routing_plans(default_target):
    """기밀 여부에 따른 생성 모델 순서. 기밀 문서는 로컬 모델만 쓴다."""
    def unique(names):
        return list(dict.fromkeys(names))

    confidential = unique([lesson.CONFIDENTIAL_TARGET, lesson.LOCAL_FALLBACK_TARGET])
    if any(lesson.GENERATION_CANDIDATES[name]["provider"] != "ollama" for name in confidential):
        raise ValueError("기밀 문서용 생성 모델은 로컬(ollama) 모델이어야 합니다.")
    normal = unique([default_target, lesson.CONFIDENTIAL_TARGET, lesson.LOCAL_FALLBACK_TARGET])
    return {"confidential": confidential, "normal": normal}


def create_routed_chain(classifier, default_target, structured=lesson.STRUCTURED_OUTPUT):
    """분류 → 정책 → (기밀 여부로 분기) → 생성(대체 경로 포함). 입력: {"text", "confidential"}."""
    plans = routing_plans(default_target)
    branch = RunnableBranch(
        (lambda item: bool(item.get("confidential")), fallback_chain(plans["confidential"],
                                                                     structured)),
        fallback_chain(plans["normal"], structured))
    # 요청마다 새 시도 기록 목록을 붙인다(대체 단계들이 같은 목록을 공유).
    start_log = RunnableLambda(lambda item: {**item, "attempt_log": []})
    return create_chain(classifier) | start_log | branch, plans


def compare_prepared(prepared, targets, structured=lesson.STRUCTURED_OUTPUT):
    # 같은 분류 결과와 프롬프트를 모든 생성 모델이 공유한다.
    return {"classification": prepared["classification"],
            "prompt_version": prepared["prompt_version"],
            "results": [generate_prepared(prepared, target, structured)
                        for target in targets]}


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
