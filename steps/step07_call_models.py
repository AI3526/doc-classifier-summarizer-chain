"""07. Ollama 또는 OpenAI에 프롬프트를 보내고 실제 응답을 읽는다.

PROVIDER를 openai로 바꾸면 .env의 OPENAI_API_KEY를 사용하며 요금이 발생한다.

VS Code 터미널에서 project2-kit 폴더를 기준으로 실행한다.
uv run python steps/step07_call_models.py
"""

import json
import os
import time

import httpx
import lesson_settings as lesson
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# 먼저 ollama로 실행한 뒤 provider만 바꾸어 같은 입력을 확인한다.
PROVIDER = "ollama"
SAMPLE_INPUTS = {
    "inquiries": "배송 현황을 확인하는 방법을 알려 주세요.",
    "documents": (
        "교육장 이용 공지: 10월 8일 오후 2시부터 3시까지 3층 실습실을 점검합니다. "
        "해당 시간에 예약한 수강생은 2층 실습실을 이용해 주세요."
    ),
}
INPUT_TEXT = SAMPLE_INPUTS[lesson.DATASET]
OLLAMA_URL = lesson.OLLAMA_URL
OLLAMA_MODEL = lesson.OLLAMA_MODEL
OPENAI_MODEL = lesson.OPENAI_MODEL
MAX_OUTPUT_TOKENS = lesson.MAX_OUTPUT_TOKENS
TIMEOUT_SECONDS = lesson.TIMEOUT_SECONDS
TEMPERATURE = 0.2


class AnswerPayload(BaseModel):
    """응답 문자열이 가져야 할 필드와 타입을 정한다."""

    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    next_steps: list[str]


def valid_output(text):
    # JSON 문법과 타입만 검사한다. 답변의 근거는 사람이 따로 읽는다.
    try:
        AnswerPayload.model_validate_json(text)
        return True
    except ValidationError:
        return False


def call_ollama(prompt):
    # stream=False이면 여러 조각 대신 완성된 응답 객체 한 개를 받는다.
    # 두 제공자에 같은 prompt를 전달하고 내부 템플릿 차이는 별도로 인정한다.
    request_body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_OUTPUT_TOKENS,
            "num_ctx": 4096,
        },
    }
    with httpx.Client(timeout=TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.post(
            OLLAMA_URL.rstrip("/") + "/api/generate", json=request_body
        )
        response.raise_for_status()
        body = response.json()

    # 제공자의 응답 원문을 그대로 보존한다. 임의로 JSON을 고치지 않는다.
    return {
        "text": body["response"],
        "model": OLLAMA_MODEL,
        "usage": {
            "input_tokens": body.get("prompt_eval_count"),
            "output_tokens": body.get("eval_count"),
        },
        "generation_complete": bool(body.get("done"))
        and body.get("done_reason") != "length",
        "finish_reason": body.get("done_reason"),
    }


def call_openai(prompt):
    # 키의 존재만 확인한다. 키 값이나 요청 헤더를 출력하지 않는다.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(".env에 OPENAI_API_KEY를 설정하세요.")

    # 자동 재시도를 끄면 한 번의 기록이 한 번의 호출 시도와 대응한다.
    with OpenAI(timeout=TIMEOUT_SECONDS, max_retries=0) as client:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            reasoning={"effort": "none"},
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )

    # output_text는 SDK가 텍스트 출력들을 모은 값이다.
    # 사용량이 없으면 0으로 바꾸지 않고 알 수 없음인 None을 남긴다.
    usage = response.usage
    return {
        "text": response.output_text,
        "model": OPENAI_MODEL,
        "usage": {
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
        },
        "generation_complete": response.status == "completed",
        "finish_reason": response.status,
    }


def call_model(provider, prompt):
    """생성 호출의 결과와 실패를 같은 필드로 반환한다."""
    if provider not in ("ollama", "openai"):
        raise ValueError("provider는 ollama 또는 openai여야 합니다.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("프롬프트는 내용이 있는 문자열이어야 합니다.")

    started = time.perf_counter()
    output = {
        "text": "", "model": None,
        "usage": {"input_tokens": None, "output_tokens": None},
        "generation_complete": False, "finish_reason": "error",
    }
    error = None
    try:
        if provider == "ollama":
            output = call_ollama(prompt)
        else:
            output = call_openai(prompt)
    except Exception as exc:  # noqa: BLE001 - 제공자 예외의 원문을 밖으로 내보내지 않는다.
        # 실패도 호출 기록이다. 원본 예외에는 비밀값이 섞일 수 있어 타입만 남긴다.
        error = type(exc).__name__
    return {
        **output,
        "provider": provider,
        "latency_seconds": time.perf_counter() - started,
        "output_format_valid": valid_output(output["text"]),
        "error": error,
    }


def generation_config():
    # 비교 파일에 실제로 적용한 설정을 함께 남긴다.
    return {
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "ollama": {
            "model": OLLAMA_MODEL, "num_ctx": 4096,
            "stream": False, "keep_alive": "5m",
        },
        "openai": {
            "model": OPENAI_MODEL, "api": "responses",
            "reasoning_effort": "none", "store": False, "max_retries": 0,
        },
    }


def build_prompt(text, dataset=lesson.DATASET):
    instructions = {
        "inquiries": (
            "한국어로 배송 확인 방법을 안내한다. 주문 조회 도구는 없다.\n"
            "배송 날짜나 조회 완료를 지어내지 않는다.\n"
            "주문번호는 공식 보안 문의창에서 확인하도록 안내한다. "
            "이 대화창에 주문번호를 보내 달라고 요구하지 않는다.\n"
        ),
        "documents": (
            "한국어 공지를 대상, 일정, 해야 할 일을 중심으로 요약한다.\n"
            "원문에 없는 날짜, 장소, 담당자를 만들지 않는다.\n"
        ),
    }
    return (
        instructions[dataset]
        + "출력은 Markdown 없이 JSON 객체 하나이며 필드는 answer와 next_steps뿐이다.\n"
        + "answer는 한국어 답변 문자열이고 next_steps는 문자열 배열이다.\n"
        + "next_steps에는 사용자가 앞으로 실행할 구체적인 행동만 적는다. "
        + "각 항목은 사용자에게 직접 안내하는 정중한 명령형 문장으로 쓴다. "
        + "모델이 요약하거나 검토했다는 작업 보고, 정책을 준수한다는 설명은 넣지 않는다. "
        + "입력과 근거에 없는 행동을 추가하지 않으며, 사용자가 할 행동이 없으면 빈 배열을 쓴다.\n"
        + "사용자 입력: " + json.dumps(text, ensure_ascii=False)
    )


def main():
    # 이 단계는 호출 자체를 연습한다. 실제 분류 모델은 다음 단계에서 연결한다.
    prompt = build_prompt(INPUT_TEXT)
    result = call_model(PROVIDER, prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["error"]:
        raise SystemExit("생성 호출이 실패했습니다. error와 모델 연결 설정을 확인하세요.")


if __name__ == "__main__":
    main()
