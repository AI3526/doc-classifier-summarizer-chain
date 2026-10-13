"""10. 학습한 분류기와 생성 체인을 FastAPI 요청으로 실행한다.

엔드포인트
- GET  /health    서버, 분류기, 기본 생성 모델, Ollama 연결 상태
- POST /classify  분류만 (생성 모델을 부르지 않음)
- POST /summarize 분류 → 정책 → 생성 → 형식 검사 (LangChain 체인 하나, /generate도 같은 동작)
                  target을 생략하면 하이브리드 라우팅: 기밀 문서(confidential=true)는 로컬 모델만,
                  그 외는 기본 생성 모델. 연결 실패·시간 초과면 대체 모델로 넘긴다.
- POST /compare   같은 분류 결과와 프롬프트로 여러 생성 모델을 비교

기본 생성 모델은 selection/generation_selected.json의 선택 결과를 따르고(가상 채점 결과는 쓰지 않음),
없으면 lesson_settings.GENERATION_DEFAULT를 쓴다. 요청마다 target으로 바꿀 수 있다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step10_serve_api.py
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import lesson_settings as lesson
import uvicorn
from common.records import read_json
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from step06_predict import load_classifier
from step08_connect_chain import (
    MAX_INPUT_CHARS,
    GenerationUnavailable,
    compare_prepared,
    create_chain,
    create_routed_chain,
    create_service_chain,
)

TargetName = Literal[tuple(lesson.GENERATION_CANDIDATES)]
HOST = "127.0.0.1"
PORT = 8000
# 분류 확신도가 이보다 낮으면 응답에 경고를 붙인다. 판단 기준이 아니라 사용자에게 알리는 용도다.
LOW_CONFIDENCE = 0.6
# 생성 실패 유형별 HTTP 상태. 예외 타입 이름으로 나눈다(원본 메시지는 비밀값이 섞일 수 있어 줄여서 보냄).
UNAVAILABLE_ERRORS = {"ConnectError", "APIConnectionError", "AuthenticationError", "RuntimeError"}
TIMEOUT_ERRORS = {"ReadTimeout", "ConnectTimeout", "TimeoutException", "APITimeoutError"}


def default_target():
    """선택 결과 파일의 생성 모델. 가상 채점(dummy) 결과는 쓰지 않는다."""
    path = lesson.SELECTION_DIR / "generation_selected.json"
    if path.exists():
        selected = read_json(path)
        if selected.get("status") != "dummy" and selected["selected"] in lesson.GENERATION_CANDIDATES:
            return selected["selected"], selected.get("status", "unknown")
    return lesson.GENERATION_DEFAULT, "default"


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)

    @field_validator("text")
    @classmethod
    def nonblank(cls, text):
        if not text.strip():
            raise ValueError("공백만 입력할 수 없습니다.")
        return text.strip()


class SummarizeRequest(TextRequest):
    target: TargetName | None = None  # 생략하면 하이브리드 라우팅
    confidential: bool = False  # 기밀 문서면 외부(상용) 모델로 보내지 않는다


class CompareRequest(TextRequest):
    targets: list[TargetName] = Field(min_length=1, max_length=len(lesson.GENERATION_CANDIDATES))


def error_response(status, code, message, **extra):
    return JSONResponse({"error": {"code": code, "message": message}, **extra}, status_code=status)


def classification_view(classification):
    return {**classification,
            "low_confidence": classification["confidence"] < LOW_CONFIDENCE}


def generation_view(generation, target):
    metrics = generation["performance_metrics"]
    return {
        "target": target, "provider": generation["provider"], "model": generation["model"],
        "status": generation["status"], "summary": generation["parsed"],
        "format_valid": generation["format_valid_lenient"],
        "latency_sec": metrics["total_time_sec"], "ttft_sec": metrics["ttft_sec"],
        "prompt_tokens": metrics["prompt_tokens"],
        "completion_tokens": metrics["completion_tokens"],
        "api_cost_usd": generation["cost_metrics"]["api_cost_usd"],
    }


def generation_failure(generation):
    """생성 실패를 HTTP 상태와 오류 코드로 바꾼다. 성공이면 None."""
    if generation["status"] == "ERROR":
        error = generation["error"] or ""
        if error in TIMEOUT_ERRORS or "Timeout" in error:
            return 504, "generation_timeout", "생성 모델 응답 시간이 초과되었습니다."
        if error in UNAVAILABLE_ERRORS:
            return 503, "generation_unavailable", "생성 모델에 연결할 수 없습니다."
        return 502, "generation_failed", "생성 모델 호출이 실패했습니다."
    if not generation["format_valid_lenient"]:
        return 502, "invalid_output", "생성 모델 응답이 출력 형식에 맞지 않습니다."
    return None


@asynccontextmanager
async def lifespan(app):
    # 서버가 시작될 때 분류기를 한 번 읽는다. 모델을 읽지 못하면 시작이 실패한다.
    classifier = load_classifier()
    app.state.classifier = classifier
    app.state.prepare_chain = create_chain(classifier)  # 분류 → 정책·프롬프트
    app.state.service_chains = {}  # 생성 모델별 전체 체인(분류 → 정책 → 생성), 처음 쓸 때 만든다
    app.state.default_target, app.state.default_status = default_target()
    # 하이브리드 라우팅 체인: 기밀 여부로 분기하고, 응답하지 못한 모델은 다음 모델로 넘긴다.
    app.state.routed_chain, app.state.routing_plans = create_routed_chain(
        classifier, app.state.default_target)
    yield


app = FastAPI(title="문서 분류 + 유형별 요약 API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # 잘못된 입력은 함수 실행 전에 막고, 어느 칸이 왜 틀렸는지 한국어로 알려 준다.
    problems = [{"field": ".".join(str(part) for part in error["loc"][1:]) or "body",
                 "reason": error["msg"]} for error in exc.errors()]
    return error_response(422, "invalid_request",
                          f"요청 형식이 올바르지 않습니다. text는 공백이 아닌 1~{MAX_INPUT_CHARS}자, "
                          f"target은 {', '.join(lesson.GENERATION_CANDIDATES)} 중 하나여야 합니다.",
                          detail=problems)


@app.get("/health")
def health(request: Request):
    # 서버와 분류기 상태, 기본 생성 모델과 그 선택 상태를 보여 준다. 생성 모델을 호출하지는 않는다.
    try:
        with httpx.Client(timeout=2, trust_env=False) as client:
            ollama_ok = client.get(lesson.OLLAMA_URL.rstrip("/") + "/api/tags").status_code == 200
    except httpx.HTTPError:
        ollama_ok = False
    return {"status": "ok", "classifier": request.app.state.classifier["identity"],
            "default_target": request.app.state.default_target,
            "default_target_status": request.app.state.default_status,
            "routing_plans": request.app.state.routing_plans,
            "targets": {name: spec["provider"] for name, spec in lesson.GENERATION_CANDIDATES.items()},
            "structured_output": lesson.STRUCTURED_OUTPUT, "ollama_reachable": ollama_ok,
            "openai_key_configured": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/classify")
def classify(body: TextRequest, request: Request):
    started = time.perf_counter()
    try:
        prepared = request.app.state.prepare_chain.invoke(body.text)
    except Exception as exc:  # noqa: BLE001 - 분류 실패는 원인 타입만 알린다.
        return error_response(500, "classification_failed", "문서 분류에 실패했습니다.",
                              detail=type(exc).__name__)
    classification = classification_view(prepared["classification"])
    warnings = (["분류 확신도가 낮습니다. 결과를 확인하세요."] if classification["low_confidence"]
                else []) + (["입력이 길어 분류기에서 일부가 잘렸습니다."]
                            if classification["truncated"] else [])
    warnings += preprocessing_warnings(prepared["preprocessing"])
    return {"classification": classification, "preprocessing": prepared["preprocessing"],
            "policy": prepared["policy"], "warnings": warnings,
            "elapsed_sec": time.perf_counter() - started}


def preprocessing_warnings(preprocessing):
    if not preprocessing["body_extracted"]:
        return []
    return [(f"참고 자료 구간을 빼고 본문만 분류·요약했습니다 "
            f"({preprocessing['original_chars']}자 → {preprocessing['used_chars']}자).")]


def is_local(target):
    return lesson.GENERATION_CANDIDATES[target]["provider"] == "ollama"


@app.post("/summarize")
@app.post("/generate")
def summarize(body: SummarizeRequest, request: Request):
    started = time.perf_counter()
    state = request.app.state
    if body.target:
        # 직접 지정: 그 모델만 쓴다. 기밀 문서를 상용 모델로 보내는 요청은 막는다.
        if body.confidential and not is_local(body.target):
            return error_response(422, "confidential_requires_local",
                                  "기밀 문서는 로컬 생성 모델로만 처리할 수 있습니다.",
                                  allowed=[name for name in lesson.GENERATION_CANDIDATES
                                           if is_local(name)])
        if body.target not in state.service_chains:
            state.service_chains[body.target] = create_service_chain(state.classifier,
                                                                     body.target)
        chain, payload = state.service_chains[body.target], body.text
        plan = [body.target]
    else:
        chain = state.routed_chain
        payload = {"text": body.text, "confidential": body.confidential}
        plan = state.routing_plans["confidential" if body.confidential else "normal"]
    routing = {"mode": "explicit" if body.target else "auto", "confidential": body.confidential,
               "plan": plan}
    try:
        result = chain.invoke(payload)
    except GenerationUnavailable as exc:
        # 목록의 모든 모델이 응답하지 못함. 분류 결과와 시도 기록을 함께 돌려준다.
        status, code, message = generation_failure(exc.generation)
        return error_response(status, code, f"{message} 대체 모델도 응답하지 않았습니다.",
                              classification=classification_view(exc.item["classification"]),
                              preprocessing=exc.item.get("preprocessing"),
                              routing={**routing, "attempts": exc.attempts},
                              elapsed_sec=time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - 분류 단계 실패
        return error_response(500, "classification_failed", "문서 분류에 실패했습니다.",
                              detail=type(exc).__name__)
    used = result.get("used_target", body.target)
    attempts = result.get("attempts", [{"target": used, "status": result["generation"]["status"],
                                        "error": result["generation"]["error"]}])
    routing = {**routing, "used_target": used, "fallback_used": len(attempts) > 1,
               "attempts": attempts}
    classification = classification_view(result["classification"])
    generation = result["generation"]
    failure = generation_failure(generation)
    view = generation_view(generation, used)
    if failure:
        status, code, message = failure
        # 실패해도 분류 결과와 실패한 호출의 기록은 돌려준다.
        return error_response(status, code, message, classification=classification,
                              preprocessing=result.get("preprocessing"), routing=routing,
                              generation={**view, "error": generation["error"],
                                          "error_message": generation["error_message"]},
                              elapsed_sec=time.perf_counter() - started)
    warnings = (["분류 확신도가 낮습니다. 요약 형식이 문서와 맞는지 확인하세요."]
                if classification["low_confidence"] else [])
    warnings += preprocessing_warnings(result["preprocessing"])
    if routing["fallback_used"]:
        warnings.append(f"{attempts[0]['target']}이(가) 응답하지 않아 {used}(으)로 처리했습니다.")
    if state.default_status == "provisional" and not body.target and not body.confidential             and used == state.default_target:
        warnings.append("기본 생성 모델은 사람 채점 전 잠정 선택입니다.")
    return {"classification": classification, "preprocessing": result["preprocessing"],
            "prompt_version": result["prompt_version"], "routing": routing, "generation": view, "warnings": warnings,
            "elapsed_sec": time.perf_counter() - started}


@app.post("/compare")
def compare(body: CompareRequest, request: Request):
    # 한 번 분류해서 만든 프롬프트를 여러 생성 모델이 공유한다.
    prepared = request.app.state.prepare_chain.invoke(body.text)
    result = compare_prepared(prepared, body.targets, lesson.STRUCTURED_OUTPUT)
    views = [{**generation_view(row, target), "error": row["error"]}
             for row, target in zip(result["results"], body.targets, strict=True)]
    failed = any(generation_failure(row) for row in result["results"])
    return JSONResponse({"classification": classification_view(prepared["classification"]),
                         "preprocessing": prepared["preprocessing"],
                         "prompt_version": result["prompt_version"], "results": views},
                        status_code=502 if failed else 200)


def main():
    print(f"API 문서: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT, access_log=False)


if __name__ == "__main__":
    main()
