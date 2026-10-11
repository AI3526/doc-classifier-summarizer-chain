"""07. Ollama 또는 OpenAI에 프롬프트를 보내고 실제 응답과 실행 기록을 읽는다.

openai 후보는 .env의 OPENAI_API_KEY를 사용하며 요금이 발생한다.
이 단계는 분류기 없이 호출만 연습하므로 train.csv 한 행의 문장과 정답 라벨로 프롬프트를 만든다.

VS Code 터미널에서 프로젝트 폴더를 기준으로 실행한다.
uv run python steps/step07_call_models.py
uv run python steps/step07_call_models.py --target gemma3-4b --index 5
"""

import argparse
import json
import os
import re
import time
from functools import cache

import httpx
import lesson_settings as lesson
from common.resources import environment_info
from openai import OpenAI
from prompts.document_policies import PROMPT_VERSION, build_prompt, parse_output
from step01_read_data import read_rows

DEFAULT_TARGET = lesson.GENERATION_DEFAULT


def resolve_target(name):
    """짧은 이름(lesson_settings.GENERATION_CANDIDATES)을 호출 대상 정보로 바꾼다."""
    if name not in lesson.GENERATION_CANDIDATES:
        raise ValueError(f"알 수 없는 생성 모델: {name}. "
                         f"후보: {', '.join(lesson.GENERATION_CANDIDATES)}")
    return {"alias": name, **lesson.GENERATION_CANDIDATES[name]}


def _ollama(path):
    return lesson.OLLAMA_URL.rstrip("/") + path


@cache
def ollama_model_details(model):
    """양자화 방식과 파라미터 크기를 Ollama에서 읽는다. 실패하면 빈 값을 둔다."""
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            details = client.post(_ollama("/api/show"), json={"model": model}).json()["details"]
        return {"quantization": details.get("quantization_level"),
                "parameter_size": details.get("parameter_size"),
                "family": details.get("family")}
    except (httpx.HTTPError, KeyError, ValueError):
        return {"quantization": None, "parameter_size": None, "family": None}


def ollama_memory_mb(model):
    """현재 불러온 모델이 차지한 전체 메모리와 그중 VRAM(MB). 응답 직후에 읽는다."""
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            loaded = client.get(_ollama("/api/ps")).json().get("models", [])
        for item in loaded:
            if item.get("name") == model or item.get("model") == model:
                return item.get("size", 0) / 2**20, item.get("size_vram", 0) / 2**20
    except (httpx.HTTPError, ValueError):
        pass
    return None, None


def ollama_unload(model):
    """모델을 메모리에서 내린다. 다음 모델의 메모리 측정이 앞 모델의 영향을 받지 않게 한다."""
    try:
        with httpx.Client(timeout=30, trust_env=False) as client:
            client.post(_ollama("/api/generate"), json={"model": model, "keep_alive": 0})
    except httpx.HTTPError:
        pass


def call_ollama(model, prompt):
    # stream=True로 받아 첫 글자가 도착한 시각(TTFT)을 잰다.
    body = {"model": model, "prompt": prompt, "stream": True, "keep_alive": "5m",
            "options": {"temperature": lesson.TEMPERATURE, "num_predict": lesson.MAX_OUTPUT_TOKENS,
                        "num_ctx": lesson.OLLAMA_NUM_CTX}}
    started = time.perf_counter()
    first_token, pieces, final = None, [], {}
    with httpx.Client(timeout=lesson.TIMEOUT_SECONDS, trust_env=False) as client, \
            client.stream("POST", _ollama("/api/generate"), json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                if first_token is None:
                    first_token = time.perf_counter()
                pieces.append(chunk["response"])
            if chunk.get("done"):
                final = chunk
    total = time.perf_counter() - started
    total_mb, vram_mb = ollama_memory_mb(model)
    eval_seconds = (final.get("eval_duration") or 0) / 1e9
    return {
        # 제공자의 응답 원문을 그대로 보존한다. 임의로 JSON을 고치지 않는다.
        "text": "".join(pieces),
        "generation_complete": bool(final.get("done")) and final.get("done_reason") != "length",
        "finish_reason": final.get("done_reason"),
        "performance_metrics": {
            "total_time_sec": total,
            "ttft_sec": None if first_token is None else first_token - started,
            "tokens_per_second": (final.get("eval_count") or 0) / eval_seconds
            if eval_seconds else None,
            "prompt_tokens": final.get("prompt_eval_count"),
            "completion_tokens": final.get("eval_count"),
            "load_time_sec": (final.get("load_duration") or 0) / 1e9,
            "peak_memory_mb": total_mb, "vram_mb": vram_mb,
        },
    }


def call_openai(model, prompt):
    # 키의 존재만 확인한다. 키 값이나 요청 헤더를 출력하지 않는다.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(".env에 OPENAI_API_KEY를 설정하세요.")
    started = time.perf_counter()
    first_token, pieces, final = None, [], None
    # 자동 재시도를 끄면 한 번의 기록이 한 번의 호출 시도와 대응한다.
    with OpenAI(timeout=lesson.TIMEOUT_SECONDS, max_retries=0) as client:
        stream = client.responses.create(
            model=model, input=prompt, reasoning={"effort": lesson.OPENAI_REASONING_EFFORT},
            temperature=lesson.TEMPERATURE, max_output_tokens=lesson.MAX_OUTPUT_TOKENS,
            store=False, stream=True)
        for event in stream:
            if event.type == "response.output_text.delta":
                if first_token is None:
                    first_token = time.perf_counter()
                pieces.append(event.delta)
            elif event.type in ("response.completed", "response.incomplete", "response.failed"):
                final = event.response
    total = time.perf_counter() - started
    usage = getattr(final, "usage", None)
    output_tokens = usage.output_tokens if usage else None
    details = getattr(usage, "input_tokens_details", None)
    generating = total - (first_token - started) if first_token else None
    return {
        "text": "".join(pieces),
        "generation_complete": getattr(final, "status", None) == "completed",
        "finish_reason": getattr(final, "status", None),
        "performance_metrics": {
            "total_time_sec": total,
            "ttft_sec": None if first_token is None else first_token - started,
            # 응답 시간에는 네트워크 왕복이 포함된다.
            "tokens_per_second": output_tokens / generating
            if output_tokens and generating else None,
            # 사용량이 없으면 0으로 바꾸지 않고 알 수 없음인 None을 남긴다.
            "prompt_tokens": usage.input_tokens if usage else None,
            # 입력 중 캐시로 처리된 토큰. 같은 앞부분이 반복되면 생길 수 있다.
            "cached_prompt_tokens": getattr(details, "cached_tokens", None),
            "completion_tokens": output_tokens,
            "load_time_sec": None, "peak_memory_mb": None, "vram_mb": None,
        },
    }


def cost_metrics(provider, metrics):
    if provider == "ollama":
        seconds = metrics["total_time_sec"]
        energy = (lesson.LOCAL_POWER_WATTS * seconds / 3600
                  if lesson.LOCAL_POWER_WATTS and seconds else None)
        return {"api_cost_usd": 0.0, "estimated": True,
                "cost_formula": "로컬 실행: API 요금 0. 장비·전력·관리 비용은 별도 추정"
                                + (f", 전력량 = {lesson.LOCAL_POWER_WATTS}W × 응답 시간"
                                   if energy is not None else ""),
                "energy_wh_estimate": energy}
    return {**openai_cost(metrics.get("prompt_tokens"), metrics.get("completion_tokens"),
                          metrics.get("cached_prompt_tokens") or 0),
            "energy_wh_estimate": None}


def openai_cost(prompt_tokens, completion_tokens, cached_tokens=0):
    """실측 토큰 수와 lesson_settings의 단가로 요청 1건의 비용(USD)을 계산한다."""
    prices = (lesson.OPENAI_INPUT_PRICE_PER_1M, lesson.OPENAI_CACHED_INPUT_PRICE_PER_1M,
              lesson.OPENAI_OUTPUT_PRICE_PER_1M)
    if None in prices or prompt_tokens is None or completion_tokens is None:
        return {"api_cost_usd": None, "estimated": True,
                "cost_formula": "단가 또는 토큰 수가 없어 계산하지 않음"}
    input_price, cached_price, output_price = prices
    # 272K 토큰을 넘는 요청은 요청 전체에 가산 단가를 적용한다.
    long_prompt = prompt_tokens > lesson.OPENAI_LONG_PROMPT_TOKENS
    if long_prompt:
        input_price *= lesson.OPENAI_LONG_PROMPT_MULTIPLIER["input"]
        cached_price *= lesson.OPENAI_LONG_PROMPT_MULTIPLIER["input"]
        output_price *= lesson.OPENAI_LONG_PROMPT_MULTIPLIER["output"]
    uncached = prompt_tokens - cached_tokens
    cost = (uncached * input_price + cached_tokens * cached_price
            + completion_tokens * output_price) / 1e6
    return {"api_cost_usd": cost, "estimated": False,
            "cost_formula": (
                f"실측 토큰 × Standard 단가: 입력 {uncached} × ${input_price}/1M + "
                f"캐시 입력 {cached_tokens} × ${cached_price}/1M + "
                f"출력 {completion_tokens} × ${output_price}/1M"
                + (" (272K 초과 가산 적용)" if long_prompt else "")
                + f", 단가 확인일 {lesson.OPENAI_PRICE_CHECKED_AT}. "
                "캐시 쓰기 토큰은 응답에 보고되지 않아 제외")}


def _safe_message(exc):
    # 원본 예외에는 비밀값이 섞일 수 있어 키 형태의 문자열을 가리고 길이를 줄인다.
    return re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-***", str(exc))[:300]


def call_model(target, prompt):
    """생성 호출의 결과와 실패를 같은 필드로 반환한다. 실패도 호출 기록이다."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("프롬프트는 내용이 있는 문자열이어야 합니다.")
    target = resolve_target(target) if isinstance(target, str) else target
    started = time.perf_counter()
    output = {"text": "", "generation_complete": False, "finish_reason": "error",
              "performance_metrics": {key: None for key in (
                  "ttft_sec", "tokens_per_second", "prompt_tokens", "completion_tokens",
                  "load_time_sec", "peak_memory_mb", "vram_mb")}}
    error = message = None
    try:
        if target["provider"] == "ollama":
            output = call_ollama(target["model"], prompt)
        elif target["provider"] == "openai":
            output = call_openai(target["model"], prompt)
        else:
            raise ValueError(f"지원하지 않는 provider: {target['provider']}")
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 기록으로 남기고 다음 호출을 계속한다.
        error, message = type(exc).__name__, _safe_message(exc)
    output["performance_metrics"]["total_time_sec"] = (
        output["performance_metrics"].get("total_time_sec") or time.perf_counter() - started)
    return {
        "alias": target["alias"], "provider": target["provider"], "model": target["model"],
        **output,
        "cost_metrics": cost_metrics(target["provider"], output["performance_metrics"]),
        "status": "ERROR" if error else "SUCCESS", "error": error, "error_message": message,
    }


def execution_conditions(target):
    """비교 기록에 남길 실제 적용 설정."""
    target = resolve_target(target) if isinstance(target, str) else target
    environment = environment_info()
    common = {"prompt_version": PROMPT_VERSION, "temperature": lesson.TEMPERATURE,
              "max_tokens": lesson.MAX_OUTPUT_TOKENS, "timeout_sec": lesson.TIMEOUT_SECONDS,
              "stream": True}
    if target["provider"] == "ollama":
        return {**common, **ollama_model_details(target["model"]),
                "num_ctx": lesson.OLLAMA_NUM_CTX, "keep_alive": "5m",
                "hardware": f"{environment['gpu']} / RAM {environment['ram_total_gb']}GB",
                "measurement": "로컬 HTTP 호출, 모델 적재 후(준비 호출 제외) 시간"}
    return {**common, "api": "responses", "reasoning_effort": lesson.OPENAI_REASONING_EFFORT,
            "store": False, "max_retries": 0, "quantization": None,
            "hardware": "OpenAI 클라우드",
            "measurement": "네트워크 왕복을 포함한 클라이언트 측 시간"}


def main():
    parser = argparse.ArgumentParser(description="생성 모델 호출 연습")
    parser.add_argument("--target", nargs="+", default=[DEFAULT_TARGET],
                        help=f"후보: {', '.join(lesson.GENERATION_CANDIDATES)}")
    parser.add_argument("--index", type=int, default=0, help="train.csv에서 사용할 행 번호")
    args = parser.parse_args()
    row = read_rows(lesson.DATA_DIR / "train.csv")[args.index]
    # train의 정답 라벨은 학습용 값이므로 연습용 프롬프트에 써도 된다.
    prompt = build_prompt(row["text"], row["label"])
    print("입력:", row["id"], row["label"], "\n", row["text"])
    failed = False
    for name in args.target:
        result = call_model(name, prompt)
        parsed = parse_output(result["text"], row["label"])
        print(f"\n===== {name} =====")
        print(json.dumps({**result, "format_valid_strict": parsed["strict"],
                          "format_valid_lenient": parsed["lenient"]},
                         ensure_ascii=False, indent=2))
        failed |= result["status"] == "ERROR"
    if failed:
        raise SystemExit("생성 호출이 실패했습니다. error와 모델 연결 설정을 확인하세요.")


if __name__ == "__main__":
    main()
