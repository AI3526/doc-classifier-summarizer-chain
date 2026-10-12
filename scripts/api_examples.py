"""API의 정상 요청과 실패 요청 예시를 실제로 실행해 저장한다.

서버를 따로 띄우지 않고 FastAPI TestClient로 같은 앱을 실행한다(시작 시 분류기 로딩 포함).
입력은 generation_development.jsonl의 text만 쓴다. openai 대상 요청은 요금이 소액 발생한다.
실패 상황은 설정을 잠시 바꿔 재현한다: Ollama 주소를 닫힌 포트로(연결 실패), 시간 제한을 아주 짧게(시간 초과).

uv run python scripts/api_examples.py
결과: artifacts/{dataset}/service/api_examples/*.json, reports/generated/api_examples.md
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps"))

import lesson_settings as lesson
from common.records import now_iso, write_json
from fastapi.testclient import TestClient
from step01_read_data import read_jsonl
from step10_serve_api import app

OUTPUT = lesson.OUTPUT_ROOT / lesson.DATASET / "service" / "api_examples"
REPORT = ROOT / "reports" / "generated" / "api_examples.md"


def main():
    cases = read_jsonl(lesson.DATA_DIR / "generation_development.jsonl")
    by_label = {}
    for case in cases:
        by_label.setdefault(case["expected_label"], case["text"])  # 유형별 첫 문서(text만 사용)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    examples = []

    def run(name, kind, method, path, payload=None, note=""):
        response = client.request(method, path, json=payload)
        body = response.json()
        record = {"name": name, "kind": kind, "request": {"method": method, "path": path,
                                                         "json": payload},
                  "status_code": response.status_code, "response": body, "note": note,
                  "recorded_at": now_iso()}
        write_json(OUTPUT / f"{len(examples) + 1:02d}_{name}.json", record)
        examples.append(record)
        code = body.get("error", {}).get("code", "-") if isinstance(body, dict) else "-"
        print(f"{response.status_code} {name} {code}")

    with TestClient(app) as client:
        run("health", "정상", "GET", "/health")
        run("classify_announcement", "정상", "POST", "/classify", {"text": by_label["announcement"]})
        run("summarize_local_minutes", "정상", "POST", "/summarize",
            {"text": by_label["minutes"], "target": "qwen2.5-7b"}, "로컬 모델 지정")
        run("summarize_default_manual", "정상", "POST", "/summarize", {"text": by_label["manual"]},
            "target 생략 → 라우팅: 일반 문서는 기본 생성 모델")
        run("route_confidential_local", "라우팅", "POST", "/summarize",
            {"text": by_label["announcement"], "confidential": True},
            "기밀 문서 → 로컬 모델로만 처리")
        run("confidential_cloud_rejected", "라우팅", "POST", "/summarize",
            {"text": by_label["announcement"], "confidential": True, "target": "openai"},
            "기밀 문서를 상용 모델로 지정한 요청은 거부")
        run("blank_text", "잘못된 입력", "POST", "/classify", {"text": "   "})
        run("too_long_text", "잘못된 입력", "POST", "/summarize",
            {"text": "가" * (lesson.MAX_INPUT_CHARS + 1)})
        run("unknown_target", "잘못된 입력", "POST", "/summarize",
            {"text": by_label["announcement"], "target": "gpt-unknown"})
        run("missing_text", "잘못된 입력", "POST", "/summarize", {"target": "qwen2.5-7b"})

        original_url, original_timeout = lesson.OLLAMA_URL, lesson.TIMEOUT_SECONDS
        try:
            lesson.OLLAMA_URL = "http://127.0.0.1:9"  # 아무도 듣지 않는 포트
            run("ollama_unreachable", "모델 호출 실패", "POST", "/summarize",
                {"text": by_label["announcement"], "target": "qwen2.5-7b"},
                "Ollama 서버에 연결할 수 없는 상황을 재현")
            lesson.OLLAMA_URL = original_url
            lesson.TIMEOUT_SECONDS = 0.001
            run("generation_timeout", "모델 호출 실패", "POST", "/summarize",
                {"text": by_label["announcement"], "target": "qwen2.5-7b"},
                "시간 제한을 0.001초로 줄여 응답 시간 초과를 재현")
            lesson.TIMEOUT_SECONDS = original_timeout
            lesson.OLLAMA_URL = "http://127.0.0.1:9"
            run("route_confidential_all_local_down", "대체 경로", "POST", "/summarize",
                {"text": by_label["minutes"], "confidential": True},
                "기밀 문서, 로컬 모델이 모두 응답하지 않음 → 상용 모델로 넘기지 않고 실패")
        finally:
            lesson.OLLAMA_URL, lesson.TIMEOUT_SECONDS = original_url, original_timeout
        key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            run("route_fallback_to_local", "대체 경로", "POST", "/summarize",
                {"text": by_label["minutes"]},
                "일반 문서, 상용 모델 사용 불가(키 없음)를 재현 → 로컬 모델로 넘김")
        finally:
            if key:
                os.environ["OPENAI_API_KEY"] = key

    lines = ["# API 요청 예시", "",
             (f"`scripts/api_examples.py`가 생성. 생성 시각 {now_iso()}. 각 응답 전문은 "
             f"`{OUTPUT.relative_to(ROOT).as_posix()}/`에 있다."), "",
             "| # | 구분 | 요청 | 상태 | 오류 코드 | 결과 |", "|---|---|---|---|---|---|"]
    for number, record in enumerate(examples, start=1):
        body = record["response"]
        request = record["request"]
        target = (request["json"] or {}).get("target", "") if request["json"] else ""
        error = body.get("error", {}) if isinstance(body, dict) else {}
        if error and body.get("detail") and isinstance(body["detail"], list):
            # 잘못된 입력: 어느 칸이 왜 틀렸는지 보여 준다.
            outcome = "; ".join(f"{item['field']}: {item['reason']}" for item in body["detail"])
        elif error:
            outcome = error.get("message", "")
            if record["note"]:
                outcome += f" ({record['note']})"
        elif "generation" in body:
            outcome = (f"{body['classification']['label']} → {body['generation']['target']} 요약 "
                       f"{body['generation']['latency_sec']:.1f}s"
                       + (f" ({record['note']})" if record["note"] else ""))
        elif "classification" in body:
            outcome = (f"{body['classification']['label']} "
                       f"(확신도 {body['classification']['confidence']:.2f})")
        else:
            outcome = f"기본 생성 모델 {body.get('default_target')} ({body.get('default_target_status')})"
        attempts = (body.get("routing") or {}).get("attempts", []) if isinstance(body, dict) else []
        if len(attempts) > 1 or (attempts and error):
            outcome += " — 시도: " + " → ".join(
                f"{a['target']}({'성공' if not a['error'] else a['error']})" for a in attempts)
        flags = []
        if isinstance(request["json"], dict) and request["json"].get("confidential"):
            flags.append("기밀")
        lines.append(f"| {number} | {record['kind']} | `{request['method']} {request['path']}`"
                     f"{' ' + target if target else ''}{' (' + ', '.join(flags) + ')' if flags else ''} | "
                     f"{record['status_code']} | "
                     f"{error.get('code', '-')} | {outcome} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("저장:", OUTPUT, REPORT)
    print(json.dumps(examples[3]["response"], ensure_ascii=False, indent=1)[:1500])


if __name__ == "__main__":
    main()
