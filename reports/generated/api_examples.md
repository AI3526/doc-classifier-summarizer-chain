# API 요청 예시

`scripts/api_examples.py`가 생성. 생성 시각 2026-10-12T17:15:08+09:00. 각 응답 전문은 `artifacts/documents/service/api_examples/`에 있다.

| # | 구분 | 요청 | 상태 | 오류 코드 | 결과 |
|---|---|---|---|---|---|
| 1 | 정상 | `GET /health` | 200 | - | 기본 생성 모델 openai (provisional) |
| 2 | 정상 | `POST /classify` | 200 | - | announcement (확신도 0.97) |
| 3 | 정상 | `POST /summarize` qwen2.5-7b | 200 | - | minutes → qwen2.5-7b 요약 6.2s (로컬 모델 지정) |
| 4 | 정상 | `POST /summarize` | 200 | - | manual → openai 요약 2.4s (target 생략 → 기본 생성 모델(선택 결과 파일)) |
| 5 | 잘못된 입력 | `POST /classify` | 422 | invalid_request | text: Value error, 공백만 입력할 수 없습니다. |
| 6 | 잘못된 입력 | `POST /summarize` | 422 | invalid_request | text: String should have at most 4000 characters |
| 7 | 잘못된 입력 | `POST /summarize` gpt-unknown | 422 | invalid_request | target: Input should be 'qwen3-4b', 'gemma3-4b', 'qwen2.5-7b', 'exaone3.5-7.8b', 'llama3.1-8b' or 'openai' |
| 8 | 잘못된 입력 | `POST /summarize` qwen2.5-7b | 422 | invalid_request | text: Field required |
| 9 | 모델 호출 실패 | `POST /summarize` qwen2.5-7b | 503 | generation_unavailable | 생성 모델에 연결할 수 없습니다. (Ollama 서버에 연결할 수 없는 상황을 재현) |
| 10 | 모델 호출 실패 | `POST /summarize` qwen2.5-7b | 504 | generation_timeout | 생성 모델 응답 시간이 초과되었습니다. (시간 제한을 0.001초로 줄여 응답 시간 초과를 재현) |
