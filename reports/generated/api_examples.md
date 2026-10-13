# API 요청 예시

`scripts/api_examples.py`가 생성. 생성 시각 2026-10-13T13:06:11+09:00. 각 응답 전문은 `artifacts/documents/service/api_examples/`에 있다.

| # | 구분 | 요청 | 상태 | 오류 코드 | 결과 |
|---|---|---|---|---|---|
| 1 | 정상 | `GET /health` | 200 | - | 기본 생성 모델 openai (provisional) |
| 2 | 정상 | `POST /classify` | 200 | - | announcement (확신도 0.97) |
| 3 | 정상 | `POST /summarize` qwen2.5-7b | 200 | - | minutes → qwen2.5-7b 요약 5.7s (로컬 모델 지정) |
| 4 | 정상 | `POST /summarize` | 200 | - | manual → openai 요약 2.7s (target 생략 → 라우팅: 일반 문서는 기본 생성 모델) |
| 5 | 라우팅 | `POST /summarize` (기밀) | 200 | - | announcement → qwen2.5-7b 요약 2.0s (기밀 문서 → 로컬 모델로만 처리) |
| 6 | 라우팅 | `POST /summarize` openai (기밀) | 422 | confidential_requires_local | 기밀 문서는 로컬 생성 모델로만 처리할 수 있습니다. (기밀 문서를 상용 모델로 지정한 요청은 거부) |
| 7 | 긴 문서 | `POST /summarize` qwen2.5-7b | 200 | - | minutes → qwen2.5-7b 요약 3.9s (참고 자료가 앞에 붙은 긴 문서 → 본문만 분류·요약) — 본문 추출 1380자 → 272자 |
| 8 | 잘못된 입력 | `POST /classify` | 422 | invalid_request | text: Value error, 공백만 입력할 수 없습니다. |
| 9 | 잘못된 입력 | `POST /summarize` | 422 | invalid_request | text: String should have at most 4000 characters |
| 10 | 잘못된 입력 | `POST /summarize` gpt-unknown | 422 | invalid_request | target: Input should be 'qwen3-4b', 'gemma3-4b', 'qwen2.5-7b', 'exaone3.5-7.8b', 'llama3.1-8b' or 'openai' |
| 11 | 잘못된 입력 | `POST /summarize` qwen2.5-7b | 422 | invalid_request | text: Field required |
| 12 | 모델 호출 실패 | `POST /summarize` qwen2.5-7b | 503 | generation_unavailable | 생성 모델에 연결할 수 없습니다. (Ollama 서버에 연결할 수 없는 상황을 재현) — 시도: qwen2.5-7b(ConnectError) |
| 13 | 모델 호출 실패 | `POST /summarize` qwen2.5-7b | 504 | generation_timeout | 생성 모델 응답 시간이 초과되었습니다. (시간 제한을 0.001초로 줄여 응답 시간 초과를 재현) — 시도: qwen2.5-7b(ReadTimeout) |
| 14 | 대체 경로 | `POST /summarize` (기밀) | 503 | generation_unavailable | 생성 모델에 연결할 수 없습니다. 대체 모델도 응답하지 않았습니다. (기밀 문서, 로컬 모델이 모두 응답하지 않음 → 상용 모델로 넘기지 않고 실패) — 시도: qwen2.5-7b(ConnectError) → qwen3-4b(ConnectError) |
| 15 | 대체 경로 | `POST /summarize` | 200 | - | minutes → qwen2.5-7b 요약 3.3s (일반 문서, 상용 모델 사용 불가(키 없음)를 재현 → 로컬 모델로 넘김) — 시도: openai(RuntimeError) → qwen2.5-7b(성공) |
