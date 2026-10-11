# CLAUDE.md

## 프로젝트 한 줄 요약
사내 문서(공지 `announcement` / 매뉴얼 `manual` / 회의록 `minutes`)를 **직접 학습한 분류기**로 분류하고, 예측 라벨에 맞는 정책으로 **생성 LLM이 요약**하는 서비스. LangChain으로 연결하고 FastAPI로 제공한다.
팀 방향: **비용 최적화 & 온프레미스** — 기밀 문서는 외부로 보내기 어렵고 긴 문서는 API 비용이 커지므로, 로컬 LLM(Ollama)을 상용 API(OpenAI)와 실측으로 비교해 근거를 만든다.

- 요구사항 원문: `projectoutline.md` / 팀의 미결 사항·기록 사양: `needtocheck.md`
- 승인된 실행 계획: 현황 진단, 문제점, 단계별 작업 (아래 "진행 상태" 참고)

## 환경
- Windows 11, Python 3.12, `uv` 사용 (`uv run python steps/<file>.py`)
- GPU: RTX 5060 Laptop **8GB VRAM**, RAM 32GB, torch cu128
- Ollama 설치 모델: `qwen3:4b-instruct-2507-q4_K_M`(기본), `gemma3:4b`, `qwen2.5:7b`, `llama3.1:8b`
- 비밀값은 `.env`(OPENAI_API_KEY 등). **키 값을 출력하거나 파일·로그에 쓰지 않는다.**

## 데이터 규칙 (반드시 지킬 것)
- 필수 실험은 `data/documents` (train 360 / validation 120 / test 120, 라벨 균형, 텍스트 ~100자).
- `data/documents_advanced`(평균 ~900자, 최대 1877자)는 **선택 과제**: 긴 문서의 비용·성능 근거용.
- `test.csv`는 **step11에서만** 읽는다. 모델 선택·튜닝에 test 결과를 쓰지 않는다.
- 분류 입력은 `text`만 사용. `id`, `group_id`, `source`는 특징에 넣지 않는다.
- 생성 JSONL의 `expected_label`, `required_points`, `forbidden_claims`, `rubric_note`는 **채점 전용**. 프롬프트나 모델 입력에 넣지 않는다.
- 서비스의 정책 분기는 정답 라벨이 아니라 **분류기의 실제 예측**을 사용한다.
- `generation_eval.jsonl`의 답변을 보고 프롬프트를 고치지 않는다(개발은 `generation_development.jsonl`).

## 실행 순서 (steps/)
| 단계 | 파일 | 내용 |
|---|---|---|
| 0 | `check_environment.py` | Python/GPU/Ollama/키 존재 확인 |
| 1–2 | `step01_read_data.py`, `step02_check_data.py` | 데이터 읽기, 분할 중복·라벨 검사 |
| 3 | `step03_train_baseline.py` | 문자 TF-IDF + 로지스틱 회귀 (필수 베이스라인) |
| 4 | `step04_train_classifier.py` | klue/bert-base 등 전체 파인튜닝 (GPU) |
| 5 | `step05_select_model.py` | validation 기준 분류기 선정 → `selected.json` |
| 6 | `step06_predict.py` | 선정 분류기 로드·단건 예측 |
| 7 | `step07_call_models.py` | Ollama / OpenAI 호출 (**OpenAI는 요금 발생**) |
| 8 | `step08_connect_chain.py` | 분류 → 정책 → 프롬프트 LangChain 체인 |
| 9 | `step09_compare.py` | 같은 입력으로 생성 모델 비교 기록 (**요금 발생**) |
| 10 | `step10_serve_api.py` | FastAPI (`/health`, `/generate`, `/compare`) |
| 11 | `step11_evaluate_final.py` | 고정된 분류기로 test 1회 평가 |

`scripts/verify_learning.py`, `scripts/verify_service.py`는 학습·비교 기록 검수용, `scripts/build_datasets.py`는 데이터 생성 원본(수정하지 않음).

## 결과 폴더
- `artifacts/{dataset}/{learner}/...` — 실행별 결과. `steps/lesson_settings.py`의 `DATASET`, `LEARNER`로 결정.
- 팀원(`learner_park`, `learner_choi`, `learner_ro`, `learner_su`)은 **각자 다른 실험**을 맡고, learner 폴더로 결과가 분리된다.
- 기존 실험 폴더를 덮어쓰지 않는다(코드가 `FileExistsError`로 막음). 새 실험은 새 이름으로.

## 기록 사양 (needtocheck.md 기준)
- 분류: 모델 ID·파라미터 수·정밀도·학습 방식, max_length/batch/lr/epochs/seed, 학습 시간(초), 1건당 추론 지연(ms), peak VRAM(`torch.cuda.max_memory_allocated`, MB), Accuracy·Macro-F1·Weighted-F1·혼동행렬, 오분류 사례.
- 생성: 입력·예측 라벨·프롬프트 버전, 모델/공급자, 하드웨어·temperature·max_tokens·양자화, 총 시간·TTFT·tokens/s·토큰 수·메모리, 비용(실측/추정 구분)과 산식, 성공 여부·오류, 원본 응답.
- 실패한 호출도 기록한다. 실측값과 추정값을 구분한다.

## 코딩 규칙
- 기존 스타일 유지: 한국어 주석·메시지, 단계별 독립 실행 파일, `lesson_settings.py`에서 경로·설정 import.
- `steps/`가 `sys.path` 기준(pytest `pythonpath = ["steps"]`). 모듈은 `from step01_read_data import ...` 형태로 import.
- ruff line-length 100.
- 문의(inquiries) 주제는 사용하지 않는다. 연습 키트에서 온 문의 전용 코드는 정리 대상.

## 프롬프트
- 정책·프롬프트·출력 형식 검사는 `steps/prompts/document_policies.py` 한 곳에 둔다(`POLICIES`, `build_prompt`, `valid_output`, `PROMPT_VERSION`).
- 프롬프트를 바꾸면 `PROMPT_VERSION`도 바꾼다.

## 진행 상태 / 알려진 문제
- [완료] 0단계 기반 정리: `.gitignore`(가중치만 제외), `pyproject.toml` 정리·의존성(joblib, matplotlib, psutil), 문의용 코드 제거, 프롬프트 분리
- step05 선정 후보에 베이스라인이 없음 (필수 요건 위반) → 베이스라인 포함 비교로 변경 예정
- step03/04/11에 시간·VRAM·지연 기록 없음
- 출력 스키마(answer/next_steps)가 요약에 맞지 않음 → 라벨별 요약 스키마로 교체 예정
- step09는 ollama 1개 vs openai 고정 → 여러 로컬 모델 비교, 비용 계산, LLM 채점 + 블라인드 사람 채점 추가 예정
- `tests/` 폴더 아직 없음 (pytest testpaths = `tests`)
