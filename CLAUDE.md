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
| 3 | `step03_train_baseline.py` | 문자 TF-IDF + 로지스틱 회귀 (필수 베이스라인). `--ngram 2 5` |
| 4 | `step04_train_classifier.py` | 사전학습 모델 전체 파인튜닝 (GPU). `--model bert roberta --lr 1e-5 2e-5 5e-5 --seed 42` → 모델×학습률×seed 조합을 차례로, 실패해도 계속 |
| 5 | `step05_select_model.py` | 팀 전체 실행(베이스라인 포함)을 모아 validation 기준 선정 → `selection/` |
| 6 | `step06_predict.py` | 선정 분류기 로드·단건 예측 (베이스라인/BERT 모두 `load_run`으로) |
| 7 | `step07_call_models.py` | Ollama / OpenAI 호출 (**OpenAI는 요금 발생**) |
| 8 | `step08_connect_chain.py` | 분류 → 정책 → 프롬프트 LangChain 체인 |
| 9 | `step09_compare.py` | 같은 입력으로 생성 모델 비교 기록 (**요금 발생**) |
| 10 | `step10_serve_api.py` | FastAPI (`/health`, `/generate`, `/compare`) |
| 11 | `step11_evaluate_final.py` | 고정된 분류기로 test 1회 평가 |

`scripts/verify_learning.py`, `scripts/verify_service.py`는 학습·비교 기록 검수용, `scripts/build_datasets.py`는 데이터 생성 원본(수정하지 않음).

## 결과 폴더
```
artifacts/{dataset}/
  {learner}/runs/{YYYYMMDD-HHMMSS}_{run_name}/   # step03/04 실행 하나
      config.json  experiment.json  validation_metrics.json  validation_errors.csv
      epochs.jsonl  learning_curve.png  training_summary.json   # BERT 실행만
      checkpoint/ 또는 baseline.joblib                          # 가중치 (git 제외)
  selection/                                     # 팀 공통, step05/11
      selected.json  comparison.md/csv  confusion_*.png  history/
      test_metrics.json  test_predictions.csv  test_case_review.csv  test_errors.csv
  {learner}/comparisons/...                      # step09 생성 비교
```
- `DATASET`, `LEARNER`는 `steps/lesson_settings.py`에서 정한다. 팀원(`learner_park`, `learner_choi`, `learner_ro`, `learner_su`)은 **각자 다른 실험**을 맡는다.
- `experiment.json`이 needtocheck의 EXP 형식이며 step05가 이것을 모아 비교한다. 지표 JSON/CSV는 커밋, 가중치는 git 제외.
- 선정 규칙(step05 `TOLERANCE`): validation macro F1 최고점에서 0.01 이내 후보 중 CPU 1건 추론이 가장 빠른 것. **결과를 본 뒤 바꾸지 않는다.**
- `selection/test_metrics.json`이 생기면 step05 재선정이 막힌다(test를 본 뒤 재선택 금지).
- 다른 팀원 실행이 선정되면 가중치가 git에 없으므로 받아 오거나 같은 설정으로 재학습해야 서비스에서 쓸 수 있다.
- Windows 경로 길이(260자) 제한에 주의: 프로젝트를 깊은 폴더에 두면 체크포인트 저장이 실패할 수 있다.

## 분류 후보 모델 (`lesson_settings.CLASSIFIER_CANDIDATES`)
- `bert`(klue/bert-base, 필수), `roberta`, `electra-s`(14M), `electra-b`, `deberta-ko`(team-lucid/deberta-v3-base-korean) — 모두 로드·학습 확인됨
- 제외: distilkobert(구형, remote code + sentencepiece 필요), ModernBERT·NeoBERT(영어 전용 토크나이저 → 한국어 처리 부적합)
- 보류: mdeberta-v3-base(다국어, sentencepiece·protobuf 설치 필요)
- 실험 순서: ① 모든 후보 × lr {1e-5, 2e-5, 5e-5}, seed 42 ② 상위 2~3개 설정만 seed 43, 44 추가 → 평균±표준편차 보고
- 실행 이름 형식: `{alias}_lr{lr}_s{seed}`

## 공통 모듈 (`steps/common/`)
- `metrics.py`: `classification_scores`(accuracy, macro/weighted F1, 혼동행렬), `save_errors`
- `records.py`: `new_run_dir`, `write_json`, `read_json`, `now_iso`, `stamp`
- `resources.py`: `measure_latency`(1건씩, 준비 호출 제외), `reset_gpu_peak`/`gpu_peak_mb`, `environment_info`
- `plots.py`: `plot_learning_curve`, `plot_confusion_matrix` (축 이름은 영어 — 한글 글꼴 문제 회피)

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
- [완료] 1단계 분류: step03/04 자원·시간 기록(`experiment.json`)과 CLI, step05 베이스라인 포함 팀 단위 선정, step06 통합 로더, step11 지연·사례 기록, `verify_learning.py` 새 구조 대응
- `data/manifest.json` 해시는 이전 데이터 버전 기준이라 현재 v2 데이터와 다름 → `verify_learning.py`는 일치 여부만 기록
- 출력 스키마(answer/next_steps)가 요약에 맞지 않음 → 라벨별 요약 스키마로 교체 예정
- step09는 ollama 1개 vs openai 고정 → 여러 로컬 모델 비교, 비용 계산, LLM 채점 + 블라인드 사람 채점 추가 예정
- `tests/` 폴더 아직 없음 (pytest testpaths = `tests`)
