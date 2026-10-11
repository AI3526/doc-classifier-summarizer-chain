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
| 9 | `step09_compare.py` | 같은 입력으로 생성 모델 비교 기록 (openai 포함 시 **요금 발생**) |
| 9b | `step09b_score.py` | 요약 품질 채점: `judge`(LLM, **요금 발생**) / `sheet`(사람 시트) / `aggregate`(집계) |
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
      selected.json  comparison.md/csv  comparison_by_config.csv  confusion_*.png  history/
      test_metrics.json  test_predictions.csv  test_case_review.csv  test_errors.csv
  {learner}/comparisons/...                      # step09 생성 비교
```
- `DATASET`, `LEARNER`는 `steps/lesson_settings.py`에서 정한다. 팀원(`learner_park`, `learner_choi`, `learner_ro`, `learner_su`)은 **각자 다른 실험**을 맡는다.
- `experiment.json`이 needtocheck의 EXP 형식이며 step05가 이것을 모아 비교한다. 지표 JSON/CSV는 커밋, 가중치는 git 제외.
- 선정 규칙(step05 상단 상수): seed만 다른 실행을 한 설정으로 묶어(`CONFIG_KEYS`) 평균 macro F1을 구한다.
  최고 평균에서 `TOLERANCE`(0.01) 이내 설정은 동점 → 모두 seed `MIN_SEEDS`(3)개 이상이어야 함(기준 모델 제외, 부족하면 step05가 멈추고 `--band` 안내)
  → CPU 추론이 가장 빠른 후보의 `LATENCY_TIE_RATIO`(1.2)배 이내는 같은 속도 → 평균 F1 → 표준편차 → 파라미터 수 순 → 그 설정에서 F1 최고 seed의 가중치 사용. **test를 본 뒤에는 바꾸지 않는다.**
- `selection/test_metrics.json`이 생기면 step05 재선정이 막힌다(test를 본 뒤 재선택 금지).
- 다른 팀원 실행이 선정되면 가중치가 git에 없으므로 받아 오거나 같은 설정으로 재학습해야 서비스에서 쓸 수 있다.
- Windows 경로 길이(260자) 제한에 주의: 프로젝트를 깊은 폴더에 두면 체크포인트 저장이 실패할 수 있다.

## 분류 후보 모델 (`lesson_settings.CLASSIFIER_CANDIDATES`)
- `bert`(klue/bert-base, 필수), `roberta`, `electra-s`(14M), `electra-b`, `deberta-ko`(team-lucid/deberta-v3-base-korean) — 모두 로드·학습 확인됨
- 제외: distilkobert(구형, remote code + sentencepiece 필요), ModernBERT·NeoBERT(영어 전용 토크나이저 → 한국어 처리 부적합)
- 보류: mdeberta-v3-base(다국어, sentencepiece·protobuf 설치 필요)
- 실험 순서: ① `--model bert roberta electra-s electra-b deberta-ko --lr 1e-5 2e-5 5e-5` (seed 42) ② `--band --extra-seeds 43 44` → 후보 범위 안 설정에 없는 seed만 학습 (`--top-k K`는 상위 K개 기준) ③ step05 → 평균±표준편차 표. seed가 늘며 평균이 바뀌면 범위가 달라질 수 있으니 step05가 멈추면 ②를 다시 실행
- 학습률·에포크는 모든 모델에 같은 조건(lr 3개 × 5에포크)으로 고정하고 추가 조정하지 않는다. 한계로 기록할 것: deberta-ko·electra-s는 lr 1e-5/2e-5에서 5에포크 안에 수렴하지 못했고 5e-5에서야 0.98~0.99, deberta-ko는 5e-5에서도 마지막 epoch가 최고점(덜 학습됐을 가능성). validation이 4개 설정에서 1.0으로 포화되어 상위 모델 간 변별력이 낮음
- `--top-k`는 매번 그 시점의 평균 순위로 고르므로, seed가 추가되며 순위가 바뀌면 다시 실행했을 때 다른 설정이 반복 대상이 될 수 있다(상위 설정이 모두 같은 seed 수를 갖게 됨).
- 실행 이름 형식: `{alias}_lr{lr}_s{seed}`

## 보고서 (`reports/`)
- `model_selection_report.md`: 최종 제출용 보고서(그림 + 설명). 단계가 끝날 때마다 해당 장을 채운다(현재 3장 분류까지).
- `experiment_log_classification.md`: 분류 실험 상세 기록. 생성 실험도 `experiment_log_generation.md`로 같은 형식으로 남긴다.
- `figures/`, `generated/`: `uv run python scripts/build_report.py`가 실험 JSON에서 자동 생성. 직접 수정하지 않고 스크립트를 고친다.
- 보고서 수치는 반드시 artifacts의 JSON에서 가져온다. 라이선스 등 외부 정보는 출처를 확인하고 적는다.

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
- 정책·프롬프트·출력 형식은 `steps/prompts/document_policies.py` 한 곳에 둔다(`POLICIES`, `SCHEMAS`, `build_prompt`, `parse_output`, `PROMPT_VERSION`).
- 라벨별 출력 JSON: announcement{summary, audience, schedule, actions[], changes} / manual{summary, prerequisites[], steps[], cautions[]} / minutes{summary, discussion[], decisions[], owners[], follow_ups[]}. 없는 정보는 "미기재" 또는 [].
- `parse_output`: strict(응답 그대로 유효) / lenient(```json 코드 블록만 걷어 내면 유효). 품질 채점은 lenient 결과를 쓴다.
- 프롬프트를 바꾸면 `PROMPT_VERSION`도 바꾼다. eval 답변을 보고 프롬프트를 고치지 않는다.

## 생성 모델 (`lesson_settings.GENERATION_CANDIDATES`)
- 로컬(Ollama): `qwen3-4b`, `gemma3-4b`, `qwen2.5-7b`, `exaone3.5-7.8b`, `llama3.1-8b`(한국어 약세 대조군) / 상용: `openai`(**요금 발생**, step09 기본 목록에서 제외)
- step07: 스트리밍 호출로 TTFT, tokens/s, 토큰 수, Ollama 적재 시간·메모리(`/api/ps`)를 기록. 비용은 `lesson_settings`의 gpt-6-luna Standard 단가(입력 $0.10, 캐시 입력 $0.01, 출력 $0.50 /1M, 272K 초과 시 입력 2배·출력 1.5배, 2026-10-11 확인)와 실측 토큰으로 계산(`openai_cost`, `.env`로 덮어쓰기 가능). 로컬은 API 0 + `LOCAL_POWER_WATTS`로 전력량 추정
- 준비 호출(warm-up): 모델마다 train 첫 문장으로 1회, `warmup.json`에만 기록하고 요약 지표에서 제외. 적재 시간은 콜드 스타트 지표로 따로 보고
- step08: `create_chain`(분류→프롬프트), `create_service_chain`(분류→프롬프트→생성), `generate_prepared`(호출+형식 검사)
- step09: `--split smoke|development|eval --targets ... --repeats N` → `artifacts/{dataset}/{learner}/comparisons/{시각}_{split}/`(config, warmup, results.jsonl, summary.json/md). 모델별로 준비 호출 후 측정, 끝나면 언로드. 기록은 needtocheck의 GEN 형식
- 검수: `uv run python scripts/verify_service.py [폴더]` (호출 수, 프롬프트 공유, 채점 기준 비노출, 요약 일치)

## 요약 품질 채점 (step09b, `common/quality.py`, `prompts/judge_prompt.py`)
- **기준 원문은 `reports/evaluation_rubric.md`**. 채점 규칙을 바꾸면 이 문서·코드 상수·`JUDGE_VERSION`을 함께 바꾼다(eval 결과를 본 뒤에는 바꾸지 않음)
- 사람과 LLM이 **같은 체크리스트**를 채움: required_points별 포함/부분/누락, forbidden_claims별 위반, 그 외 근거 없는 주장, 수치·날짜 오류, 외국어 섞임, 정책 위반, 어색한 문장
- **주 지표 = 사용 가능률**: 형식 정상 + 환각 없음 + 수치 오류 없음 + 외국어 없음 + 핵심 포함률 ≥ 0.67. 함께 보고: 핵심 포함률(포함 1/부분 0.5), 환각률, 100점 감점 점수(환각 −30, 수치 −20, 외국어 −15, 정책 −5, 어색 −5, 누락 −8/부분 −4 최대 −25, 최대 감점 합계 정확히 100). 형식 실패는 0점·사용 불가. 외국어는 자동 탐지(원문에 없는 한자·가나·영문)와 OR
- `judge`: `JUDGE_TARGET`(openai, 요금 발생, temperature 0)로 전체 응답 채점 → `judge.jsonl`, `quality_llm.json`. 이어서 실행하면 성공한 항목은 건너뜀. OpenAI도 후보라 자기 답변 편향 가능성을 한계로 적음
- `sheet`: 반복 1회분 중 형식 정상 응답에서 모델×문서 유형별 균등 무작위 36건(seed 2026) → `human/human_sheet_rater1~4.csv`(블라인드, 채점자마다 순서 다름), `human_key.json`(채점 끝날 때까지 열지 않음), `README.md`(채점 안내)
- `aggregate`: 4명 시트 → 항목별 다수결(동률은 보수적으로 문제 있음/부분, `ties`에 남겨 논의), Fleiss' kappa, LLM vs 사람 다수결 일치율·Cohen's kappa·PABAK·감점 점수 상관 → `quality_human.json`. `llm_trust_decision`: 사용 가능 판정 kappa ≥ 0.7(한쪽 판정이 85% 초과면 PABAK ≥ 0.7)이면 LLM 채점 144건으로, 아니면 사람 다수결 36건 우선으로 모델 비교
- 채점 전 개발 데이터 3~5건으로 기준 맞추기(평가 데이터로 연습 금지)
- 계획: 1차 development 15건 × 로컬 5개 → 2차 eval 24건 × 2회 × (상위 로컬 2 + openai)

## 진행 상태 / 알려진 문제
- [완료] 0단계 기반 정리: `.gitignore`(가중치만 제외), `pyproject.toml` 정리·의존성(joblib, matplotlib, psutil), 문의용 코드 제거, 프롬프트 분리
- [완료] 1단계 분류: step03/04 자원·시간 기록(`experiment.json`)과 CLI, step05 베이스라인 포함 팀 단위 선정, step06 통합 로더, step11 지연·사례 기록, `verify_learning.py` 새 구조 대응
- `data/manifest.json` 해시는 이전 데이터 버전 기준이라 현재 v2 데이터와 다름 → `verify_learning.py`는 일치 여부만 기록
- 출력 스키마(answer/next_steps)가 요약에 맞지 않음 → 라벨별 요약 스키마로 교체 예정
- step09는 ollama 1개 vs openai 고정 → 여러 로컬 모델 비교, 비용 계산, LLM 채점 + 블라인드 사람 채점 추가 예정
- `tests/` 폴더 아직 없음 (pytest testpaths = `tests`)
