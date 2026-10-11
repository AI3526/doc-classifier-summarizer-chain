# 문서 분류 + 유형별 요약 서비스

사내 문서(공지·매뉴얼·회의록)를 **직접 학습한 분류기**로 분류하고, 예측한 유형에 맞는 형식으로 **생성 LLM이 요약**하는 서비스입니다. 분류와 생성은 LangChain으로 연결하고 FastAPI로 제공합니다.

> ⚠ **진행 중**: 분류기는 validation 기준으로 선정했고 test 평가는 아직 하지 않았습니다. 생성 모델 선택은 LLM 채점 기준의 **잠정** 결과이며, 보고서의 사람 채점 수치는 **가상 채점(미리보기용)**입니다.

## 왜 만들었나

- 사내 문서는 유형마다 읽는 사람이 챙겨야 할 내용이 다릅니다. 공지는 대상·일정·할 일, 매뉴얼은 순서·주의 사항, 회의록은 결정·담당·후속 조치가 중요합니다.
- 기밀 문서가 많고 문서가 길수록 상용 API 비용이 늘어납니다. 그래서 **로컬 LLM(Ollama)과 상용 API를 품질·시간·비용·운영 조건으로 실측 비교**해, 로컬로 충분한지 근거를 만듭니다.

## 처리 흐름

```mermaid
flowchart LR
    A[문서 원문] --> B["분류기<br/>klue/roberta-base<br/>(직접 파인튜닝)"]
    B -->|공지 / 매뉴얼 / 회의록| C["유형별 정책·출력 형식 선택"]
    C --> D["생성 LLM<br/>구조화 출력(JSON 스키마)"]
    D --> E["유형별 요약 JSON<br/>+ 분류 결과·실행 기록"]
```

학습한 분류기의 **실제 예측**으로 요약 정책과 출력 형식을 고릅니다. 정답 라벨은 쓰지 않습니다.

## 결과 한눈에 보기

### 분류 모델 (validation 120건)

| 모델 | Macro-F1 | 학습 시간 | 학습 VRAM | CPU 1건 추론 | |
|---|---|---|---|---|---|
| TF-IDF + 로지스틱 회귀 (기준) | 0.950 | 0.2초 | – | 0.3ms | |
| **klue/roberta-base** (lr 1e-5) | **1.000** (seed 3개) | 26초 | 2.1GB | 21ms | **선정** |
| koelectra-small (14M) | 0.983 | 26초 | 0.3GB | 7ms | 가장 가벼운 대안 |

- 사전학습 모델 5개 × 학습률 3개, 상위 설정은 seed 3개로 반복해 **29회** 파인튜닝했습니다.
- 선정 규칙: seed 평균 Macro-F1 최고점에서 0.01 이내는 동점 → 추론 속도 → 표준편차 → 파라미터 수 순으로 선택

![분류 모델 속도 vs 정확도](reports/figures/classification/speed_vs_f1.png)

### 생성 모델 (eval 24건 × 2회, LLM 채점)

| 모델 | 사용 가능률 | 환각률 | 응답 평균 | VRAM | 1,000건 API 비용 | |
|---|---|---|---|---|---|---|
| **gpt-6-luna** (상용) | **0.81** | **0.15** | 1.84s | – | $0.113 | **잠정 선택** |
| qwen2.5:7b (로컬) | 0.52 | 0.25 | 1.90s | 4.4GB | $0 | 로컬 중 환각 최소 |
| llama3.1:8b (로컬) | 0.52 | 0.46 | 2.20s | 4.9GB | $0 | |

- **사용 가능률**: 형식 정상, 환각·수치 오류·외국어 없음, 핵심 내용 포함률 0.67 이상인 요약의 비율
- 로컬 5개 모델을 개발 데이터로 먼저 비교해 상위 2개를 골랐습니다.
- 선택 규칙(결과를 보기 전에 정함): 로컬의 사용 가능률이 상용보다 0.10 이하로만 낮고 환각률이 0.10 이하로만 높으면 로컬 우선 → 두 로컬 모델 모두 0.29 차이로 미달해 **상용 모델이 잠정 선택**되었습니다.

![생성 모델 품질](reports/figures/generation/round2_quality.png)

## 실행 방법

**준비**: Python 3.12, [uv](https://docs.astral.sh/uv/), NVIDIA GPU(분류 학습), [Ollama](https://ollama.com/)(로컬 생성)

```powershell
uv sync
copy .env.example .env      # OPENAI_API_KEY 등 설정 (키는 출력·커밋하지 않음)
ollama pull qwen2.5:7b      # 비교에 쓰는 로컬 모델: llama3.1:8b, gemma3:4b, exaone3.5:7.8b, qwen3:4b-instruct-2507-q4_K_M
```

**단계별 실행** (`steps/`, 프로젝트 폴더에서)

| 단계 | 명령 | 내용 |
|---|---|---|
| 환경 확인 | `uv run python steps/check_environment.py` | GPU, Ollama 모델, 키 설정 여부 |
| 데이터 | `uv run python steps/step02_check_data.py` | 분할 중복·라벨 검사 |
| 기준 모델 | `uv run python steps/step03_train_baseline.py` | TF-IDF + 로지스틱 회귀 |
| 파인튜닝 | `uv run python steps/step04_train_classifier.py --model bert roberta electra-s electra-b deberta-ko --lr 1e-5 2e-5 5e-5` | 모델 × 학습률 |
| seed 반복 | `uv run python steps/step04_train_classifier.py --band --extra-seeds 43 44` | 동점 범위 설정에 seed 추가 |
| 분류기 선정 | `uv run python steps/step05_select_model.py` | 팀 전체 실행 비교 → `artifacts/documents/selection/` |
| 체인 시험 | `uv run python steps/step08_connect_chain.py --target qwen2.5-7b` | 분류 → 정책 → 생성 1건 |
| 생성 비교 | `uv run python steps/step09_compare.py --split eval --targets llama3.1-8b qwen2.5-7b openai --repeats 2 --structured` | openai 포함 시 **요금 발생** |
| 품질 채점 | `uv run python steps/step09b_score.py judge` → `sheet` → `aggregate` → `select` | LLM 채점, 사람 채점 시트, 집계, 생성 모델 선택 |
| API | `uv run python steps/step10_serve_api.py` | `http://127.0.0.1:8000/docs` |
| test 평가 | `uv run python steps/step11_evaluate_final.py` | 모델을 확정한 뒤 1회만 |
| 보고서 그림 | `uv run python scripts/build_report.py` | `reports/figures/`, `reports/generated/` |

## 폴더 구조

```
steps/                 단계별 실행 파일 (01~11), 설정 lesson_settings.py
  common/              지표·기록·자원 측정·품질 채점·그림
  prompts/             요약 정책·출력 스키마·채점 프롬프트
scripts/               검수(verify_*), 보고서 생성(build_report), 데이터 생성 원본
data/documents/        기본 데이터 (train 360 / validation 120 / test 120, 생성 개발 15 / 평가 24)
data/documents_advanced/  긴 문서 심화 데이터 (선택 과제)
artifacts/documents/   실험 기록 (지표 JSON·CSV는 커밋, 가중치는 제외)
  <learner>/runs/      분류 학습 실행
  <learner>/comparisons/  생성 비교 실행과 채점
  selection/           분류기·생성 모델 선택 결과, test 평가
reports/               보고서, 실험 기록, 평가 기준, 그림
```

## 문서

| 문서 | 내용 |
|---|---|
| [모델 선정 보고서](reports/model_selection_report.md) | 문제 정의부터 분류·생성 비교, 선정 이유, 한계까지 |
| [분류 실험 기록](reports/experiment_log_classification.md) | 설정, 29회 파인튜닝 결과, 선정 과정, 오류 분석 |
| [생성 실험 기록](reports/experiment_log_generation.md) | 프롬프트 v2~v4, 구조화 출력, 1·2차 비교 |
| [요약 평가 기준](reports/evaluation_rubric.md) | 체크리스트, 사용 가능률, 감점표, 사람·LLM 채점 절차, 선택 규칙 |

## 데이터

- 제공된 교육용 합성 데이터(`original-synthetic:synthetic-ko-v2.0`)를 정해진 분할 그대로 사용합니다.
- 라벨: announcement(공지), manual(매뉴얼), minutes(회의록), 각 분할에서 균형
- 생성 데이터의 `required_points`, `forbidden_claims`, `expected_label`은 채점에만 쓰고 모델 입력에는 넣지 않습니다.

## 한계

- validation에서 상위 분류 모델 4개 설정이 1.000으로 포화되어 우열을 가리기 어렵습니다.
- 생성 품질의 LLM 채점 모델(gpt-6-luna)이 비교 후보이기도 해, 자기 답변을 후하게 봤을 가능성이 있습니다. 블라인드 사람 채점으로 점검할 예정입니다.
- 기본 데이터는 문서가 짧아(약 100자) 긴 문서의 비용·성능 차이는 심화 데이터 실험에서 확인합니다.

## 팀

| 이름 | 역할 |
|---|---|
| _작성 예정_ | |
