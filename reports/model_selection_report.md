# 모델 선정 보고서: 문서 분류 + 유형별 요약 서비스

> 작성 중인 보고서입니다. 분류 모델 비교(3장)까지 채웠고, 나머지는 단계가 끝날 때마다 채웁니다.
> 그림과 표는 `uv run python scripts/build_report.py`로 다시 만들 수 있습니다.

| 장 | 상태 |
|---|---|
| 1. 문제 정의 | 작성 |
| 2. 데이터 | 작성 |
| 3. 분류 모델 비교와 선정 | 작성 (validation 기준) |
| 4. 분류 모델 최종 평가 (test) | 예정 |
| 5. 생성 모델 비교 (로컬 vs 상용) | 예정 |
| 6. 서비스 구성 (LangChain + FastAPI) | 예정 |
| 7. 최종 선정과 한계 | 예정 |
| 8. 팀원 역할과 기여 | 예정 |

---

## 1. 문제 정의

- **대상**: 사내 공지, 매뉴얼, 회의록을 읽고 핵심을 빠르게 파악해야 하는 구성원
- **입력**: 문서 원문 한 건
- **출력**: 문서 유형(분류 결과)과 유형에 맞는 요약
- **처리 흐름**: 학습한 분류기가 유형을 예측 → 예측 유형에 맞는 요약 정책 선택 → 생성 LLM이 원문과 정책으로 요약
- **팀 방향: 비용 최적화와 온프레미스**
  - 사내 문서는 외부로 보내기 어려운 내용이 많다.
  - 문서가 길수록 상용 API의 토큰 비용이 커진다.
  - 그래서 로컬 LLM(Ollama)과 상용 API를 품질, 시간, 비용, 운영 조건으로 실측 비교해 근거를 만든다.

## 2. 데이터

| 분할 | 행 수 | 용도 |
|---|---|---|
| train | 360 (라벨별 120) | 분류 모델 학습 |
| validation | 120 (라벨별 40) | 분류 모델 비교·선정 |
| test | 120 (라벨별 40) | 분류 최종 평가 |
| generation_development | 15 | 생성 흐름 개발 |
| generation_eval | 24 | 생성 모델 최종 비교 |

- 출처: 제공된 교육용 합성 데이터(`original-synthetic:synthetic-ko-v2.0`). 정해진 분할을 그대로 쓴다.
- 분할 사이에 같은 문장이나 같은 `group_id`가 겹치지 않는 것을 확인했다(step02).
- 생성 데이터의 `expected_label`, `required_points`, `forbidden_claims`는 채점에만 쓰고 모델 입력에는 넣지 않는다.
- 선택 과제용 심화 데이터 `documents_advanced`(평균 약 900자)는 별도로 실험한다.

## 3. 분류 모델 비교와 선정

상세 기록: [문서 분류 모델 실험 기록](experiment_log_classification.md) · 전체 표: [분류 실험 표](generated/classification_tables.md)

### 3.1 후보와 비교 조건

| 모델 | ID | 파라미터 | 출처·사용 조건 | 후보 선정 이유 |
|---|---|---|---|---|
| 기준 모델 | TF-IDF(문자 2~5gram) + 로지스틱 회귀 | 0.12M | scikit-learn (BSD-3) | 필수 기준선 |
| bert | klue/bert-base | 110.6M | Hugging Face, CC-BY-SA-4.0 | 필수 후보, 한국어 KLUE |
| roberta | klue/roberta-base | 110.6M | Hugging Face, 모델 카드에 라이선스 표기 없음 (확인 필요) | 같은 KLUE 계열의 RoBERTa 학습 방식 |
| electra-s | monologg/koelectra-small-v3-discriminator | 14.1M | Hugging Face, 모델 카드에 라이선스 표기 없음 (확인 필요) | 작은 모델의 정확도·자원 절충 확인 |
| electra-b | monologg/koelectra-base-v3-discriminator | 112.9M | Hugging Face, Apache-2.0 | 다른 사전학습 방식(ELECTRA) |
| deberta-ko | team-lucid/deberta-v3-base-korean | 135.3M | Hugging Face, Apache-2.0 | 최신 구조(DeBERTa-v3)의 한국어 모델 |

> 라이선스는 2026-10-11 Hugging Face 모델 카드 메타데이터 기준이다. 표기가 없는 모델은 원 저장소(KLUE, KoELECTRA GitHub)의 라이선스를 제출 전에 확인할 것.

- 모든 사전학습 모델: 전체 파인튜닝, max_length 192, 유효 배치 16, AdamW, BF16, 5 epoch, 학습률 {1e-5, 2e-5, 5e-5}
- 지표: validation Accuracy, Macro-F1(주 지표), 학습 시간, peak VRAM, CPU 1건 추론 시간
- seed 영향을 보려고 상위 설정은 seed 42/43/44로 반복했다.
- 실험 횟수: 기준 모델 1회 + 사전학습 29회

### 3.2 결과 한눈에 보기

![속도 vs 정확도](figures/classification/speed_vs_f1.png)

- 왼쪽은 모든 설정, 오른쪽은 노란 동점 범위를 확대한 것이다.
- 기준 모델(회색 네모)은 가장 빠르지만(0.3ms) Macro-F1 0.95로 동점 범위 밖이다.
- 상위 설정들은 모두 19~21ms에 모여 있다. 이 구간에서는 속도로 우열이 갈리지 않고, seed 평균 F1과 안정성이 기준이 된다.
- 검은 별이 선정한 설정(roberta, lr 1e-5)이다.

| 모델 | 최고 설정 lr | seed 수 | val Macro-F1 평균 | 학습 시간 | peak VRAM | CPU 1건 추론 |
|---|---|---|---|---|---|---|
| 기준 모델 | – | 1 | 0.950 | 0.18초 | – | 0.3ms |
| electra-s | 5e-5 | 1 | 0.983 | 25.7초 | 292MB | 7.1ms |
| bert | 2e-5 | 3 | 0.992 ± 0.000 | 24.0초 | 2,131MB | 19.1ms |
| **roberta** | **1e-5** | **3** | **1.000 ± 0.000** | 25.9초 | 2,132MB | 20.9ms |
| electra-b | 5e-5 | 3 | 1.000 ± 0.000 | 30.3초 | 2,174MB | 20.2ms |
| deberta-ko | 5e-5 | 3 | 0.989 ± 0.005 | 48.5초 | 2,610MB | 38.1ms |

### 3.3 학습률과 seed의 영향

![모델 × 학습률](figures/classification/lr_heatmap.png)

- bert, roberta, electra-b는 세 학습률 모두 0.98 이상으로 안정적이다.
- electra-s와 deberta-ko는 학습률이 낮으면 5 epoch 안에 수렴하지 못한다(deberta-ko 1e-5: 0.437).

![seed별 편차](figures/classification/seed_spread.png)

- roberta 1e-5·5e-5, electra-b 5e-5는 seed 3개 모두 1.000이다.
- bert 5e-5는 seed 42에서 1.000이었지만 다른 seed에서 0.983으로, seed 하나의 결과만으로 고르면 안 된다는 것을 보여 준다.

### 3.4 학습 곡선

![학습 곡선](figures/classification/learning_curves.png)

- 모델별 최고 설정의 seed 42 실행이다.
- bert, roberta, electra-b는 2~3 epoch에 수렴한다.
- deberta-ko는 5 epoch까지 validation loss가 계속 내려가 더 학습할 여지가 있다.
- electra-s는 validation loss가 0.66으로 높지만 정확도는 0.98이다. 예측의 확신도가 낮은 상태다.

### 3.5 자원 비교

![자원 비교](figures/classification/resources.png)

- 모든 후보가 8GB GPU에서 여유 있게 학습된다(최대 2.6GB).
- electra-s는 VRAM 292MB, CPU 추론 7ms로 가장 가볍다.
- deberta-ko는 학습 시간과 추론 시간이 다른 base 모델의 약 2배다.

### 3.6 선정 규칙과 결과

선정 규칙:
1. seed만 다른 실행을 한 설정으로 묶어 validation Macro-F1 평균을 구한다.
2. 평균 최고점에서 0.01(validation 약 1건) 이내는 동점으로 본다.
3. 동점 후보는 모두 seed 3개 이상이어야 한다.
4. CPU 추론 시간이 가장 빠른 후보의 1.2배 이내는 같은 속도로 본다.
5. 평균 Macro-F1 → 표준편차 → 파라미터 수 순으로 고른다.

**선정 결과: klue/roberta-base (lr 1e-5)**
- 동점 범위의 5개 설정이 모두 같은 속도 구간(18.9~20.9ms)에 들어왔다.
- 그중 평균 1.000이 roberta 1e-5, roberta 5e-5, electra-b 5e-5였고, 세 설정 모두 표준편차가 0이었다.
- 파라미터가 더 적은 roberta가 남았다. 두 학습률은 모든 기준에서 같아, 정해 둔 마지막 기준(설정 이름 순)으로 1e-5를 골랐다.
- 서비스에는 seed 42의 가중치를 쓴다.

| 선정 모델 (validation) | 기준 모델 (validation) |
|---|---|
| ![선정 모델 혼동행렬](figures/classification/confusion_validation_selected.png) | ![기준 모델 혼동행렬](figures/classification/confusion_validation_baseline_reference.png) |

- 기준 모델의 오류 6건은 모두 공지↔매뉴얼 사이에서 났다. 공지 안에 이용 방법("확인한다", "싣지 않는다")이 섞인 문장이 매뉴얼로 분류됐다.
- 회의록은 모든 모델이 한 번도 틀리지 않았다.

### 3.7 분류 단계의 한계

- validation이 120건이고 상위 4개 설정이 1.000으로 포화되어, 상위 모델 사이의 우열은 validation으로 가릴 수 없다.
- 학습률과 epoch는 모든 모델에 같은 조건으로 고정했다. electra-s와 deberta-ko는 더 높은 학습률이나 더 긴 학습에서 결과가 달라질 수 있다.
- CPU 추론 시간은 노트북 한 대에서 측정했고 실행마다 약 ±15% 흔들린다.
- 합성 데이터라 실제 사내 문서보다 짧고 정형적이다.

## 4. 분류 모델 최종 평가 (test)

_예정: 선택 실험(심화 데이터)까지 마친 뒤 step11로 1회 평가한다. 선정 모델과 기준 모델의 test Accuracy·Macro-F1, 혼동행렬, 대표 성공·실패 사례를 넣는다._

## 5. 생성 모델 비교 (로컬 vs 상용)

_예정: Ollama 로컬 모델(qwen3:4b, gemma3:4b, qwen2.5:7b, llama3.1:8b)과 상용 API를 같은 입력·지시문으로 비교한다. 품질(LLM 채점 + 블라인드 사람 채점), 응답 시간, 비용(실측·추정 구분), 운영 조건을 넣는다._

## 6. 서비스 구성 (LangChain + FastAPI)

_예정: 처리 흐름도, API 엔드포인트, 정상 요청과 실패 요청 예시, 분류기와 로컬 LLM을 함께 실행할 때의 메모리를 넣는다._

## 7. 최종 선정과 한계

_예정_

## 8. 팀원 역할과 기여

_예정_
