# 문서 분류 모델 실험 기록 (documents)

- 실험일: 2026-10-11
- 실행 학습자 폴더: `artifacts/documents/learner_park/`
- 범위: 기준 모델(TF-IDF + 로지스틱 회귀) 1회, 사전학습 모델 파인튜닝 29회, validation 기준 선정
- test 평가: 2026-10-13 1회. 선정 0.983 / 기준 모델 0.967 (13장)
- 이 문서의 모든 수치는 `artifacts/documents/**/experiment.json`과 `scripts/build_report.py`가 만든 [분류 실험 표](generated/classification_tables.md)에서 가져왔다.

---

## 1. 목적

사내 문서를 공지(`announcement`), 매뉴얼(`manual`), 회의록(`minutes`) 세 유형으로 분류하는 모델을 고른다. 분류 결과는 생성 LLM에 넘길 요약 정책을 고르는 데 쓰인다. 같은 train으로 학습하고 같은 validation과 지표로 비교해 서비스에 쓸 분류기 하나를 정한다.

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| OS | Windows 11 (10.0.26200) |
| CPU | Intel64 Family 6 Model 198 (torch 스레드 24) |
| RAM | 31.4 GB |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM |
| Python / torch / transformers / scikit-learn | 3.12.13 / 2.11.0+cu128 / 4.57.6 / 1.9.1 |

## 3. 데이터

| 분할 | 행 수 | 라벨별 | 용도 |
|---|---|---|---|
| train | 360 | 각 120 | 학습 |
| validation | 120 | 각 40 | 모델 비교·선정, 체크포인트 선택 |
| test | 120 | 각 40 | 최종 평가 (선정을 고정한 뒤 13장에서 1회) |

- 라벨 정의(`data/documents/labels.json`)
  - announcement: 대상자에게 일정, 변경, 참여, 주의사항을 알리는 문서
  - manual: 사용자가 작업을 수행하도록 절차, 조건, 방법을 설명하는 문서
  - minutes: 이미 진행된 논의의 의견, 결정, 담당, 후속 작업을 기록한 문서
- 입력은 `text`만 사용하고 `id`, `group_id`, `source`는 쓰지 않는다.
- step02에서 분할 사이에 id, group_id, 정규화한 문장이 겹치지 않는 것을 확인했다.
- 한 상황(`group_id`)마다 표현만 다른 두 문장(v1/v2)이 있다. 그래서 validation 120건은 실질적으로 60개 상황이고, 오분류도 쌍으로 나타나는 경우가 많다.
- 토큰 길이(klue 토크나이저): train 평균 53.4, 최대 69. `max_length=192`에서 잘린 문장은 없다.

## 4. 공통 설정

### 4.1 기준 모델

| 항목 | 값 |
|---|---|
| 특징 | `TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)` — train에서만 fit |
| 분류기 | `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)` |
| 사전학습 가중치 | 사용하지 않음 |

### 4.2 사전학습 모델 파인튜닝 (모든 모델에 동일)

| 항목 | 값 |
|---|---|
| 학습 방식 | 본체 + 분류층 전체 파인튜닝 (`AutoModelForSequenceClassification`, 라벨 3개) |
| max_length | 192 (배치마다 동적 padding, 8의 배수) |
| 배치 | 4 × gradient accumulation 4 = 유효 배치 16 |
| optimizer | AdamW (weight_decay 0.01) |
| scheduler | 전체 update의 10% 선형 warmup 뒤 선형 감소 |
| gradient clipping | 1.0 |
| 정밀도 | BF16 autocast |
| gradient checkpointing | 사용 |
| epoch | 5 — epoch마다 validation macro F1을 계산해 **가장 높은 epoch의 가중치만 저장** |
| 학습률 | 1e-5, 2e-5, 5e-5 |
| seed | 42 (반복 실험에서 43, 44 추가) |

### 4.3 후보 모델

| 짧은 이름 | Hugging Face ID | 파라미터 | 비고 |
|---|---|---|---|
| bert | klue/bert-base | 110.6M | 필수 후보 |
| roberta | klue/roberta-base | 110.6M | |
| electra-s | monologg/koelectra-small-v3-discriminator | 14.1M | 작고 빠른 후보 |
| electra-b | monologg/koelectra-base-v3-discriminator | 112.9M | |
| deberta-ko | team-lucid/deberta-v3-base-korean | 135.3M | 한국어 전용 DeBERTa-v3 |

후보에서 뺀 모델과 이유:
- monologg/distilkobert: 오래된 모델이고, 저장소의 별도 코드 실행(`trust_remote_code`)과 sentencepiece가 필요하다.
- answerdotai/ModernBERT-base: 영어 토크나이저라 한국어 예문이 바이트 조각 90토큰으로 쪼개진다(klue는 28토큰).
- chandar-lab/NeoBERT: 영어 어휘(30,522)이고 별도 코드 실행이 필요하다.
- microsoft/mdeberta-v3-base: 다국어 비교 후보로 보류했다. sentencepiece·protobuf 설치가 필요하고 약 280M으로 무겁다.

## 5. 측정 방법

| 지표 | 측정 범위 |
|---|---|
| validation Accuracy / Macro-F1 / Weighted-F1 | 저장한(최고 epoch) 가중치 기준 |
| 학습 시간 | 학습 루프만 합산, 매 epoch validation 시간 제외 (`train_time_sec`) |
| CPU 1건 추론 시간 | 저장한 가중치를 CPU로 다시 불러와 validation 120건을 1건씩 예측. 토큰화부터 라벨 결정까지 포함, 모델 로딩 제외, 처음 3건은 준비 호출로 제외 |
| peak VRAM | `torch.cuda.max_memory_allocated()`. 모델 로딩 전에 초기화하고, 가중치·optimizer 상태·활성값을 모두 포함한 학습 전체의 최대치 |

같은 설정을 다시 실행해도 CPU 추론 시간은 18~24ms 범위에서 흔들렸다(roberta lr 1e-5: 23.8 / 19.0 / 20.0ms). 크기가 같은 모델끼리의 1~2ms 차이는 측정 오차로 본다.

## 6. 실험 절차

| 차수 | 내용 | 실행 수 | 명령 |
|---|---|---|---|
| 0 | 기준 모델 | 1 | `uv run python steps/step03_train_baseline.py` |
| 1 | 후보 5개 × 학습률 3개 (seed 42) | 15 | `uv run python steps/step04_train_classifier.py --model bert roberta electra-s electra-b deberta-ko --lr 1e-5 2e-5 5e-5` |
| 2 | 평균 1.000 동점 4개 설정에 seed 43, 44 추가 | 8 | `uv run python steps/step04_train_classifier.py --top-k 4 --extra-seeds 43 44` |
| 3 | 후보 범위(평균 최고점 −0.01 이내) 안에서 seed가 부족한 설정에 seed 43, 44 추가 | 6 | `uv run python steps/step04_train_classifier.py --band --extra-seeds 43 44` |
| 4 | 선정과 검수 | – | `uv run python steps/step05_select_model.py`, `uv run python scripts/verify_learning.py` |

모든 실행이 실패 없이 끝났다. `verify_learning.py`로 29개 사전학습 실행 모두 본체와 분류층 가중치가 초기값에서 바뀌었는지, 저장한 epoch와 validation 기록이 일치하는지 확인했다.

## 7. 결과

### 7.1 1차: 모델 × 학습률 (seed 42, validation Macro-F1)

| 모델 | lr 1e-5 | lr 2e-5 | lr 5e-5 | 최고 epoch (5e-5) |
|---|---|---|---|---|
| electra-s | 0.8290 | 0.8997 | 0.9833 | 4 |
| bert | 0.9833 | 0.9917 | 1.0000 | 3 |
| roberta | 1.0000 | 0.9917 | 1.0000 | 2 |
| electra-b | 0.9833 | 0.9833 | 1.0000 | 3 |
| deberta-ko | 0.4369 | 0.5694 | 0.9917 | 5 |
| 기준 모델 | 0.9500 (학습률 없음) | | | |

### 7.2 seed 반복 결과 (seed 42/43/44)

| 설정 | seed별 Macro-F1 | 평균 ± 표준편차 |
|---|---|---|
| roberta 1e-5 | 1.0000 / 1.0000 / 1.0000 | 1.0000 ± 0.0000 |
| roberta 5e-5 | 1.0000 / 1.0000 / 1.0000 | 1.0000 ± 0.0000 |
| electra-b 5e-5 | 1.0000 / 1.0000 / 1.0000 | 1.0000 ± 0.0000 |
| roberta 2e-5 | 0.9917 / 0.9917 / 1.0000 | 0.9944 ± 0.0048 |
| bert 2e-5 | 0.9917 / 0.9917 / 0.9917 | 0.9917 ± 0.0000 |
| deberta-ko 5e-5 | 0.9917 / 0.9917 / 0.9833 | 0.9889 ± 0.0048 |
| bert 5e-5 | 1.0000 / 0.9833 / 0.9833 | 0.9889 ± 0.0096 |

### 7.3 모델별 최고 설정과 자원

| 모델 | lr | seed 수 | Macro-F1 평균 | 학습 시간(5 epoch) | peak VRAM | CPU 1건 추론 |
|---|---|---|---|---|---|---|
| 기준 모델 | – | 1 | 0.9500 | 0.18초 | – (CPU) | 0.33ms |
| electra-s | 5e-5 | 1 | 0.9833 | 25.7초 | 292MB | 7.1ms |
| bert | 2e-5 | 3 | 0.9917 | 24.0초 | 2,131MB | 19.1ms |
| roberta | 1e-5 | 3 | 1.0000 | 25.9초 | 2,132MB | 20.9ms |
| electra-b | 5e-5 | 3 | 1.0000 | 30.3초 | 2,174MB | 20.2ms |
| deberta-ko | 5e-5 | 3 | 0.9889 | 48.5초 | 2,610MB | 38.1ms |

전체 실행 목록은 [분류 실험 표](generated/classification_tables.md)의 "실행별 전체"에 있다.

## 8. 선정

### 8.1 선정 규칙 (`steps/step05_select_model.py`)

1. seed만 다른 실행은 한 설정으로 묶어 validation Macro-F1 평균을 구한다.
2. 평균 최고점에서 0.01 이내인 설정은 동점으로 보고 후보로 둔다. validation 120건에서 1건 차이가 약 0.008이다.
3. 후보 설정은 모두 seed가 3개 이상이어야 한다. 기준 모델은 같은 데이터에서 결과가 늘 같으므로 예외다.
4. CPU 1건 추론 시간이 가장 빠른 후보의 1.2배 이내인 설정은 같은 속도로 본다.
5. 같은 속도의 설정 중 평균 Macro-F1 → 작은 표준편차 → 적은 파라미터 순으로 고른다.
6. 고른 설정 안에서는 validation Macro-F1이 가장 높은 seed의 가중치를 서비스에 쓴다.

### 8.2 적용 결과

| 단계 | 남은 설정 |
|---|---|
| 동점 범위 (≥ 0.990) | roberta 1e-5, roberta 5e-5, electra-b 5e-5, roberta 2e-5, bert 2e-5 (모두 seed 3개) |
| 같은 속도 (가장 빠른 18.9ms × 1.2 = 22.7ms 이내) | 다섯 설정 모두 (18.9~20.9ms) |
| 평균 Macro-F1 최고 | roberta 1e-5, roberta 5e-5, electra-b 5e-5 (모두 1.000) |
| 표준편차 | 세 설정 모두 0 |
| 파라미터 수 | roberta 110.6M < electra-b 112.9M → roberta 1e-5, roberta 5e-5 |
| 최종 | 남은 두 설정은 모든 기준이 같아 설정 이름 순으로 **roberta 1e-5** |

**선정: klue/roberta-base, lr 1e-5, seed 42** (`artifacts/documents/learner_park/runs/20261011-210721_roberta_lr1e-5_s42`)

- validation Accuracy 1.000, Macro-F1 1.000, 혼동행렬 `[[40,0,0],[0,40,0],[0,0,40]]`
- roberta 1e-5와 5e-5는 validation에서 구별되지 않는다. 어느 쪽을 써도 같은 수준이다.
- test에서 함께 보고할 기준 모델: `20261011-210534_baseline_char2-5`

## 9. 오류 분석 (validation)

**기준 모델** (6건 오류, 혼동행렬 `[[36,4,0],[2,38,0],[0,0,40]]`)
- 오류가 모두 공지↔매뉴얼 사이에서 났고, 3개 상황의 v1/v2 쌍이 함께 틀렸다.
  - 공지 → 매뉴얼: "무거운 교구를 옮길 바퀴 달린 상자 세 개를 새로 마련했다. 이용자는 상자를 반납할 때 내부가 비었는지 확인한다…" — 새 물품을 알리는 공지이지만 "확인한다", "싣지 않는다" 같은 지시 문장이 많다.
  - 공지 → 매뉴얼: "작품 촬영 공간의 예약 확정 알림에 다음 주부터 색상 표식을 추가한다…" — 변경 공지 안에 사용 방법이 섞여 있다.
  - 매뉴얼 → 공지: "행사 뒤 들어온 질문은 답변할 담당이 정해진 건과 미배정 건으로 나눈다…" — 절차인데 공지처럼 읽히는 문체다.

**사전학습 모델** (29회 실행에서 반복해 틀린 문장)

| 틀린 실행 수 | 정답 → 예측 | 문장 요지 |
|---|---|---|
| 11 | 공지 → 매뉴얼 | 예약 확정 알림 색상 표식 추가 (v2, 제목 포함) |
| 7 | 매뉴얼 → 공지 | 행사 뒤 질문 분류 절차 (v1) |
| 5 | 공지 → 매뉴얼 | 전시 복제본 회색 표식 (v2) |
| 5 | 매뉴얼 → 공지 | 행사 뒤 질문 분류 절차 (v2) |
| 5 | 공지 → 매뉴얼 | 바퀴 달린 상자 마련 (v2) |

- 기준 모델이 틀린 문장을 사전학습 모델도 자주 틀린다. 공지 안에 이용 방법이 섞인 문장은 라벨 정의상 경계에 있다.
- 회의록은 모든 모델, 모든 실행에서 한 번도 틀리지 않았다. 과거형 서술, 결정·담당 표현이 뚜렷하기 때문으로 보인다.

## 10. 관찰

1. **validation 포화**: 4개 설정이 seed 3개 모두에서 120건을 전부 맞혔다. validation만으로는 상위 모델들의 우열을 가릴 수 없다.
2. **seed 영향**: bert 5e-5는 seed 42에서 1.000이었지만 3개 평균은 0.989±0.010이다. seed 1개 점수는 우연히 높게 나올 수 있다.
3. **학습률 민감도**: bert, roberta, electra-b는 세 학습률 모두 0.98 이상이다. 반면 electra-s(14M)와 deberta-ko는 1e-5·2e-5에서 5 epoch 안에 수렴하지 못했다(deberta-ko 0.44, 0.57).
4. **electra-s의 효율**: Macro-F1 0.983으로 최고점보다 2건 적게 맞히는 수준이다. 하지만 VRAM은 1/7(292MB), CPU 추론은 약 2.7배 빠르다(7.1ms). 자원이 제한된 환경의 대안이 될 수 있다.
5. **electra-s의 확신도**: validation loss가 0.66으로 높은데 Accuracy는 0.983이다. 예측은 맞지만 확률이 한쪽으로 크게 쏠리지 않은 상태다(학습 곡선 참고).
6. **기준 모델**: 0.95로 사전학습 모델보다 6건 정도 더 틀린다. 대신 학습 0.2초, 추론 0.3ms, GPU가 필요 없다.

## 11. 한계

- 학습률(1e-5, 2e-5, 5e-5)과 epoch(5)는 모든 모델에 같은 조건으로 고정했고 모델별로 따로 조정하지 않았다.
  - electra-s와 deberta-ko는 가장 높은 학습률에서야 성능이 나왔으므로 더 높은 학습률이 맞을 수 있다.
  - deberta-ko는 lr 5e-5에서도 seed 3개 중 2개의 최고점이 마지막 epoch(5)여서 덜 학습됐을 가능성이 있다.
- validation이 120건(60개 상황)이고 상위권에서 포화되어 상위 모델 간 차이를 통계적으로 구분할 수 없다.
- seed 반복은 후보 범위 안의 설정에만 했다. 범위 밖 설정(예: electra-s 5e-5)은 seed 1개 결과다.
- CPU 추론 시간은 노트북 한 대에서 다른 프로세스가 도는 상태로 측정했고, 실행마다 약 ±15% 흔들린다.
- 데이터는 LLM으로 만든 교육용 합성 문서(`original-synthetic:synthetic-ko-v2.0`)라 실제 사내 문서보다 짧고 정형적이다.
- `data/manifest.json`의 해시는 이전 데이터 버전 기준이라 현재 v2 파일과 일치하지 않는다(검수 기록 `learning_verification.json`에 일치 여부를 남김).

## 12. 재현 방법과 결과 위치

```powershell
uv sync
uv run python steps/step02_check_data.py
uv run python steps/step03_train_baseline.py
uv run python steps/step04_train_classifier.py --model bert roberta electra-s electra-b deberta-ko --lr 1e-5 2e-5 5e-5
uv run python steps/step04_train_classifier.py --band --extra-seeds 43 44   # step05가 멈추면 다시 실행
uv run python steps/step05_select_model.py
uv run python scripts/verify_learning.py
uv run python scripts/build_report.py
```

| 위치 | 내용 |
|---|---|
| `artifacts/documents/learner_park/runs/<실행>/` | `config.json`, `experiment.json`, `validation_metrics.json`, `validation_errors.csv`, `epochs.jsonl`, `learning_curve.png` (가중치는 git 제외) |
| `artifacts/documents/selection/` | `selected.json`, `comparison.md/csv`, `comparison_by_config.csv`, 혼동행렬, `learning_verification.json` |
| `reports/generated/classification_tables.md` | 모델별·설정별·실행별 표 |
| `reports/figures/classification/` | 보고서용 그림 |

## 13. 최종 평가 (test, 1회)

- 실행: `uv run python steps/step11_evaluate_final.py` (2026-10-13), 결과 `artifacts/documents/selection/test_*.{json,csv,png}`
- 고정: 선정 `20261011-210721_roberta_lr1e-5_s42`, 기준 모델 `20261011-210534_baseline_char2-5`. 이 결과로 다시 고르지 않는다(step05 재선정 막힘).

| 모델 | Accuracy | Macro-F1 | Weighted-F1 | CPU 1건 추론 | 혼동행렬 (행 = 정답 announcement / manual / minutes) |
|---|---|---|---|---|---|
| klue/roberta-base (선정) | 0.9833 | 0.9833 | 0.9833 | 18.3ms | [[40, 0, 0], [2, 38, 0], [0, 0, 40]] |
| TF-IDF + 로지스틱 회귀 (기준) | 0.9667 | 0.9666 | 0.9666 | 0.5ms | [[36, 4, 0], [0, 40, 0], [0, 0, 40]] |

- 선정 분류기 오답 2건은 같은 상황("재료 요청 철회")의 두 문서로, 대상자에게 할 일을 알리는 문장이라 공지로 예측했다(확신도 0.95, 0.75).
- 기준 모델은 반대로 공지 4건을 매뉴얼로 틀렸다.
- 심화 test(본문 추출) 결과는 [긴 문서 실험 기록](experiment_log_long_documents.md) 8장.
