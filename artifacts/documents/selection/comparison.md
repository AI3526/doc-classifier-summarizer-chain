# 분류 모델 비교 (documents, validation)

- 선정 규칙: 설정별 validation macro F1 평균의 최고점에서 0.01 이내이고 seed 3개 이상인 설정 중, 가장 빠른 CPU 추론 시간의 1.2배 이내 설정에서 평균 macro F1 → 표준편차 → 파라미터 수 순으로 선택, 그 안에서 macro F1이 가장 높은 seed
- 선정: `documents/learner_park/runs/20261011-210721_roberta_lr1e-5_s42`
- 작성: 2026-10-11T21:32:41+09:00

## 설정별 (seed 평균)

| 선정 | model_alias | model | learning_rate | n_seeds | macro_f1_mean | macro_f1_std | accuracy_mean | train_time_sec_mean | cpu_latency_ms_mean | peak_vram_mb_max | params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **✔** | roberta | klue/roberta-base | 1e-05 | 3 | 1.0000 | 0.0000 | 1.0000 | 25.9 | 20.9 | 2132.2 | 110620419 |
|  | roberta | klue/roberta-base | 5e-05 | 3 | 1.0000 | 0.0000 | 1.0000 | 25.9 | 18.9 | 2132.2 | 110620419 |
|  | electra-b | monologg/koelectra-base-v3-discriminator | 5e-05 | 3 | 1.0000 | 0.0000 | 1.0000 | 30.3 | 20.2 | 2174.1 | 112923651 |
|  | roberta | klue/roberta-base | 2e-05 | 3 | 0.9944 | 0.0048 | 0.9944 | 25.7 | 19.8 | 2132.2 | 110620419 |
|  | bert | klue/bert-base | 2e-05 | 3 | 0.9917 | 0.0000 | 0.9917 | 24.0 | 19.1 | 2131.4 | 110619651 |
|  | deberta-ko | team-lucid/deberta-v3-base-korean | 5e-05 | 3 | 0.9889 | 0.0048 | 0.9889 | 48.5 | 38.1 | 2609.7 | 135272451 |
|  | bert | klue/bert-base | 5e-05 | 3 | 0.9889 | 0.0096 | 0.9889 | 25.9 | 19.3 | 2131.4 | 110619651 |
|  | bert | klue/bert-base | 1e-05 | 1 | 0.9833 | - | 0.9833 | 26.2 | 18.2 | 2131.4 | 110619651 |
|  | electra-s | monologg/koelectra-small-v3-discriminator | 5e-05 | 1 | 0.9833 | - | 0.9833 | 25.7 | 7.0740 | 292.2 | 14122755 |
|  | electra-b | monologg/koelectra-base-v3-discriminator | 1e-05 | 1 | 0.9833 | - | 0.9833 | 30.3 | 21.3 | 2174.1 | 112923651 |
|  | electra-b | monologg/koelectra-base-v3-discriminator | 2e-05 | 1 | 0.9833 | - | 0.9833 | 30.7 | 21.5 | 2174.1 | 112923651 |
|  | baseline | tfidf-char+logistic-regression | - | 1 | 0.9500 | - | 0.9500 | 0.1799 | 0.3322 | - | 118710 |
|  | electra-s | monologg/koelectra-small-v3-discriminator | 2e-05 | 1 | 0.8997 | - | 0.9000 | 25.6 | 7.1083 | 292.2 | 14122755 |
|  | electra-s | monologg/koelectra-small-v3-discriminator | 1e-05 | 1 | 0.8290 | - | 0.8333 | 25.7 | 7.3968 | 292.2 | 14122755 |
|  | deberta-ko | team-lucid/deberta-v3-base-korean | 2e-05 | 1 | 0.5694 | - | 0.6750 | 48.3 | 39.2 | 2601.4 | 135272451 |
|  | deberta-ko | team-lucid/deberta-v3-base-korean | 1e-05 | 1 | 0.4369 | - | 0.5083 | 47.2 | 39.0 | 2601.4 | 135272451 |

## 실행별

| 선정 | learner | run_name | kind | model | learning_rate | seed | best_epoch | validation_accuracy | validation_macro_f1 | validation_weighted_f1 | train_time_sec | cpu_latency_ms | peak_vram_mb | params | weights_available |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | learner_park | 20261011-210649_bert_lr5e-5_s42 | transformer | klue/bert-base | 5e-05 | 42 | 3 | 1.0000 | 1.0000 | 1.0000 | 25.9 | 19.5 | 2131.4 | 110619651 | True |
| **✔** | learner_park | 20261011-210721_roberta_lr1e-5_s42 | transformer | klue/roberta-base | 1e-05 | 42 | 4 | 1.0000 | 1.0000 | 1.0000 | 25.6 | 23.8 | 2132.2 | 110620419 | True |
|  | learner_park | 20261011-210823_roberta_lr5e-5_s42 | transformer | klue/roberta-base | 5e-05 | 42 | 2 | 1.0000 | 1.0000 | 1.0000 | 25.7 | 19.1 | 2132.2 | 110620419 | True |
|  | learner_park | 20261011-211132_electra-b_lr5e-5_s42 | transformer | monologg/koelectra-base-v3-discriminator | 5e-05 | 42 | 3 | 1.0000 | 1.0000 | 1.0000 | 30.1 | 20.4 | 2174.1 | 112923651 | True |
|  | learner_park | 20261011-211617_roberta_lr1e-5_s43 | transformer | klue/roberta-base | 1e-05 | 43 | 3 | 1.0000 | 1.0000 | 1.0000 | 26.3 | 19.0 | 2132.2 | 110620419 | True |
|  | learner_park | 20261011-211648_roberta_lr1e-5_s44 | transformer | klue/roberta-base | 1e-05 | 44 | 4 | 1.0000 | 1.0000 | 1.0000 | 25.9 | 20.0 | 2132.2 | 110620419 | True |
|  | learner_park | 20261011-211720_roberta_lr5e-5_s43 | transformer | klue/roberta-base | 5e-05 | 43 | 4 | 1.0000 | 1.0000 | 1.0000 | 26.2 | 19.0 | 2132.2 | 110620419 | True |
|  | learner_park | 20261011-211752_roberta_lr5e-5_s44 | transformer | klue/roberta-base | 5e-05 | 44 | 3 | 1.0000 | 1.0000 | 1.0000 | 25.9 | 18.8 | 2132.2 | 110620419 | True |
|  | learner_park | 20261011-211823_electra-b_lr5e-5_s43 | transformer | monologg/koelectra-base-v3-discriminator | 5e-05 | 43 | 2 | 1.0000 | 1.0000 | 1.0000 | 30.1 | 20.1 | 2174.1 | 112923651 | True |
|  | learner_park | 20261011-211858_electra-b_lr5e-5_s44 | transformer | monologg/koelectra-base-v3-discriminator | 5e-05 | 44 | 3 | 1.0000 | 1.0000 | 1.0000 | 30.7 | 20.2 | 2174.1 | 112923651 | True |
|  | learner_park | 20261011-213005_roberta_lr2e-5_s44 | transformer | klue/roberta-base | 2e-05 | 44 | 2 | 1.0000 | 1.0000 | 1.0000 | 25.6 | 19.0 | 2132.2 | 110620419 | True |
|  | learner_park | 20261011-210617_bert_lr2e-5_s42 | transformer | klue/bert-base | 2e-05 | 42 | 3 | 0.9917 | 0.9917 | 0.9917 | 26.2 | 18.5 | 2131.4 | 110619651 | True |
|  | learner_park | 20261011-210752_roberta_lr2e-5_s42 | transformer | klue/roberta-base | 2e-05 | 42 | 2 | 0.9917 | 0.9917 | 0.9917 | 25.5 | 20.4 | 2132.2 | 110620419 | True |
|  | learner_park | 20261011-211403_deberta-ko_lr5e-5_s42 | transformer | team-lucid/deberta-v3-base-korean | 5e-05 | 42 | 5 | 0.9917 | 0.9917 | 0.9917 | 47.8 | 38.8 | 2601.4 | 135272451 | True |
|  | learner_park | 20261011-212837_bert_lr2e-5_s43 | transformer | klue/bert-base | 2e-05 | 43 | 2 | 0.9917 | 0.9917 | 0.9917 | 23.4 | 18.5 | 2131.4 | 110619651 | True |
|  | learner_park | 20261011-212906_bert_lr2e-5_s44 | transformer | klue/bert-base | 2e-05 | 44 | 3 | 0.9917 | 0.9917 | 0.9917 | 22.5 | 20.5 | 2131.4 | 110619651 | True |
|  | learner_park | 20261011-212934_roberta_lr2e-5_s43 | transformer | klue/roberta-base | 2e-05 | 43 | 4 | 0.9917 | 0.9917 | 0.9917 | 25.9 | 20.0 | 2132.2 | 110620419 | True |
|  | learner_park | 20261011-213036_deberta-ko_lr5e-5_s43 | transformer | team-lucid/deberta-v3-base-korean | 5e-05 | 43 | 4 | 0.9917 | 0.9917 | 0.9917 | 48.4 | 37.7 | 2609.7 | 135272451 | True |
|  | learner_park | 20261011-210546_bert_lr1e-5_s42 | transformer | klue/bert-base | 1e-05 | 42 | 4 | 0.9833 | 0.9833 | 0.9833 | 26.2 | 18.2 | 2131.4 | 110619651 | True |
|  | learner_park | 20261011-210951_electra-s_lr5e-5_s42 | transformer | monologg/koelectra-small-v3-discriminator | 5e-05 | 42 | 4 | 0.9833 | 0.9833 | 0.9833 | 25.7 | 7.0740 | 292.2 | 14122755 | True |
|  | learner_park | 20261011-211020_electra-b_lr1e-5_s42 | transformer | monologg/koelectra-base-v3-discriminator | 1e-05 | 42 | 3 | 0.9833 | 0.9833 | 0.9833 | 30.3 | 21.3 | 2174.1 | 112923651 | True |
|  | learner_park | 20261011-211056_electra-b_lr2e-5_s42 | transformer | monologg/koelectra-base-v3-discriminator | 2e-05 | 42 | 2 | 0.9833 | 0.9833 | 0.9833 | 30.7 | 21.5 | 2174.1 | 112923651 | True |
|  | learner_park | 20261011-211515_bert_lr5e-5_s43 | transformer | klue/bert-base | 5e-05 | 43 | 2 | 0.9833 | 0.9833 | 0.9833 | 25.9 | 18.2 | 2131.4 | 110619651 | True |
|  | learner_park | 20261011-211546_bert_lr5e-5_s44 | transformer | klue/bert-base | 5e-05 | 44 | 1 | 0.9833 | 0.9833 | 0.9833 | 25.7 | 20.1 | 2131.4 | 110619651 | True |
|  | learner_park | 20261011-213134_deberta-ko_lr5e-5_s44 | transformer | team-lucid/deberta-v3-base-korean | 5e-05 | 44 | 5 | 0.9833 | 0.9833 | 0.9833 | 49.3 | 37.7 | 2601.4 | 135272451 | True |
|  | learner_park | 20261011-210534_baseline_char2-5 | baseline | tfidf-char+logistic-regression | - | 42 | - | 0.9500 | 0.9500 | 0.9500 | 0.1799 | 0.3322 | - | 118710 | True |
|  | learner_park | 20261011-210922_electra-s_lr2e-5_s42 | transformer | monologg/koelectra-small-v3-discriminator | 2e-05 | 42 | 5 | 0.9000 | 0.8997 | 0.8997 | 25.6 | 7.1083 | 292.2 | 14122755 | True |
|  | learner_park | 20261011-210854_electra-s_lr1e-5_s42 | transformer | monologg/koelectra-small-v3-discriminator | 1e-05 | 42 | 5 | 0.8333 | 0.8290 | 0.8290 | 25.7 | 7.3968 | 292.2 | 14122755 | True |
|  | learner_park | 20261011-211304_deberta-ko_lr2e-5_s42 | transformer | team-lucid/deberta-v3-base-korean | 2e-05 | 42 | 5 | 0.6750 | 0.5694 | 0.5694 | 48.3 | 39.2 | 2601.4 | 135272451 | True |
|  | learner_park | 20261011-211208_deberta-ko_lr1e-5_s42 | transformer | team-lucid/deberta-v3-base-korean | 1e-05 | 42 | 3 | 0.5083 | 0.4369 | 0.4369 | 47.2 | 39.0 | 2601.4 | 135272451 | True |
